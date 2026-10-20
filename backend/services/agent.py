"""
Job Hunter Agent Service — CareerPilot

Job search priority (AGENTS.md):
  1st  JSearch (RapidAPI)  — always try first, best BD/Dhaka coverage
  2nd  Remotive            — fallback for remote roles, no API key needed
  3rd  Tavily              — fallback for local BD sites incl. bdjobs.com

Agent State and Graph structure built with LangGraph:
  Nodes: search_node, score_node, filter_node
  Conditional Edge: should_retry
"""

import asyncio
import html
import logging
import os
import re
import httpx
from datetime import datetime
from html.parser import HTMLParser
from typing import TypedDict, List, Dict, Any, Literal
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from db.supabase import supabase
from services.cache import get_cached_jobs, cache_jobs
from services.fit_score import FIT_SCORE_VERSION, compute_fit_score, generate_fit_explanations, get_active_cv_id
from services.searcher import hybrid_search
from groq import Groq, RateLimitError as GroqRateLimitError
from services.key_pool import groq_pool, jsearch_pool, tavily_pool, KeyPoolExhausted

# TAVILY_API_KEY still needed directly for the payload dict
TAVILY_API_KEY: str = tavily_pool.current() if len(tavily_pool) > 0 else ""
logger = logging.getLogger(__name__)

_DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


def _get_candidate_models() -> list[str]:
    configured = os.environ.get("GROQ_MODEL", "").strip()
    if configured:
        return [configured] + [m for m in _DEFAULT_MODELS if m != configured]
    return _DEFAULT_MODELS


class _HTMLToTextParser(HTMLParser):
    block_tags = {"br", "div", "p", "li", "ul", "ol", "section", "article", "h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_text(self) -> str:
        return "".join(self.parts)


def _clean_job_description(value: Any, max_length: int = 1000) -> str:
    raw = str(value or "")
    if not raw:
        return ""

    parser = _HTMLToTextParser()
    try:
        parser.feed(raw)
        parser.close()
        text = parser.get_text()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", raw)

    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_length]


# ---------------------------------------------------------------------------
# Procedural Job API Integrations
# ---------------------------------------------------------------------------

async def _search_jsearch(query: str, location: str = "") -> list[dict]:
    url = "https://jsearch.p.rapidapi.com/search"
    params = {
        "query": f"{query} {location}".strip(),
        "num_pages": "1",
        "date_posted": "month",
    }
    rotation = jsearch_pool.rotate_on_rate_limit()
    for api_key in rotation:
        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "jsearch.p.rapidapi.com",
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code == 429:
                    continue  # rotate to next key
                resp.raise_for_status()
                data = resp.json()
            rotation.success()
            break
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                continue
            raise

    jobs = []
    for item in data.get("data", []):
        jobs.append({
            "title":       item.get("job_title", ""),
            "company":     item.get("employer_name", ""),
            "location":    item.get("job_city", "") or item.get("job_country", ""),
            "url":         item.get("job_apply_link", ""),
            "description": _clean_job_description(item.get("job_description", "")),
            "source":      "jsearch",
        })
    return jobs


async def _search_remotive(query: str) -> list[dict]:
    url = "https://remotive.com/api/remote-jobs"
    params = {"search": query, "limit": 10}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    jobs = []
    for item in data.get("jobs", []):
        jobs.append({
            "title":       item.get("title", ""),
            "company":     item.get("company_name", ""),
            "location":    item.get("candidate_required_location", "Remote"),
            "url":         item.get("url", ""),
            "description": _clean_job_description(item.get("description", "")),
            "source":      "remotive",
        })
    return jobs


async def _search_tavily(query: str, location: str = "") -> list[dict]:
    url = "https://api.tavily.com/search"
    rotation = tavily_pool.rotate_on_rate_limit()
    data: dict = {}
    for api_key in rotation:
        payload = {
            "api_key": api_key,
            "query": f"{query} jobs {location} site:bdjobs.com OR linkedin.com OR glassdoor.com".strip(),
            "search_depth": "basic",
            "max_results": 10,
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 429:
                    continue
                resp.raise_for_status()
                data = resp.json()
            rotation.success()
            break
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                continue
            raise

    jobs = []
    for item in data.get("results", []):
        jobs.append({
            "title":       query,
            "company":     "",
            "location":    location or "Bangladesh",
            "url":         item.get("url", ""),
            "description": _clean_job_description(item.get("content", "")),
            "source":      "tavily",
        })
    return jobs


# ---------------------------------------------------------------------------
# LangChain Tools (Day 5 requirement)
# ---------------------------------------------------------------------------

@tool
async def search_jobs_tool(query: str, location: str = "") -> List[dict]:
    """Search for jobs using the priority fallback pipeline: JSearch -> Remotive -> Tavily."""
    # JSearch
    try:
        jobs = await _search_jsearch(query, location)
        if jobs:
            return jobs
    except Exception:
        pass

    # Remotive
    try:
        jobs = await _search_remotive(query)
        if jobs:
            return jobs
    except Exception:
        pass

    # Tavily
    try:
        jobs = await _search_tavily(query, location)
        return jobs
    except Exception:
        return []


@tool
async def compute_fit_score_tool(job_description: str, user_id: str) -> dict:
    """Programmatically compute a matching fit score (0-100) and get a detailed one-sentence explanation."""
    return await compute_fit_score(job_description, user_id)


@tool
async def get_cv_context_tool(user_id: str) -> str:
    """Retrieve highly relevant section chunks from a user's CV via hybrid vector search."""
    try:
        chunks = await hybrid_search(query="", user_id=user_id, match_count=5)
        return "\n\n".join(f"[{c['section']}] {c['content']}" for c in chunks)
    except Exception:
        return "No CV context available."


@tool
def draft_cover_letter_tool(job_description: str, cv_context: str) -> str:
    """Draft a persuasive and professional cover letter tailored to a specific job description using the candidate's CV context."""
    prompt = (
        "You are a professional cover letter assistant.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate CV Context:\n{cv_context}\n\n"
        "Draft a compelling, professional cover letter tailored to this job description. "
        "Highlight key skills, experience matches, and use persuasive business language. "
        "Keep it concise, elegant, and ready for submission."
    )
    rotation = groq_pool.rotate_on_rate_limit()
    for api_key in rotation:
        try:
            client = Groq(api_key=api_key)
            for candidate_model in _get_candidate_models():
                try:
                    resp = client.chat.completions.create(
                        model=candidate_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                    )
                    rotation.success()
                    return (resp.choices[0].message.content or "").strip()
                except Exception as model_err:
                    err_str = str(model_err)
                    if "model_not_found" in err_str or "does not exist" in err_str:
                        logger.warning(
                            "Groq model '%s' unavailable for cover letter: %s",
                            candidate_model,
                            model_err,
                        )
                        continue
                    raise
        except GroqRateLimitError:
            continue
        except Exception as e:
            return f"Failed to generate cover letter: {e}"
    return "Cover letter generation unavailable — all API keys are rate-limited. Try again in a few minutes."


# ---------------------------------------------------------------------------
# LangGraph Agent Implementation
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    location: str
    user_id: str
    jobs: List[Dict[str, Any]]
    retries: int
    max_retries: int


async def search_node(state: AgentState) -> Dict[str, Any]:
    """Search node - executes priority search fallbacks."""
    query = state["query"]
    location = state["location"]
    
    # Try JSearch
    jobs = []
    try:
        jobs = await _search_jsearch(query, location)
    except Exception:
        pass

    # Try Remotive fallback
    if not jobs:
        try:
            jobs = await _search_remotive(query)
        except Exception:
            pass

    # Try Tavily fallback
    if not jobs:
        try:
            jobs = await _search_tavily(query, location)
        except Exception:
            pass
            
    return {"jobs": jobs}


async def filter_node(state: AgentState) -> Dict[str, Any]:
    """Keep the first 10 raw search results for scoring and presentation."""
    return {"jobs": [_sanitize_raw_job(job) for job in list(state["jobs"])[:10]]}


def should_retry(state: AgentState) -> Literal["retry", "end"]:
    """Conditional edge checking if we should try a broader query if zero jobs are found."""
    if not state["jobs"] and state["retries"] < state["max_retries"]:
        return "retry"
    return "end"


async def retry_node(state: AgentState) -> Dict[str, Any]:
    """Broadens query terms to trigger a retry search."""
    # Simplify query: split and take first 2 words
    words = state["query"].split()
    broader_query = " ".join(words[:2]) if len(words) > 2 else state["query"]
    return {
        "query": broader_query,
        "retries": state["retries"] + 1
    }


def _sanitize_raw_job(job: dict) -> dict:
    job_copy = dict(job)
    for key in (
        "id",
        "user_id",
        "fit_score",
        "fit_explanation",
        "scored_cv_id",
        "fit_score_calculated_at",
        "fit_score_version",
    ):
        job_copy.pop(key, None)
    return {
        "title": job_copy.get("title") or "Untitled role",
        "company": job_copy.get("company") or "",
        "location": job_copy.get("location") or "",
        "url": job_copy.get("url") or "",
        "description": _clean_job_description(job_copy.get("description")),
        "source": job_copy.get("source") or "jsearch",
        "salary_range": job_copy.get("salary_range") or "Not Disclosed",
        "deadline": job_copy.get("deadline") or "Rolling / Open",
    }


def _unavailable_score_explanation() -> str:
    return "Fit score is unavailable because the CV evidence could not be scored right now."


def _reused_score_explanation(score: int) -> str:
    return f"Fit score is {score}/100 based on weighted CV similarity across skills, experience, education, and projects."


def _is_missing_url_column_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "url" in text and "jobs" in text and (
        "schema cache" in text
        or "pgrst204" in text
        or "42703" in text
        or "does not exist" in text
    )


def _is_missing_provenance_columns_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "jobs" in text and (
        "scored_cv_id" in text or "fit_score_calculated_at" in text or "fit_score_version" in text
    ) and (
        "schema cache" in text
        or "pgrst204" in text
        or "42703" in text
        or "does not exist" in text
    )


async def _insert_or_reuse_job(raw_job: dict, user_id: str) -> tuple[dict, dict | None]:
    title = raw_job["title"]
    company = raw_job["company"]
    try:
        existing = await supabase.table("jobs").select(
            "id, fit_score, fit_explanation, scored_cv_id, fit_score_calculated_at, fit_score_version"
        ).eq("user_id", user_id).eq("title", title).eq("company", company).execute()
    except Exception as exc:
        if not _is_missing_provenance_columns_error(exc):
            raise
        try:
            existing = await supabase.table("jobs").select(
                "id, fit_score, fit_explanation"
            ).eq("user_id", user_id).eq("title", title).eq("company", company).execute()
            for row in existing.data or []:
                row["scored_cv_id"] = None
                row["fit_score_calculated_at"] = None
                row["fit_score_version"] = None
        except Exception:
            existing = None

    if existing and existing.data:
        return {**raw_job, "id": existing.data[0]["id"]}, existing.data[0]

    insert_data = {
        "user_id": user_id,
        "title": title,
        "company": company,
        "location": raw_job["location"],
        "salary_range": raw_job["salary_range"],
        "deadline": raw_job["deadline"],
        "description": raw_job["description"],
        "source": raw_job["source"],
        "url": raw_job["url"],
    }
    try:
        inserted = await supabase.table("jobs").insert(insert_data).execute()
    except Exception as exc:
        if not _is_missing_url_column_error(exc):
            raise
        fallback_insert_data = dict(insert_data)
        fallback_insert_data.pop("url", None)
        inserted = await supabase.table("jobs").insert(fallback_insert_data).execute()

    if inserted.data:
        return {**raw_job, "id": inserted.data[0]["id"]}, None
    return raw_job, None


def _score_can_be_reused(existing: dict | None, active_cv_id: str) -> bool:
    if not existing or existing.get("fit_score") is None:
        return False
    return (
        existing.get("scored_cv_id") == active_cv_id
        and existing.get("fit_score_version") == FIT_SCORE_VERSION
    )


async def _score_and_rank_jobs(raw_jobs: list[dict], user_id: str, search_query: str) -> list[dict]:
    scored_jobs = []
    explanation_inputs = []
    explanation_targets = []
    active_cv_id = await get_active_cv_id(user_id)

    for raw_job in raw_jobs[:10]:
        job, existing = await _insert_or_reuse_job(_sanitize_raw_job(raw_job), user_id)
        existing_score = existing.get("fit_score") if existing else None
        existing_explanation = existing.get("fit_explanation") if existing else None

        if _score_can_be_reused(existing, active_cv_id):
            explanation = existing_explanation or _reused_score_explanation(int(existing_score))
            job_with_existing_score = {
                **job,
                "fit_score": existing_score,
                "fit_explanation": explanation,
                "scored_cv_id": existing.get("scored_cv_id"),
                "fit_score_calculated_at": existing.get("fit_score_calculated_at"),
                "fit_score_version": existing.get("fit_score_version"),
            }
            scored_jobs.append(job_with_existing_score)
            if not existing_explanation and job.get("id"):
                await supabase.table("jobs").update({
                    "fit_explanation": explanation,
                }).eq("id", job["id"]).execute()
            continue

        try:
            job_text = "\n".join(
                value for value in [job.get("title", ""), job.get("description", "")] if value
            )
            score_res = await compute_fit_score(
                job_description=job_text or job.get("title") or "",
                user_id=user_id,
                include_explanation=False,
                search_query=search_query,
                active_cv_id=active_cv_id,
            )
            score = score_res["score"]
            calculated_at = datetime.utcnow().isoformat()
            job_with_score = {
                **job,
                "fit_score": score,
                "fit_explanation": score_res["explanation"],
                "scored_cv_id": score_res["active_cv_id"],
                "fit_score_calculated_at": calculated_at,
                "fit_score_version": score_res["fit_score_version"],
            }
            scored_jobs.append(job_with_score)
            explanation_inputs.append({
                "title": job.get("title", ""),
                "score": score,
                "job_description": job_text or job.get("title") or "",
                "evidence_snippets": score_res.get("evidence_snippets", []),
            })
            explanation_targets.append((job_with_score, job.get("id"), calculated_at))
            await asyncio.sleep(0.25)
        except Exception as exc:
            logger.warning(
                "Fit scoring failed for job '%s' (%s): %s",
                job.get("title", ""),
                job.get("id", "no-id"),
                exc,
            )
            scored_jobs.append({
                **job,
                "fit_score": None,
                "fit_explanation": _unavailable_score_explanation(),
            })

    explanations = generate_fit_explanations(explanation_inputs)
    for (job, job_id, calculated_at), explanation in zip(explanation_targets, explanations):
        job["fit_explanation"] = explanation
        if job_id:
            try:
                await supabase.table("jobs").update({
                    "fit_score": job["fit_score"],
                    "fit_explanation": explanation,
                    "scored_cv_id": job["scored_cv_id"],
                    "fit_score_calculated_at": calculated_at,
                    "fit_score_version": job["fit_score_version"],
                }).eq("id", job_id).execute()
            except Exception as exc:
                if not _is_missing_provenance_columns_error(exc):
                    raise
                await supabase.table("jobs").update({
                    "fit_score": job["fit_score"],
                    "fit_explanation": explanation,
                }).eq("id", job_id).execute()

    return sorted(
        scored_jobs,
        key=lambda item: (
            item.get("fit_score") is not None,
            item.get("fit_score") or 0,
        ),
        reverse=True,
    )


# Build LangGraph workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("search", search_node)
workflow.add_node("filter", filter_node)
workflow.add_node("retry", retry_node)

# Set entry point
workflow.set_entry_point("search")

# Add standard edges
workflow.add_edge("retry", "search")
workflow.add_edge("filter", END)

# Add conditional edges from search
workflow.add_conditional_edges(
    "search",
    should_retry,
    {
        "retry": "retry",
        "end": "filter"
    }
)

# Compile graph
graph = workflow.compile()


# ---------------------------------------------------------------------------
# Main Router Interface
# ---------------------------------------------------------------------------

async def hunt_jobs(query: str, location: str, user_id: str) -> list[dict]:
    """
    Hunt jobs using the compiled LangGraph workflow.
    Checks Upstash Redis cache first.
    """
    # 1. Check raw search cache first
    cached = await get_cached_jobs(query, location)
    if cached is not None:
        return await _score_and_rank_jobs(cached, user_id, query)

    # 2. Run LangGraph pipeline
    initial_state: AgentState = {
        "query": query,
        "location": location,
        "user_id": user_id,
        "jobs": [],
        "retries": 0,
        "max_retries": 1
    }
    
    final_state = await graph.ainvoke(initial_state)
    jobs = final_state.get("jobs", [])

    # 3. Cache raw results, then personalize scores for this user
    await cache_jobs(jobs, query, location)

    return await _score_and_rank_jobs(jobs, user_id, query)
