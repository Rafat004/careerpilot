"""
Copilot V2 service.

Validates and executes confirmed CareerPilot actions. The chat model may propose
actions, but this service is the authority before anything is shown or mutated.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import parse_qs, urlencode, urlparse

import logging

from db.supabase import supabase

logger = logging.getLogger(__name__)

ApplicationStatus = Literal["saved", "applied", "interviewing", "offer", "rejected"]

APPLICATION_STATUSES: set[str] = {"saved", "applied", "interviewing", "offer", "rejected"}
TRACKER_VIEWS: set[str] = {"today", "applications", "goals_tasks", "calendar", "progress"}
ROUTES: set[str] = {"/tracker", "/jobs", "/chat", "/cv", "/cv/preview"}
CV_PARAMS: set[str] = {"upload", "build", "preview"}
JOB_PARAMS: set[str] = {"query", "location", "auto"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
GUIDE_STEPS: list[str] = [
    "welcome",
    "preferences",
    "profile_setup",
    "job_search",
    "job_review",
    "applications",
    "goals_tasks",
    "calendar",
    "progress",
    "today",
]
GUIDANCE_LEVELS: set[str] = {"first_run", "guided", "light", "minimal"}

DEFAULT_ONBOARDING: dict[str, Any] = {
    "completed": False,
    "name": "",
    "targetRoles": [],
    "location": "",
    "workMode": "",
    "careerStage": "",
    "lastStep": "name",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_list(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text:
            cleaned.append(text)
    return cleaned[:limit]


def _clean_step_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    steps: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text in GUIDE_STEPS and text not in steps:
            steps.append(text)
    return steps


def _normalize_onboarding(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    target_roles = source.get("targetRoles") or source.get("target_roles")
    next_onboarding = {
        **DEFAULT_ONBOARDING,
        "completed": bool(source.get("completed")),
        "name": _clean_text(source.get("name")),
        "targetRoles": _clean_list(target_roles, limit=4),
        "location": _clean_text(source.get("location")),
        "workMode": _clean_text(source.get("workMode") or source.get("work_mode")),
        "careerStage": _clean_text(source.get("careerStage") or source.get("career_stage")),
    }

    if not next_onboarding["name"]:
        next_onboarding["lastStep"] = "name"
    elif not next_onboarding["targetRoles"]:
        next_onboarding["lastStep"] = "target_role"
    elif not next_onboarding["location"] and not next_onboarding["workMode"]:
        next_onboarding["lastStep"] = "location_work_mode"
    elif not next_onboarding["careerStage"]:
        next_onboarding["lastStep"] = "career_stage"
    else:
        next_onboarding["completed"] = True
        next_onboarding["lastStep"] = "complete"

    return next_onboarding


def _guidance_level(completed_steps: list[str], current_level: str | None = None) -> str:
    if current_level == "minimal":
        return "minimal"
    count = len(set(completed_steps))
    if count <= 1:
        return "first_run"
    if count < 6:
        return "guided"
    if count < len(GUIDE_STEPS):
        return "light"
    return "minimal"


def _derive_completed_steps(
    onboarding: dict[str, Any],
    profile_status: str,
    application_counts: dict[str, int],
    active_goals: list[dict[str, Any]],
    todos: list[dict[str, Any]],
    existing_steps: list[str],
) -> list[str]:
    steps = set(existing_steps)
    steps.add("welcome")
    if onboarding.get("completed"):
        steps.add("preferences")
    if profile_status == "has_profile":
        steps.add("profile_setup")
    if sum(application_counts.values()) > 0:
        steps.add("job_search")
        steps.add("job_review")
        steps.add("applications")
    if active_goals or todos:
        steps.add("goals_tasks")
    if any(_clean_text(todo.get("due_date")) for todo in todos):
        steps.add("calendar")
    if steps.intersection({"applications", "goals_tasks", "calendar"}):
        steps.add("progress")
    if todos or active_goals or application_counts.get("saved", 0) or application_counts.get("interviewing", 0):
        steps.add("today")
    return [step for step in GUIDE_STEPS if step in steps]


def _next_guide_step(completed_steps: list[str]) -> str | None:
    completed = set(completed_steps)
    for step in GUIDE_STEPS:
        if step not in completed:
            return step
    return None


def _step_prompt(step: str | None) -> str:
    prompts = {
        "welcome": "Start by learning the user's name and explaining you will guide the app.",
        "preferences": "Ask target role, location or work mode, and career stage.",
        "profile_setup": "Guide the user to upload a CV or build a profile manually.",
        "job_search": "Suggest a focused job search and prefill Jobs.",
        "job_review": "Explain fit scores, details review, and saving strong jobs.",
        "applications": "Guide the user to Applications and explain statuses and drag/drop.",
        "goals_tasks": "Suggest a preparation goal with tasks and offer to add it.",
        "calendar": "Show Calendar so the user sees due dates and deadlines.",
        "progress": "Show Progress and explain what the metrics mean.",
        "today": "Show Today as the daily action queue and next best action.",
    }
    return prompts.get(step or "", "Keep guidance short and action-focused.")


async def get_copilot_state(user_id: str) -> dict[str, Any]:
    try:
        result = await supabase.table("copilot_user_state").select(
            "user_id, onboarding, completed_steps, feature_exposures, guidance_level, "
            "last_suggested_step, updated_at"
        ).eq("user_id", user_id).limit(1).execute()

        if result.data:
            row = result.data[0]
            return {
                "user_id": user_id,
                "onboarding": _normalize_onboarding(row.get("onboarding")),
                "completed_steps": _clean_step_list(row.get("completed_steps")),
                "feature_exposures": row.get("feature_exposures") if isinstance(row.get("feature_exposures"), dict) else {},
                "guidance_level": row.get("guidance_level") if row.get("guidance_level") in GUIDANCE_LEVELS else "first_run",
                "last_suggested_step": row.get("last_suggested_step"),
                "updated_at": row.get("updated_at"),
            }
    except Exception as e:
        logger.warning("Failed to query copilot_user_state: %s. Using default state.", e)

    state = {
        "user_id": user_id,
        "onboarding": DEFAULT_ONBOARDING,
        "completed_steps": [],
        "feature_exposures": {},
        "guidance_level": "first_run",
        "last_suggested_step": None,
        "updated_at": None,
    }
    try:
        await supabase.table("copilot_user_state").upsert({
            "user_id": user_id,
            "onboarding": state["onboarding"],
            "completed_steps": state["completed_steps"],
            "feature_exposures": state["feature_exposures"],
            "guidance_level": state["guidance_level"],
            "last_suggested_step": state["last_suggested_step"],
            "updated_at": _now_iso(),
        }, on_conflict="user_id").execute()
    except Exception as e:
        logger.warning("Failed to initialize/upsert copilot_user_state: %s", e)
    return state


async def patch_copilot_state(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    current = await get_copilot_state(user_id)
    onboarding = current["onboarding"]
    completed_steps = current["completed_steps"]
    feature_exposures = current["feature_exposures"]
    guidance_level = current["guidance_level"]
    last_suggested_step = current["last_suggested_step"]

    if "onboarding" in payload:
        onboarding = _normalize_onboarding({**onboarding, **(payload.get("onboarding") or {})})

    if "completed_steps" in payload:
        completed_steps = _clean_step_list(payload.get("completed_steps"))

    if "mark_step_complete" in payload:
        step = _clean_text(payload.get("mark_step_complete"))
        if step in GUIDE_STEPS and step not in completed_steps:
            completed_steps = [*completed_steps, step]

    if "feature_exposure" in payload and isinstance(payload.get("feature_exposure"), dict):
        feature = _clean_text(payload["feature_exposure"].get("feature"))
        if feature:
            feature_exposures = {
                **feature_exposures,
                feature: {
                    "count": int((feature_exposures.get(feature) or {}).get("count", 0)) + 1
                    if isinstance(feature_exposures.get(feature), dict)
                    else 1,
                    "last_seen_at": _now_iso(),
                },
            }

    if "guidance_level" in payload and payload.get("guidance_level") in GUIDANCE_LEVELS:
        guidance_level = payload["guidance_level"]
    else:
        guidance_level = _guidance_level(completed_steps, guidance_level)

    if "last_suggested_step" in payload:
        candidate = _clean_text(payload.get("last_suggested_step"))
        last_suggested_step = candidate if candidate in GUIDE_STEPS else None

    row = {
        "user_id": user_id,
        "onboarding": onboarding,
        "completed_steps": completed_steps,
        "feature_exposures": feature_exposures,
        "guidance_level": guidance_level,
        "last_suggested_step": last_suggested_step,
        "updated_at": _now_iso(),
    }
    try:
        result = await supabase.table("copilot_user_state").upsert(
            row,
            on_conflict="user_id",
        ).execute()
        return result.data[0] if result.data else await get_copilot_state(user_id)
    except Exception as e:
        logger.warning("Failed to patch copilot_user_state: %s", e)
        return current


def _valid_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    text = _clean_text(value)
    if not DATE_RE.match(text):
        raise ValueError("Dates must use YYYY-MM-DD format.")
    return text


def _is_allowed_href(href: str) -> bool:
    if not href.startswith("/") or href.startswith("//"):
        return False

    parsed = urlparse(href)
    if parsed.scheme or parsed.netloc:
        return False
    if parsed.path not in ROUTES:
        return False

    params = parse_qs(parsed.query, keep_blank_values=True)
    param_keys = set(params.keys())

    if parsed.path == "/tracker":
        if not param_keys:
            return True
        if "view" not in param_keys or params.get("view", [""])[0] not in TRACKER_VIEWS:
            return False
        return param_keys.issubset({"view", "draft"})

    if parsed.path == "/cv":
        if not param_keys:
            return True
        return param_keys.issubset(CV_PARAMS) and all(values == ["1"] for values in params.values())

    if parsed.path == "/cv/preview":
        return not param_keys

    if parsed.path == "/jobs":
        return param_keys.issubset(JOB_PARAMS)

    return not param_keys


def _job_search_href(query: str, location: str | None, auto: bool) -> str:
    params = {"query": query}
    if location:
        params["location"] = location
    if auto:
        params["auto"] = "1"
    return f"/jobs?{urlencode(params)}"


async def get_preferences(user_id: str) -> dict[str, Any]:
    result = await supabase.table("career_preferences").select(
        "user_id, preferred_name, target_roles, preferred_locations, work_modes, "
        "seniority, weekly_capacity_hours, target_start_date, industries, updated_at"
    ).eq("user_id", user_id).limit(1).execute()

    if result.data:
        return result.data[0]

    return {
        "user_id": user_id,
        "preferred_name": None,
        "target_roles": [],
        "preferred_locations": [],
        "work_modes": [],
        "seniority": None,
        "weekly_capacity_hours": None,
        "target_start_date": None,
        "industries": [],
        "updated_at": None,
    }


async def get_context_snapshot(user_id: str) -> dict[str, Any]:
    today = datetime.now(timezone.utc).date().isoformat()
    preferences = await get_preferences(user_id)
    copilot_state = await get_copilot_state(user_id)
    onboarding = _normalize_onboarding({
        **copilot_state.get("onboarding", {}),
        "name": copilot_state.get("onboarding", {}).get("name") or preferences.get("preferred_name") or "",
        "targetRoles": copilot_state.get("onboarding", {}).get("targetRoles") or preferences.get("target_roles") or [],
        "location": copilot_state.get("onboarding", {}).get("location")
        or (preferences.get("preferred_locations") or [""])[0],
        "workMode": copilot_state.get("onboarding", {}).get("workMode")
        or (preferences.get("work_modes") or [""])[0],
        "careerStage": copilot_state.get("onboarding", {}).get("careerStage") or preferences.get("seniority") or "",
    })

    profile: dict[str, Any] | None = None
    try:
        profile_result = await supabase.table("profiles").select(
            "user_id, active_cv_id, headline, location, skills, updated_at"
        ).eq("user_id", user_id).limit(1).execute()
        if profile_result.data:
            profile = profile_result.data[0]
    except Exception:
        profile = None

    application_counts = {status: 0 for status in APPLICATION_STATUSES}
    high_fit_saved_jobs: list[dict[str, Any]] = []
    interviewing_applications: list[dict[str, Any]] = []

    try:
        app_result = await supabase.table("applications").select(
            "id, status, job_id, applied_at"
        ).eq("user_id", user_id).execute()
        applications = app_result.data or []
        for application in applications:
            status = _clean_text(application.get("status"))
            if status in application_counts:
                application_counts[status] += 1

        job_ids = list({
            _clean_text(application.get("job_id"))
            for application in applications
            if _clean_text(application.get("job_id"))
        })
        jobs_map: dict[str, dict[str, Any]] = {}
        if job_ids:
            try:
                jobs_result = await supabase.table("jobs").select(
                    "id, title, company, location, fit_score, url"
                ).in_("id", job_ids).execute()
            except Exception:
                jobs_result = await supabase.table("jobs").select(
                    "id, title, company, location, fit_score"
                ).in_("id", job_ids).execute()
            jobs_map = {job["id"]: job for job in jobs_result.data or []}

        for application in applications:
            job = jobs_map.get(_clean_text(application.get("job_id")))
            if not job:
                continue
            fit_score = job.get("fit_score")
            summary = {
                "application_id": application.get("id"),
                "job_id": job.get("id"),
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "fit_score": fit_score,
            }
            if (
                application.get("status") == "saved"
                and isinstance(fit_score, (int, float))
                and fit_score >= 70
            ):
                high_fit_saved_jobs.append(summary)
            if application.get("status") == "interviewing":
                interviewing_applications.append(summary)
    except Exception:
        pass

    try:
        todos_result = await supabase.table("todos").select(
            "id, title, due_date, completed"
        ).eq("user_id", user_id).eq("completed", False).order(
            "due_date", desc=False
        ).limit(25).execute()
        todos = todos_result.data or []
    except Exception:
        todos = []

    due_today = [
        todo for todo in todos
        if _clean_text(todo.get("due_date")) == today
    ]
    overdue = [
        todo for todo in todos
        if _clean_text(todo.get("due_date")) and _clean_text(todo.get("due_date")) < today
    ]

    try:
        goals_result = await supabase.table("goals").select(
            "id, title, target_date, completed"
        ).eq("user_id", user_id).eq("completed", False).order(
            "target_date", desc=False
        ).limit(10).execute()
        active_goals = goals_result.data or []
    except Exception:
        active_goals = []

    skills = profile.get("skills") if profile else []
    top_skills = skills[:8] if isinstance(skills, list) else []

    profile_status = "has_profile" if profile and profile.get("active_cv_id") else "no_profile"
    completed_steps = _derive_completed_steps(
        onboarding=onboarding,
        profile_status=profile_status,
        application_counts=application_counts,
        active_goals=active_goals,
        todos=todos,
        existing_steps=_clean_step_list(copilot_state.get("completed_steps")),
    )
    next_step = _next_guide_step(completed_steps)
    guidance_level = _guidance_level(completed_steps, copilot_state.get("guidance_level"))

    try:
        await supabase.table("copilot_user_state").upsert({
            "user_id": user_id,
            "onboarding": onboarding,
            "completed_steps": completed_steps,
            "feature_exposures": copilot_state.get("feature_exposures", {}),
            "guidance_level": guidance_level,
            "last_suggested_step": next_step,
            "updated_at": _now_iso(),
        }, on_conflict="user_id").execute()
    except Exception:
        pass

    remaining_steps = [step for step in GUIDE_STEPS if step not in completed_steps]

    return {
        "profile_status": profile_status,
        "profile": {
            "headline": profile.get("headline") if profile else None,
            "location": profile.get("location") if profile else None,
            "top_skills": top_skills,
            "updated_at": profile.get("updated_at") if profile else None,
        },
        "preferences": preferences,
        "applications": {
            "counts": application_counts,
            "high_fit_saved_jobs": high_fit_saved_jobs[:3],
            "interviewing": interviewing_applications[:3],
        },
        "goals": {
            "active_count": len(active_goals),
            "next": active_goals[:3],
        },
        "todos": {
            "due_today_count": len(due_today),
            "overdue_count": len(overdue),
            "next": todos[:5],
        },
        "copilot": {
            "onboarding": onboarding,
            "completed_steps": completed_steps,
            "remaining_steps": remaining_steps,
            "feature_exposures": copilot_state.get("feature_exposures", {}),
            "guidance_level": guidance_level,
            "next_step": next_step,
            "next_step_prompt": _step_prompt(next_step),
            "guide_steps": GUIDE_STEPS,
        },
    }


async def patch_preferences(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "user_id": user_id,
        "updated_at": _now_iso(),
    }

    if "preferred_name" in payload:
        row["preferred_name"] = _clean_text(payload.get("preferred_name")) or None
    if "target_roles" in payload:
        row["target_roles"] = _clean_list(payload.get("target_roles"), limit=6)
    if "preferred_locations" in payload:
        row["preferred_locations"] = _clean_list(payload.get("preferred_locations"), limit=6)
    if "work_modes" in payload:
        row["work_modes"] = _clean_list(payload.get("work_modes"), limit=4)
    if "seniority" in payload:
        row["seniority"] = _clean_text(payload.get("seniority")) or None
    if "weekly_capacity_hours" in payload:
        capacity = payload.get("weekly_capacity_hours")
        row["weekly_capacity_hours"] = int(capacity) if capacity is not None else None
    if "target_start_date" in payload:
        row["target_start_date"] = _valid_date(payload.get("target_start_date"))
    if "industries" in payload:
        row["industries"] = _clean_list(payload.get("industries"), limit=8)

    result = await supabase.table("career_preferences").upsert(
        row,
        on_conflict="user_id",
    ).execute()
    return result.data[0] if result.data else await get_preferences(user_id)


async def _log_action_event(
    user_id: str,
    action_type: str,
    status: str,
    source: str,
    summary: str,
    action: dict[str, Any],
    created_records: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    try:
        await supabase.table("copilot_action_events").insert({
            "user_id": user_id,
            "action_type": action_type,
            "status": status,
            "source": source,
            "summary": summary,
            "action": action,
            "created_records": created_records or {},
            "error": error,
        }).execute()
    except Exception:
        pass


async def _application_for_user(user_id: str, application_id: str) -> dict[str, Any]:
    result = await supabase.table("applications").select(
        "id, user_id, job_id, status, applied_at, notes"
    ).eq("id", application_id).eq("user_id", user_id).limit(1).execute()
    if not result.data:
        raise ValueError("Application not found.")
    return result.data[0]


async def _job_for_user(user_id: str, job_id: str) -> dict[str, Any]:
    result = await supabase.table("jobs").select(
        "id, user_id, title, company, location, url, fit_score"
    ).eq("id", job_id).eq("user_id", user_id).limit(1).execute()
    if not result.data:
        raise ValueError("Job not found.")
    return result.data[0]


def _base_response(action: dict[str, Any], summary: str, is_mutating: bool) -> dict[str, Any]:
    return {
        "valid": True,
        "action": action,
        "action_type": action["type"],
        "label": action.get("label") or summary,
        "summary": summary,
        "is_mutating": is_mutating,
        "confirmation_label": "Confirm" if is_mutating else "Open",
    }


async def validate_action(user_id: str, action: dict[str, Any], source: str = "widget") -> dict[str, Any]:
    action_type = _clean_text(action.get("type"))
    if not action_type:
        raise ValueError("Action type is required.")

    label = _clean_text(action.get("label"))

    if action_type == "open_route":
        href = _clean_text(action.get("href"))
        if not label or not href or not _is_allowed_href(href):
            raise ValueError("That route is not available.")
        normalized = {"type": action_type, "label": label, "href": href}
        return _base_response(normalized, label, False)

    if action_type == "prefill_job_search":
        query = _clean_text(action.get("query"))
        if not label or not query:
            raise ValueError("A job search needs a query.")
        location = _clean_text(action.get("location")) or None
        auto = action.get("auto") is True
        href = _job_search_href(query, location, auto)
        if not _is_allowed_href(href):
            raise ValueError("That job search is not available.")
        normalized = {
            "type": action_type,
            "label": label,
            "query": query,
            "location": location or "",
            "auto": auto,
        }
        response = _base_response(normalized, label, False)
        response["href"] = href
        return response

    if action_type == "create_todo":
        todo = action.get("todo") if isinstance(action.get("todo"), dict) else {}
        title = _clean_text(todo.get("title"))
        if not label or not title:
            raise ValueError("A task needs a title.")
        normalized = {
            "type": action_type,
            "label": label,
            "todo": {"title": title, "due_date": _valid_date(todo.get("due_date"))},
        }
        return _base_response(normalized, f"Add task: {title}", True)

    if action_type == "prefill_todo":
        todo = action.get("todo") if isinstance(action.get("todo"), dict) else {}
        title = _clean_text(todo.get("title"))
        if not label or not title:
            raise ValueError("A task draft needs a title.")
        normalized = {
            "type": action_type,
            "label": label,
            "todo": {"title": title, "due_date": _valid_date(todo.get("due_date"))},
        }
        response = _base_response(normalized, f"Prepare task: {title}", False)
        response["href"] = "/tracker?view=goals_tasks&draft=1"
        return response

    if action_type == "create_goal_with_todos":
        goal = action.get("goal") if isinstance(action.get("goal"), dict) else {}
        title = _clean_text(goal.get("title"))
        todos_in = action.get("todos") if isinstance(action.get("todos"), list) else []
        if not label or not title:
            raise ValueError("A goal needs a title.")
        if len(todos_in) > 5:
            raise ValueError("A plan can include at most five tasks.")
        todos = []
        for todo in todos_in:
            if not isinstance(todo, dict):
                raise ValueError("Each task must be an object.")
            todo_title = _clean_text(todo.get("title"))
            if not todo_title:
                raise ValueError("Each task needs a title.")
            todos.append({"title": todo_title, "due_date": _valid_date(todo.get("due_date"))})
        normalized = {
            "type": action_type,
            "label": label,
            "goal": {"title": title, "target_date": _valid_date(goal.get("target_date"))},
            "todos": todos,
        }
        return _base_response(normalized, f"Create goal: {title}", True)

    if action_type == "prefill_goal_with_todos":
        goal = action.get("goal") if isinstance(action.get("goal"), dict) else {}
        title = _clean_text(goal.get("title"))
        todos_in = action.get("todos") if isinstance(action.get("todos"), list) else []
        if not label or not title:
            raise ValueError("A goal draft needs a title.")
        if len(todos_in) > 5:
            raise ValueError("A draft can include at most five tasks.")
        todos = []
        for todo in todos_in:
            if not isinstance(todo, dict):
                raise ValueError("Each task must be an object.")
            todo_title = _clean_text(todo.get("title"))
            if not todo_title:
                raise ValueError("Each task needs a title.")
            todos.append({"title": todo_title, "due_date": _valid_date(todo.get("due_date"))})
        normalized = {
            "type": action_type,
            "label": label,
            "goal": {"title": title, "target_date": _valid_date(goal.get("target_date"))},
            "todos": todos,
        }
        response = _base_response(normalized, f"Prepare goal: {title}", False)
        response["href"] = "/tracker?view=goals_tasks&draft=1"
        return response

    if action_type == "create_roadmap_with_tasks":
        goals_in = action.get("goals") if isinstance(action.get("goals"), list) else []
        if not label:
            raise ValueError("A roadmap needs a label.")
        if not goals_in:
            raise ValueError("A roadmap needs at least one goal.")
        if len(goals_in) > 4:
            raise ValueError("A roadmap can include at most four goals.")

        total_todos = 0
        goals = []
        for goal in goals_in:
            if not isinstance(goal, dict):
                raise ValueError("Each roadmap goal must be an object.")
            title = _clean_text(goal.get("title"))
            if not title:
                raise ValueError("Each roadmap goal needs a title.")
            todos_in = goal.get("todos") if isinstance(goal.get("todos"), list) else []
            if len(todos_in) > 5:
                raise ValueError("Each roadmap goal can include at most five tasks.")

            todos = []
            for todo in todos_in:
                if not isinstance(todo, dict):
                    raise ValueError("Each roadmap task must be an object.")
                todo_title = _clean_text(todo.get("title"))
                if not todo_title:
                    raise ValueError("Each roadmap task needs a title.")
                todos.append({"title": todo_title, "due_date": _valid_date(todo.get("due_date"))})

            total_todos += len(todos)
            goals.append({
                "title": title,
                "target_date": _valid_date(goal.get("target_date")),
                "todos": todos,
            })

        if total_todos > 12:
            raise ValueError("A roadmap can include at most twelve tasks.")

        normalized = {
            "type": action_type,
            "label": label,
            "goals": goals,
        }
        return _base_response(normalized, f"Create roadmap: {len(goals)} goals", True)

    if action_type == "save_application":
        job_id = _clean_text(action.get("job_id"))
        status = _clean_text(action.get("status")) or "saved"
        if status not in APPLICATION_STATUSES:
            raise ValueError("Application status is not valid.")
        job = await _job_for_user(user_id, job_id)
        normalized = {
            "type": action_type,
            "label": label or f"Save {job.get('title', 'job')}",
            "job_id": job_id,
            "status": status,
        }
        return _base_response(normalized, f"Save {job.get('title', 'job')} to Applications", True)

    if action_type == "update_application_status":
        application_id = _clean_text(action.get("application_id"))
        status = _clean_text(action.get("status"))
        if status not in APPLICATION_STATUSES:
            raise ValueError("Application status is not valid.")
        await _application_for_user(user_id, application_id)
        normalized = {
            "type": action_type,
            "label": label or "Update application",
            "application_id": application_id,
            "status": status,
        }
        return _base_response(normalized, f"Move application to {status}", True)

    if action_type == "save_application_note":
        application_id = _clean_text(action.get("application_id"))
        note = _clean_text(action.get("note"))
        if not note:
            raise ValueError("A note cannot be empty.")
        await _application_for_user(user_id, application_id)
        normalized = {
            "type": action_type,
            "label": label or "Save note",
            "application_id": application_id,
            "note": note,
        }
        return _base_response(normalized, "Save note to application", True)

    if action_type == "prefill_application_note":
        application_id = _clean_text(action.get("application_id"))
        note = _clean_text(action.get("note"))
        if not label or not application_id or not note:
            raise ValueError("An application note draft needs an application and note.")
        await _application_for_user(user_id, application_id)
        normalized = {
            "type": action_type,
            "label": label,
            "application_id": application_id,
            "note": note,
        }
        response = _base_response(normalized, "Prepare application note", False)
        response["href"] = "/tracker?view=applications&draft=1"
        return response

    if action_type == "show_feature_explainer":
        feature = _clean_text(action.get("feature"))
        if not label or not feature:
            raise ValueError("A feature explainer needs a feature.")
        normalized = {
            "type": action_type,
            "label": label,
            "feature": feature,
            "body": _clean_text(action.get("body")),
            "href": _clean_text(action.get("href")) or None,
        }
        href = normalized.get("href")
        if href and not _is_allowed_href(href):
            raise ValueError("That route is not available.")
        response = _base_response(normalized, label, False)
        if href:
            response["href"] = href
        return response

    raise ValueError("Action type is not supported.")


async def execute_action(user_id: str, action: dict[str, Any], source: str = "widget") -> dict[str, Any]:
    try:
        validated = await validate_action(user_id, action, source)
    except ValueError as exc:
        await _log_action_event(
            user_id=user_id,
            action_type=_clean_text(action.get("type")) or "unknown",
            status="rejected",
            source=source,
            summary="Rejected unsafe Copilot action",
            action=action,
            error=str(exc),
        )
        raise

    normalized = validated["action"]
    action_type = validated["action_type"]
    summary = validated["summary"]
    created_records: dict[str, Any] = {}
    href = "/tracker"

    try:
        if action_type == "create_todo":
            todo = normalized["todo"]
            result = await supabase.table("todos").insert({
                "user_id": user_id,
                "goal_id": None,
                "title": todo["title"],
                "due_date": todo.get("due_date"),
                "completed": False,
            }).execute()
            created_records = {"todo": result.data[0] if result.data else None}
            href = "/tracker?view=goals_tasks"

        elif action_type == "create_goal_with_todos":
            goal = normalized["goal"]
            goal_result = await supabase.table("goals").insert({
                "user_id": user_id,
                "title": goal["title"],
                "target_date": goal.get("target_date"),
                "completed": False,
            }).execute()
            if not goal_result.data:
                raise ValueError("Failed to create goal.")
            goal_row = goal_result.data[0]
            todo_rows = []
            for todo in normalized["todos"]:
                todo_result = await supabase.table("todos").insert({
                    "user_id": user_id,
                    "goal_id": goal_row["id"],
                    "title": todo["title"],
                    "due_date": todo.get("due_date"),
                    "completed": False,
                }).execute()
                if todo_result.data:
                    todo_rows.append(todo_result.data[0])
            created_records = {"goal": goal_row, "todos": todo_rows}
            href = "/tracker?view=goals_tasks"

        elif action_type == "create_roadmap_with_tasks":
            goal_rows = []
            todo_rows = []
            for goal in normalized["goals"]:
                goal_result = await supabase.table("goals").insert({
                    "user_id": user_id,
                    "title": goal["title"],
                    "target_date": goal.get("target_date"),
                    "completed": False,
                }).execute()
                if not goal_result.data:
                    raise ValueError("Failed to create roadmap goal.")
                goal_row = goal_result.data[0]
                goal_rows.append(goal_row)

                for todo in goal["todos"]:
                    todo_result = await supabase.table("todos").insert({
                        "user_id": user_id,
                        "goal_id": goal_row["id"],
                        "title": todo["title"],
                        "due_date": todo.get("due_date"),
                        "completed": False,
                    }).execute()
                    if todo_result.data:
                        todo_rows.append(todo_result.data[0])

            created_records = {"goals": goal_rows, "todos": todo_rows}
            href = "/tracker?view=goals_tasks"

        elif action_type == "save_application":
            job_id = normalized["job_id"]
            existing = await supabase.table("applications").select(
                "id, user_id, job_id, status, applied_at"
            ).eq("user_id", user_id).eq("job_id", job_id).limit(1).execute()
            if existing.data:
                created_records = {"application": existing.data[0], "duplicate": True}
            else:
                result = await supabase.table("applications").insert({
                    "user_id": user_id,
                    "job_id": job_id,
                    "status": normalized["status"],
                }).execute()
                created_records = {"application": result.data[0] if result.data else None, "duplicate": False}
            href = "/tracker?view=applications"

        elif action_type == "update_application_status":
            application = await _application_for_user(user_id, normalized["application_id"])
            update_data: dict[str, Any] = {"status": normalized["status"]}
            if normalized["status"] == "saved":
                update_data["applied_at"] = None
            elif not application.get("applied_at"):
                update_data["applied_at"] = _now_iso()
            result = await supabase.table("applications").update(
                update_data
            ).eq("id", normalized["application_id"]).eq("user_id", user_id).execute()
            created_records = {"application": result.data[0] if result.data else None}
            href = "/tracker?view=applications"

        elif action_type == "save_application_note":
            result = await supabase.table("applications").update({
                "notes": normalized["note"],
            }).eq("id", normalized["application_id"]).eq("user_id", user_id).execute()
            created_records = {"application": result.data[0] if result.data else None}
            href = "/tracker?view=applications"

        else:
            raise ValueError("Only mutating actions should be executed.")

        await _log_action_event(
            user_id=user_id,
            action_type=action_type,
            status="executed",
            source=source,
            summary=summary,
            action=normalized,
            created_records=created_records,
        )
        return {
            "status": "executed",
            "message": "Confirmed.",
            "href": href,
            "created_records": created_records,
            "validated_action": validated,
        }
    except Exception as exc:
        await _log_action_event(
            user_id=user_id,
            action_type=action_type,
            status="failed",
            source=source,
            summary=summary,
            action=normalized,
            error=str(exc),
        )
        raise


async def list_action_events(user_id: str) -> list[dict[str, Any]]:
    result = await supabase.table("copilot_action_events").select(
        "id, user_id, action_type, status, source, summary, created_records, error, created_at"
    ).eq("user_id", user_id).order("created_at", desc=True).limit(50).execute()
    return result.data or []
