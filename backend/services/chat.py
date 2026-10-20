"""
Chat Service — CareerPilot (Pillar 3: AI Assistant)

Handles business logic for conversational assistant:
  - Fetches message history
  - Saves message logs
  - Streams conversational response from Groq Llama 3.3 70B using session history and RAG context
"""

import os
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Optional
from groq import Groq, RateLimitError as GroqRateLimitError
from db.supabase import supabase
from services.key_pool import groq_pool, KeyPoolExhausted

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

_MEMORY_LIMIT = 10
_DEFAULT_SESSION_TITLE = "New conversation"
_SESSION_TITLE_LIMIT = 42
_ACTION_PATTERN = re.compile(
    r"<careerpilot_action>\s*[\s\S]*?\s*</careerpilot_action>",
    re.MULTILINE,
)
_ONBOARDING_PATTERN = re.compile(
    r"<careerpilot_onboarding>\s*[\s\S]*?\s*</careerpilot_onboarding>",
    re.MULTILINE,
)


def _now_iso() -> str:
    """Return a timezone-aware timestamp for Supabase updates."""
    return datetime.now(timezone.utc).isoformat()


def _title_from_message(message: str) -> str:
    """Derive the persisted session title from the first user message."""
    title = " ".join(message.strip().split())
    if not title:
        return _DEFAULT_SESSION_TITLE
    if len(title) > _SESSION_TITLE_LIMIT:
        return f"{title[:_SESSION_TITLE_LIMIT]}..."
    return title


def _strip_copilot_actions(content: str) -> str:
    """Remove hidden UI directives before persisting assistant memory."""
    content = _ACTION_PATTERN.sub("", content)
    content = _ONBOARDING_PATTERN.sub("", content)
    return content.strip()


def _has_hidden_action(content: str) -> bool:
    return bool(_ACTION_PATTERN.search(content))


def _hidden_action(action: dict[str, Any]) -> str:
    return (
        "\n\n<careerpilot_action>\n"
        f"{json.dumps(action, ensure_ascii=True)}\n"
        "</careerpilot_action>"
    )


def _latest_assistant_message(history: list[dict]) -> str:
    for message in reversed(history):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _strip_markdown_label(value: str) -> str:
    return value.strip().strip("*_`").strip()


def _parse_goal_todos_from_text(content: str) -> dict[str, Any] | None:
    """Extract a simple Goal/Todos proposal from visible assistant text."""
    goal_match = re.search(
        r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?goal(?:\*\*)?\s*:\s*(.+?)\s*$",
        content,
    )
    if not goal_match:
        return None

    goal_title = _strip_markdown_label(goal_match.group(1))
    if not goal_title:
        return None

    todos_heading = re.search(
        r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?(?:todos?|tasks?)(?:\*\*)?\s*:?\s*$",
        content[goal_match.end():],
    )
    if not todos_heading:
        return None

    todos_text = content[goal_match.end() + todos_heading.end():]
    todo_lines = [_strip_markdown_label(line) for line in todos_text.splitlines()]
    todo_lines = [line for line in todo_lines if line]
    todos: list[dict[str, Any]] = []
    index = 0

    while index < len(todo_lines) and len(todos) < 5:
        line = todo_lines[index]
        if line.startswith("<careerpilot_"):
            break
        if re.match(r"(?i)^(goal|todos?|tasks?)\s*:", line):
            break

        numbered = re.match(r"^\d+[\.)]\s*(.*)$", line)
        bullet = re.match(r"^[-*]\s+(.*)$", line)
        title = ""

        if numbered:
            title = numbered.group(1).strip()
            if not title and index + 1 < len(todo_lines):
                index += 1
                title = todo_lines[index]
        elif bullet:
            title = bullet.group(1).strip()
        elif todos:
            title = line

        title = _strip_markdown_label(title)
        if title and not re.match(r"^\d+[\.)]?$", title):
            todos.append({"title": title, "due_date": None})

        index += 1

    if not todos:
        return None

    return {
        "type": "prefill_goal_with_todos",
        "label": "Review in Goals & Tasks",
        "goal": {
            "title": goal_title,
            "target_date": None,
        },
        "todos": todos,
    }


def _date_days_from_now(days: int) -> str:
    return (datetime.now(timezone.utc).date() + timedelta(days=days)).isoformat()


def _fallback_action_for_turn(
    message: str,
    history: list[dict],
    client_context: Optional[dict[str, Any]],
    visible_reply: str,
) -> dict[str, Any] | None:
    """Create deterministic action cards when the model speaks about an action but omits the directive."""
    normalized = " ".join(message.lower().split())
    previous_message = _latest_assistant_message(history)
    previous = previous_message.lower()
    profile_status = str((client_context or {}).get("profile_status") or "")

    visible_goal_action = _parse_goal_todos_from_text(visible_reply)
    if visible_goal_action:
        return visible_goal_action

    is_confirmation = normalized in {
        "ok",
        "okay",
        "yes",
        "yeah",
        "yep",
        "sure",
        "do it",
        "ok do it",
        "okay do it",
        "do that",
        "ok do that",
        "yes do that",
        "add it",
        "add them",
        "add them to my task and goal list",
        "add them to my goals and tasks",
        "add to goals and tasks",
        "confirm",
    }
    wants_open = any(
        phrase in normalized
        for phrase in (
            "show me",
            "open it",
            "open that",
            "take me",
            "go there",
            "resume preview",
            "preview resume",
            "view resume",
        )
    )
    wants_tasks = any(
        phrase in normalized
        for phrase in (
            "add the tasks",
            "add tasks",
            "add them",
            "create the tasks",
            "create tasks",
            "do that",
            "ok do that",
            "add them to my task and goal list",
            "add them to my goals and tasks",
            "add to goals and tasks",
        )
    )

    resume_context = "resume preview" in previous or "profile" in previous and "resume" in previous
    if wants_open and resume_context:
        if profile_status == "no_profile":
            return {
                "type": "open_route",
                "label": "Upload your CV",
                "href": "/cv?upload=1",
            }
        return {
            "type": "open_route",
            "label": "Open Resume Preview",
            "href": "/cv/preview",
        }

    task_context = any(
        phrase in previous
        for phrase in (
            "proposed the tasks",
            "these tasks",
            "career goal setting",
            "skill development",
            "job matching",
            "resume preview",
            "ai journey",
            "roadmap",
            "goal:",
            "todos:",
            "todo:",
            "tasks:",
            "task:",
        )
    )
    if wants_tasks and (task_context or is_confirmation):
        previous_goal_action = _parse_goal_todos_from_text(previous_message)
        if previous_goal_action:
            return previous_goal_action

        return {
            "type": "prefill_goal_with_todos",
            "label": "Review journey tasks",
            "goal": {
                "title": "Complete CareerPilot journey setup",
                "target_date": _date_days_from_now(7),
            },
            "todos": [
                {
                    "title": "Review Resume Preview for accuracy",
                    "due_date": _date_days_from_now(1),
                },
                {
                    "title": "Search matched jobs from my profile",
                    "due_date": _date_days_from_now(2),
                },
                {
                    "title": "Set one clear career goal",
                    "due_date": _date_days_from_now(4),
                },
                {
                    "title": "Choose one skill to improve this week",
                    "due_date": _date_days_from_now(7),
                },
            ],
        }

    if "job matching" in normalized or "search jobs" in normalized or "find jobs" in normalized:
        app_state = (client_context or {}).get("app_state") or {}
        preferences = app_state.get("preferences") if isinstance(app_state, dict) else {}
        target_roles = preferences.get("target_roles") if isinstance(preferences, dict) else []
        preferred_locations = preferences.get("preferred_locations") if isinstance(preferences, dict) else []
        query = target_roles[0] if isinstance(target_roles, list) and target_roles else "software engineer"
        location = preferred_locations[0] if isinstance(preferred_locations, list) and preferred_locations else ""
        return {
            "type": "prefill_job_search",
            "label": "Search matching jobs",
            "query": query,
            "location": location,
            "auto": True,
        }

    return None


def _format_client_context(client_context: Optional[dict[str, Any]]) -> str:
    if not client_context:
        return ""

    allowed_keys = {
        "current_path",
        "current_page_label",
        "profile_status",
        "onboarding",
        "preferences",
        "copilot_state",
        "app_state",
        "app_map",
    }
    compact_context = {
        key: value
        for key, value in client_context.items()
        if key in allowed_keys
    }
    return json.dumps(compact_context, ensure_ascii=True, indent=2)[:8000]


async def ensure_chat_session(user_id: str, session_id: str, message: str) -> None:
    """Create or refresh the durable chat session row."""
    existing = await supabase.table("chat_sessions").select(
        "title"
    ).eq("id", session_id).eq("user_id", user_id).limit(1).execute()

    rows = existing.data or []
    if rows:
        update_payload = {"updated_at": _now_iso()}
        if rows[0].get("title") == _DEFAULT_SESSION_TITLE:
            update_payload["title"] = _title_from_message(message)

        await supabase.table("chat_sessions").update(
            update_payload
        ).eq("id", session_id).eq("user_id", user_id).execute()
        return

    await supabase.table("chat_sessions").insert({
        "id":         session_id,
        "user_id":    user_id,
        "title":      _title_from_message(message),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }).execute()


async def get_chat_history(user_id: str, session_id: str) -> list[dict]:
    """Fetch last N messages for a chat session from Supabase in chronological order."""
    result = await supabase.table("chat_messages").select(
        "role, content"
    ).eq("user_id", user_id).eq("session_id", session_id).order(
        "created_at", desc=True
    ).limit(_MEMORY_LIMIT).execute()

    messages = result.data or []
    return list(reversed(messages))


async def save_message(user_id: str, session_id: str, role: str, content: str) -> None:
    """Persist a chat message to Supabase."""
    await supabase.table("chat_messages").insert({
        "user_id":    user_id,
        "session_id": session_id,
        "role":       role,
        "content":    content,
    }).execute()


async def stream_chat(
    user_id: str,
    session_id: str,
    message: str,
    cv_context: str,
    client_context: Optional[dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream chat response from Groq Llama 3.3 70B.
    Yields Server-Sent Events (SSE) compatible string tokens.
    """
    # 1. Fetch chat history
    history = await get_chat_history(user_id, session_id)

    # 2. Build system and conversation prompt
    formatted_client_context = _format_client_context(client_context)
    action_rules = ""
    if formatted_client_context:
        action_rules = f"""

Product context from the CareerPilot app:
{formatted_client_context}

You are also the user's in-app product guide. You should feel like a practical
career coach plus an app operator: warm, specific, low-drama, and always moving
the user toward the next useful step.

CareerPilot response shape:
- Start with the useful answer, not a generic greeting.
- Give a quick read, the reason, and the next best step.
- Use the current page and profile status from product context.
- If the user needs another app area, include a normal markdown link and, when
  useful, one hidden action directive.
- Ask at most one follow-up question when required to avoid a bad plan.
- Prefer concrete actions over long feature explanations.

Guidance playbooks:
- Work in four modes: onboarding guide, feature guide, career coach, and app
  operator. Choose the mode from the user's request and copilot_state.next_step.
- In onboarding guide mode, collect preferred name, target role, location or
  work mode, career stage or urgency, then ask whether they want to upload a CV
  or build a profile manually. Ask one missing detail at a time.
- In feature guide mode, explain the current feature in one or two sentences,
  then give the next click. Jobs explains fit score, Details, and Save to
  Applications. Applications explains statuses and drag/drop. Goals & Tasks
  explains draft plans and Create. Calendar explains due dates and deadlines.
  Progress explains metrics. Today explains the daily action queue.
- In career coach mode, answer the career question directly, grounded in CV
  context when available. After a side question, gently resume the guide, for
  example: "We can continue from job search when you're ready."
- In app operator mode, guide, prefill, route, and propose. Do not silently
  mutate product state.
- If profile_status is "no_profile", explain that CV/profile data powers fit
  scores and personalized advice. Offer Upload CV or Build profile manually.
- If profile_status is "has_profile", use CV context and preferences to suggest
  job searches, application steps, goals, tasks, cover-letter drafts, or prep.
- For job searches, propose a focused query and location. Use
  prefill_job_search when a clear search exists. Visible text must say what
  happens next, usually "Now click Search."
- For goals and tasks, prefer prefill_goal_with_todos or prefill_todo so the
  user reviews the form and clicks Create. Use direct create_* actions only when
  the user explicitly asks to create and the UI will preview the mutation.
- For application notes, prefer prefill_application_note so the note opens in
  Applications and the user clicks Save.
- For application status changes or saved jobs, only propose the change. Never
  claim the record changed before UI confirmation.
- Guidance should get shorter as copilot_state.guidance_level moves from
  first_run to guided, light, and minimal.

Hidden action rule:
Use at most one hidden action directive per assistant turn unless the user asks
for a multi-step setup. Put hidden directives at the very end of the response.
Never show or explain hidden JSON in visible text. Never claim an action has
been completed unless the UI has confirmed it after the user clicks.
If you say "I can propose", "I've proposed", "click", "open", "show", "add",
"create", or "confirm" for a CareerPilot product action, you must include the
matching hidden action directive in the same turn. Do not say an action is
proposed unless the directive is present.

Allowed hidden action directive format:
<careerpilot_action>
{{"type":"open_route","label":"Upload your CV","href":"/cv?upload=1"}}
</careerpilot_action>

Allowed hidden onboarding directive format:
<careerpilot_onboarding>
{{"name":"Alex","targetRoles":["Frontend Engineer"],"location":"Dhaka","workMode":"remote","careerStage":"fresh graduate","completed":false}}
</careerpilot_onboarding>

Allowed action types:
- open_route: href must be a CareerPilot route from the app map.
- prefill_job_search: include label, query, optional location, optional auto.
- prefill_goal_with_todos: include one goal and at most five todos. This opens
  Goals & Tasks with a draft; tell the user to review and click Create.
- prefill_todo: include one todo. This opens Goals & Tasks with a draft; tell
  the user to click Create.
- create_goal_with_todos: include one goal and at most five todos. This only proposes; the user must confirm.
- create_roadmap_with_tasks: include at most four goals and at most twelve todos total. This only proposes; the user must confirm.
- create_todo: include one todo. This only proposes; the user must confirm.
- save_application: include label, job_id, optional status. This only proposes; the user must confirm.
- update_application_status: include label, application_id, status. This only proposes; the user must confirm.
- save_application_note: include label, application_id, note. This only proposes; the user must confirm.
- prefill_application_note: include label, application_id, and note. This opens
  Applications with the note draft; tell the user to click Save.
- show_feature_explainer: include label, feature, optional body, optional href.
  Use this when the user needs a feature explanation before moving on.

Do not show or explain the hidden directive in visible text. Keep visible replies
concise, practical, and page-aware.

For first-run onboarding, behave like a smart conversation, not a rigid form:
- Infer only fields the user actually provided or corrected.
- If the user says "hi im alex" or "my name is alex", store the name as "Alex", never the full sentence.
- If the user corrects a prior value, acknowledge the correction and update it.
- If the user asks what is happening, pause onboarding and briefly explain that you are learning a few preferences to personalize job matches and next steps.
- Ask for only the most useful missing detail, and do not advance through hard-coded questions when the user is confused.
- Set completed to true only when name, targetRoles, location/workMode, and careerStage are known.
- Include at most one onboarding directive at the very end when a field should be updated.
"""

    system_prompt = (
        "You are CareerPilot, the user's expert career co-pilot. "
        "Your job is to help them make progress: find jobs, judge readiness, improve their CV, "
        "write tailored application material, prepare interviews, and turn advice into goals and tasks. "
        "Be warm, direct, and useful. Avoid generic chatbot filler. "
        "Always ground answers in the user's actual CV context when available. "
        "Only reference experience, skills, education, or projects that appear in the CV context. "
        "If CV context is missing, say what you can do after a CV/profile is added instead of inventing background. "
        "When giving career advice, make it specific enough that the user can act on it today. "
        "When proposing product actions, remember that navigation is safe but tracker/application mutations require explicit UI confirmation. "
        f"Current UTC timestamp: {_now_iso()}\n\n"
        f"User CV context:\n{cv_context or 'No CV data available.'}"
        f"{action_rules}"
    )

    groq_messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        groq_messages.append({"role": msg["role"], "content": msg["content"]})
    groq_messages.append({"role": "user", "content": message})

    # 3. Ensure durable session tab and save user's question to Supabase
    await ensure_chat_session(user_id, session_id, message)
    await save_message(user_id, session_id, "user", message)

    # 4. Stream response from Groq with key-pool fallback on 429
    full_reply = []
    try:
        rotation = groq_pool.rotate_on_rate_limit()
        for api_key in rotation:
            try:
                groq_client = Groq(api_key=api_key)
                model_succeeded = False
                for candidate_model in _get_candidate_models():
                    try:
                        stream = groq_client.chat.completions.create(
                            model=candidate_model,
                            messages=groq_messages,
                            stream=True,
                            temperature=0.35,
                            max_tokens=1800,
                        )
                        for chunk in stream:
                            token = chunk.choices[0].delta.content or ""
                            if token:
                                full_reply.append(token)
                                yield f"data: {json.dumps({'token': token})}\n\n"
                        model_succeeded = True
                        break
                    except Exception as model_err:
                        err_str = str(model_err)
                        if "model_not_found" in err_str or "does not exist" in err_str:
                            logger.warning(
                                "Groq model '%s' unavailable, trying fallback: %s",
                                candidate_model,
                                model_err,
                            )
                            continue
                        raise
                if model_succeeded:
                    rotation.success()
                    break
            except GroqRateLimitError:
                full_reply = []  # reset partial output before retry
                continue

        visible_reply = "".join(full_reply)
        if not _has_hidden_action(visible_reply):
            fallback_action = _fallback_action_for_turn(
                message=message,
                history=history,
                client_context=client_context,
                visible_reply=visible_reply,
            )
            if fallback_action:
                directive = _hidden_action(fallback_action)
                full_reply.append(directive)
                yield f"data: {json.dumps({'token': directive})}\n\n"

        # 5. Save assistant's answer once fully streamed
        await save_message(
            user_id,
            session_id,
            "assistant",
            _strip_copilot_actions("".join(full_reply)),
        )
        yield "data: [DONE]\n\n"

    except KeyPoolExhausted as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
