"""Conversation loop on Google Gemini (free tier friendly), same interface as TravelAgent.

Set LLM_PROVIDER=gemini and GEMINI_API_KEY to use it. GEMINI_MODEL sets the model, or a
comma-separated list of models to fall back through when one is overloaded, out of
free quota, or not available (e.g. "gemini-3.5-flash,gemini-2.5-flash").
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .agent import (
    MAX_TOOL_ROUNDS,
    SKIPPED_RESULT,
    TURN_DEADLINE_S,
    limited,
    notice,
    closing_notice,
    packages_shown,
    AssistantBusyError,
    AssistantUnavailableError,
    MissingCredentialsError,
    EventCallback,
    TextCallback,
    ToolCallback,
)
from .prompt import SYSTEM_PROMPT
from .tools import TOOLS, run_tool

# Tried in order. Lighter models first: they usually have more free-tier quota and
# less demand. Once one works, the conversation stays on it.
# gemini-2.5-* returned 404 "no longer available to new users" (Nov 2026).
DEFAULT_MODELS = "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-3.8-flash"
# Errors that mean "try the next model": quota/rate limit, overload, model not found.
FALLBACK_CODES = {404, 429, 500, 502, 503, 504}
# A 429 often means "too many requests this minute": Gemini says how long to wait.
MAX_WAIT_S = 60


def retry_delay(e: Any) -> float | None:
    """Seconds Gemini asks us to wait (RetryInfo.retryDelay, e.g. "37s"), if any."""
    details = getattr(e, "details", None) or {}
    for d in (details.get("error") or {}).get("details") or []:
        if isinstance(d, dict) and str(d.get("@type", "")).endswith("RetryInfo"):
            try:
                return float(str(d.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                return None
    return None
log = logging.getLogger("elotravel.gemini")

_FUNCTIONS = [
    types.FunctionDeclaration(
        name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"]
    )
    for t in TOOLS
]


class GeminiTravelAgent:
    """One user's conversation. Not thread-safe; use one instance per session."""

    def __init__(self, client: Any = None, today: dt.date | None = None, model: str | None = None):
        self._client = client
        self._today = today
        names = model or os.environ.get("GEMINI_MODEL", "") or DEFAULT_MODELS
        self.models = [m.strip() for m in names.split(",") if m.strip()]
        self._model_i = 0
        self.messages: list[types.Content] = []
        self._sleep = time.sleep

    @property
    def model(self) -> str:
        return self.models[self._model_i]

    @property
    def today(self) -> dt.date:
        return self._today or dt.date.today()

    def _get_client(self) -> Any:
        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise MissingCredentialsError("Gemini credentials are missing. Set GEMINI_API_KEY.")
            self._client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=90_000,
                    # Free-tier models are often briefly overloaded: retry a few times.
                    retry_options=types.HttpRetryOptions(
                        # Not 429: retrying a quota error only uses up more quota.
                        attempts=2, initial_delay=2, max_delay=6,
                        http_status_codes=[500, 502, 503, 504],
                    ),
                ),
            )
        return self._client

    def _config(self) -> types.GenerateContentConfig:
        date_line = (
            f"Today's date is {self.today.isoformat()} ({self.today:%A}). "
            "Resolve relative dates against it and never search past dates."
        )
        return types.GenerateContentConfig(
            system_instruction=f"{SYSTEM_PROMPT}\n\n{date_line}",
            tools=[types.Tool(function_declarations=_FUNCTIONS)],
            # We run the searches ourselves (validation, parallel calls).
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _generate(self, on_event: EventCallback | None = None) -> Any:
        client = self._get_client()
        waited: set[str] = set()
        i = self._model_i
        while i < len(self.models):
            name = self.models[i]
            try:
                resp = client.models.generate_content(model=name, contents=self.messages, config=self._config())
                self._model_i = i  # stay on the model that works
                return resp
            except genai_errors.APIError as e:
                detail = f"Gemini {e.code} {e.status or ''}: {(e.message or '')[:200]}".strip()
                log.warning("Gemini request failed (model %s): %s", name, detail)
                delay = retry_delay(e) if e.code == 429 else None
                if delay is not None and delay <= MAX_WAIT_S and name not in waited:
                    # Per-minute limit: wait as asked and try the same model once more.
                    waited.add(name)
                    log.warning("Waiting %.0f s for model %s", delay, name)
                    if on_event:
                        on_event({"type": "wait", "seconds": int(delay) + 1})
                    self._sleep(delay + 1)
                    continue
                if e.code in FALLBACK_CODES and i + 1 < len(self.models):
                    log.warning("Falling back to model %s", self.models[i + 1])
                    i += 1
                    continue
                if e.code == 429:
                    raise AssistantBusyError(detail) from e
                if e.code in (401, 403) or "API key" in (e.message or ""):
                    raise MissingCredentialsError(detail) from e
                raise AssistantUnavailableError(detail) from e
            except Exception as e:  # network errors and timeouts
                log.warning("Could not reach Gemini: %r", e)
                raise AssistantUnavailableError(f"Could not reach Gemini ({type(e).__name__})") from e
        raise AssistantUnavailableError("No Gemini model is configured.")

    def _run_tools(self, calls: list[Any], on_tool: ToolCallback | None, on_event: EventCallback | None = None) -> types.Content:
        run_now, skipped = limited(calls)
        for fc in run_now:
            if on_tool:
                on_tool(fc.name, dict(fc.args or {}))
        with ThreadPoolExecutor(max_workers=max(len(run_now), 1)) as pool:
            outcomes = list(pool.map(lambda fc: run_tool(fc.name, dict(fc.args or {}), self.today, on_event), run_now))
        outcomes += [(SKIPPED_RESULT, True)] * skipped
        parts = []
        for fc, (text, is_error) in zip(calls, outcomes):
            payload = {"error": text} if is_error else {"result": json.loads(text)}
            part = types.Part.from_function_response(name=fc.name, response=payload)
            if getattr(fc, "id", None):
                part.function_response.id = fc.id
            parts.append(part)
        # All results go back in one message.
        return types.Content(role="user", parts=parts)

    def send(
        self,
        user_text: str,
        on_text: TextCallback | None = None,
        on_tool: ToolCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> str:
        """Send a user message, run any searches the model asks for, return the reply text.

        If a call fails, the conversation is rolled back so the user can retry.
        """
        start = len(self.messages)
        try:
            return self._send(start, user_text, on_text, on_tool, on_event)
        except BaseException:
            del self.messages[start:]
            raise

    def _send(self, start: int, user_text: str, on_text: TextCallback | None, on_tool: ToolCallback | None,
              on_event: EventCallback | None = None) -> str:
        self.messages.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
        reply_parts: list[str] = []

        t0 = time.monotonic()
        for round_no in range(MAX_TOOL_ROUNDS):
            if round_no and time.monotonic() - t0 > TURN_DEADLINE_S:
                slow = notice("slow", user_text)
                if on_text:
                    on_text(slow)
                reply_parts.append(slow)
                break
            resp = self._generate(on_event)
            candidate = (resp.candidates or [None])[0]
            content = getattr(candidate, "content", None)
            if content is None or not content.parts:
                # Blocked by safety filters or an empty answer: drop the turn.
                del self.messages[start:]
                msg_ = notice("refusal", user_text)
                if on_text:
                    on_text(msg_)
                return msg_

            # Keep the model's content as-is (it can carry thought signatures).
            self.messages.append(content)
            text = "".join(p.text for p in content.parts if p.text and not p.thought)
            if text:
                reply_parts.append(text)
                if on_text:
                    on_text(text)

            calls = [p.function_call for p in content.parts if p.function_call]
            if not calls:
                break
            results = self._run_tools(calls, on_tool, on_event)
            self.messages.append(results)
            if packages_shown([fc.name for fc in calls],
                              ["error" in (p.function_response.response or {}) for p in results.parts]):
                closing = text.strip() or closing_notice([fc.name for fc in calls], user_text)
                if not reply_parts:
                    reply_parts.append(closing)
                    if on_text:
                        on_text(closing)
                # Keep user/model turns alternating for the next message.
                self.messages.append(types.Content(role="model", parts=[types.Part.from_text(text=closing)]))
                break
        else:
            stop = notice("too_many", user_text)
            if on_text:
                on_text(stop)
            reply_parts.append(stop)

        return "".join(reply_parts)
