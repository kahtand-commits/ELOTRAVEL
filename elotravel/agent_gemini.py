"""Conversation loop on Google Gemini (free tier friendly), same interface as TravelAgent.

Set LLM_PROVIDER=gemini and GEMINI_API_KEY to use it. GEMINI_MODEL overrides the model.
"""

from __future__ import annotations

import datetime as dt
import json
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
    AssistantBusyError,
    AssistantUnavailableError,
    MissingCredentialsError,
    TextCallback,
    ToolCallback,
)
from .prompt import SYSTEM_PROMPT
from .tools import TOOLS, run_tool

DEFAULT_MODEL = "gemini-3.8-flash"

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
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.messages: list[types.Content] = []

    @property
    def today(self) -> dt.date:
        return self._today or dt.date.today()

    def _get_client(self) -> Any:
        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise MissingCredentialsError("Gemini credentials are missing. Set GEMINI_API_KEY.")
            self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=90_000))
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

    def _generate(self) -> Any:
        try:
            return self._get_client().models.generate_content(
                model=self.model, contents=self.messages, config=self._config()
            )
        except genai_errors.ClientError as e:
            if e.code == 429:
                raise AssistantBusyError("Gemini rate limit or free-tier quota reached.") from e
            if e.code in (401, 403) or "API key" in str(e):
                raise MissingCredentialsError("The Gemini API key is missing or invalid.") from e
            raise AssistantUnavailableError(f"Gemini rejected the request ({e.code}).") from e
        except genai_errors.ServerError as e:
            raise AssistantUnavailableError("Gemini is unavailable right now.") from e

    def _run_tools(self, calls: list[Any], on_tool: ToolCallback | None) -> types.Content:
        run_now, skipped = limited(calls)
        for fc in run_now:
            if on_tool:
                on_tool(fc.name, dict(fc.args or {}))
        with ThreadPoolExecutor(max_workers=max(len(run_now), 1)) as pool:
            outcomes = list(pool.map(lambda fc: run_tool(fc.name, dict(fc.args or {}), self.today), run_now))
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
    ) -> str:
        """Send a user message, run any searches the model asks for, return the reply text.

        If a call fails, the conversation is rolled back so the user can retry.
        """
        start = len(self.messages)
        try:
            return self._send(start, user_text, on_text, on_tool)
        except BaseException:
            del self.messages[start:]
            raise

    def _send(self, start: int, user_text: str, on_text: TextCallback | None, on_tool: ToolCallback | None) -> str:
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
            resp = self._generate()
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
            self.messages.append(self._run_tools(calls, on_tool))
        else:
            stop = notice("too_many", user_text)
            if on_text:
                on_text(stop)
            reply_parts.append(stop)

        return "".join(reply_parts)
