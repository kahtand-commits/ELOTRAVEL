"""Conversation loop: Claude + the flight/hotel search tools."""

from __future__ import annotations

import datetime as dt
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import anthropic

from .prompt import SYSTEM_PROMPT
from .tools import TOOLS, run_tool

MODEL = os.environ.get("ELOTRAVEL_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("ELOTRAVEL_EFFORT", "medium")
MAX_TOOL_ROUNDS = 8
MAX_SEARCHES_PER_ROUND = 6
TURN_DEADLINE_S = 150  # stop starting new rounds after this many seconds

_NOTICES = {
    "refusal": ("Sorry, I can't help with that request. Could you rephrase it?",
                "מצטער, אני לא יכול לעזור עם הבקשה הזו. אפשר לנסח אותה אחרת?"),
    "too_many": ("\n\n(I stopped after too many searches in one turn. Tell me how to narrow it down.)",
                 "\n\n(עצרתי אחרי יותר מדי חיפושים. אפשר לצמצם, למשל תאריכים מדויקים?)"),
    "slow": ("\n\n(The search took too long, so I stopped here. Give me exact dates or fewer options and I'll search again.)",
             "\n\n(החיפוש לקח יותר מדי זמן ועצרתי. תן לי תאריכים מדויקים או פחות אפשרויות ואחפש שוב.)"),
    "cut": ("\n\n(The reply was cut off. Ask me to continue.)",
            "\n\n(התשובה נקטעה. בקש ממני להמשיך.)"),
}
_HEBREW = re.compile(r"[\u0590-\u05FF]")


def notice(key: str, user_text: str) -> str:
    en, he = _NOTICES[key]
    return he if _HEBREW.search(user_text or "") else en


def limited(calls: list[Any]) -> tuple[list[Any], int]:
    """Split tool calls into the ones to run now and how many are skipped."""
    return calls[:MAX_SEARCHES_PER_ROUND], max(len(calls) - MAX_SEARCHES_PER_ROUND, 0)


SKIPPED_RESULT = (
    f"Not run: at most {MAX_SEARCHES_PER_ROUND} searches per step. "
    "Run the most important ones first, or ask the user to narrow the dates."
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Server-side refusal fallback exists for Opus 5.x, Sonnet 5.5 and Fable; not for Haiku 5.5.
_FALLBACK_MODELS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5")


def fallback_args(model: str) -> dict[str, Any]:
    return {"betas": [FALLBACK_BETA], "fallbacks": "default"} if model.startswith(_FALLBACK_MODELS) else {}


FALLBACK_ARGS = fallback_args(MODEL)

TextCallback = Callable[[str], None]


class MissingCredentialsError(RuntimeError):
    """No usable API key for the selected model provider."""


class AssistantBusyError(RuntimeError):
    """The model provider is rate limiting us (or a free-tier quota ran out)."""


class AssistantUnavailableError(RuntimeError):
    """The model provider failed or rejected the request."""

ToolCallback = Callable[[str, dict], None]
EventCallback = Callable[[dict], None]  # page events such as package cards


class TravelAgent:
    """One user's conversation. Not thread-safe; use one instance per session."""

    def __init__(self, client: anthropic.Anthropic | None = None, today: dt.date | None = None):
        self.client = client or anthropic.Anthropic(timeout=120.0)
        self._today = today
        self.messages: list[dict[str, Any]] = []

    @property
    def today(self) -> dt.date:
        return self._today or dt.date.today()

    def _system(self) -> list[dict[str, Any]]:
        return [
            # Stable prompt first, cached; the date sits after the breakpoint.
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
            {
                "type": "text",
                "text": f"Today's date is {self.today.isoformat()} ({self.today:%A}). "
                "Resolve relative dates against it and never search past dates.",
            },
        ]

    def _run_tools(self, blocks: list[Any], on_tool: ToolCallback | None, on_event: EventCallback | None = None) -> list[dict[str, Any]]:
        run_now, skipped = limited(blocks)
        for b in run_now:
            if on_tool:
                on_tool(b.name, b.input if isinstance(b.input, dict) else {})
        # Searches are slow network calls; run them concurrently.
        with ThreadPoolExecutor(max_workers=max(len(run_now), 1)) as pool:
            outcomes = list(pool.map(lambda b: run_tool(b.name, b.input, self.today, on_event), run_now))
        outcomes += [(SKIPPED_RESULT, True)] * skipped
        # All results go back in a single user message.
        return [
            {"type": "tool_result", "tool_use_id": b.id, "content": text, "is_error": is_error}
            for b, (text, is_error) in zip(blocks, outcomes)
        ]

    def _open_stream(self):
        return self.client.beta.messages.stream(
            model=MODEL,
            max_tokens=64000,
            system=self._system(),
            tools=TOOLS,
            messages=self.messages,
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT},
            cache_control={"type": "ephemeral"},
            **FALLBACK_ARGS,
        )

    def send(
        self,
        user_text: str,
        on_text: TextCallback | None = None,
        on_tool: ToolCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> str:
        """Send a user message, run any searches Claude asks for, return the reply text.

        If the API call fails, the conversation is rolled back to before this
        message so the user can simply retry.
        """
        snapshot = (len(self.messages), self.messages[-1]["content"] if self.messages else None)
        try:
            return self._send(user_text, on_text, on_tool, on_event)
        except BaseException:
            del self.messages[snapshot[0]:]
            if self.messages:
                self.messages[-1]["content"] = snapshot[1]
            raise

    def _send(
        self,
        user_text: str,
        on_text: TextCallback | None,
        on_tool: ToolCallback | None,
        on_event: EventCallback | None = None,
    ) -> str:
        start = len(self.messages)
        last = self.messages[-1] if self.messages else None
        restore = None  # (message, original content) to undo a merge on refusal
        if last and last["role"] == "user":
            # A previous turn ended on tool results (cut off or round limit).
            # Keep roles alternating by adding the new text to that message.
            restore = (last, last["content"])
            prev = last["content"]
            if isinstance(prev, str):
                prev = [{"type": "text", "text": prev}]
            last["content"] = list(prev) + [{"type": "text", "text": user_text}]
        else:
            self.messages.append({"role": "user", "content": user_text})
        reply_parts: list[str] = []

        t0 = time.monotonic()
        for round_no in range(MAX_TOOL_ROUNDS):
            if round_no and time.monotonic() - t0 > TURN_DEADLINE_S:
                slow = notice("slow", user_text)
                if on_text:
                    on_text(slow)
                reply_parts.append(slow)
                break
            try:
                with self._open_stream() as stream:
                    for text in stream.text_stream:
                        if on_text:
                            on_text(text)
                    msg = stream.get_final_message()
            except TypeError as e:
                # The SDK raises TypeError when it can't find any credentials.
                if "authentication method" in str(e):
                    raise MissingCredentialsError(
                        "Anthropic credentials are missing. Set ANTHROPIC_API_KEY."
                    ) from e
                raise

            if msg.stop_reason == "refusal":
                # Every model in the fallback chain declined. Drop this turn so
                # the conversation can continue cleanly.
                del self.messages[start:]
                if restore:
                    restore[0]["content"] = restore[1]
                msg_ = notice("refusal", user_text)
                if on_text:
                    on_text("\n" + msg_)
                return msg_

            self.messages.append({"role": "assistant", "content": msg.content})
            reply_parts.extend(b.text for b in msg.content if b.type == "text")

            if msg.stop_reason == "max_tokens":
                pending = [b for b in msg.content if b.type == "tool_use"]
                if pending:  # every tool_use needs a result before the next request
                    self.messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                         "content": "Not run: the reply was cut off."} for b in pending]})
                cut = notice("cut", user_text)
                if on_text:
                    on_text(cut)
                reply_parts.append(cut)
                break

            tool_uses = [b for b in msg.content if b.type == "tool_use"]
            if msg.stop_reason != "tool_use" or not tool_uses:
                break
            self.messages.append({"role": "user", "content": self._run_tools(tool_uses, on_tool, on_event)})
        else:
            stop = notice("too_many", user_text)
            if on_text:
                on_text(stop)
            reply_parts.append(stop)

        return "".join(reply_parts)
