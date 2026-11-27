"""Minimal web chat: uvicorn elotravel.web:app --port 8000

Settings (environment variables):
  ACCESS_CODE          If set, every chat request must send this code. Always set it
                       when the app is reachable from the internet, or anyone with the
                       link can spend your API credit.
  DAILY_REQUEST_LIMIT  Max chat messages per day (UTC) for the whole server. Default 50.
"""

from __future__ import annotations

import datetime as dt
import hmac
import json
import logging
import os
import queue
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Literal

import anthropic
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import AssistantBusyError, AssistantUnavailableError, MissingCredentialsError
from .providers import make_agent
from .prompt import GREETING

log = logging.getLogger("elotravel")
app = FastAPI(title="ELOTRAVEL")
_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")

SESSION_TTL_S = 6 * 3600
# In-memory sessions: fine for a single process; use a store for production.
_sessions: dict[str, tuple[Any, threading.Lock, float]] = {}
_sessions_lock = threading.Lock()

# Server-wide daily message counter (in memory: it resets if the server restarts).
_usage = {"day": None, "count": 0}
_usage_lock = threading.Lock()


MODES = {
    "package": "[Mode: flight + hotel package]",
    "flight": "[Mode: flights only. Search and show flight-only options; no hotel. The budget is for flights.]",
    "hotel": "[Mode: hotels only. Search and show hotel-only options; no flights. The budget is for the stay.]",
    "matmid": (
        "[Mode: El Al Matmid award help. Search El Al flights only (airline LY) and show them as flight-only "
        "options. You cannot see award seat availability or points prices: say so in one sentence and point to "
        "the El Al button and points calculator on each card. Never state points prices.]"
    ),
}


class ChatIn(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)
    mode: Literal["package", "flight", "hotel", "matmid"] = "package"

    def for_agent(self) -> str:
        """The message with the app's mode switch in front, for the model."""
        return f"{MODES[self.mode]}\n{self.message}"


class ChatOut(BaseModel):
    session_id: str
    reply: str
    searches: list[str]
    packages: list[dict[str, Any]] = []


def _access_code() -> str:
    return os.environ.get("ACCESS_CODE", "")


def _daily_limit() -> int:
    try:
        return max(int(os.environ.get("DAILY_REQUEST_LIMIT", "50")), 0)
    except ValueError:
        return 50


def _check_access(code: str | None) -> None:
    expected = _access_code()
    if not expected:
        return
    if not code or not hmac.compare_digest(code.encode(), expected.encode()):
        time.sleep(1)  # slow down guessing
        raise HTTPException(401, "Wrong or missing access code.")


def _count_request() -> None:
    today = dt.datetime.now(dt.timezone.utc).date()
    with _usage_lock:
        if _usage["day"] != today:
            _usage["day"], _usage["count"] = today, 0
        if _usage["count"] >= _daily_limit():
            raise HTTPException(429, "The daily message limit has been reached. Please try again tomorrow.")
        _usage["count"] += 1


def _session(session_id: str | None) -> tuple[str, Any, threading.Lock]:
    now = time.time()
    with _sessions_lock:
        for sid in [s for s, (_, _, t) in _sessions.items() if now - t > SESSION_TTL_S]:
            del _sessions[sid]
        if session_id and session_id in _sessions:
            agent, lock, _ = _sessions[session_id]
        else:
            session_id = secrets.token_urlsafe(16)
            agent, lock = make_agent(), threading.Lock()
        _sessions[session_id] = (agent, lock, now)
    return session_id, agent, lock


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict[str, bool]:
    return {"access_code_required": bool(_access_code())}


@app.get("/api/greeting")
def greeting() -> dict[str, str]:
    return {"reply": GREETING}


@app.post("/api/check-code")
def check_code(x_access_code: str | None = Header(default=None)) -> dict[str, bool]:
    _check_access(x_access_code)
    return {"ok": True}


@app.post("/api/chat", response_model=ChatOut)
def chat(body: ChatIn, x_access_code: str | None = Header(default=None)) -> ChatOut:
    # sync def: FastAPI runs it in a thread pool
    _check_access(x_access_code)
    sid, agent, lock = _session(body.session_id)
    searches: list[str] = []
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "Still working on your previous message.")
    try:
        _count_request()
        cards: list[dict[str, Any]] = []
        reply = agent.send(
            body.for_agent(),
            on_tool=lambda name, _args: searches.append(name),
            on_event=lambda ev: cards.extend(ev.get("packages", [])),
        )
    except (MissingCredentialsError, anthropic.AuthenticationError):
        raise HTTPException(503, "The assistant isn't configured: its API key is missing or invalid.")
    except (AssistantBusyError, anthropic.RateLimitError):
        raise HTTPException(429, "The assistant is busy or out of free quota. Please try again later.")
    except (AssistantUnavailableError, anthropic.APIStatusError, anthropic.APIConnectionError):
        raise HTTPException(502, "The assistant is unavailable right now.")
    finally:
        lock.release()
    return ChatOut(session_id=sid, reply=reply, searches=searches, packages=cards)


def _error_payload(exc: BaseException) -> dict[str, Any]:
    log.warning("chat turn failed: %r", exc)
    # Our own errors carry a short, key-free reason (e.g. "Gemini 503 UNAVAILABLE: overloaded").
    reason = f" ({exc})" if isinstance(exc, (MissingCredentialsError, AssistantBusyError, AssistantUnavailableError)) and str(exc) else ""
    if isinstance(exc, (MissingCredentialsError, anthropic.AuthenticationError)):
        return {"status": 503, "message": "The assistant isn't configured: its API key is missing or invalid." + reason}
    if isinstance(exc, (AssistantBusyError, anthropic.RateLimitError)):
        return {"status": 429, "message": "The assistant is busy or out of free quota. Please try again later." + reason}
    if isinstance(exc, (AssistantUnavailableError, anthropic.APIStatusError, anthropic.APIConnectionError)):
        return {"status": 502, "message": "The assistant is unavailable right now." + reason}
    log.exception("unexpected error in chat turn", exc_info=exc)
    return {"status": 500, "message": "Something went wrong. Please try again."}


@app.post("/api/chat/stream")
def chat_stream(body: ChatIn, x_access_code: str | None = Header(default=None)) -> StreamingResponse:
    """Same as /api/chat, but streams progress as JSON lines while searches run:
    {"type":"tool",...} for each search, then {"type":"reply",...} or {"type":"error",...}.
    A {"type":"ping"} line every 10 s keeps the connection alive."""
    _check_access(x_access_code)
    sid, agent, lock = _session(body.session_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "Still working on your previous message.")
    try:
        _count_request()
    except HTTPException:
        lock.release()
        raise

    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            reply = agent.send(
                body.for_agent(),
                on_tool=lambda name, args: events.put({"type": "tool", "name": name, "args": args}),
                on_event=lambda ev: events.put({**ev, "mode": body.mode}),
            )
            events.put({"type": "reply", "session_id": sid, "reply": reply})
        except Exception as exc:  # reported to the page, never raised into the stream
            events.put({"type": "error", **_error_payload(exc)})
        finally:
            lock.release()
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def lines():
        yield json.dumps({"type": "start", "session_id": sid}) + "\n"
        while True:
            try:
                event = events.get(timeout=10)
            except queue.Empty:
                yield '{"type": "ping"}\n'
                continue
            if event is None:
                return
            yield json.dumps(event, ensure_ascii=False, default=str) + "\n"

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
