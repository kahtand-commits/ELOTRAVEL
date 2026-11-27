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
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import anthropic
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import AssistantBusyError, AssistantUnavailableError, MissingCredentialsError
from .providers import make_agent
from .prompt import GREETING

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


class ChatIn(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)


class ChatOut(BaseModel):
    session_id: str
    reply: str
    searches: list[str]


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
        reply = agent.send(body.message, on_tool=lambda name, _args: searches.append(name))
    except (MissingCredentialsError, anthropic.AuthenticationError):
        raise HTTPException(503, "The assistant isn't configured: its API key is missing or invalid.")
    except (AssistantBusyError, anthropic.RateLimitError):
        raise HTTPException(429, "The assistant is busy or out of free quota. Please try again later.")
    except (AssistantUnavailableError, anthropic.APIStatusError, anthropic.APIConnectionError):
        raise HTTPException(502, "The assistant is unavailable right now.")
    finally:
        lock.release()
    return ChatOut(session_id=sid, reply=reply, searches=searches)
