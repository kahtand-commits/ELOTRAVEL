"""Minimal web chat: uvicorn elotravel.web:app --port 8000"""

from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import TravelAgent
from .prompt import GREETING

app = FastAPI(title="ELOTRAVEL")
_STATIC = Path(__file__).parent / "static"

SESSION_TTL_S = 6 * 3600
# In-memory sessions: fine for a single process; use a store for production.
_sessions: dict[str, tuple[TravelAgent, threading.Lock, float]] = {}
_sessions_lock = threading.Lock()


class ChatIn(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)


class ChatOut(BaseModel):
    session_id: str
    reply: str
    searches: list[str]


def _session(session_id: str | None) -> tuple[str, TravelAgent, threading.Lock]:
    now = time.time()
    with _sessions_lock:
        for sid in [s for s, (_, _, t) in _sessions.items() if now - t > SESSION_TTL_S]:
            del _sessions[sid]
        if session_id and session_id in _sessions:
            agent, lock, _ = _sessions[session_id]
        else:
            session_id = secrets.token_urlsafe(16)
            agent, lock = TravelAgent(), threading.Lock()
        _sessions[session_id] = (agent, lock, now)
    return session_id, agent, lock


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/api/greeting")
def greeting() -> dict[str, str]:
    return {"reply": GREETING}


@app.post("/api/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:  # sync def: FastAPI runs it in a thread pool
    sid, agent, lock = _session(body.session_id)
    searches: list[str] = []
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "Still working on your previous message.")
    try:
        reply = agent.send(body.message, on_tool=lambda name, _args: searches.append(name))
    except anthropic.RateLimitError:
        raise HTTPException(429, "The assistant is busy. Please try again in a moment.")
    except (anthropic.APIStatusError, anthropic.APIConnectionError):
        raise HTTPException(502, "The assistant is unavailable right now.")
    finally:
        lock.release()
    return ChatOut(session_id=sid, reply=reply, searches=searches)
