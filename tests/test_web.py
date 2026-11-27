"""Web server tests: access code and daily limit (no network, fake agent)."""

import pytest
from fastapi.testclient import TestClient

from elotravel import web


class FakeAgent:
    def send(self, text, on_tool=None, on_event=None):
        return f"echo: {text}"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web, "make_agent", FakeAgent)
    monkeypatch.setattr(web.time, "sleep", lambda s: None)
    web._sessions.clear()
    web._usage.update(day=None, count=0)
    return TestClient(web.app)


def test_open_when_no_access_code(client, monkeypatch):
    monkeypatch.delenv("ACCESS_CODE", raising=False)
    assert client.get("/api/config").json() == {"access_code_required": False}
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 200 and r.json()["reply"] == "echo: [Mode: flight + hotel package]\nhi"


def test_access_code_required(client, monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "s3cret")
    assert client.get("/api/config").json() == {"access_code_required": True}
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
    assert client.post("/api/chat", json={"message": "hi"}, headers={"X-Access-Code": "nope"}).status_code == 401
    assert client.post("/api/check-code", headers={"X-Access-Code": "nope"}).status_code == 401
    assert client.post("/api/check-code", headers={"X-Access-Code": "s3cret"}).status_code == 200
    r = client.post("/api/chat", json={"message": "hi"}, headers={"X-Access-Code": "s3cret"})
    assert r.status_code == 200
    # Rejected requests don't create sessions or use the daily allowance.
    assert len(web._sessions) == 1 and web._usage["count"] == 1


def test_daily_limit(client, monkeypatch):
    monkeypatch.delenv("ACCESS_CODE", raising=False)
    monkeypatch.setenv("DAILY_REQUEST_LIMIT", "2")
    codes = [client.post("/api/chat", json={"message": "hi"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_health_and_page(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert "codeForm" in client.get("/").text


def _lines(r):
    import json
    return [json.loads(l) for l in r.text.splitlines() if l.strip()]


def test_stream_reports_searches_then_reply(client, monkeypatch):
    monkeypatch.delenv("ACCESS_CODE", raising=False)

    class SearchingAgent:
        def send(self, text, on_tool=None, on_event=None):
            on_tool("search_flights", {"origin": "TLV", "dest": "ATH"})
            on_tool("search_hotels", {"query": "Athens"})
            return "two packages"

    monkeypatch.setattr(web, "make_agent", SearchingAgent)
    events = _lines(client.post("/api/chat/stream", json={"message": "hi"}))
    assert [e["type"] for e in events] == ["start", "tool", "tool", "reply"]
    assert events[1]["args"] == {"origin": "TLV", "dest": "ATH"}
    assert events[-1]["reply"] == "two packages" and events[-1]["session_id"] == events[0]["session_id"]


def test_stream_error_is_reported_and_session_unlocked(client, monkeypatch):
    from elotravel.agent import AssistantBusyError
    monkeypatch.delenv("ACCESS_CODE", raising=False)

    class BusyAgent:
        def send(self, text, on_tool=None, on_event=None):
            raise AssistantBusyError("quota")

    monkeypatch.setattr(web, "make_agent", BusyAgent)
    first = _lines(client.post("/api/chat/stream", json={"message": "hi"}))
    assert first[-1] == {"type": "error", "status": 429,
                         "message": "The assistant is busy or out of free quota. Please try again later. (quota)"}
    sid = first[0]["session_id"]
    again = _lines(client.post("/api/chat/stream", json={"message": "hi", "session_id": sid}))
    assert again[-1]["type"] == "error"  # not a 409: the lock was released


def test_stream_requires_access_code(client, monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "s3cret")
    assert client.post("/api/chat/stream", json={"message": "hi"}).status_code == 401


def test_mode_reaches_the_model_and_the_cards(client, monkeypatch):
    monkeypatch.delenv("ACCESS_CODE", raising=False)
    seen = {}

    class ModeAgent:
        def send(self, text, on_tool=None, on_event=None):
            seen["text"] = text
            on_event({"type": "packages", "packages": [{"label": "x"}]})
            return "ok"

    monkeypatch.setattr(web, "make_agent", ModeAgent)
    events = _lines(client.post("/api/chat/stream", json={"message": "TLV-LHR", "mode": "matmid"}))
    assert seen["text"].startswith("[Mode: El Al Matmid") and seen["text"].endswith("TLV-LHR")
    assert [e for e in events if e["type"] == "packages"][0]["mode"] == "matmid"
    assert client.post("/api/chat/stream", json={"message": "x", "mode": "cruise"}).status_code == 422


def test_result_can_be_fetched_after_a_dropped_stream(client, monkeypatch):
    monkeypatch.delenv("ACCESS_CODE", raising=False)

    class CardAgent:
        def send(self, text, on_tool=None, on_event=None):
            on_event({"type": "packages", "packages": [{"label": "x"}]})
            return "done"

    monkeypatch.setattr(web, "make_agent", CardAgent)
    start = _lines(client.post("/api/chat/stream", json={"message": "hi"}))[0]
    r = client.get("/api/chat/result", params={"session_id": start["session_id"], "turn": start["turn"]})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "done" and body["reply"] == "done"
    assert body["packages"] == [{"label": "x"}] and body["mode"] == "package"
    assert client.get("/api/chat/result", params={"session_id": start["session_id"], "turn": "nope"}).status_code == 404
    monkeypatch.setenv("ACCESS_CODE", "s3cret")
    assert client.get("/api/chat/result", params={"session_id": start["session_id"], "turn": start["turn"]}).status_code == 401
