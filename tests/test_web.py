"""Web server tests: access code and daily limit (no network, fake agent)."""

import pytest
from fastapi.testclient import TestClient

from elotravel import web


class FakeAgent:
    def send(self, text, on_tool=None):
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
    assert r.status_code == 200 and r.json()["reply"] == "echo: hi"


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
