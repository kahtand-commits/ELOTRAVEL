"""Gemini agent loop and provider selection, with a fake client (no network)."""

import datetime as dt
import json
from types import SimpleNamespace as NS

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from elotravel import agent_gemini, providers
from elotravel.agent import AssistantBusyError, MissingCredentialsError
from elotravel.agent_gemini import GeminiTravelAgent

TODAY = dt.date(2026, 11, 27)


def resp(*parts):
    return NS(candidates=[NS(content=types.Content(role="model", parts=list(parts)))])


def call(name, args, id_):
    return types.Part(function_call=types.FunctionCall(name=name, args=args, id=id_))


class FakeModels:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def client(replies):
    return NS(models=FakeModels(replies))


def test_tool_round_trip(monkeypatch):
    ran = []
    monkeypatch.setattr(
        agent_gemini, "run_tool",
        lambda n, a, d: (ran.append((n, a)) or (json.dumps({"flights": [1]}), False)) if n == "search_flights"
        else ("Search failed: nope", True),
    )
    c = client([
        resp(types.Part(text="Searching..."), call("search_flights", {"origin": "TLV"}, "c1"),
             call("search_hotels", {"query": "Athens"}, "c2")),
        resp(types.Part(text="Best value: ...")),
    ])
    a = GeminiTravelAgent(client=c, today=TODAY, model="test-model")
    assert a.send("TLV to Athens") == "Searching...Best value: ..."
    assert ran == [("search_flights", {"origin": "TLV"})]
    results = a.messages[2].parts
    assert [p.function_response.id for p in results] == ["c1", "c2"]
    assert results[0].function_response.response == {"result": {"flights": [1]}}
    assert results[1].function_response.response == {"error": "Search failed: nope"}
    cfg = c.models.calls[0]["config"]
    assert "2026-11-27" in cfg.system_instruction
    assert {f.name for f in cfg.tools[0].function_declarations} == {"search_flights", "search_hotels", "get_return_flights"}
    assert cfg.automatic_function_calling.disable is True


def test_thought_text_is_not_shown():
    c = client([resp(types.Part(text="internal", thought=True), types.Part(text="Hi!"))])
    assert GeminiTravelAgent(client=c, today=TODAY).send("hi") == "Hi!"


def test_blocked_answer_rolls_back():
    a = GeminiTravelAgent(client=client([NS(candidates=[])]), today=TODAY)
    assert "can't help" in a.send("hi")
    assert a.messages == []


def test_rate_limit_maps_and_rolls_back():
    err = genai_errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    a = GeminiTravelAgent(client=client([err]), today=TODAY)
    with pytest.raises(AssistantBusyError):
        a.send("hi")
    assert a.messages == []


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(MissingCredentialsError):
        GeminiTravelAgent(today=TODAY).send("hi")


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"LLM_PROVIDER": "gemini"}, "gemini"),
        ({"LLM_PROVIDER": "claude"}, "anthropic"),
        ({"GEMINI_API_KEY": "x"}, "gemini"),
        ({"GEMINI_API_KEY": "x", "ANTHROPIC_API_KEY": "y"}, "anthropic"),
        ({}, "anthropic"),
    ],
)
def test_provider_selection(monkeypatch, env, expected):
    for k in ("LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert providers.provider_name() == expected
