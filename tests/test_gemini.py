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
        lambda n, a, d, sink=None: (ran.append((n, a)) or (json.dumps({"flights": [1]}), False)) if n == "search_flights"
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
    assert {f.name for f in cfg.tools[0].function_declarations} == {"search_flights", "search_hotels", "get_return_flights", "find_cheapest_dates", "show_packages", "ask_user"}
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
    a = GeminiTravelAgent(client=client([err]), today=TODAY, model="m")
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


def test_server_error_reason_is_kept(monkeypatch):
    from elotravel.agent import AssistantUnavailableError
    err = genai_errors.ServerError(503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}})
    a = GeminiTravelAgent(client=client([err]), today=TODAY, model="m")
    with pytest.raises(AssistantUnavailableError, match="503 UNAVAILABLE: The model is overloaded"):
        a.send("hi")


def test_bad_request_reason_is_kept():
    from elotravel.agent import AssistantUnavailableError
    err = genai_errors.ClientError(404, {"error": {"code": 404, "message": "models/x is not found", "status": "NOT_FOUND"}})
    with pytest.raises(AssistantUnavailableError, match="404 NOT_FOUND: models/x is not found"):
        GeminiTravelAgent(client=client([err]), today=TODAY, model="m").send("hi")


def test_network_error_maps_to_unavailable():
    from elotravel.agent import AssistantUnavailableError
    with pytest.raises(AssistantUnavailableError, match="Could not reach Gemini"):
        GeminiTravelAgent(client=client([TimeoutError("slow")]), today=TODAY).send("hi")


def _err(cls, code, status, msg):
    return cls(code, {"error": {"code": code, "message": msg, "status": status}})


def test_falls_back_to_next_model_and_stays_there():
    quota = _err(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", "quota")
    c = client([quota, resp(types.Part(text="first")), resp(types.Part(text="second"))])
    a = GeminiTravelAgent(client=c, today=TODAY, model="model-a, model-b")
    assert a.send("hi") == "first"
    assert a.send("again") == "second"
    assert [call["model"] for call in c.models.calls] == ["model-a", "model-b", "model-b"]


def test_overloaded_and_missing_models_fall_back():
    busy = _err(genai_errors.ServerError, 503, "UNAVAILABLE", "high demand")
    missing = _err(genai_errors.ClientError, 404, "NOT_FOUND", "not found")
    c = client([busy, missing, resp(types.Part(text="ok"))])
    a = GeminiTravelAgent(client=c, today=TODAY, model="a,b,c")
    assert a.send("hi") == "ok" and a.model == "c"


def test_all_models_out_of_quota_reports_busy():
    quota = _err(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", "quota")
    a = GeminiTravelAgent(client=client([quota, quota]), today=TODAY, model="a,b")
    with pytest.raises(AssistantBusyError, match="429"):
        a.send("hi")
    assert a.messages == []


def test_bad_request_does_not_fall_back():
    from elotravel.agent import AssistantUnavailableError
    bad = _err(genai_errors.ClientError, 400, "INVALID_ARGUMENT", "bad")
    c = client([bad])
    with pytest.raises(AssistantUnavailableError, match="400"):
        GeminiTravelAgent(client=c, today=TODAY, model="a,b").send("hi")
    assert len(c.models.calls) == 1


def test_default_model_list(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert GeminiTravelAgent(client=client([]), today=TODAY).models[0] == "gemini-3.5-flash-lite"
    monkeypatch.setenv("GEMINI_MODEL", "x , y")
    assert GeminiTravelAgent(client=client([]), today=TODAY).models == ["x", "y"]


def _quota_with_delay(seconds):
    return genai_errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
        "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": f"{seconds}s"}]}})


def test_waits_when_gemini_asks_then_retries_same_model():
    c = client([_quota_with_delay(12), resp(types.Part(text="ok"))])
    a = GeminiTravelAgent(client=c, today=TODAY, model="a,b")
    slept, events = [], []
    a._sleep = slept.append
    assert a.send("hi", on_event=events.append) == "ok"
    assert slept == [13] and events == [{"type": "wait", "seconds": 13}]
    assert [x["model"] for x in c.models.calls] == ["a", "a"]


def test_long_delay_falls_back_instead_of_waiting():
    c = client([_quota_with_delay(3600), resp(types.Part(text="ok"))])
    a = GeminiTravelAgent(client=c, today=TODAY, model="a,b")
    a._sleep = lambda s: pytest.fail("should not wait an hour")
    assert a.send("hi") == "ok" and a.model == "b"


def test_turn_ends_after_packages_are_shown(monkeypatch):
    monkeypatch.setattr(agent_gemini, "run_tool", lambda n, a, d, sink=None: ('{"shown_to_user": []}', False))
    c = client([resp(types.Part(text="הנה 3 אפשרויות."), call("show_packages", {"packages": []}, "p1"))])
    a = GeminiTravelAgent(client=c, today=TODAY, model="m")
    assert a.send("פריז") == "הנה 3 אפשרויות."
    assert len(c.models.calls) == 1  # no extra model call after the cards
    assert [m.role for m in a.messages] == ["user", "model", "user", "model"]
