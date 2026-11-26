"""Agent loop tests with a fake Anthropic client (no network)."""

import datetime as dt
from types import SimpleNamespace as NS

import pytest

from elotravel import agent as agent_mod
from elotravel.agent import TravelAgent


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


class FakeStream:
    def __init__(self, msg):
        self.msg = msg
        self.text_stream = iter([b.text for b in msg.content if b.type == "text"])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeStream(reply)


TODAY = dt.date(2026, 11, 26)


def test_tool_round_trip_runs_tools_in_one_message(monkeypatch):
    ran = []
    monkeypatch.setattr(agent_mod, "run_tool", lambda n, i, d: (ran.append(n) or f"{n} ok", False))
    client = FakeClient([
        NS(stop_reason="tool_use", content=[
            text("Searching..."),
            tool_use("a", "search_flights", {"origin": "TLV"}),
            tool_use("b", "search_hotels", {"query": "Athens"}),
        ]),
        NS(stop_reason="end_turn", content=[text("Best value: ...")]),
    ])
    a = TravelAgent(client=client, today=TODAY)
    reply = a.send("TLV to Athens")
    assert reply == "Searching...Best value: ..."
    assert sorted(ran) == ["search_flights", "search_hotels"]
    results = a.messages[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    call = client.calls[0]
    assert call["fallbacks"] == "default" and call["thinking"] == {"type": "adaptive"}
    assert "2026-11-26" in call["system"][1]["text"]
    assert "cache_control" in call["system"][0]


def test_refusal_rolls_back_turn():
    client = FakeClient([NS(stop_reason="refusal", content=[])])
    a = TravelAgent(client=client, today=TODAY)
    assert "can't help" in a.send("hi")
    assert a.messages == []


def test_api_error_rolls_back_and_reraises():
    client = FakeClient([RuntimeError("network")])
    a = TravelAgent(client=client, today=TODAY)
    with pytest.raises(RuntimeError):
        a.send("hi")
    assert a.messages == []


def test_max_tokens_with_pending_tool_then_next_message_merges():
    client = FakeClient([
        NS(stop_reason="max_tokens", content=[tool_use("x", "search_flights", {})]),
        NS(stop_reason="end_turn", content=[text("ok")]),
    ])
    a = TravelAgent(client=client, today=TODAY)
    assert "cut off" in a.send("hi")
    assert a.messages[-1]["role"] == "user"  # error tool_result for the pending call
    a.send("continue")
    roles = [m["role"] for m in client.calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]  # still alternating
    last = client.calls[1]["messages"][-1]["content"]
    assert last[0]["type"] == "tool_result" and last[-1] == {"type": "text", "text": "continue"}


def test_missing_credentials_is_reported_and_rolled_back(monkeypatch):
    from elotravel.agent import MissingCredentialsError

    class NoAuthClient(FakeClient):
        def _stream(self, **kwargs):
            raise TypeError('"Could not resolve authentication method. Expected one of api_key..."')

    a = TravelAgent(client=NoAuthClient([]), today=TODAY)
    with pytest.raises(MissingCredentialsError):
        a.send("hi")
    assert a.messages == []
