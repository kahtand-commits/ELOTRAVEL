import datetime as dt
import json

import pytest

from elotravel import serpapi, tools

TODAY = dt.date(2026, 11, 26)

FLIGHTS_PAYLOAD = {
    "search_metadata": {"google_flights_url": "https://www.google.com/travel/flights?x"},
    "price_insights": {"price_level": "typical", "typical_price_range": [300, 450]},
    "best_flights": [
        {
            "price": 410,
            "total_duration": 175,
            "flights": [
                {
                    "airline": "Aegean",
                    "flight_number": "A3 929",
                    "duration": 175,
                    "departure_airport": {"id": "TLV", "time": "2027-03-10 06:00"},
                    "arrival_airport": {"id": "ATH", "time": "2027-03-10 08:55"},
                }
            ],
        }
    ],
    "other_flights": [
        {
            "price": 280,
            "total_duration": 400,
            "layovers": [{"id": "LCA", "duration": 120}],
            "flights": [
                {"airline": "Wizz Air", "departure_airport": {"id": "TLV"}, "arrival_airport": {"id": "LCA"}},
                {"airline": "Wizz Air", "departure_airport": {"id": "LCA"}, "arrival_airport": {"id": "ATH"}},
            ],
        },
        {"total_duration": 200, "flights": []},  # unpriced: dropped
    ],
}

HOTELS_PAYLOAD = {
    "search_metadata": {"google_hotels_url": "https://www.google.com/travel/hotels?y"},
    "properties": [
        {
            "name": "Plaka Hotel",
            "type": "hotel",
            "extracted_hotel_class": 3,
            "overall_rating": 4.5,
            "reviews": 2100,
            "rate_per_night": {"lowest": "$120", "extracted_lowest": 120},
            "total_rate": {"lowest": "$600", "extracted_lowest": 600},
            "link": "https://plakahotel.example",
            "nearby_places": [{"name": "Acropolis"}],
        },
        {"name": "No Price Inn"},  # unpriced: dropped
    ],
}


def test_normalize_flights_sorts_and_drops_unpriced():
    out = serpapi.normalize_flights(FLIGHTS_PAYLOAD, "USD")
    assert [f["price"] for f in out["flights"]] == [280, 410]
    cheap = out["flights"][0]
    assert cheap["stops"] == 1 and cheap["airlines"] == ["Wizz Air"]
    assert cheap["layovers"] == [{"airport": "LCA", "duration_min": 120}]
    assert out["results_link"].startswith("https://www.google.com/travel/flights")
    assert out["flights"][1]["outbound_legs"][0]["flight_number"] == "A3 929"


def test_normalize_hotels():
    out = serpapi.normalize_hotels(HOTELS_PAYLOAD, "EUR")
    assert len(out["hotels"]) == 1
    h = out["hotels"][0]
    assert (h["price_per_night"], h["price_total_stay"], h["rating"], h["hotel_class"]) == (120, 600, 4.5, 3)
    assert h["nearby"] == ["Acropolis"] and out["currency"] == "EUR"


def test_validate_flights_ok_and_normalizes():
    args = tools.validate_flights(
        {"origin": "tlv", "dest": "ATH", "depart": "2027-03-10", "ret": "2027-03-15", "adults": 2}, TODAY
    )
    assert args["origin"] == "TLV" and args["children"] == 0 and args["currency"] == "USD"


@pytest.mark.parametrize(
    "patch, msg",
    [
        ({"depart": "2025-01-01"}, "in the past"),
        ({"ret": "2027-03-09"}, "after depart"),
        ({"origin": "Tel Aviv"}, "IATA"),
        ({"adults": 0}, "between 1 and 9"),
        ({"adults": True}, "between 1 and 9"),
        ({"bogus": 1}, "Unknown parameters"),
    ],
)
def test_validate_flights_rejects(patch, msg):
    base = {"origin": "TLV", "dest": "ATH", "depart": "2027-03-10", "ret": "2027-03-15", "adults": 2}
    with pytest.raises(tools.ToolInputError, match=msg):
        tools.validate_flights({**base, **patch}, TODAY)


def test_validate_flights_missing_required():
    # A truncated eager-streamed input can arrive without required keys.
    with pytest.raises(tools.ToolInputError, match="Missing required"):
        tools.validate_flights({"origin": "TLV", "dest": "ATH"}, TODAY)


def test_validate_hotels_rejects_bad_rating_and_order():
    base = {"query": "Athens", "check_in": "2027-03-10", "check_out": "2027-03-15", "adults": 2}
    assert tools.validate_hotels(base, TODAY)["area"] is None
    with pytest.raises(tools.ToolInputError, match="min_rating"):
        tools.validate_hotels({**base, "min_rating": 4.2}, TODAY)
    with pytest.raises(tools.ToolInputError, match="after check_in"):
        tools.validate_hotels({**base, "check_out": "2027-03-10"}, TODAY)


def test_run_tool_end_to_end_with_stubbed_http(monkeypatch):
    seen = {}

    def fake_get(params):
        seen.update(params)
        return HOTELS_PAYLOAD

    monkeypatch.setattr(serpapi, "_get", fake_get)
    text, is_error = tools.run_tool(
        "search_hotels",
        {"query": "Athens hotels", "area": "Plaka", "check_in": "2027-03-10",
         "check_out": "2027-03-15", "adults": 2, "min_rating": 4.0, "hotel_class": [3, 4]},
        TODAY,
    )
    assert not is_error
    assert seen["q"] == "Athens hotels Plaka" and seen["rating"] == "8" and seen["hotel_class"] == "3,4"
    assert json.loads(text)["hotels"][0]["name"] == "Plaka Hotel"


def test_run_tool_reports_errors(monkeypatch):
    def boom(params):
        raise serpapi.SearchError("Google Hotels hasn't returned any results for this query.")

    monkeypatch.setattr(serpapi, "_get", boom)
    text, is_error = tools.run_tool(
        "search_hotels",
        {"query": "x", "check_in": "2027-03-10", "check_out": "2027-03-11", "adults": 1}, TODAY,
    )
    assert is_error and "hasn't returned any results" in text
    assert tools.run_tool("nope", {}, TODAY) == ("Unknown tool: nope", True)


def test_missing_serpapi_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    with pytest.raises(serpapi.SearchError, match="SERPAPI_API_KEY"):
        serpapi._get({"engine": "google_flights"})


# ------------------------------------------------------------ return flights

OUTBOUND_WITH_TOKENS = {
    "best_flights": [
        {
            "price": 410,
            "departure_token": "TOKEN-A" * 50,
            "flights": [{"airline": "Aegean", "departure_airport": {"id": "TLV"}, "arrival_airport": {"id": "ATH"}}],
        }
    ],
    "other_flights": [],
}

RETURN_PAYLOAD = {
    "search_metadata": {"google_flights_url": "https://www.google.com/travel/flights?ret"},
    "best_flights": [
        {
            "price": 520,
            "total_duration": 180,
            "flights": [
                {
                    "airline": "Aegean",
                    "flight_number": "A3 930",
                    "departure_airport": {"id": "ATH", "time": "2027-03-15 09:30"},
                    "arrival_airport": {"id": "TLV", "time": "2027-03-15 12:30"},
                }
            ],
        }
    ],
    "other_flights": [
        {"price": 480, "flights": [{"airline": "Wizz Air"}, {"airline": "Wizz Air"}], "layovers": [{"id": "LCA", "duration": 90}]},
        {"flights": []},  # unpriced: dropped
    ],
}


def test_outbound_options_get_short_ids_not_tokens():
    out = serpapi.normalize_flights(OUTBOUND_WITH_TOKENS, "USD", remember=lambda tok: "F00000001")
    assert out["flights"][0]["option_id"] == "F00000001"
    assert "TOKEN-A" not in json.dumps(out)  # long tokens never reach the model
    assert "option_id" not in serpapi.normalize_flights(OUTBOUND_WITH_TOKENS, "USD")["flights"][0]


def test_normalize_return_flights():
    out = serpapi.normalize_return_flights(RETURN_PAYLOAD, "USD")
    assert [r["round_trip_price"] for r in out["return_flights"]] == [480, 520]
    assert out["return_flights"][0]["stops"] == 1
    assert out["return_flights"][1]["return_legs"][0]["depart_time"] == "2027-03-15 09:30"


def test_search_then_get_return_flights_end_to_end(monkeypatch):
    calls = []

    def fake_get(params):
        calls.append(dict(params))
        return OUTBOUND_WITH_TOKENS if "departure_token" not in params else RETURN_PAYLOAD

    monkeypatch.setattr(serpapi, "_get", fake_get)
    text, err = tools.run_tool(
        "search_flights",
        {"origin": "TLV", "dest": "ATH", "depart": "2027-03-10", "ret": "2027-03-15", "adults": 2}, TODAY,
    )
    assert not err
    option_id = json.loads(text)["flights"][0]["option_id"]

    text, err = tools.run_tool("get_return_flights", {"option_id": option_id}, TODAY)
    assert not err
    body = json.loads(text)
    assert body["option_id"] == option_id
    # The outbound search was direct-only, so the 1-stop return ($480) is dropped.
    assert [r["round_trip_price"] for r in body["return_flights"]] == [520]
    # The second request repeats the original search and adds the stored token.
    second = calls[1]
    assert second["departure_token"] == "TOKEN-A" * 50
    assert (second["departure_id"], second["arrival_id"], second["adults"], second["return_date"]) == (
        "TLV", "ATH", 2, "2027-03-15")


def test_return_flights_rejects_bad_and_unknown_ids():
    text, err = tools.run_tool("get_return_flights", {"option_id": "nope"}, TODAY)
    assert err and "option_id" in text
    text, err = tools.run_tool("get_return_flights", {"option_id": "Fdeadbeef"}, TODAY)
    assert err and "expired" in text


def test_expired_option_id(monkeypatch):
    oid = serpapi._remember("tok", {"engine": "google_flights"})
    assert serpapi._recall(oid) is not None
    monkeypatch.setattr(serpapi, "_TOKEN_TTL_S", -1)
    assert serpapi._recall(oid) is None


def test_city_codes_expand_to_airports():
    args = tools.validate_flights({"origin": "TLV", "dest": "NYC", "depart": "2027-03-10", "ret": "2027-03-15",
                                   "adults": 2}, TODAY)
    assert args["dest"] == "JFK,EWR,LGA" and args["origin"] == "TLV"
    assert tools.expand_metro("LON,LHR") == "LHR,LGW,STN,LTN,LCY"


# ------------------------------------------------------------ direct-first policy

def _opt(price, minutes, stops, token):
    legs = [{"airline": "X", "departure_airport": {"id": "TLV"}, "arrival_airport": {"id": "PRG"}}] * (stops + 1)
    return {"price": price, "total_duration": minutes, "flights": legs, "departure_token": token}


def _policy_search(monkeypatch, by_stops):
    calls = []

    def fake_get(params):
        calls.append(params["stops"])
        payload = by_stops.get(params["stops"])
        if payload is None:
            raise serpapi.SearchError("Google Flights hasn't returned any results for this query.")
        return {"best_flights": payload}

    monkeypatch.setattr(serpapi, "_get", fake_get)
    text, err = tools.run_tool("search_flights", {"origin": "TLV", "dest": "PRG", "depart": "2026-12-02",
                                                  "ret": "2026-12-06", "adults": 2}, TODAY)
    assert not err
    return json.loads(text), calls


def test_short_direct_flight_means_direct_only(monkeypatch):
    out, calls = _policy_search(monkeypatch, {1: [_opt(300, 240, 0, "d")], 2: [_opt(200, 400, 1, "c")]})
    assert calls == [1] and [f["stops"] for f in out["flights"]] == [0]
    assert out["notes"][0].startswith("Direct flights only")


def test_no_direct_flight_adds_one_stop_options(monkeypatch):
    out, calls = _policy_search(monkeypatch, {2: [_opt(200, 400, 1, "c"), _opt(150, 600, 2, "cc")]})
    assert calls == [1, 2]
    assert [f["stops"] for f in out["flights"]] == [1]  # the 2-stop option is dropped
    assert out["notes"][0].startswith("No direct flight was found")


def test_long_direct_flight_adds_one_stop_options(monkeypatch):
    out, calls = _policy_search(monkeypatch, {1: [_opt(900, 660, 0, "d")],
                                              2: [_opt(900, 660, 0, "d"), _opt(600, 800, 1, "c")]})
    assert calls == [1, 2] and sorted(f["stops"] for f in out["flights"]) == [0, 1]
    assert out["notes"][0].startswith("The direct flights take more than 7 hours")


def test_direct_only_request_never_adds_connections(monkeypatch):
    monkeypatch.setattr(serpapi, "_get", lambda p: (_ for _ in ()).throw(
        serpapi.SearchError("Google Flights hasn't returned any results for this query.")))
    text, err = tools.run_tool("search_flights", {"origin": "TLV", "dest": "PRG", "depart": "2026-12-02",
                                                  "ret": "2026-12-06", "adults": 2, "nonstop_only": True}, TODAY)
    assert not err and json.loads(text)["flights"] == []



# ------------------------------------------------------------ cheapest dates

def _calendar(monkeypatch, by_stops):
    calls = []

    def fake_get(params):
        calls.append(params)
        prices = by_stops.get(params["stops"])
        if prices is None:
            raise serpapi.SearchError("Google Flights hasn't returned any results for this query.")
        return {"prices": prices}

    monkeypatch.setattr(serpapi, "_get", fake_get)
    return calls


def test_cheapest_dates_sorted_direct_first(monkeypatch):
    calls = _calendar(monkeypatch, {1: [
        {"outbound_date": "2026-12-05", "return_date": "2026-12-09", "price": 410},
        {"outbound_date": "2026-12-12", "return_date": "2026-12-16", "price": 290},
        {"outbound_date": "2026-12-19", "return_date": "2026-12-23", "price": 520},
    ]})
    text, err = tools.run_tool("find_cheapest_dates", {"origin": "TLV", "dest": "PRG", "date_from": "2026-12-01",
                                                       "date_to": "2026-12-31", "nights": 4, "adults": 2}, TODAY)
    out = json.loads(text)
    assert not err and [c["depart"] for c in out["cheapest"]] == ["2026-12-12", "2026-12-05", "2026-12-19"]
    assert calls[0]["engine"] == "google_flights_calendar" and calls[0]["round_trip_days"] == 4
    assert calls[0]["stops"] == 1 and len(calls) == 1 and out["notes"][0] == "Direct flights only."


def test_cheapest_dates_fall_back_to_one_stop(monkeypatch):
    calls = _calendar(monkeypatch, {2: [{"outbound_date": "2026-12-05", "return_date": "2026-12-09", "price": 300}]})
    out = json.loads(tools.run_tool("find_cheapest_dates", {"origin": "TLV", "dest": "PRG", "date_from": "2026-12-01",
                                                            "date_to": "2026-12-31", "nights": 4, "adults": 1}, TODAY)[0])
    assert [c["stops"] for c in calls] == [1, 2] and out["cheapest"][0]["price"] == 300


def test_cheapest_dates_input_checks():
    base = {"origin": "TLV", "dest": "PRG", "date_from": "2026-12-01", "date_to": "2026-12-31", "nights": 4, "adults": 1}
    for patch, msg in [({"date_to": "2026-11-30"}, "on or after"), ({"date_to": "2027-03-01"}, "two months"),
                       ({"nights": 0}, "nights")]:
        text, err = tools.run_tool("find_cheapest_dates", {**base, **patch}, TODAY)
        assert err and msg in text


# ------------------------------------------------------------ hotels: hostels and link city

def test_hostels_are_left_out_unless_asked(monkeypatch):
    payload = {"properties": [
        {"name": "Sir Toby's Hostel", "type": "hotel", "rate_per_night": {"extracted_lowest": 20}},
        {"name": "Hotel Paris", "type": "hotel", "rate_per_night": {"extracted_lowest": 150}},
    ]}
    monkeypatch.setattr(serpapi, "_get", lambda p: payload)
    base = {"query": "Prague hotels", "check_in": "2027-03-10", "check_out": "2027-03-12", "adults": 2}
    names = [h["name"] for h in json.loads(tools.run_tool("search_hotels", base, TODAY)[0])["hotels"]]
    assert names == ["Hotel Paris"]
    names = [h["name"] for h in json.loads(tools.run_tool("search_hotels", {**base, "include_hostels": True}, TODAY)[0])["hotels"]]
    assert names == ["Sir Toby's Hostel", "Hotel Paris"]


def test_hebrew_query_words_stay_out_of_links():
    assert serpapi._place("מלונות בפראג Prague") == "Prague"
    assert serpapi._place("מלונות בפראג") == ""


# ------------------------------------------------------------ ask_user buttons

def test_ask_user_sends_buttons_to_the_page():
    events = []
    q = {"questions": [{"question": "כמה נוסעים?", "options": ["1", "2", "2 + ילד", "אחר"]},
                       {"question": "תאריכים?", "options": ["תאריכים מדויקים", "הכי זול בחודש"]}]}
    text, err = tools.run_tool("ask_user", q, TODAY, sink=events.append)
    assert not err and json.loads(text)["shown_to_user"] is True
    assert events == [{"type": "questions", "questions": q["questions"]}]


@pytest.mark.parametrize("bad", [
    {"questions": []},
    {"questions": [{"question": "x", "options": ["only one"]}]},
    {"questions": [{"question": "", "options": ["a", "b"]}]},
    {"questions": [{"question": "x", "options": ["a", "b"]}], "extra": 1},
])
def test_ask_user_rejects_bad_input(bad):
    text, err = tools.run_tool("ask_user", bad, TODAY)
    assert err and text.startswith("Invalid input")
