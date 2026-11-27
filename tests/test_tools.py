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
    assert body["option_id"] == option_id and body["return_flights"][0]["round_trip_price"] == 480
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
