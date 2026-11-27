"""show_packages and dated hotel links, end to end with stubbed SerpApi responses."""

import datetime as dt
import json
import urllib.parse

import pytest

from elotravel import serpapi, tools

TODAY = dt.date(2026, 11, 27)

OUTBOUND = {
    "search_metadata": {"google_flights_url": "https://www.google.com/travel/flights?out"},
    "best_flights": [{
        "price": 420, "total_duration": 290, "departure_token": "TOK-1",
        "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/AF.png",
        "flights": [{
            "airline": "Air France", "flight_number": "AF 1621", "airplane": "Airbus A320neo",
            "legroom": "29 in", "travel_class": "Economy", "duration": 290,
            "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/AF.png",
            "extensions": ["Below average legroom (29 in)", "Carbon emissions estimate: 150 kg", "Wi-Fi for a fee"],
            "departure_airport": {"id": "TLV", "name": "Ben Gurion Airport", "time": "2026-12-11 06:00"},
            "arrival_airport": {"id": "CDG", "name": "Paris Charles de Gaulle", "time": "2026-12-11 09:50"},
        }],
    }],
}
RETURNS = {
    "search_metadata": {"google_flights_url": "https://www.google.com/travel/flights?ret"},
    "best_flights": [{
        "price": 455, "total_duration": 260,
        "flights": [{
            "airline": "Air France", "flight_number": "AF 1620", "airplane": "Airbus A321",
            "departure_airport": {"id": "CDG", "time": "2026-12-13 17:30"},
            "arrival_airport": {"id": "TLV", "time": "2026-12-13 22:50"},
        }],
    }],
}
HOTELS = {
    "search_metadata": {
        "google_hotels_url": "https://www.google.com/travel/search?q=Paris+hotels+Le+Marais"
                             "&check_in_date=2026-12-11&check_out_date=2026-12-13&adults=2&hl=en"
    },
    "properties": [{
        "name": "Hôtel du Marais", "type": "hotel", "extracted_hotel_class": 4,
        "overall_rating": 4.6, "reviews": 1800,
        "rate_per_night": {"extracted_lowest": 310}, "total_rate": {"extracted_lowest": 620},
        "images": [{"thumbnail": "https://lh5.googleusercontent.com/p/hotel.jpg"}],
        "link": "https://hoteldumarais.example",
        "amenities": ["Free Wi-Fi", "Breakfast"],
    }],
}


@pytest.fixture
def searched(monkeypatch):
    """Run the three searches; return (option_id, return_id, hotel_id)."""
    def fake_get(params):
        if params["engine"] == "google_hotels":
            return HOTELS
        return RETURNS if "departure_token" in params else OUTBOUND

    monkeypatch.setattr(serpapi, "_get", fake_get)
    out, err = tools.run_tool("search_flights", {"origin": "TLV", "dest": "CDG", "depart": "2026-12-11",
                                                 "ret": "2026-12-13", "adults": 2}, TODAY)
    assert not err
    flight = json.loads(out)["flights"][0]
    assert flight["outbound_legs"][0]["airplane"] == "Airbus A320neo"
    ret, err = tools.run_tool("get_return_flights", {"option_id": flight["option_id"]}, TODAY)
    assert not err
    hotels, err = tools.run_tool("search_hotels", {"query": "Paris hotels", "area": "Le Marais",
                                                   "check_in": "2026-12-11", "check_out": "2026-12-13",
                                                   "adults": 2}, TODAY)
    assert not err
    return flight["option_id"], json.loads(ret)["return_flights"][0]["return_id"], json.loads(hotels)["hotels"][0]


def test_hotel_links_carry_dates_and_travelers(searched):
    _, _, hotel = searched
    links = hotel["links"]
    booking = urllib.parse.parse_qs(urllib.parse.urlsplit(links["booking"]).query)
    assert booking["checkin"] == ["2026-12-11"] and booking["checkout"] == ["2026-12-13"]
    assert booking["group_adults"] == ["2"] and booking["ss"] == ["Hôtel du Marais, Paris Le Marais"]
    google = urllib.parse.parse_qs(urllib.parse.urlsplit(links["google_hotels"]).query)
    assert google["q"] == ["Hôtel du Marais Paris Le Marais"]
    assert google["check_in_date"] == ["2026-12-11"] and google["adults"] == ["2"]
    assert links["website"] == "https://hoteldumarais.example"


def test_show_packages_builds_cards_from_search_data(searched):
    option_id, return_id, hotel = searched
    events = []
    text, err = tools.run_tool("show_packages", {
        "packages": [{"label": "הכי משתלם", "why": "טיסה ישירה ומלון במארה.",
                      "flight_option_id": option_id, "return_id": return_id, "hotel_id": hotel["hotel_id"]}],
        "budget": 1600,
    }, TODAY, sink=events.append)
    assert not err
    card = events[0]["packages"][0]
    # Final round-trip fare (from the chosen return) + hotel total, computed by the server.
    assert (card["flight_price"], card["hotel_total"], card["total"], card["remaining"]) == (455, 620, 1075, 525)
    assert card["nights"] == 2
    leg = card["flight"]["outbound"]["legs"][0]
    assert leg["airplane"] == "Airbus A320neo" and leg["legroom"] == "29 in"
    assert "Carbon emissions estimate: 150 kg" not in leg["extensions"]
    assert card["flight"]["return"]["legs"][0]["flight_number"] == "AF 1620"
    assert card["hotel"]["image"] == "https://lh5.googleusercontent.com/p/hotel.jpg"
    result = json.loads(text)
    assert result["shown_to_user"][0]["total"] == 1075 and result["shown_to_user"][0]["under_budget_by"] == 525


def test_without_return_uses_outbound_fare(searched):
    option_id, _, hotel = searched
    events = []
    tools.run_tool("show_packages", {"packages": [{"label": "x", "why": "y", "flight_option_id": option_id,
                                                   "hotel_id": hotel["hotel_id"]}]}, TODAY, sink=events.append)
    card = events[0]["packages"][0]
    assert card["flight_price"] == 420 and card["total"] == 1040 and card["flight"]["return_details_missing"]


def test_over_budget_is_reported(searched):
    option_id, return_id, hotel = searched
    text, _ = tools.run_tool("show_packages", {"packages": [{"label": "x", "why": "y", "flight_option_id": option_id,
                                                             "return_id": return_id, "hotel_id": hotel["hotel_id"]}],
                                               "budget": 1000}, TODAY)
    assert json.loads(text)["shown_to_user"][0]["over_budget_by"] == 75


@pytest.mark.parametrize("bad, msg", [
    ({"packages": []}, "non-empty list"),
    ({"packages": [{"label": "x", "why": "y", "flight_option_id": "nope", "hotel_id": "Hdeadbeef"}]}, "flight_option_id"),
    ({"packages": [{"label": "", "why": "y", "flight_option_id": "Fdeadbeef", "hotel_id": "Hdeadbeef"}]}, "label"),
    ({"packages": [{"label": "x", "why": "y", "flight_option_id": "Fdeadbeef", "hotel_id": "Hdeadbeef"}],
      "budget": -5}, "budget"),
])
def test_bad_input(bad, msg):
    text, err = tools.run_tool("show_packages", bad, TODAY)
    assert err and msg in text


def test_expired_ids_are_reported():
    text, err = tools.run_tool("show_packages", {"packages": [{"label": "x", "why": "y",
                                                               "flight_option_id": "Fdeadbeef", "hotel_id": "Hdeadbeef"}]}, TODAY)
    assert err and "expired" in text


def test_flight_only_and_hotel_only_options(searched):
    option_id, return_id, hotel = searched
    events = []
    text, err = tools.run_tool("show_packages", {"packages": [
        {"label": "Flight", "why": "y", "flight_option_id": option_id, "return_id": return_id},
        {"label": "Hotel", "why": "y", "hotel_id": hotel["hotel_id"]},
    ], "budget": 1000}, TODAY, sink=events.append)
    assert not err
    flight_card, hotel_card = events[0]["packages"]
    assert flight_card["hotel"] is None and flight_card["total"] == 455 and flight_card["remaining"] == 545
    assert hotel_card["flight"] is None and hotel_card["total"] == 620 and hotel_card["nights"] == 2


def test_option_needs_a_flight_or_a_hotel():
    text, err = tools.run_tool("show_packages", {"packages": [{"label": "x", "why": "y"}]}, TODAY)
    assert err and "flight_option_id, a hotel_id, or both" in text
    text, err = tools.run_tool("show_packages", {"packages": [{"label": "x", "why": "y", "return_id": "Rdeadbeef",
                                                               "hotel_id": "Hdeadbeef"}]}, TODAY)
    assert err and "return_id needs" in text


def test_airline_filter_is_sent(monkeypatch):
    seen = {}
    monkeypatch.setattr(serpapi, "_get", lambda params: seen.update(params) or OUTBOUND)
    _, err = tools.run_tool("search_flights", {"origin": "TLV", "dest": "LHR", "depart": "2026-12-11",
                                               "ret": "2026-12-13", "adults": 1, "airline": "ly"}, TODAY)
    assert not err and seen["include_airlines"] == "LY"
    text, err = tools.run_tool("search_flights", {"origin": "TLV", "dest": "LHR", "depart": "2026-12-11",
                                                  "ret": "2026-12-13", "adults": 1, "airline": "El Al"}, TODAY)
    assert err and "airline" in text
