"""Google Flights and Google Hotels searches through SerpApi.

Both functions return compact, normalized dicts so the model sees only what it
needs (and nothing it could mistake for data that isn't there). Missing fields
stay None rather than being guessed.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from typing import Any, Callable

SERPAPI_URL = "https://serpapi.com/search.json"
MAX_FLIGHTS = 8
MAX_RETURN_FLIGHTS = 6
MAX_HOTELS = 10

# SerpApi google_hotels `rating` filter codes.
_RATING_CODES = {3.5: "7", 4.0: "8", 4.5: "9"}


class SearchError(Exception):
    """A search failed; the message is safe to show to the model and user."""


# Google's departure tokens are long opaque strings. Keep them (and the search
# they belong to) here and give the model a short option_id instead.
_TOKEN_TTL_S = 30 * 60
_TOKEN_MAX = 500
_tokens: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_tokens_lock = threading.Lock()


def _remember(token: str, search_params: dict[str, Any]) -> str:
    key = json.dumps([token, search_params], sort_keys=True, default=str)
    option_id = "F" + hashlib.sha1(key.encode()).hexdigest()[:8]
    now = time.time()
    with _tokens_lock:
        _tokens[option_id] = (now, {**search_params, "departure_token": token})
        _tokens.move_to_end(option_id)
        while len(_tokens) > _TOKEN_MAX:
            _tokens.popitem(last=False)
    return option_id


def _recall(option_id: str) -> dict[str, Any] | None:
    with _tokens_lock:
        entry = _tokens.get(option_id)
        if entry and time.time() - entry[0] <= _TOKEN_TTL_S:
            return dict(entry[1])
        _tokens.pop(option_id, None)
    return None


def _put(key: str, payload: dict[str, Any]) -> None:
    with _tokens_lock:
        _tokens[key] = (time.time(), payload)
        _tokens.move_to_end(key)
        while len(_tokens) > _TOKEN_MAX:
            _tokens.popitem(last=False)


def _short_id(prefix: str, material: Any) -> str:
    key = json.dumps(material, sort_keys=True, default=str)
    return prefix + hashlib.sha1(key.encode()).hexdigest()[:8]


def lookup(key: str) -> dict[str, Any] | None:
    """Details kept for an id (outbound "F…/ui", return "R…", hotel "H…"), or None if expired."""
    return _recall(key)


def _get(params: dict[str, Any], timeout: float = 25.0) -> dict[str, Any]:
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise SearchError("SERPAPI_API_KEY is not set, so live search is unavailable.")
    query = {k: v for k, v in params.items() if v is not None}
    query["api_key"] = api_key
    url = f"{SERPAPI_URL}?{urllib.parse.urlencode(query)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        try:
            detail = json.load(e).get("error", "")
        except Exception:
            detail = ""
        raise SearchError(f"Search provider returned HTTP {e.code}. {detail}".strip()) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise SearchError(f"Could not reach the search provider: {e}") from e
    if data.get("error"):
        # SerpApi reports "no results" as an error string too.
        raise SearchError(str(data["error"]))
    return data


# ---------------------------------------------------------------- flights


def _leg(seg: dict[str, Any]) -> dict[str, Any]:
    dep = seg.get("departure_airport") or {}
    arr = seg.get("arrival_airport") or {}
    return {
        "airline": seg.get("airline"),
        "flight_number": seg.get("flight_number"),
        "from": dep.get("id"),
        "depart_time": dep.get("time"),
        "to": arr.get("id"),
        "arrive_time": arr.get("time"),
        "duration_min": seg.get("duration"),
        "airplane": seg.get("airplane"),
        "legroom": seg.get("legroom"),
    }


def _leg_ui(seg: dict[str, Any]) -> dict[str, Any]:
    dep = seg.get("departure_airport") or {}
    arr = seg.get("arrival_airport") or {}
    return {
        **_leg(seg),
        "from_name": dep.get("name"),
        "to_name": arr.get("name"),
        "airline_logo": seg.get("airline_logo"),
        "travel_class": seg.get("travel_class"),
        "overnight": bool(seg.get("overnight")),
        "often_delayed": bool(seg.get("often_delayed_by_over_30_min")),
        "extensions": [e for e in (seg.get("extensions") or []) if "carbon" not in e.lower()][:6],
    }


def _option_ui(opt: dict[str, Any]) -> dict[str, Any]:
    segs = opt.get("flights") or []
    return {
        "price": opt.get("price"),
        "airlines": list(dict.fromkeys(s.get("airline") for s in segs if s.get("airline"))),
        "airline_logo": opt.get("airline_logo"),
        "stops": max(len(segs) - 1, 0),
        "total_duration_min": opt.get("total_duration"),
        "legs": [_leg_ui(s) for s in segs],
        "layovers": [
            {"airport": l.get("id"), "name": l.get("name"), "duration_min": l.get("duration")}
            for l in (opt.get("layovers") or [])
        ],
        "extensions": [e for e in (opt.get("extensions") or []) if "carbon" not in e.lower()][:6],
    }


def normalize_flights(
    data: dict[str, Any],
    currency: str,
    remember: Callable[[str], str] | None = None,
    keep: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    options = (data.get("best_flights") or []) + (data.get("other_flights") or [])
    meta = data.get("search_metadata") or {}
    flights = []
    for opt in options:
        if opt.get("price") is None:
            continue  # unpriced options can't be part of a package
        segs = opt.get("flights") or []
        airlines = list(dict.fromkeys(s.get("airline") for s in segs if s.get("airline")))
        token = opt.get("departure_token")
        option_id = remember(token) if token and remember else None
        if option_id and keep:
            keep(option_id, {**_option_ui(opt), "currency": currency, "results_link": meta.get("google_flights_url")})
        flights.append(
            {
                **({"option_id": option_id} if option_id else {}),
                "price": opt.get("price"),
                "airlines": airlines,
                "stops": max(len(segs) - 1, 0),
                "total_duration_min": opt.get("total_duration"),
                "outbound_legs": [_leg(s) for s in segs],
                "layovers": [
                    {"airport": l.get("id"), "duration_min": l.get("duration")}
                    for l in (opt.get("layovers") or [])
                ],
            }
        )
    flights.sort(key=lambda f: f["price"])
    insights = data.get("price_insights") or {}
    return {
        "currency": currency,
        "results_link": meta.get("google_flights_url"),
        "typical_price_range": insights.get("typical_price_range"),
        "price_level": insights.get("price_level"),
        "flights": flights[:MAX_FLIGHTS],
        "notes": [
            "Source: Google Flights via SerpApi. Prices are indicative and change quickly.",
            "price is the round-trip fare Google Flights shows for this outbound flight "
            "with its cheapest return (for all passengers requested). It can change once a "
            "specific return flight is chosen; confirm per-person vs. total on the link.",
            "Only outbound legs are listed here. Call get_return_flights with an option_id "
            "to see return flights and the final round-trip price for that combination.",
            "Baggage allowances are not included in this data.",
        ],
    }


# Direct flights first. Connections only when there is no direct flight, or when
# the shortest direct flight is longer than this; then at most one stop.
DIRECT_MAX_MIN = 7 * 60
_NO_RESULTS = "hasn't returned any results"


def _get_or_empty(params: dict[str, Any]) -> dict[str, Any]:
    try:
        return _get(params)
    except SearchError as e:
        if _NO_RESULTS in str(e):
            return {}
        raise


def search_flights(
    origin: str,
    dest: str,
    depart: str,
    ret: str,
    adults: int = 1,
    children: int = 0,
    nonstop_only: bool = False,
    currency: str = "USD",
    airline: str | None = None,
) -> dict[str, Any]:
    base = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": dest,
        "outbound_date": depart,
        "return_date": ret,
        "type": 1,  # round trip
        "adults": adults,
        "children": children or None,
        "include_airlines": airline,
        "currency": currency,
        "hl": "en",
    }

    def run(stops: int) -> dict[str, Any]:
        params = {**base, "stops": stops}  # SerpApi: 1 = nonstop only, 2 = at most 1 stop
        return normalize_flights(
            _get_or_empty(params), currency,
            remember=lambda tok: _remember(tok, params),
            keep=lambda oid, ui: _put(f"{oid}/ui", ui),
        )

    result = run(1)
    result["flights"] = [f for f in result["flights"] if f["stops"] == 0]
    durations = [f["total_duration_min"] for f in result["flights"] if f["total_duration_min"]]
    if nonstop_only:
        policy = "Direct flights only, as asked."
    elif result["flights"] and (not durations or min(durations) <= DIRECT_MAX_MIN):
        policy = "Direct flights only: a direct flight of 7 hours or less exists, so connections are not shown."
    else:
        reason = ("No direct flight was found" if not result["flights"]
                  else "The direct flights take more than 7 hours")
        result = run(2)
        result["flights"] = [f for f in result["flights"] if f["stops"] <= 1]
        policy = f"{reason}, so flights with at most one stop are included."
    if not result["flights"]:
        result["notes"].insert(0, "No flights found for this search.")
    result["notes"].insert(0, policy)
    result["search"] = {
        "origin": origin, "dest": dest, "depart": depart, "ret": ret,
        "adults": adults, "children": children, "nonstop_only": nonstop_only, "airline": airline,
    }
    return result


def normalize_return_flights(
    data: dict[str, Any],
    currency: str,
    keep: Callable[[dict[str, Any], dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    options = (data.get("best_flights") or []) + (data.get("other_flights") or [])
    meta = data.get("search_metadata") or {}
    returns = []
    for opt in options:
        if opt.get("price") is None:
            continue
        segs = opt.get("flights") or []
        return_id = keep(opt, {**_option_ui(opt), "currency": currency,
                               "results_link": meta.get("google_flights_url")}) if keep else None
        returns.append(
            {
                **({"return_id": return_id} if return_id else {}),
                "round_trip_price": opt.get("price"),
                "airlines": list(dict.fromkeys(s.get("airline") for s in segs if s.get("airline"))),
                "stops": max(len(segs) - 1, 0),
                "total_duration_min": opt.get("total_duration"),
                "return_legs": [_leg(s) for s in segs],
                "layovers": [
                    {"airport": l.get("id"), "duration_min": l.get("duration")}
                    for l in (opt.get("layovers") or [])
                ],
            }
        )
    returns.sort(key=lambda r: r["round_trip_price"])
    return {
        "currency": currency,
        "results_link": meta.get("google_flights_url"),
        "return_flights": returns[:MAX_RETURN_FLIGHTS],
        "notes": [
            "Source: Google Flights via SerpApi. Prices are indicative and change quickly.",
            "round_trip_price is Google Flights' price for the whole trip (the chosen outbound "
            "flight plus this return flight) for all passengers requested. Use it as the final "
            "flight price for a package, and confirm it on the link.",
            "Baggage allowances are not included in this data.",
        ],
    }


def get_return_flights(option_id: str) -> dict[str, Any]:
    params = _recall(option_id)
    if params is None:
        raise SearchError(
            f"Unknown or expired option_id {option_id!r}. Run search_flights again to get fresh ones."
        )
    data = _get(params)

    def keep(opt: dict[str, Any], ui: dict[str, Any]) -> str:
        rid = _short_id("R", [option_id, opt.get("price"), [l.get("flight_number") for l in ui["legs"]]])
        _put(rid, {**ui, "option_id": option_id})
        return rid

    result = normalize_return_flights(data, params.get("currency", "USD"), keep=keep)
    result["option_id"] = option_id
    return result


# ----------------------------------------------------------------- hotels


def _num(rate: dict[str, Any] | None) -> Any:
    return (rate or {}).get("extracted_lowest")


def _place(query: str) -> str:
    """'Paris hotels Le Marais' -> 'Paris Le Marais' (for search links)."""
    words = [w for w in query.split() if w.lower() not in {"hotel", "hotels"}]
    return " ".join(words)


def google_hotel_link(results_link: str | None, name: str, place: str) -> str | None:
    """Google's own results URL for this search (it carries the dates and travelers),
    narrowed to one hotel by replacing the query."""
    if not results_link:
        return None
    parts = urllib.parse.urlsplit(results_link)
    query = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
    query["q"] = [f"{name} {place}".strip()]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query, doseq=True)))


def booking_link(name: str, place: str, check_in: str, check_out: str, adults: int, children: int = 0) -> str:
    params = [
        ("ss", f"{name}, {place}".strip(", ")),
        ("checkin", check_in), ("checkout", check_out),
        ("group_adults", adults), ("group_children", children), ("no_rooms", 1),
    ] + [("age", 8)] * children
    return "https://www.booking.com/searchresults.html?" + urllib.parse.urlencode(params)


def _image(p: dict[str, Any]) -> str | None:
    for img in p.get("images") or []:
        url = img.get("thumbnail") or img.get("original_image")
        if isinstance(url, str) and url.startswith("https://"):
            return url
    return None


def normalize_hotels(
    data: dict[str, Any],
    currency: str,
    search: dict[str, Any] | None = None,
    keep: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    meta = data.get("search_metadata") or {}
    search = search or {}
    place = _place(search.get("query", ""))
    hotels = []
    for p in data.get("properties") or []:
        per_night = _num(p.get("rate_per_night"))
        total = _num(p.get("total_rate"))
        if per_night is None and total is None:
            continue  # no price, can't be part of a package
        name = p.get("name") or ""
        links = {
            "google_hotels": google_hotel_link(meta.get("google_hotels_url"), name, place),
            "booking": booking_link(name, place, search["check_in"], search["check_out"],
                                    search.get("adults", 1), search.get("children", 0))
            if search.get("check_in") and search.get("check_out") else None,
            "website": p.get("link"),
        }
        hotel = {
                "name": p.get("name"),
                "type": p.get("type"),
                "hotel_class": p.get("extracted_hotel_class") or p.get("hotel_class"),
                "rating": p.get("overall_rating"),
                "reviews": p.get("reviews"),
                "price_per_night": per_night,
                "price_total_stay": total,
                "location_rating": p.get("location_rating"),
                "amenities": (p.get("amenities") or [])[:8],
                "nearby": [n.get("name") for n in (p.get("nearby_places") or [])[:3]],
                "links": {k: v for k, v in links.items() if v},
        }
        if keep:
            hotel = {"hotel_id": keep({**hotel, "image": _image(p), "currency": currency, **search}), **hotel}
        hotels.append(hotel)
    return {
        "currency": currency,
        "results_link": meta.get("google_hotels_url"),
        "hotels": hotels[:MAX_HOTELS],
        "notes": [
            "Source: Google Hotels via SerpApi. Prices are indicative and change quickly.",
            "price_total_stay is Google Hotels' lowest total for the whole stay; "
            "price_per_night is the lowest nightly rate. Taxes/fees may or may not be included.",
            "rating is the Google guest rating out of 5; hotel_class is the star level. "
            "None means the data was not provided.",
            "links.google_hotels and links.booking open a search for that hotel with the "
            "trip's dates and travelers already filled in; links.website is the hotel's own site. "
            "If a hotel has no links, use results_link.",
        ],
    }


def search_hotels(
    query: str,
    check_in: str,
    check_out: str,
    adults: int = 1,
    area: str | None = None,
    children: int = 0,
    min_rating: float | None = None,
    hotel_class: list[int] | None = None,
    max_price_per_night: int | None = None,
    sort_by: str | None = None,
    currency: str = "USD",
) -> dict[str, Any]:
    q = f"{query} {area}".strip() if area else query
    rating_code = None
    if min_rating is not None:
        eligible = [v for k, v in sorted(_RATING_CODES.items()) if k <= min_rating]
        rating_code = eligible[-1] if eligible else None
    data = _get(
        {
            "engine": "google_hotels",
            "q": q,
            "check_in_date": check_in,
            "check_out_date": check_out,
            "adults": adults,
            "children": children or None,
            "children_ages": ",".join(["8"] * children) if children else None,
            "rating": rating_code,
            "hotel_class": ",".join(str(c) for c in hotel_class) if hotel_class else None,
            "max_price": max_price_per_night,
            "sort_by": {"lowest_price": 3, "highest_rating": 8, "most_reviewed": 13}.get(sort_by or ""),
            "currency": currency,
            "hl": "en",
        }
    )
    search = {"query": q, "check_in": check_in, "check_out": check_out, "adults": adults, "children": children}

    def keep(h: dict[str, Any]) -> str:
        hid = _short_id("H", [h.get("name"), check_in, check_out, adults, children, currency])
        _put(hid, h)
        return hid

    result = normalize_hotels(data, currency, search=search, keep=keep)
    result["search"] = search
    if children:
        result["notes"].append("Children were searched with an assumed age of 8.")
    return result
