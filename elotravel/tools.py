"""Tool definitions for Claude, input validation, and dispatch."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from typing import Any, Callable

from . import serpapi

_IATA = re.compile(r"^[A-Z]{3}(,[A-Z]{3})*$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")

_DATE = {"type": "string", "description": "Date in YYYY-MM-DD format."}
_CURRENCY_PROP = {
    "type": "string",
    "description": "ISO 4217 currency code for prices, e.g. USD, EUR, ILS. Use the user's budget currency.",
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_flights",
        "description": (
            "Search Google Flights for round-trip flights. Returns up to 8 priced options "
            "sorted by price, with airlines, stops, duration, outbound legs and a results link. "
            "Call several times (in parallel) to compare alternative dates or airports."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Departure IATA airport code (e.g. TLV). Several may be comma-separated (e.g. JFK,EWR).",
                },
                "dest": {
                    "type": "string",
                    "description": "Arrival IATA airport code (e.g. ATH). Several may be comma-separated.",
                },
                "depart": _DATE,
                "ret": {**_DATE, "description": "Return date in YYYY-MM-DD format."},
                "adults": {"type": "integer", "minimum": 1, "maximum": 9},
                "children": {"type": "integer", "minimum": 0, "maximum": 8},
                "nonstop_only": {"type": "boolean", "description": "Only direct flights."},
                "currency": _CURRENCY_PROP,
            },
            "required": ["origin", "dest", "depart", "ret", "adults"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_hotels",
        "description": (
            "Search Google Hotels. Returns up to 10 priced hotels with star class, guest rating, "
            "nightly and full-stay prices, and links. Put the destination city in `query` and a "
            "neighborhood or region in `area`."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Destination, e.g. 'Athens hotels'."},
                "check_in": _DATE,
                "check_out": _DATE,
                "adults": {"type": "integer", "minimum": 1, "maximum": 9},
                "area": {"type": "string", "description": "Neighborhood or region, e.g. 'Plaka'."},
                "children": {"type": "integer", "minimum": 0, "maximum": 8},
                "min_rating": {
                    "type": "number",
                    "enum": [3.5, 4.0, 4.5],
                    "description": "Minimum Google guest rating.",
                },
                "hotel_class": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 2, "maximum": 5},
                    "description": "Allowed star levels, e.g. [4, 5].",
                },
                "max_price_per_night": {"type": "integer", "minimum": 1},
                "sort_by": {"type": "string", "enum": ["lowest_price", "highest_rating", "most_reviewed"]},
                "currency": _CURRENCY_PROP,
            },
            "required": ["query", "check_in", "check_out", "adults"],
            "additionalProperties": False,
        },
    },
]


class ToolInputError(ValueError):
    pass


def _date(value: Any, name: str, today: dt.date) -> dt.date:
    if not isinstance(value, str):
        raise ToolInputError(f"{name} must be a YYYY-MM-DD string.")
    try:
        d = dt.date.fromisoformat(value)
    except ValueError:
        raise ToolInputError(f"{name}={value!r} is not a valid YYYY-MM-DD date.") from None
    if d < today:
        raise ToolInputError(f"{name}={value} is in the past (today is {today}).")
    return d


def _int(inp: dict, key: str, lo: int, hi: int, default: int | None = None) -> int | None:
    v = inp.get(key, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ToolInputError(f"{key} must be an integer between {lo} and {hi}.")
    return v


def _currency(inp: dict) -> str:
    cur = inp.get("currency") or os.environ.get("ELOTRAVEL_CURRENCY", "USD")
    if not isinstance(cur, str) or not _CURRENCY.match(cur.upper()):
        raise ToolInputError("currency must be a 3-letter ISO code such as USD.")
    return cur.upper()


def _check_keys(inp: Any, schema: dict) -> dict:
    # With eager input streaming the API doesn't validate inputs, and a
    # truncated stream can yield a partial object, so check everything here.
    if not isinstance(inp, dict):
        raise ToolInputError("Tool input must be a JSON object.")
    props = schema["properties"]
    unknown = set(inp) - set(props)
    if unknown:
        raise ToolInputError(f"Unknown parameters: {sorted(unknown)}.")
    missing = [k for k in schema["required"] if k not in inp]
    if missing:
        raise ToolInputError(f"Missing required parameters: {missing}.")
    return inp


def validate_flights(inp: Any, today: dt.date) -> dict[str, Any]:
    inp = _check_keys(inp, TOOLS[0]["input_schema"])
    origin, dest = str(inp["origin"]).upper().replace(" ", ""), str(inp["dest"]).upper().replace(" ", "")
    for name, code in (("origin", origin), ("dest", dest)):
        if not _IATA.match(code):
            raise ToolInputError(f"{name}={code!r} must be IATA airport code(s) like TLV or JFK,EWR.")
    depart = _date(inp["depart"], "depart", today)
    ret = _date(inp["ret"], "ret", today)
    if ret <= depart:
        raise ToolInputError("ret must be after depart.")
    nonstop = inp.get("nonstop_only", False)
    if not isinstance(nonstop, bool):
        raise ToolInputError("nonstop_only must be true or false.")
    return {
        "origin": origin, "dest": dest, "depart": depart.isoformat(), "ret": ret.isoformat(),
        "adults": _int(inp, "adults", 1, 9), "children": _int(inp, "children", 0, 8, 0),
        "nonstop_only": nonstop, "currency": _currency(inp),
    }


def validate_hotels(inp: Any, today: dt.date) -> dict[str, Any]:
    schema = TOOLS[1]["input_schema"]
    inp = _check_keys(inp, schema)
    query = inp["query"]
    if not isinstance(query, str) or not query.strip():
        raise ToolInputError("query must be a non-empty string.")
    area = inp.get("area")
    if area is not None and not isinstance(area, str):
        raise ToolInputError("area must be a string.")
    check_in = _date(inp["check_in"], "check_in", today)
    check_out = _date(inp["check_out"], "check_out", today)
    if check_out <= check_in:
        raise ToolInputError("check_out must be after check_in.")
    if (check_out - check_in).days > 30:
        raise ToolInputError("Stays longer than 30 nights are not supported.")
    min_rating = inp.get("min_rating")
    if min_rating is not None and min_rating not in schema["properties"]["min_rating"]["enum"]:
        raise ToolInputError("min_rating must be 3.5, 4.0 or 4.5.")
    classes = inp.get("hotel_class")
    if classes is not None and (
        not isinstance(classes, list)
        or not all(isinstance(c, int) and not isinstance(c, bool) and 2 <= c <= 5 for c in classes)
    ):
        raise ToolInputError("hotel_class must be a list of integers from 2 to 5.")
    sort_by = inp.get("sort_by")
    if sort_by is not None and sort_by not in schema["properties"]["sort_by"]["enum"]:
        raise ToolInputError("sort_by must be lowest_price, highest_rating or most_reviewed.")
    return {
        "query": query.strip(), "check_in": check_in.isoformat(), "check_out": check_out.isoformat(),
        "adults": _int(inp, "adults", 1, 9), "area": area.strip() if area else None,
        "children": _int(inp, "children", 0, 8, 0), "min_rating": min_rating,
        "hotel_class": classes or None,
        "max_price_per_night": _int(inp, "max_price_per_night", 1, 10**7),
        "sort_by": sort_by, "currency": _currency(inp),
    }


_REGISTRY: dict[str, tuple[Callable[[Any, dt.date], dict], Callable[..., dict]]] = {
    "search_flights": (validate_flights, serpapi.search_flights),
    "search_hotels": (validate_hotels, serpapi.search_hotels),
}


def run_tool(name: str, raw_input: Any, today: dt.date | None = None) -> tuple[str, bool]:
    """Run a tool call. Returns (result_text, is_error)."""
    today = today or dt.date.today()
    if name not in _REGISTRY:
        return f"Unknown tool: {name}", True
    validate, fn = _REGISTRY[name]
    try:
        args = validate(raw_input, today)
        result = fn(**args)
    except ToolInputError as e:
        return f"Invalid input: {e}", True
    except serpapi.SearchError as e:
        return f"Search failed: {e}", True
    except Exception as e:  # never crash the conversation on a tool bug
        return f"Search failed unexpectedly ({type(e).__name__}).", True
    if not (result.get("flights") or result.get("hotels")):
        result["notes"].insert(0, "No priced results for this search.")
    return json.dumps(result, ensure_ascii=False), False
