"""show_packages: turn the model's chosen flight + hotel ids into package cards.

The model only picks ids and writes a label and a reason. Every price, time, photo
and link on the cards comes from the stored search results, and the totals are
computed here, so the numbers the user sees are exactly what the searches returned.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from . import serpapi

MAX_PACKAGES = 3
_IDS = {
    "flight_option_id": re.compile(r"^F[0-9a-f]{8}$"),
    "return_id": re.compile(r"^R[0-9a-f]{8}$"),
    "hotel_id": re.compile(r"^H[0-9a-f]{8}$"),
}

SCHEMA: dict[str, Any] = {
    "name": "show_packages",
    "description": (
        "Show the user your 3 best options as visual cards: flight + hotel packages, or flight-only / "
        "hotel-only options when the user's mode asks for that (fewer than 3 only if not enough fit). The app fills in "
        "every price, time, aircraft, photo and link from the search results and adds up the "
        "totals, so pass only ids, a short label and one sentence on why it fits. Call it once "
        "you have searched flights, return flights and hotels. After it returns, reply with only "
        "a 1-2 sentence recommendation and one next step; don't repeat the details or links."
    ),
    "eager_input_streaming": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "packages": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_PACKAGES,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "e.g. 'Best value' / 'הכי משתלם', in the user's language."},
                        "title": {"type": "string", "description": "Short headline, e.g. 'Direct flight + hotel in Plaka'."},
                        "why": {"type": "string", "description": "One sentence on why it fits this user, in their language."},
                        "flight_option_id": {"type": "string", "description": "option_id from search_flights (omit for hotel-only)."},
                        "return_id": {"type": "string", "description": "return_id from get_return_flights, if you have one."},
                        "hotel_id": {"type": "string", "description": "hotel_id from search_hotels (omit for flight-only)."},
                    },
                    "required": ["label", "why"],
                    "additionalProperties": False,
                },
            },
            "budget": {"type": "number", "description": "The user's total budget for flight + hotel, if they gave one."},
        },
        "required": ["packages"],
        "additionalProperties": False,
    },
}


class PackageError(ValueError):
    pass


def _text(value: Any, name: str, limit: int, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PackageError(f"{name} must be a non-empty string.")
    return value.strip()[:limit]


def validate(inp: Any, today: dt.date | None = None) -> dict[str, Any]:
    if not isinstance(inp, dict) or not isinstance(inp.get("packages"), list) or not inp["packages"]:
        raise PackageError("packages must be a non-empty list.")
    unknown = set(inp) - {"packages", "budget"}
    if unknown:
        raise PackageError(f"Unknown parameters: {sorted(unknown)}.")
    if len(inp["packages"]) > MAX_PACKAGES:
        raise PackageError(f"Show at most {MAX_PACKAGES} packages.")
    budget = inp.get("budget")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0):
        raise PackageError("budget must be a positive number.")
    out = []
    for i, p in enumerate(inp["packages"], 1):
        if not isinstance(p, dict):
            raise PackageError(f"Package {i} must be an object.")
        extra = set(p) - set(SCHEMA["input_schema"]["properties"]["packages"]["items"]["properties"])
        if extra:
            raise PackageError(f"Package {i}: unknown fields {sorted(extra)}.")
        item = {
            "label": _text(p.get("label"), "label", 40),
            "title": _text(p.get("title"), "title", 90, required=False),
            "why": _text(p.get("why"), "why", 300),
        }
        for key, pattern in _IDS.items():
            val = p.get(key)
            if val is None:
                item[key] = None
                continue
            if not isinstance(val, str) or not pattern.match(val):
                raise PackageError(f"Package {i}: {key} must be an id from the search results.")
            item[key] = val
        if not item["flight_option_id"] and not item["hotel_id"]:
            raise PackageError(f"Package {i}: give a flight_option_id, a hotel_id, or both.")
        if item["return_id"] and not item["flight_option_id"]:
            raise PackageError(f"Package {i}: return_id needs its flight_option_id.")
        out.append(item)
    return {"packages": out, "budget": budget}


def _nights(hotel: dict[str, Any]) -> int | None:
    try:
        return (dt.date.fromisoformat(hotel["check_out"]) - dt.date.fromisoformat(hotel["check_in"])).days
    except (KeyError, TypeError, ValueError):
        return None


MAX_ALT_FLIGHTS = 6
MAX_ALT_HOTELS = 8


def _delta(a: Any, b: Any) -> Any:
    return a - b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None


def flight_alternatives(outbound: dict[str, Any], chosen_id: str) -> list[dict[str, Any]]:
    """Other options from the same flight search, with the fare difference.

    Compared on the same basis: each option's round-trip fare with its cheapest return."""
    group = serpapi.lookup(outbound.get("group") or "") or {}
    alts = []
    for oid in group.get("ids", []):
        if oid == chosen_id:
            continue
        ui = serpapi.lookup(f"{oid}/ui")
        if not ui:
            continue
        legs = ui.get("legs") or []
        alts.append({
            "airlines": ui.get("airlines"), "airline_logo": ui.get("airline_logo"),
            "stops": ui.get("stops"), "total_duration_min": ui.get("total_duration_min"),
            "depart_time": legs[0].get("depart_time") if legs else None,
            "arrive_time": legs[-1].get("arrive_time") if legs else None,
            "from": legs[0].get("from") if legs else None, "to": legs[-1].get("to") if legs else None,
            "airplane": legs[0].get("airplane") if legs else None,
            "price": ui.get("price"), "delta": _delta(ui.get("price"), outbound.get("price")),
            "results_link": ui.get("results_link"),
        })
    alts.sort(key=lambda a: (a["price"] is None, a["price"] or 0))
    return alts[:MAX_ALT_FLIGHTS]


def hotel_alternatives(hotel: dict[str, Any], chosen_id: str, chosen_total: Any) -> list[dict[str, Any]]:
    """Other hotels from the same hotel search, with the stay-price difference."""
    group = serpapi.lookup(hotel.get("group") or "") or {}
    alts = []
    for hid in group.get("ids", []):
        if hid == chosen_id:
            continue
        h = serpapi.lookup(hid)
        if not h:
            continue
        alts.append({
            "name": h.get("name"), "image": h.get("image"), "rating": h.get("rating"),
            "reviews": h.get("reviews"), "hotel_class": h.get("hotel_class"),
            "price_per_night": h.get("price_per_night"), "price_total_stay": h.get("price_total_stay"),
            "delta": _delta(h.get("price_total_stay"), chosen_total),
            "lat": h.get("lat"), "lng": h.get("lng"), "city": h.get("city"), "links": h.get("links") or {},
        })
    alts.sort(key=lambda a: (a["price_total_stay"] is None, a["price_total_stay"] or 0))
    return alts[:MAX_ALT_HOTELS]


def resolve(args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (result for the model, card payload for the page)."""
    budget = args.get("budget")
    cards, summary, missing = [], [], []
    for p in args["packages"]:
        outbound = serpapi.lookup(f"{p['flight_option_id']}/ui") if p["flight_option_id"] else None
        ret = serpapi.lookup(p["return_id"]) if p["return_id"] else None
        hotel = serpapi.lookup(p["hotel_id"]) if p["hotel_id"] else None
        gone = [i for i, v in ((p["flight_option_id"], outbound), (p["return_id"], ret), (p["hotel_id"], hotel)) if i and v is None]
        if gone:
            missing.extend(gone)
            continue
        if ret and ret.get("option_id") != p["flight_option_id"]:
            raise PackageError(f"return_id {p['return_id']} belongs to a different outbound flight.")

        # A chosen return gives the final round-trip fare; otherwise the outbound fare.
        flight_price = (ret or outbound).get("price") if outbound else None
        nights = _nights(hotel) if hotel else None
        hotel_total = hotel.get("price_total_stay") if hotel else None
        if hotel and hotel_total is None and hotel.get("price_per_night") is not None and nights:
            hotel_total = hotel["price_per_night"] * nights
        parts = [x for x, present in ((flight_price, outbound), (hotel_total, hotel)) if present]
        total = sum(parts) if parts and all(isinstance(x, (int, float)) for x in parts) else None
        currencies = {c for c in ((outbound or {}).get("currency"), (hotel or {}).get("currency")) if c}
        mixed = len(currencies) > 1
        currency = next(iter(currencies), None)
        remaining = budget - total if budget is not None and total is not None and not mixed else None

        cards.append({
            "label": p["label"], "title": p["title"], "why": p["why"],
            "currency": currency, "mixed_currency": mixed,
            "flight_price": flight_price, "hotel_total": hotel_total, "total": None if mixed else total,
            "budget": budget, "remaining": remaining, "nights": nights,
            "flight": {"outbound": outbound, "return": ret,
                       "results_link": (ret or outbound).get("results_link"),
                       "return_details_missing": ret is None} if outbound else None,
            "hotel": hotel,
            "alternatives": {
                "flights": flight_alternatives(outbound, p["flight_option_id"]) if outbound else [],
                "hotels": hotel_alternatives(hotel, p["hotel_id"], hotel_total) if hotel else [],
            },
        })
        summary.append({
            "label": p["label"], "flight_price": flight_price, "hotel_total": hotel_total,
            "total": None if mixed else total, "currency": currency,
            "under_budget_by": remaining if remaining is not None and remaining >= 0 else None,
            "over_budget_by": -remaining if remaining is not None and remaining < 0 else None,
        })

    if missing and not cards:
        raise PackageError(
            f"These ids are unknown or expired: {missing}. Run the searches again and use the new ids."
        )
    result = {
        "shown_to_user": summary,
        "note": (
            "The options are now on screen as cards with the flight and/or hotel details, "
            "prices, totals and links. Reply with only a 1-2 sentence recommendation and one next "
            "step. Don't repeat the details, prices or links. If a package is over budget, say so."
        ),
    }
    if missing:
        result["not_shown"] = f"Unknown or expired ids were skipped: {missing}."
    return result, {"type": "packages", "packages": cards}
