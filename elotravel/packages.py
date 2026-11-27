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
        "Show the user your 2-3 best flight + hotel packages as visual cards. The app fills in "
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
                        "flight_option_id": {"type": "string", "description": "option_id from search_flights."},
                        "return_id": {"type": "string", "description": "return_id from get_return_flights, if you have one."},
                        "hotel_id": {"type": "string", "description": "hotel_id from search_hotels."},
                    },
                    "required": ["label", "why", "flight_option_id", "hotel_id"],
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
            if val is None and key == "return_id":
                item[key] = None
                continue
            if not isinstance(val, str) or not pattern.match(val):
                raise PackageError(f"Package {i}: {key} must be an id from the search results.")
            item[key] = val
        out.append(item)
    return {"packages": out, "budget": budget}


def _nights(hotel: dict[str, Any]) -> int | None:
    try:
        return (dt.date.fromisoformat(hotel["check_out"]) - dt.date.fromisoformat(hotel["check_in"])).days
    except (KeyError, TypeError, ValueError):
        return None


def resolve(args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (result for the model, card payload for the page)."""
    budget = args.get("budget")
    cards, summary, missing = [], [], []
    for p in args["packages"]:
        outbound = serpapi.lookup(f"{p['flight_option_id']}/ui")
        ret = serpapi.lookup(p["return_id"]) if p["return_id"] else None
        hotel = serpapi.lookup(p["hotel_id"])
        if outbound is None:
            missing.append(p["flight_option_id"])
        if p["return_id"] and ret is None:
            missing.append(p["return_id"])
        if hotel is None:
            missing.append(p["hotel_id"])
        if outbound is None or hotel is None or (p["return_id"] and ret is None):
            continue
        if ret and ret.get("option_id") != p["flight_option_id"]:
            raise PackageError(f"return_id {p['return_id']} belongs to a different outbound flight.")

        # A chosen return gives the final round-trip fare; otherwise the outbound fare.
        flight_price = (ret or outbound).get("price")
        nights = _nights(hotel)
        hotel_total = hotel.get("price_total_stay")
        if hotel_total is None and hotel.get("price_per_night") is not None and nights:
            hotel_total = hotel["price_per_night"] * nights
        total = flight_price + hotel_total if isinstance(flight_price, (int, float)) and isinstance(hotel_total, (int, float)) else None
        currency = outbound.get("currency") or hotel.get("currency")
        mixed = hotel.get("currency") and outbound.get("currency") and hotel["currency"] != outbound["currency"]
        remaining = budget - total if budget is not None and total is not None and not mixed else None

        cards.append({
            "label": p["label"], "title": p["title"], "why": p["why"],
            "currency": currency, "mixed_currency": bool(mixed),
            "flight_price": flight_price, "hotel_total": hotel_total, "total": None if mixed else total,
            "budget": budget, "remaining": remaining, "nights": nights,
            "flight": {"outbound": outbound, "return": ret,
                       "results_link": (ret or outbound).get("results_link"),
                       "return_details_missing": ret is None},
            "hotel": hotel,
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
            "The packages are now on screen as cards with flight times, aircraft, hotel photo, "
            "prices, totals and links. Reply with only a 1-2 sentence recommendation and one next "
            "step. Don't repeat the details, prices or links. If a package is over budget, say so."
        ),
    }
    if missing:
        result["not_shown"] = f"Unknown or expired ids were skipped: {missing}."
    return result, {"type": "packages", "packages": cards}
