"""Pick the model provider.

LLM_PROVIDER=gemini     Google Gemini (has a free tier). Needs GEMINI_API_KEY.
LLM_PROVIDER=anthropic  Claude. Needs ANTHROPIC_API_KEY.

If LLM_PROVIDER is not set: Claude when ANTHROPIC_API_KEY is set, else Gemini when
GEMINI_API_KEY is set, else Claude (which then reports the missing key).
"""

from __future__ import annotations

import os
from typing import Any


def provider_name() -> str:
    chosen = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if chosen in {"gemini", "google"}:
        return "gemini"
    if chosen in {"anthropic", "claude"}:
        return "anthropic"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "anthropic"


def make_agent() -> Any:
    if provider_name() == "gemini":
        from .agent_gemini import GeminiTravelAgent

        return GeminiTravelAgent()
    from .agent import TravelAgent

    return TravelAgent()
