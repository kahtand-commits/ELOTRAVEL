"""Terminal chat: python -m elotravel.cli"""

from __future__ import annotations

import json
import os
import sys

import anthropic

from .agent import AssistantBusyError, AssistantUnavailableError, MissingCredentialsError
from .providers import make_agent
from .prompt import GREETING


def main() -> int:
    if not os.environ.get("SERPAPI_API_KEY"):
        print("Warning: SERPAPI_API_KEY is not set; searches will fail.\n", file=sys.stderr)
    agent = make_agent()
    print(f"Assistant: {GREETING}\n(Type 'quit' to exit.)\n")

    def on_text(t: str) -> None:
        print(t, end="", flush=True)

    def on_tool(name: str, args: dict) -> None:
        print(f"\n  [searching: {name} {json.dumps(args, ensure_ascii=False)}]", flush=True)

    def on_event(event: dict) -> None:
        for p in event.get("packages", []):
            hotel = p["hotel"]
            airlines = ", ".join(p["flight"]["outbound"].get("airlines") or [])
            print(f"\n  [{p['label']}] {airlines} {p['flight_price']} + {hotel.get('name')} "
                  f"{p['hotel_total']} = {p['total']} {p['currency']}")
            for name, url in (hotel.get("links") or {}).items():
                print(f"      {name}: {url}")

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            return 0
        print("\nAssistant: ", end="", flush=True)
        try:
            agent.send(user, on_text=on_text, on_tool=on_tool, on_event=on_event)
        except (MissingCredentialsError, anthropic.AuthenticationError):
            print("\nError: the API key is missing or invalid (set GEMINI_API_KEY or ANTHROPIC_API_KEY).")
            return 1
        except (AssistantBusyError, anthropic.RateLimitError):
            print("\nError: rate limited or out of free quota. Try again later.")
        except AssistantUnavailableError as e:
            print(f"\nError: {e}")
        except anthropic.APIStatusError as e:
            print(f"\nError: Anthropic API returned {e.status_code}: {e.message}")
        except anthropic.APIConnectionError:
            print("\nError: could not reach the Anthropic API.")
        print()


if __name__ == "__main__":
    sys.exit(main())
