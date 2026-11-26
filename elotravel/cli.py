"""Terminal chat: python -m elotravel.cli"""

from __future__ import annotations

import json
import os
import sys

import anthropic

from .agent import TravelAgent
from .prompt import GREETING


def main() -> int:
    if not os.environ.get("SERPAPI_API_KEY"):
        print("Warning: SERPAPI_API_KEY is not set; searches will fail.\n", file=sys.stderr)
    agent = TravelAgent()
    print(f"Assistant: {GREETING}\n(Type 'quit' to exit.)\n")

    def on_text(t: str) -> None:
        print(t, end="", flush=True)

    def on_tool(name: str, args: dict) -> None:
        print(f"\n  [searching: {name} {json.dumps(args, ensure_ascii=False)}]", flush=True)

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
            agent.send(user, on_text=on_text, on_tool=on_tool)
        except anthropic.AuthenticationError:
            print("\nError: Anthropic credentials are missing or invalid (set ANTHROPIC_API_KEY).")
            return 1
        except anthropic.RateLimitError:
            print("\nError: rate limited by the Anthropic API. Try again in a moment.")
        except anthropic.APIStatusError as e:
            print(f"\nError: Anthropic API returned {e.status_code}: {e.message}")
        except anthropic.APIConnectionError:
            print("\nError: could not reach the Anthropic API.")
        print()


if __name__ == "__main__":
    sys.exit(main())
