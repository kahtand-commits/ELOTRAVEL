# ELOTRAVEL

A chat assistant that finds and compares **flight + hotel packages** within a budget.
It searches Google Flights and Google Hotels (through [SerpApi](https://serpapi.com)), combines
the results, and presents the best 2-3 packages with transparent totals and provider links.
It only searches and compares. It never books, pays, or collects personal or payment data.

Works in any language the user writes in, including Hebrew (the web chat renders RTL automatically).

## How it works

```
user ──► TravelAgent (Claude, adaptive thinking) ──► search_flights ─┐
                 ▲                                ──► search_hotels ──┤  SerpApi (Google Flights / Hotels)
                 └──────────── normalized JSON results ◄──────────────┘
```

| File | Purpose |
|---|---|
| `elotravel/prompt.py` | The assistant's system prompt (conversation flow, budget logic, rules, style) |
| `elotravel/serpapi.py` | SerpApi calls, plus normalization into compact results with notes on what each price covers |
| `elotravel/tools.py` | Tool schemas for Claude, strict input validation (IATA codes, future dates, ranges), dispatch |
| `elotravel/agent.py` | The conversation loop: streams replies, runs searches in parallel, and keeps history |
| `elotravel/cli.py` | Terminal chat |
| `elotravel/web.py` + `static/index.html` | Minimal web chat (FastAPI) |

Details:
- Model `claude-opus-5-5` with adaptive thinking and effort `medium` (override with `ELOTRAVEL_MODEL` / `ELOTRAVEL_EFFORT`).
- Server-side refusal fallback is on (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`).
  If Claude's safety classifiers decline a request, the API retries it on another model. If every model declines, the turn is dropped and the user is asked to rephrase.
- The system prompt and conversation history are prompt-cached. Today's date is sent outside the cached block.
- If a search fails, the error message goes back to Claude (`is_error`), which explains it and suggests alternatives instead of inventing data.
- If an API call fails, the conversation rolls back to before that message so the user can retry.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # https://console.anthropic.com
export SERPAPI_API_KEY=...     # https://serpapi.com/manage-api-key
```

See `.env.example` for optional settings (`ELOTRAVEL_CURRENCY` sets the default currency, USD).

## Run

Terminal:
```bash
python -m elotravel.cli
```

Web:
```bash
uvicorn elotravel.web:app --port 8000
# open http://localhost:8000
```

The web server keeps sessions in memory (one process, 6-hour expiry). Use a shared store before running multiple workers.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use fixtures and a fake Anthropic client. They need no keys and no network.

## Data caveats

- **Flights:** each price is Google Flights' round-trip fare for the search. Only outbound legs are listed. Return times and baggage are confirmed on the provider site.
- **Hotels:** results include the lowest nightly rate and the lowest full-stay total. Whether taxes and fees are included varies by property.
- If children are included in a hotel search, they are searched at an assumed age of 8.
- Every price is indicative. The assistant always tells users to verify on the provider's site.

## Adding sources

To add a provider (Booking, Expedia, Kayak, …), add a function that returns the same normalized shape, register a schema in `TOOLS`, and add a validator/function pair in `tools._REGISTRY`. The system prompt already tells Claude to use only the tools available in the session.
