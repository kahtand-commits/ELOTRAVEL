# ELOTRAVEL

A chat assistant that finds and compares **flight + hotel packages** within a budget.
It searches Google Flights and Google Hotels (through [SerpApi](https://serpapi.com)), combines
the results, and presents the best 2-3 packages with transparent totals and provider links.
It only searches and compares. It never books, pays, or collects personal or payment data.

Works in any language the user writes in, including Hebrew (the web chat renders RTL automatically).

## How it works

```
user ──► TravelAgent (Claude, adaptive thinking) ──► search_flights ──────┐
                 ▲                                ──► get_return_flights ──┤  SerpApi (Google Flights / Hotels)
                 │                                ──► search_hotels ────────┤
                 └──────────────── normalized JSON results ◄───────────────┘
```

| File | Purpose |
|---|---|
| `elotravel/prompt.py` | The assistant's system prompt (conversation flow, budget logic, rules, style) |
| `elotravel/serpapi.py` | SerpApi calls, plus normalization into compact results with notes on what each price covers |
| `elotravel/tools.py` | Tool schemas for Claude, strict input validation (IATA codes, future dates, ranges), dispatch |
| `elotravel/agent.py` | The Claude conversation loop: streams replies, runs searches in parallel, and keeps history |
| `elotravel/agent_gemini.py` | The same loop on Google Gemini |
| `elotravel/providers.py` | Picks Claude or Gemini from the settings |
| `elotravel/packages.py` | `show_packages`: the model picks flight, return and hotel ids; the server builds the package cards (prices, totals, aircraft, photos, links) from the stored search results |
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

The web server keeps sessions in memory (one process, 6-hour expiry). Use a shared store before running multiple workers. Set `ACCESS_CODE` to require a code (see below).

## Search modes

A switch above the message box picks what to search; the page sends it with each message:

| Mode | What it does |
|---|---|
| Package (default) | Flight + hotel packages within the budget |
| Flight | Flight-only options |
| Hotel | Hotel-only options |
| El Al Matmid | El Al flights only, across up to 7 dates in a range, listing the dates El Al flies so you know which dates to check for award seats; each card links to El Al's award ticket page. The app can't see award seats or points prices: El Al shows them only after a Matmid login. |

## Choosing the AI model

The app runs on either provider, chosen with `LLM_PROVIDER`:

| `LLM_PROVIDER` | Key | Notes |
|---|---|---|
| `gemini` | `GEMINI_API_KEY` | Google Gemini, has a free tier with usage limits. Free-tier terms may let Google use the conversations to improve its products; check them. Models: `GEMINI_MODEL`, one model or a comma-separated list tried in order when a model is overloaded, out of free quota or unavailable (default `gemini-3.5-flash-lite,gemini-3.5-flash,gemini-3.8-flash`). |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude (`claude-opus-5-5` by default, `ELOTRAVEL_MODEL` to change; `claude-haiku-5-5` is the low-cost choice). Paid per use. |

Switching is a settings change only: set `LLM_PROVIDER` and add the matching key. If `LLM_PROVIDER` is unset, the app uses Claude when `ANTHROPIC_API_KEY` is set, otherwise Gemini when `GEMINI_API_KEY` is set.

## Put it online (Render)

`render.yaml` sets up a single web service on [Render](https://render.com). Render's screens change from time to time, so the names below may differ slightly.

1. Sign in to Render with your GitHub account.
2. Choose **New → Blueprint** and pick the `ELOTRAVEL` repository. Render reads `render.yaml` from the repo's default branch.
3. Render asks for the secret values. Enter them there, never in code or chat:
   - `GEMINI_API_KEY`: your Google Gemini key (free tier, from Google AI Studio)
   - `SERPAPI_API_KEY`: your SerpApi key
   - `ACCESS_CODE`: a code you make up. Anyone using the app must type it.
4. Create the service and wait for the build to finish, then open the `…onrender.com` link it shows.

Protection built in:
- With `ACCESS_CODE` set, the chat asks for the code before anything runs, and wrong codes are slowed down. **Always set it on a public server**, or anyone with the link can spend your API credit.
- `DAILY_REQUEST_LIMIT` (default 50) caps chat messages per day across the whole server.
- On Render's free plan the service sleeps when unused, so the first visit after a while can take up to a minute.

## Tests

GitHub Actions runs these on every pull request and on pushes to `main` (Python 3.11 and 3.13).

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use fixtures and a fake Anthropic client. They need no keys and no network.

## Data caveats

- **Flights:** `search_flights` lists outbound options with Google Flights' round-trip fare for each. `get_return_flights` then fetches the return flights for a chosen option and the final round-trip price for that pair (this costs one extra SerpApi search per option). Baggage allowances are not in the data; confirm them on the provider site.
- **Hotels:** results include the lowest nightly rate and the lowest full-stay total. Whether taxes and fees are included varies by property. Each hotel gets links to Google Hotels and Booking.com searches with the trip dates and travelers filled in, plus the hotel's own website when known.
- If children are included in a hotel search, they are searched at an assumed age of 8.
- Every price is indicative. The assistant always tells users to verify on the provider's site.

## Adding sources

To add a provider (Booking, Expedia, Kayak, …), add a function that returns the same normalized shape, register a schema in `TOOLS`, and add a validator/function pair in `tools._REGISTRY`. The system prompt already tells Claude to use only the tools available in the session.
