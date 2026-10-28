# The Liquidation Run-Rate Problem — prototype

Each morning, for Farah's ~3,000 pairs on Amazon.ae and Noon: what to do on every listing that
needs it (a specific price, ad budget or exit, with an Apply button), and whether the portfolio
clears before the jobber deadline. Every number comes from deterministic code that passes one
rule check; Claude writes the morning brief and never decides.

## Run it

1. `uv sync` — installs Python 3.11 and the dependencies ([install uv](https://docs.astral.sh/uv/getting-started/installation/)).
2. `cp .env.example .env` and put your `ANTHROPIC_API_KEY` in it. Without a key the app still
   runs; the brief is then the engine's own verdict, and the page says so.
3. `uv run streamlit run app.py`, then open http://localhost:8501.

No uv? `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`, then
`streamlit run app.py`. A new day's data goes in through the upload in the sidebar.
`make verify` runs what a reviewer would: lint, 117 tests, edge cases, drills.

## What I cut, and why

- **No marketplace API, backend or database.** Apply records the change in `state.json` and
  tomorrow's plan enforces the rules against what was applied. One brand manager doesn't need
  more, and the decisions are the point.
- **No optimiser.** Ad budget goes AED 10 at a time to the listing where it adds the most
  projected recovery. Farah can follow that in one sentence, and the estimates an optimiser would
  squeeze are soft.
- **No price raises.** A style selling out early (S16 on Noon, 2.3× its needed rate) could carry a
  higher price; the tool cuts its ad spend instead. Every price move spends one of two weekly slots
  and risks parity, and this is a liquidation. It is the next lever I'd add.
- **No size-level pricing.** Sizes drive exits (a style whose core sizes are gone slows down, and
  its odd sizes go to the jobber), not prices. The sales history is by listing, not by size, so a
  per-size price would rest on a guess.
- **No ML forecasting.** Fourteen days of data. Conversion is shrunk toward category peers,
  elasticity is a stated prior, and the tool checks its own forecasts after every change it
  applies: the honest form of learning at this data size.

## More

- `HOW_IT_WORKS.md` — what it does step by step, the file layout, and where the build departs
  from the approach note.
- `DECISIONS.md` — the judgement calls and why. `assumptions.md` — every assumption.
  `VERIFICATION.md` — each rule mapped to the test that proves it, and what passing doesn't prove.
- `approach_note.md` — the Stage 1 note, as sent.
