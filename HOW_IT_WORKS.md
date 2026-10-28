# How it works

The detail behind the README: what the tool does step by step, where each part lives, and where
the build departs from the approach note. The judgement calls are in `DECISIONS.md`, every
assumption in `assumptions.md`, and each rule mapped to the test that proves it in `VERIFICATION.md`.

## What it does

1. **Reads the three CSVs** (`data/`), loosely: column names are matched by synonym, blanks
   coerced, duplicates dropped, and impossible rows (sessions below ad clicks — Noon reports
   zero sessions on 29–30 Sep while orders continued) are kept for their orders and distrusted
   for their sessions.
2. **Estimates per listing**: run-rate needed vs actual, conversion (Beta-Binomial with a
   category × channel prior worth 150 sessions, last 7 days weighted double), click cost,
   price elasticity (prior 2.0, updated only from observed price changes — the history has one).
3. **Decides, in this order**: diagnose each listing (visibility, price, both, broken, new,
   on track) → switch off ads that lose money against the jobber at today's price → fix prices
   that already break a rule → choose prices per style, both channels together because parity
   ties them, only where the diagnosis says price is the lever (a channel with no reason of its
   own follows to the shallowest legal price) → set ads per channel: each listing at the spend the
   projection values highest, then the freed budget AED 10 at a time to the listing where it adds
   the most, under the cap → exit the pairs that won't sell even on an optimistic read at the
   price the listing is heading for; a listing that looks broken gets a human check first.
4. **Guardrail**: one function checks the whole plan against the five rules (floor, MRP, ±5%
   parity after both moves, per-channel budget, 2 changes per rolling 7 days incl. parity-forced
   ones). A plan with a violation is not shown. A price that already breaks a rule in the data
   (a floor raised overnight, parity drifted) is fixed first and ranked first; if it can't be
   fixed without breaking another rule, it is shown to Farah as a conflict to decide.
5. **Verdict**: projected sell-through and net recovery at the deadline under "nothing changes"
   vs "today's actions", with a range from the conversion posterior, against the baseline of
   jobbering everything today.
6. **LLM (one call)**: reads the computed plan and writes Farah's brief and flags for a human.
   Every number in the brief is checked against the numbers the model was given; anything it
   invented is called out under the brief. It never produces a number that gets applied. Claude
   Sonnet 5.5 through the official `anthropic` SDK, with the reply's shape enforced by structured
   outputs and effort set to medium (it explains a computed plan; it doesn't need to reason long).
   Without a key the engine's own verdict is shown and the page says so.
7. **Apply** writes to `state.json`; tomorrow's rules are enforced against what was actually
   done. **New day's data** is a drop-in upload: stock rolls forward, the uploaded days are kept
   under `.session/` so a refresh or restart comes back to the same place, and predicted vs
   actual for every applied action lands in the learning log, which re-weights a lever that
   keeps over- or under-delivering. The page, `make plan` and the drills all load the session
   the same way (`engine/session.py`), so they always agree on what the data is.

## Layout

```
app.py                 the page              engine/load.py       CSVs in, issues out
run_plan.py            same plan, terminal   engine/estimate.py   rates, conversion, elasticity, slots
engine/decide.py       the plan, in steps    engine/economics.py  the one projection function
engine/allocate.py     ad budget per channel engine/guardrail.py  the five rules
engine/apply.py        state changes         engine/llm.py        the one LLM call
engine/state.py        state.json            engine/settings.py   Anthropic key, model, effort, timeout from env / .env (key as SecretStr)
engine/session.py      one way to load data + state for the page, the terminal and the drills; keeps uploads under .session/
engine/config.py       every business assumption as a number, in one place
engine/schema.py       the input contract (columns each file must have) and the channel names
engine/actions.py      Action: one recommendation, shared by decide, guardrail, apply, llm and the page
data/                  Opptra's CSVs         tools/               next-day simulator, for rehearsing the daily loop
scenarios/             condition-change drills and edge cases, with results
tests/                 117 tests: the brief as an independent checker (test_invariants), each rule at its boundary (test_rules), the engine on the data (test_engine)
VERIFICATION.md        requirement → code → test, and what passing does not prove
pyproject.toml         deps, ruff, pytest; Makefile; .github/workflows/ci.yml
```

## Other entry points

`make` lists them: `make verify` (lint, format check, 117 tests, edge cases, drills: everything a
reviewer would run), `make test`, `make lint`, `make plan` (the same plan in a terminal),
`make drills` (re-runs the condition changes in `scenarios/drill_results.md`), `make edge`
(awkward inputs: new listing, floor raised overnight, parity already broken, zero budget, messy
multi-day upload), `make next-day` (simulates tomorrow's data to rehearse the daily loop). CI runs
lint, the format check, the tests and the edge cases on every push.

## Where the build departs from the note

- The note says cuts are "the smallest step that gets the listing on track". That holds when a
  legal cut can do it. When none can, a critical style takes the legal cut with the best
  projected recovery (up to the 20% step allowed per change) and a merely-behind listing takes a
  measured ~10% first move; either way the listing is not cut again for 3 days. My first version
  maximised projected recovery for every listing, and with elasticity above 1 that drove cuts to
  the 20% bound even for listings only slightly behind. The bound was deciding, not the
  economics, so the maximising rule is now reserved for styles that are critical.
- The partner channel of a cut follows to the shallowest in-band price, not the same depth. An
  earlier version cut the follower as deep as the leader, giving away margin on a channel that
  had no reason of its own to move; the follower is labelled "parity" and the style-level number
  is shown next to both.
- Every change has to beat holding by a margin, and that now applies to ad raises too (AED 150
  of projected gain). Allocation values each AED 10 of budget with the same projection the plan
  is judged by, so a raise can't look good in one place and bad in another. Day 1 ad spend drops
  from AED 1,342 to ~465 on Amazon.ae and from 860 to ~845 on Noon: most Amazon ads were buying
  sales that paid less than the jobber, or sales that would have happened anyway.
- Switching a listing's ads off is one rule, judged by the projection: the listing recovers more
  over the days left with no ads than with today's spend at the planned price. The per-sale
  cost shown next to it is the explanation, not the test.
- Ad budget changes also get a 3-day cooldown; one day of data after a raise is noise.
- Learning is measured on the *uplift* a lever was credited with (actual − baseline over
  predicted − baseline), not on raw rates, which a good listing would pass whatever the lever
  did. The ratio is shrunk toward 1 until there are enough actions, bounded 0.5–1.5, and scales
  uplift only, never the baseline; a combined price+ads action scores the price lever only. With a
  handful of scored actions and one sale a day it is noisy by nature, and the tool says so.
- **Exits wait for the human.** The note said "exit what won't sell even with the best legal
  actions". The first build read that at today's price and exited 422 pairs on day 1, 234 of them
  from a listing the tool itself flagged as probably broken (S19: 3 sales in 14 days at 65% off).
  Now a broken listing gets its ads stopped and a *check* action (images, size chart, buy box);
  its exit is proposed only once the check is 3 days old and the listing is still dead. Fixing a
  listing recovers far more than the jobber, and 3 days of holding cost AED 70. Other exits are
  read at the price the listing is heading for: today's if nothing says cut it (the odd sizes of
  an on-track style, S04), the deepest price the rules allow if it is being cut. Day 1 exits 79
  pairs, not 422; exits are irreversible and cheap to delay.
- **Stopping ads is no longer all-or-nothing.** The projection is evaluated at every AED 10 from
  zero to today's spend; a listing is stopped only if zero is best, trimmed if a lower spend is
  (S01 Amazon.ae goes 219 → 70/day, enough to sell out by the deadline, instead of to zero), and
  the reason on the card says which case it is: ads that lose money against the jobber, or a
  listing that sells out on organic traffic anyway.
- **What is shown is what applies.** Impact and verdict are computed after the guardrail, so a
  withdrawn move can't leave its number on the page; applying one channel of a parity pair
  applies the partner's price only, not its ad change or exit.
- Not in the note at all: fixing rules that are already broken in the incoming data, and
  admitting when two rules conflict. Stress-testing showed that a raised floor or a tightened
  parity band would otherwise leave the tool unable to produce a legal plan.
- The note's LLM flags turned out to need a template fallback and a check on every number in
  the brief; both are in.
