# Assumptions

Everything below is a call I made because the brief left it open, or because the data
needed interpreting. Each one is written so it can be replaced in a sentence if Opptra
says otherwise. The decision engine reads most of these from one config block, so
changing an assumption is a one-line change, not a rewrite.

---

## 0. Data provenance

The three CSVs from Opptra are used unchanged in `data/`. Every recommendation in the app
is based on them.

---

## 1. What the data says (facts, not assumptions)

### 1.1 Shape
| Item | Value |
|---|---|
| History window | 21 Sep – 4 Oct 2026, 14 days, 40 listings × 14 rows, no gaps, no duplicates, no blanks |
| "Today" | 4 Oct 2026 (last day in the file). Day 1 = 5 Oct, day 60 = 3 Dec 2026 |
| Styles | 20: 8 men's (S01–S08), 7 women's (S09–S15), 5 kids' (S16–S20) |
| Channels | `amazon_ae`, `noon` |
| Sizes | Men EU 39–46 (8), women EU 36–41 (6), kids EU 28–35 (8) |
| Units on hand today | **3,060** (3,643 on 21 Sep; 583 sold in the window) |
| Inventory vs history | Reconcile exactly for every listing (start − on hand = units sold) |
| Unit cost | AED 60–170. Cost value of stock on hand: AED 332,750 |
| Jobber value of everything today | AED 83,188 (25% of cost) |
| Revenue if every pair sold at today's prices | AED 647,640 |
| Holding cost if nothing moved for 60 days | AED 18,360 |

### 1.2 Run-rate
| Item | Value |
|---|---|
| Needed to clear in 60 days | 51 units/day |
| Actual, last 7 days | 40 units/day (41.6 over 14 days) |
| Straight-line projection if nothing changes | ~2,400 sold, ~650 pairs to the jobber on day 60, before fade and stock-outs on the fast listings make it worse |
| Listings at or above required rate | 14 of 40 |
| Listings below half the required rate | 10 of 40 |

### 1.3 Levers as they stand
| Item | Value |
|---|---|
| Ad spend today | Amazon.ae ≈ AED 1,340 of 1,500; Noon ≈ AED 860 of 1,000. So the headroom is ~AED 300/day; the real opportunity is reallocation, not more money |
| Click cost | ≈ AED 1.93 on Amazon.ae, AED 1.56 on Noon (ad spend ÷ ad clicks over the window) |
| Share of sessions that are paid | 42% overall; 68–100% on the heaviest-funded listings |
| Price changes in the window | **One.** S08 on Noon, 349 → 332 on 28 Sep, which is a parity move (−4.9% vs Amazon.ae). There is no price-response signal in the history |
| Parity | All styles inside ±5%. S08 sits at −4.9%, so any Amazon.ae move on S08 forces Noon to move too |
| Prices vs MRP | Most styles 15–21% below MRP. S06 is at 55% off and S19 at 65% off |
| Prices vs floor | S06 on Amazon.ae is **at** its floor (169; the floor is above cost for S06). S19 is 7% above its floor. Everyone else has 12–52% of room |

### 1.4 Things in the data that shape decisions
- **Noon sessions are zero on 29 and 30 Sep for every listing**, while Noon ad clicks, ad spend and units sold continued at normal levels. That is a reporting outage, not a demand collapse. Conversion rates computed naively from those two days are infinite. The tool treats sessions on those rows as missing (sessions < ad clicks is impossible) and says so.
- **S19 Mini Trek (kids)**: 260 pairs, priced at 59 (65% off MRP), absorbing AED ~250/day of ads across both channels, 140 sessions a day on Amazon.ae, and **three pairs sold in 14 days**. At that conversion, price is not the problem; the listing is. Economically it is the single largest leak in the portfolio (~AED 15,000 of ads over 60 days for a few dozen sales).
- **S06 Retro Hi, Amazon.ae**: already at floor, 0.3% conversion, AED 78/day of ads for 0.4 pairs a day. Each ad-driven sale costs more than the pair earns. Noon converts five times better on the same style.
- **S09 Cloud Step, Amazon.ae**: AED 193/day of ads at 0.45% conversion → AED 338 of ads per pair sold against a 249 price.
- **S15 Breeze Runner**: 165 pairs, least discounted style (15% off), 0.2% conversion on both channels, 0.14 pairs a day. Too slow to learn from; has to be decided on economics.
- **S04 Urban Knit**: 60% of remaining stock is in sizes 39, 45 and 46; the middle sizes are nearly gone. Current velocity is being carried by sizes that are about to run out.
- **Starved converters**: S14 on Noon (7% conversion, 14 sessions/day, AED 10/day of ads), S07 on Noon (4%, 11 sessions, AED 8), S03 on Amazon.ae (4.8%, 21 sessions, AED 10). These convert four to seven times better than the portfolio average and get almost no traffic.
- **S01 Amazon.ae** is on track at 1.9× the required rate while spending AED 216/day. It could give budget to the starved listings and stay on track.

---

## 2. Business-rule interpretations

| Rule in the brief | How I read it |
|---|---|
| Daily ad budget AED 1,500 (Amazon.ae) / AED 1,000 (Noon) | A **cap per channel per day**, not a target. Unspent is fine. Budgets do not move between channels. |
| Max 2 price changes per listing in any rolling 7 days | A change on day D is allowed if the listing has had fewer than 2 changes on days D−6…D−1. **A change forced by parity counts.** Direction doesn't matter. The S08 Noon change on 28 Sep drops out of the window on 5 Oct. |
| Noon within ±5% of Amazon.ae | `abs(noon / amazon − 1) ≤ 0.05`, Amazon.ae is the reference. Both channels' prices are decided together per style. |
| Floor | Per style, both channels. No recommendation ever goes below it. MRP is the ceiling; raising prices is out of scope in a liquidation. |
| Holding cost AED 3/unit/month | Accrues daily: AED 0.10 per unit per day on units held. Small, but in every projection. |
| Jobber at 25% of cost, any day, any quantity | A standing option, not a day-60 event. Partial exits (one channel, or the dead sizes of one style) are allowed. Proceeds immediate; no transaction cost. |
| "Today" | Last date in the history file. Recommendations are for the next day. |

Things the brief does not mention that I had to decide:

| Topic | Assumption |
|---|---|
| Marketplace commission / fulfilment fees | Not in the brief, so **default 0** in the economics. Exposed as `marketplace_fee_pct` in config because in reality it moves the jobber comparison; a slider on the page. |
| When a price change takes effect | Next day (marketplaces re-index). Ad changes take effect the same day. |
| Moving stock between channels | Not a lever here. In reality it might be (same warehouse); it is on the "would ask" list. |
| Returns / cancellations | Ignored; `units_sold` is treated as net. |
| Cross-channel cannibalisation | Ignored beyond the parity rule. |
| Size-level pricing | Not done. One price per listing. Sizes only inform exits. |
| Minimum ad spend per listing | None. A listing can be set to 0. |
| Maximum ad spend per listing per day | 2.5× its current spend, so the allocator cannot pile a whole channel's budget into one listing on a thin estimate. |

---

## 3. Modelling assumptions (decision engine)

| Component | Assumption |
|---|---|
| Required run-rate | `units_remaining ÷ days_left` per listing |
| Actual run-rate | Trailing 7-day mean of units sold. Units are trusted on every day (orders are hard data); only sessions are treated as missing on the Noon outage days |
| Status | On track ≥ 100% of required; Behind 50–100%; Critical < 50% |
| Diagnosis | "Traffic problem" when sessions are below the median for that channel × category; "price problem" when conversion is below that median; both → price first (paying for traffic to a listing that doesn't convert is the expensive mistake); conversion near zero at a deep discount → "listing problem", flagged for a human, ads stopped |
| Conversion estimate | Beta-Binomial: category × channel prior worth 150 sessions, updated with the listing's own sessions and units. Category × channel conversion in the data runs 1.3–1.9%, which sets the priors |
| Click cost | Channel average (1.93 / 1.56), updated per listing where it has spent |
| Price response | `units_multiplier = (new_price / old_price)^(−elasticity)`. Prior elasticity 2.0 for every category (SD 0.6); there is no evidence in the data to differentiate categories. Updated from the tool's own price changes. Once 4+ applied actions have 3 days of actuals, each lever's uplift is re-weighted by Σ(actual − baseline) / Σ(predicted − baseline), shrunk toward 1 by n/(n+8), bounded 0.5–1.5; a price+ads action scores the price lever only |
| Ad response | Incremental units = incremental spend ÷ CPC × conversion, with a 0.7 factor on spend beyond 1.5× current (diminishing returns). Budget is allocated AED 10 at a time to the listing whose next AED 10 adds the most projected net recovery; a raise stands only if it is projected to add ≥ AED 150 by the deadline |
| Stopping / trimming ads | The projection is evaluated at every AED 10 from zero to today's spend at the planned price. Zero best → stop (if it saves ≥ AED 150, or the listing is broken, sold out, or past the deadline); a lower spend best → trim to it; else keep. The card says which case applies: ads losing money against the jobber, or a listing that sells out on organic traffic anyway |
| Price-cut sizing | Smallest cut, in AED 5 steps, that lifts projected sell-through to the target, bounded by floor, parity and the change limit. If none can: a critical style takes the legal cut with the best projected recovery (within the 20% step), a behind one a ~10% first move; 3 days before cutting again. The partner channel follows to the shallowest in-band price. A style's cuts must beat holding by AED 300 per listing that moves |
| Exit rule | Exit now the units the plan is *confident* will not sell by day 60 (90th-percentile conversion, elasticity +1 SD) at the price the listing is heading for: today's if nothing says cut it (the odd sizes of an on-track style), the deepest price the rules allow in the time left if it is being cut. A listing flagged as broken is checked by a human first; its exit is proposed 3 days later if it is still dead. Marginal units stay on the marketplace, because the jobber is still there on day 60 and holding cost is only AED 6/unit for the whole period |
| Unit economics test | A marketplace sale is worth making if `price × (1 − fee) − ad_cost_per_unit − holding_to_sale > jobber_value + AED 5` |
| Ranking | Actions ranked by expected change in net recovery (AED) over the remaining horizon versus doing nothing. At most 8 shown as "do today"; the rest are "hold" |
| Verdict | Projected sell-through and net recovery at day 60 under (a) nothing changes, (b) today's actions applied, each with a confidence band from the posterior spread; compared with "jobber everything today" (AED 83,188) |

---

## 4. Product and scope assumptions

- One page. Verdict and LLM brief on top, action table with Apply buttons below, upload at the side.
- State (applied prices, ad budgets, exits, change timestamps, predicted-vs-actual log) lives in `state.json`. No database.
- The LLM makes one call per morning: it receives the computed table and returns (a) a three- to five-sentence brief in plain language and (b) flags on listings where the arithmetic may be misleading. It never returns a number that gets applied. If the key is missing or the call fails, the page renders a template brief and says so.
- Anthropic only, through the official `anthropic` SDK; `ANTHROPIC_API_KEY` in `.env` (the evaluator can use their own). Model `claude-sonnet-5-5` by default, `LLM_MODEL` / `LLM_EFFORT` to change.
- Streamlit, Python 3.10+, no Docker, three commands to run.
- New-day ingestion accepts a file in the `daily_history.csv` layout holding one or more new dates; column names are matched case-insensitively with a small synonym list.

---

## 5. What I would confirm with Opptra on day 1

1. Marketplace fees and fulfilment costs per channel (changes the jobber comparison).
2. Whether a parity-forced Noon change counts against Noon's limit (I assume yes).
3. Whether stock can be moved between channels (S06 sells on Noon, not on Amazon.ae).
4. What happened to Noon sessions on 29–30 Sep, and whether S19's listing is broken (images, size chart, suppressed buy box) rather than overpriced.
5. Whether Amazon.ae and Noon budgets can flex into each other (I assume no).
6. Return rates by category (kids' footwear returns are usually higher).
