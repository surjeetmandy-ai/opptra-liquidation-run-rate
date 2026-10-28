"""Roll the real data forward by one day the tool has never seen.

Fits a simple per-listing model to data/*.csv (organic sessions, click cost, conversion at
the current price) and simulates the next date under whatever prices and ad budgets are in
state.json (what was applied) or, with --as-is, under today's settings. Hidden truths that
the tool must learn: elasticity (drawn once per style, seed-fixed) and a mild demand fade.

Writes tools/next_day.csv in the same layout as daily_history.csv, ready to upload. Used to
rehearse the daily loop (apply, new day, re-plan) before real next-day data exists.

    python tools/simulate_next_day.py                     # after applying actions in the app
    python tools/simulate_next_day.py --as-is             # nothing applied
    python tools/simulate_next_day.py --shock noon_outage # or: spike, competitor_cut
"""

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.load import load_from_paths  # noqa: E402

SEED = 11


def main() -> None:
    """Simulate the next day(s) from the real data and what was applied, and write tools/next_day.csv."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--state", default=str(ROOT / "state.json"))
    ap.add_argument("--out", default=str(ROOT / "tools"))
    ap.add_argument("--as-is", action="store_true", help="ignore state.json; simulate under today's settings")
    ap.add_argument("--shock", default="", choices=["", "noon_outage", "spike", "competitor_cut"])
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument(
        "--history", default="", help="use this history file instead of data/daily_history.csv (e.g. one already rolled forward)"
    )
    args = ap.parse_args()

    d = Path(args.data)
    ds = load_from_paths(d / "inventory.csv", args.history or d / "daily_history.csv", d / "constraints.csv")
    hist = ds.history
    state = {}
    if not args.as_is and Path(args.state).exists():
        state = json.loads(Path(args.state).read_text())
    rng = np.random.default_rng(SEED)
    style_elasticity = {s: float(np.clip(rng.normal(2.0, 0.6), 0.8, 3.5)) for s in ds.inventory["style_id"].unique()}

    rows = []
    for (s, ch), g in hist.groupby(["style_id", "channel"]):
        g = g.sort_values("date")
        lid = f"{s}:{ch}"
        valid = g[g["sessions_valid"]]
        last7 = g[g["date"] > ds.today - timedelta(days=7)]
        v7 = last7[last7["sessions_valid"]]
        src = v7 if len(v7) >= 3 else valid
        organic = float((src["sessions"] - src["ad_clicks"]).clip(lower=0).mean()) if len(src) else 20.0
        clicks, spend = float(g["ad_clicks"].sum()), float(g["ad_spend_aed"].sum())
        cpc = spend / clicks if clicks > 0 else 1.8
        p_ref = float(g["price_aed"].iloc[-1])
        at_ref = valid[valid["price_aed"] == p_ref]
        cvr = (at_ref["units_sold"].sum() + 1.5) / (at_ref["sessions"].sum() + 100)  # lightly smoothed truth
        price = float(state.get("applied_prices", {}).get(lid, {}).get("price", p_ref))
        ad = float(state.get("applied_ads", {}).get(lid, {}).get("spend", last7["ad_spend_aed"].mean() if len(last7) else 0.0))
        on_hand = int(ds.inventory[(ds.inventory["style_id"] == s) & (ds.inventory["channel"] == ch)]["units_on_hand"].sum())
        exited = sum(e["units"] for e in state.get("exits", []) if e["listing_id"] == lid)
        on_hand = max(0, on_hand - exited)
        e = style_elasticity[s]
        for k in range(1, args.days + 1):
            day = ds.today + timedelta(days=k)
            fade = 1 - 0.004 * k
            org = organic * fade * rng.lognormal(0, 0.18)
            if args.shock == "competitor_cut" and ch == "amazon_ae":
                org *= 0.7
            sp = max(0.0, ad * (1 + rng.normal(0, 0.06))) if ad > 0 else 0.0
            cl = int(rng.poisson(sp / cpc)) if sp > 0 else 0
            sessions = int(round(org)) + cl
            cvr_p = float(np.clip(cvr * (price / p_ref) ** (-e) * rng.lognormal(0, 0.12), 0.0005, 0.3))
            units = int(min(on_hand, rng.binomial(sessions, cvr_p)))
            if args.shock == "spike" and lid == "S08:amazon_ae" and k == 1:
                units += 12
                sessions += 15
            on_hand -= units
            if args.shock == "noon_outage" and ch == "noon":
                sessions = 0
            rows.append(
                dict(
                    date=day.isoformat(),
                    style_id=s,
                    channel=ch,
                    price_aed=int(round(price)),
                    sessions=sessions,
                    ad_spend_aed=round(sp, 2),
                    ad_clicks=cl,
                    units_sold=units,
                )
            )
    out = pd.DataFrame(rows).sort_values(["date", "style_id", "channel"])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / "next_day.csv"
    out.to_csv(path, index=False)
    print(
        f"wrote {path}: {len(out)} rows for {out['date'].nunique()} day(s), {int(out['units_sold'].sum())} units sold, "
        f"ad spend AED {out['ad_spend_aed'].sum():,.0f}"
    )


if __name__ == "__main__":
    main()
