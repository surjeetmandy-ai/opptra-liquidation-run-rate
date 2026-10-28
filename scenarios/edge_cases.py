"""Things the live session might throw at the tool. Each case must load, plan, and stay legal.

python scenarios/edge_cases.py
"""

import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.config import Config  # noqa: E402
from engine.decide import build_plan  # noqa: E402
from engine.load import build_dataset, load_from_paths, merge_new_history  # noqa: E402
from engine.state import State  # noqa: E402

D = ROOT / "data"
INV, HIST, CONS = pd.read_csv(D / "inventory.csv"), pd.read_csv(D / "daily_history.csv"), pd.read_csv(D / "constraints.csv")


FAILURES: list = []  # an illegal plan or a crash in any scenario makes the script exit non-zero


def run(name, inv=None, h=None, c=None, cfg=None, ds=None):
    """Plan on a modified copy of the data (or a given dataset) and print one line: legal or not, number of actions,
    sell-through, and any notes.
    """
    try:
        ds = ds or build_dataset(
            (INV if inv is None else inv).copy(), (HIST if h is None else h).copy(), (CONS if c is None else c).copy()
        )
        p = build_plan(ds, State(), cfg or Config())
        notes = [n for n in p.notes if "sessions below" not in n]
        status = "OK " if not p.violations else "VIOLATION"
        if p.violations:
            FAILURES.append(name)
        print(
            f"{status} {name:38s} actions {len(p.changes):2d} | sell-through {p.verdict['sell_through_plan']:.0%} | "
            f"violations {[(v.rule, v.listing_id) for v in p.violations]} | conflicts {[(v.rule, v.listing_id) for v in getattr(p, 'conflicts', [])]}"
        )
        for n in notes[:3]:
            print(f"      note: {n}")
        return p
    except Exception as e:
        FAILURES.append(name)
        print(f"CRASH {name:38s} {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)


def main():
    """Run each awkward input in turn."""
    run("baseline")
    new = INV[(INV.style_id == "S05") & (INV.channel == "noon")].copy()
    new["style_id"], new["style_name"] = "S21", "New Kicks"
    c2 = pd.concat(
        [
            CONS,
            pd.DataFrame(
                [{"style_id": "S21", "style_name": "New Kicks", "category": "men", "mrp_aed": 249, "liquidation_floor_aed": 90}]
            ),
        ]
    )
    run("new listing, no history", inv=pd.concat([INV, new]), c=c2)
    c3 = CONS.copy()
    c3.loc[c3.style_id == "S05", "liquidation_floor_aed"] = 170
    run("floor raised above current price", c=c3)
    h3 = HIST.copy()
    h3.loc[(h3.style_id == "S12") & (h3.channel == "noon") & (h3.date == "2026-10-04"), "price_aed"] = 199
    run("parity already broken in data", h=h3)
    run("style missing from constraints", c=CONS[CONS.style_id != "S07"])
    i5 = INV.copy()
    i5.loc[(i5.style_id == "S16") & (i5.channel == "noon"), "units_on_hand"] = 0
    run("listing sold out", inv=i5)
    cfg = Config()
    cfg.ad_budget = {"amazon_ae": 1500.0, "noon": 0.0}
    run("Noon ad budget = 0", cfg=cfg)
    cfg = Config()
    cfg.ad_budget = {"amazon_ae": 750.0, "noon": 500.0}
    run("both budgets halved", cfg=cfg)
    cfg = Config()
    cfg.max_price_changes = 1
    cfg.parity_band = 0.03
    run("1 price change/week, parity ±3%", cfg=cfg)
    cfg = Config()
    cfg.horizon_days = 30
    run("deadline 30 days", cfg=cfg)
    cfg = Config()
    cfg.jobber_pct = 0.40
    run("jobber pays 40%", cfg=cfg)
    cfg = Config()
    cfg.jobber_pct = 0.10
    run("jobber pays 10%", cfg=cfg)
    cfg = Config()
    cfg.marketplace_fee_pct = 0.15
    run("15% marketplace fee", cfg=cfg)
    h9 = pd.concat(
        [
            HIST,
            pd.DataFrame(
                [
                    {
                        "date": "2026-10-04",
                        "style_id": "S99",
                        "channel": "noon",
                        "price_aed": 100,
                        "sessions": 10,
                        "ad_spend_aed": 0,
                        "ad_clicks": 0,
                        "units_sold": 0,
                    }
                ]
            ),
        ]
    )
    run("unknown style in history", h=h9)

    # a messy multi-day upload: renamed columns, display-name marketplaces, one style missing
    ds = load_from_paths(D / "inventory.csv", D / "daily_history.csv", D / "constraints.csv")
    nd = HIST[HIST.date.isin(["2026-10-03", "2026-10-04"])].copy()
    nd["date"] = (pd.to_datetime(nd.date) + pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d")
    nd = nd[nd.style_id != "S11"].rename(
        columns={
            "units_sold": "Units Sold",
            "price_aed": "Price",
            "sessions": "Sessions",
            "ad_spend_aed": "Ad Spend",
            "ad_clicks": "Clicks",
            "channel": "Marketplace",
            "style_id": "Style",
        }
    )
    nd["Marketplace"] = nd["Marketplace"].map({"amazon_ae": "Amazon.ae", "noon": "Noon"})
    try:
        ds2, issues = merge_new_history(ds, nd)
        print(f"     multi-day messy upload: data through {ds2.today}; " + "; ".join(issues[:3]))
        run("after messy 2-day upload", ds=ds2)
    except Exception as e:
        FAILURES.append("multi-day messy upload")
        print("CRASH multi-day messy upload", e)
        traceback.print_exc(limit=3)
    if FAILURES:
        raise SystemExit(f"edge cases failed: {FAILURES}")
    print("every edge case produced a legal plan")


if __name__ == "__main__":
    main()
