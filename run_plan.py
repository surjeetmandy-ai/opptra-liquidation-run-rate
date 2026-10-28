"""Print tomorrow's plan in the terminal. Same engine as the app; handy for a quick check,
for tests, and as a fallback if Streamlit misbehaves in a demo.

    python run_plan.py                      # real data in data/
    python run_plan.py --data path/to/folder  # any folder with the three CSVs
"""

import argparse
from pathlib import Path

from engine.config import Config
from engine.decide import build_plan, verdict_text
from engine.session import load_session
from engine.settings import DATA_DIR, STATE_PATH


def main() -> None:
    """Print tomorrow's plan in the terminal: verdict, ad totals, ranked actions with reasons and, with --all, the holds."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--state", default=str(STATE_PATH))
    ap.add_argument("--all", action="store_true", help="also list holds")
    args = ap.parse_args()
    ds, state = load_session(Path(args.data), Path(args.state))
    cfg = Config.from_dict(state.config_overrides)
    plan = build_plan(ds, state, cfg)

    print(f"Plan for {plan.plan_date}  ·  {plan.days_left} days left  ·  {len(plan.changes)} actions, {len(plan.holds)} holds")
    print(verdict_text(plan.verdict))
    print(
        "Ad spend tomorrow: "
        + ", ".join(f"{ch} AED {v:,.0f} (now {plan.ad_totals['now'][ch]:,.0f})" for ch, v in plan.ad_totals["plan"].items())
    )
    if plan.violations:
        print("!! GUARDRAIL VIOLATIONS:", plan.violations)
    print()
    for a in plan.changes:
        print(f"#{a.rank:2d}  {a.impact_aed:+8,.0f}  {a.headline()}")
        print(f"        {a.reason}")
        if a.flags:
            print(f"        flags: {'; '.join(a.flags)}")
    if args.all:
        print("\nHolds / watch:")
        for a in plan.holds:
            print(f"   {a.kind:5s} {a.listing_id:15s} {a.reason}")
    for n in plan.notes:
        print("note:", n)


if __name__ == "__main__":
    main()
