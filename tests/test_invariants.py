"""The brief's hard rules, re-stated as checks that know nothing about the engine's internals.

`tests/test_engine.py` tests the parts. This module tests the whole: whatever the conditions, every
plan the tool produces must satisfy the rules Farah is held to, and the same inputs must give the
same plan. The checker below is written from the brief, not from `engine/guardrail.py`, so the two
can disagree, and a disagreement is a bug in one of them.

Rules checked (from the brief):
  1. price >= liquidation floor, and <= MRP
  2. Noon price within +/- parity band of Amazon.ae, per style, after both channels move
  3. total daily ad spend per channel <= that channel's cap
  4. at most `max_price_changes` price changes per listing in any rolling 7-day window,
     counting changes visible in the history and changes the tool itself applied
  5. never exit more pairs than are on hand, per listing and per size
  6. one decision per listing per day; a same-day re-plan proposes nothing new
  7. a rule already broken in the incoming data is never made worse; it is fixed or declared
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from engine.apply import apply_action, score_predictions
from engine.config import Config
from engine.decide import build_plan
from engine.load import load_from_paths, merge_new_history
from engine.state import State

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AMAZON, NOON = "amazon_ae", "noon"


def load_real():
    return load_from_paths(DATA / "inventory.csv", DATA / "daily_history.csv", DATA / "constraints.csv")


def history_price_change_dates(history: pd.DataFrame, listing_id: str) -> set:
    """Dates on which a listing's price in the history differs from the day before (>= 1%)."""
    style, ch = listing_id.split(":")
    h = history[(history["style_id"] == style) & (history["channel"] == ch)].sort_values("date")
    prices = h["price_aed"].astype(float).to_numpy()
    dates = pd.to_datetime(h["date"]).dt.date.to_numpy()
    out = set()
    for i in range(1, len(prices)):
        if prices[i - 1] > 0 and abs(prices[i] / prices[i - 1] - 1) >= 0.01:
            out.add(dates[i])
    return out


def rules_broken(plan, ds, state: State, cfg: Config) -> list:
    """Every way the plan breaks the brief, as plain sentences. Empty means legal."""
    t = plan.table.set_index("listing_id")
    problems = []
    seen = set()

    def planned_price(lid):
        a = by_listing.get(lid)
        p = a.price_to if a is not None and a.price_to is not None and a.price_to == a.price_to else None
        return p if p is not None else (float(t.loc[lid, "price_now"]) if pd.notna(t.loc[lid, "price_now"]) else None)

    def planned_ads(lid):
        a = by_listing.get(lid)
        return float(a.ads_to) if a is not None and a.ads_to is not None else float(t.loc[lid, "ad_now"])

    by_listing = {}
    for a in plan.actions:
        if a.listing_id in seen:
            problems.append(f"{a.listing_id}: more than one action for the listing")
        seen.add(a.listing_id)
        by_listing[a.listing_id] = a
    for lid in t.index:
        if lid not in seen:
            problems.append(f"{lid}: no decision (not even a hold)")

    # 1. floor and MRP, never made worse than the data arrived
    for lid in t.index:
        p = planned_price(lid)
        if p is None:
            continue
        floor, mrp, p0 = t.loc[lid, "floor"], t.loc[lid, "mrp"], t.loc[lid, "price_now"]
        if pd.notna(floor) and p < floor - 1e-9 and not (pd.notna(p0) and p0 < floor - 1e-9 and p >= p0 - 1e-9):
            problems.append(f"{lid}: price {p} below floor {floor}")
        if pd.notna(mrp) and p > mrp + 1e-9 and not (pd.notna(p0) and p0 > mrp + 1e-9 and p <= p0 + 1e-9):
            problems.append(f"{lid}: price {p} above MRP {mrp}")

    # 2. parity per style, after both channels move; a pre-existing breach may stay but never widen
    for style, grp in t.groupby("style_id"):
        la, ln = f"{style}:{AMAZON}", f"{style}:{NOON}"
        if la not in grp.index or ln not in grp.index:
            continue
        pa, pn = planned_price(la), planned_price(ln)
        if pa is None or pn is None or pa <= 0:
            continue
        dev = abs(pn / pa - 1)
        if dev > cfg.parity_band + 1e-9:
            pa0, pn0 = t.loc[la, "price_now"], t.loc[ln, "price_now"]
            dev0 = abs(pn0 / pa0 - 1) if pd.notna(pa0) and pd.notna(pn0) and pa0 > 0 else 0.0
            if dev0 <= cfg.parity_band + 1e-9 or dev > dev0 + 1e-9:
                problems.append(f"{style}: parity {dev:.1%} after plan (was {dev0:.1%}), band {cfg.parity_band:.0%}")

    # 3. per-channel daily cap
    for ch, grp in t.groupby("channel"):
        total = sum(planned_ads(lid) for lid in grp.index)
        cap = float(cfg.ad_budget.get(ch, 0.0))
        if total > cap + 1e-6:
            problems.append(f"{ch}: planned ads {total:.2f} over the cap {cap:.0f}")
        for lid in grp.index:
            if planned_ads(lid) < 0:
                problems.append(f"{lid}: negative ad spend")

    # 4. rolling window of price changes, counted from history + the tool's own applied changes
    lo, hi = plan.plan_date - timedelta(days=cfg.change_window_days - 1), plan.plan_date
    for a in plan.changes:
        if a.price_to is None or a.price_from is None or a.price_to != a.price_to or a.price_to == a.price_from:
            continue
        dates = history_price_change_dates(ds.history, a.listing_id) | state.price_change_dates(a.listing_id)
        n = sum(1 for d in dates if lo <= d <= hi)
        if n + 1 > cfg.max_price_changes:
            problems.append(
                f"{a.listing_id}: {n} price changes already in {lo}..{hi}, plus today's, limit {cfg.max_price_changes}"
            )

    # 5. exits within stock, per listing and per size
    for a in plan.changes:
        if a.exit_units <= 0:
            continue
        if a.exit_units > float(t.loc[a.listing_id, "on_hand"]) + 1e-9:
            problems.append(f"{a.listing_id}: exits {a.exit_units} exceed stock {t.loc[a.listing_id, 'on_hand']}")
        if a.exit_by_size is not None:
            stock = plan.sizes[a.listing_id]["on_hand"]
            if len(a.exit_by_size) != len(stock) or any(
                e > s + 1e-9 or e < 0 for e, s in zip(a.exit_by_size, stock, strict=True)
            ):
                problems.append(f"{a.listing_id}: exits by size {a.exit_by_size} do not fit stock {stock}")
    return problems


def random_conditions(seed: int) -> Config:
    """A plausible but arbitrary set of conditions Opptra could change in the live session."""
    rng = np.random.default_rng(seed)
    caps = [0.0, 200.0, 500.0, 1000.0, 1500.0, 2500.0]
    return Config(
        horizon_days=int(rng.integers(15, 91)),
        jobber_pct=float(rng.uniform(0.05, 0.6)),
        ad_budget={AMAZON: float(rng.choice(caps)), NOON: float(rng.choice(caps))},
        marketplace_fee_pct=float(rng.uniform(0.0, 0.2)),
        parity_band=float(rng.uniform(0.02, 0.1)),
        max_price_changes=int(rng.integers(1, 4)),
    )


class TestInvariants(unittest.TestCase):
    def test_default_conditions_are_legal(self):
        ds = load_real()
        plan = build_plan(ds, State(), Config())
        self.assertEqual(rules_broken(plan, ds, State(), Config()), [])
        self.assertEqual(plan.violations, [])

    def test_random_conditions_are_always_legal(self):
        ds = load_real()
        for seed in range(12):
            cfg = random_conditions(seed)
            with self.subTest(
                seed=seed,
                cfg=cfg.model_dump(include={"horizon_days", "jobber_pct", "ad_budget", "parity_band", "max_price_changes"}),
            ):
                plan = build_plan(ds, State(), cfg)
                self.assertEqual(rules_broken(plan, ds, State(), cfg), [])
                self.assertEqual(plan.violations, [], "the engine's own guardrail disagrees with the brief checker")
                self.assertFalse(any("guardrail withdrew" in n for n in plan.notes), "the generator proposed something illegal")

    def test_the_checker_itself_catches_broken_plans(self):
        """A checker that never fires proves nothing: break each rule by hand and make sure it does."""
        ds = load_real()
        cfg = Config()
        plan = build_plan(ds, State(), cfg)
        t = plan.table.set_index("listing_id")
        self.assertEqual(rules_broken(plan, ds, State(), cfg), [])

        def tampered(fn):
            p = build_plan(ds, State(), cfg)
            fn(p)
            return rules_broken(p, ds, State(), cfg)

        def below_floor(p):
            a = next(x for x in p.actions if x.channel == AMAZON)
            a.price_from, a.price_to = float(t.loc[a.listing_id, "price_now"]), float(t.loc[a.listing_id, "floor"]) - 5

        def over_cap(p):
            a = next(x for x in p.actions if x.channel == NOON)
            a.ads_from, a.ads_to = float(t.loc[a.listing_id, "ad_now"]), cfg.ad_budget[NOON] + 50

        def out_of_band(p):
            a = next(x for x in p.actions if x.channel == NOON)
            partner = next(x for x in p.actions if x.listing_id == f"{a.style_id}:{AMAZON}")
            pa = float(partner.price_to if partner.price_to == partner.price_to else t.loc[partner.listing_id, "price_now"])
            a.price_from, a.price_to = float(t.loc[a.listing_id, "price_now"]), round(pa * 0.8)

        def too_many_exits(p):
            a = next(x for x in p.actions if x.exit_units == 0)
            a.kind, a.exit_units = "exit", int(t.loc[a.listing_id, "on_hand"]) + 1

        def above_mrp(p):
            a = next(x for x in p.actions if x.channel == AMAZON)
            a.price_from, a.price_to = float(t.loc[a.listing_id, "price_now"]), float(t.loc[a.listing_id, "mrp"]) + 5

        def negative_ads(p):
            a = next(x for x in p.actions if x.channel == NOON)
            a.ads_from, a.ads_to = float(t.loc[a.listing_id, "ad_now"]), -10.0

        def third_change_in_window(p):
            a = next(x for x in p.actions if x.price_to is not None and x.price_to == x.price_to and x.price_to != x.price_from)
            st = State()
            for k in range(cfg.max_price_changes):
                st.apply_price(a.listing_id, 300.0 - k, 299.0 - k, p.plan_date - timedelta(days=k + 1))
            return rules_broken(p, ds, st, cfg)

        for name, fn in [
            ("floor", below_floor),
            ("budget", over_cap),
            ("parity", out_of_band),
            ("exits", too_many_exits),
            ("mrp", above_mrp),
            ("negative ads", negative_ads),
        ]:
            with self.subTest(rule=name):
                self.assertTrue(tampered(fn), f"the checker missed a broken {name} rule")
        with self.subTest(rule="rolling window"):
            self.assertTrue(third_change_in_window(build_plan(ds, State(), cfg)))

    def test_same_inputs_same_plan(self):
        ds = load_real()
        a = build_plan(ds, State(), Config())
        b = build_plan(ds, State(), Config())
        self.assertEqual([x.headline() for x in a.actions], [x.headline() for x in b.actions])
        self.assertEqual([round(x.impact_aed, 6) for x in a.actions], [round(x.impact_aed, 6) for x in b.actions])
        self.assertEqual(a.verdict["plan"]["net"], b.verdict["plan"]["net"])

    def test_every_change_moves_something(self):
        plan = build_plan(load_real(), State(), Config())
        for a in plan.changes:
            moved = (
                (a.price_to is not None and a.price_from is not None and a.price_to != a.price_from)
                or (a.ads_to is not None and a.ads_from is not None and a.ads_to != a.ads_from)
                or a.exit_units > 0
            )
            self.assertTrue(moved, f"{a.listing_id} is listed as a change but changes nothing: {a.headline()}")

    def test_four_day_loop_stays_legal_and_does_not_repeat_itself(self):
        """Apply everything, simulate the next day, merge, re-plan: the rolling window and the
        cooldowns are only tested for real once the tool has its own changes in the log."""
        ds = load_real()
        state = State()
        state.campaign_start = ds.today + timedelta(days=1)
        cfg = Config()
        tmp = Path(tempfile.mkdtemp())
        try:
            for day in range(1, 5):
                plan = build_plan(ds, state, cfg)
                with self.subTest(day=day):
                    self.assertEqual(rules_broken(plan, ds, state, cfg), [])
                    self.assertEqual(plan.violations, [])
                    self.assertEqual(plan.days_left, cfg.horizon_days - (day - 1))
                for a in plan.changes:
                    apply_action(ds, state, a, plan.plan_date, plan.sizes[a.listing_id]["sizes"])
                again = build_plan(ds, state, cfg)
                self.assertEqual([a.headline() for a in again.changes], [], f"day {day}: a same-day re-plan proposed changes")
                self.assertEqual(rules_broken(again, ds, state, cfg), [])
                state.save(tmp / "state.json")
                ds.history.drop(columns=["sessions_valid"]).to_csv(tmp / "hist.csv", index=False)
                r = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tools" / "simulate_next_day.py"),
                        "--state",
                        str(tmp / "state.json"),
                        "--history",
                        str(tmp / "hist.csv"),
                        "--out",
                        str(tmp),
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(ROOT),
                )
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                ds, _ = merge_new_history(ds, pd.read_csv(tmp / "next_day.csv"))
                score_predictions(ds, state)
            # the state survives a round trip through disk with every change intact
            state.save(tmp / "state.json")
            back = State.load(tmp / "state.json")
            self.assertEqual(len(back.change_log), len(state.change_log))
            self.assertEqual(json.loads((tmp / "state.json").read_text())["campaign_start"], str(state.campaign_start))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
