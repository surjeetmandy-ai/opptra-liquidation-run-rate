"""Unit tests for the rules that size a decision, on small hand-built fixtures.

`test_engine.py` runs the engine on the real data; `test_invariants.py` checks every plan against the
brief. This file pins the individual rules and their boundaries, so that a change to a threshold, a
window length or a comparison operator fails a test instead of quietly changing Farah's plan.
An independent audit found that about half of 51 realistic mutations survived the suite before these
existed; each test here is named for the mutation it kills.
"""

import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from engine.actions import Action
from engine.config import Config
from engine.decide import build_plan, verdict_text
from engine.economics import expected_rate, project
from engine.guardrail import check_plan
from engine.load import build_dataset, clean_history, load_from_paths, merge_new_history, normalise_channel
from engine.state import State

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AMAZON, NOON = "amazon_ae", "noon"


def load_real():
    return load_from_paths(DATA / "inventory.csv", DATA / "daily_history.csv", DATA / "constraints.csv")


def row(**over) -> pd.Series:
    """One listing as the estimator would describe it, with sensible defaults; override what the test is about."""
    base = dict(
        listing_id="S01:amazon_ae",
        style_id="S01",
        channel=AMAZON,
        on_hand=60.0,
        jobber_value=30.0,
        mrp=300.0,
        floor=150.0,
        price_ref=200.0,
        price_now=200.0,
        ad_now=50.0,
        cpc=2.0,
        organic_7=40.0,
        sessions_7=65.0,
        locked=False,
        days_at_price=999,
        days_at_ads=999,
        calib_price=1.0,
        calib_ads=1.0,
        cvr_mean=0.02,
        cvr_p10=0.012,
        cvr_p90=0.03,
        elasticity=2.0,
        elasticity_sd=0.6,
        required_rate=1.0,
        ratio=1.0,
        status="behind",
        slots=2,
        changes_in_window=0,
        is_new=False,
        listing_problem=False,
        checked_days=-1,
        exited_today=False,
        at_floor=False,
        notes=[],
    )
    base.update(over)
    return pd.Series(base)


def table(*rows: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in rows])


def action(lid: str, **over) -> Action:
    style, ch = lid.split(":")
    base = dict(listing_id=lid, style_id=style, channel=ch, style_name=style, kind="price")
    base.update(over)
    return Action(**base)


class TestConfigMatchesBrief(unittest.TestCase):
    def test_the_briefs_numbers_are_the_defaults(self):
        cfg = Config()
        self.assertEqual(cfg.horizon_days, 60)
        self.assertEqual(cfg.jobber_pct, 0.25)
        self.assertAlmostEqual(cfg.holding_per_unit_per_day, 3.0 / 30)
        self.assertEqual(cfg.ad_budget, {AMAZON: 1500.0, NOON: 1000.0})
        self.assertEqual(cfg.parity_band, 0.05)
        self.assertEqual(cfg.max_price_changes, 2)
        self.assertEqual(cfg.change_window_days, 7)
        self.assertEqual(cfg.marketplace_fee_pct, 0.0)  # the brief has no fee
        self.assertEqual(cfg.sell_through_target, 0.95)  # my assumption for "make it", pinned so it can't drift silently


class TestGuardrailBoundaries(unittest.TestCase):
    """check_plan on two listings of one style, at the exact edge of every rule."""

    def setUp(self):
        self.cfg = Config()
        self.t = table(
            row(listing_id="S01:amazon_ae", channel=AMAZON, price_now=200.0, ad_now=100.0, on_hand=60.0),
            row(listing_id="S01:noon", channel=NOON, price_now=200.0, ad_now=100.0, on_hand=40.0),
            row(
                listing_id="S02:amazon_ae", style_id="S02", channel=AMAZON, price_now=200.0, ad_now=0.0, on_hand=40.0
            ),  # one channel only
        )
        self.cfg.ad_budget = {AMAZON: 150.0, NOON: 150.0}

    def rules(self, *actions):
        return sorted(v.rule for v in check_plan(list(actions), self.t, self.cfg))

    def test_parity_at_exactly_the_band_passes_and_a_hair_over_fails(self):
        self.assertEqual(self.rules(action("S01:noon", price_from=200.0, price_to=210.0)), [])  # +5.0%
        self.assertEqual(self.rules(action("S01:noon", price_from=200.0, price_to=190.0)), [])  # -5.0%
        self.assertEqual(self.rules(action("S01:noon", price_from=200.0, price_to=210.5)), ["parity"])
        self.assertEqual(self.rules(action("S01:noon", price_from=200.0, price_to=189.5)), ["parity"])

    def test_parity_is_judged_after_both_channels_move(self):
        both = (action("S01:amazon_ae", price_from=200.0, price_to=160.0), action("S01:noon", price_from=200.0, price_to=168.0))
        self.assertEqual(self.rules(*both), [])  # 168/160 = +5.0%
        self.assertEqual(self.rules(both[0]), ["parity"])  # Amazon alone: Noon 200 vs 160 = +25%

    def test_budget_at_the_cap_passes_and_one_fil_over_fails(self):
        self.assertEqual(self.rules(action("S01:amazon_ae", kind="ads", ads_from=100.0, ads_to=150.0)), [])
        self.assertEqual(self.rules(action("S01:amazon_ae", kind="ads", ads_from=100.0, ads_to=150.01)), ["budget"])
        # the cap is per channel: Noon's spend does not count against Amazon's cap
        self.assertEqual(
            self.rules(
                action("S01:amazon_ae", kind="ads", ads_from=100.0, ads_to=150.0),
                action("S01:noon", kind="ads", ads_from=100.0, ads_to=150.0),
            ),
            [],
        )

    def test_negative_ad_spend_is_a_violation(self):
        self.assertEqual(self.rules(action("S01:amazon_ae", kind="ads", ads_from=100.0, ads_to=-1.0)), ["ads"])

    def test_floor_and_mrp_edges(self):
        self.assertEqual(self.rules(action("S02:amazon_ae", price_from=200.0, price_to=150.0)), [])  # at the floor
        self.assertEqual(self.rules(action("S02:amazon_ae", price_from=200.0, price_to=149.0)), ["floor"])
        self.assertEqual(self.rules(action("S02:amazon_ae", price_from=200.0, price_to=300.0)), [])  # at MRP
        self.assertEqual(self.rules(action("S02:amazon_ae", price_from=200.0, price_to=301.0)), ["mrp"])

    def test_exits_up_to_stock_pass_and_one_more_fails(self):
        self.assertEqual(self.rules(action("S02:amazon_ae", kind="exit", exit_units=40)), [])
        self.assertEqual(self.rules(action("S02:amazon_ae", kind="exit", exit_units=41)), ["exit"])

    def test_a_price_change_with_no_slot_left_is_caught(self):
        self.t.loc[self.t.listing_id == "S01:noon", "slots"] = 0
        self.assertEqual(self.rules(action("S01:noon", price_from=200.0, price_to=195.0)), ["change_limit"])
        self.assertEqual(self.rules(action("S01:noon", price_from=200.0, price_to=200.0)), [])  # no change, no slot used


class TestRollingWindow(unittest.TestCase):
    """The 7-day window ends on the plan date: a change 6 days before counts, 7 days before does not."""

    def setUp(self):
        self.ds = load_real()
        self.plan_date = self.ds.today + timedelta(days=1)

    def slots(self, state: State) -> int:
        t = build_plan(self.ds, state, Config()).table.set_index("listing_id")
        return int(t.loc["S05:amazon_ae", "slots"])

    def test_window_length_is_exactly_seven_days(self):
        self.assertEqual(self.slots(State()), 2)
        st = State()
        st.apply_price("S05:amazon_ae", 159.0, 155.0, self.plan_date - timedelta(days=6))
        self.assertEqual(self.slots(st), 1)
        st = State()
        st.apply_price("S05:amazon_ae", 159.0, 155.0, self.plan_date - timedelta(days=7))
        self.assertEqual(self.slots(st), 2)

    def test_a_change_on_the_plan_date_itself_counts(self):
        st = State()
        st.apply_price("S05:amazon_ae", 159.0, 155.0, self.plan_date)
        self.assertEqual(self.slots(st), 1)

    def test_a_shorter_window_setting_is_honoured(self):
        cfg = Config()
        cfg.change_window_days = 3
        st = State()
        st.apply_price("S05:amazon_ae", 159.0, 155.0, self.plan_date - timedelta(days=3))
        t = build_plan(self.ds, st, cfg).table.set_index("listing_id")
        self.assertEqual(int(t.loc["S05:amazon_ae", "slots"]), 2)
        st.apply_price("S05:amazon_ae", 155.0, 150.0, self.plan_date - timedelta(days=2))
        t = build_plan(self.ds, st, cfg).table.set_index("listing_id")
        self.assertEqual(int(t.loc["S05:amazon_ae", "slots"]), 1)


class TestAllocator(unittest.TestCase):
    """allocate_ads on a hand-built channel of three listings."""

    def setUp(self):
        from engine.allocate import allocate_ads

        self.allocate = allocate_ads
        self.cfg = Config()
        self.days = 60

    def sizes_for(self, t):
        return {
            r["listing_id"]: {"on_hand": [float(r["on_hand"])], "demand_share": [1.0], "sizes": ["42"]} for _, r in t.iterrows()
        }

    def test_a_raise_never_exceeds_the_per_listing_cap(self):
        starved = row(
            listing_id="S02:noon",
            style_id="S02",
            channel=NOON,
            ad_now=10.0,
            organic_7=5.0,
            sessions_7=10.0,
            cvr_mean=0.04,
            cvr_p90=0.05,
            on_hand=200.0,
            required_rate=3.4,
            ratio=0.2,
            status="critical",
        )
        t = table(starved)
        self.cfg.ad_budget = {NOON: 1000.0}
        out = self.allocate(t, {}, self.cfg, self.days, self.sizes_for(t))
        spend = out["S02:noon"]["spend"]
        self.assertGreater(spend, 10.0)
        self.assertLessEqual(spend, max(self.cfg.max_ad_multiplier * 10.0, self.cfg.min_ad_cap))

    def test_an_on_track_listing_never_gets_a_raise(self):
        # on track by its overall rate, but with a skewed size curve two sizes will be left over, so a
        # raise would look worthwhile to the greedy step: only the on-track rule keeps the money away
        fine = row(
            listing_id="S03:noon",
            style_id="S03",
            channel=NOON,
            ad_now=20.0,
            cpc=0.8,
            organic_7=30.0,
            sessions_7=55.0,
            cvr_mean=0.05,
            cvr_p90=0.06,
            ratio=1.05,
            status="on_track",
            required_rate=2.6,
            on_hand=156.0,
        )
        t = table(fine)
        self.cfg.ad_budget = {NOON: 1000.0}
        sizes = {"S03:noon": {"on_hand": [100.0, 30.0, 26.0], "demand_share": [0.8, 0.12, 0.08], "sizes": ["41", "42", "43"]}}
        self.assertLess(project(fine, 200.0, 20.0, None, self.days, self.cfg, sizes["S03:noon"]).rate, 1.2 * 2.6)
        self.assertGreater(
            project(fine, 200.0, 40.0, None, self.days, self.cfg, sizes["S03:noon"]).net,
            project(fine, 200.0, 20.0, None, self.days, self.cfg, sizes["S03:noon"]).net,
        )  # the raise would pay by the projection
        out = self.allocate(t, {}, self.cfg, self.days, sizes)
        self.assertLessEqual(out["S03:noon"]["spend"], 20.0)
        # the same listing, merely behind, is funded
        t2 = table(row(**{**fine.to_dict(), "ratio": 0.95, "status": "behind"}))
        out2 = self.allocate(t2, {}, self.cfg, self.days, sizes)
        self.assertGreater(out2["S03:noon"]["spend"], 20.0)

    def test_a_listing_whose_projection_peaks_in_the_middle_is_trimmed_not_stopped(self):
        # organic traffic alone sells most of the stock; a little paid traffic finishes it, a lot only costs
        r = row(
            listing_id="S04:noon",
            style_id="S04",
            channel=NOON,
            ad_now=200.0,
            cpc=2.0,
            organic_7=40.0,
            sessions_7=140.0,
            cvr_mean=0.02,
            on_hand=55.0,
            required_rate=0.9,
            ratio=1.8,
            status="on_track",
        )
        t = table(r)
        self.cfg.ad_budget = {NOON: 1000.0}
        sizes = self.sizes_for(t)
        out = self.allocate(t, {}, self.cfg, self.days, sizes)
        best = out["S04:noon"]["spend"]
        self.assertEqual(out["S04:noon"]["kind"], "trim")
        self.assertGreater(best, 0.0)
        self.assertLess(best, 200.0)
        net = lambda s: project(r, 200.0, s, None, self.days, self.cfg, sizes["S04:noon"]).net  # noqa: E731
        self.assertGreaterEqual(net(best), net(0.0))
        self.assertGreaterEqual(net(best), net(200.0))

    def test_when_the_cap_drops_below_todays_spend_the_locked_listing_pays_last(self):
        a = row(
            listing_id="S05:noon",
            style_id="S05",
            channel=NOON,
            ad_now=100.0,
            days_at_ads=1,
            on_hand=300.0,
            required_rate=5.0,
            ratio=0.3,
            status="critical",
        )
        b = row(
            listing_id="S06:noon",
            style_id="S06",
            channel=NOON,
            ad_now=100.0,
            on_hand=300.0,
            required_rate=5.0,
            ratio=0.3,
            status="critical",
        )
        t = table(a, b)
        self.cfg.ad_budget = {NOON: 150.0}
        out = self.allocate(t, {}, self.cfg, self.days, self.sizes_for(t))
        total = sum(v["spend"] for v in out.values())
        self.assertLessEqual(total, 150.0 + 1e-6)
        self.assertGreaterEqual(out["S05:noon"]["spend"], out["S06:noon"]["spend"])  # the locked one keeps more


class TestPriceCandidatesAndPick(unittest.TestCase):
    def test_cut_candidates_are_five_dirham_steps_between_five_and_twenty_percent_never_below_floor(self):
        from engine.decide import _cut_candidates

        cfg = Config()
        self.assertEqual(
            _cut_candidates(row(price_now=200.0, floor=150.0), cfg), [200.0, 190.0, 185.0, 180.0, 175.0, 170.0, 165.0, 160.0]
        )
        self.assertEqual(_cut_candidates(row(price_now=200.0, floor=172.0), cfg), [200.0, 190.0, 185.0, 180.0, 175.0])
        self.assertEqual(_cut_candidates(row(price_now=200.0, floor=150.0, days_at_price=1), cfg), [200.0])  # cooldown
        self.assertEqual(_cut_candidates(row(price_now=200.0, floor=150.0, slots=0), cfg), [200.0])  # no slot
        self.assertEqual(_cut_candidates(row(price_now=200.0, floor=150.0, is_new=True), cfg), [200.0])  # too little data

    def test_pick_modes(self):
        from engine.decide import _pick

        cfg = Config()
        p0 = {AMAZON: 200.0, NOON: 200.0}
        scored = [(200.0, 200.0, 1000.0), (190.0, 190.0, 1100.0), (180.0, 180.0, 1300.0), (160.0, 160.0, 1250.0)]
        # closing: the shallowest pair that reaches the target
        closing = [(180.0, 180.0, 1300.0), (160.0, 160.0, 1250.0)]
        self.assertEqual(_pick(scored, closing, False, p0, cfg)[:2], (180.0, 180.0))
        # critical and nothing closes: the best projected recovery
        self.assertEqual(_pick(scored, [], True, p0, cfg)[:2], (180.0, 180.0))
        # behind and nothing closes: the pair closest to the default first step (10%)
        self.assertEqual(_pick(scored, [], False, p0, cfg)[:2], (180.0, 180.0))
        scored2 = [(200.0, 200.0, 1000.0), (185.0, 185.0, 1050.0), (170.0, 170.0, 1400.0)]  # 7.5% vs 15%: 7.5% is closer
        self.assertEqual(_pick(scored2, [], False, p0, cfg)[:2], (185.0, 185.0))

    def test_the_price_gate_holds_a_style_whose_cut_does_not_pay_enough(self):
        ds = load_real()
        loose, tight = Config(), Config()
        loose.price_min_gain_aed = 0.0
        tight.price_min_gain_aed = 1_000_000.0
        cuts = lambda cfg: sum(1 for a in build_plan(ds, State(), cfg).changes if a.price_to != a.price_from and not a.must_do)  # noqa: E731
        self.assertEqual(cuts(tight), 0)
        self.assertGreater(cuts(Config()), 0)
        self.assertGreaterEqual(cuts(loose), cuts(Config()))

    def test_the_generator_never_relies_on_the_guardrail(self):
        plan = build_plan(load_real(), State(), Config())
        self.assertFalse(any("guardrail withdrew" in n for n in plan.notes), plan.notes)


class TestEstimator(unittest.TestCase):
    def setUp(self):
        self.inv, self.h, self.c = (pd.read_csv(DATA / f) for f in ("inventory.csv", "daily_history.csv", "constraints.csv"))

    def table_for(self, h=None, inv=None):
        return build_plan(
            build_dataset(inv if inv is not None else self.inv, h if h is not None else self.h, self.c), State(), Config()
        ).table.set_index("listing_id")

    def test_run_rate_is_the_trailing_seven_day_mean(self):
        h = self.h.copy()
        sel = (h.style_id == "S05") & (h.channel == AMAZON)
        dates = sorted(h.loc[sel, "date"].unique())
        h.loc[sel & h.date.isin(dates[-7:]), "units_sold"] = 2
        h.loc[sel & h.date.isin(dates[:-7]), "units_sold"] = 0
        t = self.table_for(h)
        self.assertAlmostEqual(float(t.loc["S05:amazon_ae", "rr7"]), 2.0)
        self.assertAlmostEqual(float(t.loc["S05:amazon_ae", "rr14"]), 1.0)

    def test_status_thresholds_are_one_and_a_half(self):
        t = self.table_for()
        for lid, r in t.iterrows():
            if r["on_hand"] <= 0:
                self.assertEqual(r["status"], "sold_out", lid)
            elif r["ratio"] >= 1.0:
                self.assertEqual(r["status"], "on_track", lid)
            elif r["ratio"] >= 0.5:
                self.assertEqual(r["status"], "behind", lid)
            else:
                self.assertEqual(r["status"], "critical", lid)
        self.assertTrue({"on_track", "behind", "critical"} <= set(t["status"]))

    def test_days_with_impossible_sessions_are_missing_for_conversion_but_their_orders_count(self):
        before = self.table_for()
        h = self.h.copy()
        sel = (h.style_id == "S05") & (h.channel == NOON) & (h.date == "2026-09-28")
        h.loc[sel, "sessions"] = 0  # sessions below ad clicks: a reporting gap, not a demand collapse
        h.loc[sel, "ad_clicks"] = 10
        after = self.table_for(h)
        dropped = self.table_for(self.h[~sel])  # the same day simply absent from the export
        self.assertAlmostEqual(float(after.loc["S05:noon", "cvr_mean"]), float(dropped.loc["S05:noon", "cvr_mean"]), places=9)
        self.assertNotAlmostEqual(float(after.loc["S05:noon", "cvr_mean"]), float(before.loc["S05:noon", "cvr_mean"]), places=9)
        self.assertEqual(float(before.loc["S05:noon", "rr7"]), float(after.loc["S05:noon", "rr7"]))  # orders are kept

    def test_a_new_listing_borrows_its_price_from_the_other_channel(self):
        inv = self.inv.copy()
        new = inv[(inv.style_id == "S05") & (inv.channel == NOON)].copy()
        new["channel"] = AMAZON
        inv = pd.concat([inv[~((inv.style_id == "S05") & (inv.channel == AMAZON))], new])
        h = self.h[~((self.h.style_id == "S05") & (self.h.channel == AMAZON))]
        t = self.table_for(h, inv)
        self.assertTrue(bool(t.loc["S05:amazon_ae", "is_new"]))
        self.assertEqual(float(t.loc["S05:amazon_ae", "price_now"]), float(t.loc["S05:noon", "price_now"]))

    def test_calibration_weights_scale_the_uplift_only(self):
        cfg = Config()
        r = row(price_ref=200.0, price_now=200.0)
        base = expected_rate(r, 200.0, 50.0, cfg)
        full = expected_rate(r, 160.0, 50.0, cfg)
        r2 = row(price_ref=200.0, price_now=200.0, calib_price=0.5)
        half = expected_rate(r2, 160.0, 50.0, cfg)
        self.assertAlmostEqual(expected_rate(r2, 200.0, 50.0, cfg), base)  # baseline untouched
        self.assertAlmostEqual(half, base + 0.5 * (full - base))
        r3 = row(calib_ads=0.5)
        up = expected_rate(r, 200.0, 100.0, cfg)
        self.assertAlmostEqual(expected_rate(r3, 200.0, 100.0, cfg), base + 0.5 * (up - base))

    def test_calibration_is_bounded(self):
        from engine.estimate import lever_calibration

        cfg = Config()
        cfg.calibration_shrink_actions = 0  # no shrinkage: the bound alone must hold the weight
        st = State()
        for _ in range(cfg.calibration_min_actions):
            st.record_prediction(date(2026, 10, 1), "S01:amazon_ae", "ads", 2.0, 1.0)
            st.predictions[-1].update(actual_rate=10.0, days_observed=5)  # delivered 9x the promise
        self.assertEqual(lever_calibration(st, cfg)["ads"], cfg.calibration_bounds[1])
        for p in st.predictions:
            p["actual_rate"] = 0.0  # delivered nothing at all
        self.assertEqual(lever_calibration(st, cfg)["ads"], cfg.calibration_bounds[0])


class TestEconomicsEdges(unittest.TestCase):
    def test_an_ad_driven_sale_must_beat_the_jobber_by_the_safety_margin(self):
        from engine.economics import ads_pay_for_themselves

        cfg = Config()  # safety margin AED 5
        r = row(price_now=100.0, price_ref=100.0, cpc=2.0, cvr_mean=0.04)  # AED 50 of ads per sale: AED 50 left of a 100 price
        self.assertTrue(ads_pay_for_themselves(row(**{**r.to_dict(), "jobber_value": 44.0}), 100.0, cfg))  # 50 > 44 + 5
        self.assertFalse(ads_pay_for_themselves(row(**{**r.to_dict(), "jobber_value": 47.0}), 100.0, cfg))  # 50 < 47 + 5
        self.assertFalse(
            ads_pay_for_themselves(row(**{**r.to_dict(), "jobber_value": 45.0}), 100.0, cfg)
        )  # 50 = 45 + 5: not strictly better

    def test_exits_are_sized_on_the_optimistic_read_and_floored_per_size(self):
        from engine.decide import _size_exit, _Work

        cfg = Config()
        r = row(
            listing_id="S09:noon",
            style_id="S09",
            channel=NOON,
            on_hand=120.0,
            price_now=200.0,
            price_ref=200.0,
            ad_now=0.0,
            organic_7=30.0,
            sessions_7=30.0,
            cvr_mean=0.02,
            cvr_p90=0.03,
            status="on_track",
            ratio=1.1,
            required_rate=2.0,
        )
        sizes = {"S09:noon": {"on_hand": [50.0, 40.0, 30.0], "demand_share": [0.7, 0.2, 0.1], "sizes": ["41", "42", "43"]}}
        w = _Work(table=table(r), rows={"S09:noon": r}, sizes=sizes, cfg=cfg, days_left=60, plan_date=date(2026, 10, 5))
        w.diag = {"S09:noon": {"kind": "on_track", "why": ""}}
        by_size, units, why = _size_exit(w, "S09:noon", 200.0, 0.0)
        optimistic = project(
            r, 200.0, 0.0, None, 60, cfg, sizes["S09:noon"], cvr=r["cvr_p90"], elasticity=r["elasticity"] + r["elasticity_sd"]
        )
        expected = [int(np.floor(u)) for u in optimistic.unsold_by_size]
        self.assertEqual(by_size, expected)  # exactly the optimistic leftover, floored size by size
        self.assertEqual(units, sum(expected))
        mean_read = project(r, 200.0, 0.0, None, 60, cfg, sizes["S09:noon"])
        self.assertLess(units, sum(int(np.floor(u)) for u in mean_read.unsold_by_size))  # the mean read would exit more
        self.assertIn("optimistic read", why)


class TestVerdict(unittest.TestCase):
    def test_make_it_follows_the_sell_through_target(self):
        ds = load_real()
        cfg = Config()
        cfg.sell_through_target = 0.5  # the plan clears 50%: "yes"
        v = build_plan(ds, State(), cfg).verdict
        self.assertTrue(v["make_it"])
        self.assertTrue(verdict_text(v).startswith("Yes"))
        cfg.sell_through_target = 0.99
        v = build_plan(ds, State(), cfg).verdict
        self.assertFalse(v["make_it"])
        self.assertTrue(verdict_text(v).startswith("Not all of it"))
        self.assertIn(f"{v['plan']['net']:,.0f}", verdict_text(v))


class TestLoadEffects(unittest.TestCase):
    def test_duplicate_rows_are_dropped_and_the_last_one_wins(self):
        h = pd.read_csv(DATA / "daily_history.csv")
        dup = h.iloc[[0]].copy()
        dup["units_sold"] = 99
        cleaned, issues, _ = clean_history(pd.concat([h, dup]))
        self.assertEqual(len(cleaned), len(h))
        key = (
            (cleaned.style_id == h.iloc[0].style_id)
            & (cleaned.channel == h.iloc[0].channel)
            & (cleaned.date == pd.Timestamp(h.iloc[0].date).date())
        )
        self.assertEqual(int(cleaned.loc[key, "units_sold"].iloc[0]), 99)
        self.assertTrue(any("duplicate" in i for i in issues))

    def test_re_uploading_a_day_replaces_it(self):
        ds = load_real()
        again = ds.history[ds.history.date == ds.today].drop(columns=["sessions_valid"]).copy()
        again["units_sold"] = 0
        ds2, _ = merge_new_history(ds, again)
        self.assertEqual(ds2.today, ds.today)
        self.assertEqual(int(ds2.history.loc[ds2.history.date == ds.today, "units_sold"].sum()), 0)
        self.assertEqual(len(ds2.history), len(ds.history))

    def test_zero_sessions_with_orders_is_a_reporting_gap_even_without_ad_clicks(self):
        h = pd.read_csv(DATA / "daily_history.csv")
        sel = (h.style_id == "S05") & (h.channel == AMAZON) & (h.date == "2026-09-28")
        h.loc[sel, ["sessions", "ad_clicks"]] = 0
        h.loc[sel, "units_sold"] = 2  # two orders from nobody: impossible, so sessions are missing, orders stay
        cleaned, _, _ = clean_history(h)
        r = cleaned[(cleaned.style_id == "S05") & (cleaned.channel == AMAZON) & (cleaned.date == date(2026, 9, 28))].iloc[0]
        self.assertFalse(bool(r["sessions_valid"]))
        self.assertEqual(int(r["units_sold"]), 2)
        # a genuinely quiet day (no sessions, no clicks, no orders) is valid data
        h.loc[sel, "units_sold"] = 0
        cleaned, _, _ = clean_history(h)
        r = cleaned[(cleaned.style_id == "S05") & (cleaned.channel == AMAZON) & (cleaned.date == date(2026, 9, 28))].iloc[0]
        self.assertTrue(bool(r["sessions_valid"]))

    def test_channel_names_from_exports_are_recognised(self):
        raw = pd.Series(["Amazon.ae", "amazon", "AMAZON_AE", "Noon", "noon.com"])
        self.assertEqual(normalise_channel(raw).tolist(), [AMAZON, AMAZON, AMAZON, NOON, NOON])


class TestLockAndBrief(unittest.TestCase):
    def test_a_listing_changed_today_cannot_be_moved_again_whatever_the_economics(self):
        from engine.apply import apply_action

        ds = load_real()
        cfg = Config()
        cfg.cooldown_days = 0  # take the cooldown out of the way: only the same-day lock should hold the line
        cfg.price_min_gain_aed = 0.0
        plan = build_plan(ds, State(), cfg)
        t = plan.table.set_index("listing_id")
        st = State()
        # a critical style cut on both channels today: by the projection a deeper cut would pay even more
        a = next(
            x
            for x in plan.changes
            if x.price_to < x.price_from and not x.forced_by_parity and t.loc[x.listing_id, "status"] == "critical"
        )
        for x in plan.actions:
            if x.style_id == a.style_id and x.price_to != x.price_from:
                apply_action(ds, st, x, plan.plan_date, plan.sizes[x.listing_id]["sizes"])
        again = build_plan(ds, st, cfg)
        t2 = again.table.set_index("listing_id")
        for x in again.actions:
            if x.style_id == a.style_id:
                self.assertTrue(bool(t2.loc[x.listing_id, "locked"]))
                self.assertEqual(x.price_to, x.price_from, x.headline())
                self.assertEqual(int(t2.loc[x.listing_id, "slots"]), cfg.max_price_changes - 1)  # one slot left, but locked

    def test_the_same_day_lock_blocks_price_candidates_even_with_a_slot_left(self):
        from engine.decide import _can_move_price, _cut_candidates

        self.assertTrue(_can_move_price(row(slots=1, locked=False)))
        self.assertFalse(_can_move_price(row(slots=2, locked=True)))
        self.assertFalse(_can_move_price(row(slots=0, locked=False)))
        self.assertEqual(_cut_candidates(row(price_now=200.0, floor=150.0, locked=True), Config()), [200.0])

    def test_a_percentage_the_model_was_given_as_a_fraction_is_not_flagged(self):
        from engine import llm

        payload = {"verdict": {"sell_through_with_plan": 0.72, "net_recovery_plan_aed": 394757}}
        self.assertEqual(llm._check_numbers("Sell-through 72% and AED 394,757.", payload), [])
        self.assertTrue(llm._check_numbers("Sell-through 73%.", payload))


class TestApplyBySize(unittest.TestCase):
    def test_exits_by_size_remove_exactly_those_sizes(self):
        from engine.apply import apply_action

        ds = load_real()
        inv = ds.inventory
        mask = (inv.style_id == "S04") & (inv.channel == AMAZON)
        sizes = [str(s) for s in inv.loc[mask, "size_eu"]]
        before = inv.loc[mask, "units_on_hand"].astype(int).tolist()
        by_size = [0] * len(sizes)
        by_size[0], by_size[-1] = min(3, before[0]), min(2, before[-1])
        a = action("S04:amazon_ae", kind="exit", exit_units=sum(by_size), exit_by_size=by_size, exit_value=100.0)
        apply_action(ds, State(), a, ds.today + timedelta(days=1), sizes)
        after = ds.inventory.loc[mask, "units_on_hand"].astype(int).tolist()
        self.assertEqual(after, [b - e for b, e in zip(before, by_size, strict=True)])


class TestGuardrailWithdrawal(unittest.TestCase):
    def test_a_parity_breach_withdraws_both_channels_of_the_style(self):
        from engine.decide import _apply_guardrail, _Work

        ds = load_real()
        cfg = Config()
        plan = build_plan(ds, State(), cfg)
        w = _Work(
            table=plan.table,
            rows={r["listing_id"]: r for _, r in plan.table.iterrows()},
            sizes=plan.sizes,
            cfg=cfg,
            days_left=plan.days_left,
            plan_date=plan.plan_date,
        )
        a = next(x for x in plan.changes if x.price_to != x.price_from and x.channel == NOON and not x.must_do)
        a.price_to = round(a.price_from * 0.7)  # 30% off on one channel only
        violations, _ = _apply_guardrail(w, plan.actions)
        self.assertEqual(violations, [])
        for x in plan.actions:
            if x.style_id == a.style_id:
                self.assertEqual(x.price_to, x.price_from, x.headline())
                self.assertTrue(any("withdrawn" in f for f in x.flags))
        self.assertTrue(any("guardrail withdrew" in n for n in w.notes))


class TestAwkwardInputs(unittest.TestCase):
    """The scenarios from scenarios/edge_cases.py that only printed before, now asserted."""

    def setUp(self):
        self.inv, self.h, self.c = (pd.read_csv(DATA / f) for f in ("inventory.csv", "daily_history.csv", "constraints.csv"))

    def test_a_style_missing_from_constraints_gets_no_price_move_and_a_note(self):
        c = self.c[self.c.style_id != "S07"]
        ds = build_dataset(self.inv, self.h, c)
        plan = build_plan(ds, State(), Config())
        self.assertEqual(plan.violations, [])
        self.assertTrue(all(a.price_to == a.price_from for a in plan.actions if a.style_id == "S07"))
        self.assertTrue(any("S07 has no row in constraints" in n for n in plan.notes))

    def test_an_unknown_style_in_the_history_is_dropped_with_a_note(self):
        extra = self.h[self.h.style_id == "S01"].copy()
        extra["style_id"] = "S99"
        ds = build_dataset(self.inv, pd.concat([self.h, extra]), self.c)
        self.assertNotIn("S99", set(ds.history.style_id))
        self.assertTrue(any("S99" in i and "no inventory rows" in i for i in ds.issues))
        self.assertEqual(build_plan(ds, State(), Config()).violations, [])

    def test_a_messy_two_day_upload_with_export_names_merges(self):
        ds = load_real()
        d1, d2 = ds.today + timedelta(days=1), ds.today + timedelta(days=2)
        last = ds.history[ds.history.date == ds.today].drop(columns=["sessions_valid"])
        nd = pd.concat([last.assign(date=d1), last.assign(date=d2)])
        nd = nd[nd.style_id != "S11"]  # one style missing from the export
        nd = nd.rename(columns={"channel": "Marketplace", "units_sold": "Units", "price_aed": "Price (AED)", "style_id": "Style"})
        nd["Marketplace"] = nd["Marketplace"].map({AMAZON: "Amazon.ae", NOON: "Noon"})
        ds2, issues = merge_new_history(ds, nd)
        self.assertEqual(ds2.today, d2)
        self.assertTrue(any("S11" in i for i in issues + ds2.issues))
        plan = build_plan(ds2, State(), Config())
        self.assertEqual(plan.violations, [])
        self.assertEqual(int(ds2.inventory.units_on_hand.sum()), int(ds.inventory.units_on_hand.sum()) - int(nd["Units"].sum()))


if __name__ == "__main__":
    unittest.main()
