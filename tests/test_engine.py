"""Run with:  python -m pytest -q     (or)   python -m unittest -v

The guardrail tests are the ones that matter: a plan that breaks a business rule is a
failure however well it is explained."""

import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd

from engine.actions import Action
from engine.apply import apply_action
from engine.config import Config
from engine.decide import build_plan
from engine.guardrail import check_plan
from engine.load import identify_kind, load_from_paths, merge_new_history, normalise_columns
from engine.state import State

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_real():
    return load_from_paths(DATA / "inventory.csv", DATA / "daily_history.csv", DATA / "constraints.csv")


class TestLoad(unittest.TestCase):
    def test_real_data_loads_and_flags_noon_outage(self):
        ds = load_real()
        self.assertEqual(ds.history.groupby(["style_id", "channel"]).ngroups, 40)
        self.assertEqual(int((~ds.history["sessions_valid"]).sum()), 40)  # 20 listings x 2 days
        self.assertTrue(any("sessions below ad clicks" in i for i in ds.issues))
        # orders on those days are kept, only sessions are distrusted
        bad = ds.history[~ds.history["sessions_valid"]]
        self.assertGreater(bad["units_sold"].sum(), 0)

    def test_column_synonyms(self):
        df = pd.DataFrame(
            {
                "Date": ["2026-10-05"],
                "Style": ["s01"],
                "Marketplace": ["Amazon.ae"],
                "Price": [279],
                "Views": [100],
                "Spend": [50],
                "Clicks": [20],
                "Orders": [2],
            }
        )
        self.assertEqual(identify_kind(df), "history")
        cols = set(normalise_columns(df).columns)
        self.assertTrue(
            {"date", "style_id", "channel", "price_aed", "sessions", "ad_spend_aed", "ad_clicks", "units_sold"} <= cols
        )

    def test_messy_export_loads(self):
        """Blank ad cells, a duplicated row and a missing day: cleaned and reported, not fatal."""
        from engine.load import build_dataset

        h = pd.read_csv(DATA / "daily_history.csv")
        h.loc[h.index[:5], ["ad_spend_aed", "ad_clicks"]] = None
        h = pd.concat([h, h.iloc[[10]]])
        h = h[~((h.style_id == "S03") & (h.channel == "noon") & (h.date == "2026-09-27"))]
        ds = build_dataset(pd.read_csv(DATA / "inventory.csv"), h, pd.read_csv(DATA / "constraints.csv"))
        self.assertTrue(any("blank ad" in i for i in ds.issues))
        self.assertTrue(any("duplicate" in i for i in ds.issues))
        self.assertTrue(any("missing from history" in i for i in ds.issues))

    def test_new_day_rolls_stock_forward(self):
        ds = load_real()
        before = int(ds.inventory["units_on_hand"].sum())
        nxt = ds.history[ds.history["date"] == ds.today].copy()
        nxt["date"] = ds.today + timedelta(days=1)
        nxt["units_sold"] = 1
        ds2, issues = merge_new_history(ds, nxt.drop(columns=["sessions_valid"]))
        self.assertEqual(ds2.today, ds.today + timedelta(days=1))
        self.assertEqual(int(ds2.inventory["units_on_hand"].sum()), before - 40)


class TestGuardrail(unittest.TestCase):
    def setUp(self):
        self.ds = load_real()
        self.cfg = Config()
        self.plan = build_plan(self.ds, State(), self.cfg)
        self.table = self.plan.table

    def _row(self, lid):
        return self.table.set_index("listing_id").loc[lid]

    def test_plan_has_no_violations(self):
        self.assertEqual(self.plan.violations, [])

    def test_floor_is_caught(self):
        r = self._row("S01:amazon_ae")
        a = Action(
            "S01:amazon_ae",
            "S01",
            "amazon_ae",
            "x",
            "price",
            price_from=r["price_now"],
            price_to=r["floor"] - 5,
            ads_from=r["ad_now"],
            ads_to=r["ad_now"],
        )
        v = check_plan([a], self.table, self.cfg)
        self.assertTrue(any(x.rule == "floor" for x in v))

    def test_parity_is_caught_and_counts_both_channels(self):
        r = self._row("S01:amazon_ae")
        a = Action(
            "S01:amazon_ae",
            "S01",
            "amazon_ae",
            "x",
            "price",
            price_from=r["price_now"],
            price_to=r["price_now"] * 0.85,
            ads_from=r["ad_now"],
            ads_to=r["ad_now"],
        )
        v = check_plan([a], self.table, self.cfg)
        self.assertTrue(any(x.rule == "parity" for x in v))
        # moving Noon along inside the band clears it
        rn = self._row("S01:noon")
        b = Action(
            "S01:noon",
            "S01",
            "noon",
            "x",
            "price",
            price_from=rn["price_now"],
            price_to=r["price_now"] * 0.85,
            ads_from=rn["ad_now"],
            ads_to=rn["ad_now"],
        )
        self.assertFalse(any(x.rule == "parity" for x in check_plan([a, b], self.table, self.cfg)))

    def test_budget_is_caught(self):
        r = self._row("S01:amazon_ae")
        a = Action(
            "S01:amazon_ae",
            "S01",
            "amazon_ae",
            "x",
            "ads",
            price_from=r["price_now"],
            price_to=r["price_now"],
            ads_from=r["ad_now"],
            ads_to=r["ad_now"] + 1000,
        )
        self.assertTrue(any(x.rule == "budget" for x in check_plan([a], self.table, self.cfg)))

    def test_change_limit_rolling_window(self):
        """Two applied changes inside the window lock the listing; a change 7 days earlier does not."""
        st = State()
        d = self.plan.plan_date
        st.apply_price("S05:amazon_ae", 159, 154, d - timedelta(days=3))
        st.apply_price("S05:amazon_ae", 154, 149, d - timedelta(days=1))
        st.apply_price("S05:noon", 159, 149, d - timedelta(days=7))  # outside the 7-day window on plan date
        t = build_plan(self.ds, st, self.cfg).table.set_index("listing_id")
        self.assertEqual(int(t.loc["S05:amazon_ae", "slots"]), 0)
        self.assertEqual(int(t.loc["S05:noon", "slots"]), 2)
        a = Action("S05:amazon_ae", "S05", "amazon_ae", "x", "price", price_from=159, price_to=150, ads_from=0, ads_to=0)
        self.assertTrue(any(x.rule == "change_limit" for x in check_plan([a], t.reset_index(), self.cfg)))

    def test_parity_forced_change_counts(self):
        st = State()
        st.apply_price("S05:noon", 159, 149, self.plan.plan_date - timedelta(days=1), forced=True)
        t = build_plan(self.ds, st, self.cfg).table.set_index("listing_id")
        self.assertEqual(int(t.loc["S05:noon", "slots"]), 1)


class TestDecisions(unittest.TestCase):
    def setUp(self):
        self.ds = load_real()
        self.cfg = Config()
        self.plan = build_plan(self.ds, State(), self.cfg)
        self.by = {a.listing_id: a for a in self.plan.actions}

    def test_budgets_respected_and_parity_holds(self):
        for ch, total in self.plan.ad_totals["plan"].items():
            self.assertLessEqual(total, self.cfg.ad_budget[ch] + 1e-6)
        prices = {a.listing_id: a.price_to for a in self.plan.actions}
        for s in {a.style_id for a in self.plan.actions}:
            pa, pn = prices.get(f"{s}:amazon_ae"), prices.get(f"{s}:noon")
            if pa and pn:
                self.assertLessEqual(abs(pn / pa - 1), self.cfg.parity_band + 1e-9)
        t = self.plan.table.set_index("listing_id")
        for a in self.plan.actions:
            self.assertGreaterEqual(a.price_to, t.loc[a.listing_id, "floor"] - 1e-9)

    def test_broken_listing_gets_ads_stopped_and_a_check_before_any_exit(self):
        a = self.by["S19:amazon_ae"]
        self.assertEqual(a.ads_to, 0.0)
        self.assertTrue(a.check)
        self.assertEqual(a.exit_units, 0)  # irreversible; a fixed listing recovers far more than the jobber
        self.assertIn("exit held", a.reason)
        self.assertTrue(any("human check" in f for f in a.flags))
        # the check was requested 3 days ago and the listing is still dead: now the pairs go
        st = State()
        st.record_check("S19:amazon_ae", self.plan.plan_date - timedelta(days=3))
        later = {x.listing_id: x for x in build_plan(self.ds, st, self.cfg).actions}["S19:amazon_ae"]
        self.assertFalse(later.check)
        self.assertGreater(later.exit_units, 50)
        # one day after the check: still held, and the check is not asked for twice
        st2 = State()
        st2.record_check("S19:amazon_ae", self.plan.plan_date - timedelta(days=1))
        soon = {x.listing_id: x for x in build_plan(self.ds, st2, self.cfg).actions}["S19:amazon_ae"]
        self.assertFalse(soon.check)
        self.assertEqual(soon.exit_units, 0)

    def test_starved_converter_gets_ads(self):
        a = self.by["S14:noon"]
        self.assertGreater(a.ads_to, a.ads_from)

    def test_donor_is_trimmed(self):
        a = self.by["S01:amazon_ae"]
        self.assertLess(a.ads_to, a.ads_from)

    def test_listing_at_floor_is_never_cut(self):
        a = self.by["S06:amazon_ae"]
        self.assertEqual(a.price_to, a.price_from)

    def test_impact_of_changes_is_positive_at_style_level(self):
        for s in {a.style_id for a in self.plan.changes}:
            total = sum(a.impact_aed for a in self.plan.actions if a.style_id == s)
            self.assertGreater(total, 0)

    def test_apply_then_replan_does_not_repeat(self):
        st = State()
        a = max(self.plan.changes, key=lambda x: x.exit_units)  # the biggest exit of the day (S04 on day 1)
        self.assertGreater(a.exit_units, 0)
        apply_action(self.ds, st, a, self.plan.plan_date, self.plan.sizes[a.listing_id]["sizes"])
        plan2 = build_plan(self.ds, st, self.cfg)
        a2 = {x.listing_id: x for x in plan2.actions}[a.listing_id]
        self.assertEqual(a2.ads_from, a.ads_to)  # the applied spend is now the baseline
        self.assertEqual(a2.exit_units, 0)  # units already gone are not exited again
        self.assertFalse(a2.is_change)  # one decision per listing per day
        # a check applied today is not asked for again
        c = self.by["S19:amazon_ae"]
        apply_action(self.ds, st, c, self.plan.plan_date, self.plan.sizes[c.listing_id]["sizes"])
        c2 = {x.listing_id: x for x in build_plan(self.ds, st, self.cfg).actions}["S19:amazon_ae"]
        self.assertFalse(c2.check)
        self.assertFalse(c2.is_change)


class TestScenarios(unittest.TestCase):
    """The live-session 'change some conditions' drills: each must stay legal and move the right way."""

    def setUp(self):
        self.ds = load_real()
        self.base = build_plan(self.ds, State(), Config())

    def test_half_budget(self):
        cfg = Config()
        cfg.ad_budget = {"amazon_ae": 750.0, "noon": 500.0}
        p = build_plan(self.ds, State(), cfg)
        self.assertEqual(p.violations, [])
        for ch, total in p.ad_totals["plan"].items():
            self.assertLessEqual(total, cfg.ad_budget[ch] + 1e-6)

    def test_shorter_deadline_means_more_urgency(self):
        cfg = Config()
        cfg.horizon_days = 30
        p = build_plan(self.ds, State(), cfg)
        self.assertEqual(p.violations, [])
        self.assertGreater(p.verdict["n_critical"], self.base.verdict["n_critical"])
        cuts = lambda plan: sum(1 for a in plan.changes if a.price_to is not None and a.price_to < a.price_from)  # noqa: E731
        self.assertGreater(cuts(p), cuts(self.base))
        # exits are irreversible and cheap to delay: a shorter deadline brings cuts forward, not a fire sale
        self.assertLessEqual(sum(a.exit_units for a in p.actions), self.base.verdict["on_hand"] * 0.1)

    def test_richer_jobber_means_more_exits(self):
        cfg = Config()
        cfg.jobber_pct = 0.45
        p = build_plan(self.ds, State(), cfg)
        self.assertEqual(p.violations, [])
        self.assertGreaterEqual(sum(a.exit_units for a in p.actions), sum(a.exit_units for a in self.base.actions))

    def test_fees_reduce_ad_spend(self):
        cfg = Config()
        cfg.marketplace_fee_pct = 0.15
        p = build_plan(self.ds, State(), cfg)
        self.assertEqual(p.violations, [])
        self.assertLessEqual(sum(p.ad_totals["plan"].values()), sum(self.base.ad_totals["plan"].values()) + 1e-6)


class TestEdgeCases(unittest.TestCase):
    """Things the live session may throw at the tool: each must plan without breaking a rule."""

    def setUp(self):
        self.inv = pd.read_csv(DATA / "inventory.csv")
        self.h = pd.read_csv(DATA / "daily_history.csv")
        self.c = pd.read_csv(DATA / "constraints.csv")

    def _plan(self, inv=None, h=None, c=None, cfg=None, state=None):
        from engine.load import build_dataset

        ds = build_dataset(
            (self.inv if inv is None else inv).copy(), (self.h if h is None else h).copy(), (self.c if c is None else c).copy()
        )
        return build_plan(ds, state or State(), cfg or Config())

    def test_new_listing_without_history(self):
        new = self.inv[(self.inv.style_id == "S05") & (self.inv.channel == "noon")].copy()
        new["style_id"] = "S21"
        c = pd.concat(
            [
                self.c,
                pd.DataFrame(
                    [{"style_id": "S21", "style_name": "x", "category": "men", "mrp_aed": 249, "liquidation_floor_aed": 90}]
                ),
            ]
        )
        p = self._plan(inv=pd.concat([self.inv, new]), c=c)
        self.assertEqual(p.violations, [])
        a = {x.listing_id: x for x in p.actions}["S21:noon"]
        self.assertEqual(a.price_to, a.price_from)  # no price move on a listing with no history

    def test_floor_raised_overnight_is_fixed_first(self):
        c = self.c.copy()
        c.loc[c.style_id == "S05", "liquidation_floor_aed"] = 170
        p = self._plan(c=c)
        self.assertEqual(p.violations, [])
        top = p.changes[:2]
        self.assertTrue(all(a.must_do and a.price_to == 170 for a in top))

    def test_parity_broken_in_data_is_fixed(self):
        h = self.h.copy()
        h.loc[(h.style_id == "S12") & (h.channel == "noon") & (h.date == "2026-10-04"), "price_aed"] = 199
        p = self._plan(h=h)
        self.assertEqual(p.violations, [])
        self.assertEqual(p.changes[0].listing_id, "S12:noon")

    def test_unfixable_rule_break_is_a_conflict_not_a_violation(self):
        from engine.load import build_dataset

        h = self.h.copy()
        h.loc[(h.style_id == "S12") & (h.channel == "noon") & (h.date == "2026-10-04"), "price_aed"] = 199
        ds = build_dataset(self.inv.copy(), h, self.c.copy())
        st, d = State(), ds.today + timedelta(days=1)
        for ch in ("amazon_ae", "noon"):
            st.apply_price(f"S12:{ch}", 219, 218, d - timedelta(days=2))
            st.apply_price(f"S12:{ch}", 218, 219, d - timedelta(days=1))
        p = build_plan(ds, st, Config())
        self.assertEqual(p.violations, [])
        self.assertTrue(any(v.listing_id == "S12:noon" for v in p.conflicts))

    def test_zero_budget_channel(self):
        cfg = Config()
        cfg.ad_budget = {"amazon_ae": 1500.0, "noon": 0.0}
        p = self._plan(cfg=cfg)
        self.assertEqual(p.violations, [])
        self.assertLessEqual(p.ad_totals["plan"]["noon"], 1e-6)

    def test_tighter_rules(self):
        cfg = Config()
        cfg.parity_band = 0.03
        cfg.max_price_changes = 1
        self.assertEqual(self._plan(cfg=cfg).violations, [])


class TestLLM(unittest.TestCase):
    """The real call is mocked here (no key in CI); this checks the request, parsing, the
    numbers check and the fallback, which is everything except the network."""

    def setUp(self):
        # never read the developer's own .env in tests: results must not depend on whose machine runs them
        from unittest import mock

        from engine.settings import Settings

        patcher = mock.patch.dict(Settings.model_config, {"env_file": None})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ds = load_real()
        self.cfg = Config()
        self.plan = build_plan(self.ds, State(), self.cfg)

    def _env(self, **extra):
        """Environment for one test: a key, and nothing from the developer's own shell."""
        import os
        from unittest import mock

        env = {"ANTHROPIC_API_KEY": "k", "LLM_MODEL": "", "LLM_EFFORT": "", "LLM_TIMEOUT_S": "120", **extra}
        return mock.patch.dict(os.environ, env)

    def _client(self, reply: str = "", stop_reason: str = "end_turn", error: Exception | None = None):
        """A stand-in for anthropic.Anthropic that records what it was built with and asked, and answers
        with `reply` (or raises `error`). Returns (class to patch in, the record)."""
        from types import SimpleNamespace

        seen: dict = {}

        class FakeMessages:
            def create(self, **kw):
                seen["request"] = kw
                if error is not None:
                    raise error
                blocks = [SimpleNamespace(type="thinking", thinking="..."), SimpleNamespace(type="text", text=reply)]
                return SimpleNamespace(content=blocks, stop_reason=stop_reason)

        class FakeAnthropic:
            def __init__(self, **kw):
                seen["client"] = kw
                self.messages = FakeMessages()

        return FakeAnthropic, seen

    @staticmethod
    def _sdk_error(cls, status: int | None = None, message: str = "boom"):
        """An SDK exception of the given class, built without an HTTP response (constructors differ between SDK
        versions; the code only reads the class, .status_code and .message)."""
        e = cls.__new__(cls)
        Exception.__init__(e, message)
        e.message, e.status_code, e.body = message, status, None
        return e

    def test_the_sdk_call_parses_and_checks_numbers(self):
        import json
        from unittest import mock

        import engine.llm as llm

        v = self.plan.verdict
        brief = f"Not all of it: about {v['plan']['sold']:,.0f} of {v['on_hand']:,} pairs sell, and 999,999 is invented."
        reply = json.dumps(
            {"brief": brief, "flags": [{"listing_id": "S19:amazon_ae", "note": "check images"}], "questions": ["q"]}
        )
        fake, seen = self._client(reply)
        with self._env(), mock.patch("anthropic.Anthropic", fake):
            out = llm.generate_brief(self.plan, self.cfg)
        self.assertIsNone(out["error"])
        self.assertEqual(out["provider"], "anthropic:claude-sonnet-5-5 (effort medium)")
        req = seen["request"]
        self.assertEqual(req["model"], "claude-sonnet-5-5")
        self.assertEqual(req["system"], llm.SYSTEM)
        self.assertTrue(req["messages"][0]["content"].startswith("PLAN AS COMPUTED"))
        self.assertEqual(req["output_config"]["effort"], "medium")
        self.assertEqual(req["output_config"]["format"], {"type": "json_schema", "schema": llm.BRIEF_SCHEMA})
        self.assertNotIn("temperature", req)  # API version 2026-10-21 rejects sampling parameters
        self.assertEqual(seen["client"], {"api_key": "k", "timeout": 120.0, "max_retries": llm.MAX_RETRIES})
        self.assertEqual(out["flags"], [{"listing_id": "S19:amazon_ae", "note": "check images"}])
        self.assertIn("999999", out["unverified_numbers"])
        self.assertNotIn(f"{v['on_hand']}", out["unverified_numbers"])

    def test_model_effort_and_timeout_come_from_settings(self):
        import json
        from unittest import mock

        import engine.llm as llm

        fake, seen = self._client(json.dumps({"brief": "Not all of it.", "flags": [], "questions": []}))
        with (
            self._env(LLM_MODEL="claude-haiku-5-5", LLM_EFFORT="low", LLM_TIMEOUT_S="45"),
            mock.patch("anthropic.Anthropic", fake),
        ):
            out = llm.generate_brief(self.plan, self.cfg)
        self.assertIsNone(out["error"])
        self.assertEqual(seen["request"]["model"], "claude-haiku-5-5")
        self.assertEqual(seen["request"]["output_config"]["effort"], "low")
        self.assertEqual(seen["client"]["timeout"], 45.0)

    def test_every_api_error_falls_back_with_a_sentence_that_says_what_to_do(self):
        from unittest import mock

        import anthropic

        import engine.llm as llm

        cases = [
            (self._sdk_error(anthropic.APITimeoutError), "didn't answer within 120 s"),
            (self._sdk_error(anthropic.APIConnectionError), "check the network"),
            (self._sdk_error(anthropic.AuthenticationError, 401), "ANTHROPIC_API_KEY"),
            (self._sdk_error(anthropic.PermissionDeniedError, 403), "isn't allowed"),
            (self._sdk_error(anthropic.NotFoundError, 404), "LLM_MODEL"),
            (self._sdk_error(anthropic.RateLimitError, 429), "wait a minute"),
            (self._sdk_error(anthropic.APIStatusError, 529, "Overloaded"), "HTTP 529: Overloaded"),
            (self._sdk_error(anthropic.BadRequestError, 400, "Your credit balance is too low"), "credit balance"),
        ]
        for err, says in cases:
            with self.subTest(error=type(err).__name__):
                fake, _ = self._client(error=err)
                with self._env(), mock.patch("anthropic.Anthropic", fake):
                    out = llm.generate_brief(self.plan, self.cfg)
                self.assertEqual(out["provider"], "template")
                self.assertIn(says, out["error"])
                self.assertTrue(out["brief"])  # the page still has a verdict to show

    def test_a_truncated_or_refused_reply_falls_back(self):
        from unittest import mock

        import engine.llm as llm

        for stop, says in (("max_tokens", "token limit"), ("refusal", "declined")):
            with self.subTest(stop_reason=stop):
                fake, _ = self._client('{"brief": "Not all', stop_reason=stop)
                with self._env(), mock.patch("anthropic.Anthropic", fake):
                    out = llm.generate_brief(self.plan, self.cfg)
                self.assertEqual(out["provider"], "template")
                self.assertIn(says, out["error"])

    def test_no_key_means_the_engine_verdict_and_no_call(self):
        from unittest import mock

        import engine.llm as llm

        fake, seen = self._client("{}")
        with self._env(ANTHROPIC_API_KEY=""), mock.patch("anthropic.Anthropic", fake):
            out = llm.generate_brief(self.plan, self.cfg)
        self.assertEqual(seen, {})  # nothing was sent
        self.assertEqual(out["provider"], "template")
        self.assertIn("ANTHROPIC_API_KEY", out["error"])
        self.assertTrue(out["brief"].startswith("Not all of it") or out["brief"].startswith("Yes"))

    def test_an_unexpected_failure_still_falls_back(self):
        from unittest import mock

        import engine.llm as llm

        fake, _ = self._client(error=OSError("network down"))
        with self._env(), mock.patch("anthropic.Anthropic", fake):
            out = llm.generate_brief(self.plan, self.cfg)
        self.assertEqual(out["provider"], "template")
        self.assertIn("network down", out["error"])


class TestBoundaries(unittest.TestCase):
    """Data that comes from outside the code (sidebar, state.json, the LLM) is checked on the way in."""

    def test_config_rejects_out_of_range_values(self):
        from pydantic import ValidationError

        cfg = Config()
        with self.assertRaises(ValidationError):
            cfg.jobber_pct = 1.5
        with self.assertRaises(ValidationError):
            cfg.ad_budget = {"amazon_ae": -10.0, "noon": 1000.0}
        self.assertEqual(cfg.jobber_pct, 0.25)  # a rejected assignment leaves the old value

    def test_bad_saved_override_is_dropped_good_ones_kept(self):
        cfg = Config.from_dict({"horizon_days": 30, "parity_band": -1, "not_a_setting": 5})
        self.assertEqual(cfg.horizon_days, 30)
        self.assertEqual(cfg.parity_band, 0.05)

    def test_state_round_trip(self):
        import tempfile

        st = State()
        d = load_real().today
        st.campaign_start = d
        st.apply_price("S01:amazon_ae", 279, 259, d)
        st.apply_exit("S19:noon", 10, 150.0, d)
        st.exits[-1]["by_size"] = [1, 2]
        st.record_prediction(d, "S01:amazon_ae", "price", 1.5, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            st.save(path)
            back = State.load(path)
        self.assertEqual(back.campaign_start, d)
        self.assertEqual(back.change_log[0]["to"], 259.0)
        self.assertEqual(back.exits[0]["by_size"], [1, 2])
        self.assertIsInstance(back.exits[0], dict)  # records stay plain dicts

    def test_damaged_state_file_fails_loudly(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text('{"exits": [{"listing_id": "S19:noon", "units": "lots"}]}')
            with self.assertRaises(ValueError) as cm:
                State.load(path)
        self.assertIn("can't be read", str(cm.exception))

    def test_malformed_llm_reply_falls_back_to_engine_verdict(self):
        import json
        import os
        from types import SimpleNamespace
        from unittest import mock

        import engine.llm as llm
        from engine.settings import Settings

        plan = build_plan(load_real(), State(), Config())
        reply = json.dumps({"flags": "not a list"})  # no brief, wrong type

        class FakeAnthropic:
            def __init__(self, **kw):
                self.messages = SimpleNamespace(
                    create=lambda **k: SimpleNamespace(content=[SimpleNamespace(type="text", text=reply)], stop_reason="end_turn")
                )

        with (
            mock.patch.dict(Settings.model_config, {"env_file": None}),
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}),
            mock.patch("anthropic.Anthropic", FakeAnthropic),
        ):
            out = llm.generate_brief(plan, Config())
        self.assertEqual(out["provider"], "template")
        self.assertIn("expected shape", out["error"])

    def test_the_schema_sent_to_the_model_comes_from_the_model_class(self):
        import engine.llm as llm

        schema = llm.BRIEF_SCHEMA
        self.assertEqual(set(schema["properties"]), set(llm.BriefOut.model_fields))
        self.assertEqual(set(schema["required"]), set(llm.BriefOut.model_fields))  # every field always present
        flag = schema["$defs"]["BriefFlag"]
        self.assertEqual(set(flag["properties"]), set(llm.BriefFlag.model_fields))

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False)
                self.assertFalse(set(node) & llm._NOT_ON_THE_WIRE, node)
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(schema)  # only keywords structured outputs accepts
        llm.BriefOut.model_validate({"brief": "x", "flags": [{"listing_id": "S01:noon", "note": "y"}], "questions": []})


class TestSettings(unittest.TestCase):
    def _settings(self, env: dict, env_file: str | None = None):
        import os
        from unittest import mock

        from engine.settings import Settings

        keys = ["ANTHROPIC_API_KEY", "LLM_MODEL", "LLM_EFFORT", "LLM_TIMEOUT_S", "LOG_LEVEL"]
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        with mock.patch.dict(os.environ, {**clean, **env}, clear=True):
            return Settings(_env_file=env_file)

    def test_blank_model_and_effort_mean_the_defaults(self):
        s = self._settings({"ANTHROPIC_API_KEY": "a", "LLM_MODEL": "", "LLM_EFFORT": ""})
        self.assertEqual(s.describe_llm(), "claude-sonnet-5-5 · effort medium")
        self.assertEqual(self._settings({}).describe_llm(), "none (no ANTHROPIC_API_KEY)")
        s = self._settings({"ANTHROPIC_API_KEY": "a", "LLM_MODEL": "claude-haiku-5-5", "LLM_EFFORT": "low"})
        self.assertEqual(s.describe_llm(), "claude-haiku-5-5 · effort low")

    def test_old_provider_keys_in_an_existing_env_are_ignored(self):
        s = self._settings({"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o", "GEMINI_API_KEY": "g", "LLM_PROVIDER": "gemini"})
        self.assertEqual(s.anthropic_api_key.get_secret_value(), "a")

    def test_typo_in_effort_fails_loudly(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self._settings({"LLM_EFFORT": "medim"})

    def test_unknown_log_level_fails_loudly(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self._settings({"LOG_LEVEL": "LOUD"})

    def test_key_never_appears_in_repr(self):
        s = self._settings({"ANTHROPIC_API_KEY": "sk-very-secret"})
        self.assertNotIn("sk-very-secret", repr(s))
        self.assertEqual(s.anthropic_api_key.get_secret_value(), "sk-very-secret")

    def test_env_file_is_read_and_environment_wins(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("ANTHROPIC_API_KEY=from-file\nLLM_MODEL=file-model\n")
        s = self._settings({"LLM_MODEL": "env-model"}, env_file=f.name)
        self.assertEqual(s.anthropic_api_key.get_secret_value(), "from-file")
        self.assertEqual(s.llm_model, "env-model")


class TestEconomics(unittest.TestCase):
    """The projection's closed form against an explicit day loop, on random cases."""

    def test_projection_matches_day_loop(self):
        import numpy as np

        from engine.economics import expected_rate, project

        cfg = Config()
        rng = np.random.default_rng(0)
        for _ in range(400):
            n = int(rng.integers(1, 8))
            stock = rng.integers(0, 30, n).astype(float)
            share = rng.random(n)
            row = {
                "on_hand": stock.sum(),
                "jobber_value": 20.0,
                "cvr_mean": float(rng.uniform(0.002, 0.05)),
                "elasticity": 2.0,
                "price_ref": 200.0,
                "cpc": 1.8,
                "ad_now": float(rng.uniform(0, 100)),
                "organic_7": float(rng.uniform(0, 150)),
            }
            price, ads, days = float(rng.uniform(120, 220)), float(rng.uniform(0, 150)), int(rng.integers(1, 61))
            exits = None if rng.random() < 0.5 else list(rng.integers(0, 5, n))
            sizes = {"on_hand": stock.tolist(), "demand_share": share.tolist()}
            got = project(row, price, ads, exits, days, cfg, sizes)
            # the loop
            s = stock - (np.minimum(np.array(exits, dtype=float), stock) if exits is not None else 0)
            rate = expected_rate(row, price, ads, cfg)
            full = share[s > 0].sum()
            sold = holding = adcost = 0.0
            for _d in range(days):
                if s.sum() <= 1e-9:
                    break
                avail = share[s > 0].sum()
                alloc = np.zeros(n)
                if avail > 0:
                    alloc[s > 0] = rate * (avail / full) * share[s > 0] / avail
                alloc = np.minimum(alloc, s)
                s = s - alloc
                sold += alloc.sum()
                holding += s.sum() * cfg.holding_per_unit_per_day
                adcost += ads
            self.assertAlmostEqual(got.sold, sold, places=6)
            self.assertAlmostEqual(got.unsold, s.sum(), places=6)
            net = sold * price - adcost - holding + (got.exit_units + s.sum()) * 20.0
            self.assertAlmostEqual(got.net, net, places=4)


class TestPlanEconomics(unittest.TestCase):
    """Every recommendation must pay by the plan's own projection."""

    def setUp(self):
        from engine.economics import project

        self.project = project
        self.ds = load_real()
        self.cfg = Config()
        self.plan = build_plan(self.ds, State(), self.cfg)
        self.t = self.plan.table.set_index("listing_id")

    def _net(self, lid, price, spend, exits=None):
        return self.project(self.t.loc[lid], price, spend, exits, self.plan.days_left, self.cfg, self.plan.sizes[lid]).net

    def test_no_ad_change_loses_money_at_the_plan_price(self):
        for a in self.plan.changes:
            if abs(a.ads_to - a.ads_from) < 1:
                continue
            gain = self._net(a.listing_id, a.price_to, a.ads_to) - self._net(a.listing_id, a.price_to, a.ads_from)
            if a.ads_to > a.ads_from:
                self.assertGreaterEqual(gain, self.cfg.ads_min_gain_aed - 1e-6, a.headline())
            else:
                self.assertGreaterEqual(gain, -1e-6, a.headline())  # stops and trims never cost

    def test_parity_follower_takes_the_shallowest_in_band_price(self):
        import math

        by = {a.listing_id: a for a in self.plan.actions}
        for a in self.plan.changes:
            if not a.forced_by_parity:
                continue
            other = next(b for b in self.plan.actions if b.style_id == a.style_id and b.listing_id != a.listing_id)
            band = self.cfg.parity_band
            if abs(a.price_from / other.price_to - 1) <= band + 1e-9:
                self.assertEqual(a.price_to, a.price_from, a.headline())
            else:
                want = (
                    math.floor(other.price_to * (1 + band))
                    if a.price_from > other.price_to
                    else math.ceil(other.price_to * (1 - band))
                )
                self.assertEqual(a.price_to, want, a.headline())
            self.assertGreaterEqual(by[a.listing_id].impact_aed + other.impact_aed, self.cfg.price_min_gain_aed * 2 - 1e-6)

    def test_every_action_beats_hold_or_is_a_parity_follower(self):
        for a in self.plan.changes:
            if a.forced_by_parity or a.must_do:
                continue
            self.assertGreaterEqual(a.impact_aed, 0, a.headline())

    def test_build_plan_does_not_touch_the_state(self):
        st = State()
        before = st.model_dump_json()
        build_plan(self.ds, st, self.cfg)
        self.assertEqual(st.model_dump_json(), before)


class TestSlotsAndWindows(unittest.TestCase):
    def test_history_price_change_counts_toward_slots(self):
        from engine.load import build_dataset

        inv, h, c = (pd.read_csv(DATA / f) for f in ("inventory.csv", "daily_history.csv", "constraints.csv"))
        h.loc[(h.style_id == "S05") & (h.channel == "amazon_ae") & (h.date >= "2026-10-03"), "price_aed"] = 149
        t = build_plan(build_dataset(inv, h, c), State(), Config()).table.set_index("listing_id")
        self.assertEqual(int(t.loc["S05:amazon_ae", "slots"]), 1)

    def test_applied_change_seen_in_export_a_day_later_counts_once(self):
        from engine.load import build_dataset

        inv, h, c = (pd.read_csv(DATA / f) for f in ("inventory.csv", "daily_history.csv", "constraints.csv"))
        h.loc[(h.style_id == "S05") & (h.channel == "amazon_ae") & (h.date == "2026-10-04"), "price_aed"] = 149
        st = State()
        st.apply_price("S05:amazon_ae", 159, 149, timedelta(days=0) + pd.Timestamp("2026-10-03").date())
        t = build_plan(build_dataset(inv, h, c), st, Config()).table.set_index("listing_id")
        self.assertEqual(int(t.loc["S05:amazon_ae", "slots"]), 1)

    def test_any_real_price_move_counts_but_rounding_noise_does_not(self):
        from engine.load import build_dataset

        inv, h, c = (pd.read_csv(DATA / f) for f in ("inventory.csv", "daily_history.csv", "constraints.csv"))
        h["price_aed"] = h["price_aed"].astype(float)
        sel = (h.style_id == "S05") & (h.channel == "amazon_ae") & (h.date == "2026-10-04")
        h.loc[sel, "price_aed"] = 159.2  # a fifth of a dirham: an export rounding wobble
        t = build_plan(build_dataset(inv, h.copy(), c), State(), Config()).table.set_index("listing_id")
        self.assertEqual(int(t.loc["S05:amazon_ae", "slots"]), 2)
        h.loc[sel, "price_aed"] = 160.0  # one dirham: a real reprice, under 1% but it used a slot
        t = build_plan(build_dataset(inv, h, c), State(), Config()).table.set_index("listing_id")
        self.assertEqual(int(t.loc["S05:amazon_ae", "slots"]), 1)

    def test_scoring_window_closes_at_the_listings_next_action(self):
        from engine.apply import score_predictions

        ds = load_real()
        st = State()
        d1, d2 = ds.today - timedelta(days=5), ds.today - timedelta(days=2)
        st.record_prediction(d1, "S01:amazon_ae", "ads", 2.0, 1.0)
        st.record_prediction(d2, "S01:amazon_ae", "price", 3.0, 1.0)
        score_predictions(ds, st)
        self.assertEqual(st.predictions[0]["days_observed"], 3)  # d1, d1+1, d1+2: stops before d2
        self.assertEqual(st.predictions[1]["days_observed"], 3)  # d2 .. today

    def test_calibration_weighs_uplift_over_baseline_and_shrinks_toward_one(self):
        from engine.estimate import lever_calibration

        cfg = Config()
        st = State()
        d = pd.Timestamp("2026-10-01").date()
        for _ in range(4):  # promised +1.0/day, delivered +0.5/day on top of the same baseline
            st.record_prediction(d, "S01:amazon_ae", "ads", 2.0, 1.0)
            st.predictions[-1].update(actual_rate=1.5, days_observed=5)
        raw = 0.5  # delivered / promised
        expected = 1 + (raw - 1) * 4 / (4 + cfg.calibration_shrink_actions)  # four actions are not much evidence
        self.assertAlmostEqual(lever_calibration(st, cfg)["ads"], expected, places=6)
        self.assertEqual(lever_calibration(st, cfg)["price"], 1.0)  # no price actions scored
        # with many actions the weight approaches the raw ratio (and stays within the bounds)
        for _ in range(96):
            st.record_prediction(d, "S01:amazon_ae", "ads", 2.0, 1.0)
            st.predictions[-1].update(actual_rate=1.5, days_observed=5)
        self.assertAlmostEqual(lever_calibration(st, cfg)["ads"], 1 + (raw - 1) * 100 / 108, places=6)
        # a combined price+ads action scores the price lever only: its uplift came from the cut
        st3 = State()
        for _ in range(4):
            st3.record_prediction(d, "S02:noon", "price+ads", 2.0, 1.0)
            st3.predictions[-1].update(actual_rate=1.0, days_observed=5)
        self.assertEqual(lever_calibration(st3, cfg)["ads"], 1.0)
        self.assertLess(lever_calibration(st3, cfg)["price"], 1.0)


class TestSession(unittest.TestCase):
    def test_guardrail_withdraws_a_budget_breach(self):
        from engine.decide import _apply_guardrail, _Work

        ds = load_real()
        cfg = Config()
        plan = build_plan(ds, State(), cfg)
        w = _Work(table=plan.table, rows={}, sizes=plan.sizes, cfg=cfg, days_left=plan.days_left, plan_date=plan.plan_date)
        actions = plan.actions
        for a in actions:  # force the channel over its cap
            if a.channel == "noon" and a.is_change:
                a.ads_to = a.ads_from + 400
        violations, _ = _apply_guardrail(w, actions)
        self.assertEqual(violations, [])
        self.assertLessEqual(sum(a.ads_to for a in actions if a.channel == "noon"), cfg.ad_budget["noon"] + 1e-6)

    def test_load_session_replays_exits(self):
        import tempfile

        from engine.session import load_session

        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "state.json"
            st = State()
            st.apply_exit("S19:amazon_ae", 100, 1500.0, load_real().today)
            st.save(sp)
            ds, st2 = load_session(DATA, sp)
        self.assertEqual(
            int(ds.inventory[(ds.inventory.style_id == "S19") & (ds.inventory.channel == "amazon_ae")].units_on_hand.sum()), 40
        )
        self.assertIsNotNone(st2.campaign_start)

    def test_future_dated_upload_is_refused(self):
        from engine.session import check_upload_dates

        ds = load_real()
        ok = ds.history[ds.history.date == ds.today].copy()
        ok["date"] = ds.today + timedelta(days=1)
        check_upload_dates(ok, ds.today)
        bad = ok.copy()
        bad["date"] = ds.today + timedelta(days=400)
        with self.assertRaises(ValueError):
            check_upload_dates(bad, ds.today)

    def test_invented_percentage_in_brief_is_flagged(self):
        import json
        import re

        import engine.llm as llm

        plan = build_plan(load_real(), State(), Config())
        payload = llm._payload(plan, Config())
        present = set(re.findall(r"\d+", json.dumps(payload, default=str)))
        invented = next(str(n) for n in range(99, 10, -1) if str(n) not in present)  # a percentage the plan never mentions
        self.assertIn(invented, llm._check_numbers(f"Roughly {invented}% of pairs will sell.", payload))
        self.assertEqual(llm._check_numbers("3 listings are critical.", payload), [])


class TestReviewFindings(unittest.TestCase):
    """Each test here is a bug an independent review found; the test is what stops it coming back."""

    def _stock(self, ds, style, channel):
        inv = ds.inventory
        return int(inv[(inv.style_id == style) & (inv.channel == channel)].units_on_hand.sum())

    def test_a_listing_with_no_usable_price_does_not_crash_the_plan(self):
        from engine.load import build_dataset

        inv, h, c = (pd.read_csv(DATA / f) for f in ("inventory.csv", "daily_history.csv", "constraints.csv"))
        h["price_aed"] = h["price_aed"].astype(object)
        h.loc[(h.style_id == "S17") & (h.channel == "noon"), "price_aed"] = None  # blank for every day of the export
        plan = build_plan(build_dataset(inv, h, c), State(), Config())
        self.assertEqual(plan.violations, [])
        s17 = [
            a
            for a in plan.changes
            if a.style_id == "S17" and a.price_to is not None and a.price_to == a.price_to and a.price_to != a.price_from
        ]
        self.assertEqual(s17, [])  # no price move for a style whose Noon price is unknown

    def test_restart_after_an_upload_does_not_take_exits_out_twice(self):
        import tempfile

        import engine.session as session
        from engine.apply import apply_action
        from engine.session import load_session, persist_session

        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "state.json"
            session.SESSION_DIR = Path(tmp) / ".session"  # keep the repo's own session out of the test
            try:
                ds, st = load_session(DATA, sp)
                plan = build_plan(ds, st, Config())
                a = max(plan.changes, key=lambda x: x.exit_units)
                before = self._stock(ds, a.style_id, a.channel)
                apply_action(ds, st, a, plan.plan_date, plan.sizes[a.listing_id]["sizes"])
                after_apply = self._stock(ds, a.style_id, a.channel)
                self.assertEqual(after_apply, before - a.exit_units)
                st.save(sp)
                persist_session(ds, st)  # what the page does after an upload
                ds2, st2 = load_session(DATA, sp)  # a browser refresh or a restart
                self.assertEqual(self._stock(ds2, a.style_id, a.channel), after_apply)
                self.assertEqual(sum(ds2.exited.values()), a.exit_units)
                self.assertFalse(any("inventory moved" in i and a.style_id in i for i in ds2.issues))
                # an exit applied after the last upload is still replayed
                plan2 = build_plan(ds2, st2, Config())
                nxt = [x for x in plan2.changes if x.exit_units > 0 and x.listing_id != a.listing_id]
                if nxt:
                    b = nxt[0]
                    apply_action(ds2, st2, b, plan2.plan_date, plan2.sizes[b.listing_id]["sizes"])
                    st2.save(sp)
                    ds3, _ = load_session(DATA, sp)
                    self.assertEqual(self._stock(ds3, b.style_id, b.channel), self._stock(ds2, b.style_id, b.channel))
                    self.assertEqual(self._stock(ds3, a.style_id, a.channel), after_apply)
            finally:
                session.SESSION_DIR = ROOT / ".session"

    def test_the_cli_data_folder_is_not_overridden_by_a_persisted_session(self):
        import tempfile

        import engine.session as session

        with tempfile.TemporaryDirectory() as tmp:
            session.SESSION_DIR = Path(tmp) / ".session"
            try:
                session.SESSION_DIR.mkdir()
                for f in session.FILES:
                    (session.SESSION_DIR / f).write_text("x")
                self.assertEqual(session.session_data_dir(session.DATA_DIR), session.SESSION_DIR)
                self.assertEqual(session.session_data_dir(Path(tmp)), Path(tmp))
            finally:
                session.SESSION_DIR = ROOT / ".session"

    def test_a_floor_above_the_mrp_is_a_conflict_not_a_rule_fix(self):
        from engine.load import build_dataset

        inv, h, c = (pd.read_csv(DATA / f) for f in ("inventory.csv", "daily_history.csv", "constraints.csv"))
        c.loc[c.style_id == "S05", "liquidation_floor_aed"] = 220.0  # above its MRP of 199
        plan = build_plan(build_dataset(inv, h, c), State(), Config())
        self.assertEqual(plan.violations, [])
        s05 = [a for a in plan.actions if a.style_id == "S05"]
        self.assertTrue(all(a.price_to == a.price_from for a in s05))  # no price can be legal: nothing is moved
        self.assertFalse(any(a.must_do for a in s05))
        self.assertTrue(any("floor" in n and "above the MRP" in n for n in plan.notes))

    def test_a_sold_out_listing_with_a_budget_still_set_gets_it_switched_off(self):
        from engine.load import build_dataset

        inv, h, c = (pd.read_csv(DATA / f) for f in ("inventory.csv", "daily_history.csv", "constraints.csv"))
        inv.loc[(inv.style_id == "S05") & (inv.channel == "noon"), "units_on_hand"] = 0  # sold out, budget still set
        plan = build_plan(build_dataset(inv, h, c), State(), Config())
        a = {x.listing_id: x for x in plan.actions}["S05:noon"]
        self.assertGreater(a.ads_from, 0)
        self.assertEqual(a.ads_to, 0.0, a.headline())
        self.assertIn("sold out", a.reason)
        self.assertEqual(a.exit_units, 0)
        self.assertEqual(a.price_to, a.price_from)
        self.assertEqual(plan.violations, [])

    def test_ads_are_trimmed_to_the_spend_the_projection_values_highest_not_switched_off(self):
        from engine.economics import project

        plan = build_plan(load_real(), State(), Config())
        by = {a.listing_id: a for a in plan.actions}
        t = plan.table.set_index("listing_id")
        a = by["S01:amazon_ae"]  # on track at ~1.9x: ads pay per sale, but today's AED 219/day over-buys
        self.assertGreater(a.ads_to, 0.0)
        self.assertLess(a.ads_to, a.ads_from)
        r = t.loc[a.listing_id]
        net = lambda s: project(r, float(r["price_now"]), s, None, plan.days_left, Config(), plan.sizes[a.listing_id]).net  # noqa: E731
        self.assertGreater(net(a.ads_to), net(0.0))
        self.assertGreater(net(a.ads_to), net(a.ads_from))
        self.assertIn("sell out", a.reason)

    def test_stop_reasons_tell_the_truth(self):
        from engine.economics import ads_pay_for_themselves

        plan = build_plan(load_real(), State(), Config())
        t = plan.table.set_index("listing_id")
        cfg = Config()
        for a in plan.changes:
            if a.ads_to == 0.0 and a.ads_from > 0:
                r = t.loc[a.listing_id]
                pays = ads_pay_for_themselves(r, float(a.price_to), cfg)
                if "more than the pair earns over the jobber" in a.reason:
                    self.assertFalse(pays, a.headline())
                if pays and r["on_hand"] > 0 and not r["listing_problem"]:
                    self.assertIn("organic traffic alone", a.reason, a.headline())

    def test_exits_are_read_at_the_price_the_listing_is_heading_for(self):
        plan = build_plan(load_real(), State(), Config())
        t = plan.table.set_index("listing_id")
        from engine.decide import PRICE_LEVER_KINDS, diagnose

        diag = diagnose(plan.table, Config())
        for a in plan.changes:
            if a.exit_units > 0:
                if diag[a.listing_id]["kind"] in PRICE_LEVER_KINDS:
                    self.assertIn("lowest price the rules allow", a.reason, a.headline())
                else:
                    self.assertIn("staying at", a.reason, a.headline())
                self.assertFalse(t.loc[a.listing_id, "listing_problem"])  # broken listings are checked first

    def test_past_the_deadline_everything_left_goes_to_the_jobber(self):
        ds = load_real()
        st = State(campaign_start=ds.today - timedelta(days=70))
        plan = build_plan(ds, st, Config())
        self.assertEqual(plan.days_left, 0)
        self.assertEqual(plan.violations, [])
        t = plan.table.set_index("listing_id")
        for a in plan.actions:
            if t.loc[a.listing_id, "on_hand"] > 0:
                self.assertEqual(a.exit_units, int(t.loc[a.listing_id, "on_hand"]), a.headline())
            self.assertEqual(a.ads_to, 0.0)
        self.assertTrue(any("deadline has passed" in n for n in plan.notes))

    def test_a_listing_nobody_has_seen_is_checked_not_advertised(self):
        from engine.load import build_dataset

        inv, h, c = (pd.read_csv(DATA / f) for f in ("inventory.csv", "daily_history.csv", "constraints.csv"))
        sel = (h.style_id == "S13") & (h.channel == "noon")
        h.loc[sel, ["sessions", "ad_clicks", "ad_spend_aed", "units_sold"]] = 0
        plan = build_plan(build_dataset(inv, h, c), State(), Config())
        a = {x.listing_id: x for x in plan.actions}["S13:noon"]
        self.assertEqual(a.ads_to, 0.0)
        self.assertTrue(a.check)
        self.assertTrue(any("no sessions at all" in f for f in a.flags))

    def test_impact_is_computed_from_what_will_actually_be_applied(self):
        from engine.decide import _project_action, _Work

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
        a = next(x for x in plan.changes if x.ads_to > x.ads_from)
        before = a.impact_aed
        a.ads_to = a.ads_from  # what the guardrail does to a raise that breaks the budget
        _project_action(w, a)
        self.assertNotEqual(a.impact_aed, before)
        self.assertAlmostEqual(a.impact_aed, 0.0 if a.price_to == a.price_from and a.exit_units == 0 else a.impact_aed, places=6)

    def test_a_parity_partner_is_applied_price_only(self):
        from engine.actions import price_only

        plan = build_plan(load_real(), State(), Config())
        full = next(a for a in plan.changes if a.price_to != a.price_from and a.ads_to != a.ads_from)
        part = price_only(full)
        self.assertEqual((part.price_from, part.price_to), (full.price_from, full.price_to))
        self.assertEqual(part.ads_to, part.ads_from)
        self.assertEqual((part.exit_units, part.exit_by_size, part.check), (0, None, False))
        self.assertEqual(part.predicted_rate, part.baseline_rate)  # not scored as the whole action
        self.assertEqual(full.ads_to != full.ads_from, True)  # the original is untouched


if __name__ == "__main__":
    unittest.main()
