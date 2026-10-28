"""Turn the estimates into tomorrow's actions.

build_plan runs these steps, each its own function:
  0. diagnose every listing (visibility problem / price problem / both / broken listing / on track)
  1. switch off ads that lose money against the jobber at today's price (economics.should_stop_ads)
  2. fix prices that already break a rule (floor raised overnight, parity drifted)
  3. choose prices per style, both channels together because parity ties them; a channel with no
     reason of its own to cut only follows to the shallowest price that keeps parity
  4. ads per channel (allocate.py) at the chosen prices; raises and trims must pay by the plan's
     own projection; then the price gate is re-checked at the final ad levels
  5. exits for the pairs that won't sell even on an optimistic read; one Action per listing
  6. guardrail over the whole plan (withdraw anything that breaks a rule), rank by AED impact, verdict

build_plan reads the state and never writes to it: scoring yesterday's predictions and fixing
day 1 of the campaign happen when data is loaded (session.py), not here.
"""

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .actions import Action
from .allocate import allocate_ads
from .config import Config
from .economics import ads_pay_for_themselves, marginal_ad_unit_cost, project, should_stop_ads
from .estimate import estimate_listings
from .guardrail import check_plan
from .load import Dataset
from .schema import AMAZON, CHANNEL_LABELS, NOON
from .state import State

PRICE_LEVER_KINDS = ("price", "both", "volume")  # diagnoses for which a price cut is on the table


@dataclass
class Plan:
    """Tomorrow's plan: one Action per listing (ranked, holds included), the verdict, the per-listing table it was
    built from, and anything the rules or the data flagged.
    """

    plan_date: date
    days_left: int
    table: pd.DataFrame
    sizes: dict
    actions: list  # one Action per listing, ranked; holds included
    verdict: dict
    violations: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    ad_totals: dict = field(default_factory=dict)
    conflicts: list = field(default_factory=list)  # rules already broken in the data that can't be fixed legally today

    @property
    def changes(self) -> list:
        """Actions that change something, in rank order."""
        return [a for a in self.actions if a.is_change]

    @property
    def holds(self) -> list:
        """Listings left as they are: 'hold' when on track, 'watch' when behind but no legal move is worth making."""
        return [a for a in self.actions if not a.is_change]


@dataclass
class _Work:
    """Everything the steps share while a plan is being built."""

    table: pd.DataFrame
    rows: dict  # listing_id -> row (a pandas Series)
    sizes: dict
    cfg: Config
    days_left: int
    plan_date: date
    diag: dict = field(default_factory=dict)
    ads_base: dict = field(default_factory=dict)  # listing_id -> spend assumed while prices are chosen
    price_plan: dict = field(default_factory=dict)  # listing_id -> price tomorrow
    price_reason: dict = field(default_factory=dict)
    forced: dict = field(default_factory=dict)  # listing_id -> True if the price moved only to keep parity
    must_do: set = field(default_factory=set)  # listings whose price move fixes a rule already broken
    fixed_styles: set = field(default_factory=set)
    ads_plan: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def net(self, lid: str, price: float, spend: float, exit_by_size: list | None = None, **kw) -> float:
        return project(self.rows[lid], price, spend, exit_by_size, self.days_left, self.cfg, self.sizes[lid], **kw).net

    def siblings(self, style_id: str) -> dict:
        """channel -> listing_id for one style."""
        return {r["channel"]: lid for lid, r in self.rows.items() if r["style_id"] == style_id}


# -------------------------------------------------------------------------------------------
# 0. diagnosis
# -------------------------------------------------------------------------------------------
def diagnose(table: pd.DataFrame, cfg: Config) -> dict:
    """Why each listing is where it is, and so which lever may be used.

    Returns listing_id -> {kind, why}. kind is one of: on_track, visibility (ads), price (cut), both (price first,
    ads only where they pay), listing (broken: stop spending, flag a human), new (too little data), volume (just
    too much stock). why is the sentence Farah reads. Traffic and conversion are compared with the channel median.
    """
    med_sessions = table.groupby("channel")["sessions_7"].transform("median")
    med_cvr = table.groupby("channel")["cvr_mean"].transform("median")
    out = {}
    for i, (_, r) in enumerate(table.iterrows()):
        head = f"{r['status']} at {r['ratio']:.1f}x the required {r['required_rate']:.1f}/day; "
        if r["status"] == "sold_out":
            kind, why = "sold_out", "sold out"
        elif r["status"] == "on_track":
            kind, why = "on_track", f"on track at {r['ratio']:.1f}x the required {r['required_rate']:.1f}/day"
        elif bool(r["is_new"]):
            kind, why = (
                "new",
                (
                    f"new listing with too little history to judge; conversion borrowed from peers ({r['cvr_mean']:.1%}). "
                    f"Hold the price, test demand with ads where they pay, decide after {cfg.cooldown_days} days"
                ),
            )
        elif bool(r["listing_problem"]):
            kind, why = (
                "listing",
                (
                    f"{r['status']} at {r['ratio']:.1f}x required; {r['discount']:.0%} off MRP and still {r['cvr_mean']:.1%} "
                    f"conversion on {r['sessions_7']:.0f} sessions/day: the listing is broken, not the price"
                ),
            )
        else:
            low_traffic = r["sessions_7"] < med_sessions.iloc[i]
            low_cvr = r["cvr_mean"] < med_cvr.iloc[i]
            if low_cvr and low_traffic:
                kind, why = (
                    "both",
                    head
                    + (
                        f"both traffic ({r['sessions_7']:.0f} sessions/day) and conversion ({r['cvr_mean']:.1%}) are below "
                        f"the channel median; price first, ads only where they pay"
                    ),
                )
            elif low_cvr:
                kind, why = (
                    "price",
                    head
                    + (
                        f"traffic is fine ({r['sessions_7']:.0f} sessions/day) but conversion is low ({r['cvr_mean']:.1%} "
                        f"vs median {med_cvr.iloc[i]:.1%}): a price problem"
                    ),
                )
            elif low_traffic and ads_pay_for_themselves(r, float(r["price_now"]), cfg):
                kind, why = (
                    "visibility",
                    head
                    + (
                        f"conversion is good ({r['cvr_mean']:.1%} vs median {med_cvr.iloc[i]:.1%}) but traffic is thin "
                        f"({r['sessions_7']:.0f} sessions/day): a visibility problem, ads before price"
                    ),
                )
            elif low_traffic:
                kind, why = (
                    "both",
                    head
                    + (
                        f"traffic is thin ({r['sessions_7']:.0f} sessions/day) and paid traffic doesn't pay at today's price "
                        f"(~AED {marginal_ad_unit_cost(r, float(r['price_now'])):,.0f} of ads per sale); fix the price first, "
                        f"ads only if the new price makes them pay"
                    ),
                )
            else:
                kind, why = (
                    "volume",
                    head + ("traffic and conversion are around the channel median; the stock is simply large for the time left"),
                )
        out[r["listing_id"]] = {"kind": kind, "why": why.replace("on_track", "on track")}
    return out


# -------------------------------------------------------------------------------------------
# 1. ads that lose money stop before anything else is decided
# -------------------------------------------------------------------------------------------
def _stop_losing_ads(w: _Work) -> None:
    """The spend each listing is assumed to run while prices are chosen: today's, or zero where
    economics.should_stop_ads says the ads lose money at today's price. The allocator applies the
    same rule at the chosen prices, so the two can only differ where the price moved."""
    w.ads_base = {
        lid: (0.0 if should_stop_ads(r, float(r["price_now"]), w.cfg, w.days_left, w.sizes[lid]) else float(r["ad_now"]))
        for lid, r in w.rows.items()
    }


# -------------------------------------------------------------------------------------------
# 2. prices that already break a rule are fixed first
# -------------------------------------------------------------------------------------------
def _can_move_price(r) -> bool:
    return int(r["slots"]) >= 1 and not bool(r["locked"])


def _fix_rule_breaks(w: _Work) -> None:
    """A price that already breaks a rule (floor raised overnight, parity drifted, a tighter band)
    moves to the nearest legal price before any optimisation. If the fix needs a slot the listing
    doesn't have, the rules conflict and it is reported for a human."""
    cfg, rows = w.cfg, w.rows
    for style, lids in {r["style_id"]: w.siblings(r["style_id"]) for r in rows.values()}.items():
        target, why = {}, {}
        impossible = False
        for lid in lids.values():
            r = rows[lid]
            p = float(r["price_now"])
            if not np.isfinite(p):
                continue
            if pd.notna(r["floor"]) and pd.notna(r["mrp"]) and r["floor"] > r["mrp"] + 1e-9:
                w.notes.append(
                    f"{lid}: the floor ({r['floor']:.0f}) is above the MRP ({r['mrp']:.0f}), so no price is legal; "
                    f"the constraints conflict and need a human decision; no price move proposed"
                )
                impossible = True
                continue
            if pd.notna(r["floor"]) and p < r["floor"] - 1e-9:
                p, why[lid] = float(math.ceil(r["floor"])), f"below the floor of {r['floor']:.0f}"
            if pd.notna(r["mrp"]) and p > r["mrp"] + 1e-9:
                p, why[lid] = float(math.floor(r["mrp"])), f"above MRP {r['mrp']:.0f}"
            target[lid] = p
        a_id, n_id = lids.get(AMAZON), lids.get(NOON)
        if a_id in target and n_id in target and target[a_id] > 0:
            pa, pn, band = target[a_id], target[n_id], cfg.parity_band
            if abs(pn / pa - 1) > band + 1e-9:
                rn = rows[n_id]
                pn_fix = float(min(max(pn, math.ceil(pa * (1 - band))), math.floor(pa * (1 + band))))
                noon_can = (
                    _can_move_price(rn)
                    and (pd.isna(rn["floor"]) or pn_fix >= rn["floor"])
                    and (pd.isna(rn["mrp"]) or pn_fix <= rn["mrp"])
                )
                if noon_can:
                    target[n_id], why[n_id] = pn_fix, f"{pn / pa - 1:+.1%} vs Amazon.ae, outside ±{band:.0%}"
                else:
                    pa_fix = float(min(max(pa, math.ceil(pn / (1 + band))), math.floor(pn / (1 - band))))
                    target[a_id], why[a_id] = pa_fix, f"Noon is {pn / pa - 1:+.1%} vs Amazon.ae and Noon cannot move today"
        if impossible:
            w.fixed_styles.add(style)  # neither channel's price is optimised while the constraints contradict each other
            continue
        moved = {lid: p for lid, p in target.items() if p != float(rows[lid]["price_now"])}
        if not moved:
            continue
        w.fixed_styles.add(style)
        for lid, p in moved.items():
            r = rows[lid]
            if not _can_move_price(r):
                w.notes.append(
                    f"{lid}: price {r['price_now']:.0f} breaks a rule ({why.get(lid, 'parity')}) but the listing has no "
                    f"price-change slot left this week; the rules conflict, so this needs a human decision"
                )
                continue
            w.price_plan[lid] = p
            w.must_do.add(lid)
            w.price_reason[lid] = (
                f"price {r['price_now']:.0f} breaks a rule today ({why.get(lid, 'parity')}); "
                f"moved to {p:.0f}, the nearest legal price, before any optimisation"
            )


# -------------------------------------------------------------------------------------------
# 3. prices, per style
# -------------------------------------------------------------------------------------------
def _own_reason_to_cut(r, diag_kind: str) -> bool:
    return diag_kind in PRICE_LEVER_KINDS and r["status"] not in ("on_track", "sold_out")


def _cut_candidates(r, cfg: Config) -> list[float]:
    """Prices a listing with its own reason to cut may be at tomorrow: current, then cuts in AED 5
    steps between the minimum cut and the deeper of (floor, max cut)."""
    p0 = float(r["price_now"])
    if not np.isfinite(p0) or p0 <= 0 or bool(r["is_new"]) or pd.isna(r["floor"]) or not _can_move_price(r):
        return [p0]
    if int(r["days_at_price"]) < cfg.cooldown_days:
        return [p0]
    lowest = max(float(r["floor"]), p0 * (1 - cfg.max_cut_pct))
    out, p = [p0], p0
    for _ in range(int(p0 // max(cfg.price_step, 1)) + 1):  # bounded, whatever the inputs
        p = p - cfg.price_step
        if p < lowest - 1e-9:
            break
        if (p0 - p) / p0 >= cfg.min_cut_pct - 1e-9:
            out.append(float(p))
    return out


def _follower_price(r, leader_price: float, cfg: Config) -> float | None:
    """The price a channel with no reason of its own to cut takes when the other channel moves:
    its current price if that is still within the parity band, else the shallowest in-band price.
    None if no legal price exists (floor, MRP, or no slot to move)."""
    p0 = float(r["price_now"])
    band = cfg.parity_band
    if not np.isfinite(p0) or leader_price is None or not np.isfinite(leader_price) or leader_price <= 0:
        return None
    if abs(p0 / leader_price - 1) <= band + 1e-9:
        return p0
    p = float(math.floor(leader_price * (1 + band))) if p0 > leader_price else float(math.ceil(leader_price * (1 - band)))
    if not _can_move_price(r) or bool(r["is_new"]):
        return None
    if pd.notna(r["floor"]) and p < r["floor"] - 1e-9:
        return None
    if pd.notna(r["mrp"]) and p > r["mrp"] + 1e-9:
        return None
    return p


def _price_options(w: _Work, lids: dict, own: dict) -> list[tuple]:
    """Legal (amazon_price, noon_price) pairs for a style. Both channels with their own reason to cut:
    every in-band pair of their candidates. One channel: its candidates, the other following."""
    a_id, n_id = lids.get(AMAZON), lids.get(NOON)
    cfg, rows = w.cfg, w.rows
    if a_id and n_id and own[a_id] and own[n_id]:
        return [
            (pa, pn)
            for pa in _cut_candidates(rows[a_id], cfg)
            for pn in _cut_candidates(rows[n_id], cfg)
            if abs(pn / pa - 1) <= cfg.parity_band + 1e-9
        ]
    if a_id and n_id:
        lead, follow = (a_id, n_id) if own[a_id] else (n_id, a_id)
        options = []
        for p_lead in _cut_candidates(rows[lead], cfg):
            p_follow = _follower_price(rows[follow], p_lead, cfg)
            if p_follow is not None:
                options.append((p_lead, p_follow) if lead == a_id else (p_follow, p_lead))
        return options
    lid = a_id or n_id
    return [(p, None) if a_id else (None, p) for p in _cut_candidates(rows[lid], cfg)]


def _pick(scored: list, closing: list, critical: bool, p0_by: dict, cfg: Config) -> tuple:
    """Which (pa, pn, net) to take. Smallest cut that reaches the sell-through target; otherwise, for
    a critical style, the cut with the best projected recovery; otherwise the cut closest to the
    default first step."""
    if closing:
        return max(closing, key=lambda x: (x[0] or 0) + (x[1] or 0))
    if critical:
        return max(scored, key=lambda x: x[2])

    def distance_from_default(x):
        return sum(
            abs((1 - p / p0_by[ch]) - cfg.default_cut_pct)
            for ch, p in ((AMAZON, x[0]), (NOON, x[1]))
            if p is not None and p0_by.get(ch)
        )

    moved = [x for x in scored if any(p is not None and p != p0_by.get(ch) for ch, p in ((AMAZON, x[0]), (NOON, x[1])))]
    return min(moved, key=distance_from_default) if moved else max(scored, key=lambda x: x[2])


def _choose_prices(w: _Work) -> None:
    """Per style: the legal price pairs, their projected net recovery (at the ads assumed in ads_base),
    the pick, and the gate: the style must beat holding by price_min_gain_aed per listing that moves."""
    cfg, rows = w.cfg, w.rows
    for style in {r["style_id"] for r in rows.values()}:
        if style in w.fixed_styles:
            continue
        lids = w.siblings(style)
        own = {lid: _own_reason_to_cut(rows[lid], w.diag[lid]["kind"]) for lid in lids.values()}
        if not any(own.values()):
            continue
        a_id, n_id = lids.get(AMAZON), lids.get(NOON)
        pa0 = float(rows[a_id]["price_now"]) if a_id else None
        pn0 = float(rows[n_id]["price_now"]) if n_id else None
        options = _price_options(w, lids, own)
        if len(options) <= 1:
            continue
        memo: dict = {}

        def proj(lid, p, _memo=memo):
            if (lid, p) not in _memo:
                _memo[(lid, p)] = project(rows[lid], p, w.ads_base[lid], None, w.days_left, cfg, w.sizes[lid])
            return _memo[(lid, p)]

        scored, closing = [], []
        for pa, pn in options:
            pr_a = proj(a_id, pa) if a_id else None
            pr_n = proj(n_id, pn) if n_id else None
            net = (pr_a.net if pr_a else 0.0) + (pr_n.net if pr_n else 0.0)
            scored.append((pa, pn, net))
            ok_a = pr_a is None or pr_a.sold >= cfg.sell_through_target * rows[a_id]["on_hand"] - 1e-9
            ok_n = pr_n is None or pr_n.sold >= cfg.sell_through_target * rows[n_id]["on_hand"] - 1e-9
            if ok_a and ok_n:
                closing.append((pa, pn, net))
        hold_net = (proj(a_id, pa0).net if a_id else 0.0) + (proj(n_id, pn0).net if n_id else 0.0)
        critical = any(rows[lid]["status"] == "critical" for lid in lids.values() if own[lid])
        pa, pn, net = _pick(scored, closing, critical, {AMAZON: pa0, NOON: pn0}, cfg)
        moves = [(lid, p_new, p_old) for lid, p_new, p_old in ((a_id, pa, pa0), (n_id, pn, pn0)) if lid and p_new != p_old]
        if not moves or net - hold_net < cfg.price_min_gain_aed * len(moves):
            # say so when a listing's own cut would pay but dragging the other channel along loses
            for lid in lids.values():
                other = [o for o in lids.values() if o != lid]
                if own[lid] and other:
                    p0 = float(rows[lid]["price_now"])
                    solo = max(proj(lid, p).net - proj(lid, p0).net for p in _cut_candidates(rows[lid], cfg))
                    if solo >= cfg.price_min_gain_aed:
                        w.price_reason[lid] = (
                            f"a cut here on its own would add ~AED {solo:,.0f}, but parity would pull "
                            f"{CHANNEL_LABELS.get(rows[other[0]]['channel'])} down with it and the style loses overall; held"
                        )
            continue
        mode = "closing" if closing else ("critical" if critical else "measured")
        for lid, p_new, p_old in moves:
            w.price_plan[lid] = float(p_new)
            w.forced[lid] = not own[lid]
            cut = 1 - p_new / p_old
            if not own[lid]:
                w.price_reason[lid] = (
                    f"cut {cut:.0%}, the smallest move that keeps Noon within ±{cfg.parity_band:.0%} of Amazon.ae after the other channel's cut"
                )
            elif mode == "closing":
                w.price_reason[lid] = (
                    f"cut {cut:.0%}: the smallest step that reaches {cfg.sell_through_target:.0%} sell-through by the deadline"
                )
            elif mode == "critical":
                w.price_reason[lid] = (
                    f"cut {cut:.0%}: no legal cut reaches the target, so a critical style takes the cut with the best projected "
                    f"recovery (at most {cfg.max_cut_pct:.0%} per change); what it still can't sell goes to exits"
                )
            else:
                w.price_reason[lid] = (
                    f"cut {cut:.0%}: a measured first move; the next slot reads the response before going deeper"
                )


def _unwind_style(w: _Work, style: str) -> None:
    """Take back a style's price moves (used when the gate fails once ads are final)."""
    for lid in w.siblings(style).values():
        if lid in w.price_plan and lid not in w.must_do:
            del w.price_plan[lid]
            w.forced.pop(lid, None)
            w.price_reason.pop(lid, None)


# -------------------------------------------------------------------------------------------
# 4. ads at the chosen prices
# -------------------------------------------------------------------------------------------
def _price_of(w: _Work, lid: str) -> float:
    return float(w.price_plan.get(lid, w.rows[lid]["price_now"]))


def _finalise_ads(w: _Work) -> None:
    """Allocate, then drop raises and trims that don't earn their keep: every one must beat the
    current spend by ads_min_gain_aed at the planned price (by the same projection the verdict
    uses), and move by at least ads_min_delta_aed. Stops and cap cuts are never dropped: the first
    lose money, the second are the budget rule."""
    cfg, rows = w.cfg, w.rows
    w.ads_plan = allocate_ads(w.table, w.price_plan, cfg, w.days_left, w.sizes)
    reverted: dict[str, float] = {}
    for lid, r in rows.items():
        rec = w.ads_plan.get(lid)
        if not rec or rec["kind"] not in ("raise", "trim"):
            continue
        small = abs(rec["spend"] - r["ad_now"]) < cfg.ads_min_delta_aed
        gain = w.net(lid, _price_of(w, lid), rec["spend"]) - w.net(lid, _price_of(w, lid), float(r["ad_now"]))
        if small or gain < cfg.ads_min_gain_aed:
            reverted[lid] = rec["spend"]
            w.ads_plan[lid] = {"spend": float(r["ad_now"]), "reason": "", "kind": "keep"}
    # reverting a trim must not push a channel back over its cap
    for ch in w.table["channel"].unique():
        cap = float(cfg.ad_budget.get(ch, 0.0))
        while sum(w.ads_plan[lid]["spend"] for lid in rows if lid.endswith(":" + ch)) > cap + 1e-6:
            cands = [lid for lid in reverted if lid.endswith(":" + ch) and reverted[lid] < rows[lid]["ad_now"]]
            if not cands:
                break
            lid = max(cands, key=lambda x: rows[x]["ad_now"] - reverted[x])
            w.ads_plan[lid] = {
                "spend": reverted.pop(lid),
                "reason": "trimmed to keep the channel inside its daily budget",
                "kind": "trim",
            }


def _recheck_price_gate(w: _Work) -> bool:
    """Prices were chosen assuming the ads in ads_base; the allocator may have set them differently.
    Re-run the gate at the final ad levels and unwind any style that no longer clears it.
    Returns True if anything was unwound (the caller then re-allocates once)."""
    cfg, rows = w.cfg, w.rows
    unwound = False
    for style in {rows[lid]["style_id"] for lid in w.price_plan if lid not in w.must_do}:
        lids = w.siblings(style)
        moved = [lid for lid in lids.values() if lid in w.price_plan and lid not in w.must_do]
        if not moved:
            continue
        gain = 0.0
        for lid in lids.values():
            spend = w.ads_plan[lid]["spend"]
            gain += w.net(lid, _price_of(w, lid), spend) - w.net(lid, float(rows[lid]["price_now"]), spend)
        if gain < cfg.price_min_gain_aed * len(moved):
            _unwind_style(w, style)
            unwound = True
    return unwound


# -------------------------------------------------------------------------------------------
# 5. exits, then one Action per listing
# -------------------------------------------------------------------------------------------
def _deepest_legal_price(w: _Work, lid: str, p_new: float) -> float:
    """The lowest price the rules let this listing reach in the days left: the floor, or as far as the
    allowed changes (max_cut_pct each, max_price_changes per week, one every cooldown) get toward it."""
    r, cfg = w.rows[lid], w.cfg
    if not np.isfinite(p_new) or p_new <= 0:
        return p_new
    floor = float(r["floor"]) if pd.notna(r["floor"]) else 0.0
    weeks = max(1, math.ceil(w.days_left / cfg.change_window_days))
    n_cuts = min(cfg.max_price_changes * weeks, max(1, w.days_left // max(cfg.cooldown_days, 1)))
    reach = p_new * (1 - cfg.max_cut_pct) ** n_cuts
    return float(max(floor, reach, 0.0))


def _size_exit(w: _Work, lid: str, p_new: float, a_new: float) -> tuple[list | None, int, str]:
    """Pairs that won't sell by the deadline even at the deepest price the rules allow, with
    conversion at the top of its range and a cut working better than expected. Exits are
    irreversible and pay a quarter of cost, so the read is deliberately optimistic, and a listing
    that may simply be broken is checked by a human before any of it goes.

    Returns (exits by size, units, why) — why is the sentence for the card, also when nothing exits for a reason
    Farah should hear."""
    r, cfg = w.rows[lid], w.cfg
    if bool(r["exited_today"]) or float(r["on_hand"]) <= 0:
        return None, 0, ""
    if w.days_left <= 0:
        by_size = [int(round(u)) for u in w.sizes[lid]["on_hand"]]
        return by_size, sum(by_size), "the deadline has passed: whatever is left goes to the jobber now"
    if bool(r["listing_problem"]):
        checked = int(r["checked_days"])
        if checked < 0:
            return (
                None,
                0,
                (
                    "exit held: a listing this dead at this discount may be broken (images, size chart, buy box), and a fixed "
                    "listing recovers far more than the jobber; check it first"
                ),
            )
        if checked < cfg.listing_check_days:
            return (
                None,
                0,
                (
                    f"exit held: the listing check was requested {checked} day(s) ago; the exit waits "
                    f"{cfg.listing_check_days} days for the fix to show in the data"
                ),
            )
    # the price the listing is headed for: today's if nothing says cut it (the tails of an on-track style
    # are not worth cutting the whole listing for), the deepest the rules allow if it is being cut
    p_read = _deepest_legal_price(w, lid, p_new) if w.diag[lid]["kind"] in PRICE_LEVER_KINDS else p_new
    optimistic = project(
        r, p_read, a_new, None, w.days_left, cfg, w.sizes[lid], cvr=r["cvr_p90"], elasticity=r["elasticity"] + r["elasticity_sd"]
    )
    by_size = [int(np.floor(u)) for u in optimistic.unsold_by_size]
    units = sum(by_size)
    if units < max(cfg.exit_min_units, cfg.exit_min_share * r["on_hand"]):
        return None, 0, ""
    if w.net(lid, p_new, a_new, by_size) - w.net(lid, p_new, a_new) < cfg.exit_min_gain_aed:
        return None, 0, ""
    at = (
        f"even at the lowest price the rules allow in the time left (AED {p_read:.0f})"
        if p_read < p_new - 1e-9
        else f"at AED {p_new:.0f}, the price this listing is staying at"
    )
    why = (
        f"{at} and on an optimistic read ({r['cvr_p90']:.1%} conversion) {units} pairs won't sell by the deadline; "
        f"sending them now saves holding cost and takes the jobber-price risk off the table"
    )
    return by_size, units, why


def _watch_reason(w: _Work, lid: str) -> str:
    r, cfg = w.rows[lid], w.cfg
    if bool(r["locked"]):
        return "already changed today; one move per listing per day"
    if int(r["days_at_price"]) < cfg.cooldown_days and int(r["slots"]) >= 1 and w.diag[lid]["kind"] in PRICE_LEVER_KINDS:
        return f"price changed {int(r['days_at_price'])} day(s) ago; reading the response before cutting again"
    if r["at_floor"]:
        return "at the floor, so price cannot move; ads already at what they can profitably do"
    if int(r["slots"]) < 1:
        return "both price changes for this rolling week are used; the next slot opens when the oldest change ages out"
    if w.diag[lid]["kind"] == "listing":
        return "needs a human: check images, size chart and buy-box status before spending anything more on it"
    return "no legal move beats holding by enough to be worth a change today"


def _build_action(w: _Work, lid: str) -> Action:
    """The Action for one listing. Its impact and projections are filled in by _project_action once
    the guardrail has had its say, so what is shown is what will be applied."""
    r = w.rows[lid]
    p_new = _price_of(w, lid)
    a_new = float(w.ads_plan[lid]["spend"])
    p_old, a_old = float(r["price_now"]), float(r["ad_now"])
    exit_by_size, exit_units, exit_why = _size_exit(w, lid, p_new, a_new)
    clears_stock = exit_units > 0 and exit_units >= float(r["on_hand"]) - 1e-9
    if clears_stock and a_new > 0:
        a_new = 0.0  # nothing left to advertise once everything has gone
    check = w.diag[lid]["kind"] == "listing" and int(r["checked_days"]) < 0

    kinds = [
        k
        for k, changed in (
            ("price", p_new != p_old),
            ("ads", abs(a_new - a_old) >= 1),
            ("exit", exit_units > 0),
            ("check", check),
        )
        if changed
    ]
    kind = "+".join(kinds) if kinds else ("watch" if r["status"] in ("behind", "critical") else "hold")

    if w.forced.get(lid):
        reasons = [w.price_reason[lid], "on its own merits: " + w.diag[lid]["why"]]
    else:
        reasons = [w.diag[lid]["why"]]
        if lid in w.price_reason:
            reasons.append(w.price_reason[lid])
    if "ads" in kinds:
        reasons.append(
            w.ads_plan[lid]["reason"] or ("ads stop with the stock: nothing left to advertise" if clears_stock else "")
        )
    if exit_why:
        reasons.append(exit_why)
    if kind == "watch":
        reasons.append(_watch_reason(w, lid))
    flags = list(r["notes"])
    if lid in w.must_do:
        flags.insert(0, "rule fix: do this first")
    if w.forced.get(lid):
        flags.append("price moved for parity, not on its own merits")
    if w.diag[lid]["kind"] == "listing":
        flags.append("human check: listing content / buy box")

    return Action(
        listing_id=lid,
        style_id=r["style_id"],
        channel=r["channel"],
        style_name=r["style_name"],
        kind=kind,
        price_from=p_old,
        price_to=p_new,
        ads_from=a_old,
        ads_to=a_new,
        exit_units=int(exit_units),
        exit_by_size=exit_by_size,
        exit_value=float(exit_units * r["jobber_value"]),
        forced_by_parity=bool(w.forced.get(lid, False)),
        reason=" · ".join(x for x in reasons if x),
        flags=flags,
        must_do=lid in w.must_do,
        check=check,
    )


def _project_action(w: _Work, a: Action) -> dict:
    """Hold vs plan projections for the action as it finally stands (after the guardrail), with the
    conversion range; also sets the action's impact and predicted/baseline rates from the same numbers."""
    r, cfg, lid = w.rows[a.listing_id], w.cfg, a.listing_id
    p_old, a_old = float(r["price_now"]), float(r["ad_now"])
    p_new = a.price_to if a.price_to is not None and a.price_to == a.price_to else p_old
    a_new = a.ads_to if a.ads_to is not None else a_old

    def pr(price, spend, exits=None, **kw):
        return project(r, price, spend, exits, w.days_left, cfg, w.sizes[lid], **kw)

    hold, plan = pr(p_old, a_old), pr(p_new, a_new, a.exit_by_size)
    a.impact_aed = float(plan.net - hold.net)
    a.predicted_rate, a.baseline_rate = float(plan.rate), float(hold.rate)
    return {
        "hold": hold,
        "plan": plan,
        "hold_p10": pr(p_old, a_old, cvr=r["cvr_p10"]),
        "hold_p90": pr(p_old, a_old, cvr=r["cvr_p90"]),
        "plan_p10": pr(p_new, a_new, a.exit_by_size, cvr=r["cvr_p10"]),
        "plan_p90": pr(p_new, a_new, a.exit_by_size, cvr=r["cvr_p90"]),
    }


# -------------------------------------------------------------------------------------------
# 6. guardrail, rank, verdict
# -------------------------------------------------------------------------------------------
def _withdraw(a: Action, why: str) -> None:
    a.kind, a.price_to, a.ads_to, a.exit_units, a.exit_by_size, a.impact_aed = "watch", a.price_from, a.ads_from, 0, None, 0.0
    a.flags.append(why)


def _apply_guardrail(w: _Work, actions: list) -> tuple[list, list]:
    """Check the whole plan. A rule already broken in the data that the plan could not fix is a
    conflict to report; anything the plan itself would break is withdrawn, listing by listing
    (both channels of a style for parity; every raise in the channel for a budget breach)."""
    inherited = {(v.rule, v.listing_id) for v in check_plan([], w.table, w.cfg)}
    violations = [
        v for v in check_plan([a for a in actions if a.is_change], w.table, w.cfg) if (v.rule, v.listing_id) not in inherited
    ]
    conflicts = [
        v for v in check_plan([a for a in actions if a.is_change], w.table, w.cfg) if (v.rule, v.listing_id) in inherited
    ]
    if not violations:
        return [], conflicts
    for v in violations:
        for a in actions:
            if v.rule == "budget" and a.channel == v.listing_id and a.ads_to > a.ads_from:
                a.ads_to = a.ads_from
                a.flags.append("ad raise withdrawn: the channel's budget was exceeded")
                if a.price_to == a.price_from and a.exit_units == 0:
                    a.kind = "watch"
            elif v.rule == "parity" and a.style_id == v.listing_id.split(":")[0] or a.listing_id == v.listing_id:
                _withdraw(a, "a proposed move failed the guardrail and was withdrawn")
    w.notes.append("guardrail withdrew: " + "; ".join(f"{v.listing_id} {v.rule}: {v.detail}" for v in violations))
    remaining = [
        v for v in check_plan([a for a in actions if a.is_change], w.table, w.cfg) if (v.rule, v.listing_id) not in inherited
    ]
    return remaining, conflicts


def _verdict(w: _Work, actions: list, projections: dict) -> dict:
    table, cfg = w.table, w.cfg
    on_hand = int(table["on_hand"].sum())

    def agg(key):
        ps = [projections[a.listing_id][key] for a in actions]
        return dict(
            sold=sum(p.sold for p in ps),
            unsold=sum(p.unsold for p in ps),
            exits=sum(p.exit_units for p in ps),
            net=sum(p.net for p in ps),
        )

    hold_t, plan_t = agg("hold"), agg("plan")
    st_plan = plan_t["sold"] / on_hand if on_hand else 0.0
    return dict(
        plan_date=w.plan_date,
        days_left=w.days_left,
        on_hand=on_hand,
        required_rate=on_hand / w.days_left if w.days_left else 0.0,
        actual_rate=float(table["rr7"].sum()),
        hold=hold_t,
        plan=plan_t,
        plan_range=(agg("plan_p10")["sold"] / on_hand, agg("plan_p90")["sold"] / on_hand) if on_hand else (0, 0),
        sell_through_hold=hold_t["sold"] / on_hand if on_hand else 0.0,
        sell_through_plan=st_plan,
        jobber_today=float((table["on_hand"] * table["jobber_value"]).sum()),
        make_it=st_plan >= cfg.sell_through_target,
        n_changes=sum(1 for a in actions if a.is_change),
        n_on_track=int((table["status"] == "on_track").sum()),
        n_behind=int((table["status"] == "behind").sum()),
        n_critical=int((table["status"] == "critical").sum()),
    )


def build_plan(ds: Dataset, state: State, cfg: Config) -> Plan:
    """Build tomorrow's plan from the data, what has been applied so far, and the config.

    Pure with respect to its inputs: nothing in `state` is changed. See the module docstring for
    the steps.
    """
    plan_date = ds.today + timedelta(days=1)
    table, sizes = estimate_listings(ds, state, cfg, plan_date)
    w = _Work(
        table=table,
        rows={r["listing_id"]: r for _, r in table.iterrows()},
        sizes=sizes,
        cfg=cfg,
        days_left=int(table.attrs["days_left"]),
        plan_date=plan_date,
    )
    w.diag = diagnose(table, cfg)
    _stop_losing_ads(w)
    _fix_rule_breaks(w)
    _choose_prices(w)
    _finalise_ads(w)
    if _recheck_price_gate(w):
        _finalise_ads(w)

    actions = [_build_action(w, lid) for lid in w.rows]
    violations, conflicts = _apply_guardrail(w, actions)
    projections = {a.listing_id: _project_action(w, a) for a in actions}  # after the guardrail: what is shown is what applies
    if w.days_left <= 0:
        w.notes.append("the deadline has passed: everything left goes to the jobber today; ads stop")

    # a parity follower is judged with its leader: show the style-level number on both
    for style in {a.style_id for a in actions if a.forced_by_parity}:
        moved = [a for a in actions if a.style_id == style and a.price_to != a.price_from]
        parts = ", ".join(f"{CHANNEL_LABELS.get(a.channel, a.channel)} {a.impact_aed:+,.0f}" for a in moved)
        for a in moved:
            a.reason += f" · style-level net {sum(x.impact_aed for x in moved):+,.0f} AED across channels ({parts})"

    actions.sort(key=lambda a: (not a.is_change, not a.must_do, -a.impact_aed))
    for i, a in enumerate(actions):
        a.rank = i + 1

    ad_totals = {"plan": {}, "now": {}}
    for ch in table["channel"].unique():
        ad_totals["plan"][ch] = float(sum(a.ads_to for a in actions if a.channel == ch))
        ad_totals["now"][ch] = float(table.loc[table["channel"] == ch, "ad_now"].sum())
    return Plan(
        plan_date=plan_date,
        days_left=w.days_left,
        table=table,
        sizes=sizes,
        actions=actions,
        verdict=_verdict(w, actions, projections),
        violations=violations,
        notes=w.notes + ds.issues,
        ad_totals=ad_totals,
        conflicts=conflicts,
    )


def verdict_text(v: dict) -> str:
    """Deterministic verdict; the LLM brief is layered on top of this, and this is what Farah
    sees if the LLM is unavailable."""
    h, p = v["hold"], v["plan"]
    lines = []
    if v["make_it"]:
        lines.append(
            f"Yes, if today's actions are applied: projected sell-through {v['sell_through_plan']:.0%} of {v['on_hand']:,} pairs "
            f"by the deadline (range {v['plan_range'][0]:.0%}–{v['plan_range'][1]:.0%})."
        )
    else:
        lines.append(
            f"Not all of it. If nothing changes, about {h['sold']:,.0f} of {v['on_hand']:,} pairs sell by the deadline "
            f"({v['sell_through_hold']:.0%}) and ~{h['unsold']:,.0f} go to the jobber. With today's actions: ~{p['sold']:,.0f} sold "
            f"({v['sell_through_plan']:.0%}, range {v['plan_range'][0]:.0%}–{v['plan_range'][1]:.0%}), {p['exits']:,.0f} exited now, "
            f"~{p['unsold']:,.0f} left for the jobber on the last day."
        )
    lines.append(
        f"Net recovery: AED {p['net']:,.0f} with the plan vs AED {h['net']:,.0f} if nothing changes "
        f"(+AED {p['net'] - h['net']:,.0f}); jobbering everything today would pay AED {v['jobber_today']:,.0f}."
    )
    lines.append(
        f"Run-rate needed {v['required_rate']:.0f}/day, actual {v['actual_rate']:.0f}/day. "
        f"{v['n_on_track']} listings on track, {v['n_behind']} behind, {v['n_critical']} critical; {v['n_changes']} actions today."
    )
    return " ".join(lines)
