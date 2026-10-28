"""The five rules, in one place, checked on the whole plan before Farah sees any of it.

A recommendation that breaks a rule is a failure however well it is explained, so this is
not advisory: decide.py refuses to emit a plan with violations, and the app refuses to
apply one.
"""

from dataclasses import dataclass

import pandas as pd

from .actions import Action
from .config import Config
from .schema import AMAZON, CHANNEL_LABELS, NOON


@dataclass
class Violation:
    """A broken rule: which rule, where, and the numbers that break it."""

    rule: str
    listing_id: str
    detail: str


def _planned_prices(actions: list[Action], table: pd.DataFrame) -> dict:
    prices = {r["listing_id"]: float(r["price_now"]) for _, r in table.iterrows()}
    for a in actions:
        if a.price_to is not None:
            prices[a.listing_id] = float(a.price_to)
    return prices


def _planned_ads(actions: list[Action], table: pd.DataFrame) -> dict:
    ads = {r["listing_id"]: float(r["ad_now"]) for _, r in table.iterrows()}
    for a in actions:
        if a.ads_to is not None:
            ads[a.listing_id] = float(a.ads_to)
    return ads


def check_plan(actions: list[Action], table: pd.DataFrame, cfg: Config) -> list[Violation]:
    """Returns an empty list when every rule holds for the plan as a whole."""
    v: list[Violation] = []
    t = table.set_index("listing_id")
    prices = _planned_prices(actions, table)
    ads = _planned_ads(actions, table)

    # 1. floor and MRP ceiling, per listing
    for lid, p in prices.items():
        if p != p:  # no price known (new listing with nothing to copy): nothing to check
            continue
        floor, mrp = t.loc[lid, "floor"], t.loc[lid, "mrp"]
        if pd.notna(floor) and p < floor - 1e-9:
            v.append(Violation("floor", lid, f"price {p:.0f} below floor {floor:.0f}"))
        if pd.notna(mrp) and p > mrp + 1e-9:
            v.append(Violation("mrp", lid, f"price {p:.0f} above MRP {mrp:.0f}"))

    # 2. cross-channel parity, per style, after all moves
    by_style: dict[str, dict] = {}
    for lid, p in prices.items():
        s, ch = lid.split(":")
        by_style.setdefault(s, {})[ch] = p
    for s, chans in by_style.items():
        if AMAZON in chans and NOON in chans and chans[AMAZON] > 0 and chans[NOON] == chans[NOON]:
            gap = chans[NOON] / chans[AMAZON] - 1
            if abs(gap) > cfg.parity_band + 1e-9:
                v.append(
                    Violation("parity", f"{s}:noon", f"Noon {chans[NOON]:.0f} is {gap:+.1%} vs Amazon.ae {chans[AMAZON]:.0f}")
                )

    # 3. daily ad budget per channel
    totals: dict[str, float] = {}
    for lid, a in ads.items():
        ch = lid.split(":")[1]
        totals[ch] = totals.get(ch, 0.0) + a
        if a < 0:
            v.append(Violation("ads", lid, "negative ad spend"))
    for ch, total in totals.items():
        cap = cfg.ad_budget.get(ch)
        if cap is not None and total > cap + 1e-6:
            v.append(Violation("budget", ch, f"{CHANNEL_LABELS.get(ch, ch)} ads AED {total:,.0f} exceed cap AED {cap:,.0f}"))

    # 4. two price changes per listing per rolling 7 days (slots already count the window)
    for a in actions:
        changed = a.price_to is not None and a.price_from is not None and a.price_to != a.price_from
        if changed and int(t.loc[a.listing_id, "slots"]) < 1:
            v.append(
                Violation(
                    "change_limit",
                    a.listing_id,
                    f"already {int(t.loc[a.listing_id, 'changes_in_window'])} price changes in the rolling window",
                )
            )

    # 5. exits cannot exceed stock
    for a in actions:
        if a.exit_units > int(t.loc[a.listing_id, "on_hand"]):
            v.append(Violation("exit", a.listing_id, f"exit {a.exit_units} exceeds stock {int(t.loc[a.listing_id, 'on_hand'])}"))
    return v
