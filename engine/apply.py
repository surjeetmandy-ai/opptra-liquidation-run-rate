"""Apply = a convincing state change. No marketplace API; the state file and the in-memory
inventory move, and tomorrow's plan is built against them."""

from datetime import date

import numpy as np

from .actions import Action
from .load import Dataset
from .state import State


def _remove_units(ds: Dataset, style_id: str, channel: str, units: int, by_size: list | None, sizes_order: list | None) -> None:
    inv = ds.inventory
    mask = (inv["style_id"] == style_id) & (inv["channel"] == channel)
    if by_size is not None and sizes_order is not None:
        for sz, u in zip(sizes_order, by_size, strict=False):
            if u <= 0:
                continue
            m = mask & (inv["size_eu"].astype(str) == str(sz))
            inv.loc[m, "units_on_hand"] = (inv.loc[m, "units_on_hand"] - int(u)).clip(lower=0)
        return
    # proportional removal (exits recorded without a size breakdown)
    on_hand = inv.loc[mask, "units_on_hand"].astype(float)
    total = on_hand.sum()
    if total <= 0 or units <= 0:
        return
    take = np.floor(on_hand / total * min(units, total)).astype(int)
    remainder = int(min(units, total) - take.sum())
    order = np.argsort(-(on_hand / total * min(units, total) - take).values)
    for idx in order[:remainder]:
        take.iloc[idx] += 1
    inv.loc[mask, "units_on_hand"] = (on_hand - take).clip(lower=0).astype(int)


def apply_action(ds: Dataset, state: State, action: Action, plan_date: date, sizes_order: list | None = None) -> list[str]:
    """Mutates state and ds.inventory. Returns a list of one-line confirmations."""
    done = []
    if action.price_to is not None and action.price_from is not None and action.price_to != action.price_from:
        state.apply_price(action.listing_id, action.price_from, action.price_to, plan_date, action.forced_by_parity)
        done.append(f"{action.listing_id}: price {action.price_from:.0f} → {action.price_to:.0f} from {plan_date}")
    if action.ads_to is not None and action.ads_from is not None and abs(action.ads_to - action.ads_from) >= 1:
        state.apply_ads(action.listing_id, action.ads_to, plan_date)
        done.append(f"{action.listing_id}: ads AED {action.ads_from:.0f} → {action.ads_to:.0f}/day from {plan_date}")
    if action.check:
        state.record_check(action.listing_id, plan_date)
        done.append(f"{action.listing_id}: listing check requested on {plan_date}")
    if action.exit_units > 0:
        state.apply_exit(action.listing_id, action.exit_units, action.exit_value, plan_date, action.exit_by_size, sizes_order)
        _remove_units(ds, action.style_id, action.channel, action.exit_units, action.exit_by_size, sizes_order)
        ds.exited[action.listing_id] = ds.exited.get(action.listing_id, 0) + int(action.exit_units)
        done.append(f"{action.listing_id}: {action.exit_units} pairs to the jobber for AED {action.exit_value:,.0f}")
    if done:
        state.record_prediction(plan_date, action.listing_id, action.kind, action.predicted_rate, action.baseline_rate)
    return done


def replay_exits(ds: Dataset, state: State, skip: int = 0) -> None:
    """On a fresh load of the CSVs, take out the units already sent to the jobber. `skip` = the first
    exit records that are already reflected in the inventory being loaded (a persisted session)."""
    for e in state.exits[skip:]:
        s, ch = e["listing_id"].split(":")
        _remove_units(ds, s, ch, int(e["units"]), e.get("by_size"), e.get("sizes"))
        ds.exited[e["listing_id"]] = ds.exited.get(e["listing_id"], 0) + int(e["units"])


def score_predictions(ds: Dataset, state: State) -> None:
    """Fill in what actually happened after each applied action, once the data has arrived.

    The actual run-rate is measured from the action's date up to (not including) the listing's
    next applied action, so overlapping actions on one listing don't score each other's days.
    Writes into state.predictions; call it when new data is loaded, not while planning.
    """
    hist = ds.history
    by_listing: dict[str, list] = {}
    for p in state.predictions:
        by_listing.setdefault(p["listing_id"], []).append(p)
    for lid, preds in by_listing.items():
        s, ch = lid.split(":")
        h = hist[(hist["style_id"] == s) & (hist["channel"] == ch)]
        preds.sort(key=lambda p: p["date"])
        for i, p in enumerate(preds):
            start = date.fromisoformat(p["date"])
            end = date.fromisoformat(preds[i + 1]["date"]) if i + 1 < len(preds) else None
            window = h[(h["date"] >= start) & ((h["date"] < end) if end else True)]
            if len(window):
                p["actual_rate"] = float(window["units_sold"].mean())
                p["days_observed"] = int(len(window))
