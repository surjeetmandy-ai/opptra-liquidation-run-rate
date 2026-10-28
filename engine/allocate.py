"""Ad budget allocation, per channel, under the daily cap.

Greedy and incremental on purpose: for every listing that spends, find the spend between zero
and today's that the projection values highest (stop if that is zero, trim if it is lower), then
push the freed money plus any headroom to the listings that are behind, AED 10 at a time, to
whichever listing's next AED 10 adds the most projected net recovery, until the cap or until no
step pays. Forty listings and a brand manager who has to act on it: an optimiser would be more
exact and less explainable, and the estimate it would optimise is itself soft.

Every step is valued with economics.project(), the same function that values the final plan,
so a raise can't look good here and bad there.
"""

import numpy as np
import pandas as pd

from .config import Config
from .economics import AD_STEP, ads_pay_for_themselves, best_spend_up_to, marginal_ad_unit_cost, project, should_stop_ads

STEP = AD_STEP


def _round5(x: float) -> float:
    return float(5 * round(x / 5))


def _stop_reason(r, price: float, cfg: Config, saved: float) -> str:
    """Why the ads stop, in the terms that are actually true for this listing."""
    if float(r["on_hand"]) <= 0:
        return (
            f"sold out; a budget of AED {r['ad_now']:.0f}/day is still set on a listing with nothing left to sell: switch it off"
        )
    if bool(r["listing_problem"]):
        return (
            "near-zero conversion at a deep discount; ads cannot fix a broken listing. "
            f"Stopping is projected to recover AED {saved:,.0f} more by the deadline"
        )
    if not ads_pay_for_themselves(r, price, cfg):
        mcost = marginal_ad_unit_cost(r, price)
        return (
            f"each ad-driven sale costs ~AED {mcost:,.0f} against a price of {price:.0f}, more than the pair earns over "
            f"the jobber (AED {r['jobber_value']:.0f}); stopping is projected to recover AED {saved:,.0f} more by the deadline"
        )
    return (
        "the stock sells by the deadline on organic traffic alone, so ads only bring the same sales forward; "
        f"stopping is projected to recover AED {saved:,.0f} more"
    )


def allocate_ads(table: pd.DataFrame, price_plan: dict, cfg: Config, days_left: int, sizes: dict) -> dict:
    """Tomorrow's ad spend per listing, given the prices the plan has chosen.

    Returns listing_id -> {"spend", "reason", "kind"}; kind is one of stop / trim / cap / raise / keep.
    """
    out: dict = {}
    for ch, grp in table.groupby("channel"):
        cap = float(cfg.ad_budget.get(ch, 0.0))
        rows = {r["listing_id"]: r for _, r in grp.iterrows()}
        price = {lid: float(price_plan.get(lid, r["price_now"])) for lid, r in rows.items()}
        spend = {lid: float(r["ad_now"]) for lid, r in rows.items()}
        reason = {lid: "" for lid in rows}
        kind = {lid: "keep" for lid in rows}

        def net_at(lid: str, s: float, _rows=rows, _price=price) -> float:
            return project(_rows[lid], _price[lid], s, None, days_left, cfg, sizes[lid]).net

        # locked: changed today, or changed in the last few days and one or two days of data are
        # noise. A broken listing is never locked: its ads stop regardless.
        locked = {
            lid
            for lid, r in rows.items()
            if (bool(r["locked"]) or int(r["days_at_ads"]) < cfg.cooldown_days) and not bool(r["listing_problem"])
        }

        # 1. stop where the projection says no spend beats zero, or there is nothing to advertise
        for lid, r in rows.items():
            if lid in locked or not should_stop_ads(r, price[lid], cfg, days_left, sizes[lid]):
                continue
            saved = net_at(lid, 0.0) - net_at(lid, spend[lid])
            spend[lid] = 0.0
            kind[lid] = "stop"
            reason[lid] = _stop_reason(r, price[lid], cfg, saved)

        # 2. trim where a lower spend is worth more than today's: enough to sell out by the deadline
        for lid, r in rows.items():
            if lid in locked or kind[lid] != "keep" or spend[lid] <= 0:
                continue
            best, _ = best_spend_up_to(r, price[lid], cfg, days_left, sizes[lid])
            if best < spend[lid] - 1e-9:
                old = spend[lid]
                spend[lid] = best
                kind[lid] = "trim"
                reason[lid] = (
                    f"AED {best:.0f}/day is the spend the projection values highest at this price: enough to sell out by "
                    f"the deadline; freeing AED {old - best:.0f}/day for listings that are behind"
                )

        # 2b. if the channel is still over its cap (a cap cut below today's spend), take money back
        #     from the listings whose last dirhams add the least. The budget is a hard rule, so this
        #     overrides cooldowns and same-day locks (those go last).
        pool = cap - sum(spend.values())
        for _ in range(int(-pool // STEP) + 2):
            if pool >= -1e-9:
                break
            worst, worst_key = None, None
            for lid in rows:
                if spend[lid] <= 1e-9:
                    continue
                step = min(STEP, spend[lid])
                val = (net_at(lid, spend[lid]) - net_at(lid, spend[lid] - step)) / step
                key = (lid in locked, val)
                if worst_key is None or key < worst_key:
                    worst, worst_key = lid, key
            if worst is None:
                break
            take = min(STEP, spend[worst], -pool)
            spend[worst] -= take
            pool += take
            if kind[worst] in ("keep", "trim", "raise"):
                kind[worst] = "cap"
                reason[worst] = "the channel's budget is below today's spend; this listing's last dirhams add the least"

        # 3. push the pool, AED 10 at a time, to the listing whose next AED 10 adds the most net
        #    recovery, among listings that are behind and whose ads pay at the planned price
        candidates = [
            lid
            for lid, r in rows.items()
            if lid not in locked
            and kind[lid] != "stop"
            and not bool(r["listing_problem"])  # a broken listing gets a human check, not more traffic
            and r["status"] not in ("on_track", "sold_out")
            and ads_pay_for_themselves(r, price[lid], cfg)
        ]
        for _ in range(int(pool // STEP) + 1):
            if pool < STEP or not candidates:
                break
            best, best_val = None, 0.0
            for lid in candidates:
                r = rows[lid]
                cap_l = max(cfg.max_ad_multiplier * float(r["ad_now"]), cfg.min_ad_cap)
                if spend[lid] + STEP > cap_l:
                    continue
                now = project(r, price[lid], spend[lid], None, days_left, cfg, sizes[lid])
                if now.rate >= cfg.raise_until_ratio * float(r["required_rate"]):
                    continue
                val = net_at(lid, spend[lid] + STEP) - now.net
                if val > best_val:
                    best, best_val = lid, val
            if best is None:
                break
            spend[best] += STEP
            pool -= STEP
            if kind[best] == "keep":
                kind[best] = "raise"

        for lid, r in rows.items():
            if kind[lid] == "raise":
                gain = net_at(lid, spend[lid]) - net_at(lid, float(r["ad_now"]))
                reason[lid] = (
                    f"converts at {r['cvr_mean']:.1%} on {r['sessions_7']:.0f} sessions/day; the extra budget is projected "
                    f"to add AED {gain:,.0f} by the deadline, the best use of the channel's remaining budget"
                )
            if kind[lid] == "cap":
                val = float(np.floor(spend[lid] / 5) * 5) if spend[lid] >= 5 else 0.0  # round down, never up
            elif kind[lid] == "keep":
                val = float(r["ad_now"])
            else:
                val = _round5(spend[lid])
            out[lid] = {"spend": val, "reason": reason[lid], "kind": kind[lid]}
        # rounding must not breach the cap
        total = sum(v["spend"] for k, v in out.items() if k.split(":")[1] == ch)
        if total > cap:
            excess = total - cap
            for lid in sorted(
                (k for k in out if k.split(":")[1] == ch and out[k]["spend"] > 0),
                key=lambda k: (out[k]["kind"] != "raise", -out[k]["spend"]),
            ):
                take = min(excess, out[lid]["spend"])
                out[lid]["spend"] -= take
                excess -= take
                if excess <= 0:
                    break
    return out
