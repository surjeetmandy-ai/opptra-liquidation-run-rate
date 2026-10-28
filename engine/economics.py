"""The money. One projection function, used for every candidate action and for the verdict,
so "impact" always means the same thing: net recovery under the plan minus net recovery
under hold, over the days left.

Net recovery = marketplace revenue x (1 - fee) - ad spend - holding cost
               + jobber proceeds for exits now + jobber proceeds for what is left at the deadline.
"""

from dataclasses import dataclass

import numpy as np

from .config import Config


def conversion_at(row, price: float, cvr: float | None = None, elasticity: float | None = None) -> float:
    """Conversion at a price: conversion at the reference price x (price / reference) ^ (-elasticity).

    The one place this formula lives; every cost-per-sale and every projection goes through it.
    """
    cvr = float(row["cvr_mean"] if cvr is None else cvr)
    e = float(row["elasticity"] if elasticity is None else elasticity)
    p_ref = float(row["price_ref"]) if row["price_ref"] and not np.isnan(row["price_ref"]) else price
    out = cvr * (price / p_ref) ** (-e) if p_ref > 0 and price > 0 else cvr
    return float(np.clip(out, 0.0, 0.5))


def _clicks(row, ad_spend: float, cfg: Config) -> float:
    """Ad clicks bought by a daily spend, with diminishing returns past 1.5x what the listing runs today.

    A listing with no ads today gets its first AED min_ad_cap at full value.
    """
    cpc = max(float(row["cpc"]), 0.1)
    base_spend = float(row["ad_now"])
    threshold = cfg.diminishing_threshold * base_spend if base_spend > 0 else cfg.min_ad_cap
    if ad_spend <= threshold:
        return ad_spend / cpc
    return threshold / cpc + (ad_spend - threshold) / cpc * cfg.diminishing_factor


def expected_rate(
    row, price: float, ad_spend: float, cfg: Config, cvr: float | None = None, elasticity: float | None = None
) -> float:
    """Expected units/day for a listing at a given price and daily ad spend.

    sessions = organic + ad clicks; units = sessions x conversion at that price. The learning
    log's weights (calib_ads, calib_price, 1.0 until the tool has scored enough of its own
    actions) scale the *uplift* a raise or a cut is credited with, never the baseline.
    """
    base_spend = float(row["ad_now"])
    clicks = _clicks(row, ad_spend, cfg)
    if ad_spend > base_spend:
        clicks_base = _clicks(row, base_spend, cfg)
        clicks = clicks_base + (clicks - clicks_base) * float(row.get("calib_ads", 1.0))
    sessions = float(row["organic_7"]) + clicks
    rate = sessions * conversion_at(row, price, cvr, elasticity)
    p_ref = float(row["price_ref"]) if row["price_ref"] and not np.isnan(row["price_ref"]) else price
    if price < p_ref:
        rate_at_ref = sessions * conversion_at(row, p_ref, cvr, elasticity)
        rate = rate_at_ref + (rate - rate_at_ref) * float(row.get("calib_price", 1.0))
    return max(0.0, float(rate))


@dataclass
class Projection:
    """Expected outcome for one listing over the days left, at a given price, daily ad spend and exit.

    Units are expected values (fractional); money is AED.
    """

    rate: float  # units/day at the start of the plan
    sold: float  # units expected to sell on the marketplace by the deadline
    unsold: float  # units expected to be left at the deadline (go to jobber then)
    unsold_by_size: list  # same, by size
    exit_units: int  # units sent to the jobber now
    net: float  # revenue x (1 - fee) - ad spend - storage + jobber value (exits now + leftovers at the deadline)


def project(
    row,
    price: float,
    ad_spend: float,
    exit_by_size: list | None,
    days: int,
    cfg: Config,
    size_info: dict | None = None,
    cvr: float | None = None,
    elasticity: float | None = None,
) -> Projection:
    """Expected outcome over `days` with the size curve, so a listing whose core sizes are gone
    slows down as it should. Fractional units throughout (expected values)."""
    sizes = size_info or {"on_hand": [float(row["on_hand"])], "demand_share": [1.0]}
    stock = np.array(sizes["on_hand"], dtype=float)
    share = np.array(sizes["demand_share"], dtype=float)
    exits = np.zeros_like(stock) if exit_by_size is None else np.minimum(np.array(exit_by_size, dtype=float), stock)
    stock = stock - exits
    exit_units = int(round(exits.sum()))
    jobber_value = float(row["jobber_value"])
    fee = cfg.marketplace_fee_pct

    rate = expected_rate(row, price, ad_spend, cfg, cvr=cvr, elasticity=elasticity)
    # Each in-stock size sells at a constant daily rate (its share of demand) until it runs out;
    # demand for a size that is gone is lost, not moved to other sizes. So the 60 days have a
    # closed form per size, no day loop needed.
    has = stock > 1e-9
    full_share = share[has].sum() if has.any() else 0.0
    r = np.where(has, rate * share / full_share, 0.0) if full_share > 0 else np.zeros_like(stock)
    sold_i = np.minimum(stock, r * days)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_out = np.where(r > 0, stock / r, np.inf)  # day each size runs out
    m = np.minimum(days, np.floor(t_out + 1e-9))  # days with stock left at day end
    m = np.where(np.isfinite(m), m, days)
    held = np.where(r > 0, m * stock - r * m * (m + 1) / 2, stock * days)
    holding = float(np.clip(held, 0, None).sum() * cfg.holding_per_unit_per_day)
    if not has.any():
        sellout_day = (
            0.0  # nothing to sell: the marketplace stops serving the ads (a budget still set is hygiene, see should_stop_ads)
        )
    elif np.all(r[has] > 0) and np.max(t_out[has]) <= days + 1e-9:
        sellout_day = float(np.ceil(np.max(t_out[has]) - 1e-9))
    else:
        sellout_day = float(days)
    ad_cost = float(ad_spend * sellout_day)
    sold = float(sold_i.sum())
    stock = stock - sold_i
    unsold = float(stock.sum())
    revenue = sold * price
    net = revenue * (1 - fee) - ad_cost - holding + exit_units * jobber_value + unsold * jobber_value
    return Projection(
        rate=rate,
        sold=float(sold),
        unsold=unsold,
        unsold_by_size=stock.tolist(),
        exit_units=exit_units,
        net=float(net),
    )


def marginal_ad_unit_cost(row, price: float) -> float:
    """What one more marketplace sale bought with ads costs, at this price: CPC / conversion."""
    return float(row["cpc"]) / max(conversion_at(row, price), 1e-4)


def ads_pay_for_themselves(row, price: float, cfg: Config) -> bool:
    """An ad-driven sale must earn more than the jobber would pay, with a safety margin."""
    return (
        price * (1 - cfg.marketplace_fee_pct) - marginal_ad_unit_cost(row, price)
        > float(row["jobber_value"]) + cfg.safety_margin_aed
    )


AD_STEP = 10.0  # AED; the grain at which ad spend is valued and allocated


def best_spend_up_to(row, price: float, cfg: Config, days: int, size_info: dict | None = None) -> tuple[float, float]:
    """The daily spend between zero and today's that the projection values highest at this price,
    and its net. Today's spend wins a tie, so the answer only moves when moving pays.

    Grid of AD_STEP from zero up to today's spend (today's exact figure included)."""
    spend = float(row["ad_now"])
    grid = sorted({0.0, spend} | {float(x) for x in np.arange(AD_STEP, spend, AD_STEP)})
    best, best_net = spend, project(row, price, spend, None, days, cfg, size_info).net
    for s in grid:
        if s == spend:
            continue
        net = project(row, price, s, None, days, cfg, size_info).net
        if net > best_net + 1e-9:
            best, best_net = s, net
    return best, best_net


def should_stop_ads(row, price: float, cfg: Config, days: int, size_info: dict | None = None) -> bool:
    """The one rule for switching a listing's ads off: it is spending, and either there is nothing
    left to sell, or the listing is broken (near-zero conversion at a deep discount), or the
    deadline has passed, or no spend between zero and today's beats zero by the plan's own
    projection and stopping is worth at least ads_min_gain_aed.

    Judged by project(), the same arithmetic that values every other move, so a stop can never
    contradict the impact shown next to it. The per-sale cost (marginal_ad_unit_cost) is one of
    the explanations Farah reads, not the test."""
    spend = float(row["ad_now"])
    if spend <= 0:
        return False
    if float(row["on_hand"]) <= 0 or bool(row["listing_problem"]) or days <= 0:
        return True
    best, best_net = best_spend_up_to(row, price, cfg, days, size_info)
    if best > 0:
        return False
    return best_net - project(row, price, spend, None, days, cfg, size_info).net >= cfg.ads_min_gain_aed
