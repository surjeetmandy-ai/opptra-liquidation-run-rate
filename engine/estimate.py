"""Per-listing estimates: run-rates, conversion, click cost, elasticity, status.

Fourteen days of thin data, so every rate is shrunk towards its peers (category x channel)
and the shrinkage is visible in the output (cvr_prior, cvr_obs_sessions), not hidden.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import Config
from .load import Dataset
from .state import State


def listing_id(style_id: str, channel: str) -> str:
    """The key used for a listing everywhere: 'S01' + 'amazon_ae' -> 'S01:amazon_ae'."""
    return f"{style_id}:{channel}"


def days_left(plan_date: date, state: State, cfg: Config) -> int:
    """Days remaining in the 60, counting the plan date itself."""
    start = state.campaign_start or plan_date
    return max(0, cfg.horizon_days - (plan_date - start).days)


def _beta_quantiles(a: float, b: float, qs=(0.10, 0.90), n: int = 4000) -> tuple:
    rng = np.random.default_rng(7)
    draws = rng.beta(max(a, 1e-3), max(b, 1e-3), n)
    return tuple(float(np.quantile(draws, q)) for q in qs)


def category_size_curve(inv: pd.DataFrame) -> dict:
    """Demand share by size within a category = share of units bought (units_at_start).
    Merchandisers buy to the expected size curve, so this is the best proxy we have."""
    curves = {}
    for cat, g in inv.groupby("category"):
        s = g.groupby("size_eu")["units_at_start"].sum()
        s = s / s.sum() if s.sum() > 0 else s
        curves[cat] = s.to_dict()
    return curves


def _size_sort_key(s: str):
    try:
        return float(s)
    except ValueError:
        return s


PRICE_MOVE_MIN_AED = 0.5  # any move of half a dirham or more is a price change for the weekly limit; less is rounding
PRICE_MOVE_TOLERANCE = 0.01  # a move under 1% says nothing usable about elasticity
NO_RECENT_CHANGE = 999  # "days since" when there has never been a change


def price_change_dates_from_history(h: pd.DataFrame) -> set:
    """Dates on which the price differs from the day before, read from the history.

    These count toward the 2-changes-a-week limit, together with changes applied in the tool.
    """
    h = h.sort_values("date")
    p = h["price_aed"].values
    d = h["date"].values
    return {d[i] for i in range(1, len(p)) if p[i - 1] > 0 and abs(p[i] - p[i - 1]) >= PRICE_MOVE_MIN_AED}


def merge_change_dates(from_history: set, applied: set) -> set:
    """One set of price-change dates. A change the tool applied on day D shows in the export on
    D (or D+1 if the marketplace re-indexed overnight); count it once, not twice."""
    keep = set(applied)
    for d in from_history:
        if not any(abs((d - a).days) <= 1 for a in applied):
            keep.add(d)
    return keep


def elasticity_posterior(h: pd.DataFrame, cfg: Config, prior_rate: float) -> tuple[float, float]:
    """Normal-normal update of the price elasticity of conversion from price-change events
    seen in this listing's own history. With one change in the data this mostly returns the prior."""
    mu, var = cfg.elasticity_prior_mean, cfg.elasticity_prior_sd**2
    h = h.sort_values("date").reset_index(drop=True)
    changes = [i for i in range(1, len(h)) if h.loc[i, "price_aed"] != h.loc[i - 1, "price_aed"]]
    for i in changes:
        p_old, p_new = float(h.loc[i - 1, "price_aed"]), float(h.loc[i, "price_aed"])
        if p_old <= 0 or p_new <= 0 or abs(p_new / p_old - 1) < PRICE_MOVE_TOLERANCE:
            continue
        before = h.iloc[max(0, i - 7) : i]
        after = h.iloc[i : i + 7]
        before = before[(before["price_aed"] == p_old) & before["sessions_valid"]]
        after = after[(after["price_aed"] == p_new) & after["sessions_valid"]]
        if len(before) < 3 or len(after) < 3:
            continue
        ub, nb = before["units_sold"].sum(), before["sessions"].sum()
        ua, na = after["units_sold"].sum(), after["sessions"].sum()
        if nb <= 0 or na <= 0 or (ub + ua) < 2:
            continue
        k = cfg.cvr_smoothing_sessions  # light smoothing so a zero-sales week doesn't give log(0)
        cvr_b = (ub + prior_rate * k) / (nb + k)
        cvr_a = (ua + prior_rate * k) / (na + k)
        obs = -np.log(cvr_a / cvr_b) / np.log(p_new / p_old)
        obs = float(np.clip(obs, -1.0, 6.0))
        obs_var = (1.0 / (ub + 1) + 1.0 / (ua + 1)) / (np.log(p_new / p_old) ** 2)
        obs_var = max(obs_var, 0.05)
        post_var = 1.0 / (1.0 / var + 1.0 / obs_var)
        mu = post_var * (mu / var + obs / obs_var)
        var = post_var
    mu = float(np.clip(mu, 0.5, 5.0))
    return mu, float(np.sqrt(var))


def _scores_lever(kind: str, lever: str) -> bool:
    """Which applied actions count toward which lever's score. A price action (with or without an ad
    change) scores the price lever: the cut is the bigger lever and the ad change rides on it. Only an
    ad change made on its own scores the ads lever; crediting a combined action to both would score
    the ads lever on the price cut's uplift."""
    if lever == "price":
        return "price" in kind
    return "ads" in kind and "price" not in kind


def lever_calibration(state: State, cfg: Config) -> dict:
    """How much of the uplift the tool promised did its applied actions deliver, per lever?

    For each applied action the log holds the predicted run-rate, the hold baseline and (once
    the data arrives) the actual rate until the listing's next change. Raw weight for a lever =
    sum(actual - baseline) / sum(predicted - baseline) over the actions that promised an uplift
    through that lever, once at least calibration_min_actions of them have 3+ days of actuals.
    With a handful of actions and ~1 sale a day that ratio is noise, so it is pulled toward 1 by
    n / (n + calibration_shrink_actions) and bounded. Below 1 means the tool has been
    over-promising; future uplifts from that lever are scaled by it. Stops and trims are not
    scored: removing paid clicks is mechanical, not a forecast.
    """
    lo, hi = cfg.calibration_bounds
    out = {"price": 1.0, "ads": 1.0}
    for lever in out:
        scored = [
            p
            for p in state.predictions
            if p.get("actual_rate") is not None
            and _scores_lever(p["kind"], lever)
            and p.get("days_observed", 0) >= 3
            and p["predicted_rate"] - p["baseline_rate"] > 0.05
        ]
        n = len(scored)
        if n >= cfg.calibration_min_actions:
            promised = sum(p["predicted_rate"] - p["baseline_rate"] for p in scored)
            delivered = sum(p["actual_rate"] - p["baseline_rate"] for p in scored)
            raw = delivered / promised
            shrunk = 1.0 + (raw - 1.0) * n / (n + cfg.calibration_shrink_actions)
            out[lever] = float(np.clip(shrunk, lo, hi))
    return out


def estimate_listings(ds: Dataset, state: State, cfg: Config, plan_date: date | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (table with one row per listing, size vectors per listing)."""
    plan_date = plan_date or (ds.today + timedelta(days=1))
    calib = lever_calibration(state, cfg)
    horizon = days_left(plan_date, state, cfg)
    inv, hist, cons = ds.inventory, ds.history, ds.constraints.set_index("style_id")
    curves = category_size_curve(inv)

    valid = hist[hist["sessions_valid"]]
    # pooled conversion priors: category x channel, falling back to channel
    cat_of = inv.groupby("style_id")["category"].first()
    valid = valid.assign(category=valid["style_id"].map(cat_of))
    pool_cc = valid.groupby(["category", "channel"]).agg(u=("units_sold", "sum"), n=("sessions", "sum"))
    pool_c = valid.groupby("channel").agg(u=("units_sold", "sum"), n=("sessions", "sum"))
    overall = valid["units_sold"].sum() / max(valid["sessions"].sum(), 1)

    def prior_for(cat, ch):
        if (cat, ch) in pool_cc.index and pool_cc.loc[(cat, ch), "n"] >= cfg.prior_pool_min_sessions:
            r = pool_cc.loc[(cat, ch)]
            return float(r["u"] / r["n"])
        if ch in pool_c.index and pool_c.loc[ch, "n"] >= cfg.prior_pool_min_sessions:
            r = pool_c.loc[ch]
            return float(r["u"] / r["n"])
        return float(overall)

    # click cost: channel level, listing-adjusted
    clicked = hist[hist["ad_clicks"] > 0]
    cpc_ch = (clicked.groupby("channel")["ad_spend_aed"].sum() / clicked.groupby("channel")["ad_clicks"].sum()).to_dict()
    cpc_default = float(clicked["ad_spend_aed"].sum() / max(clicked["ad_clicks"].sum(), 1)) if len(clicked) else 2.0

    rows, sizes = [], {}
    window_start = ds.today - timedelta(days=cfg.run_rate_days - 1)
    last_price = hist.sort_values("date").groupby(["style_id", "channel"])["price_aed"].last().to_dict() if len(hist) else {}
    for (s, ch), g in inv.groupby(["style_id", "channel"]):
        lid = listing_id(s, ch)
        h = hist[(hist["style_id"] == s) & (hist["channel"] == ch)].sort_values("date")
        cat = g["category"].iloc[0]
        name = g["style_name"].iloc[0]
        on_hand = int(g["units_on_hand"].sum())
        cost = float(g["unit_cost_aed"].iloc[0])
        mrp = float(cons.loc[s, "mrp_aed"]) if s in cons.index else np.nan
        floor = float(cons.loc[s, "liquidation_floor_aed"]) if s in cons.index else np.nan

        last7 = h[h["date"] >= window_start]
        rr7 = float(last7["units_sold"].mean()) if len(last7) else 0.0
        rr14 = float(h["units_sold"].mean()) if len(h) else 0.0
        sold_window = int(h["units_sold"].sum()) if len(h) else 0
        is_new = len(h) < 3
        price_source = "history"
        if len(h):
            price_ref = float(h["price_aed"].iloc[-1])
        else:
            # a listing with no history has no observed price: take the other channel's (parity) or MRP
            other = [v for (ss, cc), v in last_price.items() if ss == s and cc != ch]
            price_ref, price_source = (
                (float(other[0]), "other channel") if other else ((mrp, "MRP") if not np.isnan(mrp) else (np.nan, "none"))
            )
        spend_7 = float(last7["ad_spend_aed"].mean()) if len(last7) else 0.0
        spend_last = float(h["ad_spend_aed"].iloc[-1]) if len(h) else 0.0  # ads are a setting: the latest day is the current one
        clicks_7 = float(last7["ad_clicks"].mean()) if len(last7) else 0.0

        # planned/applied overrides that the data has not caught up with yet
        price_now = price_ref
        rec = state.applied_prices.get(lid)
        if rec and date.fromisoformat(rec["date"]) > ds.today:
            price_now = float(rec["price"])
        ad_now = spend_last
        rec = state.applied_ads.get(lid)
        if rec and date.fromisoformat(rec["date"]) > ds.today:
            ad_now = float(rec["spend"])
        # one decision per listing per day: anything applied for the plan date locks the listing
        rec_p = state.applied_prices.get(lid)
        exited_today = any(e["listing_id"] == lid and e["date"] == plan_date.isoformat() for e in state.exits)
        locked = bool(
            (rec and date.fromisoformat(rec["date"]) == plan_date)
            or (rec_p and date.fromisoformat(rec_p["date"]) == plan_date)
            or exited_today
        )
        # how long since the ad budget last moved (applied or visible in the data)
        spends = h["ad_spend_aed"].round(0).values if len(h) else np.array([])
        # day-to-day wobble of a few percent is pacing noise, not a budget change
        ad_change_days = [
            h["date"].values[i]
            for i in range(1, len(spends))
            if abs(spends[i] - spends[i - 1]) >= max(10.0, 0.15 * max(spends[i - 1], 1.0))
        ]
        if rec:
            ad_change_days.append(date.fromisoformat(rec["date"]))
        past_ads = [d for d in ad_change_days if d <= ds.today]
        days_at_ads = (ds.today - max(past_ads)).days + 1 if past_ads else NO_RECENT_CHANGE

        v7 = last7[last7["sessions_valid"]]
        vall = h[h["sessions_valid"]]
        src = v7 if len(v7) >= 3 else vall
        sessions_7 = float(src["sessions"].mean()) if len(src) else 0.0
        organic_7 = float((src["sessions"] - src["ad_clicks"]).clip(lower=0).mean()) if len(src) else 0.0
        invalid_days = int((~h["sessions_valid"]).sum())

        # conversion at the reference price, from valid rows at that price. Recent days count
        # in full, older days half: last week is a better guide to next week than the week before.
        at_ref = vall[vall["price_aed"] == price_ref]
        w = np.where(at_ref["date"] >= window_start, 1.0, 0.5)
        units_obs, sess_obs = float((at_ref["units_sold"] * w).sum()), float((at_ref["sessions"] * w).sum())
        prior = prior_for(cat, ch)
        a0, b0 = prior * cfg.cvr_prior_strength, (1 - prior) * cfg.cvr_prior_strength
        a, b = a0 + units_obs, b0 + max(sess_obs - units_obs, 0)
        cvr_mean = a / (a + b)
        cvr_p10, cvr_p90 = _beta_quantiles(a, b)

        # click cost
        lc, ls = float(h["ad_clicks"].sum()), float(h["ad_spend_aed"].sum())
        k = cfg.cpc_prior_clicks
        cpc = (ls + k * cpc_ch.get(ch, cpc_default)) / (lc + k)

        e_mu, e_sd = elasticity_posterior(h, cfg, prior) if len(h) else (cfg.elasticity_prior_mean, cfg.elasticity_prior_sd)

        # price-change slots in the rolling window ending on the plan date
        change_dates = merge_change_dates(price_change_dates_from_history(h), state.price_change_dates(lid))
        # window = the 7 days ending on the plan date, so a change already applied for that
        # date counts and a second one the same day is refused
        lo, hi = plan_date - timedelta(days=cfg.change_window_days - 1), plan_date
        n_changes = sum(1 for d in change_dates if lo <= d <= hi)
        slots = max(0, cfg.max_price_changes - n_changes)
        # how long the listing has been at its current price: one day of data after a cut is noise
        past = [d for d in change_dates if d <= ds.today]
        days_at_price = (ds.today - max(past)).days + 1 if past else NO_RECENT_CHANGE
        if locked:
            days_at_price = 0

        required = on_hand / horizon if horizon > 0 else float("inf")
        ratio = rr7 / required if required > 0 else float("inf")
        status = "on_track" if ratio >= 1.0 else ("behind" if ratio >= 0.5 else "critical")
        if on_hand == 0:
            status = "sold_out"

        # sizes
        gs = g.copy()
        gs["_k"] = gs["size_eu"].map(_size_sort_key)
        gs = gs.sort_values("_k")
        curve = curves.get(cat, {})
        shares = np.array([curve.get(sz, 0.0) for sz in gs["size_eu"]], dtype=float)
        if shares.sum() <= 0:
            shares = np.ones(len(gs)) / max(len(gs), 1)
        shares = shares / shares.sum()
        stock_vec = gs["units_on_hand"].astype(float).values
        sizes[lid] = {"sizes": gs["size_eu"].tolist(), "on_hand": stock_vec.tolist(), "demand_share": shares.tolist()}
        core = shares >= np.median(shares)
        core_stock_share = float(stock_vec[core].sum() / stock_vec.sum()) if stock_vec.sum() > 0 else 1.0
        broken_run = bool(on_hand > 0 and core_stock_share < cfg.broken_run_core_share)

        discount = float(1 - price_now / mrp) if mrp and mrp > 0 else np.nan
        at_floor = bool(not np.isnan(floor) and price_now <= floor + 1e-9)
        ad_per_unit = (spend_7 / rr7) if rr7 > 0 else (np.inf if spend_7 > 0 else 0.0)

        notes = []
        if is_new:
            notes.append(
                f"new listing: {len(h)} day(s) of history; conversion borrowed from category peers"
                + (f"; price taken from {price_source}" if price_source != "history" else "")
            )
        if invalid_days:
            notes.append(f"{invalid_days} day(s) with missing sessions excluded from conversion")
        if broken_run:
            notes.append(f"broken size run: core sizes hold {core_stock_share:.0%} of stock")
        if at_floor:
            notes.append("already at the price floor")
        listing_problem = bool(
            not np.isnan(discount) and discount >= cfg.listing_problem_discount and cvr_mean < cfg.listing_problem_cvr
        )
        if listing_problem:
            notes.append("near-zero conversion at a deep discount: check the listing, not the price")
        # a listing nobody has seen in two weeks is not live, whatever its price; ads on prior conversion would be a guess
        no_traffic = bool(not is_new and len(vall) >= 7 and float(vall["sessions"].sum()) == 0 and on_hand > 0)
        if no_traffic:
            listing_problem = True
            notes.append(f"no sessions at all in {len(vall)} days: is the listing live? check it before spending anything")
        # a human check requested in the tool: how many days ago (-1 = never)
        checked_days = (plan_date - date.fromisoformat(state.checks[lid])).days if lid in state.checks else -1

        rows.append(
            dict(
                listing_id=lid,
                style_id=s,
                channel=ch,
                style_name=name,
                category=cat,
                on_hand=on_hand,
                jobber_value=cost * cfg.jobber_pct,
                mrp=mrp,
                floor=floor,
                price_ref=price_ref,
                price_now=price_now,
                discount=discount,
                at_floor=at_floor,
                is_new=is_new,
                ad_now=ad_now,
                spend_7=spend_7,
                spend_last=spend_last,
                clicks_7=clicks_7,
                cpc=cpc,
                organic_7=organic_7,
                sessions_7=sessions_7,
                locked=locked,
                exited_today=exited_today,
                days_at_price=days_at_price,
                days_at_ads=days_at_ads,
                calib_price=calib.get("price", 1.0),
                calib_ads=calib.get("ads", 1.0),
                rr7=rr7,
                rr14=rr14,
                sold_window=sold_window,
                ad_per_unit=ad_per_unit,
                cvr_mean=cvr_mean,
                cvr_p10=cvr_p10,
                cvr_p90=cvr_p90,
                cvr_prior=prior,
                cvr_obs_sessions=sess_obs,
                elasticity=e_mu,
                elasticity_sd=e_sd,
                required_rate=required,
                ratio=ratio,
                status=status,
                changes_in_window=n_changes,
                slots=slots,
                core_stock_share=core_stock_share,
                broken_run=broken_run,
                notes=notes,
                listing_problem=listing_problem,
                checked_days=checked_days,
            )
        )
    table = pd.DataFrame(rows)
    table.attrs["plan_date"] = plan_date
    table.attrs["days_left"] = horizon
    table.attrs["calibration"] = calib
    return table, sizes
