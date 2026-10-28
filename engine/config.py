"""All the knobs in one place. Every assumption in assumptions.md that is a number
lives here, so the live-session "what if we change X" is a slider, not a code change.

A pydantic model rather than a dataclass because values arrive from outside the code (the
sidebar, and overrides saved in state.json): each one is range-checked on the way in, and
again on every assignment.
"""

import logging
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .schema import AMAZON, NOON

log = logging.getLogger(__name__)


class Config(BaseModel):
    """Business rules from the brief and the tunable numbers behind every decision.

    Range-checked when created and on every assignment.
    """

    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    # --- business rules from the brief -------------------------------------------------
    horizon_days: int = Field(60, ge=1, le=365)  # deadline: 60 days starting the day after "today"
    jobber_pct: float = Field(0.25, ge=0, le=1)  # jobber pays 25% of unit cost, any day, any quantity
    holding_per_unit_per_day: float = Field(3.0 / 30, ge=0)  # AED 3 per unit per month
    ad_budget: dict[str, float] = Field(default_factory=lambda: {AMAZON: 1500.0, NOON: 1000.0})
    parity_band: float = Field(0.05, gt=0, le=0.5)  # Noon within ±5% of Amazon.ae
    max_price_changes: int = Field(2, ge=0, le=7)  # per listing ...
    change_window_days: int = Field(7, ge=1, le=31)  # ... in any rolling 7 days

    # --- things the brief does not say (see assumptions.md §2) ---------------------------
    marketplace_fee_pct: float = Field(0.0, ge=0, lt=1)  # referral/fulfilment fees; 0 because the brief has none
    max_actions: int = Field(8, ge=1)  # what Farah sees as "do today"; the rest is folded away

    # --- when a change is worth making ----------------------------------------------------
    # Hold is the default. A move has to beat holding by these amounts (projected net recovery,
    # AED, over the days left) before it is proposed.
    price_min_gain_aed: float = Field(300.0, ge=0)  # per listing whose price moves; a change burns a scarce weekly slot
    ads_min_gain_aed: float = Field(150.0, ge=0)  # for an ad raise or trim
    exit_min_gain_aed: float = Field(100.0, ge=0)  # for sending pairs to the jobber now rather than on day 60
    ads_min_delta_aed: float = Field(20.0, ge=0)  # ad changes smaller than this are fiddling, not decisions
    cooldown_days: int = Field(3, ge=0)  # after moving a price or a budget, read the response before moving it again

    # --- price moves ------------------------------------------------------------------------
    price_step: int = Field(5, ge=1)  # cuts move in AED 5 steps
    min_cut_pct: float = Field(0.05, ge=0, le=1)  # cuts below this are noise to the buyer and a waste of a slot
    default_cut_pct: float = Field(0.10, gt=0, le=1)  # first move on a listing that is behind: a measured experiment
    max_cut_pct: float = Field(0.20, gt=0, le=0.9)  # no single cut deeper than this; evidence first

    # --- ad moves ----------------------------------------------------------------------------
    max_ad_multiplier: float = Field(2.5, ge=1)  # a listing's ads can't jump beyond 2.5x current in a day
    min_ad_cap: float = Field(60.0, ge=0)  # ... but can always be raised to at least this (AED/day)
    safety_margin_aed: float = Field(5.0, ge=0)  # an ad-driven sale must beat jobber value by this much
    raise_until_ratio: float = Field(1.2, ge=1)  # stop funding a listing once it is projected this far ahead
    listing_check_days: int = Field(3, ge=0)  # a broken listing gets a human check; its exit waits this long for the fix
    calibration_shrink_actions: int = Field(8, ge=0)  # a lever's measured weight is pulled toward 1 until this many actions

    # --- estimation ---------------------------------------------------------------------
    run_rate_days: int = Field(7, ge=1)  # trailing window for the actual run-rate
    cvr_prior_strength: float = Field(150.0, gt=0)  # the category x channel prior is worth this many sessions
    prior_pool_min_sessions: int = Field(500, ge=1)  # a category x channel pool thinner than this falls back to the channel
    cvr_smoothing_sessions: float = Field(20.0, gt=0)  # smoothing when reading elasticity from a price change
    cpc_prior_clicks: float = Field(50.0, gt=0)  # the channel's click cost is worth this many clicks per listing
    elasticity_prior_mean: float = Field(2.0, gt=0)
    elasticity_prior_sd: float = Field(0.6, gt=0)
    diminishing_threshold: float = Field(1.5, ge=0)  # ad clicks beyond 1.5x current spend ...
    diminishing_factor: float = Field(0.7, ge=0, le=1)  # ... are worth 70% as much
    listing_problem_cvr: float = Field(0.005, ge=0, le=1)  # conversion below 0.5% ...
    listing_problem_discount: float = Field(0.40, ge=0, le=1)  # ... at >= 40% off MRP is a listing problem, not a price problem
    broken_run_core_share: float = Field(0.30, ge=0, le=1)  # core sizes holding less than this of the stock = broken size run
    exit_min_units: int = Field(10, ge=0)  # don't bother the jobber with fewer than this
    exit_min_share: float = Field(0.10, ge=0, le=1)  # ... or less than 10% of a listing's stock
    sell_through_target: float = Field(0.95, gt=0, le=1)  # "make it" = at least this share sold on marketplaces
    calibration_min_actions: int = Field(4, ge=1)  # applied actions with 3+ days of actuals before a lever is re-weighted
    calibration_bounds: tuple[float, float] = (0.5, 1.5)  # a lever's weight never leaves this range

    @field_validator("ad_budget")
    @classmethod
    def _budgets_not_negative(cls, v: dict[str, float]) -> dict[str, float]:
        bad = {k: x for k, x in v.items() if x < 0}
        if bad:
            raise ValueError(f"ad budgets can't be negative: {bad}")
        return v

    def to_dict(self) -> dict:
        """Every value as a plain dict, for saving as overrides in state.json."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Saved overrides: keep every value that is valid, drop (and log) any that isn't, so one
        bad value in state.json doesn't throw away the rest."""
        d = dict(d or {})
        try:
            return cls.model_validate(d)
        except ValidationError as e:
            bad = {str(err["loc"][0]) for err in e.errors() if err["loc"]}
            log.warning("ignoring invalid saved settings %s: %s", sorted(bad), e.errors()[0]["msg"])
            return cls.model_validate({k: v for k, v in d.items() if k not in bad})
