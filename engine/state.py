"""What has been applied so far. A JSON file, because this is a prototype and one brand
manager uses it. Everything the rules need to count (price changes, exits) is here, so
tomorrow's guardrail runs against what was actually done, not what was suggested.

A pydantic model because it is read back from disk: a damaged or hand-edited file is
reported on load, not discovered three functions later. Each record stays a plain dict (a
TypedDict is validated but not converted), so the code that reads them is unchanged.

If the file can't be read, loading fails loudly rather than starting fresh: a fresh start
would forget applied price changes, and the tool could then break the 2-changes-a-week rule.
"""

from datetime import date
from pathlib import Path
from typing import NotRequired, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import TypedDict  # pydantic needs typing_extensions' TypedDict on Python < 3.12


class PriceRec(TypedDict):
    """A price applied to a listing, and the date it takes effect."""

    price: float
    date: str


class AdsRec(TypedDict):
    """A daily ad budget applied to a listing, and the date it takes effect."""

    spend: float
    date: str


ChangeRec = TypedDict("ChangeRec", {"listing_id": str, "date": str, "from": float, "to": float, "forced_by_parity": bool})


class ExitRec(TypedDict):
    """Pairs sent to the jobber from one listing: when, how many, what they fetched, and the split by size when known."""

    listing_id: str
    date: str
    units: int
    value_aed: float
    by_size: NotRequired[list[int] | None]
    sizes: NotRequired[list[str] | None]


class PredictionRec(TypedDict):
    """What an applied action was expected to do (units/day) versus holding, and what actually happened once the
    data arrived.
    """

    date: str
    listing_id: str
    kind: str
    predicted_rate: float
    baseline_rate: float
    actual_rate: float | None
    days_observed: NotRequired[int]


def _d(s) -> date | None:
    return date.fromisoformat(s) if isinstance(s, str) else s


class State(BaseModel):
    """Everything applied so far: prices, ad budgets, the price-change log, exits, predictions and saved setting
    overrides. Saved to state.json after every change.
    """

    model_config = ConfigDict(extra="ignore")

    campaign_start: date | None = None  # day 1 of the 60 (the day after the first "today")
    applied_prices: dict[str, PriceRec] = Field(default_factory=dict)
    applied_ads: dict[str, AdsRec] = Field(default_factory=dict)
    change_log: list[ChangeRec] = Field(default_factory=list)
    exits: list[ExitRec] = Field(default_factory=list)
    checks: dict[str, str] = Field(default_factory=dict)  # listing_id -> date a human check of the listing was requested
    predictions: list[PredictionRec] = Field(default_factory=list)
    config_overrides: dict = Field(default_factory=dict)

    # ---- persistence ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> Self:
        """Read state.json; a missing file means a fresh start.

        Raises ValueError with a readable message if the file is damaged.
        """
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            return cls.model_validate_json(path.read_text())
        except ValidationError as e:
            first = e.errors()[0]
            where = ".".join(map(str, first["loc"]))
            raise ValueError(
                f"{path.name} can't be read ({e.error_count()} problem(s); first: {where}: {first['msg']}). "
                f"Fix it, or delete it to start again (that forgets every applied action)."
            ) from e

    def save(self, path: Path) -> None:
        """Write the state to state.json."""
        Path(path).write_text(self.model_dump_json(indent=2))

    # ---- queries ------------------------------------------------------------------------
    def exited_value(self) -> float:
        """AED received from the jobber so far, across all listings."""
        return float(sum(e["value_aed"] for e in self.exits))

    def price_change_dates(self, listing_id: str) -> set:
        """Dates of price changes applied in the tool for this listing, parity-forced ones included."""
        return {_d(c["date"]) for c in self.change_log if c["listing_id"] == listing_id}

    # ---- mutations ----------------------------------------------------------------------
    def apply_price(self, listing_id: str, old: float, new: float, on: date, forced: bool = False) -> None:
        """Record a new price from date `on`.

        A change is logged (and counts against the weekly limit) only if the price actually moved.
        """
        self.applied_prices[listing_id] = {"price": float(new), "date": on.isoformat()}
        if float(old) != float(new):
            self.change_log.append(
                {
                    "listing_id": listing_id,
                    "date": on.isoformat(),
                    "from": float(old),
                    "to": float(new),
                    "forced_by_parity": bool(forced),
                }
            )

    def apply_ads(self, listing_id: str, new: float, on: date) -> None:
        """Record a new daily ad budget from date `on`.

        The rules don't limit ad changes; the engine adds its own 3-day cooldown.
        """
        self.applied_ads[listing_id] = {"spend": float(new), "date": on.isoformat()}

    def apply_exit(
        self,
        listing_id: str,
        units: int,
        value_aed: float,
        on: date,
        by_size: list[int] | None = None,
        sizes: list[str] | None = None,
    ) -> None:
        """Record pairs sent to the jobber on date `on`, what they fetched, and the split by size when known."""
        self.exits.append(
            {
                "listing_id": listing_id,
                "date": on.isoformat(),
                "units": int(units),
                "value_aed": float(value_aed),
                "by_size": by_size,
                "sizes": sizes,
            }
        )

    def record_check(self, listing_id: str, on: date) -> None:
        """Note that a human was asked to check the listing (content, size chart, buy box) on date `on`.
        The first request is the one that counts; the exit of a broken listing waits for it."""
        self.checks.setdefault(listing_id, on.isoformat())

    def record_prediction(self, on: date, listing_id: str, kind: str, predicted_rate: float, baseline_rate: float) -> None:
        """Log what an applied action is expected to do (units/day) next to the hold baseline.

        The actual rate is filled in when the next days' data arrives (see apply.score_predictions).
        """
        self.predictions.append(
            {
                "date": on.isoformat(),
                "listing_id": listing_id,
                "kind": kind,
                "predicted_rate": float(predicted_rate),
                "baseline_rate": float(baseline_rate),
                "actual_rate": None,
            }
        )
