"""A recommendation: the one object every part of the tool passes around.

Built by decide.py, checked by guardrail.py, applied by apply.py, described by llm.py and
shown by app.py. It lives in its own module because it has no single owner, and because
defining it next to the code that builds it (decide.py) would make decide and guardrail
import each other.
"""

from dataclasses import dataclass, field, replace

from .schema import CHANNEL_LABELS


@dataclass
class Action:
    """One recommendation for one listing: what to change (price, ads, exit), by how much, why, and what it is
    worth versus holding. kind 'hold' or 'watch' means no change.
    """

    listing_id: str
    style_id: str
    channel: str
    style_name: str
    kind: str  # "price" | "ads" | "exit" | "check" (joined with "+" when combined) | "hold" | "watch"
    price_from: float | None = None
    price_to: float | None = None
    ads_from: float | None = None
    ads_to: float | None = None
    exit_units: int = 0
    exit_by_size: list | None = None
    exit_value: float = 0.0
    forced_by_parity: bool = False
    impact_aed: float = 0.0  # net recovery under plan minus under hold, this listing
    reason: str = ""
    flags: list = field(default_factory=list)
    predicted_rate: float = 0.0
    baseline_rate: float = 0.0
    rank: int = 0
    must_do: bool = False  # fixes a rule already broken in the data; shown first whatever its AED impact
    check: bool = False  # asks a human to check the listing (content, size chart, buy box); applying it records the date

    @property
    def is_change(self) -> bool:
        """True if applying this action changes anything."""
        return self.kind not in ("hold", "watch")

    def headline(self) -> str:
        """The one line Farah reads, e.g. 'S15 Breeze Runner · Amazon.ae: cut price AED 289 → 234; stop ads (was AED 64/day)'."""
        ch = CHANNEL_LABELS.get(self.channel, self.channel)
        parts = []
        if self.price_to is not None and self.price_from is not None and self.price_to != self.price_from:
            verb = "raise" if self.price_to > self.price_from else "cut"
            parts.append(
                f"{verb} price AED {self.price_from:.0f} → {self.price_to:.0f}"
                + (" (parity)" if self.forced_by_parity else "")
                + (" (rule fix)" if self.must_do else "")
            )
        if self.ads_to is not None and self.ads_from is not None and abs(self.ads_to - self.ads_from) >= 1:
            if self.ads_to == 0:
                parts.append(f"stop ads (was AED {self.ads_from:.0f}/day)")
            else:
                verb = "raise" if self.ads_to > self.ads_from else "cut"
                parts.append(f"{verb} ads AED {self.ads_from:.0f} → {self.ads_to:.0f}/day")
        if self.exit_units > 0:
            parts.append(f"exit {self.exit_units} pairs to jobber (AED {self.exit_value:,.0f})")
        if self.check:
            parts.append("check the listing (images, size chart, buy box)")
        if not parts:
            parts.append("hold" if self.kind == "hold" else "watch")
        return f"{self.style_id} {self.style_name} · {ch}: " + "; ".join(parts)


def price_only(a: Action) -> Action:
    """A copy of an action reduced to its price move, for applying a parity partner's price without its ad
    change, exit or check. Its predicted rate equals the baseline, so the partial move is not scored as if
    it were the whole action."""
    return replace(
        a,
        kind="price",
        ads_to=a.ads_from,
        exit_units=0,
        exit_by_size=None,
        exit_value=0.0,
        check=False,
        predicted_rate=a.baseline_rate,
    )
