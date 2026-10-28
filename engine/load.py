"""Read the three CSVs (and any new day's file), normalise them, and say what is wrong with them.

Marketplace exports are never quite the same twice, so column names are matched loosely,
blanks are coerced, duplicates are dropped, and impossible rows (sessions below ad clicks)
are kept for their orders but not trusted for their sessions.
"""

import io
import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .schema import CHANNEL_ALIASES, COLUMN_SYNONYMS, CONSTRAINTS, FILES, HISTORY, INVENTORY, FileSchema


def _tidy(name: str) -> str:
    """'Units Sold ' -> 'units_sold'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


# tidied name -> our column name: each column matches itself and its synonyms
_COLUMN_LOOKUP = {_tidy(s): canon for canon, syns in COLUMN_SYNONYMS.items() for s in [canon, *syns]}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to our names where they match (after tidying); unknown columns are kept."""
    out = df.copy()
    out.columns = [_COLUMN_LOOKUP.get(_tidy(c), _tidy(c)) for c in df.columns]
    return out


def normalise_channel(s: pd.Series) -> pd.Series:
    """Map however an export writes the marketplace ('Amazon.ae', 'amazon', 'Noon.com') to our id.

    Unknown values pass through unchanged.
    """
    key = s.astype(str).str.strip().str.lower().str.replace(" ", "").str.replace("-", "_")
    return key.map(lambda k: CHANNEL_ALIASES.get(k, CHANNEL_ALIASES.get(k.replace(".", "_"), k)))


def identify_kind(df: pd.DataFrame) -> str | None:
    """Which of the three files is this? Decided by the columns present."""
    cols = set(normalise_columns(df).columns)
    return next((f.kind for f in FILES if f.signature <= cols), None)


def _conform(df: pd.DataFrame, schema: FileSchema) -> pd.DataFrame:
    """The boundary check: our column names, required columns present, numbers as numbers."""
    df = normalise_columns(df)
    missing = [c for c in schema.required if c not in df.columns]
    if missing:
        raise ValueError(f"{schema.name} is missing columns: {missing} (found: {list(df.columns)})")
    df = df[[c for c in schema.columns if c in df.columns]].copy()
    for c in schema.numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@dataclass
class Dataset:
    """The three files after cleaning, plus 'today' (the last date in the history), the data-quality notes shown on
    the page, the rows with impossible sessions, and units already sent to the jobber.
    """

    inventory: pd.DataFrame
    history: pd.DataFrame
    constraints: pd.DataFrame
    today: date
    issues: list = field(default_factory=list)  # human-readable data-quality notes
    anomalies: pd.DataFrame = field(default_factory=pd.DataFrame)
    exited: dict = field(default_factory=dict)  # listing_id -> units sent to the jobber (not sales)


# -------------------------------------------------------------------------------------------
# cleaning
# -------------------------------------------------------------------------------------------
def clean_history(df: pd.DataFrame) -> tuple[pd.DataFrame, list, pd.DataFrame]:
    """Check and clean the daily history. Returns (history, notes, anomalies).

    Blank ad cells become 0, duplicate days keep the last row, negatives are clipped. Rows whose sessions are
    impossible (below ad clicks, or zero with orders) keep their orders but get sessions_valid=False, so conversion
    ignores them.
    """
    df = _conform(df, HISTORY)
    issues: list[str] = []

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        issues.append(f"{bad_dates} history rows had unreadable dates and were dropped")
        df = df[df["date"].notna()]
    df["style_id"] = df["style_id"].astype(str).str.strip().str.upper()
    df["channel"] = normalise_channel(df["channel"])
    blanks = int(df[["ad_spend_aed", "ad_clicks"]].isna().sum().sum())
    if blanks:
        issues.append(f"{blanks} blank ad spend/click cells read as 0 (exports leave them blank on no-ad days)")
    df["ad_spend_aed"] = df["ad_spend_aed"].fillna(0.0)
    df["ad_clicks"] = df["ad_clicks"].fillna(0)
    df["units_sold"] = df["units_sold"].fillna(0)
    df["sessions"] = df["sessions"].fillna(0)
    df["price_aed"] = df.groupby(["style_id", "channel"])["price_aed"].transform(lambda s: s.ffill().bfill())

    dups = df.duplicated(subset=["date", "style_id", "channel"], keep="last").sum()
    if dups:
        issues.append(f"{dups} duplicate (date, style, channel) rows dropped, last occurrence kept")
        df = df.drop_duplicates(subset=["date", "style_id", "channel"], keep="last")

    neg = (df[list(HISTORY.numeric)] < 0).any(axis=1)
    if neg.any():
        issues.append(f"{int(neg.sum())} rows with negative values clipped to 0")
        df.loc[neg, ["sessions", "ad_spend_aed", "ad_clicks", "units_sold"]] = df.loc[
            neg, ["sessions", "ad_spend_aed", "ad_clicks", "units_sold"]
        ].clip(lower=0)

    # Sessions below ad clicks is impossible (every click is a session). Zero sessions with
    # orders or clicks is the same thing. The orders are real; the sessions are missing.
    impossible = (df["sessions"] < df["ad_clicks"]) | ((df["sessions"] == 0) & ((df["ad_clicks"] > 0) | (df["units_sold"] > 0)))
    df["sessions_valid"] = ~impossible
    anomalies = df[impossible].copy()
    if impossible.any():
        by_day = anomalies.groupby(["date", "channel"]).size()
        for (d, ch), n in by_day.items():
            issues.append(f"{ch}: {n} listings report sessions below ad clicks on {d} (treated as missing sessions; orders kept)")

    df = df.sort_values(["date", "style_id", "channel"]).reset_index(drop=True)
    return df, issues, anomalies


def clean_inventory(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Check and clean the inventory. Returns (inventory, notes).

    Style ids upper-cased, channels mapped to our ids, category tidied ("men's" -> 'men'), 'EU' stripped from sizes,
    blank stock read as 0, duplicate rows dropped.
    """
    df = _conform(df, INVENTORY)
    issues: list[str] = []
    df["style_id"] = df["style_id"].astype(str).str.strip().str.upper()
    df["channel"] = normalise_channel(df["channel"])
    if "category" in df.columns:
        df["category"] = (
            df["category"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("'s", "", regex=False)
            .str.replace("s'", "s", regex=False)
        )
    else:
        df["category"] = "all"
    if "style_name" not in df.columns:
        df["style_name"] = df["style_id"]
    df["size_eu"] = df["size_eu"].astype(str).str.replace("EU", "", regex=False).str.strip()
    n_na = int(df[list(INVENTORY.numeric)].isna().sum().sum())
    if n_na:
        issues.append(f"{n_na} blank inventory cells read as 0")
    df[["units_at_start", "units_on_hand"]] = df[["units_at_start", "units_on_hand"]].fillna(0)
    df["unit_cost_aed"] = df.groupby("style_id")["unit_cost_aed"].transform(lambda s: s.ffill().bfill())
    dups = df.duplicated(subset=["style_id", "channel", "size_eu"], keep="last").sum()
    if dups:
        issues.append(f"{dups} duplicate inventory rows dropped")
        df = df.drop_duplicates(subset=["style_id", "channel", "size_eu"], keep="last")
    return df.reset_index(drop=True), issues


def clean_constraints(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Check and clean the per-style MRP and floor. Returns (constraints, notes).

    A style without a floor gets no price actions.
    """
    df = _conform(df, CONSTRAINTS)
    df["style_id"] = df["style_id"].astype(str).str.strip().str.upper()
    issues: list[str] = []
    if df[list(CONSTRAINTS.numeric)].isna().any().any():
        issues.append("some styles have no MRP or floor; those styles get no price recommendations")
    return df.drop_duplicates("style_id").reset_index(drop=True), issues


# -------------------------------------------------------------------------------------------
# assembly
# -------------------------------------------------------------------------------------------
def build_dataset(
    inventory: pd.DataFrame, history: pd.DataFrame, constraints: pd.DataFrame, exited: dict | None = None
) -> Dataset:
    """Clean the three files and cross-check them.

    Flags listings that appear in one file but not another, days missing from the history, and stock movement that
    doesn't match units sold (allowing for pairs already sent to the jobber). Findings become notes on the page,
    not errors.
    """
    exited = dict(exited or {})
    inv, i1 = clean_inventory(inventory)
    hist, i2, anomalies = clean_history(history)
    cons, i3 = clean_constraints(constraints)
    issues = i1 + i2 + i3

    # listings in history but not in inventory, and vice versa
    inv_keys = set(map(tuple, inv[["style_id", "channel"]].drop_duplicates().values))
    hist_keys = set(map(tuple, hist[["style_id", "channel"]].drop_duplicates().values))
    for k in sorted(hist_keys - inv_keys):
        issues.append(f"{k[0]} on {k[1]} has history but no inventory rows; ignored")
    for k in sorted(inv_keys - hist_keys):
        issues.append(f"{k[0]} on {k[1]} has inventory but no history; treated as a new listing with no sales")
    hist = hist[[tuple(x) in inv_keys for x in hist[["style_id", "channel"]].values]]
    missing_cons = set(inv["style_id"]) - set(cons["style_id"])
    for s in sorted(missing_cons):
        issues.append(f"{s} has no row in constraints; no price actions will be proposed for it")

    today = max(hist["date"]) if len(hist) else date.today()
    # missing days per listing
    all_days = pd.date_range(min(hist["date"]), today).date if len(hist) else []
    counts = hist.groupby(["style_id", "channel"]).size()
    short = counts[counts < len(all_days)]
    for (s, ch), n in short.items():
        issues.append(f"{s} on {ch}: {len(all_days) - n} day(s) missing from history")

    # reconciliation: start - on hand should equal units sold in the window
    sold = hist.groupby(["style_id", "channel"])["units_sold"].sum()
    stock = inv.groupby(["style_id", "channel"])[["units_at_start", "units_on_hand"]].sum()
    stock["delta"] = stock["units_at_start"] - stock["units_on_hand"]
    rec = stock.join(sold.rename("sold"), how="left").fillna(0)
    rec["exited"] = [exited.get(f"{s}:{ch}", 0) for s, ch in rec.index]
    off = rec[(rec["delta"] - rec["sold"] - rec["exited"]).abs() > 0]
    for (s, ch), r in off.iterrows():
        issues.append(
            f"{s} on {ch}: inventory moved by {int(r['delta'])} but history shows {int(r['sold'])} sold"
            + (f" and {int(r['exited'])} exited" if r["exited"] else "")
        )

    return Dataset(inventory=inv, history=hist, constraints=cons, today=today, issues=issues, anomalies=anomalies, exited=exited)


def load_from_paths(inventory_path, history_path, constraints_path, exited: dict | None = None) -> Dataset:
    """Read the three CSVs from disk and build the dataset. `exited` = units per listing already taken out of
    the inventory file for the jobber, so the stock-vs-sales reconciliation doesn't flag them."""
    return build_dataset(pd.read_csv(inventory_path), pd.read_csv(history_path), pd.read_csv(constraints_path), exited)


def read_uploaded(file_bytes: bytes, name: str = "") -> pd.DataFrame:
    """CSV or Excel bytes -> DataFrame."""
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    text = file_bytes.decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(text))


def merge_new_history(ds: Dataset, new_history: pd.DataFrame) -> tuple[Dataset, list]:
    """Append a new day's (or several days') rows. Rows for dates already present replace the old ones.

    Stock on hand is rolled forward by subtracting the new units sold, spread across sizes in
    proportion to what is on hand, and units_at_start is left alone, so the file still
    reconciles (start - on hand = all units sold). If the user also uploads a fresh inventory
    file, that replaces this approximation.
    """
    new, issues, _ = clean_history(new_history)
    old_today = ds.today
    keep = ~ds.history.set_index(["date", "style_id", "channel"]).index.isin(new.set_index(["date", "style_id", "channel"]).index)
    combined = pd.concat([ds.history[keep], new], ignore_index=True).drop(columns=["sessions_valid"])

    inv = ds.inventory.copy()
    newer = new[new["date"] > old_today]
    if len(newer):
        sold = newer.groupby(["style_id", "channel"])["units_sold"].sum()
        for (s, ch), n in sold.items():
            mask = (inv["style_id"] == s) & (inv["channel"] == ch)
            on_hand = inv.loc[mask, "units_on_hand"].astype(float)
            total = on_hand.sum()
            if total <= 0 or n <= 0:
                continue
            take = on_hand / total * min(n, total)
            take_int = np.floor(take).astype(int)
            remainder = int(min(n, total) - take_int.sum())
            order = np.argsort(-(take - take_int).values)
            for idx in order[:remainder]:
                take_int.iloc[idx] += 1
            inv.loc[mask, "units_on_hand"] = (on_hand - take_int).clip(lower=0).astype(int)
        issues.append(f"stock rolled forward from {old_today} to {max(newer['date'])} using units sold ({int(sold.sum())} pairs)")
    ds2 = build_dataset(inv, combined, ds.constraints, ds.exited)
    return ds2, issues + [i for i in ds2.issues if i not in ds.issues]


def replace_inventory(ds: Dataset, new_inventory: pd.DataFrame) -> tuple[Dataset, list]:
    """Use a freshly uploaded inventory file. It is taken as the truth, so past exits are not subtracted again."""
    ds2 = build_dataset(
        new_inventory, ds.history.drop(columns=["sessions_valid"]), ds.constraints
    )  # a fresh file is the truth; past exits are in it already
    return ds2, ["inventory replaced by uploaded file"] + ds2.issues


def replace_constraints(ds: Dataset, new_constraints: pd.DataFrame) -> tuple[Dataset, list]:
    """Use a freshly uploaded constraints file (new MRPs or floors); everything else is kept."""
    ds2 = build_dataset(ds.inventory, ds.history.drop(columns=["sessions_valid"]), new_constraints, ds.exited)
    return ds2, ["constraints replaced by uploaded file"] + ds2.issues
