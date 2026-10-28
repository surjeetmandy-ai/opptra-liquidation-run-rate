"""One way to load the working session, for the app, the terminal and the scenario scripts.

A session is the three files plus what has been applied. Loading it means: read the data
(uploaded days persisted under .session/ take precedence over the originals in data/), take
out the pairs already sent to the jobber, score yesterday's predictions against the new data,
and fix day 1 of the campaign on the first load. These are the side effects planning must not
have, so they live here, and build_plan can stay pure.
"""

import json
import shutil
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .apply import replay_exits, score_predictions
from .load import Dataset, load_from_paths
from .settings import DATA_DIR, ROOT, STATE_PATH
from .state import State

SESSION_DIR = ROOT / ".session"
FILES = ("inventory.csv", "daily_history.csv", "constraints.csv")
META = "meta.json"  # how many of state.exits are already taken out of the persisted inventory


def session_data_dir(data_dir: Path = DATA_DIR) -> Path:
    """Where the current data lives: the persisted session if every file is there (and the caller
    asked for the default data folder), else the folder given."""
    if data_dir != DATA_DIR:
        return data_dir
    return SESSION_DIR if all((SESSION_DIR / f).exists() for f in FILES) else data_dir


def _exits_already_in(d: Path) -> int:
    """Exit records already reflected in the inventory file under `d` (0 for the original files)."""
    if d != SESSION_DIR or not (d / META).exists():
        return 0
    try:
        return int(json.loads((d / META).read_text()).get("exits_applied", 0))
    except (ValueError, TypeError):
        return 0


def load_session(data_dir: Path = DATA_DIR, state_path: Path = STATE_PATH) -> tuple[Dataset, State]:
    """The dataset and state every entry point should plan from.

    A persisted session's inventory already has the exits applied up to the moment it was written
    (apply takes units out of the inventory at once), so only exits recorded after that are replayed;
    replaying all of them would take the same pairs out twice."""
    state = State.load(state_path)
    d = session_data_dir(data_dir)
    baked = min(_exits_already_in(d), len(state.exits))
    exited: dict = {}
    for e in state.exits[:baked]:
        exited[e["listing_id"]] = exited.get(e["listing_id"], 0) + int(e["units"])
    ds = load_from_paths(d / "inventory.csv", d / "daily_history.csv", d / "constraints.csv", exited)
    replay_exits(ds, state, skip=baked)
    score_predictions(ds, state)
    if state.campaign_start is None:
        state.campaign_start = ds.today + timedelta(days=1)
    return ds, state


def persist_session(ds: Dataset, state: State) -> None:
    """Keep uploaded days and rolled-forward stock beside state.json, so a browser refresh or a
    restart comes back to the same place. The inventory written here already reflects every exit in
    `state`, and meta.json says so, so a later load does not take them out again."""
    SESSION_DIR.mkdir(exist_ok=True)
    ds.inventory.to_csv(SESSION_DIR / "inventory.csv", index=False)
    ds.history.drop(columns=["sessions_valid"], errors="ignore").to_csv(SESSION_DIR / "daily_history.csv", index=False)
    ds.constraints.to_csv(SESSION_DIR / "constraints.csv", index=False)
    (SESSION_DIR / META).write_text(json.dumps({"exits_applied": len(state.exits)}))


def clear_session(state_path: Path = STATE_PATH) -> None:
    """Back to the original files and nothing applied."""
    shutil.rmtree(SESSION_DIR, ignore_errors=True)
    Path(state_path).unlink(missing_ok=True)


def check_upload_dates(new_history: pd.DataFrame, today, max_days_ahead: int = 31) -> None:
    """A typo in a date column can jump 'today' by a year, collapse the days left and trigger mass
    exits. Refuse dates more than a month past the data we have."""
    dates = pd.to_datetime(new_history.get("date", pd.Series(dtype=str)), errors="coerce").dropna().dt.date
    if not len(dates):
        return
    limit = today + timedelta(days=max_days_ahead)
    bad = sorted({d for d in dates if d > limit})
    if bad:
        raise ValueError(
            f"the upload has dates more than {max_days_ahead} days after the data on hand ({today}): {bad[:3]}; check the file"
        )
