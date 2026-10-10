import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "usage.sqlite3")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS daily (day TEXT PRIMARY KEY, seconds REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS daily_state (
                day TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('absent','unknown')),
                seconds REAL NOT NULL, PRIMARY KEY(day,state)
            );
            CREATE TABLE IF NOT EXISTS events (at TEXT NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS watch_daily (day TEXT PRIMARY KEY, seconds REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS watch_events (
                at TEXT NOT NULL, active INTEGER NOT NULL CHECK(active IN (0,1)),
                photo_path TEXT
            );
        """)

    def record(self, at: datetime, elapsed: float, state: str, changed: bool, interval_state: str | None = None):
        # Split intervals at local midnight; do not assign yesterday's usage to today.
        category = interval_state if interval_state is not None else state
        if elapsed > 0 and category not in ("absent", "unknown"):
            raise ValueError("Counted intervals must be absent or unknown")
        start = at - timedelta(seconds=elapsed)
        with self.db:
            while start < at:
                boundary = datetime.combine(start.date() + timedelta(days=1), datetime.min.time())
                end = min(boundary, at)
                seconds = (end - start).total_seconds()
                self.db.execute(
                    "INSERT INTO daily_state VALUES (?, ?, ?) ON CONFLICT(day,state) DO UPDATE SET seconds=seconds+excluded.seconds",
                    (start.date().isoformat(), category, seconds),
                )
                start = end
            if changed:
                self.db.execute("INSERT INTO events VALUES (?, ?)", (at.isoformat(), state))

    def record_watch(self, at: datetime, elapsed: float, active: bool, changed: bool,
                     photo_path: str | None = None):
        """Save suspect-use intervals independently from phone-presence totals.

        ``elapsed`` is the already counted suspect time ending at ``at``. It can
        be positive when ``active`` is false on the transition out of activity.
        """
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Suspect-use elapsed time must be finite and nonnegative")
        start = at - timedelta(seconds=elapsed)
        with self.db:
            while start < at:
                boundary = datetime.combine(
                    start.date() + timedelta(days=1), datetime.min.time(), tzinfo=start.tzinfo,
                )
                end = min(boundary, at)
                seconds = (end - start).total_seconds()
                self.db.execute(
                    "INSERT INTO watch_daily VALUES (?, ?) ON CONFLICT(day) DO UPDATE SET seconds=seconds+excluded.seconds",
                    (start.date().isoformat(), seconds),
                )
                start = end
            if changed or photo_path is not None:
                self.db.execute(
                    "INSERT INTO watch_events VALUES (?, ?, ?)",
                    (at.isoformat(), int(bool(active)), photo_path),
                )

    def watch_total(self, day: str) -> float:
        row = self.db.execute("SELECT seconds FROM watch_daily WHERE day=?", (day,)).fetchone()
        return row[0] if row else 0.0

    def totals(self, day: str) -> dict[str, float]:
        totals = {"absent": 0.0, "unknown": 0.0, "legacy": 0.0}
        for state, seconds in self.db.execute("SELECT state,seconds FROM daily_state WHERE day=?", (day,)):
            totals[state] = seconds
        row = self.db.execute("SELECT seconds FROM daily WHERE day=?", (day,)).fetchone()
        totals["legacy"] = row[0] if row else 0.0
        return totals

    def total(self, day: str) -> float:
        return sum(self.totals(day).values())

    def close(self):
        self.db.close()
