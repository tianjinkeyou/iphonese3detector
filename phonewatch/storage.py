import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "usage.sqlite3")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS daily (day TEXT PRIMARY KEY, seconds REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events (at TEXT NOT NULL, state TEXT NOT NULL);
        """)

    def record(self, at: datetime, elapsed: float, state: str, changed: bool):
        # Split intervals at local midnight; do not assign yesterday's usage to today.
        start = at - timedelta(seconds=elapsed)
        with self.db:
            while start < at:
                boundary = datetime.combine(start.date() + timedelta(days=1), datetime.min.time())
                end = min(boundary, at)
                seconds = (end - start).total_seconds()
                self.db.execute(
                    "INSERT INTO daily VALUES (?, ?) ON CONFLICT(day) DO UPDATE SET seconds=seconds+excluded.seconds",
                    (start.date().isoformat(), seconds),
                )
                start = end
            if changed:
                self.db.execute("INSERT INTO events VALUES (?, ?)", (at.isoformat(), state))

    def total(self, day: str) -> float:
        row = self.db.execute("SELECT seconds FROM daily WHERE day=?", (day,)).fetchone()
        return row[0] if row else 0.0

    def close(self):
        self.db.close()
