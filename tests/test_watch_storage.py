import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from phonewatch.storage import Store


class WatchStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_suspect_time_is_separate_from_presence_and_legacy_totals(self):
        at = datetime(2026, 10, 10, 12)
        self.store.record(at, 15, "absent", True)
        self.store.record(at, 20, "unknown", True)
        self.store.record_watch(at, 30, True, True)
        self.store.record_watch(at, 10, False, True)
        self.assertEqual(self.store.watch_total("2026-10-10"), 40)
        self.assertEqual(self.store.watch_total("2026-10-09"), 0)
        self.assertEqual(self.store.totals("2026-10-10"), {
            "absent": 15, "unknown": 20, "legacy": 0,
        })
        self.assertEqual(self.store.total("2026-10-10"), 35)
        self.assertEqual(self.store.db.execute("SELECT state FROM events").fetchall(), [
            ("absent",), ("unknown",),
        ])

    def test_midnight_split_persists_after_reopen(self):
        self.store.record_watch(datetime(2026, 10, 11, 0, 0, 5), 10, True, True)
        self.store.record_watch(datetime(2026, 10, 11, 0, 0, 15), 10, False, True)
        self.store.close()
        self.store = Store(self.path)
        self.assertEqual(self.store.watch_total("2026-10-10"), 5)
        self.assertEqual(self.store.watch_total("2026-10-11"), 15)
        self.assertEqual(self.store.db.execute("SELECT active FROM watch_events").fetchall(), [
            (1,), (0,),
        ])

    def test_long_interval_splits_across_multiple_local_days(self):
        tz = timezone(timedelta(hours=9))
        self.store.record_watch(datetime(2026, 10, 12, 0, 0, 5, tzinfo=tz),
                                86410, True, False)
        self.assertEqual(self.store.watch_total("2026-10-10"), 5)
        self.assertEqual(self.store.watch_total("2026-10-11"), 86400)
        self.assertEqual(self.store.watch_total("2026-10-12"), 5)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM watch_events").fetchone()[0], 0)

    def test_events_save_photo_paths_without_duplicate_transition_rows(self):
        at = datetime(2026, 10, 10, 12)
        self.store.record_watch(at, 0, True, True, "photos/start.jpg")
        self.store.record_watch(at, 3, True, False)
        self.store.record_watch(at, 0, True, False, "photos/check.jpg")
        self.store.record_watch(at, 2, False, True)
        rows = self.store.db.execute("SELECT at,active,photo_path FROM watch_events").fetchall()
        self.assertEqual(rows, [
            (at.isoformat(), 1, "photos/start.jpg"),
            (at.isoformat(), 1, "photos/check.jpg"),
            (at.isoformat(), 0, None),
        ])
        self.assertEqual(self.store.watch_total("2026-10-10"), 5)

    def test_invalid_elapsed_does_not_write_anything(self):
        for elapsed in (-1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                self.store.record_watch(datetime(2026, 10, 10, 12), elapsed, True, True)
        self.assertEqual(self.store.watch_total("2026-10-10"), 0)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM watch_events").fetchone()[0], 0)

    def test_daily_totals_and_transition_event_commit_atomically(self):
        self.store.db.execute("""
            CREATE TRIGGER reject_watch_event BEFORE INSERT ON watch_events
            BEGIN SELECT RAISE(ABORT, 'test failure'); END
        """)
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.record_watch(datetime(2026, 10, 11, 0, 0, 5), 10, True, True)
        self.assertEqual(self.store.watch_total("2026-10-10"), 0)
        self.assertEqual(self.store.watch_total("2026-10-11"), 0)


class WatchLegacyStorageTests(unittest.TestCase):
    def test_additive_migration_preserves_old_data_and_dictionary_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            db = sqlite3.connect(path / "usage.sqlite3")
            db.executescript("""
                CREATE TABLE daily (day TEXT PRIMARY KEY, seconds REAL NOT NULL);
                CREATE TABLE daily_state (
                    day TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('absent','unknown')),
                    seconds REAL NOT NULL, PRIMARY KEY(day,state)
                );
                CREATE TABLE events (at TEXT NOT NULL, state TEXT NOT NULL);
                INSERT INTO daily VALUES ('2026-10-10', 600);
                INSERT INTO daily_state VALUES ('2026-10-10', 'absent', 12);
                INSERT INTO daily_state VALUES ('2026-10-10', 'unknown', 17);
                INSERT INTO events VALUES ('2026-10-10T12:00:00', 'absent');
            """)
            db.close()
            for index in range(2):
                store = Store(path)
                try:
                    self.assertEqual(store.totals("2026-10-10"), {
                        "absent": 12, "unknown": 17, "legacy": 600,
                    })
                    self.assertEqual(store.total("2026-10-10"), 629)
                    self.assertEqual(store.watch_total("2026-10-10"), 9 * index)
                    self.assertEqual(store.db.execute("SELECT * FROM events").fetchall(), [
                        ("2026-10-10T12:00:00", "absent"),
                    ])
                    if index == 0:
                        store.record_watch(datetime(2026, 10, 10, 12), 9, True, True)
                finally:
                    store.close()


if __name__ == "__main__":
    unittest.main()
