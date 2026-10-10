import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from phonewatch.event_log import EventJournal
from phonewatch.storage import Store


def event_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")]


class EventJournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.journal = EventJournal(self.directory)
        self.tz = timezone(timedelta(hours=9))
        self.at = datetime(2026, 10, 10, 12, 0, 0, 123456, tzinfo=self.tz)

    def tearDown(self):
        self.journal.close()
        self.temporary.cleanup()

    def test_record_automatically_creates_utf8_text_with_single_line_json(self):
        details = {"原因": "手机离开\n支架", "photo": "照片/拿起.jpg"}
        event_id = self.journal.record(self.at, "pickup", details)
        path = self.directory / "events" / "2026-10-10.txt"
        content = path.read_text(encoding="utf-8")
        self.assertIn("不能证明手机身份", content)
        self.assertIn("当日拿起次数（估计）：1", content)
        self.assertIn("未推算旧历史", content)
        self.assertIn("摄像头中断期间", content)
        self.assertIn("只有 unknown 不会计次", content)
        self.assertIn("rapid_empty=true", content)
        self.assertIn("全程遮挡，无法可靠计次", content)
        self.assertIn("gap_recovered=true、timing_uncertain=true", content)
        self.assertIn("不含之前的 unknown 间隙", content)
        self.assertIn("手机离开\\n支架", content)
        self.assertEqual(event_lines(path), [{
            "id": event_id, "time": self.at.isoformat(), "event": "pickup",
            "session": self.journal.session_id, "details": details,
        }])

    def test_pickup_count_uses_only_pickup_rows(self):
        for kind in ("session_start", "pickup", "returned", "unknown", "pickup", "suspect_start"):
            self.journal.record(self.at, kind)
        self.assertEqual(self.journal.pickup_total("2026-10-10"), 2)
        self.assertEqual(self.journal.pickup_total("2026-10-11"), 0)
        self.assertIn("当日拿起次数（估计）：2",
                      self.journal.export_day("2026-10-10").read_text(encoding="utf-8"))

    def test_local_midnight_keeps_events_on_their_supplied_dates(self):
        first = datetime(2026, 10, 10, 23, 59, 59, tzinfo=self.tz)
        second = datetime(2026, 10, 11, 0, 0, 1, tzinfo=self.tz)
        self.journal.record(first, "pickup")
        self.journal.record(second, "pickup")
        self.assertEqual(self.journal.pickup_total("2026-10-10"), 1)
        self.assertEqual(self.journal.pickup_total("2026-10-11"), 1)
        self.assertEqual(event_lines(self.journal.export_day("2026-10-10"))[0]["time"],
                         first.isoformat())
        self.assertEqual(event_lines(self.journal.export_day("2026-10-11"))[0]["time"],
                         second.isoformat())

    def test_naive_timestamp_gets_an_explicit_local_offset(self):
        self.journal.record(datetime(2026, 10, 10, 12), "present")
        row = event_lines(self.journal.export_day("2026-10-10"))[0]
        self.assertIsNotNone(datetime.fromisoformat(row["time"]).utcoffset())

    def test_restart_preserves_sessions_and_exports_without_duplicate_events(self):
        first_session = self.journal.session_id
        first_id = self.journal.record(self.at, "pickup")
        self.journal.close()
        self.journal = EventJournal(self.directory)
        second_id = self.journal.record(self.at - timedelta(seconds=10), "returned")
        self.assertNotEqual(first_session, self.journal.session_id)
        path = self.journal.export_day("2026-10-10")
        original = path.read_bytes()
        self.journal.export_day("2026-10-10")
        self.assertEqual(path.read_bytes(), original)
        rows = event_lines(path)
        self.assertEqual([row["id"] for row in rows], [first_id, second_id])
        self.assertEqual([row["session"] for row in rows], [first_session, self.journal.session_id])
        self.assertEqual(self.journal.pickup_total("2026-10-10"), 1)

    def test_export_recovers_deleted_and_truncated_text(self):
        self.journal.record(self.at, "pickup")
        path = self.journal.export_day("2026-10-10")
        original = path.read_bytes()
        path.unlink()
        self.journal.export_day("2026-10-10")
        self.assertEqual(path.read_bytes(), original)
        path.write_text("damaged\n", encoding="utf-8")
        self.journal.export_day("2026-10-10")
        self.assertEqual(path.read_bytes(), original)
        path.write_text("damaged\n", encoding="utf-8")
        self.journal.record(self.at, "returned")
        self.assertEqual(len(event_lines(path)), 2)

    def test_invalid_input_changes_neither_database_nor_text(self):
        self.journal.record(self.at, "pickup")
        path = self.journal.export_day("2026-10-10")
        original = path.read_bytes()
        cases = [
            ("2026-10-10", "pickup", {}),
            (self.at, "", {}), (self.at, "  ", {}), (self.at, 4, {}),
            (self.at, "pickup", []),
            (self.at, "pickup", {"score": float("nan")}),
            (self.at, "pickup", {"score": float("inf")}),
            (self.at, "pickup", {"object": object()}),
        ]
        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises((ValueError, TypeError)):
                    self.journal.record(*values)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(self.journal.db.execute("SELECT COUNT(*) FROM activity_events").fetchone()[0], 1)
        for day in ("../other", "2026-13-01", "20261010", 20261010):
            with self.subTest(day=day), self.assertRaises(ValueError):
                self.journal.export_day(day)
            with self.subTest(day=day), self.assertRaises(ValueError):
                self.journal.pickup_total(day)
        self.assertEqual(path.read_bytes(), original)

    def test_empty_day_export_has_zero_count_and_no_invented_events(self):
        path = self.journal.export_day("2026-10-09")
        self.assertEqual(event_lines(path), [])
        self.assertIn("当日拿起次数（估计）：0", path.read_text(encoding="utf-8"))

    def test_text_failure_leaves_the_committed_event_recoverable(self):
        # A text export can be repaired; it must not erase the canonical event.
        (self.directory / "events").write_text("blocks directory", encoding="utf-8")
        with self.assertRaises(OSError):
            self.journal.record(self.at, "pickup")
        self.assertEqual(self.journal.pickup_total("2026-10-10"), 1)
        (self.directory / "events").unlink()
        self.assertEqual(len(event_lines(self.journal.export_day("2026-10-10"))), 1)


class EventJournalMigrationTests(unittest.TestCase):
    def test_additive_schema_preserves_legacy_and_store_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            db = sqlite3.connect(directory / "usage.sqlite3")
            db.executescript("""
                CREATE TABLE daily (day TEXT PRIMARY KEY, seconds REAL NOT NULL);
                INSERT INTO daily VALUES ('2026-10-10', 600);
            """)
            db.close()
            store = Store(directory)
            at = datetime(2026, 10, 10, 12)
            store.record(at, 12, "absent", True)
            store.record(at, 17, "unknown", True)
            store.record_watch(at, 9, True, True, "photos/check.jpg")
            journal = EventJournal(directory)
            try:
                journal.record(at, "pickup")
                self.assertEqual(store.totals("2026-10-10"), {
                    "absent": 12, "unknown": 17, "legacy": 600,
                })
                self.assertEqual(store.watch_total("2026-10-10"), 9)
                self.assertEqual(store.db.execute("SELECT state FROM events").fetchall(),
                                 [("absent",), ("unknown",)])
                self.assertEqual(store.db.execute("SELECT photo_path FROM watch_events").fetchall(),
                                 [("photos/check.jpg",)])
                text = journal.export_day("2026-10-10").read_text(encoding="utf-8")
                self.assertIn("absent=12.000 unknown=17.000 suspected_use=9.000", text)
                self.assertEqual(journal.pickup_total("2026-10-10"), 1)
            finally:
                journal.close()
                store.close()


if __name__ == "__main__":
    unittest.main()
