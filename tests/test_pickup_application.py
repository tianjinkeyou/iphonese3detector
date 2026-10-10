import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from phonewatch.__main__ import main
from phonewatch.event_log import EventJournal


class ClockedCamera:
    """Advance the observation clock only when a camera frame is acquired."""

    def __init__(self, observations, clock):
        self.observations = iter(observations)
        self.clock = clock
        self.raw_presence = None
        self.frame = np.full((320, 480, 3), 150, dtype=np.uint8)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        seconds, presence, available = next(self.observations)
        self.clock[0] = seconds
        self.raw_presence = presence
        if not available:
            return False, None
        frame = self.frame.copy()
        if presence is True:
            # A visible phone and acquisition-dependent shade let regressions
            # distinguish the cached prior frame from both absence and return.
            shade = 20 + int(seconds) % 30
            cv2.rectangle(frame, (180, 80), (300, 240), (shade, shade, shade), -1)
        return True, frame

    def release(self):
        self.released = True


class PickupApplicationTests(unittest.TestCase):
    day = "2026-10-10"
    base = datetime(2026, 10, 10, 12, 0)

    def run_monitor(self, directory, observations, *, keys=None, photo_failure=False,
                    export_failure=False):
        observations = [entry if len(entry) == 3 else (*entry, True)
                        for entry in observations]
        clock = [observations[0][0]]
        camera = ClockedCamera(observations, clock)
        detector = Mock()
        detector.detect.side_effect = lambda region: (camera.raw_presence, "mock smart", [])
        keys = keys or [-1] * (len(observations) - 1) + [ord("q")]
        output = io.StringIO()
        argv = ["phonewatch", "--detector", "smart", "--delay", "5",
                "--pickup-delay", "1", "--data-dir", str(directory)]
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch("cv2.VideoCapture", return_value=camera))
            stack.enter_context(patch("cv2.imshow"))
            stack.enter_context(patch("cv2.destroyAllWindows"))
            stack.enter_context(patch("cv2.waitKey", side_effect=keys))
            texts = stack.enter_context(patch("cv2.putText", wraps=cv2.putText))
            stack.enter_context(patch("phonewatch.smart.SmartDetector", return_value=detector))
            stack.enter_context(patch("phonewatch.__main__.time.monotonic",
                                      side_effect=lambda: clock[0]))
            stack.enter_context(patch("phonewatch.__main__.time.sleep"))
            dates = stack.enter_context(patch("phonewatch.__main__.datetime", wraps=datetime))
            dates.now.side_effect = lambda: self.base + timedelta(seconds=clock[0])
            if photo_failure:
                stack.enter_context(patch("cv2.imwrite", return_value=False))
            exports = None
            if export_failure:
                exports = stack.enter_context(patch.object(
                    EventJournal, "export_day", side_effect=PermissionError("TXT is read-only")))
            with redirect_stdout(output):
                main()
        self.assertTrue(camera.released)
        return {
            "texts": [call.args[1] for call in texts.call_args_list],
            "stdout": output.getvalue(),
            "export_calls": exports.call_count if exports is not None else None,
        }

    def records(self, directory):
        with sqlite3.connect(Path(directory) / "usage.sqlite3") as db:
            return [
                {"id": event_id, "event": kind, "time": at,
                 "session": session, "details": json.loads(details)}
                for event_id, kind, at, session, details in db.execute(
                    "SELECT id,kind,at,session_id,details FROM activity_events ORDER BY id")
            ]

    def exported_records(self, directory):
        path = Path(directory) / "events" / f"{self.day}.txt"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")]

    def events_of(self, records, kind):
        return [event for event in records if event["event"] == kind]

    def test_short_pickup_counts_before_usage_absent_debounce(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_monitor(directory, [
                (0, True), (1, True), (6, True), (7, False),
                (8, False), (9, True), (10, True),
            ])
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            returns = self.events_of(records, "return")
            self.assertEqual(len(pickups), 1)
            self.assertEqual(len(returns), 1)
            self.assertEqual(pickups[0]["details"]["state"], "present")
            self.assertEqual(pickups[0]["details"]["confirmed_after_seconds"], 1)
            self.assertEqual(returns[0]["details"]["away_seconds"], 2)
            self.assertEqual(pickups[0]["details"]["cycle_id"],
                             returns[0]["details"]["cycle_id"])
            self.assertEqual(pickups[0]["details"]["cycle_id"],
                             f"{pickups[0]['session']}:1")
            self.assertTrue(pickups[0]["details"]["before_photo_time"].startswith(
                "2026-10-10T12:00:06"))
            self.assertFalse(any(event["details"]["to_state"] == "absent"
                                 for event in self.events_of(records, "state_change")))
            with sqlite3.connect(Path(directory) / "usage.sqlite3") as db:
                absent = db.execute(
                    "SELECT seconds FROM daily_state WHERE day=? AND state='absent'",
                    (self.day,),
                ).fetchone()
            self.assertEqual(absent[0] if absent else 0, 0)
            self.assertIn("PICKUPS today (estimated): 1", result["texts"])
            for event in pickups + returns:
                path = Path(directory) / event["details"]["photo_path"]
                self.assertTrue(path.is_file())
                self.assertIsNotNone(cv2.imread(str(path)))
            before_path = Path(directory) / pickups[0]["details"]["before_photo_path"]
            self.assertTrue(before_path.name.startswith("before_pickup_"))
            before = cv2.imread(str(before_path))
            after = cv2.imread(str(Path(directory) / pickups[0]["details"]["photo_path"]))
            returned = cv2.imread(str(Path(directory) / returns[0]["details"]["photo_path"]))
            self.assertAlmostEqual(float(before[160, 240].mean()), 26, delta=2)
            self.assertAlmostEqual(float(after[160, 240].mean()), 150, delta=2)
            self.assertAlmostEqual(float(returned[160, 240].mean()), 30, delta=2)
            self.assertEqual(len(list((Path(directory) / "photos").glob("pickup_*.jpg"))), 1)
            self.assertEqual(self.exported_records(directory), records)

    def test_startup_absent_and_restart_do_not_fabricate_pickups(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, False), (1, False), (6, False), (7, True), (8, True),
            ])
            self.assertEqual(self.events_of(self.records(directory), "pickup"), [])
            self.run_monitor(directory, [
                (10, True), (11, True), (12, False), (13, False),
            ])
            result = self.run_monitor(directory, [(20, False), (21, False), (22, False)])
            records = self.records(directory)
            self.assertEqual(len(self.events_of(records, "pickup")), 1)
            self.assertEqual(self.events_of(records, "return"), [])
            self.assertEqual(len({event["session"] for event in records}), 3)
            self.assertIn("PICKUPS today (estimated): 1", result["texts"])
            baselines = self.events_of(records, "baseline")
            self.assertEqual([event["details"]["to_state"] for event in baselines],
                             ["absent", "present", "absent"])
            self.assertEqual(self.exported_records(directory), records)

    def test_camera_loss_discards_baseline_but_later_observed_pickup_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, True), (1, True), (2, None, False),
                (3, False), (4, False), (5, True), (6, True),
                (7, False), (8, False), (9, True), (10, True),
            ])
            records = self.records(directory)
            self.assertEqual(len(self.events_of(records, "camera_lost")), 1)
            self.assertEqual(len(self.events_of(records, "camera_restored")), 1)
            baselines = self.events_of(records, "baseline")
            self.assertEqual([event["details"]["to_state"] for event in baselines],
                             ["present", "absent"])
            pickups = self.events_of(records, "pickup")
            self.assertEqual(len(pickups), 1)
            self.assertTrue(pickups[0]["time"].startswith("2026-10-10T12:00:08"))
            # Recovery must not attach the pre-disconnection cached phone.
            self.assertTrue(pickups[0]["details"]["before_photo_time"].startswith(
                "2026-10-10T12:00:06"))
            before = cv2.imread(str(Path(directory) / pickups[0]["details"]["before_photo_path"]))
            self.assertAlmostEqual(float(before[160, 240].mean()), 26, delta=2)
            self.assertNotIn("before_photo_path", baselines[1]["details"])
            self.assertEqual(len(self.events_of(records, "return")), 1)
            self.assertEqual(self.events_of(records, "unobserved_change"), [])
            self.assertEqual(self.exported_records(directory), records)

    def test_short_unknown_bridge_reports_uncertain_duration_with_pickup(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, True), (1, True), (2, None), (3, False),
                (4, False), (5, True), (6, True),
            ])
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            self.assertEqual(len(pickups), 1)
            self.assertEqual(pickups[0]["details"]["unknown_seconds"], 1)
            self.assertEqual(len(self.events_of(records, "return")), 1)
            self.assertEqual(self.events_of(records, "unobserved_change"), [])

    def test_long_unknown_change_is_logged_without_inventing_pickup(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, True), (1, True), (2, None), (5, None),
                (6, False), (7, False),
            ])
            records = self.records(directory)
            self.assertEqual(self.events_of(records, "pickup"), [])
            self.assertEqual(self.events_of(records, "return"), [])
            changes = self.events_of(records, "unobserved_change")
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0]["details"]["from_state"], "present")
            self.assertEqual(changes[0]["details"]["to_state"], "absent")
            self.assertEqual(changes[0]["details"]["unknown_seconds"], 4)
            self.assertTrue(changes[0]["details"]["before_photo_time"].startswith(
                "2026-10-10T12:00:01"))
            before = cv2.imread(str(Path(directory) / changes[0]["details"]["before_photo_path"]))
            after = cv2.imread(str(Path(directory) / changes[0]["details"]["photo_path"]))
            self.assertAlmostEqual(float(before[160, 240].mean()), 21, delta=2)
            self.assertAlmostEqual(float(after[160, 240].mean()), 150, delta=2)
            self.assertNotIn("cycle_id", changes[0]["details"])
            self.assertEqual(self.exported_records(directory), records)

    def test_long_unknown_returning_present_records_before_after_without_pickup(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, True), (1, True), (2, None), (5, None),
                (6, True), (7, True),
            ])
            records = self.records(directory)
            self.assertEqual(self.events_of(records, "pickup"), [])
            self.assertEqual(self.events_of(records, "return"), [])
            changes = self.events_of(records, "unobserved_change")
            self.assertEqual(len(changes), 1)
            details = changes[0]["details"]
            self.assertEqual((details["from_state"], details["to_state"]),
                             ("present", "present"))
            self.assertEqual(details["unknown_seconds"], 4)
            self.assertTrue(details["before_photo_time"].startswith("2026-10-10T12:00:01"))
            before = cv2.imread(str(Path(directory) / details["before_photo_path"]))
            after = cv2.imread(str(Path(directory) / details["photo_path"]))
            self.assertAlmostEqual(float(before[160, 240].mean()), 21, delta=2)
            self.assertAlmostEqual(float(after[160, 240].mean()), 27, delta=2)
            self.assertEqual(self.exported_records(directory), records)

    def test_snapshot_failure_preserves_pickup_and_error_in_txt(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_monitor(directory, [
                (0, True), (1, True), (2, False), (3, False),
                (4, True), (5, True),
            ], photo_failure=True)
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            returns = self.events_of(records, "return")
            self.assertEqual(len(pickups), 1)
            self.assertEqual(len(returns), 1)
            for event in pickups + returns:
                self.assertIsNone(event["details"]["photo_path"])
                self.assertIn("Could not save event photo", event["details"]["photo_error"])
            self.assertIsNone(pickups[0]["details"]["before_photo_path"])
            self.assertIn("Could not save event photo", pickups[0]["details"]["before_photo_error"])
            self.assertIn("PICKUPS today (estimated): 1", result["texts"])
            self.assertEqual(self.exported_records(directory), records)
            self.assertEqual(list((Path(directory) / "photos").glob("*.jpg")), [])

    def test_export_permission_failure_keeps_all_events_and_counts_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_monitor(directory, [
                (0, True), (1, True), (2, False), (3, False),
                (4, True), (5, True),
            ], export_failure=True)
            records = self.records(directory)
            self.assertEqual(len(self.events_of(records, "pickup")), 1)
            self.assertEqual(len(self.events_of(records, "return")), 1)
            self.assertEqual(len(self.events_of(records, "session_end")), 1)
            self.assertEqual(len(self.events_of(records, "summary")), 1)
            self.assertEqual(result["export_calls"], len(records))
            self.assertEqual(result["stdout"].count("TXT export unavailable"), 1)
            self.assertIn("PICKUPS today (estimated): 1", result["texts"])
            journal = EventJournal(Path(directory))
            try:
                self.assertEqual(journal.pickup_total(self.day), 1)
                journal.export_day(self.day)
            finally:
                journal.close()
            self.assertEqual(self.exported_records(directory), records)
            self.assertEqual(self.records(directory), records)

    def test_reference_capture_resets_pending_pickup_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, True), (1, True), (2, False), (3, False), (4, False),
                (5, True), (6, True), (7, False), (8, False),
            ], keys=[-1, -1, ord("e"), -1, -1, -1, -1, -1, ord("q")])
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            self.assertEqual(len(pickups), 1)
            self.assertTrue(pickups[0]["time"].startswith("2026-10-10T12:00:08"))
            self.assertTrue(pickups[0]["details"]["before_photo_time"].startswith(
                "2026-10-10T12:00:06"))
            saved = self.events_of(records, "reference_saved")
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["details"]["label"], "empty")
            resets = self.events_of(records, "tracking_reset")
            self.assertEqual(len(resets), 1)
            self.assertEqual(resets[0]["details"]["key"], "e")
            baselines = self.events_of(records, "baseline")
            self.assertEqual([event["details"]["to_state"] for event in baselines],
                             ["present", "absent"])
            self.assertEqual(self.exported_records(directory), records)


if __name__ == "__main__":
    unittest.main()
