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
from phonewatch.reference import ReferenceDetector


class ParkingCamera:
    """Provide independent model observations and physical parking scenes."""

    def __init__(self, observations, clock):
        self.observations = iter(observations)
        self.clock = clock
        self.current = None
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        self.current = next(self.observations)
        seconds, _, parked, _, available = self.current
        self.clock[0] = seconds
        if not available:
            return False, None
        frame = np.full((380, 480, 3), 150, dtype=np.uint8)
        if parked is True:
            shade = 20 + int(seconds) % 30
            cv2.rectangle(frame, (180, 80), (300, 240), (shade, shade, shade), -1)
        return True, frame

    def release(self):
        self.released = True


class ParkingApplicationTests(unittest.TestCase):
    day = "2026-10-10"
    base = datetime(2026, 10, 10, 12)

    def run_monitor(self, directory, observations, *, pickup_mode=None, ready=True,
                    use_real_parking=False, instant_empty=False):
        observations = [tuple(entry) + (True,) * (5 - len(entry))
                        for entry in observations]
        clock = [observations[0][0]]
        camera = ParkingCamera(observations, clock)
        smart = Mock()
        smart.detect.side_effect = lambda region: (camera.current[1], "mock phone model", [])
        parking = Mock()
        parking.ready.return_value = ready

        def parking_detection(region, *, phone_polygons, screen_polygon):
            _, _, parked, reliable, _ = camera.current
            return parked, "mock parked-body evidence", {
                "reliable_empty": reliable,
                "immediate_empty": instant_empty if parked is False else False,
            }

        parking.detect.side_effect = parking_detection
        argv = ["phonewatch", "--detector", "smart", "--delay", "5",
                "--pickup-delay", "1", "--data-dir", str(directory)]
        if pickup_mode is not None:
            argv.extend(["--pickup-mode", pickup_mode])
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch("cv2.VideoCapture", return_value=camera))
            stack.enter_context(patch("cv2.imshow"))
            stack.enter_context(patch("cv2.destroyAllWindows"))
            stack.enter_context(patch("cv2.waitKey", side_effect=
                                      [-1] * (len(observations) - 1) + [ord("q")]))
            texts = stack.enter_context(patch("cv2.putText", wraps=cv2.putText))
            stack.enter_context(patch("phonewatch.smart.SmartDetector", return_value=smart))
            if not use_real_parking:
                stack.enter_context(patch("phonewatch.parking.ParkingDetector", return_value=parking))
            stack.enter_context(patch("phonewatch.__main__.time.monotonic",
                                      side_effect=lambda: clock[0]))
            stack.enter_context(patch("phonewatch.__main__.time.sleep"))
            dates = stack.enter_context(patch("phonewatch.__main__.datetime", wraps=datetime))
            dates.now.side_effect = lambda: self.base + timedelta(seconds=clock[0])
            with redirect_stdout(io.StringIO()):
                main()
        self.assertTrue(camera.released)
        return parking, [call.args[1] for call in texts.call_args_list]

    def records(self, directory):
        with sqlite3.connect(Path(directory) / "usage.sqlite3") as db:
            return [
                {"id": event_id, "event": kind, "time": at,
                 "session": session, "details": json.loads(details)}
                for event_id, kind, at, session, details in db.execute(
                    "SELECT id,kind,at,session_id,details FROM activity_events ORDER BY id")
            ]

    def events_of(self, records, kind):
        return [event for event in records if event["event"] == kind]

    def assert_export_matches(self, directory, records):
        path = Path(directory) / "events" / f"{self.day}.txt"
        exported = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line and not line.startswith("#")]
        self.assertEqual(exported, records)

    def assert_before_after(self, directory, details, before_seconds):
        self.assertTrue(details["before_photo_time"].startswith(
            f"2026-10-10T12:00:{before_seconds:02}"))
        before = cv2.imread(str(Path(directory) / details["before_photo_path"]))
        after = cv2.imread(str(Path(directory) / details["photo_path"]))
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertAlmostEqual(float(before[160, 240].mean()), 20 + before_seconds % 30, delta=2)
        self.assertAlmostEqual(float(after[160, 240].mean()), 150, delta=2)

    def test_unknown_phone_model_still_counts_reliable_empty_parking(self):
        with tempfile.TemporaryDirectory() as directory:
            parking, texts = self.run_monitor(directory, [
                (0, None, True), (1, None, True), (2, None, False),
                (3, None, False), (4, None, True), (5, None, True),
            ])
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            returns = self.events_of(records, "return")
            self.assertEqual(len(pickups), 1)
            self.assertEqual(len(returns), 1)
            details = pickups[0]["details"]
            self.assertEqual(details["pickup_source"], "parking")
            self.assertEqual(details["phone_observation"], "unknown")
            self.assertEqual(details["state"], "unknown")
            self.assertIs(details["parking_evidence"]["reliable_empty"], True)
            self.assertEqual(details["cycle_id"], returns[0]["details"]["cycle_id"])
            self.assert_before_after(directory, details, 1)
            self.assertEqual(parking.detect.call_count, 6)
            self.assertIn("Pickup evidence: EMPTY (parking)", texts)
            self.assertIn("PICKUPS today (estimated): 1", texts)
            with sqlite3.connect(Path(directory) / "usage.sqlite3") as db:
                states = dict(db.execute("SELECT state,seconds FROM daily_state WHERE day=?", (self.day,)))
            self.assertEqual(states.get("absent", 0), 0)
            self.assertEqual(states["unknown"], 5)
            self.assert_export_matches(directory, records)

    def test_real_parking_references_drive_pickup_and_persist_across_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            references = ReferenceDetector(Path(directory) / "references")
            empty = np.full((380, 480, 3), 150, dtype=np.uint8)
            phone = empty.copy()
            cv2.rectangle(phone, (180, 80), (300, 240), (20, 20, 20), -1)
            references.save("empty", empty)
            references.save("phone", phone)
            self.run_monitor(directory, [
                (0, None, True), (1, None, True),
                (2, None, False), (3, None, False),
            ], use_real_parking=True)
            _, texts = self.run_monitor(directory, [
                (10, None, False), (11, None, False),
            ], use_real_parking=True)
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            self.assertEqual(len(pickups), 1)
            details = pickups[0]["details"]
            self.assertEqual(details["pickup_source"], "parking")
            self.assertEqual(details["phone_observation"], "unknown")
            self.assertIs(details["parking_evidence"]["reliable_empty"], True)
            self.assertIs(details["parking_evidence"]["immediate_empty"], True)
            self.assertIs(details["rapid_empty"], True)
            self.assertTrue(pickups[0]["time"].startswith("2026-10-10T12:00:02"))
            self.assertGreater(details["parking_evidence"]["body_pixels"], 0)
            self.assertEqual(details["parking_evidence"]["poses"], 1)
            self.assert_before_after(directory, details, 1)
            self.assertEqual(len({event["session"] for event in records}), 2)
            self.assertIn("PICKUPS today (estimated): 1", texts)
            self.assertIn("Pickup evidence: EMPTY (parking)", texts)
            self.assertEqual(self.events_of(records, "return"), [])
            self.assert_export_matches(directory, records)

    def test_reliable_empty_recovers_long_unknown_gap_once_with_uncertain_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, True, True), (1, True, True), (2, None, None),
                (5, None, None), (6, None, False), (7, None, False),
                (8, None, False), (9, None, True), (10, None, True),
            ])
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            self.assertEqual(len(pickups), 1)
            details = pickups[0]["details"]
            self.assertTrue(pickups[0]["time"].startswith("2026-10-10T12:00:07"))
            self.assertIs(details["gap_recovered"], True)
            self.assertIs(details["timing_uncertain"], True)
            self.assertEqual(details["unknown_seconds"], 4)
            self.assertEqual(details["confirmed_after_seconds"], 1)
            self.assertEqual((details["from_state"], details["to_state"]), ("present", "absent"))
            self.assert_before_after(directory, details, 1)
            returns = self.events_of(records, "return")
            self.assertEqual(len(returns), 1)
            self.assertEqual(returns[0]["details"]["away_seconds"], 3)
            self.assertEqual(returns[0]["details"]["cycle_id"], details["cycle_id"])
            self.assertEqual(self.events_of(records, "unobserved_change"), [])
            self.assert_export_matches(directory, records)

    def test_unknown_parking_alone_cannot_create_pickup_even_with_reliable_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            _, texts = self.run_monitor(directory, [
                (0, True, True), (1, True, True), (2, None, None),
                (5, None, None), (8, None, None),
            ])
            records = self.records(directory)
            self.assertEqual(self.events_of(records, "pickup"), [])
            self.assertEqual(self.events_of(records, "return"), [])
            self.assertEqual(self.events_of(records, "unobserved_change"), [])
            self.assertIn("Pickup evidence: UNCERTAIN (parking)", texts)
            self.assertEqual(list((Path(directory) / "photos").glob("*.jpg")), [])
            self.assert_export_matches(directory, records)

    def test_camera_reset_prevents_unknown_gap_recovery_and_discards_old_before_photo(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, True, True), (1, True, True), (2, None, None),
                (5, None, None), (6, None, None, True, False),
                (7, None, False), (8, None, False),
                (9, None, True), (10, None, True),
                (11, None, False), (12, None, False),
            ])
            records = self.records(directory)
            self.assertEqual(len(self.events_of(records, "camera_lost")), 1)
            self.assertEqual(len(self.events_of(records, "camera_restored")), 1)
            baselines = self.events_of(records, "baseline")
            self.assertEqual([event["details"]["to_state"] for event in baselines],
                             ["present", "absent"])
            self.assertTrue(baselines[1]["time"].startswith("2026-10-10T12:00:08"))
            pickups = self.events_of(records, "pickup")
            self.assertEqual(len(pickups), 1)
            self.assertTrue(pickups[0]["time"].startswith("2026-10-10T12:00:12"))
            self.assertNotIn("gap_recovered", pickups[0]["details"])
            self.assert_before_after(directory, pickups[0]["details"], 10)
            self.assert_export_matches(directory, records)

    def test_present_phone_model_and_empty_parking_count_without_absent_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, True, True), (1, True, True), (6, True, True),
                (7, True, False), (8, True, False),
                (9, True, True), (10, True, True),
            ])
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            self.assertEqual(len(pickups), 1)
            details = pickups[0]["details"]
            self.assertEqual(details["phone_observation"], "present")
            self.assertEqual(details["state"], "present")
            self.assertEqual(details["pickup_source"], "parking")
            self.assert_before_after(directory, details, 6)
            self.assertFalse(any(event["details"]["to_state"] == "absent"
                                 for event in self.events_of(records, "state_change")))
            with sqlite3.connect(Path(directory) / "usage.sqlite3") as db:
                absent = db.execute("SELECT seconds FROM daily_state WHERE day=? AND state='absent'",
                                    (self.day,)).fetchone()
            self.assertEqual(absent[0] if absent else 0, 0)
            self.assert_export_matches(directory, records)

    def test_presence_mode_retains_legacy_unobserved_gap_without_parking_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            parking, texts = self.run_monitor(directory, [
                (0, True, True), (1, True, True), (2, None, None),
                (5, None, None), (6, False, False), (7, False, False),
            ], pickup_mode="presence")
            records = self.records(directory)
            self.assertEqual(self.events_of(records, "pickup"), [])
            changes = self.events_of(records, "unobserved_change")
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0]["details"]["pickup_source"], "phone_presence")
            self.assertEqual(changes[0]["details"]["parking_evidence"], {})
            self.assertEqual(changes[0]["details"]["unknown_seconds"], 4)
            parking.ready.assert_not_called()
            parking.detect.assert_not_called()
            self.assertIn("Pickup evidence: EMPTY (phone_presence)", texts)
            self.assert_export_matches(directory, records)

    def test_unready_parking_detector_falls_back_to_phone_presence(self):
        with tempfile.TemporaryDirectory() as directory:
            parking, texts = self.run_monitor(directory, [
                (0, True, None), (1, True, None),
                (2, False, None), (3, False, None),
            ], ready=False)
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            self.assertEqual(len(pickups), 1)
            self.assertEqual(pickups[0]["details"]["pickup_source"], "phone_presence")
            self.assertEqual(parking.ready.call_count, 4)
            parking.detect.assert_not_called()
            self.assertIn("Pickup evidence: EMPTY (phone_presence)", texts)
            self.assert_export_matches(directory, records)

    def test_truthy_nonboolean_empty_flag_cannot_resolve_long_unknown_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, True, True), (1, True, True), (2, None, None),
                (5, None, None), (6, None, False, 1), (7, None, False, 1),
            ])
            records = self.records(directory)
            self.assertEqual(self.events_of(records, "pickup"), [])
            changes = self.events_of(records, "unobserved_change")
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0]["details"]["unknown_seconds"], 4)
            self.assertNotIn("gap_recovered", changes[0]["details"])
            self.assert_export_matches(directory, records)

    def test_one_strong_empty_frame_counts_quick_replacement_and_paired_return(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, None, True), (1, None, True),
                (2, None, False), (2.1, None, True), (3.1, None, True),
            ], instant_empty=True)
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            returns = self.events_of(records, "return")
            self.assertEqual(len(pickups), 1)
            self.assertEqual(len(returns), 1)
            details = pickups[0]["details"]
            self.assertTrue(pickups[0]["time"].startswith("2026-10-10T12:00:02"))
            self.assertIs(details["rapid_empty"], True)
            self.assertEqual(details["confirmed_after_seconds"], 0)
            self.assert_before_after(directory, details, 1)
            self.assertEqual(returns[0]["details"]["cycle_id"], details["cycle_id"])
            self.assertAlmostEqual(returns[0]["details"]["away_seconds"], 1.1)
            self.assertIsNotNone(cv2.imread(str(Path(directory) / returns[0]["details"]["photo_path"])))
            self.assert_export_matches(directory, records)

    def test_repeated_strong_empty_frames_during_return_confirmation_count_once(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, None, True), (1, None, True), (2, None, False),
                (2.1, None, True), (2.2, None, False),
                (2.3, None, True), (2.4, None, False),
                (2.5, None, True), (3.5, None, True),
            ], instant_empty=True)
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            returns = self.events_of(records, "return")
            self.assertEqual(len(pickups), 1)
            self.assertEqual(len(returns), 1)
            self.assertTrue(returns[0]["time"].startswith("2026-10-10T12:00:03.500000"))
            self.assertAlmostEqual(returns[0]["details"]["away_seconds"], 1.5)
            self.assertEqual(returns[0]["details"]["cycle_id"], pickups[0]["details"]["cycle_id"])
            self.assertEqual(len(list((Path(directory) / "photos").glob("pickup_*.jpg"))), 1)
            self.assert_export_matches(directory, records)

    def test_false_or_truthy_nonboolean_immediate_flag_keeps_one_second_confirmation(self):
        for flag in (False, 1):
            with self.subTest(immediate_empty=flag), tempfile.TemporaryDirectory() as directory:
                self.run_monitor(directory, [
                    (0, None, True), (1, None, True),
                    (2, None, False), (2.1, None, True), (3.1, None, True),
                    (4, None, False), (5, None, False),
                    (6, None, True), (7, None, True),
                ], instant_empty=flag)
                records = self.records(directory)
                pickups = self.events_of(records, "pickup")
                self.assertEqual(len(pickups), 1)
                self.assertTrue(pickups[0]["time"].startswith("2026-10-10T12:00:05"))
                self.assertEqual(pickups[0]["details"]["confirmed_after_seconds"], 1)
                self.assertFalse(pickups[0]["details"].get("rapid_empty", False))
                self.assertEqual(len(self.events_of(records, "return")), 1)
                self.assert_export_matches(directory, records)

    def test_strong_empty_cannot_fabricate_startup_or_after_camera_reset_pickup(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_monitor(directory, [
                (0, None, False), (.1, None, False), (1, None, False),
                (2, None, True), (3, None, True),
                (4, None, None, True, False),
                (5, None, False), (5.1, None, False), (6, None, False),
            ], instant_empty=True)
            records = self.records(directory)
            self.assertEqual(self.events_of(records, "pickup"), [])
            self.assertEqual(self.events_of(records, "return"), [])
            self.assertEqual(self.events_of(records, "unobserved_change"), [])
            self.assertEqual(len(self.events_of(records, "camera_lost")), 1)
            self.assertEqual(len(self.events_of(records, "camera_restored")), 1)
            self.assertEqual(list((Path(directory) / "photos").glob("*.jpg")), [])
            self.assert_export_matches(directory, records)


if __name__ == "__main__":
    unittest.main()
