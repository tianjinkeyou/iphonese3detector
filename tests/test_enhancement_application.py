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
from phonewatch.image_enhancement import ImageEnhancer
from phonewatch.perspective import rectify, save_plane
from phonewatch.reference import ReferenceDetector


class EnhancementCamera:
    def __init__(self, observations, clock):
        self.observations = iter(observations)
        self.clock = clock
        self.presence = None
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        seconds, frame, self.presence = next(self.observations)
        self.clock[0] = seconds
        return True, frame.copy()

    def release(self):
        self.released = True


class EnhancementApplicationTests(unittest.TestCase):
    day = "2026-10-11"
    base = datetime(2026, 10, 11, 12)
    roi = (120, 30, 240, 280)

    def scenes(self):
        empty = np.full((380, 480, 3), 80, np.uint8)
        phone = empty.copy()
        cv2.rectangle(phone, (180, 80), (300, 240), (20, 20, 20), -1)
        return empty, phone

    def seed(self, directory, empty, phone):
        references = ReferenceDetector(Path(directory) / "references")
        references.save("empty", empty)
        references.save("phone", phone)

    def run_monitor(self, directory, observations, *, mode=None, strength=None,
                    keys=None, options=(), companion_failure=False, watch=False,
                    selected_roi=None, screen_activity=()):
        clock = [observations[0][0]]
        camera = EnhancementCamera(observations, clock)
        references = []
        detection_inputs = []
        previews = []
        selections = []
        hand_inputs = []
        screen_inputs = []
        smart = Mock()

        def detection(image):
            detection_inputs.append(image.copy())
            return camera.presence, "mock phone model", []

        smart.detect.side_effect = detection

        def detector_factory(refs, model_factory):
            references.append(refs)
            return smart

        def select_roi(window, frame, **kwargs):
            selections.append(frame.copy())
            return selected_roi

        argv = ["phonewatch", "--detector", "smart-watch" if watch else "smart",
                "--delay", "0", "--pickup-delay", "1", "--data-dir", str(directory)]
        if mode is not None:
            argv.extend(["--image-mode", mode])
        if strength is not None:
            argv.extend(["--image-strength", str(strength)])
        if watch:
            argv.extend(["--watch-screen", "--hand-mode", "landmarks"])
        argv.extend(options)
        keys = keys or [-1] * (len(observations) - 1) + [ord("q")]
        original_imwrite = cv2.imwrite

        def imwrite(path, image, *args):
            if "enhanced" in Path(path).parts:
                return False
            return original_imwrite(path, image, *args)

        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch("cv2.VideoCapture", return_value=camera))
            stack.enter_context(patch("cv2.destroyAllWindows"))
            stack.enter_context(patch("cv2.imshow", side_effect=lambda _, image: previews.append(image.copy())))
            stack.enter_context(patch("cv2.waitKey", side_effect=keys))
            stack.enter_context(patch("phonewatch.smart.SmartDetector", side_effect=detector_factory))
            stack.enter_context(patch("phonewatch.__main__.time.monotonic", side_effect=lambda: clock[0]))
            dates = stack.enter_context(patch("phonewatch.__main__.datetime", wraps=datetime))
            dates.now.side_effect = lambda: self.base + timedelta(seconds=clock[0])
            if selected_roi is not None:
                stack.enter_context(patch("cv2.selectROI", side_effect=select_roi))
            if companion_failure:
                stack.enter_context(patch("cv2.imwrite", side_effect=imwrite))
            if watch:
                hands = Mock(source="test model")
                hands.detect.side_effect = lambda frame, **kwargs: hand_inputs.append(frame.copy()) or []
                screen = Mock()
                motion = iter(screen_activity)
                screen.detect.side_effect = lambda frame, polygon: screen_inputs.append(frame.copy()) or (next(motion, False), .05, "SCREEN: test activity")
                stack.enter_context(patch("phonewatch.hand_tracking.HandTracker", return_value=hands))
                stack.enter_context(patch("phonewatch.screen_activity.ScreenActivityDetector", return_value=screen))
            with redirect_stdout(output):
                main()
        self.assertTrue(camera.released)
        return {"references": references[0], "inputs": detection_inputs,
                "previews": previews, "selections": selections,
                "hand_inputs": hand_inputs, "screen_inputs": screen_inputs,
                "stdout": output.getvalue()}

    def records(self, directory):
        with sqlite3.connect(Path(directory) / "usage.sqlite3") as db:
            records = [
                {"id": event_id, "event": kind, "time": at,
                 "session": session, "details": json.loads(details)}
                for event_id, kind, at, session, details in db.execute(
                    "SELECT id,kind,at,session_id,details FROM activity_events ORDER BY id")
            ]
        exported = [json.loads(line) for line in
                    (Path(directory) / "events" / f"{self.day}.txt").read_text(encoding="utf-8").splitlines()
                    if line and not line.startswith("#")]
        self.assertEqual(exported, records)
        return records

    def events_of(self, records, kind):
        return [event for event in records if event["event"] == kind]

    def assert_jpeg(self, directory, path, expected):
        actual = cv2.imread(str(Path(directory) / path))
        self.assertIsNotNone(actual)
        success, encoded = cv2.imencode(".jpg", expected)
        self.assertTrue(success)
        np.testing.assert_array_equal(actual, cv2.imdecode(encoded, cv2.IMREAD_COLOR))

    def test_runtime_roi_and_reference_are_enhanced_once_but_saved_pngs_are_raw(self):
        empty, phone = self.scenes()
        x, y, width, height = self.roi
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_monitor(directory, [(0, empty, False), (1, phone, True), (2, phone, True)],
                                      mode="adaptive", strength=.5, keys=[ord("e"), ord("f"), ord("q")],
                                      options=["--roi", *map(str, self.roi)])
            raw_phone = phone[y:y + height, x:x + width]
            raw_empty = empty[y:y + height, x:x + width]
            enhancer = ImageEnhancer("adaptive", .5)
            expected = enhancer.process(raw_phone)
            self.assertFalse(np.array_equal(expected, enhancer.process(expected)))
            for image in result["inputs"][1:]:
                np.testing.assert_array_equal(image, expected)
            references = result["references"]
            self.assertEqual(references.counts, {"empty": 1, "phone": 1})
            np.testing.assert_array_equal(references.compatible("phone", raw_phone.shape)[0], expected)
            np.testing.assert_array_equal(references.raw_view().compatible("phone", raw_phone.shape)[0], raw_phone)
            for name, raw in (("phone", raw_phone), ("empty", raw_empty)):
                path, = (Path(directory) / "references" / name).glob("*.png")
                np.testing.assert_array_equal(cv2.imread(str(path)), raw)
            session = self.events_of(self.records(directory), "session_start")[0]["details"]
            self.assertEqual(session["image_mode"], "adaptive")
            self.assertEqual(session["image_strength"], .5)
            self.assertEqual(session["enhancement"], enhancer.config)

    def test_tilted_plane_rectifies_raw_pixels_before_runtime_and_reference_enhancement(self):
        _, phone = self.scenes()
        plane = np.array([[170, 65], [310, 85], [300, 250], [170, 240]], np.float32)
        raw_plane, _ = rectify(phone, plane)
        enhanced_plane = ImageEnhancer("adaptive", .5).process(raw_plane)
        with tempfile.TemporaryDirectory() as directory:
            save_plane(Path(directory) / "parking-plane.json", plane, phone.shape)
            result = self.run_monitor(directory, [(0, phone, True), (1, phone, True)],
                                      mode="adaptive", keys=[ord("f"), ord("q")])
            for actual in result["inputs"]:
                np.testing.assert_array_equal(actual, enhanced_plane)
            path, = (Path(directory) / "references" / "phone").glob("*.png")
            np.testing.assert_array_equal(cv2.imread(str(path)), raw_plane)
            np.testing.assert_array_equal(result["references"].raw_view().compatible(
                "phone", raw_plane.shape)[0], raw_plane)
            np.testing.assert_array_equal(result["references"].compatible(
                "phone", raw_plane.shape)[0], enhanced_plane)

    def test_pickup_return_and_periodic_photos_keep_raw_and_enhanced_companions(self):
        empty, phone = self.scenes()
        enhancer = ImageEnhancer("adaptive", .5)
        with tempfile.TemporaryDirectory() as directory:
            self.seed(directory, empty, phone)
            self.run_monitor(directory, [(0, phone, None), (1, phone, None), (2, empty, None),
                                         (3, empty, None), (4, phone, None), (5, phone, None)],
                             mode="adaptive", options=["--photo-interval", "2"])
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            returns = self.events_of(records, "return")
            periodic = self.events_of(records, "photo_saved")
            self.assertEqual(len(pickups), 1)
            self.assertEqual(len(returns), 1)
            self.assertEqual(len(periodic), 2)
            expected_photos = [(pickups[0], empty), (returns[0], phone),
                               (periodic[0], empty), (periodic[1], phone)]
            for event, raw in expected_photos:
                details = event["details"]
                self.assert_jpeg(directory, details["photo_path"], raw)
                self.assert_jpeg(directory, details["enhanced_photo_path"], enhancer.process(raw))
                self.assertEqual(Path(details["enhanced_photo_path"]).parent, Path("photos/enhanced"))
                self.assertEqual(Path(details["enhanced_photo_path"]).name, Path(details["photo_path"]).name)
            details = pickups[0]["details"]
            self.assert_jpeg(directory, details["before_photo_path"], phone)
            self.assert_jpeg(directory, details["before_enhanced_photo_path"], enhancer.process(phone))
            self.assertTrue(details["before_photo_time"].startswith("2026-10-11T12:00:01"))
            self.assertEqual(details["cycle_id"], returns[0]["details"]["cycle_id"])

    def test_raw_clipped_quality_stays_unknown_and_skips_phone_model(self):
        rng = np.random.default_rng(12)
        noise = rng.integers(0, 7, (380, 480, 3), dtype=np.uint8)
        bright = np.full_like(noise, 249)
        for raw in (noise, bright):
            with self.subTest(luminance=float(raw.mean())), tempfile.TemporaryDirectory() as directory:
                result = self.run_monitor(directory, [(0, raw, True), (1, raw, True)], mode="adaptive")
                self.assertEqual(result["inputs"], [])
                records = self.records(directory)
                self.assertEqual(self.events_of(records, "state_change"), [])
                self.assertEqual(self.events_of(records, "pickup"), [])
                summary = self.events_of(records, "summary")[-1]["details"]
                self.assertEqual(summary["state"], "unknown")
                self.assertEqual(summary["unknown_seconds"], 1)

    def test_parking_counts_and_timing_are_identical_with_enhancement_and_clipping_cannot_fake_empty(self):
        empty, phone = self.scenes()
        signatures = []
        for mode in ("off", "adaptive"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                self.seed(directory, empty, phone)
                self.run_monitor(directory, [(0, phone, None), (1, phone, None), (2, empty, None),
                                             (3, empty, None), (4, phone, None), (5, phone, None)], mode=mode)
                records = self.records(directory)
                signatures.append([(event["event"], event["time"], {
                    key: event["details"].get(key) for key in (
                        "from_state", "to_state", "unknown_seconds", "confirmed_after_seconds",
                        "rapid_empty", "away_seconds", "parking_evidence")})
                    for event in records if event["event"] in ("pickup", "return")])
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual([event[0] for event in signatures[0]], ["pickup", "return"])
        with tempfile.TemporaryDirectory() as directory:
            self.seed(directory, empty, phone)
            dark = np.full_like(empty, 3)
            clipped = np.full_like(empty, 250)
            result = self.run_monitor(directory, [(0, phone, True), (1, phone, True),
                                                 (2, dark, True), (3, clipped, True)], mode="adaptive")
            self.assertEqual(len(result["inputs"]), 2)
            records = self.records(directory)
            self.assertEqual(self.events_of(records, "pickup"), [])
            self.assertEqual(self.events_of(records, "unobserved_change"), [])
            self.assertEqual(self.events_of(records, "summary")[-1]["details"]["state"], "unknown")

    def test_companion_write_failure_keeps_raw_photos_count_and_loop_alive(self):
        empty, phone = self.scenes()
        with tempfile.TemporaryDirectory() as directory:
            self.seed(directory, empty, phone)
            self.run_monitor(directory, [(0, phone, None), (1, phone, None), (2, empty, None),
                                         (3, empty, None), (4, phone, None), (5, phone, None)],
                             mode="adaptive", companion_failure=True, options=["--photo-interval", "2"])
            records = self.records(directory)
            pickups = self.events_of(records, "pickup")
            returns = self.events_of(records, "return")
            self.assertEqual(len(pickups), 1)
            self.assertEqual(len(returns), 1)
            for event in pickups + returns + self.events_of(records, "photo_saved"):
                details = event["details"]
                self.assertTrue((Path(directory) / details["photo_path"]).is_file())
                self.assertIsNone(details["enhanced_photo_path"])
                self.assertIn("enhanced photo", details["enhanced_photo_error"])
            details = pickups[0]["details"]
            self.assert_jpeg(directory, details["before_photo_path"], phone)
            self.assertIsNone(details["before_enhanced_photo_path"])
            self.assertIn("enhanced photo", details["before_enhanced_photo_error"])
            self.assertEqual(self.events_of(records, "error"), [])
            self.assertEqual(self.events_of(records, "session_end")[-1]["details"]["reason"], "normal")

    def test_default_off_preserves_raw_inputs_and_writes_no_extra_photos(self):
        empty, phone = self.scenes()
        with tempfile.TemporaryDirectory() as directory:
            self.seed(directory, empty, phone)
            result = self.run_monitor(directory, [(0, phone, None), (1, phone, None), (2, empty, None),
                                                 (3, empty, None), (4, phone, None), (5, phone, None)])
            for actual, expected in zip(result["inputs"], (phone, phone, empty, empty, phone, phone)):
                np.testing.assert_array_equal(actual, expected)
            records = self.records(directory)
            pickup = self.events_of(records, "pickup")[0]
            self.assertNotIn("enhanced_photo_path", pickup["details"])
            self.assertNotIn("before_enhanced_photo_path", pickup["details"])
            self.assertFalse((Path(directory) / "photos" / "enhanced").exists())
            self.assertEqual(len(list((Path(directory) / "photos").glob("*.jpg"))), 3)
            self.assertEqual(self.events_of(records, "session_start")[0]["details"]["image_mode"], "off")
        with tempfile.TemporaryDirectory() as directory:
            raw = np.full_like(empty, 3)
            result = self.run_monitor(directory, [(0, raw, True)])
            self.assertEqual(len(result["inputs"]), 1)
            self.assertEqual(self.events_of(self.records(directory), "summary")[-1]["details"]["state"], "present")

    def test_v_changes_preview_only_and_roi_selection_and_reference_capture_stay_raw(self):
        _, phone = self.scenes()
        x, y, width, height = self.roi
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_monitor(directory, [(0, phone, True), (1, phone, True),
                                                 (2, phone, True), (3, phone, True)],
                                      mode="adaptive", keys=[ord("v"), ord("r"), ord("f"), ord("q")],
                                      selected_roi=self.roi)
            full_enhanced = ImageEnhancer("adaptive", .5).process(phone)
            np.testing.assert_array_equal(result["previews"][0][300, 470], full_enhanced[300, 470])
            np.testing.assert_array_equal(result["previews"][1][300, 470], phone[300, 470])
            self.assertEqual(len(result["selections"]), 1)
            np.testing.assert_array_equal(result["selections"][0], phone)
            crop = phone[y:y + height, x:x + width]
            expected = ImageEnhancer("adaptive", .5).process(crop)
            for actual in result["inputs"][2:]:
                np.testing.assert_array_equal(actual, expected)
            path, = (Path(directory) / "references" / "phone").glob("*.png")
            np.testing.assert_array_equal(cv2.imread(str(path)), crop)
            np.testing.assert_array_equal(result["references"].compatible("phone", crop.shape)[0], expected)

    def test_watch_hands_receive_enhanced_frame_but_screen_motion_receives_raw_frame(self):
        _, phone = self.scenes()
        with tempfile.TemporaryDirectory() as directory:
            screen = np.array([[190, 90], [290, 90], [290, 230], [190, 230]], np.float32)
            save_plane(Path(directory) / "screen-plane.json", screen, phone.shape)
            result = self.run_monitor(directory, [(0, phone, True), (1, phone, True)], mode="adaptive",
                                      watch=True, options=["--roi", *map(str, self.roi)])
            self.assertEqual(len(result["hand_inputs"]), 2)
            self.assertEqual(len(result["screen_inputs"]), 2)
            enhanced = ImageEnhancer("adaptive", .5).process(phone)
            for actual in result["hand_inputs"]:
                np.testing.assert_array_equal(actual, enhanced)
            for actual in result["screen_inputs"]:
                np.testing.assert_array_equal(actual, phone)

    def test_suspect_photos_keep_raw_store_path_with_companion_success_or_failure(self):
        _, phone = self.scenes()
        screen = np.array([[190, 90], [290, 90], [290, 230], [190, 230]], np.float32)
        for fail in (False, True):
            with self.subTest(companion_failure=fail), tempfile.TemporaryDirectory() as directory:
                save_plane(Path(directory) / "screen-plane.json", screen, phone.shape)
                self.run_monitor(directory, [(0, phone, True), (1, phone, True)],
                                 mode="adaptive", watch=True, companion_failure=fail,
                                 screen_activity=[True, False],
                                 options=["--roi", *map(str, self.roi), "--watch-delay", "0"])
                records = self.records(directory)
                starts = self.events_of(records, "suspect_start")
                ends = self.events_of(records, "suspect_end")
                self.assertEqual(len(starts), 1)
                self.assertEqual(len(ends), 1)
                details = starts[0]["details"]
                self.assertEqual(details["sources"], ["screen_motion"])
                self.assert_jpeg(directory, details["photo_path"], phone)
                self.assertIsNone(details["photo_error"])
                if fail:
                    self.assertIsNone(details["enhanced_photo_path"])
                    self.assertIn("enhanced photo", details["enhanced_photo_error"])
                else:
                    self.assert_jpeg(directory, details["enhanced_photo_path"],
                                     ImageEnhancer("adaptive", .5).process(phone))
                with sqlite3.connect(Path(directory) / "usage.sqlite3") as db:
                    watch = db.execute("SELECT active,photo_path FROM watch_events ORDER BY at").fetchall()
                    seconds = db.execute("SELECT seconds FROM watch_daily WHERE day=?", (self.day,)).fetchone()[0]
                self.assertEqual(watch, [(1, details["photo_path"]), (0, None)])
                self.assertEqual(seconds, 1)
                self.assertEqual(self.events_of(records, "error"), [])
                self.assertEqual(self.events_of(records, "session_end")[-1]["details"]["reason"], "normal")


if __name__ == "__main__":
    unittest.main()
