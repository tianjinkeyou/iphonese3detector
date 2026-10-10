import unittest

import cv2
import numpy as np

from phonewatch.screen_activity import ScreenActivityDetector


class ScreenActivityTests(unittest.TestCase):
    polygon = np.float32([[50, 50], [249, 50], [249, 149], [50, 149]])

    @staticmethod
    def frame():
        # Deterministic textured background enables real camera-motion checks.
        rng = np.random.default_rng(72)
        gray = rng.integers(25, 155, (200, 300), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray[50:150, 50:250] = 45
        cv2.putText(gray, "PHONE", (65, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, 180, 2)
        cv2.rectangle(gray, (75, 115), (100, 135), 160, -1)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_first_observation_then_static_frame(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        self.assertIsNone(detector.detect(frame, self.polygon)[0])
        moving, score, label = detector.detect(frame, self.polygon)
        self.assertFalse(moving)
        self.assertEqual(score, 0)
        self.assertIn("stable", label)

    def test_spatial_content_changes_are_activity(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        detector.detect(frame, self.polygon)
        changed = frame.copy()
        cv2.rectangle(changed, (130, 105), (190, 135), (210, 210, 210), -1)
        moving, score, label = detector.detect(changed, self.polygon)
        self.assertTrue(moving)
        self.assertGreater(score, .025)
        self.assertIn("possible use", label)

    def test_camera_exposure_shift_is_not_activity(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        detector.detect(frame, self.polygon)
        changed = cv2.add(frame, np.full(frame.shape, 25, dtype=np.uint8))
        self.assertFalse(detector.detect(changed, self.polygon)[0])

    def test_uniform_screen_transition_with_static_surroundings_is_activity(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        frame[50:150, 50:250] = 45
        detector.detect(frame, self.polygon)
        changed = frame.copy()
        changed[50:150, 50:250] = 150
        moving, score, _ = detector.detect(changed, self.polygon)
        self.assertTrue(moving)
        self.assertGreater(score, .95)

    def test_flat_screen_whole_camera_exposure_is_not_activity(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        frame[50:150, 50:250] = 45
        detector.detect(frame, self.polygon)
        changed = cv2.add(frame, np.full(frame.shape, 35, dtype=np.uint8))
        moving, score, _ = detector.detect(changed, self.polygon)
        self.assertFalse(moving)
        self.assertLess(score, .025)

    def test_uniform_screen_change_without_surroundings_is_uncertain(self):
        detector = ScreenActivityDetector()
        frame = np.full((100, 200, 3), 45, dtype=np.uint8)
        polygon = [[0, 0], [199, 0], [199, 99], [0, 99]]
        detector.detect(frame, polygon)
        moving, _, label = detector.detect(np.full_like(frame, 150), polygon)
        self.assertIsNone(moving)
        self.assertIn("include background", label)

    def test_screen_color_transition_with_static_surroundings_is_activity(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        frame[50:150, 50:250] = (180, 40, 40)
        detector.detect(frame, self.polygon)
        changed = frame.copy()
        changed[50:150, 50:250] = (40, 180, 40)
        self.assertTrue(detector.detect(changed, self.polygon)[0])

    def test_small_camera_translation_uses_surrounding_background(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        detector.detect(frame, self.polygon)
        moved = cv2.warpAffine(frame, np.float32([[1, 0, 3], [0, 1, 2]]), (300, 200))
        moving, score, _ = detector.detect(moved, self.polygon)
        self.assertFalse(moving)
        self.assertLess(score, .025)

    def test_small_camera_rotation_is_compensated(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        detector.detect(frame, self.polygon)
        transform = cv2.getRotationMatrix2D((150, 100), 1, 1)
        moved = cv2.warpAffine(frame, transform, (300, 200))
        moving, score, _ = detector.detect(moved, self.polygon)
        self.assertFalse(moving)
        self.assertLess(score, .025)

    def test_scrolling_content_is_not_registered_away(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        detector.detect(frame, self.polygon)
        changed = frame.copy()
        screen = frame[50:150, 50:250]
        changed[50:150, 50:250] = cv2.warpAffine(screen, np.float32([[1, 0, 3], [0, 1, 2]]), (200, 100), borderValue=(45, 45, 45))
        self.assertTrue(detector.detect(changed, self.polygon)[0])

    def test_motion_outside_screen_is_not_activity(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        detector.detect(frame, self.polygon)
        changed = frame.copy()
        changed[:45, :] = 220
        changed[155:, :] = 220
        self.assertFalse(detector.detect(changed, self.polygon)[0])

    def test_background_only_brightness_change_is_not_screen_activity(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        detector.detect(frame, self.polygon)
        changed = cv2.add(frame, np.full(frame.shape, 20, dtype=np.uint8))
        changed[50:150, 50:250] = frame[50:150, 50:250]
        self.assertFalse(detector.detect(changed, self.polygon)[0])

    def test_landscape_crop_is_not_forced_to_portrait(self):
        detector = ScreenActivityDetector()
        detector.detect(self.frame(), self.polygon)
        height, width = detector._previous_crop.shape
        self.assertGreater(width, height)
        self.assertLessEqual(max(width, height), 320)

    def test_large_screen_is_resized_without_changing_aspect(self):
        detector = ScreenActivityDetector()
        detector.detect(np.full((600, 1000, 3), 100, np.uint8), [[100, 100], [900, 100], [900, 500], [100, 500]])
        self.assertEqual(detector._previous_crop.shape, (160, 320))

    def test_invalid_geometry_resets_history(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        detector.detect(frame, self.polygon)
        for points in (None, [], [[1, 1]] * 4, [[-1, 1], [60, 1], [60, 60], [-1, 60]], [[50, 50], [249, 149], [249, 50], [50, 149]]):
            with self.subTest(points=points):
                self.assertIsNone(detector.detect(frame, points)[0])
                self.assertIsNone(detector.detect(frame, self.polygon)[0])

    def test_changed_polygon_or_camera_size_restarts_history(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        detector.detect(frame, self.polygon)
        self.assertIsNone(detector.detect(frame, self.polygon + [1, 0])[0])
        self.assertFalse(detector.detect(frame, self.polygon + [1, 0])[0])
        larger = np.pad(frame, ((0, 20), (0, 20), (0, 0)))
        self.assertIsNone(detector.detect(larger, self.polygon + [1, 0])[0])

    def test_blackout_is_unobservable_and_recovery_restarts_history(self):
        detector = ScreenActivityDetector()
        frame = self.frame()
        detector.detect(frame, self.polygon)
        moving, score, label = detector.detect(np.zeros_like(frame), self.polygon)
        self.assertIsNone(moving)
        self.assertEqual(score, 0)
        self.assertIn("dark", label)
        self.assertIsNone(detector.detect(frame, self.polygon)[0])
        self.assertFalse(detector.detect(frame, self.polygon)[0])

    def test_saturated_white_is_unobservable(self):
        detector = ScreenActivityDetector()
        white = np.full_like(self.frame(), 255)
        self.assertIsNone(detector.detect(white, self.polygon)[0])
        moving, _, label = detector.detect(white, self.polygon)
        self.assertIsNone(moving)
        self.assertIn("overexposed", label)

    def test_reset_discards_history(self):
        detector = ScreenActivityDetector()
        detector.detect(self.frame(), self.polygon)
        detector.reset()
        self.assertIsNone(detector.detect(self.frame(), self.polygon)[0])

    def test_threshold_configuration_is_validated(self):
        for value in (0, -1, 2, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ScreenActivityDetector(threshold=value)
                with self.assertRaises(ValueError):
                    ScreenActivityDetector(intensity_threshold=value)


if __name__ == "__main__":
    unittest.main()
