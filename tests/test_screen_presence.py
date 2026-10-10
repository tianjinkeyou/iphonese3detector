import unittest

import cv2
import numpy as np

from phonewatch.screen_presence import screen_body_presence


class ScreenPresenceTests(unittest.TestCase):
    def setUp(self):
        self.empty = np.full((240, 200, 3), 145, np.uint8)
        self.phone = self.empty.copy()
        cv2.rectangle(self.phone, (40, 20), (160, 220), (25, 25, 25), -1)
        # A body detail ensures an unrelated uniformly dark object isn't enough.
        cv2.circle(self.phone, (100, 205), 8, (110, 110, 110), 2)
        self.polygon = np.float32([[50, 40], [150, 40], [150, 180], [50, 180]])

    def classify(self, image, empty=None, phone=None, polygon=None):
        return screen_body_presence(image, self.polygon if polygon is None else polygon,
                                    [self.empty] if empty is None else empty,
                                    [self.phone] if phone is None else phone)[0]

    def test_bright_and_dynamic_screen_preserves_visible_body(self):
        for color in (255, 100):
            with self.subTest(color=color):
                frame = self.phone.copy()
                frame[40:181, 50:151] = color
                cv2.putText(frame, 'VIDEO', (55, 110), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 70, 170), 2)
                self.assertTrue(self.classify(frame))

    def test_removed_phone_is_empty(self):
        self.assertFalse(self.classify(self.empty))

    def test_changed_background_cannot_establish_phone_presence(self):
        frame = self.empty.copy()
        frame[:, :30] = 25
        frame[:, 170:] = 25
        self.assertFalse(self.classify(frame))
        unrelated = self.empty.copy()
        cv2.rectangle(unrelated, (40, 20), (160, 220), (80, 100, 130), -1)
        self.assertIsNone(self.classify(unrelated))

    def test_incorrect_shape_and_missing_references_are_unknown(self):
        self.assertIsNone(self.classify(self.phone, empty=[]))
        self.assertIsNone(self.classify(self.phone, phone=[]))
        self.assertIsNone(self.classify(self.phone, phone=[self.phone[:100]]))
        self.assertIsNone(self.classify(self.phone, empty=[self.empty[:100]]))

    def test_ambiguous_references_are_unknown(self):
        self.assertIsNone(self.classify(self.phone, empty=[self.phone]))
        self.assertIsNone(self.classify(self.phone, empty=[self.empty, self.phone]))

    def test_tiny_body_and_screen_only_crop_are_unknown(self):
        tiny = self.empty.copy()
        cv2.rectangle(tiny, (48, 38), (152, 182), (25, 25, 25), -1)
        self.assertIsNone(self.classify(tiny, phone=[tiny]))
        frame = np.full((140, 100, 3), 60, np.uint8)
        points = [[0, 0], [99, 0], [99, 139], [0, 139]]
        self.assertIsNone(screen_body_presence(frame, points, [frame + 30], [frame])[0])

    def test_multiple_phone_references_match_either_pose(self):
        second = self.phone.copy()
        cv2.rectangle(second, (35, 20), (45, 220), (60, 60, 60), -1)
        cv2.rectangle(second, (155, 20), (165, 220), (60, 60, 60), -1)
        for frame in (self.phone, second):
            self.assertTrue(self.classify(frame, phone=[self.phone, second]))

    def test_modest_uniform_exposure_changes_are_compensated(self):
        for shift in (-15, 15):
            with self.subTest(shift=shift):
                frame = np.clip(self.phone.astype(np.int16) + shift, 0, 255).astype(np.uint8)
                self.assertTrue(self.classify(frame))
        brighter_reference = cv2.add(self.phone, np.full_like(self.phone, 15))
        self.assertTrue(self.classify(self.phone, phone=[brighter_reference]))

    def test_tight_rectified_phone_crop_matches_body_details(self):
        # P rectifies exactly the handset corners, leaving no exterior backdrop.
        # The original references' phone rectangle becomes this tight crop.
        empty = self.empty[20:221, 40:161].copy()
        phone = self.phone[20:221, 40:161].copy()
        polygon = self.polygon - [40, 20]
        # The lower home-button detail is outside the independently selected
        # active-screen rectangle; content there can change arbitrarily.
        for screen in (80, 255):
            with self.subTest(screen=screen):
                current = phone.copy()
                current[20:161, 10:111] = screen
                cv2.putText(current, 'VIDEO', (12, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (40, 80, 150), 2)
                self.assertTrue(screen_body_presence(current, polygon, [empty], [phone])[0])
        self.assertFalse(screen_body_presence(empty, polygon, [empty], [phone])[0])

    def test_tight_flat_body_cannot_use_brightness_as_presence(self):
        empty = self.empty[20:221, 40:161].copy()
        phone = np.full_like(empty, 25)
        polygon = self.polygon - [40, 20]
        current = phone.copy()
        current[20:161, 10:111] = 255
        self.assertIsNone(screen_body_presence(current, polygon, [empty], [phone])[0])
        self.assertIsNone(screen_body_presence(empty, polygon, [empty], [phone])[0])

    def test_tight_body_detail_labelled_both_empty_and_phone_is_unknown(self):
        empty = self.empty[20:221, 40:161].copy()
        phone = self.phone[20:221, 40:161].copy()
        polygon = self.polygon - [40, 20]
        conflicting_empty = cv2.add(phone, np.full_like(phone, 100))
        self.assertIsNone(screen_body_presence(phone, polygon,
                                              [empty, conflicting_empty], [phone])[0])

    def test_tight_crop_new_body_texture_remains_unknown(self):
        empty = self.empty[20:221, 40:161].copy()
        phone = self.phone[20:221, 40:161].copy()
        polygon = self.polygon - [40, 20]
        unrelated = phone.copy()
        cv2.rectangle(unrelated, (2, 2), (118, 12), (160, 160, 160), -1)
        self.assertIsNone(screen_body_presence(unrelated, polygon, [empty], [phone])[0])

    def test_blackout_and_invalid_geometry_are_unknown(self):
        self.assertIsNone(self.classify(np.zeros_like(self.phone)))
        for points in (None, [], [[1, 1]] * 4, [[-1, 1], [60, 1], [60, 60], [-1, 60]],
                       [[50, 40], [150, 180], [150, 40], [50, 180]]):
            self.assertIsNone(screen_body_presence(self.phone, points, [self.empty], [self.phone])[0])


if __name__ == '__main__':
    unittest.main()
