import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from phonewatch.parking import ParkingDetector
from phonewatch.reference import ReferenceDetector


class ParkingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.refs = ReferenceDetector(Path(self.tmp.name))
        rng = np.random.default_rng(42)
        # Textured wall and stand continue to occupy R after removal. A broad
        # phone classifier can therefore return UNKNOWN for this empty scene.
        texture = rng.integers(-14, 15, (180, 220, 1))
        self.empty = np.uint8(np.clip(np.full((180, 220, 3), 155) + texture, 0, 255))
        cv2.rectangle(self.empty, (70, 145), (135, 168), (95, 100, 105), -1)
        self.phone = self.with_phone(self.empty)
        self.screen = np.array([[67, 35], [118, 35], [118, 133], [67, 133]], np.float32)
        self.refs.save('empty', self.empty)
        self.refs.save('phone', self.phone)
        self.detector = ParkingDetector(self.refs)

    @staticmethod
    def with_phone(background, offset=0):
        result = background.copy()
        cv2.rectangle(result, (55 + offset, 20), (130 + offset, 145), (24, 24, 24), -1)
        cv2.rectangle(result, (67 + offset, 35), (118 + offset, 133), (10, 10, 10), -1)
        cv2.circle(result, (92 + offset, 139), 3, (65, 65, 65), 1)
        return result

    def test_parked_body_and_empty_footprint(self):
        self.assertTrue(self.detector.ready(self.phone.shape))
        parked, _, evidence = self.detector.detect(self.phone)
        self.assertIs(parked, True)
        self.assertFalse(evidence['reliable_empty'])
        self.assertFalse(evidence['immediate_empty'])
        empty, label, evidence = self.detector.detect(self.empty)
        self.assertIs(empty, False)
        self.assertTrue(evidence['reliable_empty'])
        self.assertTrue(evidence['immediate_empty'])
        self.assertIn('E=', label)
        self.assertIn('F=', label)
        json.dumps(evidence, allow_nan=False)

    def test_weaker_empty_match_does_not_bypass_time_confirmation(self):
        image = self.empty.copy()
        cv2.rectangle(image, (80, 60), (89, 89), (215, 215, 215), -1)
        state, _, evidence = self.detector.detect(image)
        self.assertIs(state, False)
        self.assertTrue(evidence['reliable_empty'])
        self.assertFalse(evidence['immediate_empty'])

    def test_phone_elsewhere_in_roi_does_not_hide_a_pickup(self):
        moved = self.with_phone(self.empty, offset=90)
        polygon = np.array([[145, 20], [220, 20], [220, 145], [145, 145]])
        result = self.detector.detect(moved, phone_polygons=[polygon])
        self.assertIs(result[0], False)
        self.assertTrue(result[2]['reliable_empty'])

    def test_phone_model_at_old_footprint_conflicts_with_empty(self):
        polygon = np.array([[55, 20], [130, 20], [130, 145], [55, 145]])
        state, _, evidence = self.detector.detect(self.empty, phone_polygons=[polygon])
        self.assertIsNone(state)
        self.assertFalse(evidence['reliable_empty'])

    def test_occluding_hand_is_unknown_instead_of_empty(self):
        hand = self.empty.copy()
        cv2.rectangle(hand, (58, 24), (135, 146), (100, 150, 210), -1)
        state, _, evidence = self.detector.detect(hand)
        self.assertIsNone(state)
        self.assertFalse(evidence['reliable_empty'])

    def test_new_unrecognised_object_is_not_empty(self):
        object_image = self.empty.copy()
        cv2.rectangle(object_image, (55, 20), (130, 145), (25, 125, 210), -1)
        state, _, evidence = self.detector.detect(object_image)
        self.assertIsNone(state)
        self.assertFalse(evidence['reliable_empty'])

    def test_live_screen_content_does_not_remove_a_parked_phone(self):
        changed = self.phone.copy()
        rng = np.random.default_rng(12)
        changed[35:134, 67:119] = rng.integers(20, 256, (99, 52, 3), dtype=np.uint8)
        self.assertIs(self.detector.detect(changed)[0], True)
        self.assertIs(self.detector.detect(changed, screen_polygon=self.screen)[0], True)

    def test_glare_in_screen_still_leaves_body_evidence(self):
        glare = self.phone.copy()
        glare[35:134, 67:119] = 255
        self.assertIs(self.detector.detect(glare, screen_polygon=self.screen)[0], True)

    def test_partial_hand_obscures_screen_but_not_phone_identity(self):
        image = self.phone.copy()
        cv2.rectangle(image, (78, 50), (110, 100), (100, 150, 210), -1)
        self.assertIs(self.detector.detect(image, screen_polygon=self.screen)[0], True)

    def test_uniform_exposure_change_is_not_a_pickup(self):
        for value in (-25, 25):
            changed = np.clip(self.phone.astype(np.int16) + value, 0, 255).astype(np.uint8)
            self.assertIs(self.detector.detect(changed)[0], True)
            empty = np.clip(self.empty.astype(np.int16) + value, 0, 255).astype(np.uint8)
            self.assertIs(self.detector.detect(empty)[0], False)

    def test_mild_camera_rotation_and_translation_preserve_pose(self):
        matrix = cv2.getRotationMatrix2D((110, 90), 3, 1)
        matrix[:, 2] += [3, -3]
        for image, expected in ((self.phone, True), (self.empty, False)):
            shifted = cv2.warpAffine(image, matrix, (220, 180), borderMode=cv2.BORDER_REFLECT101)
            self.assertIs(self.detector.detect(shifted)[0], expected)

    def test_phone_stand_small_shift_is_still_parked(self):
        for offset in (-3, -2, -1, 1, 2, 3):
            shifted = self.with_phone(self.empty, offset=offset)
            self.assertIs(self.detector.detect(shifted)[0], True)

    def test_all_saved_phone_poses_must_be_empty(self):
        alternative = self.with_phone(self.empty, offset=85)
        self.refs.save('phone', alternative)
        self.assertIs(self.detector.detect(self.phone)[0], True)
        self.assertIs(self.detector.detect(alternative)[0], True)
        self.assertIs(self.detector.detect(self.empty)[0], False)
        occluded_alternative = self.empty.copy()
        cv2.rectangle(occluded_alternative, (145, 30), (210, 140), (100, 150, 210), -1)
        self.assertIsNone(self.detector.detect(occluded_alternative)[0])

    def test_multiple_empty_lighting_samples(self):
        darker = np.clip(self.empty.astype(np.int16) - 35, 0, 255).astype(np.uint8)
        self.refs.save('empty', darker)
        self.assertIs(self.detector.detect(darker)[0], False)
        self.assertIs(self.detector.detect(self.phone)[0], True)

    def test_missing_conflicting_and_mismatched_calibrations(self):
        self.refs.clear()
        self.assertFalse(self.detector.ready(self.empty.shape))
        self.assertIsNone(self.detector.detect(self.empty)[0])
        self.refs.save('empty', self.empty)
        self.refs.save('phone', self.empty)
        self.assertFalse(self.detector.ready(self.empty.shape))
        self.assertIsNone(self.detector.detect(self.empty)[0])
        self.refs.save('phone', self.phone)
        self.assertIsNone(self.detector.detect(self.phone)[0])
        self.assertIsNone(self.detector.detect(np.zeros((200, 220, 3), np.uint8))[0])

    def test_blackout_cannot_be_a_pickup(self):
        state, _, evidence = self.detector.detect(np.zeros_like(self.empty))
        self.assertIsNone(state)
        self.assertFalse(evidence['reliable_empty'])

    def test_clipped_dark_exposure_cannot_manufacture_empty_evidence(self):
        dark = np.uint8(np.clip(self.phone.astype(float) * .35 - 40, 0, 255))
        state, _, evidence = self.detector.detect(dark, screen_polygon=self.screen)
        self.assertIsNone(state)
        self.assertFalse(evidence['reliable_empty'])

    def test_blown_out_camera_cannot_be_a_pickup(self):
        state, _, evidence = self.detector.detect(np.full_like(self.empty, 255))
        self.assertIsNone(state)
        self.assertFalse(evidence['reliable_empty'])

    def test_exposure_that_erases_scene_contrast_cannot_be_a_pickup(self):
        for gain in (.05, .35):
            image = np.uint8(np.clip(self.phone.astype(float) * gain + 150, 0, 255))
            state, _, evidence = self.detector.detect(image, screen_polygon=self.screen)
            self.assertIsNone(state)
            self.assertFalse(evidence['reliable_empty'])

    def test_tight_p_crop_retains_body_evidence_without_background(self):
        empty = np.full((180, 120, 3), 155, np.uint8)
        phone = np.full_like(empty, 24)
        phone[15:164, 12:108] = 10
        screen = np.array([[12, 15], [107, 15], [107, 163], [12, 163]], np.float32)
        self.refs.clear()
        self.refs.save('empty', empty)
        self.refs.save('phone', phone)
        self.assertTrue(self.detector.ready(phone.shape))
        self.assertIs(self.detector.detect(phone, screen_polygon=screen)[0], True)
        changed_screen = phone.copy()
        changed_screen[15:164, 12:108] = 155
        self.assertIs(self.detector.detect(changed_screen, screen_polygon=screen)[0], True)
        self.assertIs(self.detector.detect(empty, screen_polygon=screen)[0], False)

    def test_recalibration_invalidates_old_pose_cache(self):
        self.assertIs(self.detector.detect(self.phone)[0], True)
        self.refs.clear()
        self.refs.save('empty', self.empty)
        other_pose = self.with_phone(self.empty, offset=85)
        self.refs.save('phone', other_pose)
        self.assertIs(self.detector.detect(other_pose)[0], True)
        self.assertIs(self.detector.detect(self.phone)[0], False)

    def test_textureless_body_with_explicit_screen(self):
        background = np.full((180, 220, 3), 180, np.uint8)
        phone = self.with_phone(background)
        self.refs.clear()
        self.refs.save('empty', background)
        self.refs.save('phone', phone)
        phone[35:134, 67:119] = 180  # screen resembles the background
        self.assertIs(self.detector.detect(phone, screen_polygon=self.screen)[0], True)
        self.assertIs(self.detector.detect(background, screen_polygon=self.screen)[0], False)

    def test_correcting_screen_calibration_relearns_without_changing_ef(self):
        wrong = np.array([[45, 10], [140, 10], [140, 155], [45, 155]], np.float32)
        self.assertFalse(self.detector.ready(self.phone.shape, screen_polygon=wrong))
        self.assertTrue(self.detector.ready(self.phone.shape, screen_polygon=self.screen))
        self.assertIs(self.detector.detect(self.phone, screen_polygon=self.screen)[0], True)

    def test_uniform_whole_frame_brightness_change_is_not_phone_calibration(self):
        self.refs.clear()
        empty = np.full_like(self.empty, 155)
        phone = np.full_like(self.empty, 95)
        self.refs.save('empty', empty)
        self.refs.save('phone', phone)
        self.assertFalse(self.detector.ready(empty.shape))
        self.assertIsNone(self.detector.detect(phone)[0])
        self.assertFalse(self.detector.ready(empty.shape, screen_polygon=self.screen))
        self.assertIsNone(self.detector.detect(phone, screen_polygon=self.screen)[0])

    def test_horizontal_background_texture_loss_cannot_be_a_pickup(self):
        empty = np.full_like(self.empty, 155)
        rows = np.where((np.arange(empty.shape[0]) // 4) % 2, 7, -7)
        empty = np.uint8(empty.astype(np.int16) + rows[:, None, None])
        phone = self.with_phone(empty)
        self.refs.clear()
        self.refs.save('empty', empty)
        self.refs.save('phone', phone)
        self.assertIs(self.detector.detect(empty, screen_polygon=self.screen)[0], False)
        contrast_lost = np.uint8(np.clip(phone.astype(float) * .05 + 150, 0, 255))
        state, _, evidence = self.detector.detect(contrast_lost, screen_polygon=self.screen)
        self.assertIsNone(state)
        self.assertFalse(evidence['reliable_empty'])


if __name__ == '__main__':
    unittest.main()
