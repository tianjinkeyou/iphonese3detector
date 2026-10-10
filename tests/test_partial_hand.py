import unittest

import cv2
import numpy as np

from phonewatch.partial_hand import partial_hand_regions


class PartialHandTests(unittest.TestCase):
    skin = (105, 155, 205)  # BGR appearance; not evidence of human identity.

    @staticmethod
    def phone(shape=(160, 200, 3)):
        image = np.full(shape, 35, np.uint8)
        cv2.rectangle(image, (10, 10), (shape[1] - 11, shape[0] - 11), (15, 15, 15), -1)
        return image

    def test_partial_finger_entering_roi_without_palm_is_candidate(self):
        reference = self.phone()
        current = reference.copy()
        cv2.rectangle(current, (130, 65), (199, 85), self.skin, -1)
        regions = partial_hand_regions(current, [reference])
        self.assertEqual(len(regions), 1)
        x, y, width, height = cv2.boundingRect(regions[0])
        self.assertGreaterEqual(x, 128)
        self.assertLessEqual(y, 67)
        self.assertGreater(width, 60)
        self.assertGreater(height, 15)

    def test_static_warm_background_and_screen_are_not_candidates(self):
        reference = self.phone()
        cv2.rectangle(reference, (20, 20), (100, 100), self.skin, -1)
        self.assertEqual(partial_hand_regions(reference.copy(), [reference]), [])
        brighter = np.clip(reference.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        self.assertEqual(partial_hand_regions(brighter, [reference]), [])

    def test_finger_remains_candidate_with_uniform_exposure_change(self):
        reference = self.phone()
        current = np.clip(reference.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        cv2.rectangle(current, (130, 65), (199, 85), self.skin, -1)
        self.assertEqual(len(partial_hand_regions(current, [reference])), 1)

    def test_novel_white_glare_gray_or_dark_patch_is_not_skin_like(self):
        reference = self.phone()
        for colour in ((255, 255, 255), (150, 150, 150), (0, 0, 0)):
            with self.subTest(colour=colour):
                current = reference.copy()
                cv2.rectangle(current, (90, 40), (199, 95), colour, -1)
                self.assertEqual(partial_hand_regions(current, [reference]), [])

    def test_missing_or_incompatible_references_do_not_activate_fallback(self):
        current = self.phone()
        cv2.rectangle(current, (130, 65), (199, 85), self.skin, -1)
        for references in (None, [], [self.phone((80, 100, 3))], [None]):
            with self.subTest(references_type=type(references)):
                self.assertEqual(partial_hand_regions(current, references), [])

    def test_tiny_specks_and_small_pixel_area_are_ignored(self):
        reference = self.phone()
        current = reference.copy()
        cv2.rectangle(current, (120, 50), (131, 61), self.skin, -1)
        current[20::20, 20::20] = self.skin
        self.assertEqual(partial_hand_regions(current, [reference]), [])
        small = self.phone((40, 40, 3))
        current = small.copy()
        cv2.rectangle(current, (12, 12), (18, 18), self.skin, -1)
        self.assertEqual(partial_hand_regions(current, [small]), [])

    def test_minimum_area_scales_with_roi(self):
        for scale in (1, 2, 3):
            with self.subTest(scale=scale):
                reference = self.phone((160 * scale, 200 * scale, 3))
                current = reference.copy()
                cv2.rectangle(current, (130 * scale, 65 * scale),
                              (200 * scale - 1, 85 * scale), self.skin, -1)
                self.assertEqual(len(partial_hand_regions(current, [reference])), 1)
        reference = self.phone((320, 400, 3))
        current = reference.copy()
        # Small patches stay below the threshold even after its large-ROI cap.
        cv2.rectangle(current, (130, 65), (145, 80), self.skin, -1)
        self.assertEqual(partial_hand_regions(current, [reference]), [])

    def test_large_parking_roi_does_not_dilute_a_narrow_partial_finger(self):
        reference = self.phone((480, 640, 3))
        current = reference.copy()
        cv2.rectangle(current, (609, 190), (639, 290), self.skin, -1)
        regions = partial_hand_regions(current, [reference])
        self.assertEqual(len(regions), 1)
        x, y, width, height = cv2.boundingRect(regions[0])
        self.assertGreaterEqual(x, 607)
        self.assertGreater(width, 25)
        self.assertGreater(height, 95)

    def test_each_pixel_explained_by_any_f_reference_is_not_novel(self):
        first = self.phone()
        second = first.copy()
        cv2.rectangle(second, (130, 65), (199, 85), self.skin, -1)
        self.assertEqual(partial_hand_regions(second, [first, second]), [])
        self.assertEqual(partial_hand_regions(second, [second, first]), [])
        # Different samples can explain different static warm regions.
        cv2.rectangle(first, (20, 30), (80, 50), self.skin, -1)
        composite = second.copy()
        cv2.rectangle(composite, (20, 30), (80, 50), self.skin, -1)
        self.assertEqual(partial_hand_regions(composite, [first, second]), [])

    def test_near_full_roi_colour_change_is_not_usable_finger_evidence(self):
        reference = self.phone()
        current = np.full_like(reference, self.skin)
        self.assertEqual(partial_hand_regions(current, [reference]), [])

    def test_novel_skin_coloured_object_can_match_appearance_fallback(self):
        # This explicitly preserves the limit: colour/novelty cannot prove a
        # hand, so callers must label the signal as possible interaction.
        reference = self.phone()
        current = reference.copy()
        cv2.rectangle(current, (80, 50), (120, 85), self.skin, -1)
        self.assertEqual(len(partial_hand_regions(current, [reference])), 1)


if __name__ == "__main__":
    unittest.main()
