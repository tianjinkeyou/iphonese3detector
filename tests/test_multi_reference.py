"""Regression coverage for adding E/F examples instead of overwriting them."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from phonewatch.reference import ReferenceDetector
from phonewatch.smart import SmartDetector, moving_reference


class MissModel:
    def detect(self, image):
        return [], 'miss'


class MultiReferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.references = ReferenceDetector(self.directory)
        self.empty = np.full((240, 320, 3), 185, np.uint8)
        self.other_empty = self.empty.copy()
        cv2.rectangle(self.other_empty, (240, 15), (310, 100), (110, 110, 110), -1)
        self.phone = self.empty.copy()
        cv2.rectangle(self.phone, (35, 35), (110, 210), (18, 18, 18), -1)
        self.other_phone = self.empty.copy()
        corners = np.array([[165, 25], [255, 45], [215, 215], [130, 195]])
        cv2.fillConvexPoly(self.other_phone, corners, (25, 25, 25))

    def save_examples(self):
        for name, images in (
            ('empty', (self.empty, self.other_empty)),
            ('phone', (self.phone, self.other_phone)),
        ):
            for image in images:
                self.references.save(name, image)

    def test_multiple_phone_poses_and_empty_backgrounds_remain_usable(self):
        self.save_examples()
        for image in (self.phone, self.other_phone):
            with self.subTest(label='phone', position=np.argmin(image)):
                self.assertTrue(self.references.detect(image)[0])
        for image in (self.empty, self.other_empty):
            with self.subTest(label='empty', position=np.argmin(image)):
                self.assertFalse(self.references.detect(image)[0])

    def test_save_appends_and_reloads_all_examples(self):
        initial_revision = self.references.revision
        self.references.save('empty', self.empty)
        self.assertGreater(self.references.revision, initial_revision)
        self.assertFalse(self.references.ready)
        self.save_examples()
        self.assertEqual(self.references.counts, {'empty': 3, 'phone': 2})
        np.testing.assert_array_equal(self.references.images['phone'], self.other_phone)
        for name, expected_count in self.references.counts.items():
            self.assertEqual(len(list((self.directory / name).glob('*.png'))), expected_count)
            self.assertEqual(len(self.references.samples[name]), expected_count)
        reloaded = ReferenceDetector(self.directory)
        self.assertTrue(reloaded.ready)
        self.assertEqual(reloaded.counts, self.references.counts)
        for name in ('empty', 'phone'):
            for original in self.references.samples[name]:
                self.assertTrue(any(np.array_equal(original, saved) for saved in reloaded.samples[name]))
        self.assertTrue(reloaded.detect(self.phone)[0])
        self.assertTrue(reloaded.detect(self.other_phone)[0])

    def test_saving_copies_frame_instead_of_retaining_mutable_input(self):
        frame = self.phone.copy()
        self.references.save('empty', self.empty)
        self.references.save('phone', frame)
        frame.fill(0)
        self.assertTrue(self.references.detect(self.phone)[0])
        np.testing.assert_array_equal(self.references.samples['phone'][0], self.phone)

    def test_legacy_images_preserved_without_duplicate_imports(self):
        legacy_paths = {'empty': self.directory / 'empty.png', 'phone': self.directory / 'phone.png'}
        for name, image in (('empty', self.empty), ('phone', self.phone)):
            self.assertTrue(cv2.imwrite(str(legacy_paths[name]), image))
        legacy_bytes = {name: path.read_bytes() for name, path in legacy_paths.items()}
        legacy = ReferenceDetector(self.directory)
        self.assertEqual(legacy.counts, {'empty': 1, 'phone': 1})
        legacy.save('empty', self.other_empty)
        legacy.save('phone', self.other_phone)
        self.assertEqual(legacy.counts, {'empty': 2, 'phone': 2})
        for name, path in legacy_paths.items():
            self.assertEqual(path.read_bytes(), legacy_bytes[name])
            self.assertEqual(len(list((self.directory / name).glob('*.png'))), 1)
        for _ in range(2):
            legacy = ReferenceDetector(self.directory)
            self.assertEqual(legacy.counts, {'empty': 2, 'phone': 2})
            self.assertTrue(legacy.detect(self.phone)[0])
            self.assertTrue(legacy.detect(self.other_phone)[0])

    def test_clear_removes_all_examples_but_preserves_unrelated_files(self):
        self.save_examples()
        self.assertTrue(cv2.imwrite(str(self.directory / 'empty.png'), self.empty))
        self.assertTrue(cv2.imwrite(str(self.directory / 'phone.png'), self.phone))
        unrelated = self.directory / 'notes.txt'
        unrelated.write_text('Keep this local file.', encoding='utf-8')
        unrelated_image = self.directory / 'empty' / 'manual-copy.png'
        self.assertTrue(cv2.imwrite(str(unrelated_image), self.empty))
        unrelated_image_bytes = unrelated_image.read_bytes()
        previous_revision = self.references.revision
        self.references.clear()
        self.assertGreater(self.references.revision, previous_revision)
        self.assertEqual(self.references.counts, {'empty': 0, 'phone': 0})
        self.assertFalse(self.references.ready)
        self.assertFalse(list(self.directory.glob('*.png')))
        self.assertEqual(list((self.directory / 'empty').glob('*.png')), [unrelated_image])
        self.assertFalse(list((self.directory / 'phone').glob('*.png')))
        self.assertEqual(unrelated.read_text(encoding='utf-8'), 'Keep this local file.')
        self.assertEqual(unrelated_image.read_bytes(), unrelated_image_bytes)
        self.assertFalse(ReferenceDetector(self.directory).ready)

    def test_overlapping_labels_are_unknown_without_poisoning_other_examples(self):
        self.save_examples()
        self.references.save('empty', self.other_phone)
        self.assertIsNone(self.references.detect(self.other_phone)[0])
        self.assertTrue(self.references.detect(self.phone)[0])
        self.assertFalse(self.references.detect(self.empty)[0])

    def test_compatible_sizes_work_even_when_latest_examples_use_another_size(self):
        self.references.save('empty', self.empty)
        self.references.save('phone', self.phone)
        self.references.save('empty', cv2.resize(self.empty, (160, 120)))
        self.references.save('phone', cv2.resize(self.other_phone, (160, 120)))
        self.assertTrue(self.references.detect(self.phone)[0])
        self.assertFalse(self.references.detect(self.empty)[0])
        self.assertTrue(self.references.detect(cv2.resize(self.other_phone, (160, 120)))[0])
        self.assertIsNone(self.references.detect(cv2.resize(self.phone, (100, 100)))[0])

    def test_each_label_needs_a_compatible_example_for_the_current_size(self):
        self.references.save('empty', self.empty)
        self.references.save('phone', cv2.resize(self.phone, (160, 120)))
        self.assertTrue(self.references.ready)
        self.assertIsNone(self.references.detect(self.empty)[0])
        self.assertIsNone(self.references.detect(cv2.resize(self.phone, (160, 120)))[0])

    def test_moving_reference_uses_every_saved_pose_and_background(self):
        self.save_examples()
        for image in (self.phone, self.other_phone):
            shifted = cv2.warpAffine(image, np.float32([[1, 0, 4], [0, 1, 3]]),
                                     (320, 240), borderValue=(185, 185, 185))
            self.assertTrue(moving_reference(self.references, shifted)[0])
        self.assertFalse(moving_reference(self.references, self.empty)[0])
        self.assertFalse(moving_reference(self.references, self.other_empty)[0])

    def test_smart_empty_veto_includes_older_empty_examples(self):
        class FalsePositive:
            def detect(self, image):
                return [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])], 'false phone'

        self.save_examples()
        detector = SmartDetector(self.references, FalsePositive)
        for empty in (self.empty, self.other_empty):
            state, _, boxes = detector.detect(empty)
            self.assertFalse(state)
            self.assertEqual(boxes, [])

    def test_smart_conflicting_labels_override_model_and_feature_positives(self):
        from unittest.mock import patch

        class FalsePositive:
            def detect(self, image):
                return [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])], 'phone'

        self.save_examples()
        self.references.save('empty', self.other_phone)
        for factory in (FalsePositive, MissModel):
            detector = SmartDetector(self.references, factory)
            with patch.object(detector, '_match', return_value=True):
                state, label, boxes = detector.detect(self.other_phone)
            self.assertIsNone(state)
            self.assertIn('CONFLICT', label)
            self.assertEqual(boxes, [])

    def test_smart_features_retain_old_pose_and_learn_appended_pose(self):
        textured = []
        for seed, left in ((7, 30), (19, 175)):
            image = self.empty.copy()
            cv2.rectangle(image, (left, 20), (left + 105, 220), (20, 20, 20), -1)
            rng = np.random.default_rng(seed)
            for _ in range(130):
                x, y = map(int, rng.integers([left + 8, 30], [left + 95, 210]))
                color = int(rng.integers(50, 210))
                cv2.circle(image, (x, y), int(rng.integers(2, 5)), (color, color, color), -1)
            textured.append(image)
        self.references.save('empty', self.empty)
        self.references.save('phone', textured[0])
        detector = SmartDetector(self.references, MissModel)
        self.assertTrue(detector._match(textured[0]))
        self.references.save('phone', textured[1])
        for image in textured:
            matrix = cv2.getRotationMatrix2D((160, 120), 3, 1.01)
            matrix[:, 2] += [3, 2]
            shifted = cv2.warpAffine(image, matrix, (320, 240), borderValue=(185, 185, 185))
            state, label, _ = detector.detect(shifted)
            self.assertTrue(state)
            self.assertIn('moving phone features', label)


if __name__ == '__main__':
    unittest.main()
