"""Enhanced detection keeps original calibration pixels available and on disk."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from phonewatch.reference import ReferenceDetector, feature


class InPlaceBrightener:
    def __init__(self):
        self.calls = 0

    def process(self, image):
        self.calls += 1
        image[:] = cv2.convertScaleAbs(image, alpha=2, beta=30)
        return image


class EnhancedReferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.empty = np.full((160, 120, 3), 80, np.uint8)
        self.phone = self.empty.copy()
        cv2.rectangle(self.phone, (25, 15), (95, 145), (15, 15, 15), -1)
        self.transform = InPlaceBrightener()
        self.references = ReferenceDetector(self.directory, image_transform=self.transform.process)

    def calibrate(self):
        self.references.save('empty', self.empty)
        self.references.save('phone', self.phone)

    def test_current_frame_and_reference_share_the_transform_exactly_once(self):
        self.calibrate()
        for name, raw, expected_state in (
            ('empty', self.empty, False), ('phone', self.phone, True),
        ):
            enhanced = cv2.convertScaleAbs(raw, alpha=2, beta=30)
            state, _, scores = self.references.evidence(enhanced)
            self.assertEqual(state, expected_state)
            self.assertAlmostEqual(scores[name], 0., places=7)
            np.testing.assert_array_equal(self.references.compatible(name, raw.shape)[0], enhanced)
        self.assertEqual(self.transform.calls, 2)
        for raw in (self.empty, self.phone):
            self.references.detect(cv2.convertScaleAbs(raw, alpha=2, beta=30))
        self.assertEqual(self.transform.calls, 2)

    def test_save_and_in_place_transform_never_modify_raw_calibration(self):
        self.calibrate()
        self.assertEqual(self.transform.calls, 0)
        for name, raw in (('empty', self.empty), ('phone', self.phone)):
            self.references.compatible(name, raw.shape)
            np.testing.assert_array_equal(self.references.images[name], raw)
            np.testing.assert_array_equal(self.references.samples[name][0], raw)
            np.testing.assert_array_equal(self.references.raw_compatible(name, raw.shape)[0], raw)
            path, = (self.directory / name).glob('*.png')
            np.testing.assert_array_equal(cv2.imread(str(path)), raw)

    def test_derived_images_are_reused_across_feature_and_motion_requests(self):
        self.calibrate()
        first = self.references.compatible('phone', self.phone.shape)
        self.assertIs(self.references.compatible('phone', self.phone.shape), first)
        self.references.features('phone', self.phone.shape)
        self.references.moving_features('phone', self.phone.shape)
        self.references.features('phone', self.phone.shape)
        self.assertEqual(self.transform.calls, 1)
        self.assertIs(self.references.compatible('phone', self.phone.shape)[0], first[0])

    def test_append_invalidates_transformed_features_and_preserves_old_samples(self):
        self.calibrate()
        old_images = self.references.compatible('phone', self.phone.shape)
        old_features = self.references.features('phone', self.phone.shape)
        old_motion = self.references.moving_features('phone', self.phone.shape)
        shifted = self.empty.copy()
        cv2.rectangle(shifted, (10, 15), (80, 145), (15, 15, 15), -1)
        self.references.save('phone', shifted)
        current = self.references.compatible('phone', self.phone.shape)
        self.assertEqual(len(current), 2)
        self.assertIsNot(current, old_images)
        self.assertIsNot(self.references.features('phone', self.phone.shape), old_features)
        self.assertIsNot(self.references.moving_features('phone', self.phone.shape), old_motion)
        self.assertEqual(self.transform.calls, 3)
        np.testing.assert_array_equal(current[0], cv2.convertScaleAbs(self.phone, alpha=2, beta=30))
        np.testing.assert_array_equal(current[1], cv2.convertScaleAbs(shifted, alpha=2, beta=30))
        for raw in (self.phone, shifted):
            self.assertTrue(self.references.detect(cv2.convertScaleAbs(raw, alpha=2, beta=30))[0])

    def test_legacy_reload_and_restart_mode_toggle_preserve_every_png(self):
        for name, image in (('empty', self.empty), ('phone', self.phone)):
            self.assertTrue(cv2.imwrite(str(self.directory / f'{name}.png'), image))
        enhanced = ReferenceDetector(self.directory, image_transform=self.transform.process)
        enhanced.save('empty', self.empty)
        enhanced.save('phone', self.phone)
        original_files = {path: path.read_bytes() for path in self.directory.rglob('*.png')}
        self.assertEqual(enhanced.counts, {'empty': 2, 'phone': 2})
        self.assertTrue(enhanced.detect(cv2.convertScaleAbs(self.phone, alpha=2, beta=30))[0])
        raw_mode = ReferenceDetector(self.directory)
        self.assertTrue(raw_mode.detect(self.phone)[0])
        np.testing.assert_array_equal(raw_mode.compatible('phone', self.phone.shape)[0], self.phone)
        reenabled = ReferenceDetector(self.directory, image_transform=self.transform.process)
        self.assertFalse(reenabled.detect(cv2.convertScaleAbs(self.empty, alpha=2, beta=30))[0])
        self.assertEqual(reenabled.counts, {'empty': 2, 'phone': 2})
        self.assertEqual({path: path.read_bytes() for path in self.directory.rglob('*.png')}, original_files)

    def test_raw_view_sees_live_append_and_clear_without_enhancement(self):
        view = self.references.raw_view()
        self.assertIs(self.references.raw_view(), view)
        original_revision = view.revision
        self.assertFalse(view.ready)
        self.calibrate()
        self.assertGreater(view.revision, original_revision)
        self.assertTrue(view.ready)
        self.assertEqual(view.counts, {'empty': 1, 'phone': 1})
        for name, raw in (('empty', self.empty), ('phone', self.phone)):
            np.testing.assert_array_equal(view.compatible(name, raw.shape)[0], raw)
        self.assertEqual(self.transform.calls, 0)
        self.references.compatible('phone', self.phone.shape)
        revision = view.revision
        self.references.clear()
        self.assertGreater(view.revision, revision)
        self.assertFalse(view.ready)
        self.assertEqual(view.counts, {'empty': 0, 'phone': 0})
        self.assertEqual(view.compatible('phone', self.phone.shape), [])
        self.assertEqual(self.references.compatible('phone', self.phone.shape), [])
        self.references.save('phone', self.empty)
        np.testing.assert_array_equal(view.compatible('phone', self.phone.shape)[0], self.empty)
        np.testing.assert_array_equal(self.references.compatible('phone', self.phone.shape)[0],
                                      cv2.convertScaleAbs(self.empty, alpha=2, beta=30))

    def test_mixed_sizes_have_separate_derived_cache_entries(self):
        self.calibrate()
        smaller = cv2.resize(self.phone, (60, 80))
        self.references.save('phone', smaller)
        self.references.compatible('phone', self.phone.shape)
        self.references.features('phone', self.phone.shape)
        self.references.compatible('phone', smaller.shape)
        self.assertEqual(self.transform.calls, 2)
        self.assertEqual(self.references.compatible('phone', (20, 30, 3)), [])
        self.assertEqual(self.transform.calls, 2)
        np.testing.assert_array_equal(self.references.raw_compatible('phone', smaller.shape)[0], smaller)

    def test_transform_result_cannot_alias_a_reused_output_buffer(self):
        buffer = np.empty_like(self.empty)

        def shared_buffer(image):
            buffer[:] = cv2.convertScaleAbs(image, alpha=2, beta=30)
            return buffer

        references = ReferenceDetector(self.directory, image_transform=shared_buffer)
        references.save('phone', self.phone)
        references.save('phone', self.empty)
        images = references.compatible('phone', self.phone.shape)
        np.testing.assert_array_equal(images[0], cv2.convertScaleAbs(self.phone, alpha=2, beta=30))
        np.testing.assert_array_equal(images[1], cv2.convertScaleAbs(self.empty, alpha=2, beta=30))

    def test_invalid_shape_transform_fails_without_altering_saved_raw_image(self):
        references = ReferenceDetector(self.directory, image_transform=lambda image: image[:20, :20])
        references.save('phone', self.phone)
        with self.assertRaisesRegex(ValueError, 'shape and dtype'):
            references.compatible('phone', self.phone.shape)
        np.testing.assert_array_equal(references.raw_compatible('phone', self.phone.shape)[0], self.phone)

    def test_no_transform_preserves_existing_raw_reference_behavior(self):
        references = ReferenceDetector(self.directory)
        references.save('empty', self.empty)
        references.save('phone', self.phone)
        raw = references.raw_compatible('phone', self.phone.shape)[0]
        self.assertIs(references.compatible('phone', self.phone.shape)[0], raw)
        self.assertTrue(references.detect(self.phone)[0])
        self.assertFalse(references.detect(self.empty)[0])
        expected_features = feature(self.phone)
        for expected, actual in zip(expected_features, references.features('phone', self.phone.shape)[0]):
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
