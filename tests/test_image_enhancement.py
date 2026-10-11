import json
import unittest

import cv2
import numpy as np

from phonewatch.image_enhancement import ImageEnhancer


class ImageEnhancementTests(unittest.TestCase):
    def test_off_and_zero_strength_are_exact_independent_copies(self):
        image = np.random.default_rng(4).integers(0, 256, (53, 79, 3), dtype=np.uint8)
        for enhancer in (ImageEnhancer(), ImageEnhancer("adaptive", 0)):
            result = enhancer.process(image)
            np.testing.assert_array_equal(result, image)
            self.assertFalse(np.shares_memory(result, image))

    def test_adaptive_preserves_shape_dtype_input_and_determinism(self):
        image = np.random.default_rng(7).integers(0, 256, (61, 87, 3), dtype=np.uint8)
        original = image.copy()
        image.flags.writeable = False
        enhancer = ImageEnhancer("adaptive")
        first = enhancer.process(image)
        enhancer.process(np.full_like(image, 230))
        second = enhancer.process(image)
        self.assertEqual(first.shape, image.shape)
        self.assertEqual(first.dtype, np.uint8)
        self.assertFalse(np.shares_memory(first, image))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(image, original)

    def test_dark_captured_detail_is_brighter_and_more_separated(self):
        image = np.empty((128, 160, 3), np.uint8)
        image[:, :80] = 8
        image[:, 80:] = 20
        result = ImageEnhancer("adaptive").process(image)
        left = float(result[16:112, 16:64].mean())
        right = float(result[16:112, 96:144].mean())
        self.assertGreater(left, 8)
        self.assertGreater(right, 20)
        self.assertGreater(right - left, 12)

    def test_clipped_black_and_white_do_not_acquire_invented_details(self):
        enhancer = ImageEnhancer("adaptive", 1)
        for value in (0, 255):
            image = np.full((77, 113, 3), value, np.uint8)
            np.testing.assert_array_equal(enhancer.process(image), image)
        image = np.zeros((128, 160, 3), np.uint8)
        image[:, 80:] = 255
        np.testing.assert_array_equal(enhancer.process(image), image)

    def test_dark_sensor_grain_reduces_without_blurring_a_strong_edge(self):
        rng = np.random.default_rng(7)
        gray = np.clip(30 + rng.normal(0, 3, (128, 160)), 0, 255).astype(np.uint8)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        result = ImageEnhancer("adaptive").process(image)
        self.assertLess(result[8:-8, 8:-8, 0].std(), gray[8:-8, 8:-8].std())

        edge = np.empty_like(image)
        edge[:, :80] = 24
        edge[:, 80:] = 180
        processed = ImageEnhancer("adaptive").process(edge)
        profile = processed[16:112, :, 0].mean(axis=0)
        # The original edge location remains sharp, without mixing bright pixels
        # into the immediately adjacent dark phone outline.
        self.assertGreater(profile[80] - profile[79], 120)
        self.assertLess(abs(profile[79] - profile[75]), 2)
        self.assertLess(abs(profile[80] - profile[84]), 2)

    def test_common_channel_gain_retains_red_and_skin_hues(self):
        colors = np.uint8([[[15, 25, 50], [40, 70, 120], [0, 0, 90],
                            [90, 0, 0], [0, 100, 0], [180, 200, 220]]])
        image = np.repeat(np.repeat(colors, 32, axis=0), 32, axis=1)
        result = ImageEnhancer("adaptive").process(image)
        before = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        after = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.int16)
        self.assertLessEqual(int(np.max(np.abs(before[:, :, 0] - after[:, :, 0]))), 1)
        self.assertLessEqual(int(np.max(np.abs(before[:, :, 1] - after[:, :, 1]))), 3)
        self.assertGreater(result[:, :32].mean(), image[:, :32].mean())

    def test_normal_and_bright_scene_changes_are_modest(self):
        row = np.linspace(128, 240, 160).astype(np.uint8)
        image = np.repeat(np.repeat(row[None, :, None], 128, axis=0), 3, axis=2)
        result = ImageEnhancer("adaptive").process(image)
        delta = result.astype(np.int16) - image.astype(np.int16)
        self.assertLess(float(np.abs(delta).mean()), 6)
        self.assertLessEqual(int(delta.max()), 12)
        self.assertLessEqual(int(np.abs(delta[:, 120:]).max()), 3)

    def test_strength_controls_shadow_lift(self):
        image = np.full((96, 128, 3), 24, np.uint8)
        low = ImageEnhancer("adaptive", 0.25).process(image)
        high = ImageEnhancer("adaptive", 0.75).process(image)
        self.assertGreater(high.mean(), low.mean())
        self.assertGreater(low.mean(), image.mean())

    def test_small_and_noncontiguous_images_work(self):
        enhancer = ImageEnhancer("adaptive")
        single = np.full((1, 1, 3), 24, np.uint8)
        self.assertEqual(enhancer.process(single).shape, single.shape)
        image = np.full((64, 80, 3), 24, np.uint8)[:, ::2]
        self.assertFalse(image.flags.c_contiguous)
        self.assertEqual(enhancer.process(image).shape, image.shape)

    def test_config_and_exposure_stats_are_json_safe(self):
        enhancer = ImageEnhancer("adaptive", 0.6)
        image = np.zeros((40, 80, 3), np.uint8)
        image[:, 40:] = 255
        description = enhancer.describe(image)
        self.assertEqual(description["clipped_dark_fraction"], 0.5)
        self.assertEqual(description["clipped_bright_fraction"], 0.5)
        self.assertEqual(description["mean_luminance"], 127.5)
        self.assertEqual(description["p05"], 0.0)
        self.assertEqual(description["p95"], 255.0)
        self.assertEqual(enhancer.config["mode"], "adaptive")
        json.dumps({"config": enhancer.config, "exposure": description}, allow_nan=False)
        # Callers cannot change the enhancer by mutating a configuration snapshot.
        enhancer.config["mode"] = "off"
        self.assertEqual(enhancer.mode, "adaptive")

    def test_raw_percentiles_identify_lost_detail_without_enhancement(self):
        enhancer = ImageEnhancer("adaptive")
        for value in (0, 2, 255):
            with self.subTest(value=value):
                image = np.full((40, 80, 3), value, np.uint8)
                description = enhancer.describe(image)
                self.assertEqual(description["p05"], float(value))
                self.assertEqual(description["p95"], float(value))
                json.dumps(description, allow_nan=False)
        gray = np.arange(100, dtype=np.uint8).reshape(10, 10)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        description = enhancer.describe(image)
        self.assertAlmostEqual(description["p05"], 4.95)
        self.assertAlmostEqual(description["p95"], 94.05)

    def test_invalid_mode_and_strength_are_rejected(self):
        with self.assertRaises(ValueError):
            ImageEnhancer("hdr")
        for strength in (-0.1, 1.1, float("nan"), float("inf"), None, True, "0.5", object()):
            with self.subTest(strength=strength):
                with self.assertRaises(ValueError):
                    ImageEnhancer("adaptive", strength)

    def test_invalid_images_are_rejected_even_when_off(self):
        images = (None, [], np.zeros((4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8),
                  np.zeros((4, 4, 3), np.float32), np.zeros((0, 4, 3), np.uint8),
                  np.zeros((4, 0, 3), np.uint8))
        for mode in ("off", "adaptive"):
            enhancer = ImageEnhancer(mode)
            for image in images:
                with self.subTest(mode=mode, shape=getattr(image, "shape", None)):
                    with self.assertRaises(ValueError):
                        enhancer.process(image)
                    with self.assertRaises(ValueError):
                        enhancer.describe(image)


if __name__ == "__main__":
    unittest.main()
