"""Bounded, stateless webcam enhancement for visibility and detection inputs.

This improves details that the camera captured. It does not reconstruct clipped
highlights or invent details in a completely black image. The fixed tone curve
and modest local correction make the same stored reference and live image take
exactly the same processing path, without a frame-to-frame exposure controller.
"""

import math

import cv2
import numpy as np


class ImageEnhancer:
    """Optional luminance enhancement with no history or input mutation.

    ``strength`` is in [0, 1]; 0 is an exact pass-through. ``off`` always returns
    an independent, identical copy. Color channels receive a common gain, capped
    before any channel clips, which keeps hue and saturation close to the input.
    """

    def __init__(self, mode="off", strength=0.5):
        if mode not in ("off", "adaptive"):
            raise ValueError("image mode must be off or adaptive")
        if isinstance(strength, (bool, str, bytes)):
            raise ValueError("image strength must be a finite number from 0 to 1")
        try:
            strength = float(strength)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("image strength must be a finite number from 0 to 1") from exc
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("image strength must be a finite number from 0 to 1")
        self.mode = mode
        self.strength = strength
        values = np.arange(256, dtype=np.float32)
        # Fixed, monotonic shadow lift. Exact endpoints retain black and white;
        # the boost smoothly vanishes in bright areas, so highlights are bounded.
        lift = 70 * strength * (1 - values / 255) ** 2 * (1 - np.exp(-values / 16))
        self._tone = values + lift
        self._clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    @property
    def config(self):
        """Small JSON-safe configuration for event records and diagnostics."""
        return {
            "mode": self.mode,
            "strength": self.strength,
            "algorithm": "bounded-luminance-v1",
        }

    @staticmethod
    def _validate(image):
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a nonempty BGR uint8 array")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a nonempty BGR uint8 array")
        if not image.shape[0] or not image.shape[1]:
            raise ValueError("image must be a nonempty BGR uint8 array")

    def process(self, image):
        """Return an equally sized BGR uint8 image; do not alter ``image``."""
        self._validate(image)
        if self.mode == "off" or self.strength == 0:
            return image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        luminance = gray.astype(np.float32)
        # Small bilateral neighborhoods remove dark sensor grain while preserving
        # high-contrast edges. The blend is zero for brighter portions of a frame.
        filtered = cv2.bilateralFilter(gray, 5, 5.0, 1.1).astype(np.float32)
        dark_weight = np.clip((128 - luminance) / 96, 0, 1)
        denoise_weight = min(0.85, self.strength * 1.7) * dark_weight
        clean = luminance + (filtered - luminance) * denoise_weight
        clean_u8 = np.clip(np.rint(clean), 0, 255).astype(np.uint8)
        target = self._tone[clean_u8]

        # CLAHE's correction is bounded in luminance units, limiting both noise
        # amplification and sensitivity to the distribution of a scene's colors.
        local = self._clahe.apply(clean_u8).astype(np.float32)
        correction = np.clip(local - clean, -14, 14)
        local_weight = self.strength * (0.08 + 0.30 * dark_weight)
        target += correction * local_weight
        target = np.clip(target, 0, 255)

        # Use one gain for all three channels instead of independent adjustments.
        # Completely black pixels stay black; already saturated channels restrict
        # the available gain rather than being clipped and changing hue.
        gain = np.divide(target, luminance, out=np.ones_like(target), where=luminance > 0)
        maximum = image.max(axis=2).astype(np.float32)
        available = np.divide(255.0, maximum, out=np.ones_like(maximum), where=maximum > 0)
        gain = np.minimum(gain, available)
        result = np.rint(image.astype(np.float32) * gain[:, :, None])
        return np.clip(result, 0, 255).astype(np.uint8)

    def describe(self, image):
        """Report input exposure statistics without an adaptive exposure loop."""
        self._validate(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p05, p95 = np.percentile(gray, (5, 95))
        return {
            "mean_luminance": round(float(gray.mean()), 2),
            "p05": float(p05),
            "p95": float(p95),
            "shadow_fraction": round(float(np.mean(gray < 48)), 4),
            "clipped_dark_fraction": round(float(np.mean(gray <= 2)), 4),
            "clipped_bright_fraction": round(float(np.mean(gray >= 253)), 4),
        }
