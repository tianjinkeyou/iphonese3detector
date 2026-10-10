"""Optional visual activity on a calibrated phone screen.

Screen changes are evidence of possible use, not proof of interaction. A video,
notification, reflection or automatic animation can produce the same evidence.
Only the selected screen is compared; neither portrait proportions nor a phone
model are assumed. Camera motion is estimated from the surrounding scene, so
scrolling screen content is not registered away as if it were camera shake.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


class ScreenActivityDetector:
    """Compare successive, slightly smoothed screen crops.

    ``detect`` returns ``(moving, score, label)``. ``moving`` is ``None`` when
    calibration/history or image visibility prevents an observation. ``score``
    is the fraction of observable interior pixels with meaningful changes.
    Pixel coordinates must be four corners in TL/TR/BR/BL perimeter order.
    """

    def __init__(self, threshold: float = 0.025, intensity_threshold: float = 12 / 255):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("screen change threshold must be in (0, 1]")
        if not math.isfinite(intensity_threshold) or not 0 < intensity_threshold <= 1:
            raise ValueError("screen intensity threshold must be in (0, 1]")
        self.threshold = threshold
        self.intensity_threshold = intensity_threshold
        self.reset()

    def reset(self) -> None:
        """Discard temporal history after calibration/camera interruptions."""
        self._previous_crop = None
        self._previous_scene = None
        self._geometry = None

    @staticmethod
    def _calibration(frame, polygon):
        if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.dtype != np.uint8:
            raise ValueError("screen frame must be a nonempty uint8 image")
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] in (3, 4):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if frame.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("screen frame must be grayscale, BGR or BGRA")
        points = np.asarray(polygon, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("select four finite screen corners")
        height, width = gray.shape
        if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
            raise ValueError("screen corners must be inside the image")
        contour = points.reshape((-1, 1, 2))
        if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 400:
            raise ValueError("screen corners must form a sufficiently large convex area")
        edges = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        crop_width = float((edges[0] + edges[2]) / 2)
        crop_height = float((edges[1] + edges[3]) / 2)
        if min(edges) < 20:
            raise ValueError("screen must be at least 20 pixels on each edge")
        scale = min(1.0, 320 / max(crop_width, crop_height))
        size = (round(crop_width * scale), round(crop_height * scale))
        if min(size) < 20:
            raise ValueError("screen is too narrow for reliable changes")
        target = np.float32([[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]])
        return gray, points, size, target

    @staticmethod
    def _crop(gray, points, size, target):
        matrix = cv2.getPerspectiveTransform(np.float32(points), target)
        crop = cv2.warpPerspective(gray, matrix, size, flags=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(crop, (3, 3), 0).astype(np.float32) / 255

    @staticmethod
    def _scene(gray):
        scale = min(1.0, 480 / max(gray.shape))
        scene = cv2.resize(gray, (round(gray.shape[1] * scale), round(gray.shape[0] * scale)))
        return scene, scale

    @staticmethod
    def _camera_transform(previous, current, points, scale):
        """Estimate only small camera motion from features outside the screen.

        Missing or inconsistent surrounding features simply disable alignment.
        This intentionally avoids fitting a transform to a scrolling video or
        game. Bounds reject motion over 5% of the screen dimensions, rotation
        over 5 degrees, and changes in scale over 1%.
        """
        mask = np.full(previous.shape, 255, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(points * scale).astype(np.int32), 0)
        margin = max(3, round(max(np.ptp(points, axis=0)) * scale * 0.03))
        mask = cv2.erode(mask, np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8))
        features = cv2.goodFeaturesToTrack(previous, 100, 0.01, 8, mask=mask)
        if features is None or len(features) < 8:
            return None
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, features, None)
        if tracked is None or status is None:
            return None
        restored, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, tracked, None)
        if restored is None or back_status is None:
            return None
        valid = (status.ravel() == 1) & (back_status.ravel() == 1)
        valid &= np.linalg.norm(restored - features, axis=2).ravel() < 1.0
        source = features[valid].reshape(-1, 2)
        destination = tracked[valid].reshape(-1, 2)
        if len(source) < 8 or not np.isfinite(destination).all():
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(source, destination, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        if matrix is None or inliers is None or int(inliers.sum()) < 8 or inliers.mean() < 0.6:
            return None
        factor = math.hypot(matrix[0, 0], matrix[1, 0])
        angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
        if abs(factor - 1) > 0.01 or abs(angle) > 5:
            return None
        scaled_points = points * scale
        moved = cv2.transform(scaled_points[None], matrix)[0]
        displacement = np.max(np.abs(moved - scaled_points), axis=0)
        limits = np.ptp(scaled_points, axis=0) * 0.05
        if np.any(displacement > limits):
            return None
        full_matrix = matrix.copy()
        full_matrix[:, 2] /= scale
        return full_matrix

    @staticmethod
    def _dark(crop):
        return float(np.percentile(crop, 95)) < 8 / 255

    @staticmethod
    def _saturated(crop):
        return float(np.mean(crop >= 250 / 255)) > 0.98 and float(np.std(crop)) < 3 / 255

    def _exposure_offset(self, previous, current, points, scale, transform, screen_median):
        """Estimate common exposure from the surrounding scene, not the screen.

        A phone can legitimately switch between two flat-colored screens. An
        offset estimated from its own pixels would erase that activity. Camera
        exposure usually also changes the background. Reject an outside offset
        that disagrees with the screen's common shift, as it could be a moving
        hand/background rather than a camera exposure adjustment.
        """
        aligned = current
        valid = np.full(previous.shape, 255, dtype=np.uint8)
        if transform is not None:
            scene_transform = transform.copy()
            scene_transform[:, 2] *= scale
            size = (previous.shape[1], previous.shape[0])
            aligned = cv2.warpAffine(current, scene_transform, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            valid = cv2.warpAffine(valid, scene_transform, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        cv2.fillConvexPoly(valid, np.rint(points * scale).astype(np.int32), 0)
        margin = max(3, round(max(np.ptp(points, axis=0)) * scale * 0.08))
        valid = cv2.erode(valid, np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8))
        observable = valid > 0
        if np.count_nonzero(observable) < max(64, previous.size * 0.05):
            return None
        changes = (aligned.astype(np.float32) - previous.astype(np.float32)) / 255
        offset = float(np.median(changes[observable]))
        # Changes confined to the surrounding scene cannot themselves count as
        # screen activity. Correct only a shift supported by both regions.
        if abs(offset - screen_median) > self.intensity_threshold:
            return 0.0
        return offset

    def detect(self, frame, polygon) -> tuple[bool | None, float, str]:
        """Observe screen changes, compensating exposure and bounded camera shake."""
        try:
            gray, points, size, target = self._calibration(frame, polygon)
        except (ValueError, TypeError, OverflowError, cv2.error):
            self.reset()
            return None, 0.0, "SCREEN: invalid calibration"

        geometry = (gray.shape, tuple(points.ravel()), size)
        crop = self._crop(gray, points, size, target)
        scene, scale = self._scene(gray)
        previous_crop = self._previous_crop
        previous_scene = self._previous_scene
        changed_geometry = geometry != self._geometry
        self._geometry = geometry
        self._previous_crop = crop
        self._previous_scene = scene
        if self._dark(crop):
            # Restart history when visibility returns instead of treating a
            # camera blackout/recovery as phone-screen activity.
            self._previous_crop = None
            self._previous_scene = None
            return None, 0.0, "SCREEN: too dark to observe"
        if changed_geometry or previous_crop is None or previous_scene is None:
            return None, 0.0, "SCREEN: waiting for next frame"
        if self._saturated(crop) and self._saturated(previous_crop):
            return None, 0.0, "SCREEN: overexposed, no detail"

        transform = self._camera_transform(previous_scene, scene, points, scale)
        aligned = crop
        if transform is not None:
            moved_points = cv2.transform(points[None], transform)[0]
            if (np.all(moved_points >= 0) and np.all(moved_points[:, 0] < gray.shape[1])
                    and np.all(moved_points[:, 1] < gray.shape[0])):
                aligned = self._crop(gray, moved_points, size, target)

        border = max(2, round(min(size) * 0.04))
        difference = (aligned - previous_crop)[border:-border, border:-border]
        screen_median = float(np.median(difference))
        offset = self._exposure_offset(previous_scene, scene, points, scale, transform, screen_median)
        if offset is None:
            # Without surrounding pixels, a flat screen transition and camera
            # exposure are visually indistinguishable. Do not call it stable.
            residual = difference - screen_median
            residual_score = float(np.mean(np.abs(residual) > self.intensity_threshold))
            if abs(screen_median) > self.intensity_threshold and residual_score < self.threshold:
                return None, residual_score, "SCREEN: exposure/change unclear; include background"
            offset = screen_median
        difference -= offset
        score = float(np.mean(np.abs(difference) > self.intensity_threshold))
        moving = score >= self.threshold
        label = "SCREEN: changing (possible use)" if moving else "SCREEN: stable"
        return moving, score, label
