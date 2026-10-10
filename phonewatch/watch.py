"""Optional monitoring of sustained hand activity near a parked phone.

This is evidence of possible interaction, not proof that the phone is in use.
The caller supplies confirmed phone presence and normalized hand landmarks.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class WatchTimer:
    """Count confirmed hand-near intervals after a continuous activation delay.

    Activation is not charged retroactively: the first counted interval starts
    when ``active`` becomes true. Losing the camera/hand observation or phone
    presence discards the last interval and immediately clears activation.
    An observed hand moving away ends activation but charges its prior interval.
    """

    delay: float = 3.0
    active: bool = False
    previous: float | None = None
    candidate_since: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.delay) or self.delay < 0:
            raise ValueError("watch delay must be finite and nonnegative")

    def _validate_time(self, now: float):
        if not math.isfinite(now):
            raise ValueError("timestamps must be finite")
        if self.previous is not None and now < self.previous:
            raise ValueError("timestamps must be monotonic")

    def update(
        self, near: bool | None, phone_present: bool, now: float
    ) -> tuple[float, bool, bool]:
        """Return elapsed seconds, activation edge, and whether active changed."""
        self._validate_time(now)
        before = self.active
        elapsed = 0.0
        if self.previous is not None and before and phone_present and near is not None:
            elapsed = now - self.previous
        self.previous = now

        if not phone_present or near is not True:
            self.active = False
            self.candidate_since = None
        elif not self.active:
            if self.candidate_since is None:
                self.candidate_since = now
            if now - self.candidate_since >= self.delay:
                self.active = True
                self.candidate_since = None

        return elapsed, self.active and not before, self.active != before

    def reset(self, now: float) -> None:
        """Discard any running interval, for example while changing calibration."""
        self._validate_time(now)
        self.active = False
        self.candidate_since = None
        self.previous = now


def hand_near_polygon(hands, polygon, frame_shape, margin_fraction: float = 0.08) -> bool:
    """Test fingertip proximity to a parking polygon in full-frame pixels.

    ``hands`` contains lists of normalized ``(x, y)`` landmarks. The proximity
    margin is a fraction of the polygon's largest bounding-box dimension, so
    it scales with the selected phone area rather than the entire camera view.
    Wrists and palms alone cannot trigger possible phone interaction.
    """
    if not math.isfinite(margin_fraction) or margin_fraction < 0:
        raise ValueError("hand margin must be finite and nonnegative")
    try:
        height, width = frame_shape[:2]
    except (TypeError, ValueError):
        raise ValueError("frame shape must include a positive height and width") from None
    if not all(math.isfinite(value) and value > 0 for value in (height, width)):
        raise ValueError("frame shape must include a positive height and width")

    try:
        points = np.asarray(polygon, dtype=np.float32)
    except (TypeError, ValueError):
        raise ValueError("parking polygon must contain finite pixel coordinates") from None
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("parking polygon must contain at least three finite pixel coordinates")
    contour = points.reshape((-1, 1, 2))
    if abs(cv2.contourArea(contour)) <= 0:
        raise ValueError("parking polygon must have a positive area")
    margin = float(np.ptp(points, axis=0).max()) * margin_fraction

    for hand in hands or []:
        if hand is None:
            continue
        for index in (4, 8, 12, 16, 20):
            try:
                x, y = hand[index][:2]
                x, y = float(x), float(y)
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
                continue
            distance = cv2.pointPolygonTest(contour, (x * width, y * height), True)
            if distance >= -margin:
                return True
    return False
