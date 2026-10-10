"""Conservative appearance fallback for fingers without a visible whole hand.

The returned regions have skin-like colour and differ from all compatible
phone-present (F) references. They are possible contact/occlusion evidence,
not a semantic hand detection: new skin-coloured screen content or objects
can also match, while gloves and poorly lit fingers can be missed.
"""

import cv2
import numpy as np


def partial_hand_regions(image, phone_references) -> list[np.ndarray]:
    """Return candidate contours in the supplied parking image's coordinates.

    ``image`` and the F references must be uint8 BGR images of the same size.
    Uncalibrated or incompatible images yield no fallback candidates. Every
    compatible reference participates: a pixel already explained by any F
    sample is excluded, including static warm backgrounds or screen content.
    Modest uniform exposure changes are compensated before comparing pixels.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 8
        or phone_references is None
    ):
        return []
    references = [
        reference
        for reference in phone_references
        if isinstance(reference, np.ndarray)
        and reference.dtype == np.uint8
        and reference.shape == image.shape
    ]
    if not references:
        return []

    # Requiring both colour spaces excludes gray/white glare and many saturated
    # reds. These broad skin-colour bounds are still only appearance evidence.
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_ycrcb = cv2.inRange(ycrcb, (40, 133, 75), (255, 180, 135))
    skin_hsv = cv2.inRange(hsv, (0, 25, 40), (25, 175, 255))
    skin = cv2.bitwise_and(skin_ycrcb, skin_hsv)
    if not np.any(skin):
        return []

    current = cv2.GaussianBlur(image, (5, 5), 0).astype(np.float32)
    residual = np.full(image.shape[:2], np.inf, dtype=np.float32)
    for reference in references:
        baseline = cv2.GaussianBlur(reference, (5, 5), 0).astype(np.float32)
        # The median stays robust to a finger covering a minority of the ROI.
        # Use a single channel-independent offset to retain colour differences.
        exposure = float(np.clip(np.median(current - baseline), -40, 40))
        aligned = np.clip(baseline + exposure, 0, 255)
        difference = np.mean(np.abs(current - aligned), axis=2)
        np.minimum(residual, difference, out=residual)

    novel = np.where((skin != 0) & (residual >= 28), 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    novel = cv2.morphologyEx(novel, cv2.MORPH_OPEN, kernel)
    novel = cv2.morphologyEx(novel, cv2.MORPH_CLOSE, kernel)

    roi_area = image.shape[0] * image.shape[1]
    # A nearly full ROI more likely means camera/screen/lighting changes than
    # usable finger evidence, so refrain from producing fallback candidates.
    if np.count_nonzero(novel) > roi_area * 0.65:
        return []
    # A generously selected parking ROI must not dilute a visible fingertip
    # below the percentage threshold. Keep the proportional threshold for
    # small regions but cap it for larger selections; isolated specks still
    # cannot pass the absolute area requirement.
    minimum_area = max(64, min(roi_area * 0.015, 512))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(novel, 8)
    regions = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < minimum_area or area > roi_area * 0.65:
            continue
        component = np.where(labels == index, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            regions.append(max(contours, key=cv2.contourArea))
    return regions
