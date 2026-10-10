"""Conservative phone-body evidence while the calibrated display changes.

Only E/F differences near the screen, outside its interior, are evidence. A
matching background by itself cannot establish that the phone is present.
"""
import cv2
import numpy as np


def _compatible(references, shape):
    return [ref for ref in references if isinstance(ref, np.ndarray)
            and ref.shape == shape and ref.dtype == np.uint8]



def _detail_evidence(current, ref_empty, ref_phone, body_area):
    """Compare spatial bezel details when a tight crop has no stable background.

    Each body's own median is removed, so a flat dark body against a flat light
    empty sample provides zero evidence. A positive requires visible details
    matching F, including the rest of the observable body's spatial appearance.
    """
    body_count = int(np.count_nonzero(body_area))
    required = max(64, int(np.ceil(body_count * .015)))
    centered = [frame - np.median(frame[body_area], axis=0)
                for frame in (current, ref_empty, ref_phone)]
    target, empty, phone = centered
    difference = np.max(np.abs(phone - empty), axis=2)
    evidence = body_area & (difference >= 24)
    if np.count_nonzero(evidence) < required:
        # When the same observable body detail is labelled E and F, a matching
        # current appearance must veto positive evidence from another pair.
        # Flat bodies have no such detail and cannot supply this ambiguity vote.
        shared_detail = (body_area & (np.max(np.abs(phone), axis=2) >= 24)
                         & (np.max(np.abs(empty), axis=2) >= 24))
        if np.count_nonzero(shared_detail) >= required:
            de = float(np.abs(target - empty)[shared_detail].mean() / 255)
            dp = float(np.abs(target - phone)[shared_detail].mean() / 255)
            if max(de, dp) < .035:
                return None, (de, dp)
        return None
    separation = float(np.abs(phone - empty)[evidence].mean() / 255)
    if separation < .10:
        return None
    errors_empty = np.abs(target - empty)[evidence].mean(axis=1) / 255
    errors_phone = np.abs(target - phone)[evidence].mean(axis=1) / 255
    de, dp = float(errors_empty.mean()), float(errors_phone.mean())
    # Also inspect the whole visible body: new texture outside the selected
    # details must not disappear merely because E/F did not differ there.
    body_empty = np.abs(target - empty)[body_area].mean() / 255
    body_phone = np.abs(target - phone)[body_area].mean() / 255
    if (dp < .045 and body_phone < .025 and de - dp > .07
            and dp < de * .35 and np.mean(errors_phone < .06) >= .85):
        return True, (de, dp)
    if (de < .045 and body_empty < .025 and dp - de > .07
            and de < dp * .35 and np.mean(errors_empty < .06) >= .85):
        return False, (de, dp)
    return None


def screen_body_presence(image, polygon, empty_references, phone_references):
    """Return E/F body evidence without comparing the screen's contents.

    ``polygon`` is four calibrated screen corners in ``image`` coordinates.
    An unrelated appearance, insufficient visible bezel, or conflicting samples
    gives None. This appearance comparison is a fallback, not an object detector.
    """
    unknown = (None, 'screen body: insufficient E/F evidence')
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8):
        return unknown
    height, width = image.shape[:2]
    try:
        points = np.asarray(polygon, dtype=np.float32)
    except (TypeError, ValueError):
        return unknown
    if (points.shape != (4, 2) or not np.isfinite(points).all()
            or np.any(points < 0) or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] >= height)
            or not cv2.isContourConvex(points)
            or cv2.contourArea(points) < 64):
        return unknown
    empty = _compatible(empty_references, image.shape)
    phone = _compatible(phone_references, image.shape)
    if not empty or not phone:
        return None, 'screen body: save compatible E and F samples'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.percentile(gray, 95) < 8:
        return None, 'screen body: too dark'

    interior = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(interior, np.rint(points).astype(np.int32), 1)
    excluded = cv2.dilate(interior, np.ones((7, 7), np.uint8)) > 0
    # Limit evidence to visible bezel/body near the calibrated screen. Changes
    # elsewhere in the parking area cannot dominate a positive decision.
    short_edge = min(np.linalg.norm(points[(i + 1) % 4] - points[i]) for i in range(4))
    radius = max(6, min(48, int(round(short_edge * .30))))
    outer = cv2.dilate(interior, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))) > 0
    body_area = outer & ~excluded
    outside = ~excluded
    required = max(64, int(np.ceil(height * width * .015)))
    if np.count_nonzero(body_area) < required:
        return unknown

    def smooth(frame):
        return cv2.GaussianBlur(frame, (3, 3), 0).astype(np.float32)

    current = smooth(image)
    empties = [smooth(ref) for ref in empty]
    phones = [smooth(ref) for ref in phone]
    decisions = set()
    ambiguous_match = False
    best = None
    for ref_empty in empties:
        for ref_phone in phones:
            # Most exterior pixels should retain the background. Remove only a
            # modest global exposure offset; larger illumination shifts are not
            # sufficiently observable for this conservative fallback.
            offset_phone = np.median((ref_phone - ref_empty)[outside], axis=0)
            if np.max(np.abs(offset_phone)) > 24:
                detail = _detail_evidence(current, ref_empty, ref_phone, body_area)
                if detail is not None:
                    state, scores = detail
                    if state is None:
                        ambiguous_match = True
                    else:
                        decisions.add(state)
                    if best is None or min(scores) < min(best):
                        best = scores
                continue
            adjusted_phone = ref_phone - offset_phone
            difference = np.max(np.abs(adjusted_phone - ref_empty), axis=2)
            common = outside & (difference < 12)
            if np.count_nonzero(common) < required:
                detail = _detail_evidence(current, ref_empty, ref_phone, body_area)
                if detail is not None:
                    state, scores = detail
                    if state is None:
                        ambiguous_match = True
                    else:
                        decisions.add(state)
                    if best is None or min(scores) < min(best):
                        best = scores
                continue
            offset_current = np.median((current - ref_empty)[common], axis=0)
            if np.max(np.abs(offset_current)) > 24:
                continue
            adjusted_current = current - offset_current
            evidence = body_area & (difference >= 24)
            if np.count_nonzero(evidence) < required:
                # A sample labelled both E and F must not let another pair
                # confidently classify its matching appearance as present.
                residual_empty = np.abs(adjusted_current - ref_empty)[body_area].mean() / 255
                residual_phone = np.abs(adjusted_current - adjusted_phone)[body_area].mean() / 255
                if max(residual_empty, residual_phone) < .035:
                    ambiguous_match = True
                continue
            separation = float(np.abs(adjusted_phone - ref_empty)[evidence].mean() / 255)
            if separation < .10:
                continue
            errors_empty = np.abs(adjusted_current - ref_empty)[evidence].mean(axis=1) / 255
            errors_phone = np.abs(adjusted_current - adjusted_phone)[evidence].mean(axis=1) / 255
            de, dp = float(errors_empty.mean()), float(errors_phone.mean())
            if best is None or min(de, dp) < min(best):
                best = (de, dp)
            # Require both a tight match and a clear separation, over most of
            # the evidence. Matching a small portion of an unrelated object is
            # insufficient even when its average color resembles a phone.
            if dp < .055 and de - dp > .07 and dp < de * .40 and np.mean(errors_phone < .075) >= .80:
                decisions.add(True)
            elif de < .055 and dp - de > .07 and de < dp * .40 and np.mean(errors_empty < .075) >= .80:
                decisions.add(False)
    label = 'screen body'
    if best is not None:
        label += f' E={best[0]:.3f} F={best[1]:.3f}'
    if ambiguous_match or len(decisions) != 1:
        return None, f'UNCERTAIN {label}'
    return decisions.pop(), label
