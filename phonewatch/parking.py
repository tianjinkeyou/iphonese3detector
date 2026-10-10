"""Conservative evidence that a calibrated phone parking pose is occupied.

This intentionally answers a narrower question than general phone detection:
whether the body learned from F still occupies its original parking footprint.
An empty result requires E to explain every learned footprint, even when a phone
is still visible elsewhere in the camera image.
"""
import cv2
import numpy as np


def _small(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _channels(image):
    # Retain colour so a skin-coloured occluder cannot match a grey background
    # solely because it has the same brightness.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.
    return cv2.GaussianBlur(lab, (3, 3), 0)


def _distance(a, b):
    delta = np.abs(a - b)
    return delta[..., 0] * .8 + delta[..., 1:].mean(axis=-1) * .2


def _exposure(source, target, mask):
    """Estimate lighting from visible background, never from the phone body."""
    result = source.copy()
    if np.count_nonzero(mask) < 24:
        return result
    # Background gain is identifiable only when the background has texture.
    x, y = source[:, :, 0][mask], target[:, :, 0][mask]
    xp, yp = np.percentile(x, [10, 90]), np.percentile(y, [10, 90])
    gain = np.clip((yp[1] - yp[0]) / (xp[1] - xp[0]), .7, 1.4) if xp[1] - xp[0] > .1 else 1.
    result[:, :, 0] *= gain
    offsets = np.median(target[mask] - result[mask], axis=0)
    # A large correction cannot recover detail lost to clipping. In a tight
    # P crop it can also mistake the black phone itself for dark background.
    # Leave those images uncorrected rather than manufacture empty evidence.
    if abs(float(offsets[0])) > .15:
        return source.copy()
    # Very large chromatic changes are not an ordinary exposure adjustment.
    offsets[1:] = np.clip(offsets[1:], -.035, .035)
    return np.clip(result + offsets, 0., 1.)


def _align(source, target, mask=None):
    """Align modest camera motion using the background, with bounded geometry."""
    height, width = source.shape[:2]
    a = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    if mask is None:
        mask = np.ones((height, width), np.uint8) * 255
    else:
        mask = np.uint8(mask) * 255
    identity = np.float32([[1, 0, 0], [0, 1, 0]])
    matrix = identity.copy()
    try:
        # ECC handles uniform brightness changes. Its matrix maps template to
        # input; invert it before warping the input into the template frame.
        _, inverse = cv2.findTransformECC(
            b.astype(np.float32) / 255., a.astype(np.float32) / 255.,
            identity.copy(), cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 35, .001), mask, 3)
        candidate = cv2.invertAffineTransform(inverse)
        centre = np.array([width / 2., height / 2., 1.])
        displacement = candidate @ centre - centre[:2]
        angle = abs(np.degrees(np.arctan2(candidate[1, 0], candidate[0, 0])))
        if angle <= 4.1 and np.max(np.abs(displacement)) <= max(5., max(width, height) * .035):
            matrix = candidate
    except cv2.error:
        pass  # A textureless background provides no registration evidence.
    aligned = cv2.warpAffine(source, matrix, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT101)
    valid = cv2.warpAffine(np.ones((height, width), np.uint8), matrix, (width, height),
                           flags=cv2.INTER_NEAREST).astype(bool)
    return aligned, valid


class ParkingDetector:
    def __init__(self, references):
        self.references = references
        self._cache = None
        self._templates = []
        self._reason = 'save E=empty and F=parked phone'
        self._size = None
        self._scale = None
        self._screen = None

    def _learn(self, shape, screen_polygon=None):
        screen_key = None
        if screen_polygon is not None:
            points = np.asarray(screen_polygon, np.float32).reshape(-1, 2)
            if len(points) == 4 and np.isfinite(points).all():
                screen_key = tuple(float(x) for x in points.ravel())
        key = (self.references.revision, tuple(shape), screen_key)
        if self._cache == key:
            return
        self._cache = key
        self._templates = []
        self._reason = 'save E=empty and F=parked phone'
        height, width = shape[:2]
        ratio = min(1., 256. / max(height, width))
        self._size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        self._scale = np.array([self._size[0] / width, self._size[1] / height], np.float32)
        empties = self.references.compatible('empty', shape)
        phones = self.references.compatible('phone', shape)
        if not empties or not phones:
            return
        empty_images = [_small(image, self._size) for image in empties]
        screen_mask = np.zeros(self._size[::-1], np.uint8)
        if screen_key is not None:
            points = np.asarray(screen_key, np.float32).reshape(4, 2) * self._scale
            cv2.fillConvexPoly(screen_mask, np.rint(points).astype(np.int32), 1)
            screen_mask = cv2.dilate(screen_mask, np.ones((3, 3), np.uint8))
        self._screen = screen_mask.astype(bool)
        for phone in phones:
            phone_image = _small(phone, self._size)
            # Find the compatible E scene requiring least change. This also
            # rejects an F sample accidentally duplicated under the E label.
            candidates = []
            for empty_image in empty_images:
                aligned, _ = _align(phone_image, empty_image)
                e, f = _channels(empty_image), _channels(aligned)
                background = np.zeros(e.shape[:2], bool)
                border = max(3, round(min(e.shape[:2]) * .08))
                background[:border] = background[-border:] = True
                background[:, :border] = background[:, -border:] = True
                f = _exposure(f, e, background)
                difference = _distance(e, f)
                # Refine exposure after removing likely added-object pixels.
                background = difference < max(.035, float(np.percentile(difference, 45)))
                f = _exposure(_channels(aligned), e, background)
                difference = _distance(e, f)
                candidates.append((float(difference.mean()), empty_image, e, f, difference))
            _, empty_image, e, f, difference = min(candidates, key=lambda item: item[0])
            noise = float(np.percentile(difference, 35))
            # A P crop can contain almost entirely phone. In that case its
            # lower difference percentile is body contrast, not scene noise.
            threshold = max(.055, noise + .04) if noise < .12 else .055
            changed = np.uint8(difference > threshold)
            changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(changed, 8)
            footprint = np.zeros_like(changed)
            for index in range(1, count):
                if stats[index, cv2.CC_STAT_AREA] >= max(40, changed.size * .008):
                    component = np.uint8(labels == index)
                    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(footprint, contours, -1, 1, cv2.FILLED)
            area = int(footprint.sum())
            if area < max(80, footprint.size * .015):
                self._templates = []
                self._reason = 'E/F do not isolate a reliable phone footprint'
                return
            if area > footprint.size * .9 and float(np.std(f[:, :, 0])) < .012:
                self._templates = []
                self._reason = 'featureless full crop: include visible phone body or screen edges'
                return
            # The outer body/rim remains informative when the screen changes.
            radius = max(2, round(min(self._size) * .025))
            inner = cv2.erode(footprint, np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8),
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
            rim = footprint.astype(bool) & ~inner.astype(bool) & ~self._screen
            rim &= difference > .035
            if int(rim.sum()) < max(32, footprint.size * .0015):
                self._templates = []
                self._reason = 'E/F body evidence is too small; include the phone bezel'
                return
            separation = float(difference[rim].mean())
            if separation < .065:
                self._templates = []
                self._reason = 'E/F phone body is too similar to the empty scene'
                return
            mask = footprint.astype(bool)
            background = ~cv2.dilate(footprint, np.ones((9, 9), np.uint8)).astype(bool)
            background[:3] = background[-3:] = False
            background[:, :3] = background[:, -3:] = False
            # Local phone movement is independent of camera registration.
            # Keep shifts small so movement out of the parking pose is visible.
            shifts = (0, -1, 1, -2, 2, -3, 3)
            variants = []
            for angle in (0, -3, 3):
                rotation = cv2.getRotationMatrix2D((self._size[0] / 2., self._size[1] / 2.), angle, 1.)
                for dx in shifts:
                    for dy in shifts:
                        motion = rotation.copy()
                        motion[:, 2] += [dx, dy]
                        view = cv2.warpAffine(f, motion, self._size, borderMode=cv2.BORDER_REFLECT101)
                        body = cv2.warpAffine(np.uint8(rim), motion, self._size,
                                              flags=cv2.INTER_NEAREST).astype(bool)
                        yy, xx = (axis.astype(np.int16) for axis in np.nonzero(body))
                        # Retain only body pixels, keeping many motion poses
                        # inexpensive even with several saved phone images.
                        variants.append((view[yy, xx], yy, xx))
            # Compare to every E lighting sample, in this pose's coordinates.
            empty_views = []
            for alternative in empty_images:
                aligned, valid = _align(alternative, empty_image, background)
                empty_views.append((_channels(aligned), valid))
            self._templates.append({'empty_image': empty_image, 'empty': e, 'phone': f,
                                    'mask': mask, 'rim': rim, 'background': background,
                                    'separation': separation, 'variants': variants,
                                    'empty_views': empty_views})
        self._reason = 'ready'

    def ready(self, shape, screen_polygon=None):
        self._learn(shape, screen_polygon)
        return bool(self._templates)

    def detect(self, image, phone_polygons=None, screen_polygon=None):
        self._learn(image.shape, screen_polygon)
        evidence = {'reliable_empty': False, 'immediate_empty': False, 'empty': None, 'phone': None,
                    'poses': len(self._templates), 'body_pixels': 0}
        if not self._templates:
            evidence['reason'] = self._reason
            return None, 'PARKING: ' + self._reason, evidence
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if np.percentile(gray, 95) < 20:
            evidence['reason'] = 'camera too dark'
            return None, 'PARKING: too dark', evidence
        if np.percentile(gray, 5) > 245:
            evidence['reason'] = 'camera overexposed'
            return None, 'PARKING: overexposed', evidence
        current_image = _small(image, self._size)
        poses = []
        for template in self._templates:
            aligned, valid = _align(current_image, template['empty_image'], template['background'])
            current = _exposure(_channels(aligned), template['empty'], template['background'] & valid)
            body = template['rim'] & valid
            footprint = template['mask'] & valid
            if body.sum() < template['rim'].sum() * .95 or footprint.sum() < template['mask'].sum() * .95:
                poses.append((None, 1., 1., int(body.sum()), False))
                continue
            best_phone = 1.
            phone_match = False
            for phone, yy, xx in template['variants']:
                use = valid[yy, xx]
                if use.sum() < template['rim'].sum() * .95:
                    continue
                residual = _distance(current[yy[use], xx[use]], phone[use])
                score = float(residual.mean())
                best_phone = min(best_phone, score)
                threshold = max(.025, template['separation'] * .22)
                if score < threshold and float(np.mean(residual < max(.055, template['separation'] * .4))) >= .88:
                    phone_match = True
                    break
            best_empty = 1.
            empty_match = False
            strict_empty = False
            for empty, empty_valid in template['empty_views']:
                use = footprint & empty_valid
                if use.sum() < template['mask'].sum() * .95:
                    continue
                candidate = _exposure(current, empty, template['background'] & valid & empty_valid)
                residual = _distance(candidate[use], empty[use])
                score = float(residual.mean())
                best_empty = min(best_empty, score)
                background = template['background'] & valid & empty_valid
                visible_background = True
                if int(background.sum()) >= 64:
                    # Brightness compensation must not invent a recognizable
                    # E scene from washed-out or contrast-clipped video.
                    background_error = _distance(candidate[background], empty[background])
                    visible_background = float(np.mean(background_error < .055)) >= .55
                    e_edge = cv2.Sobel(empty[:, :, 0], cv2.CV_32F, 1, 0)
                    c_edge = cv2.Sobel(candidate[:, :, 0], cv2.CV_32F, 1, 0)
                    e_vertical = cv2.Sobel(empty[:, :, 0], cv2.CV_32F, 0, 1)
                    c_vertical = cv2.Sobel(candidate[:, :, 0], cv2.CV_32F, 0, 1)
                    expected_texture = float(np.sqrt(np.mean(e_edge[background] ** 2 + e_vertical[background] ** 2)))
                    actual_texture = float(np.sqrt(np.mean(c_edge[background] ** 2 + c_vertical[background] ** 2)))
                    if expected_texture > .018 and actual_texture < expected_texture * .25:
                        visible_background = False
                # Require the full footprint to be explained by E; checking
                # only the rim could mistake an occluding hand for departure.
                threshold = max(.02, min(.035, template['separation'] * .16))
                if visible_background and score < threshold and float(np.mean(residual > .055)) <= .06:
                    empty_match = True
                    if score < .015 and float(np.mean(residual > .055)) <= .015:
                        strict_empty = True
            state = True if phone_match and not empty_match else False if empty_match and not phone_match else None
            poses.append((state, best_empty, best_phone, int(body.sum()), strict_empty))
            if state is True:
                break  # Any learned pose is sufficient to establish parked.
        evidence['empty'] = float(min(p[1] for p in poses))
        evidence['phone'] = float(min(p[2] for p in poses))
        evidence['body_pixels'] = int(sum(p[3] for p in poses))
        # Any calibrated pose can explain a parked phone. Empty must explain
        # ALL of them, otherwise a different saved pose could still be occupied.
        if any(p[0] is True for p in poses):
            state, reason = True, 'body matches F'
        elif all(p[0] is False for p in poses):
            state, reason = False, 'all parking footprints match E'
        else:
            state, reason = None, 'body/empty evidence uncertain'
        # Model detections elsewhere in R do not prevent recording a pickup.
        # A model phone overlapping the expected parking body is a conflict.
        if state is False and phone_polygons:
            model_mask = np.zeros(self._size[::-1], np.uint8)
            for polygon in phone_polygons:
                points = np.asarray(polygon, np.float32).reshape(-1, 2)
                if len(points) >= 3 and np.isfinite(points).all():
                    cv2.fillPoly(model_mask, [np.rint(points * self._scale).astype(np.int32)], 1)
            for template in self._templates:
                if float(model_mask[template['rim']].mean()) > .25:
                    state, reason = None, 'model phone overlaps the parking body; conflicting evidence'
                    break
        evidence['reliable_empty'] = state is False
        evidence['immediate_empty'] = state is False and all(p[4] for p in poses)
        evidence['reason'] = reason
        label = f"PARKING: {'parked' if state is True else 'empty' if state is False else 'uncertain'} E={evidence['empty']:.3f} F={evidence['phone']:.3f}"
        return state, label, evidence
