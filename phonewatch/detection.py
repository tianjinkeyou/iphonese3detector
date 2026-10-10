"""Phone detection with bounded rotation and dark-scene fallbacks."""
import cv2
import numpy as np


def rotate_view(image, angle):
    """Expand canvas so diagonal phone corners are not clipped."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    out_width = int(np.ceil(height * sine + width * cosine))
    out_height = int(np.ceil(height * cosine + width * sine))
    matrix[0, 2] += (out_width - width) / 2
    matrix[1, 2] += (out_height - height) / 2
    rotated = cv2.warpAffine(image, matrix, (out_width, out_height))
    return rotated, matrix


def enhance_dark(image):
    """Boost luminance contrast without changing color channels independently."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class PhoneDetector:
    def __init__(self, model, classes, confidence=0.25, mode="robust"):
        self.model = model
        self.classes = classes
        self.confidence = confidence
        self.mode = mode

    def detect(self, image):
        """Return original-coordinate polygons and the successful view label.

        First try normal inference. Only misses pay for fallback inference.
        Rotations handle in-plane orientation, not missing visual information
        or a phone seen edge-on.
        """
        identity = np.array([[1., 0., 0.], [0., 1., 0.]])
        views = [(image, identity, "normal")]
        height, width = image.shape[:2]
        dark = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() < 85
        # Generate fallbacks lazily: normal detections avoid extra processing.
        def candidates():
            yield from views
            if self.mode == "fast":
                return
            source = image
            if dark:
                source = enhance_dark(image)
                yield source, identity, "low-light"
            for angle in (45, -45, 90):
                view, matrix = rotate_view(source, angle)
                yield view, matrix, f"rotation {angle}"

        for view, matrix, label in candidates():
            result = self.model.predict(view, classes=self.classes, conf=self.confidence, verbose=False, device="cpu")[0]
            polygons = []
            inverse = cv2.invertAffineTransform(matrix)
            for box in result.boxes.xyxy.cpu().numpy():
                left, top, right, bottom = box
                corners = np.array([[[left, top], [right, top], [right, bottom], [left, bottom]]], dtype=np.float32)
                points = cv2.transform(corners, inverse)[0]
                points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                polygons.append(np.rint(points).astype(np.int32))
            if polygons:
                return polygons, label
        return [], "no phone"
