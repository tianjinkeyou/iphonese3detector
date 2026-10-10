"""Rectify a fixed phone parking plane; calibration is local, not a phone image."""
import json
from pathlib import Path

import cv2
import numpy as np


def plane_matrix(points):
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Select four finite corners in perimeter order")
    contour = points.reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 100:
        raise ValueError("Corners must form a convex plane with sufficient area")
    # Restore the physical portrait proportions of an iPhone SE3 (67.3 x 138.4 mm).
    width = round(max(np.linalg.norm(points[1]-points[0]), np.linalg.norm(points[2]-points[3])))
    height = round(max(np.linalg.norm(points[3]-points[0]), np.linalg.norm(points[2]-points[1])))
    if min(width, height) < 10:
        raise ValueError("Phone plane is too narrow; change the camera angle")
    height = int(np.clip(max(height, width / (67.3 / 138.4)), 128, 640))
    width = round(height * (67.3 / 138.4))
    target = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    return cv2.getPerspectiveTransform(points, target), (width, height)


def rectify(frame, points):
    matrix, size = plane_matrix(points)
    return cv2.warpPerspective(frame, matrix, size), np.linalg.inv(matrix)


def map_polygon(polygon, inverse):
    return np.rint(cv2.perspectiveTransform(np.float32([polygon]), inverse)[0]).astype(np.int32)


def save_plane(path: Path, points, frame_shape):
    plane_matrix(points)
    path.write_text(json.dumps({"points": np.asarray(points).tolist(), "size": list(frame_shape[:2])}))


def load_plane(path: Path, frame_shape):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data["size"] != list(frame_shape[:2]):
        raise ValueError("Camera resolution changed. Delete parking-plane.json or recalibrate with --ignore-plane")
    points = np.asarray(data["points"], dtype=np.float32)
    plane_matrix(points)
    height, width = frame_shape[:2]
    if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
        raise ValueError("Saved plane is outside the image")
    return points


def select_plane(frame):
    window = "Click phone corners: top-left, top-right, bottom-right, bottom-left. Enter saves; Esc cancels"
    points = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            image = frame.copy()
            for number, point in enumerate(points, 1):
                cv2.circle(image, point, 5, (0, 255, 255), -1)
                cv2.putText(image, str(number), point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window, image)
            key = cv2.waitKey(30) & 0xff
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (10, 13) and len(points) == 4:
                try:
                    plane_matrix(points)
                    return np.float32(points)
                except ValueError as exc:
                    print(f"Invalid plane: {exc}. Click the four corners again.", flush=True)
                    points.clear()
    finally:
        cv2.destroyWindow(window)
