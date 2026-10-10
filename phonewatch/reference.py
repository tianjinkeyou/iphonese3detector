"""Two-reference classifier for a fixed camera and parking area."""
from pathlib import Path
import cv2
import numpy as np


def feature(image):
    gray = cv2.cvtColor(cv2.resize(image, (96, 128)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Suppress uniform exposure changes, retain the dark phone silhouette.
    gray = np.clip(gray - np.median(gray) + .5, 0, 1)
    edge = cv2.Sobel(gray, cv2.CV_32F, 1, 0) + cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return gray, edge


class ReferenceDetector:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.images = {}
        for name in ('empty', 'phone'):
            path = directory / f'{name}.png'
            if path.exists():
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f'Cannot read {path}')
                self.images[name] = image

    @property
    def ready(self):
        return all(name in self.images for name in ('empty', 'phone'))

    def clear(self):
        self.images.clear()
        for name in ('empty', 'phone'):
            (self.directory / f'{name}.png').unlink(missing_ok=True)

    def save(self, name, image):
        if not cv2.imwrite(str(self.directory / f'{name}.png'), image):
            raise RuntimeError('Could not save reference')
        self.images[name] = image.copy()

    @staticmethod
    def distance(a, b):
        return float(.75 * np.abs(a[0]-b[0]).mean() + .25 * np.abs(a[1]-b[1]).mean())

    def detect(self, image):
        if not self.ready:
            return None, 'reference: save E=empty and F=phone'
        if any(ref.shape != image.shape for ref in self.images.values()):
            return None, 'reference size changed: recalibrate E/F'
        current = feature(image)
        empty = feature(self.images['empty'])
        phone = feature(self.images['phone'])
        separation = self.distance(empty, phone)
        if separation < .025:
            return None, 'references too similar: reselect area and save E/F'
        de, dp = self.distance(current, empty), self.distance(current, phone)
        # Require a clear winner and reject appearances far from both references.
        if abs(de-dp) < max(.012, separation*.2) or min(de, dp) > max(.08, separation*.65):
            return None, f'UNCERTAIN empty={de:.3f} phone={dp:.3f}'
        return dp < de, f'reference empty={de:.3f} phone={dp:.3f}'
