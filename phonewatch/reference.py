"""Nearest-reference classification with multiple E/F samples per parking area."""
from pathlib import Path
from uuid import uuid4
from time import time_ns
import cv2
import numpy as np


def feature(image):
    gray = cv2.cvtColor(cv2.resize(image, (96, 128)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = np.clip(gray - np.median(gray) + .5, 0, 1)
    edge = cv2.Sobel(gray, cv2.CV_32F, 1, 0) + cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return gray, edge


class ReferenceDetector:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.samples = {'empty': [], 'phone': []}
        self.images = {}  # Latest sample, for compatibility with earlier callers.
        self.revision = 0
        self._feature_cache = {}
        self._moving_cache = {}
        for name in self.samples:
            legacy = directory / f'{name}.png'
            paths = ([legacy] if legacy.exists() else []) + sorted((directory/name).glob('sample-*.png'))
            for path in paths:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f'Cannot read reference: {path}')
                self.samples[name].append(image)
            if self.samples[name]:
                self.images[name] = self.samples[name][-1]

    @property
    def ready(self):
        return all(self.samples.values())

    @property
    def counts(self):
        return {name: len(images) for name, images in self.samples.items()}

    def _invalidate(self):
        self.revision += 1
        self._feature_cache.clear()
        self._moving_cache.clear()

    def clear(self):
        # Remove only managed reference files; retain unrelated user files.
        for name in self.samples:
            (self.directory/f'{name}.png').unlink(missing_ok=True)
            for path in (self.directory/name).glob('sample-*.png'):
                path.unlink()
            self.samples[name].clear()
        self.images.clear()
        self._invalidate()

    def save(self, name, image):
        if name not in self.samples:
            raise ValueError('Reference label must be empty or phone')
        directory = self.directory/name
        directory.mkdir(exist_ok=True)
        path = directory/f'sample-{time_ns():020d}-{uuid4().hex[:8]}.png'
        if not cv2.imwrite(str(path), image):
            raise RuntimeError('Could not save reference')
        saved = image.copy()
        self.samples[name].append(saved)
        self.images[name] = saved
        self._invalidate()

    def compatible(self, name, shape):
        return [image for image in self.samples[name] if image.shape == shape]

    def features(self, name, shape):
        key = (name, tuple(shape))
        if key not in self._feature_cache:
            self._feature_cache[key] = [feature(image) for image in self.compatible(name, shape)]
        return self._feature_cache[key]

    def moving_features(self, name, shape):
        key = (name, tuple(shape))
        if key not in self._moving_cache:
            prepared = []
            for ref in self.features(name, shape):
                views = []
                for angle in (-5, 0, 5):
                    matrix = cv2.getRotationMatrix2D((48, 64), angle, 1)
                    rotated = tuple(cv2.warpAffine(channel, matrix, (96, 128)) for channel in ref)
                    for dx in (-6,-3,0,3,6):
                        for dy in (-6,-3,0,3,6):
                            views.append(tuple(channel[10+dy:118+dy,10+dx:86+dx] for channel in rotated))
                prepared.append(views)
            self._moving_cache[key] = prepared
        return self._moving_cache[key]

    @staticmethod
    def distance(a, b):
        return float(.75*np.abs(a[0]-b[0]).mean() + .25*np.abs(a[1]-b[1]).mean())

    def evidence(self, image, moving=False):
        if not self.ready:
            return None, 'reference: save E=empty and F=phone', None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if np.percentile(gray, 95) < 8:
            return None, 'too dark; counting uncertain time', None
        groups = {name: self.features(name, image.shape) for name in self.samples}
        if not all(groups.values()):
            return None, 'reference size changed: recalibrate E/F', None
        current = feature(image)
        target = tuple(channel[10:118,10:86] for channel in current)
        best = {}
        for name, refs in groups.items():
            if moving:
                distances = [min(self.distance(view,target) for view in views)
                             for views in self.moving_features(name,image.shape)]
            else:
                distances = [self.distance(ref,current) for ref in refs]
            index = int(np.argmin(distances))
            best[name] = (distances[index], refs[index])
        de, dp = best['empty'][0], best['phone'][0]
        separation = self.distance(best['empty'][1], best['phone'][1])
        ambiguous = min(de,dp) < .02 and abs(de-dp) < .012
        scores = {'empty': de, 'phone': dp, 'ambiguous': ambiguous}
        prefix = 'fallback R/E/F' if moving else 'reference'
        label = f'{prefix} E={de:.3f} F={dp:.3f}'
        if separation < .025 or abs(de-dp) < max(.012,separation*.2) or min(de,dp) > max(.08,separation*.65):
            return None, f'UNCERTAIN {label}', scores
        return dp < de, label, scores

    def detect(self, image):
        state, label, _ = self.evidence(image)
        return state, label
