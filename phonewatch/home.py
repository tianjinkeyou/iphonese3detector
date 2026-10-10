"""Learn a visible Home-button patch at a fixed camera location.

This matches the user's selected local structure, not a semantic button model.
Missing/occluded structure is unknown unless an empty-area reference agrees.
"""
import json
from pathlib import Path
import cv2
import numpy as np


def edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32) / 255
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def contrast_distance(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    # Compare local structure, allowing a uniform exposure offset.
    ga -= np.median(ga)
    gb -= np.median(gb)
    return float(np.abs(ga-gb).mean() + .25*np.abs(edges(a)-edges(b)).mean())


class HomeDetector:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.template = None
        self.box = None
        self.size = None
        self.empty = cv2.imread(str(directory/'empty.png')) if (directory/'empty.png').exists() else None
        metadata = directory/'home.json'
        if metadata.exists():
            data = json.loads(metadata.read_text())
            self.box, self.size = data['box'], data['size']
            self.template = cv2.imread(str(directory/'button.png'))
            if self.template is None:
                raise ValueError('Cannot read Home-button calibration')

    @property
    def ready(self):
        return self.template is not None

    def save_empty(self, frame):
        if not cv2.imwrite(str(self.directory/'empty.png'), frame):
            raise RuntimeError('Could not save empty camera frame')
        self.empty = frame.copy()

    def save_button(self, frame, box):
        x,y,w,h = map(int,box)
        if min(w,h) < 12 or min(x,y) < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            raise ValueError('Select the complete Home button, at least 12 pixels wide/high')
        patch = frame[y:y+h,x:x+w]
        if edges(patch).std() < .035:
            raise ValueError('Selected patch has no visible structure. Include the button ring and add light.')
        if not cv2.imwrite(str(self.directory/'button.png'),patch):
            raise RuntimeError('Could not save Home-button patch')
        self.box = [x,y,w,h]
        self.size = list(frame.shape[:2])
        self.template = patch.copy()
        (self.directory/'home.json').write_text(json.dumps({'box':self.box,'size':self.size}))

    def detect(self, frame):
        if not self.ready:
            return None, 'Home mode: H=button; E=empty stand'
        if list(frame.shape[:2]) != self.size:
            return None, 'Home: camera size changed, recalibrate H/E'
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if np.percentile(gray,95) < 8:
            return None, 'Home: image too dark, timer paused'
        x,y,w,h=self.box
        margin=max(6,round(max(w,h)*.3))
        left,top=max(0,x-margin),max(0,y-margin)
        right,bottom=min(frame.shape[1],x+w+margin),min(frame.shape[0],y+h+margin)
        search=edges(frame[top:bottom,left:right])
        template=edges(self.template)
        best=-1.0
        for scale in (.9,1.,1.1):
            candidate=cv2.resize(template,(max(2,round(w*scale)),max(2,round(h*scale))))
            if candidate.shape[0] > search.shape[0] or candidate.shape[1] > search.shape[1]:
                continue
            if search.std() < .01 or candidate.std() < .01:
                continue
            score=cv2.matchTemplate(search,candidate,cv2.TM_CCOEFF_NORMED).max()
            best=max(best,float(score))
        empty_distance=None
        if self.empty is not None and self.empty.shape == frame.shape:
            empty_patch=self.empty[y:y+h,x:x+w]
            if contrast_distance(self.template,empty_patch) < .04:
                return None, 'Home: empty and button references too similar; redo H/E'
            current_patch=frame[y:y+h,x:x+w]
            empty_gray=cv2.cvtColor(self.empty,cv2.COLOR_BGR2GRAY).astype(np.float32)
            exposure=float(np.median(gray)-np.median(empty_gray))
            local_gray=cv2.cvtColor(current_patch,cv2.COLOR_BGR2GRAY).astype(np.float32)
            background_gray=empty_gray[y:y+h,x:x+w]
            # Keep local darkness as evidence after compensating global exposure.
            # Otherwise a covered black button and uniform empty stand both
            # normalize to a flat patch and incorrectly imply absence.
            brightness_error=float(np.abs(local_gray-exposure-background_gray).mean()/255)
            empty_distance=contrast_distance(current_patch,empty_patch)+.5*brightness_error
        # A visible learned ring is positive evidence. Empty needs its own evidence.
        if best >= .78 and (empty_distance is None or empty_distance > .04):
            return True, f'Home match={best:.2f}'
        if best < .45 and empty_distance is not None and empty_distance < .025:
            return False, f'Home: empty stand match={best:.2f}'
        return None, f'Home uncertain score={best:.2f}; timer paused'
