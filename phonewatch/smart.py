"""Smart positive evidence first; conservative reference fallback on misses."""
import cv2
import numpy as np
from .reference import feature, ReferenceDetector


def moving_reference(references, image):
    """Bounded translations/rotations, never an unrestricted background match."""
    if not references.ready:
        return None, 'fallback: press R, E=empty, F=phone'
    if any(ref.shape != image.shape for ref in references.images.values()):
        return None, 'fallback: area changed; recalibrate R/E/F'
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if np.percentile(gray,95) < 8:
        return None, 'too dark; timer paused'
    current=feature(image)
    empty=feature(references.images['empty'])
    phone=feature(references.images['phone'])
    separation=ReferenceDetector.distance(empty,phone)
    if separation < .025:
        return None, 'fallback: references too similar; redo R/E/F'
    # Features are 96x128; allow ~6% camera/stand jitter and +/-5 degrees.
    distances=[]
    for ref in (empty,phone):
        best=float('inf')
        for angle in (-5,0,5):
            matrix=cv2.getRotationMatrix2D((48,64),angle,1)
            rotated=tuple(cv2.warpAffine(channel,matrix,(96,128)) for channel in ref)
            for dx in (-6,-3,0,3,6):
                for dy in (-6,-3,0,3,6):
                    aligned=tuple(channel[10+dy:118+dy,10+dx:86+dx] for channel in rotated)
                    target=tuple(channel[10:118,10:86] for channel in current)
                    best=min(best,ReferenceDetector.distance(aligned,target))
        distances.append(best)
    de,dp=distances
    if abs(de-dp) < max(.012,separation*.2) or min(de,dp) > max(.08,separation*.65):
        return None,f'fallback uncertain E={de:.3f} F={dp:.3f}'
    return dp < de,f'fallback R/E/F E={de:.3f} F={dp:.3f}'


class SmartDetector:
    def __init__(self,references,model_factory):
        self.references=references
        self.model_factory=model_factory
        self.model=None
        self.model_failed=False
        self.sift=cv2.SIFT_create(nfeatures=700,contrastThreshold=.015)
        self.cache=None
        self.keypoints=None
        self.descriptors=None

    def _learn(self):
        if not self.references.ready:
            self.cache=None
            self.keypoints,self.descriptors=None,None
            return
        phone=self.references.images['phone']
        empty=self.references.images['empty']
        cache=(id(phone),id(empty))
        if self.cache == cache:
            return
        self.cache=cache
        self.keypoints,self.descriptors=None,None
        if phone.shape != empty.shape:
            return
        pg=cv2.cvtColor(phone,cv2.COLOR_BGR2GRAY)
        eg=cv2.cvtColor(empty,cv2.COLOR_BGR2GRAY)
        # Match features of the object added by F, excluding static background.
        difference=cv2.absdiff(cv2.GaussianBlur(pg,(5,5),0),cv2.GaussianBlur(eg,(5,5),0))
        mask=np.uint8(difference>25)*255
        mask=cv2.dilate(mask,np.ones((5,5),np.uint8))
        self.keypoints,self.descriptors=self.sift.detectAndCompute(pg,mask)
        # E/F can differ slightly in exposure or position. Remove features also
        # recognizable on the empty stand instead of learning those as phone.
        _,background=self.sift.detectAndCompute(eg,None)
        if self.descriptors is not None and background is not None and len(background)>=2:
            pairs=cv2.BFMatcher().knnMatch(self.descriptors,background,k=2)
            reused={pair[0].queryIdx for pair in pairs if len(pair)==2 and pair[0].distance<250 and pair[0].distance<.7*pair[1].distance}
            keep=[i for i in range(len(self.descriptors)) if i not in reused]
            self.keypoints=[self.keypoints[i] for i in keep]
            self.descriptors=self.descriptors[keep] if keep else None

    def _match(self,image):
        self._learn()
        if self.descriptors is None or len(self.descriptors)<6:
            return False
        points,descriptors=self.sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None)
        if descriptors is None or len(descriptors)<2:
            return False
        matches=cv2.BFMatcher().knnMatch(self.descriptors,descriptors,k=2)
        good=[pair[0] for pair in matches if len(pair)==2 and pair[0].distance < .7*pair[1].distance]
        if len(good)<6:
            return False
        source=np.float32([self.keypoints[m.queryIdx].pt for m in good])
        target=np.float32([points[m.trainIdx].pt for m in good])
        matrix,inliers=cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=3)
        if matrix is None or inliers is None or inliers.sum()<6 or inliers.mean()<.6:
            return False
        # Localize distributed phone features, not six copies of one tiny ring.
        valid=source[inliers.ravel().astype(bool)]
        height,width=image.shape[:2]
        if np.ptp(valid[:,0]) < width*.08 or np.ptp(valid[:,1]) < height*.08:
            return False
        scale=np.hypot(matrix[0,0],matrix[1,0])
        angle=abs(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])))
        return .7 < scale < 1.4 and angle<20 and abs(matrix[0,2])<width*.25 and abs(matrix[1,2])<height*.25

    def detect(self,image):
        if np.percentile(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),95)<8:
            return None,'too dark; timer paused',[]
        if not self.model_failed:
            try:
                if self.model is None:
                    self.model=self.model_factory()
                polygons,label=self.model.detect(image)
                if polygons:
                    return True,f'smart YOLO: {label}',polygons
            except Exception as exc:
                self.model_failed=True
                print(f'Smart model unavailable ({type(exc).__name__}: {exc}); continuing feature/reference detection.',flush=True)
        if self._match(image):
            return True,'smart: moving phone features',[]
        state,label=moving_reference(self.references,image)
        return state,label,[]
