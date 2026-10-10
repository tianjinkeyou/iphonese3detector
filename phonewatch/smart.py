"""Smart positive evidence first; conservative reference fallback on misses."""
import cv2
import numpy as np


def _reference_evidence(references, image):
    return references.evidence(image, moving=True)


def moving_reference(references,image):
    state,label,_=_reference_evidence(references,image)
    return state,label


class SmartDetector:
    def __init__(self,references,model_factory):
        self.references=references
        self.model_factory=model_factory
        self.model=None
        self.model_failed=False
        self.sift=cv2.SIFT_create(nfeatures=700,contrastThreshold=.015)
        self.cache=None
        self.templates=[]

    def _learn(self, shape):
        cache = (self.references.revision, tuple(shape))
        if self.cache == cache:
            return
        self.cache = cache
        self.templates = []
        phones = self.references.compatible('phone',shape)
        empties = self.references.compatible('empty',shape)
        if not phones or not empties:
            return
        backgrounds = []
        for empty in empties:
            eg = cv2.cvtColor(empty,cv2.COLOR_BGR2GRAY)
            _,descriptors = self.sift.detectAndCompute(eg,None)
            backgrounds.append((eg,descriptors))
        matcher = cv2.BFMatcher()
        for phone in phones:
            pg = cv2.cvtColor(phone,cv2.COLOR_BGR2GRAY)
            # Use the closest empty scene to isolate this pose's added object.
            eg,_ = min(backgrounds,key=lambda item: float(cv2.absdiff(pg,item[0]).mean()))
            difference = cv2.absdiff(cv2.GaussianBlur(pg,(5,5),0),cv2.GaussianBlur(eg,(5,5),0))
            mask = cv2.dilate(np.uint8(difference>25)*255,np.ones((5,5),np.uint8))
            keypoints,descriptors = self.sift.detectAndCompute(pg,mask)
            if descriptors is None:
                continue
            reused = set()
            # Exclude features recognizable in ANY empty sample.
            for _,background in backgrounds:
                if background is None or len(background)<2:
                    continue
                pairs = matcher.knnMatch(descriptors,background,k=2)
                reused.update(pair[0].queryIdx for pair in pairs if len(pair)==2 and pair[0].distance<250 and pair[0].distance<.7*pair[1].distance)
            keep = [i for i in range(len(descriptors)) if i not in reused]
            if len(keep)>=6:
                self.templates.append(([keypoints[i] for i in keep],descriptors[keep]))

    def _match(self,image):
        self._learn(image.shape)
        if not self.templates:
            return False
        points,descriptors = self.sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None)
        if descriptors is None or len(descriptors)<2:
            return False
        height,width = image.shape[:2]
        for keypoints,template in self.templates:
            matches = cv2.BFMatcher().knnMatch(template,descriptors,k=2)
            good = [pair[0] for pair in matches if len(pair)==2 and pair[0].distance<.7*pair[1].distance]
            if len(good)<6:
                continue
            source = np.float32([keypoints[m.queryIdx].pt for m in good])
            target = np.float32([points[m.trainIdx].pt for m in good])
            matrix,inliers = cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=3)
            if matrix is None or inliers is None or inliers.sum()<6 or inliers.mean()<.6:
                continue
            valid = source[inliers.ravel().astype(bool)]
            if np.ptp(valid[:,0])<width*.08 or np.ptp(valid[:,1])<height*.08:
                continue
            scale = np.hypot(matrix[0,0],matrix[1,0])
            angle = abs(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])))
            if .7<scale<1.4 and angle<20 and abs(matrix[0,2])<width*.25 and abs(matrix[1,2])<height*.25:
                return True
        return False

    def detect(self,image):
        if np.percentile(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),95)<8:
            return None,'too dark; counting time',[]
        ref_state,ref_label,evidence=_reference_evidence(self.references,image)
        def reconcile(label,polygons):
            if evidence is not None and evidence.get("ambiguous"):
                return None,"CONFLICT matching both E/F samples; redo calibration",[]
            if ref_state is False:
                de,dp=evidence["empty"],evidence["phone"]
                if de < .02 and dp-de > max(.025,3*de):
                    return False,f"E empty confirmed; smart rejected E={de:.3f} F={dp:.3f}",[]
                return None,f"CONFLICT smart vs E; uncertain E={de:.3f} F={dp:.3f}",[]
            return True,label,polygons
        if not self.model_failed:
            try:
                if self.model is None:
                    self.model=self.model_factory()
                polygons,label=self.model.detect(image)
                if polygons:
                    return reconcile(f'smart YOLO: {label}',polygons)
            except Exception as exc:
                self.model_failed=True
                print(f'Smart model unavailable ({type(exc).__name__}: {exc}); continuing feature/reference detection.',flush=True)
        if self._match(image):
            return reconcile('smart: moving phone features',[])
        return ref_state,ref_label,[]
