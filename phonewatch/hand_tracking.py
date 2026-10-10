"""Optional full-frame and crop/rotation MediaPipe hand landmarks."""
import cv2
import numpy as np
from .watch import hand_near_polygon


def hand_search_box(frame_shape, polygon):
    height,width=frame_shape[:2]
    if polygon is None:
        return 0,0,width,height
    points=np.asarray(polygon,dtype=np.float32)
    left,top=points.min(axis=0)
    right,bottom=points.max(axis=0)
    # Include the incoming palm outside the phone area, rather than cropping it off.
    margin=max(60,(right-left)*.5,(bottom-top)*.5)
    x=max(0,int(left-margin)); y=max(0,int(top-margin))
    right=min(width,int(np.ceil(right+margin))); bottom=min(height,int(np.ceil(bottom+margin)))
    return x,y,right-x,bottom-y


def map_landmarks(landmarks,box,frame_shape,rotation=0):
    x,y,w,h=box
    height,width=frame_shape[:2]
    mapped=[]
    for point in landmarks:
        px,py=point.x,point.y
        if rotation==90:
            px,py=py,1-px
        elif rotation==270:
            px,py=1-py,px
        mapped.append(((x+px*w)/width,(y+py*h)/height))
    return mapped


class HandTracker:
    def __init__(self,mode='robust'):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError('smart-watch needs optional dependencies: python -m pip install -r requirements-watch.txt') from exc
        self.mode=mode
        self.source='none'
        try:
            self.engine=mp.solutions.hands.Hands(
                static_image_mode=False,max_num_hands=2,model_complexity=1,
                min_detection_confidence=.3,min_tracking_confidence=.4)
            self.fallback=None
            if mode=='robust':
                self.fallback=mp.solutions.hands.Hands(
                    static_image_mode=True,max_num_hands=2,model_complexity=1,
                    min_detection_confidence=.25,min_tracking_confidence=.4)
        except AttributeError as exc:
            raise RuntimeError('Unsupported MediaPipe version. Install requirements-watch.txt (Python 3.10-3.12).') from exc

    def detect(self,frame,area=None):
        result=self.engine.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        height,width=frame.shape[:2]
        full_hands = [map_landmarks(hand.landmark,(0,0,width,height),frame.shape)
                      for hand in result.multi_hand_landmarks or []]
        if full_hands:
            self.source='full-frame'
            if self.fallback is None or area is None or hand_near_polygon(full_hands,area,frame.shape):
                return full_hands
        else:
            self.source='none'
        if self.fallback is None:
            return full_hands
        box=hand_search_box(frame.shape,area)
        x,y,w,h=box
        crop=frame[y:y+h,x:x+w]
        # Luminance contrast enhancement retains color; candidate skin matching
        # uses raw frames separately and is not affected by this enhancement.
        lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB)
        lab[:,:,0]=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(lab[:,:,0])
        crop=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
        for rotation in (0,90,270):
            view=crop if rotation==0 else cv2.rotate(crop,cv2.ROTATE_90_CLOCKWISE if rotation==90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
            result=self.fallback.process(cv2.cvtColor(view,cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                targeted = [map_landmarks(hand.landmark,box,frame.shape,rotation)
                            for hand in result.multi_hand_landmarks]
                if area is None or hand_near_polygon(targeted,area,frame.shape):
                    self.source=f'full-frame+crop/{rotation}' if full_hands else f'crop/{rotation}'
                    return full_hands+targeted
        return full_hands

    def close(self):
        self.engine.close()
        if self.fallback is not None:
            self.fallback.close()
