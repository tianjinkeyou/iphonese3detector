"""Optional local MediaPipe hand landmarks; imported only for smart-watch."""
import cv2


class HandTracker:
    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError('smart-watch needs optional dependencies: python -m pip install -r requirements-watch.txt') from exc
        try:
            self.engine = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2, model_complexity=1,
                min_detection_confidence=.5, min_tracking_confidence=.5,
            )
        except AttributeError as exc:
            raise RuntimeError('Unsupported MediaPipe version. Install requirements-watch.txt (Python 3.10-3.12).') from exc

    def detect(self, frame):
        result = self.engine.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        return [[(point.x,point.y) for point in hand.landmark]
                for hand in (result.multi_hand_landmarks or [])]

    def close(self):
        self.engine.close()
