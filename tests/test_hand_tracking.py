import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
from phonewatch.hand_tracking import HandTracker,hand_search_box,map_landmarks


class HandTrackingTests(unittest.TestCase):
    def test_crop_includes_incoming_palm_and_clips_to_frame(self):
        box=hand_search_box((480,640,3),[[100,200],[500,200],[500,420],[100,420]])
        self.assertEqual(box,(0,0,640,480))
        self.assertEqual(hand_search_box((480,640,3),None),(0,0,640,480))

    def test_rotated_crop_landmarks_map_back_to_full_frame(self):
        point=SimpleNamespace(x=.2,y=.3)
        box=(100,40,200,100)
        plain=map_landmarks([point],box,(200,400,3),0)[0]
        cw=map_landmarks([point],box,(200,400,3),90)[0]
        ccw=map_landmarks([point],box,(200,400,3),270)[0]
        np.testing.assert_allclose(plain,(.35,.35))
        np.testing.assert_allclose(cw,(.4,.6))
        np.testing.assert_allclose(ccw,(.6,.3))

    def test_robust_crop_fallback_after_full_frame_miss(self):
        tracker=HandTracker.__new__(HandTracker)
        tracker.engine=MagicMock()
        tracker.fallback=MagicMock()
        tracker.engine.process.return_value=SimpleNamespace(multi_hand_landmarks=[])
        hand=SimpleNamespace(landmark=[SimpleNamespace(x=.5,y=.5)]*21)
        tracker.fallback.process.side_effect=[SimpleNamespace(multi_hand_landmarks=[]),SimpleNamespace(multi_hand_landmarks=[hand])]
        frame=np.full((240,320,3),160,np.uint8)
        hands=tracker.detect(frame,[[100,50],[200,50],[200,150],[100,150]])
        self.assertEqual(len(hands),1)
        self.assertEqual(tracker.source,'crop/90')
        self.assertEqual(tracker.fallback.process.call_count,2)

    def test_landmarks_only_keeps_full_frame_behavior(self):
        tracker=HandTracker.__new__(HandTracker)
        tracker.engine=MagicMock()
        tracker.engine.process.return_value=SimpleNamespace(multi_hand_landmarks=[])
        tracker.fallback=None
        self.assertEqual(tracker.detect(np.zeros((100,100,3),np.uint8)),[])

    def test_remote_hand_does_not_block_targeted_search(self):
        tracker=HandTracker.__new__(HandTracker)
        tracker.engine=MagicMock()
        tracker.fallback=MagicMock()
        remote=SimpleNamespace(landmark=[SimpleNamespace(x=.95,y=.95)]*21)
        near=SimpleNamespace(landmark=[SimpleNamespace(x=.5,y=.5)]*21)
        tracker.engine.process.return_value=SimpleNamespace(multi_hand_landmarks=[remote])
        tracker.fallback.process.return_value=SimpleNamespace(multi_hand_landmarks=[near])
        hands=tracker.detect(np.full((240,320,3),160,np.uint8),[[100,50],[200,50],[200,150],[100,150]])
        self.assertEqual(len(hands),2)
        self.assertEqual(tracker.source,'full-frame+crop/0')
        tracker.fallback.process.assert_called_once()
