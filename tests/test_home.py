import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from phonewatch.home import HomeDetector


class HomeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.detector=HomeDetector(Path(self.tmp.name))
        self.empty=np.full((180,240,3),120,np.uint8)
        self.phone=self.empty.copy()
        cv2.rectangle(self.phone,(60,20),(180,160),(12,12,12),-1)
        cv2.ellipse(self.phone,(120,140),(13,8),-10,0,360,(90,90,90),2)
        self.box=(100,126,40,28)

    def calibrate(self):
        self.detector.save_button(self.phone,self.box)
        self.detector.save_empty(self.empty)

    def test_black_screen_button_and_empty(self):
        self.calibrate()
        self.assertTrue(self.detector.detect(self.phone)[0])
        self.assertFalse(self.detector.detect(self.empty)[0])
        loaded=HomeDetector(Path(self.tmp.name))
        self.assertTrue(loaded.detect(self.phone)[0])

    def test_entirely_black_frame_is_unknown(self):
        self.calibrate()
        self.assertIsNone(self.detector.detect(np.zeros_like(self.phone))[0])

    def test_brightness_shift_and_small_movement(self):
        self.calibrate()
        shifted=cv2.warpAffine(self.phone,np.float32([[1,0,3],[0,1,2]]),(240,180),borderValue=(120,120,120))
        self.assertTrue(self.detector.detect(shifted)[0])
        brighter=np.clip(self.phone.astype(np.int16)+20,0,255).astype(np.uint8)
        self.assertTrue(self.detector.detect(brighter)[0])

    def test_hidden_button_does_not_mean_absence(self):
        self.calibrate()
        hidden=self.phone.copy()
        x,y,w,h=self.box
        hidden[y:y+h,x:x+w]=(40,80,180)
        self.assertIsNone(self.detector.detect(hidden)[0])
        # A textured cover must also be unknown.
        rng=np.random.default_rng(1)
        hidden[y:y+h,x:x+w]=rng.integers(0,255,(h,w,3),dtype=np.uint8)
        self.assertIsNone(self.detector.detect(hidden)[0])

    def test_missing_empty_and_invalid_patch(self):
        self.detector.save_button(self.phone,self.box)
        self.assertIsNone(self.detector.detect(self.empty)[0])
        with self.assertRaises(ValueError):
            self.detector.save_button(self.empty,self.box)
        with self.assertRaises(ValueError):
            self.detector.save_button(self.phone,(0,0,3,3))
        self.assertIsNone(self.detector.detect(np.zeros((90,120,3),np.uint8))[0])
