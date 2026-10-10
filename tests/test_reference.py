import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from phonewatch.reference import ReferenceDetector


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.detector = ReferenceDetector(Path(self.tmp.name))
        self.empty = np.full((160,120,3),160,np.uint8)
        self.phone = self.empty.copy()
        cv2.rectangle(self.phone,(25,15),(95,145),(20,20,20),-1)

    def calibrate(self):
        self.detector.save('empty',self.empty)
        self.detector.save('phone',self.phone)

    def test_uncalibrated_is_unknown(self):
        self.assertIsNone(self.detector.detect(self.empty)[0])
        self.detector.save('empty',self.empty)
        self.assertIsNone(self.detector.detect(self.empty)[0])

    def test_presence_absence_and_reload(self):
        self.calibrate()
        self.assertTrue(self.detector.detect(self.phone)[0])
        self.assertFalse(self.detector.detect(self.empty)[0])
        loaded=ReferenceDetector(Path(self.tmp.name))
        self.assertTrue(loaded.detect(self.phone)[0])
        self.detector.clear()
        self.assertFalse(ReferenceDetector(Path(self.tmp.name)).ready)

    def test_moderate_uniform_exposure_change(self):
        self.calibrate()
        brighter=np.clip(self.phone.astype(np.int16)+20,0,255).astype(np.uint8)
        self.assertTrue(self.detector.detect(brighter)[0])

    def test_ambiguous_image_pauses(self):
        self.calibrate()
        mixture=cv2.addWeighted(self.phone,.5,self.empty,.5,0)
        self.assertIsNone(self.detector.detect(mixture)[0])

    def test_similar_references_and_resolution_change(self):
        self.detector.save('empty',self.empty)
        self.detector.save('phone',self.empty)
        self.assertIsNone(self.detector.detect(self.empty)[0])
        self.calibrate()
        self.assertIsNone(self.detector.detect(np.zeros((320,240,3),np.uint8))[0])
