import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from phonewatch.reference import ReferenceDetector
from phonewatch.smart import SmartDetector,moving_reference


class MissModel:
    def detect(self,image):
        return [],'miss'


class SmartTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.refs=ReferenceDetector(Path(self.tmp.name))
        self.empty=np.full((240,200,3),160,np.uint8)
        self.phone=self.empty.copy()
        cv2.rectangle(self.phone,(40,25),(160,215),(20,20,20),-1)
        rng=np.random.default_rng(7)
        for _ in range(100):
            x,y=map(int,rng.integers([50,35],[150,205]))
            color=int(rng.integers(50,200))
            cv2.circle(self.phone,(x,y),int(rng.integers(2,5)),(color,color,color),-1)
        self.refs.save('empty',self.empty)
        self.refs.save('phone',self.phone)

    def test_model_positive_has_priority(self):
        class Model:
            def detect(self,image):
                return [np.array([[0,0],[10,0],[10,10],[0,10]])],'normal'
        smart=SmartDetector(self.refs,Model)
        self.assertTrue(smart.detect(self.empty)[0])
        self.assertIn('YOLO',smart.detect(self.empty)[1])

    def test_phone_features_survive_translation_and_rotation(self):
        matrix=cv2.getRotationMatrix2D((100,120),4,1.02)
        matrix[:,2]+=[4,3]
        image=cv2.warpAffine(self.phone,matrix,(200,240),borderValue=(160,160,160))
        result=SmartDetector(self.refs,MissModel).detect(image)
        self.assertTrue(result[0])
        self.assertIn('moving phone features',result[1])

    def test_miss_uses_reference_and_empty_is_not_feature_match(self):
        result=SmartDetector(self.refs,MissModel).detect(self.empty)
        self.assertFalse(result[0])
        self.assertIn('fallback R/E/F',result[1])

    def test_feature_failure_falls_back_for_textureless_black_phone(self):
        plain=self.empty.copy()
        cv2.rectangle(plain,(40,25),(160,215),(20,20,20),-1)
        self.refs.save('phone',plain)
        smart=SmartDetector(self.refs,MissModel)
        class NoTexture:
            def detectAndCompute(self,image,mask):
                return [],None
        smart.sift=NoTexture()
        result=smart.detect(plain)
        self.assertTrue(result[0])
        self.assertIn('fallback R/E/F',result[1])

    def test_reference_handles_small_camera_shift(self):
        shifted=cv2.warpAffine(self.phone,np.float32([[1,0,5],[0,1,4]]),(200,240),borderValue=(160,160,160))
        self.assertTrue(moving_reference(self.refs,shifted)[0])
        self.assertFalse(moving_reference(self.refs,self.empty)[0])

    def test_model_error_does_not_stop_reference_mode(self):
        calls=[]
        def unavailable():
            calls.append(1)
            raise OSError('model not available')
        smart=SmartDetector(self.refs,unavailable)
        self.assertFalse(smart.detect(self.empty)[0])
        self.assertFalse(smart.detect(self.empty)[0])
        self.assertEqual(len(calls),1)

    def test_dark_frame_and_uncalibrated_misses_are_unknown(self):
        smart=SmartDetector(self.refs,MissModel)
        self.assertIsNone(smart.detect(np.zeros_like(self.phone))[0])
        self.refs.clear()
        self.assertIsNone(smart.detect(self.empty)[0])

    def test_recalibration_invalidates_learned_features(self):
        smart=SmartDetector(self.refs,MissModel)
        self.assertTrue(smart.detect(self.phone)[0])
        self.refs.clear()
        self.assertIsNone(smart.detect(self.phone)[0])
