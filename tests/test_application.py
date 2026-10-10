import itertools
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from phonewatch.__main__ import main


class Camera:
    def __init__(self, frames):
        self.frames=iter(frames)
        self.released=False
    def isOpened(self):
        return True
    def read(self):
        return True,next(self.frames).copy()
    def release(self):
        self.released=True


class ApplicationTests(unittest.TestCase):
    def test_reference_camera_loop_records_photos_and_usage(self):
        empty=np.full((160,120,3),160,np.uint8)
        phone=empty.copy()
        cv2.rectangle(phone,(25,15),(95,145),(20,20,20),-1)
        camera=Camera([empty,phone,phone,phone,empty,empty,empty])
        with tempfile.TemporaryDirectory() as tmp:
            argv=['phonewatch','--detector','reference','--delay','0','--photo-interval','1','--data-dir',tmp]
            with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',side_effect=[ord('e'),ord('f'),-1,-1,-1,-1,ord('q')]), patch('phonewatch.__main__.time.monotonic',side_effect=itertools.count(0,1)):
                main()
            self.assertTrue(camera.released)
            photos=list((Path(tmp)/'photos').glob('*.jpg'))
            self.assertEqual(len(photos),7)
            self.assertIsNotNone(cv2.imread(str(photos[0])))
            db=sqlite3.connect(Path(tmp)/'usage.sqlite3')
            states=[row[0] for row in db.execute('SELECT state FROM events')]
            self.assertIn('present',states)
            self.assertIn('absent',states)
            self.assertGreater(db.execute('SELECT SUM(seconds) FROM daily_state').fetchone()[0],0)
            db.close()

    def test_smart_fallback_roi_persists_across_restart(self):
        from phonewatch.smart import SmartDetector
        class Miss:
            def detect(self,image):
                return [],'miss'
        factory=lambda refs,model_factory: SmartDetector(refs,Miss)
        empty=np.full((160,120,3),160,np.uint8)
        phone=empty.copy()
        cv2.rectangle(phone,(25,15),(95,145),(20,20,20),-1)
        with tempfile.TemporaryDirectory() as tmp:
            argv=['phonewatch','--detector','smart','--delay','0','--data-dir',tmp]
            camera=Camera([empty,empty,phone,phone,empty])
            with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.selectROI',return_value=(10,10,100,140)), patch('cv2.waitKey',side_effect=[ord('r'),ord('e'),ord('f'),-1,ord('q')]), patch('phonewatch.smart.SmartDetector',side_effect=factory):
                main()
            self.assertTrue((Path(tmp)/'parking-area.json').exists())
            camera=Camera([phone,empty])
            with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',side_effect=[-1,ord('q')]), patch('phonewatch.smart.SmartDetector',side_effect=factory):
                main()
            db=sqlite3.connect(Path(tmp)/'usage.sqlite3')
            states=[row[0] for row in db.execute('SELECT state FROM events')]
            self.assertEqual(states[-2:],['present','absent'])
            db.close()

    def test_multiple_capture_clear_confirmation_and_data_preservation(self):
        from phonewatch.reference import ReferenceDetector
        empty=np.full((160,120,3),160,np.uint8)
        phone=empty.copy()
        cv2.rectangle(phone,(25,15),(95,145),(20,20,20),-1)
        with tempfile.TemporaryDirectory() as tmp:
            argv=['phonewatch','--detector','reference','--delay','0','--data-dir',tmp]
            camera=Camera([empty,empty,phone,phone,phone,phone])
            with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',side_effect=[ord('e'),ord('e'),ord('f'),ord('f'),ord('c'),ord('q')]):
                main()
            refs=ReferenceDetector(Path(tmp)/'references')
            self.assertEqual(refs.counts,{'empty':2,'phone':2})
            marker=Path(tmp)/'photos'/'keep.txt'
            marker.write_text('keep this file')
            camera=Camera([phone,phone,phone])
            with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',side_effect=[ord('c'),ord('c'),ord('q')]):
                main()
            self.assertEqual(ReferenceDetector(Path(tmp)/'references').counts,{'empty':0,'phone':0})
            self.assertEqual(marker.read_text(),'keep this file')
            db=sqlite3.connect(Path(tmp)/'usage.sqlite3')
            self.assertGreater(db.execute('SELECT COUNT(*) FROM events').fetchone()[0],0)
            db.close()
