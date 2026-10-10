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
            self.assertGreater(db.execute('SELECT SUM(seconds) FROM daily').fetchone()[0],0)
            db.close()
