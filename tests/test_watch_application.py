import itertools
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from phonewatch.__main__ import main


class WatchApplicationTests(unittest.TestCase):
    def test_watch_events_photo_and_camera_interruption_are_separate(self):
        frame=np.full((180,240,3),160,np.uint8)
        near=[[(.5,.5)]*21]
        far=[[(.95,.95)]*21]
        with tempfile.TemporaryDirectory() as tmp:
            camera=MagicMock()
            camera.isOpened.return_value=True
            camera.read.side_effect=[(True,frame)]*4+[(False,None)]+[(True,frame)]*3
            tracker=MagicMock()
            tracker.detect.side_effect=[near,near,far,near,near]
            detections=[(True,'test',[])]*5+[(False,'test',[])]*2
            argv=['phonewatch','--detector','smart-watch','--delay','0','--watch-delay','0','--roi','70','40','100','100','--data-dir',tmp]
            with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',side_effect=[-1]*7+[ord('q')]), patch('phonewatch.smart.SmartDetector.detect',side_effect=detections), patch('phonewatch.hand_tracking.HandTracker',return_value=tracker), patch('phonewatch.__main__.time.monotonic',side_effect=itertools.count(0,1)):
                main()
            tracker.close.assert_called_once()
            camera.release.assert_called_once()
            photos=list((Path(tmp)/'photos').glob('suspect_*.jpg'))
            self.assertEqual(len(photos),3)
            self.assertIsNotNone(cv2.imread(str(photos[0])))
            db=sqlite3.connect(Path(tmp)/'usage.sqlite3')
            events=list(db.execute('SELECT active,photo_path FROM watch_events'))
            self.assertEqual([active for active,_ in events],[1,0,1,0,1,0])
            self.assertEqual(sum(path is not None for _,path in events),3)
            self.assertGreater(db.execute('SELECT SUM(seconds) FROM watch_daily').fetchone()[0],0)
            states=dict(db.execute('SELECT state,seconds FROM daily_state'))
            self.assertGreater(states['absent'],0)
            self.assertGreater(states['unknown'],0)
            db.close()

    def test_watch_requires_selected_parking_area(self):
        frame=np.full((180,240,3),160,np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            camera=MagicMock()
            camera.isOpened.return_value=True
            camera.read.return_value=(True,frame)
            tracker=MagicMock()
            argv=['phonewatch','--detector','smart-watch','--delay','0','--watch-delay','0','--data-dir',tmp]
            with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',side_effect=[-1,ord('q')]), patch('phonewatch.smart.SmartDetector.detect',return_value=(True,'test',[])), patch('phonewatch.hand_tracking.HandTracker',return_value=tracker):
                main()
            tracker.detect.assert_not_called()
            self.assertFalse(list((Path(tmp)/'photos').glob('suspect_*.jpg')))
            db=sqlite3.connect(Path(tmp)/'usage.sqlite3')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM watch_events').fetchone()[0],0)
            db.close()

    def test_normal_smart_does_not_initialize_hand_tracking(self):
        frame=np.full((180,240,3),160,np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            camera=MagicMock()
            camera.isOpened.return_value=True
            camera.read.return_value=(True,frame)
            argv=['phonewatch','--detector','smart','--delay','0','--data-dir',tmp]
            with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',return_value=ord('q')), patch('phonewatch.smart.SmartDetector.detect',return_value=(True,'test',[])), patch('phonewatch.hand_tracking.HandTracker',side_effect=AssertionError('optional tracker must not initialize')) as tracker:
                main()
            tracker.assert_not_called()

    def test_exiting_active_watch_closes_event_without_offline_charge(self):
        frame=np.full((180,240,3),160,np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            camera=MagicMock()
            camera.isOpened.return_value=True
            camera.read.return_value=(True,frame)
            tracker=MagicMock()
            tracker.detect.return_value=[[(.5,.5)]*21]
            argv=['phonewatch','--detector','smart-watch','--delay','0','--watch-delay','0','--roi','70','40','100','100','--data-dir',tmp]
            with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',return_value=ord('q')), patch('phonewatch.smart.SmartDetector.detect',return_value=(True,'test',[])), patch('phonewatch.hand_tracking.HandTracker',return_value=tracker):
                main()
            db=sqlite3.connect(Path(tmp)/'usage.sqlite3')
            self.assertEqual([row[0] for row in db.execute('SELECT active FROM watch_events')],[1,0])
            self.assertIsNone(db.execute('SELECT SUM(seconds) FROM watch_daily').fetchone()[0])
            db.close()
