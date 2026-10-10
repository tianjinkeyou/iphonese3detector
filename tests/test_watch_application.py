import itertools
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from phonewatch.__main__ import main
from phonewatch.perspective import save_plane,load_plane
from phonewatch.reference import ReferenceDetector


class WatchApplicationTests(unittest.TestCase):
    @staticmethod
    def screen_frames():
        empty=np.full((180,240,3),160,np.uint8)
        phone=empty.copy()
        cv2.rectangle(phone,(50,40),(200,140),(20,20,20),-1)
        screen=np.float32([[60,50],[190,50],[190,130],[60,130]])
        frames=[]
        for value in (45,45,100,45,100,45):
            frame=phone.copy()
            cv2.rectangle(frame,(60,50),(190,130),(value,value,value),-1)
            frames.append(frame)
        return empty,frames,screen

    def test_screen_alone_recovers_body_presence_calibrates_and_counts(self):
        empty,frames,screen=self.screen_frames()
        with tempfile.TemporaryDirectory() as tmp:
            references=ReferenceDetector(Path(tmp)/'references')
            references.save('empty',empty[30:150,40:210])
            references.save('phone',frames[0][30:150,40:210])
            camera=MagicMock()
            camera.isOpened.return_value=True
            camera.read.side_effect=[(True,frame) for frame in frames]
            tracker=MagicMock()
            tracker.detect.return_value=[]
            argv=['phonewatch','--detector','smart-watch','--watch-screen','--delay','0','--watch-delay','0','--roi','40','30','170','120','--data-dir',tmp]
            with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',side_effect=[ord('s')]+[-1]*4+[ord('q')]), patch('phonewatch.smart.SmartDetector.detect',return_value=(None,'screen content changed',[])), patch('phonewatch.hand_tracking.HandTracker',return_value=tracker), patch('phonewatch.perspective.select_plane',return_value=screen), patch('phonewatch.__main__.time.monotonic',side_effect=itertools.count(0,1)):
                main()
            np.testing.assert_allclose(load_plane(Path(tmp)/'screen-plane.json',frames[0].shape),screen)
            self.assertEqual(tracker.detect.call_count,len(frames))
            self.assertEqual(len(list((Path(tmp)/'photos').glob('suspect_*.jpg'))),1)
            db=sqlite3.connect(Path(tmp)/'usage.sqlite3')
            self.assertEqual([row[0] for row in db.execute('SELECT active FROM watch_events')],[1,0])
            self.assertGreater(db.execute('SELECT SUM(seconds) FROM watch_daily').fetchone()[0],0)
            self.assertGreater(db.execute("SELECT seconds FROM daily_state WHERE state='unknown'").fetchone()[0],0)
            db.close()

    def test_screen_calibration_reloads_and_motion_is_optional(self):
        _,frames,screen=self.screen_frames()
        for enabled in (False,True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                save_plane(Path(tmp)/'screen-plane.json',screen,frames[0].shape)
                camera=MagicMock()
                camera.isOpened.return_value=True
                camera.read.side_effect=[(True,frames[0]),(True,frames[2])]
                tracker=MagicMock()
                tracker.detect.return_value=[]
                argv=['phonewatch','--detector','smart-watch','--delay','0','--watch-delay','0','--roi','40','30','170','120','--data-dir',tmp]+(['--watch-screen'] if enabled else [])
                with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',side_effect=[-1,ord('q')]), patch('phonewatch.smart.SmartDetector.detect',return_value=(True,'phone present',[])), patch('phonewatch.hand_tracking.HandTracker',return_value=tracker):
                    main()
                self.assertEqual(len(list((Path(tmp)/'photos').glob('suspect_*.jpg'))),int(enabled))

    def test_unknown_or_absent_phone_still_shows_evidence_without_suspect_time(self):
        _,frames,screen=self.screen_frames()
        for phone_state in (None,False):
            with self.subTest(phone_state=phone_state), tempfile.TemporaryDirectory() as tmp:
                save_plane(Path(tmp)/'screen-plane.json',screen,frames[0].shape)
                camera=MagicMock()
                camera.isOpened.return_value=True
                camera.read.side_effect=[(True,frames[0]),(True,frames[2])]
                tracker=MagicMock()
                tracker.detect.return_value=[[(.5,.5)]*21]
                argv=['phonewatch','--detector','smart-watch','--watch-screen','--delay','0','--watch-delay','0','--roi','40','30','170','120','--data-dir',tmp]
                with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',side_effect=[-1,ord('q')]), patch('cv2.putText',wraps=cv2.putText) as text, patch('phonewatch.smart.SmartDetector.detect',return_value=(phone_state,'unconfirmed',[])), patch('phonewatch.hand_tracking.HandTracker',return_value=tracker):
                    main()
                self.assertEqual(tracker.detect.call_count,2)
                labels=[call.args[1] for call in text.call_args_list]
                self.assertTrue(any('Hands: 1' in label for label in labels))
                self.assertTrue(any('SCREEN: changing' in label for label in labels))
                self.assertFalse(list((Path(tmp)/'photos').glob('suspect_*.jpg')))
                db=sqlite3.connect(Path(tmp)/'usage.sqlite3')
                self.assertEqual(db.execute('SELECT COUNT(*) FROM watch_events').fetchone()[0],0)
                db.close()

    def test_repeated_screen_activity_bridges_pauses_but_single_change_does_not(self):
        _,frames,screen=self.screen_frames()
        for repeated in (False,True):
            with self.subTest(repeated=repeated), tempfile.TemporaryDirectory() as tmp:
                save_plane(Path(tmp)/'screen-plane.json',screen,frames[0].shape)
                values=(45,100,100,45,45,100,100,45) if repeated else (45,100,100,100,100,100,100,100)
                clock=[0.0]
                reads=iter(enumerate(values))
                def read_frame():
                    index,value=next(reads)
                    clock[0]=index*.6
                    frame=frames[0].copy()
                    cv2.rectangle(frame,(60,50),(190,130),(value,value,value),-1)
                    return True,frame
                camera=MagicMock()
                camera.isOpened.return_value=True
                camera.read.side_effect=read_frame
                tracker=MagicMock()
                tracker.detect.return_value=[]
                argv=['phonewatch','--detector','smart-watch','--watch-screen','--delay','0','--watch-delay','3','--roi','40','30','170','120','--data-dir',tmp]
                with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',side_effect=[-1]*7+[ord('q')]), patch('phonewatch.smart.SmartDetector.detect',return_value=(True,'phone present',[])), patch('phonewatch.hand_tracking.HandTracker',return_value=tracker), patch('phonewatch.__main__.time.monotonic',side_effect=lambda:clock[0]):
                    main()
                self.assertEqual(len(list((Path(tmp)/'photos').glob('suspect_*.jpg'))),int(repeated))

    def test_screen_outside_selected_phone_area_cannot_trigger(self):
        _,frames,screen=self.screen_frames()
        with tempfile.TemporaryDirectory() as tmp:
            save_plane(Path(tmp)/'screen-plane.json',screen,frames[0].shape)
            camera=MagicMock()
            camera.isOpened.return_value=True
            camera.read.side_effect=[(True,frames[0]),(True,frames[2])]
            tracker=MagicMock()
            tracker.detect.return_value=[]
            argv=['phonewatch','--detector','smart-watch','--watch-screen','--delay','0','--watch-delay','0','--roi','0','0','50','100','--data-dir',tmp]
            with patch('sys.argv',argv), patch('cv2.VideoCapture',return_value=camera), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',side_effect=[-1,ord('q')]), patch('cv2.putText',wraps=cv2.putText) as text, patch('phonewatch.smart.SmartDetector.detect',return_value=(True,'phone present',[])), patch('phonewatch.hand_tracking.HandTracker',return_value=tracker):
                main()
            self.assertTrue(any('SCREEN: select screen inside' in call.args[1] for call in text.call_args_list))
            self.assertFalse(list((Path(tmp)/'photos').glob('suspect_*.jpg')))

    def test_watch_events_photo_and_camera_interruption_are_separate(self):
        frame=np.full((180,240,3),160,np.uint8)
        near=[[(.5,.5)]*21]
        far=[[(.95,.95)]*21]
        with tempfile.TemporaryDirectory() as tmp:
            camera=MagicMock()
            camera.isOpened.return_value=True
            camera.read.side_effect=[(True,frame)]*4+[(False,None)]+[(True,frame)]*3
            tracker=MagicMock()
            tracker.detect.side_effect=[near,near,far,near,near,far,far]
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
            self.assertEqual(tracker.detect.call_count,2)
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
