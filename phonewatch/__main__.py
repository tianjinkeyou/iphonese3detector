import argparse
import math
import os
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from .storage import Store
from .timer import UsageTimer
from .pickup import PickupTracker
from .event_log import EventJournal


STATE_COLORS = {"present": (0, 255, 0), "absent": (0, 0, 255), "unknown": (0, 255, 255)}


def parse_args():
    parser = argparse.ArgumentParser(description="Local phone parking monitor. Esc/Q exits; R selects area; P calibrates tilted phone plane.")
    parser.add_argument("--camera", default="0", help="camera index or /dev/video path")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path (first run downloads weights)")
    parser.add_argument("--detector", choices=("auto", "smart", "smart-watch", "reference", "yolo"), default="auto", help="smart: phone detection; smart-watch: optional hand proximity, suspect time/photos; reference: R/E/F only")
    parser.add_argument("--detection-mode", choices=("robust", "fast"), default="robust", help="robust: rotation/low-light fallback; fast: single view")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--watch-screen", action="store_true", help="optional screen-motion signal for smart-watch; S calibrates screen corners")
    parser.add_argument("--hand-mode", choices=("robust", "landmarks"), default="robust", help="robust: crop/rotation and partial hand candidates; landmarks: full-frame only")
    parser.add_argument("--screen-threshold", type=float, default=0.025, help="minimum changed screen fraction, e.g. 0.025 = 2.5%%")
    parser.add_argument("--watch-delay", type=float, default=3.0, help="continuous hand proximity or screen activity seconds before suspected use")
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--pickup-delay", type=float, default=1.0, help="normal empty/return confirmation seconds; strict parking-empty frames can count immediately; independent of --delay")
    parser.add_argument("--pickup-mode", choices=("parking", "presence"), default="parking", help="parking: use E/F parked-body evidence; presence: legacy region-presence transitions")
    parser.add_argument("--photo-interval", type=float, default=120.0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--ignore-plane", action="store_true", help="ignore saved perspective calibration for this run")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.delay, args.pickup_delay, args.watch_delay, args.photo_interval, args.confidence)) or min(args.delay,args.pickup_delay,args.watch_delay) < 0 or args.photo_interval <= 0 or not 0 < args.confidence <= 1:
        parser.error("delay/pickup-delay/watch-delay must be >=0, photo-interval >0, confidence in (0,1]")
    if args.watch_screen and args.detector != "smart-watch":
        parser.error("--watch-screen requires --detector smart-watch")
    if not math.isfinite(args.screen_threshold) or not 0 < args.screen_threshold <= 1:
        parser.error("screen-threshold must be in (0,1]")
    if args.roi and (min(args.roi[:2]) < 0 or min(args.roi[2:]) <= 0):
        parser.error("ROI needs nonnegative X/Y and positive width/height")
    return args


def main():
    args = parse_args()
    # Private permissions for newly created local records/photos.
    os.umask(0o077)
    config_dir = args.data_dir / "yolo"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir.resolve()))
    import cv2
    from .smart import SmartDetector
    from .reference import ReferenceDetector
    from .detection import PhoneDetector
    from .perspective import load_plane, save_plane, select_plane, rectify, map_polygon
    from .parking import ParkingDetector

    camera = int(args.camera) if args.camera.isdecimal() else args.camera
    capture = cv2.VideoCapture(camera)
    resources = ExitStack()
    resources.callback(capture.release)
    resources.callback(cv2.destroyAllWindows)
    store = None
    journal = None
    hand_tracker = None
    watch_timer = None
    screen_detector = None
    session_started = None
    camera_available = None
    method = "initializing"
    exit_reason = "normal"
    export_warning = None
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open camera. Check --camera and Ubuntu camera permissions.")
        references = ReferenceDetector(args.data_dir / "references")
        parking_detector = ParkingDetector(references)
        detector = None
        def make_detector():
            from ultralytics import YOLO
            model = YOLO(args.model)
            classes=[i for i,label in model.names.items() if label == "cell phone"]
            if not classes:
                raise RuntimeError("Model has no cell phone class")
            return PhoneDetector(model,classes,args.confidence,args.detection_mode)
        smart = SmartDetector(references,make_detector)
        store = Store(args.data_dir)
        resources.callback(store.close)
        journal = EventJournal(args.data_dir)
        resources.callback(journal.close)
        photos = args.data_dir / "photos"
        photos.mkdir(exist_ok=True)
        timer = UsageTimer(args.delay)
        pickups = PickupTracker(args.pickup_delay)

        def record_event(kind, at=None, **details):
            nonlocal export_warning
            try:
                event_id = journal.record(at or datetime.now(),kind,details)
            except OSError as exc:
                # The event is already committed. Do not retry the insert or
                # inflate pickup counts when the shareable export is unavailable.
                if str(exc) != export_warning:
                    print(f"TXT export unavailable; events remain in SQLite: {exc}",flush=True)
                    export_warning = str(exc)
                return None
            if export_warning is not None:
                print("TXT export recovered from SQLite.",flush=True)
                export_warning = None
            return event_id

        def update_usage(observation, now, at=None, reason="detection"):
            at = at or datetime.now()
            before = timer.state
            elapsed, changed = timer.update(observation,now)
            store.record(at,elapsed,timer.state,changed,timer.interval_state)
            if changed:
                record_event("state_change",at,from_state=before,to_state=timer.state,
                             observation="unknown" if observation is None else ("present" if observation else "absent"),
                             method=method,reason=reason)
            return elapsed,changed

        def summary(at, now, reason):
            day = at.date().isoformat()
            totals = store.totals(day)
            record_event("summary",at,reason=reason,state=timer.state,pickup_state=pickups.state,
                         pickups=journal.pickup_total(day),absent_seconds=totals["absent"],
                         unknown_seconds=totals["unknown"],legacy_seconds=totals["legacy"],
                         suspected_seconds=store.watch_total(day),camera_available=camera_available,
                         session_runtime_seconds=max(0.0,now-session_started),references=references.counts)

        def save_event_photo(kind, at, image=None):
            path = photos / f"{kind}_{at:%Y-%m-%d_%H-%M-%S_%f}.jpg"
            if not cv2.imwrite(str(path),frame if image is None else image):
                raise RuntimeError(f"Could not save event photo: {path}")
            return str(path.relative_to(args.data_dir))
        if args.detector == "smart-watch":
            from .hand_tracking import HandTracker
            from .watch import WatchTimer, hand_near_polygon
            from .partial_hand import partial_hand_regions
            hand_tracker = HandTracker(args.hand_mode)
            resources.callback(hand_tracker.close)
            watch_timer = WatchTimer(args.watch_delay)
            if args.watch_screen:
                from .screen_activity import ScreenActivityDetector
                from .screen_presence import screen_body_presence
                screen_detector = ScreenActivityDetector(threshold=args.screen_threshold)
        watch_label = "WATCH: select parking area with R or P"
        watch_total = 0.0
        screen_polygon = None
        screen_checked = False
        screen_path = args.data_dir / "screen-plane.json"
        screen_label = "SCREEN: press S to select actual screen corners"
        last_screen_activity = None
        roi_path=args.data_dir / "parking-area.json"
        roi = tuple(args.roi) if args.roi else None
        if roi is None and roi_path.exists():
            import json
            saved=json.loads(roi_path.read_text())
            roi=tuple(saved["roi"])
        roi_size=None if args.roi or not roi_path.exists() else saved["size"]
        next_photo = time.monotonic() + args.photo_interval
        plane_path = args.data_dir / "parking-plane.json"
        plane = None
        plane_checked = False
        window = "Phone parking monitor - Q: quit / R: area / P: plane / E: empty / F: phone / C twice: reset refs / S: screen"
        last_frame = None
        last_log = 0.0
        clear_deadline = 0.0
        last_present_frame = None
        last_present_at = None
        pickup_cycle = 0
        pickup_cycle_id = None
        session_started = time.monotonic()
        next_summary = session_started+300
        summary_day = datetime.now().date().isoformat()
        record_event("session_start",camera=args.camera,detector=args.detector,
                     detection_mode=args.detection_mode,delay=args.delay,pickup_delay=args.pickup_delay,
                     pickup_mode=args.pickup_mode,
                     pickup_unknown_gap=pickups.unknown_gap,watch_delay=args.watch_delay,
                     watch_screen=args.watch_screen,screen_threshold=args.screen_threshold,
                     hand_mode=args.hand_mode,photo_interval=args.photo_interval,
                     roi=list(roi) if roi else None,ignore_plane=args.ignore_plane,
                     references=references.counts)
        while True:
            ok, frame = capture.read()
            pickup_observation = None
            pickup_source = "camera_unavailable"
            pickup_method = "camera unavailable"
            pickup_evidence = {}
            screen_in_region = None
            if ok and frame is not None:
                height, width = frame.shape[:2]
                if roi_size is not None and roi_size != [height,width]:
                    raise RuntimeError("Camera size changed. Use --roi to override saved area and recalibrate R/E/F.")
                x, y, w, h = roi or (0, 0, width, height)
                if x + w > width or y + h > height:
                    raise RuntimeError("ROI exceeds camera image. Select a smaller area.")
                if not plane_checked:
                    plane = None if args.ignore_plane or args.roi else load_plane(plane_path, frame.shape)
                    plane_checked = True
                region = frame[y:y+h, x:x+w]
                inverse = None
                if plane is not None:
                    region, inverse = rectify(frame, plane)
                if screen_detector is not None and not screen_checked:
                    try:
                        screen_polygon = load_plane(screen_path,frame.shape)
                    except (ValueError, KeyError, TypeError) as exc:
                        print(f"Screen calibration needs S again: {exc}",flush=True)
                    screen_checked = True
                if screen_polygon is not None:
                    import numpy as np
                    screen_in_region = (cv2.perspectiveTransform(np.float32([screen_polygon]),np.linalg.inv(inverse))[0]
                                        if inverse is not None else screen_polygon-(x,y))
                started = time.monotonic()
                polygons = []
                if args.detector in ("auto", "smart", "smart-watch"):
                    present, method, polygons = smart.detect(region)
                    if present is None and screen_detector is not None and screen_polygon is not None:
                        body_present, body_label = screen_body_presence(
                            region,screen_in_region,references.compatible("empty",region.shape),references.compatible("phone",region.shape))
                        if body_present is not None:
                            present, method = body_present, body_label
                elif args.detector == "reference":
                    present, method = references.detect(region)
                else:
                    if detector is None:
                        detector = make_detector()
                    polygons, method = detector.detect(region)
                    present = bool(polygons)
                if args.pickup_mode == "parking" and parking_detector.ready(region.shape,screen_polygon=screen_in_region):
                    pickup_observation,pickup_method,pickup_evidence = parking_detector.detect(
                        region,phone_polygons=polygons,screen_polygon=screen_in_region)
                    pickup_source = "parking"
                else:
                    pickup_observation = present
                    pickup_source = "phone_presence"
                    pickup_method = method if args.pickup_mode=="presence" else "E/F parking unavailable; presence fallback"
                inference_ms = (time.monotonic() - started) * 1000
                display = frame.copy()
                for polygon in polygons:
                    mapped = map_polygon(polygon, inverse) if inverse is not None else polygon + (x, y)
                    cv2.polylines(display, [mapped], True, (0, 255, 0), 2)
                counts = references.counts
                sample_label = f"E samples: {counts['empty']} | F samples: {counts['phone']}"
                if time.monotonic() < clear_deadline:
                    sample_label = "Press C again to clear all E/F samples"
                cv2.putText(display,sample_label,(12,height-42),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
                cv2.putText(display, f"{method} | {inference_ms:.0f} ms", (12, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if plane is not None:
                    cv2.polylines(display, [plane.astype("int32")], True, (255, 180, 0), 2)
                else:
                    cv2.rectangle(display, (x, y), (x+w-1, y+h-1), (255, 180, 0), 2)
                last_frame = display
            else:
                present = None
                method = "camera unavailable"
            # Timestamp after inference so slow inference is not hidden from accounting.
            now = time.monotonic()
            at = datetime.now()
            available = ok and frame is not None
            if available != camera_available:
                camera_event = ("camera_ready" if camera_available is None else "camera_restored") if available else "camera_lost"
                record_event(camera_event,at,reason="valid frame" if available else "read failed",
                             frame_size=list(frame.shape[:2]) if available else None)
                if not available or camera_available is False:
                    pickups.reset(now)
                    last_present_frame = None
                    last_present_at = None
                    pickup_cycle_id = None
                camera_available = available
            elapsed, changed = update_usage(present,now,at)
            if available:
                resolve_gap = pickup_source=="parking" and pickup_observation is False and pickup_evidence.get("reliable_empty") is True
                immediate_empty = resolve_gap and pickup_evidence.get("immediate_empty") is True
                for event in pickups.update(pickup_observation,now,resolve_gap=resolve_gap,immediate_empty=immediate_empty):
                    details = dict(event)
                    kind = details.pop("kind")
                    details.update(method=pickup_method,state=timer.state,pickup_source=pickup_source,
                                   phone_method=method,phone_observation="unknown" if present is None else "present" if present else "absent",
                                   parking_evidence=pickup_evidence)
                    if kind == "pickup":
                        pickup_cycle += 1
                        pickup_cycle_id = f"{journal.session_id}:{pickup_cycle}"
                    if kind in ("pickup","return"):
                        details["cycle_id"] = pickup_cycle_id
                    if kind in ("pickup","return","unobserved_change"):
                        try:
                            details["photo_path"] = save_event_photo(kind,at)
                        except (RuntimeError,OSError,cv2.error) as exc:
                            details.update(photo_path=None,photo_error=str(exc))
                            print(f"{kind} observed; event photo unavailable: {exc}",flush=True)
                        if kind != "return" and details["from_state"]=="present" and last_present_frame is not None:
                            try:
                                details["before_photo_path"] = save_event_photo(f"before_{kind}",at,last_present_frame)
                                details["before_photo_time"] = last_present_at.astimezone().isoformat()
                            except (RuntimeError,OSError,cv2.error) as exc:
                                details.update(before_photo_path=None,before_photo_error=str(exc))
                    record_event(kind,at,**details)
                    if kind in ("return","unobserved_change","baseline"):
                        pickup_cycle_id = None
                if pickup_observation is True and pickups.state=="present":
                    last_present_frame = frame.copy()
                    last_present_at = at
            else:
                pickups.reset(now)
            if watch_timer is not None:
                near = None
                phone_confirmed = ok and present is True and timer.state == "present"
                hand_count = 0
                candidate_count = 0
                landmark_near = False
                hand_source = "camera unavailable"
                screen_moving = False
                screen_active = False
                screen_score = 0.0
                # Show hand detection independently of phone state. Only the
                # suspect timer requires confirmed presence and parking geometry.
                if ok and frame is not None:
                    import numpy as np
                    watch_polygon = None
                    if plane is not None:
                        watch_polygon = plane
                    elif roi is not None:
                        watch_polygon = np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
                    hands = hand_tracker.detect(frame,area=watch_polygon)
                    hand_count = len(hands)
                    hand_source = getattr(hand_tracker,"source","model")
                    if watch_polygon is not None:
                        near = hand_near_polygon(hands,watch_polygon,frame.shape)
                        landmark_near = near
                        if not near and args.hand_mode == "robust":
                            candidates = partial_hand_regions(region,references.compatible('phone',region.shape))
                            candidate_count = len(candidates)
                            if candidates:
                                near = True
                                for contour in candidates:
                                    polygon = contour.reshape(-1,2)
                                    mapped = map_polygon(polygon,inverse) if inverse is not None else polygon+(x,y)
                                    cv2.polylines(last_frame,[mapped],True,(0,165,255),2)
                    if screen_detector is not None:
                        if screen_polygon is not None:
                            cv2.polylines(last_frame,[screen_polygon.astype("int32")],True,(255,255,0),2)
                            if watch_polygon is not None and all(cv2.pointPolygonTest(watch_polygon,tuple(map(float,point)),False)>=0 for point in screen_polygon):
                                screen_moving,screen_score,screen_label = screen_detector.detect(frame,screen_polygon)
                                screen_label += f" | change {screen_score:.1%} / {args.screen_threshold:.1%}"
                                if screen_moving is True:
                                    last_screen_activity = time.monotonic()
                                    screen_active = True
                                elif screen_moving is None:
                                    last_screen_activity = None
                                elif last_screen_activity is not None:
                                    # Bridge short observed pauses, never camera loss or
                                    # invisible screens. One change cannot sustain the
                                    # full confirmation delay on its own.
                                    hold = min(1.5,args.watch_delay/2)
                                    screen_active = time.monotonic()-last_screen_activity < hold
                                    if screen_active:
                                        screen_label += " | recent motion"
                                if screen_active:
                                    near = True
                            else:
                                screen_detector.reset()
                                last_screen_activity = None
                                screen_label = "SCREEN: select screen inside R/P parking area"
                        else:
                            screen_label = "SCREEN: press S to select actual screen corners"
                    for hand in hands:
                        for index in (4,8,12,16,20):
                            px,py = hand[index]
                            cv2.circle(last_frame,(int(px*width),int(py*height)),4,(255,0,255),-1)
                if screen_detector is not None and (not ok or frame is None):
                    screen_detector.reset()
                    last_screen_activity = None
                    screen_label = "SCREEN: camera unavailable"
                watch_now = time.monotonic()
                watch_elapsed, watch_started, watch_changed = watch_timer.update(near,bool(phone_confirmed),watch_now)
                watch_at = datetime.now()
                photo = None
                photo_error = None
                if watch_started:
                    try:
                        photo = save_event_photo("suspect",watch_at)
                        print(f"Suspected use started; photo saved: {photo}",flush=True)
                    except (RuntimeError,OSError,cv2.error) as exc:
                        photo_error = str(exc)
                        print(f"Suspected use started; event photo unavailable: {exc}",flush=True)
                store.record_watch(watch_at,watch_elapsed,watch_timer.active,watch_changed,photo)
                if watch_changed:
                    sources = (["screen_motion"] if screen_active else []) + (["partial_candidate"] if candidate_count else []) + (["hand_landmarks"] if landmark_near else [])
                    record_event("suspect_start" if watch_timer.active else "suspect_end",watch_at,
                                 sources=sources,method=method,state=timer.state,
                                 reason="activity" if watch_timer.active else ("camera unavailable" if not available else "phone unconfirmed" if not phone_confirmed else "activity ended"),
                                 hands=hand_count,hand_source=str(hand_source),partial_candidates=candidate_count,
                                 screen_change_fraction=screen_score,photo_path=photo,photo_error=photo_error)
                watch_total = store.watch_total(watch_at.date().isoformat())
                if roi is None and plane is None:
                    watch_label = "WATCH: select parking area with R or P"
                elif not phone_confirmed:
                    watch_label = "WATCH: evidence shown; waiting for PRESENT"
                elif watch_timer.active:
                    watch_label = "WATCH: SUSPECTED USE - screen motion" if screen_active else ("WATCH: SUSPECTED USE - partial candidate" if candidate_count else "WATCH: SUSPECTED USE - hand near phone")
                elif near:
                    watch_label = "WATCH: screen motion, confirming..." if screen_active else ("WATCH: partial candidate, confirming..." if candidate_count else "WATCH: hand near phone, confirming...")
                else:
                    watch_label = "WATCH: no activity near phone"
                hand_label = f"Hands: {hand_count} ({hand_source}) | partial: {candidate_count}"
                if ok and frame is not None:
                    total_ms = (time.monotonic()-started)*1000
                    cv2.putText(last_frame,f"Frame processing: {total_ms:.0f} ms",(12,285),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
            totals = store.totals(at.date().isoformat())
            pickup_total = journal.pickup_total(at.date().isoformat())
            if now >= next_summary or at.date().isoformat() != summary_day:
                if at.date().isoformat() != summary_day:
                    try:
                        journal.export_day(summary_day)
                    except OSError as exc:
                        print(f"Previous-day TXT export unavailable; events remain in SQLite: {exc}",flush=True)
                summary(at,now,"periodic" if at.date().isoformat()==summary_day else "new_day")
                next_summary = now+300
                summary_day = at.date().isoformat()
            if changed or now - last_log >= 10:
                print(f"{at:%Y-%m-%d %H:%M:%S} state={timer.state} pickups={pickup_total} absent={totals['absent'] / 60:.1f} min unknown={totals['unknown'] / 60:.1f} min legacy={totals['legacy'] / 60:.1f} min", flush=True)
                last_log = now
            if ok and frame is not None and now >= next_photo:
                path = photos / f"{at:%Y-%m-%d_%H-%M-%S_%f}.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not save photo: {path}")
                record_event("photo_saved",at,reason="periodic",photo_path=str(path.relative_to(args.data_dir)),
                             state=timer.state,method=method)
                next_photo = now + args.photo_interval
            if last_frame is not None:
                display = last_frame.copy()
                color = STATE_COLORS[timer.state]
                cv2.putText(display, timer.state.upper(), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display, f"ABSENT today: {totals['absent']/60:.1f} min", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, STATE_COLORS["absent"], 2)
                cv2.putText(display, f"UNKNOWN / UNCERTAIN: {totals['unknown']/60:.1f} min", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, STATE_COLORS["unknown"], 2)
                cv2.putText(display,f"PICKUPS today (estimated): {pickup_total}",(12,120),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                parking_state = "unknown" if pickup_observation is None else "present" if pickup_observation else "absent"
                parking_word = "PARKED" if pickup_observation is True else "EMPTY" if pickup_observation is False else "UNCERTAIN"
                cv2.putText(display,f"Pickup evidence: {parking_word} ({pickup_source})",(12,335),cv2.FONT_HERSHEY_SIMPLEX,0.5,STATE_COLORS[parking_state],1)
                cv2.putText(display,pickup_method,(12,355),cv2.FONT_HERSHEY_SIMPLEX,0.45,STATE_COLORS[parking_state],1)
                if totals['legacy']:
                    cv2.putText(display, f"Legacy unsplit: {totals['legacy']/60:.1f} min", (12, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
                if watch_timer is not None:
                    cv2.putText(display,f"SUSPECTED today: {watch_total/60:.1f} min",(12,180),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,0,255),2)
                    if screen_detector is not None:
                        cv2.putText(display,screen_label,(12,260),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,0),2)
                    cv2.putText(display,hand_label,(12,235),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,0,255),2)
                    cv2.putText(display,watch_label,(12,210),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,0,255),2)
                if present is None:
                    cv2.putText(display, "UNCERTAIN - counting time" if ok else "CAMERA ERROR - counting time", (12, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, STATE_COLORS["unknown"], 2)
                cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("c"),ord("e"),ord("f"),ord("p"),ord("r")) or (key==ord("s") and screen_detector is not None):
                pickups.reset(time.monotonic())
                last_present_frame = None
                last_present_at = None
                pickup_cycle_id = None
                record_event("tracking_reset",reason="configuration",key=chr(key))
            if watch_timer is not None and key in (ord("c"),ord("e"),ord("f"),ord("p"),ord("r"),ord("s")):
                watch_elapsed, _, watch_changed = watch_timer.update(None,False,time.monotonic())
                store.record_watch(datetime.now(),watch_elapsed,watch_timer.active,watch_changed)
                if watch_changed:
                    record_event("suspect_end",reason="configuration",key=chr(key),state=timer.state)
            if screen_detector is not None and key in (ord("c"),ord("e"),ord("f"),ord("p"),ord("r"),ord("s")):
                screen_detector.reset()
                last_screen_activity = None
            if key == ord("s"):
                if screen_detector is None:
                    print("Screen monitoring is optional: restart smart-watch with --watch-screen.",flush=True)
                elif ok and frame is not None:
                    update_usage(None,time.monotonic(),reason="screen calibration")
                    record_event("calibration_start",key="s",target="screen")
                    selected = select_plane(frame,window_title="Screen corners: top-left, top-right, bottom-right, bottom-left. Enter saves; Esc cancels")
                    if selected is not None:
                        save_plane(screen_path,selected,frame.shape)
                        screen_polygon = selected
                        print("Saved actual screen area. Recalibrate S if camera or phone angle moves.",flush=True)
                    record_event("calibration_end",key="s",target="screen",saved=selected is not None,
                                 points=selected.tolist() if selected is not None else None)
            if key == ord("c"):
                pressed_at = time.monotonic()
                if pressed_at < clear_deadline:
                    references.clear()
                    clear_deadline = 0.0
                    update_usage(None,pressed_at,reason="references cleared")
                    record_event("references_cleared",references=references.counts)
                    print("All E/F reference samples cleared. Recollect E=empty and F=phone.",flush=True)
                else:
                    clear_deadline = pressed_at + 3
                    record_event("clear_requested",confirmation_seconds=3)
                    print("Press C again within 3 seconds to clear all E/F samples.",flush=True)
            if key in (ord("e"), ord("f")) and ok and frame is not None:
                references.save("empty" if key == ord("e") else "phone", region)
                update_usage(None,time.monotonic(),reason="reference capture")
                clear_deadline = 0.0
                counts = references.counts
                record_event("reference_saved",label="empty" if key==ord("e") else "phone",references=counts)
                print(f"Reference added: E={counts['empty']} F={counts['phone']}. E=empty, F=phone.", flush=True)
            if key == ord("p") and ok and frame is not None:
                update_usage(None,time.monotonic(),reason="phone plane calibration")
                record_event("calibration_start",key="p",target="parking_plane")
                selected = select_plane(frame)
                if selected is not None:
                    save_plane(plane_path, selected, frame.shape)
                    plane = selected
                    roi = None
                    roi_size = None
                    roi_path.unlink(missing_ok=True)
                    references.clear()
                    print("Saved tilted phone plane. Recalibrate if camera or stand moves.", flush=True)
                record_event("calibration_end",key="p",target="parking_plane",saved=selected is not None,
                             points=selected.tolist() if selected is not None else None,references=references.counts)
            if key == ord("r") and ok and frame is not None:
                # Selection is unobserved time: count it as unknown until observations stabilize.
                update_usage(None,time.monotonic(),reason="parking area calibration")
                record_event("calibration_start",key="r",target="parking_area")
                selected = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
                if selected[2] and selected[3]:
                    roi = selected
                    roi_size=[height,width]
                    import json
                    roi_path.write_text(json.dumps({"roi":list(roi),"size":roi_size}))
                    plane = None
                    references.clear()
                    plane_path.unlink(missing_ok=True)
                    print(f"Reuse this area with --roi {' '.join(map(str, roi))}", flush=True)
                record_event("calibration_end",key="r",target="parking_area",saved=bool(selected[2] and selected[3]),
                             roi=list(roi) if roi else None,references=references.counts)
            if not ok:
                time.sleep(0.2)
    except BaseException as exc:
        exit_reason = "interrupted" if isinstance(exc,KeyboardInterrupt) else "error"
        if journal is not None:
            try:
                record_event("error",reason=exit_reason,error_type=type(exc).__name__,message=str(exc))
            except Exception as log_error:
                print(f"Could not record error event: {log_error}",flush=True)
        raise
    finally:
        try:
            if watch_timer is not None and watch_timer.active and store is not None:
                watch_timer.reset(time.monotonic())
                store.record_watch(datetime.now(),0.0,False,True)
                if journal is not None:
                    record_event("suspect_end",reason="session ended",state=timer.state)
            if journal is not None and session_started is not None:
                ended_at = datetime.now()
                ended_now = time.monotonic()
                summary(ended_at,ended_now,"session_end")
                record_event("session_end",ended_at,reason=exit_reason,state=timer.state,
                             session_runtime_seconds=max(0.0,ended_now-session_started))
        except Exception as log_error:
            print(f"Could not record final events: {log_error}",flush=True)
        finally:
            resources.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
