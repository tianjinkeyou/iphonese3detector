import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path

from .storage import Store
from .timer import UsageTimer


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
    parser.add_argument("--photo-interval", type=float, default=120.0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--ignore-plane", action="store_true", help="ignore saved perspective calibration for this run")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.delay, args.watch_delay, args.photo_interval, args.confidence)) or args.delay < 0 or args.watch_delay < 0 or args.photo_interval <= 0 or not 0 < args.confidence <= 1:
        parser.error("delay/watch-delay must be >=0, photo-interval >0, confidence in (0,1]")
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

    camera = int(args.camera) if args.camera.isdecimal() else args.camera
    capture = cv2.VideoCapture(camera)
    store = None
    hand_tracker = None
    watch_timer = None
    screen_detector = None
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open camera. Check --camera and Ubuntu camera permissions.")
        references = ReferenceDetector(args.data_dir / "references")
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
        photos = args.data_dir / "photos"
        photos.mkdir(exist_ok=True)
        timer = UsageTimer(args.delay)
        if args.detector == "smart-watch":
            from .hand_tracking import HandTracker
            from .watch import WatchTimer, hand_near_polygon
            from .partial_hand import partial_hand_regions
            hand_tracker = HandTracker(args.hand_mode)
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
        while True:
            ok, frame = capture.read()
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
                started = time.monotonic()
                polygons = []
                if args.detector in ("auto", "smart", "smart-watch"):
                    present, method, polygons = smart.detect(region)
                    if present is None and screen_detector is not None and screen_polygon is not None:
                        import numpy as np
                        screen_in_region = (cv2.perspectiveTransform(np.float32([screen_polygon]),np.linalg.inv(inverse))[0]
                                            if inverse is not None else screen_polygon-(x,y))
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
            # Timestamp after inference so slow inference is not hidden from accounting.
            now = time.monotonic()
            at = datetime.now()
            elapsed, changed = timer.update(present, now)
            store.record(at, elapsed, timer.state, changed, timer.interval_state)
            if watch_timer is not None:
                near = None
                phone_confirmed = ok and present is True and timer.state == "present"
                hand_count = 0
                candidate_count = 0
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
                if watch_started:
                    photo_path = photos / f"suspect_{watch_at:%Y-%m-%d_%H-%M-%S_%f}.jpg"
                    if not cv2.imwrite(str(photo_path),frame):
                        raise RuntimeError(f"Could not save suspected-use photo: {photo_path}")
                    photo = str(photo_path.relative_to(args.data_dir))
                    print(f"Suspected use started; photo saved: {photo}",flush=True)
                store.record_watch(watch_at,watch_elapsed,watch_timer.active,watch_changed,photo)
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
            if changed or now - last_log >= 10:
                print(f"{at:%Y-%m-%d %H:%M:%S} state={timer.state} absent={totals['absent'] / 60:.1f} min unknown={totals['unknown'] / 60:.1f} min legacy={totals['legacy'] / 60:.1f} min", flush=True)
                last_log = now
            if ok and frame is not None and now >= next_photo:
                path = photos / f"{at:%Y-%m-%d_%H-%M-%S_%f}.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not save photo: {path}")
                next_photo = now + args.photo_interval
            if last_frame is not None:
                display = last_frame.copy()
                color = STATE_COLORS[timer.state]
                cv2.putText(display, timer.state.upper(), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display, f"ABSENT today: {totals['absent']/60:.1f} min", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, STATE_COLORS["absent"], 2)
                cv2.putText(display, f"UNKNOWN / UNCERTAIN: {totals['unknown']/60:.1f} min", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, STATE_COLORS["unknown"], 2)
                if totals['legacy']:
                    cv2.putText(display, f"Legacy unsplit: {totals['legacy']/60:.1f} min", (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
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
            if watch_timer is not None and key in (ord("c"),ord("e"),ord("f"),ord("p"),ord("r"),ord("s")):
                watch_elapsed, _, watch_changed = watch_timer.update(None,False,time.monotonic())
                store.record_watch(datetime.now(),watch_elapsed,watch_timer.active,watch_changed)
            if screen_detector is not None and key in (ord("c"),ord("e"),ord("f"),ord("p"),ord("r"),ord("s")):
                screen_detector.reset()
                last_screen_activity = None
            if key == ord("s"):
                if screen_detector is None:
                    print("Screen monitoring is optional: restart smart-watch with --watch-screen.",flush=True)
                elif ok and frame is not None:
                    elapsed, changed = timer.update(None,time.monotonic())
                    store.record(datetime.now(),elapsed,"unknown",changed,timer.interval_state)
                    selected = select_plane(frame,window_title="Screen corners: top-left, top-right, bottom-right, bottom-left. Enter saves; Esc cancels")
                    if selected is not None:
                        save_plane(screen_path,selected,frame.shape)
                        screen_polygon = selected
                        print("Saved actual screen area. Recalibrate S if camera or phone angle moves.",flush=True)
            if key == ord("c"):
                pressed_at = time.monotonic()
                if pressed_at < clear_deadline:
                    references.clear()
                    clear_deadline = 0.0
                    elapsed, changed = timer.update(None, pressed_at)
                    store.record(datetime.now(),elapsed,"unknown",changed,timer.interval_state)
                    print("All E/F reference samples cleared. Recollect E=empty and F=phone.",flush=True)
                else:
                    clear_deadline = pressed_at + 3
                    print("Press C again within 3 seconds to clear all E/F samples.",flush=True)
            if key in (ord("e"), ord("f")) and ok and frame is not None:
                references.save("empty" if key == ord("e") else "phone", region)
                elapsed, changed = timer.update(None, time.monotonic())
                store.record(datetime.now(), elapsed, "unknown", changed, timer.interval_state)
                clear_deadline = 0.0
                counts = references.counts
                print(f"Reference added: E={counts['empty']} F={counts['phone']}. E=empty, F=phone.", flush=True)
            if key == ord("p") and ok and frame is not None:
                elapsed, changed = timer.update(None, time.monotonic())
                store.record(datetime.now(), elapsed, "unknown", changed, timer.interval_state)
                selected = select_plane(frame)
                if selected is not None:
                    save_plane(plane_path, selected, frame.shape)
                    plane = selected
                    roi = None
                    roi_size = None
                    roi_path.unlink(missing_ok=True)
                    references.clear()
                    print("Saved tilted phone plane. Recalibrate if camera or stand moves.", flush=True)
            if key == ord("r") and ok and frame is not None:
                # Selection is unobserved time: count it as unknown until observations stabilize.
                elapsed, changed = timer.update(None, time.monotonic())
                store.record(datetime.now(), elapsed, "unknown", changed, timer.interval_state)
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
            if not ok:
                time.sleep(0.2)
    finally:
        if watch_timer is not None and watch_timer.active and store is not None:
            watch_timer.reset(time.monotonic())
            store.record_watch(datetime.now(),0.0,False,True)
        capture.release()
        cv2.destroyAllWindows()
        if hand_tracker is not None:
            hand_tracker.close()
        if store is not None:
            store.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
