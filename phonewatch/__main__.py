import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path

from .storage import Store
from .timer import UsageTimer


def parse_args():
    parser = argparse.ArgumentParser(description="Local phone parking monitor. Esc/Q exits; R selects area; P calibrates tilted phone plane.")
    parser.add_argument("--camera", default="0", help="camera index or /dev/video path")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path (first run downloads weights)")
    parser.add_argument("--detector", choices=("auto", "home", "reference", "yolo"), default="auto", help="auto prefers H button features, then E/F references; home/reference avoid model downloads")
    parser.add_argument("--detection-mode", choices=("robust", "fast"), default="robust", help="robust: rotation/low-light fallback; fast: single view")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--photo-interval", type=float, default=120.0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--ignore-plane", action="store_true", help="ignore saved perspective calibration for this run")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.delay, args.photo_interval, args.confidence)) or args.delay < 0 or args.photo_interval <= 0 or not 0 < args.confidence <= 1:
        parser.error("delay must be >=0, photo-interval >0, confidence in (0,1]")
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
    from .home import HomeDetector
    from .reference import ReferenceDetector
    from .detection import PhoneDetector
    from .perspective import load_plane, save_plane, select_plane, rectify, map_polygon

    camera = int(args.camera) if args.camera.isdecimal() else args.camera
    capture = cv2.VideoCapture(camera)
    store = None
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open camera. Check --camera and Ubuntu camera permissions.")
        references = ReferenceDetector(args.data_dir / "references")
        home = HomeDetector(args.data_dir / "home")
        detector = None
        store = Store(args.data_dir)
        photos = args.data_dir / "photos"
        photos.mkdir(exist_ok=True)
        timer = UsageTimer(args.delay)
        roi = tuple(args.roi) if args.roi else None
        next_photo = time.monotonic() + args.photo_interval
        plane_path = args.data_dir / "parking-plane.json"
        plane = None
        plane_checked = False
        window = "Phone parking monitor - Q: quit / R: area / P: plane / E: empty / F: phone / H: Home button"
        last_frame = None
        last_log = 0.0
        while True:
            ok, frame = capture.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
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
                started = time.monotonic()
                polygons = []
                if args.detector == "home" or (args.detector == "auto" and home.ready):
                    present, method = home.detect(frame)
                elif args.detector == "reference" or (args.detector == "auto" and references.ready):
                    present, method = references.detect(region)
                else:
                    if detector is None:
                        from ultralytics import YOLO
                        model = YOLO(args.model)
                        phone_classes = [i for i, label in model.names.items() if label == "cell phone"]
                        if not phone_classes:
                            raise RuntimeError("Model has no cell phone class")
                        detector = PhoneDetector(model, phone_classes, args.confidence, args.detection_mode)
                    polygons, method = detector.detect(region)
                    present = bool(polygons)
                inference_ms = (time.monotonic() - started) * 1000
                display = frame.copy()
                for polygon in polygons:
                    mapped = map_polygon(polygon, inverse) if inverse is not None else polygon + (x, y)
                    cv2.polylines(display, [mapped], True, (0, 255, 0), 2)
                cv2.putText(display, f"{method} | {inference_ms:.0f} ms", (12, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if home.ready and args.detector in ("auto", "home"):
                    hx,hy,hw,hh = home.box
                    cv2.rectangle(display,(hx,hy),(hx+hw,hy+hh),(255,0,255),2)
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
            store.record(at, elapsed, timer.state, changed)
            total = store.total(at.date().isoformat())
            if changed or now - last_log >= 10:
                print(f"{at:%Y-%m-%d %H:%M:%S} state={timer.state} estimated_usage={total / 60:.1f} min", flush=True)
                last_log = now
            if ok and frame is not None and now >= next_photo:
                path = photos / f"{at:%Y-%m-%d_%H-%M-%S_%f}.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not save photo: {path}")
                next_photo = now + args.photo_interval
            if last_frame is not None:
                display = last_frame.copy()
                color = (0, 0, 255) if timer.state == "unknown" else (0, 255, 0)
                cv2.putText(display, f"{timer.state.upper()} | usage today: {total/60:.1f} min", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if present is None:
                    cv2.putText(display, "UNCERTAIN - timer paused" if ok else "CAMERA ERROR - timer paused", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("e"), ord("f")) and ok and frame is not None:
                references.save("empty" if key == ord("e") else "phone", region)
                if key == ord("e"):
                    home.save_empty(frame)
                timer.update(None, time.monotonic())
                store.record(datetime.now(), 0, "unknown", True)
                print("Reference saved. Keep camera/area fixed. E=empty, F=phone.", flush=True)
            if key == ord("h") and ok and frame is not None:
                timer.update(None, time.monotonic())
                store.record(datetime.now(),0,"unknown",True)
                selected = cv2.selectROI(window,frame,showCrosshair=True,fromCenter=False)
                if selected[2] and selected[3]:
                    try:
                        home.save_button(frame,selected)
                        print("Home feature saved. Remove phone, leave view, press E to save empty stand.",flush=True)
                    except ValueError as exc:
                        print(f"Home calibration rejected: {exc}",flush=True)
            if key == ord("p") and ok and frame is not None:
                timer.update(None, time.monotonic())
                store.record(datetime.now(), 0, "unknown", True)
                selected = select_plane(frame)
                if selected is not None:
                    save_plane(plane_path, selected, frame.shape)
                    plane = selected
                    references.clear()
                    print("Saved tilted phone plane. Recalibrate if camera or stand moves.", flush=True)
            if key == ord("r") and ok and frame is not None:
                # Selection is unobserved time: stop counting until fresh observations stabilize.
                timer.update(None, time.monotonic())
                store.record(datetime.now(), 0, "unknown", True)
                selected = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
                if selected[2] and selected[3]:
                    roi = selected
                    plane = None
                    references.clear()
                    plane_path.unlink(missing_ok=True)
                    print(f"Reuse this area with --roi {' '.join(map(str, roi))}", flush=True)
            if not ok:
                time.sleep(0.2)
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if store is not None:
            store.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
