"""
Baseline test: YOLOv8 vehicle detection on a single image or video.

Usage:
    python scripts/baseline_test.py --source <image_or_video_path>
    python scripts/baseline_test.py --source 0  # webcam
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from src.detection import VehicleDetector
from src.utils.io import load_config


def run_image(detector: VehicleDetector, source: str) -> None:
    import cv2
    from src.utils.io import load_image, save_image

    frame = load_image(source)
    detections = detector.detect(frame)
    output = detector.draw(frame, detections)

    print(f"Detected {len(detections)} vehicle(s):")
    for d in detections:
        print(f"  {d['class_name']:12s} conf={d['confidence']:.3f}  bbox={[round(x) for x in d['bbox']]}")

    out_path = f"outputs/visualizations/baseline_{Path(source).stem}.jpg"
    save_image(output, out_path)
    print(f"Saved: {out_path}")

    cv2.imshow("Baseline Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(detector: VehicleDetector, source: str | int) -> None:
    from src.utils.io import video_frames

    for frame in video_frames(source):
        detections = detector.detect(frame)
        output = detector.draw(frame, detections)
        cv2.imshow("Baseline Detection", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Image/video path or camera index")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    detector = VehicleDetector(model_path=args.model, conf=args.conf)

    source = args.source
    if source.isdigit():
        run_video(detector, int(source))
    elif Path(source).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        run_video(detector, source)
    else:
        run_image(detector, source)


if __name__ == "__main__":
    main()
