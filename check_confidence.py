#!/usr/bin/env python3
"""Check detections at different confidence thresholds."""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.detection.vehicle_detector import VehicleDetector

def main():
    frame = cv2.imread('data/raw/araba1.jpeg')
    if frame is None:
        print("Error: Could not load image")
        return
    
    print("Checking detections at different confidence thresholds:")
    print("Format: conf_threshold -> number of detections (list of class_name:confidence)")
    
    for conf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        detector = VehicleDetector(conf=conf)
        detections = detector.detect(frame)
        print(f"Conf {conf}: {len(detections)} detections", end="")
        if len(detections) > 0:
            details = []
            for i, det in enumerate(detections):
                details.append(f"{det['class_name']}:{det['confidence']:.3f}")
            print(" (" + ", ".join(details) + ")")
        else:
            print()

if __name__ == "__main__":
    main()