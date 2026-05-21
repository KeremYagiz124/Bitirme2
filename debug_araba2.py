#!/usr/bin/env python3
"""Debug script for araba2.jpg"""

import cv2
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.detection.vehicle_detector import VehicleDetector
from src.parking import ZoneLoader, ParkingAnalyzer

def main():
    image_path = "data/raw/araba2.jpg"
    zones_path = "data/raw/araba2.json"
    
    print(f"Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"Image shape: {frame.shape}")
    
    # Load zones
    print(f"Loading zones: {zones_path}")
    loader = ZoneLoader(zones_path)
    print(f"Loaded {len(loader.parking_zones)} parking zones")
    print(f"Loaded {len(loader.forbidden_zones)} forbidden zones")
    
    for i, zone in enumerate(loader.parking_zones):
        print(f"  Zone {zone.id}: {zone.type} - points shape: {zone.polygon.shape}")
    
    # Initialize detector and analyzer
    detector = VehicleDetector(conf=0.5)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    # Run detection
    print("\nRunning vehicle detection...")
    detections = detector.detect(frame)
    print(f"Detected {len(detections)} vehicles:")
    for i, det in enumerate(detections):
        print(f"  {i}: {det['class_name']} (conf={det['confidence']:.3f}) bbox={det['bbox']}")
    
    # Run analysis
    print("\nRunning parking analysis...")
    result = analyzer.analyze(detections)
    
    print(f"Results:")
    print(f"  Available: {result.available}")
    print(f"  Occupied: {result.occupied}")
    print(f"  Forbidden vehicles: {result.forbidden_vehicles}")
    
    print("\nDetailed zone status:")
    for zs in result.zone_statuses:
        if zs.zone.type == "parking":
            print(f"  Zone {zs.zone.id}: {zs.status}")
            if zs.vehicle_bbox:
                print(f"    Occupied by bbox: {zs.vehicle_bbox}")
            if zs.occupied_since:
                import time
                dur = time.time() - zs.occupied_since
                print(f"    Occupied since: {dur:.1f} seconds")
    
    # Save visualization without displaying
    print("\nGenerating visualization...")
    out = analyzer.draw(frame, result, detections)
    
    output_dir = Path("outputs/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "debug_araba2_analysis.jpg"
    cv2.imwrite(str(output_path), out)
    print(f"Saved visualization to: {output_path}")
    
    # Also save the original with detections drawn
    det_only = detector.draw(frame, detections)
    det_path = output_dir / "debug_araba2_detections.jpg"
    cv2.imwrite(str(det_path), det_only)
    print(f"Saved detections only to: {det_path}")

if __name__ == "__main__":
    main()