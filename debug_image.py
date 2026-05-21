#!/usr/bin/env python3
"""Debug script for testing parking detection on various images."""

import cv2
import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.detection.vehicle_detector import VehicleDetector
from src.parking import ZoneLoader, ParkingAnalyzer

def main(image_path, zones_path):
    print(f"Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
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
    stem = Path(image_path).stem
    output_path = output_dir / f"debug_{stem}_analysis.jpg"
    cv2.imwrite(str(output_path), out)
    print(f"Saved visualization to: {output_path}")
    
    # Also save the original with detections drawn
    det_only = detector.draw(frame, detections)
    det_path = output_dir / f"debug_{stem}_detections.jpg"
    cv2.imwrite(str(det_path), det_only)
    print(f"Saved detections only to: {det_path}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_image.py <image_path> <zones_path>")
        print("Example: python debug_image.py data/raw/araba2.jpg data/raw/araba2.json")
        sys.exit(1)
    
    image_path = sys.argv[1]
    zones_path = sys.argv[2]
    
    success = main(image_path, zones_path)
    sys.exit(0 if success else 1)