#!/usr/bin/env python3
"""Test script to check edge cases in parking detection."""

import cv2
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.detection.vehicle_detector import VehicleDetector
from src.parking import ZoneLoader, ParkingAnalyzer

def test_edge_case():
    """Create a test case that might reveal issues."""
    
    # Create a simple test image
    frame = np.zeros((800, 1200, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Dark gray background
    
    # Draw some parking zones (as rectangles for simplicity)
    zones_data = {
        "image": "test.jpg",
        "zones": [
            # Zone 1: Left zone - should be empty
            {"id": 1, "type": "parking", "points": [[100, 400], [300, 400], [300, 600], [100, 600]]},
            # Zone 2: Middle zone - should be occupied
            {"id": 2, "type": "parking", "points": [[400, 400], [600, 400], [600, 600], [400, 600]]},
            # Zone 3: Right zone - should be empty
            {"id": 3, "type": "parking", "points": [[700, 400], [900, 400], [900, 600], [700, 600]]},
        ]
    }
    
    # Save zones to temp file
    zones_path = "temp_zones.json"
    with open(zones_path, 'w') as f:
        json.dump(zones_data, f)
    
    # Draw zones on frame for visualization
    for zone in zones_data["zones"]:
        pts = np.array(zone["points"], dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(frame, f"Zone {zone['id']}", (cx - 20, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add a vehicle detection in the middle zone
    # This should occupy zone 2 only
    detections = [{
        "bbox": [450, 450, 550, 550],  # Square in middle of zone 2
        "class_name": "car",
        "confidence": 0.9
    }]
    
    # Also test a detection that overlaps multiple zones
    detections_overlap = [{
        "bbox": [250, 450, 650, 550],  # Spans zones 1 and 2
        "class_name": "car",
        "confidence": 0.9
    }]
    
    # Test normal case
    print("=== TEST 1: Vehicle in middle zone ===")
    loader = ZoneLoader(zones_path)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    result = analyzer.analyze(detections)
    print(f"Results: Available={result.available}, Occupied={result.occupied}")
    
    for zs in result.zone_statuses:
        if zs.zone.type == "parking":
            print(f"  Zone {zs.zone.id}: {zs.status}")
    
    # Test overlapping case
    print("\n=== TEST 2: Vehicle overlapping left and middle zones ===")
    result2 = analyzer.analyze(detections_overlap)
    print(f"Results: Available={result2.available}, Occupied={result2.occupied}")
    
    for zs in result2.zone_statuses:
        if zs.zone.type == "parking":
            print(f"  Zone {zs.zone.id}: {zs.status}")
            
    # Test edge case: vehicle right on boundary
    print("\n=== TEST 3: Vehicle on zone boundary ===")
    boundary_detections = [{
        "bbox": [300, 450, 400, 550],  # Right on boundary between zone 1 and 2
        "class_name": "car",
        "confidence": 0.9
    }]
    
    result3 = analyzer.analyze(boundary_detections)
    print(f"Results: Available={result3.available}, Occupied={result3.occupied}")
    
    for zs in result3.zone_statuses:
        if zs.zone.type == "parking":
            print(f"  Zone {zs.zone.id}: {zs.status}")
    
    # Clean up
    import os
    if os.path.exists(zones_path):
        os.remove(zones_path)

if __name__ == "__main__":
    test_edge_case()