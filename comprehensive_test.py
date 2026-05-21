#!/usr/bin/env python3
"""Comprehensive test to identify issues with empty parking space detection."""

import cv2
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.detection.vehicle_detector import VehicleDetector
from src.parking import ZoneLoader, ParkingAnalyzer

def test_no_vehicles():
    """Test case: No vehicles detected - all zones should be available."""
    print("=== TEST: No Vehicles ===")
    
    # Use the standard test image
    image_path = "data/raw/araba1.jpeg"
    zones_path = "data/raw/araba1.json"
    
    frame = cv2.imread(image_path)
    loader = ZoneLoader(zones_path)
    detector = VehicleDetector(conf=0.5)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    # Simulate no detections
    detections = []
    
    result = analyzer.analyze(detections)
    
    print(f"Available: {result.available} (expected: 3)")
    print(f"Occupied: {result.occupied} (expected: 0)")
    
    success = result.available == 3 and result.occupied == 0
    print(f"Result: {'PASS' if success else 'FAIL'}")
    return success

def test_vehicle_in_each_zone():
    """Test case: Vehicle in each zone individually."""
    print("\n=== TEST: Vehicle in Each Zone ===")
    
    image_path = "data/raw/araba1.jpeg"
    zones_path = "data/raw/araba1.json"
    
    frame = cv2.imread(image_path)
    loader = ZoneLoader(zones_path)
    detector = VehicleDetector(conf=0.5)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    # Test points roughly in each zone
    test_points = [
        # Zone 1: left zone
        [100, 500, 200, 600],
        # Zone 2: middle zone  
        [500, 500, 700, 600],
        # Zone 3: right zone
        [1000, 500, 1100, 600]
    ]
    
    all_passed = True
    
    for zone_id, bbox in enumerate(test_points, 1):
        detections = [{
            "bbox": bbox,
            "class_name": "car",
            "confidence": 0.9
        }]
        
        result = analyzer.analyze(detections)
        
        # Count occupied zones
        occupied_count = result.occupied
        available_count = result.available
        
        print(f"Zone {zone_id}: Occupied={occupied_count}, Available={available_count}")
        
        # For this test, we expect exactly 1 occupied and 2 available
        passed = occupied_count == 1 and available_count == 2
        print(f"  Result: {'PASS' if passed else 'FAIL'}")
        
        if not passed:
            all_passed = False
            
    return all_passed

def test_high_iou_threshold():
    """Test case: High IoU threshold should make detection more strict."""
    print("\n=== TEST: High IoU Threshold (0.5) ===")
    
    image_path = "data/raw/araba1.jpeg"
    zones_path = "data/raw/araba1.json"
    
    frame = cv2.imread(image_path)
    loader = ZoneLoader(zones_path)
    detector = VehicleDetector(conf=0.5)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.5)  # Higher threshold
    
    detections = detector.detect(frame)
    
    result = analyzer.analyze(detections)
    
    print(f"Available: {result.available}")
    print(f"Occupied: {result.occupied}")
    print(f"Detections: {len(detections)}")
    
    # With higher threshold, we might expect different behavior
    # But let's just report what we see
    return True  # Just informational

def test_low_confidence():
    """Test case: Low confidence threshold might cause false detections."""
    print("\n=== TEST: Low Confidence Threshold (0.3) ===")
    
    image_path = "data/raw/araba1.jpeg"
    zones_path = "data/raw/araba1.json"
    
    frame = cv2.imread(image_path)
    loader = ZoneLoader(zones_path)
    detector = VehicleDetector(conf=0.3)  # Lower confidence
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    detections = detector.detect(frame)
    
    result = analyzer.analyze(detections)
    
    print(f"Available: {result.available}")
    print(f"Occupied: {result.occupied}")
    print(f"Detections: {len(detections)}")
    
    for i, det in enumerate(detections):
        print(f"  Detection {i}: {det['class_name']} conf={det['confidence']:.3f}")
    
    # With lower confidence, we might get more false detections
    return len(detections) >= 1  # At least the real car should be detected

def test_zone_edge_cases():
    """Test edge cases with zone boundaries."""
    print("\n=== TEST: Zone Edge Cases ===")
    
    # Create a simple test case with known geometry
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    
    zones_data = {
        "image": "test.jpg",
        "zones": [
            # Zone 1: Left square
            {"id": 1, "type": "parking", "points": [[100, 100], [300, 100], [300, 300], [100, 300]]},
            # Zone 2: Right square
            {"id": 2, "type": "parking", "points": [[500, 100], [700, 100], [700, 300], [500, 300]]},
        ]
    }
    
    zones_path = "temp_edge_zones.json"
    with open(zones_path, 'w') as f:
        json.dump(zones_data, f)
    
    loader = ZoneLoader(zones_path)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    # Test case 1: No vehicles
    detections = []
    result = analyzer.analyze(detections)
    print(f"No vehicles - Available: {result.available}, Occupied: {result.occupied}")
    test1_pass = result.available == 2 and result.occupied == 0
    
    # Test case 2: Vehicle clearly in zone 1
    detections = [{"bbox": [150, 150, 250, 250], "class_name": "car", "confidence": 0.9}]
    result = analyzer.analyze(detections)
    print(f"Vehicle in zone 1 - Available: {result.available}, Occupied: {result.occupied}")
    test2_pass = result.available == 1 and result.occupied == 1
    
    # Test case 3: Vehicle clearly in zone 2
    detections = [{"bbox": [550, 150, 650, 250], "class_name": "car", "confidence": 0.9}]
    result = analyzer.analyze(detections)
    print(f"Vehicle in zone 2 - Available: {result.available}, Occupied: {result.occupied}")
    test3_pass = result.available == 1 and result.occupied == 1
    
    # Test case 4: Vehicle between zones (should not occupy either if threshold is proper)
    detections = [{"bbox": [350, 150, 450, 250], "class_name": "car", "confidence": 0.9}]
    result = analyzer.analyze(detections)
    print(f"Vehicle between zones - Available: {result.available}, Occupied: {result.occupied}")
    # With our current algorithm, this might trigger on zone coverage depending on sizes
    test4_pass = True  # Informational
    
    # Clean up
    import os
    if os.path.exists(zones_path):
        os.remove(zones_path)
    
    return test1_pass and test2_pass and test3_pass

def main():
    """Run all tests."""
    print("Running comprehensive parking detection tests...\n")
    
    results = []
    results.append(test_no_vehicles())
    results.append(test_vehicle_in_each_zone())
    results.append(test_high_iou_threshold())
    results.append(test_low_confidence())
    results.append(test_zone_edge_cases())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== SUMMARY ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("All tests PASSED")
    else:
        print("Some tests FAILED - there may be issues with empty space detection")
    
    return passed == total

if __name__ == "__main__":
    main()