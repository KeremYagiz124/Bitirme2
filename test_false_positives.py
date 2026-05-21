#!/usr/bin/env python3
"""Test for false positives: empty spots marked as occupied."""

import cv2
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.detection.vehicle_detector import VehicleDetector
from src.parking import ZoneLoader, ParkingAnalyzer

def test_false_positive_scenario():
    """Create a scenario where we expect false positives."""
    print("=== TESTING FOR FALSE POSITIVES ===")
    print("Looking for cases where empty spots are marked as occupied\n")
    
    # Test with the standard image first
    image_path = "data/raw/araba1.jpeg"
    zones_path = "data/raw/araba1.json"
    
    frame = cv2.imread(image_path)
    loader = ZoneLoader(zones_path)
    detector = VehicleDetector(conf=0.5)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    # Get actual detections
    detections = detector.detect(frame)
    
    print(f"Actual detections in image: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  {i}: {det['class_name']} conf={det['confidence']:.3f}")
    
    result = analyzer.analyze(detections)
    
    print(f"\nAnalysis results:")
    print(f"  Available spots: {result.available} (expected: 2)")
    print(f"  Occupied spots: {result.occupied} (expected: 1)")
    
    # Check each zone
    false_positives = []
    for zs in result.zone_statuses:
        if zs.zone.type == "parking":
            is_actually_occupied = False
            # Check if any detection actually overlaps this zone
            for det in detections:
                if zone_actually_contains_vehicle(zs.zone, det["bbox"], analyzer.iou_threshold):
                    is_actually_occupied = True
                    break
            
            if zs.status == "Park Alanı Dolu" and not is_actually_occupied:
                false_positives.append(zs.zone.id)
                print(f"  FALSE POSITIVE: Zone {zs.zone.id} marked occupied but no vehicle actually there")
            elif zs.status == "Park Edilebilir" and is_actually_occupied:
                print(f"  FALSE NEGATIVE: Zone {zs.zone.id} marked available but vehicle actually there")
            else:
                print(f"  CORRECT: Zone {zs.zone.id} ({zs.status})")
    
    if false_positives:
        print(f"\nFound {len(false_positives)} false positives: zones {false_positives}")
        return False
    else:
        print("\nNo false positives detected in standard test")
        return True

def zone_actually_contains_vehicle(zone, bbox, threshold=0.25):
    """Check if a vehicle bbox actually indicates occupancy based on our criteria."""
    return zone.coverage_by_bbox(bbox) >= threshold or zone.contains_center(bbox)

def test_synthetic_scenario():
    """Create a synthetic test case designed to trigger false positives."""
    print("\n=== TESTING SYNTHETIC SCENARIO ===")
    
    # Create a test image
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame[:] = (100, 100, 100)  # Gray background
    
    # Define zones that are close together
    zones_data = {
        "image": "synthetic.jpg",
        "zones": [
            # Zone 1: Left parking spot
            {"id": 1, "type": "parking", "points": [[100, 200], [200, 200], [200, 400], [100, 400]]},
            # Zone 2: Right parking spot  
            {"id": 2, "type": "parking", "points": [[300, 200], [400, 200], [400, 400], [300, 400]]},
            # Add a obstacle zone between them that might cause confusion
            {"id": 3, "type": "forbidden", "points": [[200, 100], [300, 100], [300, 150], [200, 150]]},
        ]
    }
    
    zones_path = "synthetic_zones.json"
    with open(zones_path, 'w') as f:
        json.dump(zones_data, f)
    
    # Add some visual markers to the frame
    cv2.rectangle(frame, (100, 200), (200, 400), (0, 255, 0), 2)  # Zone 1
    cv2.rectangle(frame, (300, 200), (400, 400), (0, 255, 0), 2)  # Zone 2
    cv2.rectangle(frame, (200, 100), (300, 150), (0, 0, 255), 2)  # Forbidden zone
    
    loader = ZoneLoader(zones_path)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    # Test 1: No vehicles - should all be available
    print("Test 1: No vehicles")
    detections = []
    result = analyzer.analyze(detections)
    print(f"  Available: {result.available}, Occupied: {result.occupied}")
    
    zone1_status = next(zs for zs in result.zone_statuses if zs.zone.id == 1 and zs.zone.type == "parking")
    zone2_status = next(zs for zs in result.zone_statuses if zs.zone.id == 2 and zs.zone.type == "parking")
    
    test1_pass = (zone1_status.status == "Park Edilebilir" and 
                  zone2_status.status == "Park Edilebilir")
    print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")
    
    # Test 2: Place a vehicle where it might trigger false positives due to edge cases
    # Let's try placing a detection that barely touches or is near zones
    print("\nTest 2: Vehicle near zone boundaries")
    
    # Try a detection that's between the zones but might trigger coverage
    detections = [{
        "bbox": [220, 180, 280, 220],  # Small box between zones, slightly overlapping forbidden
        "class_name": "car",
        "confidence": 0.9
    }]
    
    result = analyzer.analyze(detections)
    print(f"  Detection bbox: {detections[0]['bbox']}")
    print(f"  Available: {result.available}, Occupied: {result.occupied}")
    
    for zs in result.zone_statuses:
        if zs.zone.type == "parking":
            print(f"  Zone {zs.zone.id}: {zs.status}")
    
    # Check for false positives
    false_positives = []
    for zs in result.zone_statuses:
        if zs.zone.type == "parking":
            is_actually_occupied = zone_actually_contains_vehicle(zs.zone, detections[0]["bbox"], analyzer.iou_threshold)
            if zs.status == "Park Alanı Dolu" and not is_actually_occupied:
                false_positives.append(zs.zone.id)
    
    if false_positives:
        print(f"  FALSE POSITIVES in zones: {false_positives}")
        test2_pass = False
    else:
        print(f"  No false positives")
        test2_pass = True
    
    # Test 3: Vehicle that clearly should occupy a zone
    print("\nTest 3: Vehicle clearly in zone 1")
    detections = [{
        "bbox": [120, 220, 180, 380],  # Clearly in zone 1
        "class_name": "car",
        "confidence": 0.9
    }]
    
    result = analyzer.analyze(detections)
    zone1_status = next(zs for zs in result.zone_statuses if zs.zone.id == 1 and zs.zone.type == "parking")
    zone2_status = next(zs for zs in result.zone_statuses if zs.zone.id == 2 and zs.zone.type == "parking")
    
    print(f"  Zone 1: {zone1_status.status} (should be occupied)")
    print(f"  Zone 2: {zone2_status.status} (should be available)")
    
    test3_pass = (zone1_status.status == "Park Alanı Dolu" and 
                  zone2_status.status == "Park Edilebilir")
    print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")
    
    # Clean up
    import os
    if os.path.exists(zones_path):
        os.remove(zones_path)
    
    return test1_pass and test2_pass and test3_pass

def test_perspective_issues():
    """Test if perspective/distortion causes issues."""
    print("\n=== TESTING PERSPECTIVE ISSUES ===")
    
    # The original image has perspective - let's check if that causes issues
    image_path = "data/raw/araba1.jpeg"
    zones_path = "data/raw/araba1.json"
    
    frame = cv2.imread(image_path)
    loader = ZoneLoader(zones_path)
    detector = VehicleDetector(conf=0.5)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    # Get detections
    detections = detector.detect(frame)
    
    print(f"Image has {len(detections)} detections")
    
    # For each detection, check what zones it affects
    for i, det in enumerate(detections):
        print(f"\nDetection {i}: {det['class_name']} conf={det['confidence']:.3f}")
        print(f"  Bbox: {det['bbox']}")
        
        for zone in loader.parking_zones:
            coverage = zone.coverage_by_bbox(det["bbox"])
            contains_center = zone.contains_center(det["bbox"])
            
            print(f"  Zone {zone.id}: coverage={coverage:.3f}, center_in={contains_center}")
            
            # This is the key condition from the analyzer
            would_trigger = coverage >= analyzer.iou_threshold or contains_center
            print(f"    Would trigger occupancy: {would_trigger} (threshold={analyzer.iou_threshold})")
    
    # Run the actual analysis
    result = analyzer.analyze(detections)
    print(f"\nActual results: Available={result.available}, Occupied={result.occupied}")
    
    return True

def main():
    """Run all tests."""
    print("Testing for false positives in parking detection...\n")
    
    results = []
    results.append(test_false_positive_scenario())
    results.append(test_synthetic_scenario())
    results.append(test_perspective_issues())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Test groups passed: {passed}/{total}")
    
    if passed == total:
        print("All tests PASSED - no obvious false positive issues detected")
    else:
        print("Some test groups had issues - investigate further")
    
    return passed == total

if __name__ == "__main__":
    main()