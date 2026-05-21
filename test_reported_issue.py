#!/usr/bin/env python3
"""Test to see if empty spaces are being incorrectly marked as occupied (the reported issue)."""

import cv2
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.detection.vehicle_detector import VehicleDetector
from src.parking import ZoneLoader, ParkingAnalyzer

def test_empty_spaces_marked_occupied():
    """Test specifically for the issue: empty spaces marked as occupied."""
    
    print("=== TESTING FOR EMPTY SPACES BEING MARKED AS OCCUPIED ===")
    print("This is the reported issue: boş park alanlarının yanlışlıkla dolu olarak işaretlenmesi\n")
    
    # Test with the standard image
    image_path = "data/raw/araba1.jpeg"
    zones_path = "data/raw/araba1.json"
    
    frame = cv2.imread(image_path)
    loader = ZoneLoader(zones_path)
    detector = VehicleDetector(conf=0.5)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    # Get actual detections
    detections = detector.detect(frame)
    
    print(f"Actual detections: {len(detections)} vehicles")
    for i, det in enumerate(detections):
        print(f"  {i}: {det['class_name']} conf={det['confidence']:.3f}")
    
    # Run analysis
    result = analyzer.analyze(detections)
    
    print(f"\nAnalysis results:")
    print(f"  Available spots: {result.available}")
    print(f"  Occupied spots: {result.occupied}")
    
    # Check each zone to see if any EMPTY zones are marked as OCCUPIED
    false_occupancies = []  # Zones that are empty but marked occupied
    correct_zones = []
    
    print(f"\nDetailed zone analysis:")
    for zs in result.zone_statuses:
        if zs.zone.type == "parking":
            zone_id = zs.zone.id
            is_marked_occupied = (zs.status == "Park Alanı Dolu")
            
            # Check if zone is actually empty (no vehicle detection overlaps significantly)
            is_actually_empty = True
            for det in detections:
                if zone_indicates_occupation(zs.zone, det["bbox"], analyzer.iou_threshold):
                    is_actually_empty = False
                    break
            
            print(f"  Zone {zone_id}:")
            print(f"    Marked as: {zs.status}")
            print(f"    Actually empty: {is_actually_empty}")
            
            if is_marked_occupied and is_actually_empty:
                false_occupancies.append(zone_id)
                print(f"    --> FALSE OCCUPANCY: Empty zone marked as occupied!")
            elif not is_marked_occupied and not is_actually_empty:
                print(f"    --> FALSE EMPTY: Occupied zone marked as empty!")
            else:
                print(f"    --> CORRECT")
                correct_zones.append(zone_id)
    
    print(f"\n=== RESULTS ===")
    print(f"Correctly classified zones: {len(correct_zones)}")
    print(f"False occupancies (empty marked as occupied): {len(false_occupancies)}")
    
    if false_occupancies:
        print(f"FALSE OCCUPANCIES DETECTED in zones: {false_occupancies}")
        print("THIS IS THE REPORTED BUG!")
        return False
    else:
        print("No false occupancies detected.")
        # But let's double-check by looking at what the JSON expects
        print("\nChecking against JSON expected values...")
        with open(zones_path) as f:
            zone_data = json.load(f)
        
        expected_map = {}
        for zone in zone_data['zones']:
            if zone['type'] == 'parking':
                expected_map[zone['id']] = zone.get('expected', 'unknown')
        
        mismatches = []
        for zs in result.zone_statuses:
            if zs.zone.type == "parking":
                zone_id = zs.zone.id
                expected = expected_map.get(zone_id, "unknown")
                actual = zs.status
                
                # Convert to comparable values
                actual_simple = "available" if actual == "Park Edilebilir" else "occupied" if actual == "Park Alanı Dolu" else actual
                
                if expected != "unknown" and expected != actual_simple:
                    mismatches.append((zone_id, expected, actual_simple))
        
        if mismatches:
            print(f"MISMATCHES WITH JSON EXPECTED VALUES:")
            for zone_id, expected, actual in mismatches:
                print(f"  Zone {zone_id}: Expected {expected}, Got {actual}")
            return False
        else:
            print("All zones match JSON expected values.")
            return True

def zone_indicates_occupation(zone, bbox, threshold=0.25):
    """Return True if the bbox indicates the zone should be considered occupied."""
    return zone.coverage_by_bbox(bbox) >= threshold or zone.contains_center(bbox)

def test_with_different_scenarios():
    """Test with different scenarios to try to reproduce the issue."""
    
    print("\n=== TESTING DIFFERENT SCENARIOS ===")
    
    # Scenario 1: No detections at all
    print("\nScenario 1: No vehicle detections")
    image_path = "data/raw/araba1.jpeg"
    zones_path = "data/raw/araba1.json"
    
    frame = cv2.imread(image_path)
    loader = ZoneLoader(zones_path)
    detector = VehicleDetector(conf=0.5)  # This won't matter since we'll override detections
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    # Force no detections
    detections = []
    result = analyzer.analyze(detections)
    
    print(f"  Results: Available={result.available}, Occupied={result.occupied}")
    
    # All should be available
    expected_available = 3
    if result.available == expected_available and result.occupied == 0:
        print("  --> CORRECT: All zones available when no detections")
    else:
        print(f"  --> ERROR: Expected {expected_available} available, 0 occupied")
        return False
    
    # Scenario 2: Very low confidence detections (might cause false positives)
    print("\nScenario 2: Very low confidence threshold (might cause false detections)")
    detector_low_conf = VehicleDetector(conf=0.1)  # Very low confidence
    
    detections = detector_low_conf.detect(frame)
    print(f"  Detections at conf=0.1: {len(detections)}")
    
    for i, det in enumerate(detections[:5]):  # Show first 5
        print(f"    {i}: {det['class_name']} conf={det['confidence']:.3f}")
    
    if len(detections) > 0:
        result = analyzer.analyze(detections)
        print(f"  Results: Available={result.available}, Occupied={result.occupied}")
        
        # Check for false occupancies
        false_occupancies = []
        for zs in result.zone_statuses:
            if zs.zone.type == "parking":
                is_marked_occupied = (zs.status == "Park Alanı Dolu")
                is_actually_empty = True
                for det in detections:
                    if zone_indicates_occupation(zs.zone, det["bbox"], analyzer.iou_threshold):
                        is_actually_empty = False
                        break
                
                if is_marked_occupied and is_actually_empty:
                    false_occupancies.append(zs.zone.id)
        
        if false_occupancies:
            print(f"  --> FALSE OCCUPANCIES: {false_occupancies}")
            return False
        else:
            print("  --> No false occupancies with low confidence")
    else:
        print("  --> No detections even at low confidence")
    
    return True

def main():
    """Run all tests."""
    print("Testing for the specific issue: empty parking spaces marked as occupied\n")
    
    results = []
    results.append(test_empty_spaces_marked_occupied())
    results.append(test_with_different_scenarios())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Test groups passed: {passed}/{total}")
    
    if passed == total:
        print("No evidence of the reported bug found in current testing.")
        print("The system appears to be working correctly for the test cases examined.")
    else:
        print("EVIDENCE OF THE REPORTED BUG FOUND!")
        print("Empty parking spaces are being incorrectly marked as occupied.")
    
    return passed == total

if __name__ == "__main__":
    main()