"""Stres ve kenar durum testleri."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ui.alert_system import AlertSystem, AlertLevel
from src.parking.zone_loader import ZoneLoader
from src.parking.parking_analyzer import ParkingAnalyzer, STATUS_AVAILABLE, STATUS_OCCUPIED
from src.detection.street_parking_detector import StreetParkingDetector


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_zone_json(tmp_path, zones: list) -> Path:
    data = {"image": "test.jpg", "zones": zones}
    p = tmp_path / "zones.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _blank_frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _det(x1, y1, x2, y2, cls=2, conf=0.9):
    """VehicleDetector çıktı formatında tespit."""
    return {"bbox": [x1, y1, x2, y2], "class_id": cls, "confidence": conf, "class_name": "car"}


# ── AlertSystem ───────────────────────────────────────────────────────────────

class TestAlertSystem:

    def test_throttle_prevents_duplicate(self):
        a = AlertSystem(throttle_sec=60.0)
        assert a.fire("x", AlertLevel.INFO, "msg") is True
        assert a.fire("x", AlertLevel.INFO, "msg") is False

    def test_different_codes_not_throttled(self):
        a = AlertSystem(throttle_sec=60.0)
        assert a.fire("a", AlertLevel.INFO, "msg") is True
        assert a.fire("b", AlertLevel.INFO, "msg") is True

    def test_clear_code_resets_throttle(self):
        a = AlertSystem(throttle_sec=60.0)
        a.fire("x", AlertLevel.INFO, "msg")
        a.clear_code("x")
        assert a.fire("x", AlertLevel.INFO, "msg") is True

    def test_history_capped(self):
        a = AlertSystem(throttle_sec=0.0, max_history=5)
        for i in range(10):
            a.fire(f"code{i}", AlertLevel.INFO, "msg")
        assert len(a._history) == 5

    def test_listener_called(self):
        a = AlertSystem(throttle_sec=0.0)
        received = []
        a.add_listener(received.append)
        a.fire("x", AlertLevel.WARNING, "test")
        assert len(received) == 1
        assert received[0].code == "x"

    def test_listener_exception_does_not_crash(self):
        a = AlertSystem(throttle_sec=0.0)
        a.add_listener(lambda al: 1 / 0)
        a.fire("x", AlertLevel.CRITICAL, "boom")  # hata fırlatmamalı

    def test_check_occupancy_full(self):
        a = AlertSystem(throttle_sec=0.0)
        fired = []
        a.add_listener(fired.append)
        a.check_occupancy(available=0, occupied=10)
        assert any(al.code == "park_full" for al in fired)

    def test_check_occupancy_high(self):
        a = AlertSystem(throttle_sec=0.0)
        fired = []
        a.add_listener(fired.append)
        a.check_occupancy(available=2, occupied=18)
        assert any(al.code == "park_high" for al in fired)

    def test_check_occupancy_low_clears(self):
        a = AlertSystem(throttle_sec=0.0)
        a.fire("park_full", AlertLevel.CRITICAL, "x")
        a.check_occupancy(available=5, occupied=5)
        assert "park_full" not in a._last_fired
        assert "park_high" not in a._last_fired

    def test_check_forbidden(self):
        a = AlertSystem(throttle_sec=0.0)
        fired = []
        a.add_listener(fired.append)
        a.check_forbidden(3)
        assert any(al.code == "forbidden_park" for al in fired)

    def test_check_no_fit_triggers(self):
        a = AlertSystem(throttle_sec=0.0)
        fired = []
        a.add_listener(fired.append)
        a.check_no_fit(fit_count=0, total_empty=3)
        assert any(al.code == "no_fit" for al in fired)

    def test_check_no_fit_no_empty_no_fire(self):
        a = AlertSystem(throttle_sec=0.0)
        fired = []
        a.add_listener(fired.append)
        a.check_no_fit(fit_count=0, total_empty=0)
        assert not fired

    def test_rapid_fire_many_codes(self):
        a = AlertSystem(throttle_sec=0.0, max_history=100)
        for i in range(200):
            a.fire(f"c{i}", AlertLevel.INFO, "msg")
        assert len(a._history) == 100


# ── ParkingAnalyzer ───────────────────────────────────────────────────────────

class TestParkingAnalyzerStress:

    def _make_analyzer(self, tmp_path, zone_count=5):
        size = 100
        zones = [
            {
                "id": i + 1,
                "type": "parking",
                "points": [
                    [i * size, 0],
                    [(i + 1) * size, 0],
                    [(i + 1) * size, size],
                    [i * size, size],
                ],
            }
            for i in range(zone_count)
        ]
        p = _make_zone_json(tmp_path, zones)
        loader = ZoneLoader(str(p))
        return ParkingAnalyzer(loader, iou_threshold=0.25)

    def test_no_detections(self, tmp_path):
        analyzer = self._make_analyzer(tmp_path)
        result = analyzer.analyze([])
        statuses = [zs.status for zs in result.zone_statuses]
        assert all(s == STATUS_AVAILABLE for s in statuses)

    def test_all_zones_occupied(self, tmp_path):
        analyzer = self._make_analyzer(tmp_path, zone_count=5)
        dets = [_det(i * 100 + 10, 10, i * 100 + 90, 90) for i in range(5)]
        result = analyzer.analyze(dets)
        statuses = [zs.status for zs in result.zone_statuses]
        assert all(s == STATUS_OCCUPIED for s in statuses)

    def test_many_detections(self, tmp_path):
        analyzer = self._make_analyzer(tmp_path, zone_count=3)
        dets = [_det(j * 5, j * 5, j * 5 + 50, j * 5 + 50) for j in range(50)]
        result = analyzer.analyze(dets)
        assert len(result.zone_statuses) == 3

    def test_no_zones(self, tmp_path):
        p = _make_zone_json(tmp_path, [])
        loader = ZoneLoader(str(p))
        analyzer = ParkingAnalyzer(loader)
        result = analyzer.analyze([_det(0, 0, 100, 100)])
        assert result.zone_statuses == []

    def test_single_zone_repeated_calls(self, tmp_path):
        analyzer = self._make_analyzer(tmp_path, zone_count=1)
        for _ in range(100):
            result = analyzer.analyze([_det(10, 10, 90, 90)])
        assert len(result.zone_statuses) == 1

    def test_overlapping_detections(self, tmp_path):
        analyzer = self._make_analyzer(tmp_path, zone_count=2)
        dets = [_det(10, 10, 90, 90)] * 20
        result = analyzer.analyze(dets)
        assert len(result.zone_statuses) == 2


# ── StreetParkingDetector ─────────────────────────────────────────────────────

class TestStreetParkingDetectorStress:

    def test_empty_frame_no_vehicles(self):
        det = StreetParkingDetector()
        frame = _blank_frame()
        result = det.analyze(frame, [])
        assert isinstance(result["empty_count"], int)
        assert isinstance(result["occupied_count"], int)

    def test_single_vehicle(self):
        det = StreetParkingDetector()
        frame = _blank_frame()
        dets = [_det(100, 300, 300, 430)]
        result = det.analyze(frame, dets)
        assert result["occupied_count"] >= 0

    def test_many_vehicles_no_crash(self):
        det = StreetParkingDetector()
        frame = _blank_frame(w=1280, h=720)
        dets = [_det(i * 60, 400, i * 60 + 55, 500) for i in range(20)]
        result = det.analyze(frame, dets)
        assert result["empty_count"] >= 0

    def test_tiny_frame(self):
        det = StreetParkingDetector()
        frame = _blank_frame(w=32, h=32)
        result = det.analyze(frame, [])
        assert "empty_count" in result

    def test_very_large_frame(self):
        det = StreetParkingDetector()
        frame = _blank_frame(w=3840, h=2160)
        dets = [_det(500, 1800, 900, 2100)]
        result = det.analyze(frame, dets)
        assert "empty_count" in result

    def test_reset_history_clears_state(self):
        det = StreetParkingDetector()
        frame = _blank_frame()
        dets = [_det(100, 300, 300, 430)]
        det.analyze(frame, dets)
        det.reset_history()
        result = det.analyze(frame, [])
        assert result["empty_count"] >= 0

    def test_repeated_calls_stable(self):
        det = StreetParkingDetector()
        frame = _blank_frame(w=800, h=600)
        dets = [_det(50 + i * 120, 400, 150 + i * 120, 500) for i in range(4)]
        results = [det.analyze(frame, dets) for _ in range(30)]
        counts = [r["empty_count"] for r in results]
        assert max(counts) - min(counts) <= 3  # kararlı yakınsama beklenir

    def test_vehicles_at_frame_edges(self):
        det = StreetParkingDetector()
        frame = _blank_frame()
        dets = [
            _det(0, 0, 50, 50),
            _det(590, 430, 639, 479),
        ]
        result = det.analyze(frame, dets)
        assert "empty_count" in result
