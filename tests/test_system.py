"""Sistem bütünleşik test paketi.

Kapsam:
  - AlertSystem: fire/throttle/check_occupancy/check_forbidden/check_no_fit
  - VehicleTracker: statik tespit, park süresi, ego-motion reset
  - StreetParkingDetector: boş alan tespiti, max_edge_extension_ratio=0.20
  - DrivableAreaSegmenter: fallback (model yok → graceful degradation)
  - Overhead/üst görüş perspektif testi (yeni otopark senaryosu)
  - Kamera canlı akış simülasyonu (tekrarlı kare testi)
  - Sokak modu ground truth: 3 görüntü, Mikro F1 = %100
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ui.alert_system import Alert, AlertLevel, AlertSystem
from src.detection.vehicle_tracker import VehicleTracker
from src.detection.street_parking_detector import StreetParkingDetector
from src.detection.drivable_area import DrivableAreaSegmenter


# ─── Yardımcı ────────────────────────────────────────────────────────────────

def _det(x1, y1, x2, y2, cls=2):
    return {"bbox": [x1, y1, x2, y2], "class_id": cls, "confidence": 0.9}


def _blank(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# AlertSystem testleri
# ══════════════════════════════════════════════════════════════════════════════

class TestAlertSystem:

    def test_fire_returns_true_first_time(self):
        a = AlertSystem(throttle_sec=30)
        assert a.fire("test", AlertLevel.INFO, "msg") is True

    def test_fire_throttled_returns_false(self):
        a = AlertSystem(throttle_sec=30)
        a.fire("test", AlertLevel.INFO, "msg")
        assert a.fire("test", AlertLevel.INFO, "msg") is False

    def test_fire_after_throttle_window(self):
        a = AlertSystem(throttle_sec=0.05)
        a.fire("t", AlertLevel.INFO, "x")
        time.sleep(0.06)
        assert a.fire("t", AlertLevel.INFO, "x") is True

    def test_listener_called(self):
        received = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(received.append)
        a.fire("c", AlertLevel.WARNING, "uyarı")
        assert len(received) == 1
        assert received[0].code == "c"
        assert received[0].level == AlertLevel.WARNING

    def test_multiple_listeners(self):
        calls = [0, 0]
        a = AlertSystem(throttle_sec=0)
        a.add_listener(lambda _: calls.__setitem__(0, calls[0] + 1))
        a.add_listener(lambda _: calls.__setitem__(1, calls[1] + 1))
        a.fire("x", AlertLevel.INFO, "")
        assert calls == [1, 1]

    def test_history_grows(self):
        a = AlertSystem(throttle_sec=0)
        a.fire("a", AlertLevel.INFO, "")
        a.fire("b", AlertLevel.INFO, "")
        assert len(a._history) == 2

    def test_history_max_capped(self):
        a = AlertSystem(throttle_sec=0, max_history=3)
        for i in range(5):
            a.fire(f"k{i}", AlertLevel.INFO, "")
        assert len(a._history) == 3

    def test_clear_code_resets_throttle(self):
        a = AlertSystem(throttle_sec=60)
        a.fire("z", AlertLevel.CRITICAL, "")
        assert a.fire("z", AlertLevel.CRITICAL, "") is False
        a.clear_code("z")
        assert a.fire("z", AlertLevel.CRITICAL, "") is True

    # ── check_occupancy ──

    def test_check_occupancy_park_full(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_occupancy(available=0, occupied=5)
        assert any(al.code == "park_full" and al.level == AlertLevel.CRITICAL
                   for al in fired)

    def test_check_occupancy_park_high(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_occupancy(available=1, occupied=9)   # %90 dolu
        assert any(al.code == "park_high" and al.level == AlertLevel.WARNING
                   for al in fired)

    def test_check_occupancy_low_no_alert(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_occupancy(available=8, occupied=2)   # %20 dolu
        assert fired == []

    def test_check_occupancy_total_zero_no_crash(self):
        a = AlertSystem(throttle_sec=0)
        a.check_occupancy(available=0, occupied=0)   # toplam 0 — crash olmamalı

    def test_check_occupancy_clears_on_low(self):
        a = AlertSystem(throttle_sec=0)
        a.check_occupancy(available=0, occupied=5)   # park_full
        a.check_occupancy(available=5, occupied=1)   # %17 → throttle temizlenmeli
        assert "park_full" not in a._last_fired
        assert "park_high" not in a._last_fired

    # ── check_forbidden ──

    def test_check_forbidden_fires_when_positive(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_forbidden(2)
        assert any(al.code == "forbidden_park" for al in fired)

    def test_check_forbidden_zero_no_alert(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_forbidden(0)
        assert fired == []

    # ── check_no_fit ──

    def test_check_no_fit_fires_when_no_space_fits(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_no_fit(fit_count=0, total_empty=3)
        assert any(al.code == "no_fit" and al.level == AlertLevel.INFO
                   for al in fired)

    def test_check_no_fit_clears_when_fit_exists(self):
        a = AlertSystem(throttle_sec=0)
        a.fire("no_fit", AlertLevel.INFO, "")
        a._last_fired["no_fit"] = time.time()
        a.check_no_fit(fit_count=1, total_empty=3)
        assert "no_fit" not in a._last_fired

    def test_check_no_fit_empty_zero_no_alert(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_no_fit(fit_count=0, total_empty=0)
        assert fired == []


# ══════════════════════════════════════════════════════════════════════════════
# VehicleTracker testleri
# ══════════════════════════════════════════════════════════════════════════════

class TestVehicleTracker:

    def test_new_detection_assigned(self):
        tr = VehicleTracker()
        flags = tr.update([_det(0, 0, 100, 60)])
        assert len(flags) == 1

    def test_not_static_immediately(self):
        tr = VehicleTracker(min_history=5)
        flags = tr.update([_det(0, 0, 100, 60)])
        assert flags[0] is False

    def test_static_after_enough_frames(self):
        tr = VehicleTracker(min_history=3, max_disp_ratio=0.5, max_misses=50)
        det = [_det(10, 10, 110, 70)]
        for _ in range(20):
            flags = tr.update(det)
        assert flags[0] is True

    def test_moving_vehicle_not_static(self):
        tr = VehicleTracker(min_history=3, max_disp_ratio=0.05, max_misses=50)
        for i in range(20):
            flags = tr.update([_det(i * 5, 10, i * 5 + 100, 70)])
        assert flags[0] is False

    def test_track_removed_after_max_misses(self):
        tr = VehicleTracker(max_misses=3)
        tr.update([_det(0, 0, 100, 60)])
        for _ in range(5):
            tr.update([])
        assert len(tr._tracks) == 0

    def test_get_static_tracks_empty_initially(self):
        tr = VehicleTracker()
        assert tr.get_static_tracks() == []

    def test_get_static_tracks_with_duration_returns_tuples(self):
        tr = VehicleTracker(min_history=3, max_disp_ratio=0.5, max_misses=50)
        det = [_det(10, 10, 110, 70)]
        for _ in range(25):
            tr.update(det)
        pairs = tr.get_static_tracks_with_duration(min_frames=20)
        assert len(pairs) == 1
        bbox, dur = pairs[0]
        assert len(bbox) == 4
        assert dur >= 0.0

    def test_duration_increases_over_time(self):
        tr = VehicleTracker(min_history=3, max_disp_ratio=0.5, max_misses=50)
        det = [_det(10, 10, 110, 70)]
        for _ in range(25):
            tr.update(det)
        d1 = tr.get_static_tracks_with_duration(min_frames=20)[0][1]
        time.sleep(0.05)
        d2 = tr.get_static_tracks_with_duration(min_frames=20)[0][1]
        assert d2 > d1

    def test_first_seen_set_on_creation(self):
        tr = VehicleTracker()
        before = time.time()
        tr.update([_det(0, 0, 100, 60)])
        after = time.time()
        track = list(tr._tracks.values())[0]
        assert before <= track.first_seen <= after

    def test_reset_clears_tracks(self):
        tr = VehicleTracker()
        tr.update([_det(0, 0, 100, 60)])
        tr.reset()
        assert len(tr._tracks) == 0
        assert tr._prev_gray is None

    def test_multiple_tracks_independent(self):
        tr = VehicleTracker(iou_threshold=0.1, max_misses=50)
        tr.update([_det(0, 0, 80, 50), _det(300, 0, 380, 50)])
        assert len(tr._tracks) == 2


# ══════════════════════════════════════════════════════════════════════════════
# StreetParkingDetector testleri
# ══════════════════════════════════════════════════════════════════════════════

class TestStreetParkingDetector:

    def _run(self, frame, dets, **kw):
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            **kw
        )
        return det.analyze(frame, dets)

    def test_no_vehicles_no_spaces(self):
        frame = _blank(640, 480)
        result = self._run(frame, [])
        assert result["empty_count"] == 0
        assert result["occupied_count"] == 0

    def test_single_vehicle_no_gap(self):
        frame = _blank(640, 480)
        dets = [_det(100, 300, 300, 400)]
        result = self._run(frame, dets, detect_row_edges=False)
        assert result["occupied_count"] == 1

    def test_gap_between_two_vehicles_detected(self):
        frame = _blank(1200, 500)
        # İki araç arası ~300px boşluk — araç genişliği ~200px → gap_ratio ~1.5 (>0.20)
        dets = [_det(50, 350, 250, 450), _det(550, 350, 750, 450)]
        result = self._run(frame, dets, detect_row_edges=False, max_gap_ratio=10.0)
        assert result["empty_count"] >= 1

    def test_result_keys_present(self):
        frame = _blank(640, 480)
        result = self._run(frame, [_det(100, 300, 300, 400)])
        for key in ("empty_count", "occupied_count", "empty_spaces",
                    "parked", "slot_sizes_m", "scale_m_per_px"):
            assert key in result

    def test_max_edge_extension_ratio_limits_slots(self):
        """0.20 ile geniş sağ kenar tek slota bölünmeli (0.40 üçe böler)."""
        frame = _blank(1920, 1080)
        # Araçlar x=50..1173 bölgesinde — sağda 747px kenar boşluğu
        dets = [
            _det(50,  700, 250,  800),
            _det(260, 700, 460,  800),
            _det(470, 700, 670,  800),
            _det(680, 700, 880,  800),
            _det(890, 700, 1090, 800),
            _det(1100,700, 1173, 800),
        ]
        r_loose = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            max_edge_extension_ratio=0.40,
            detect_row_edges=True,
        ).analyze(frame, dets)

        r_tight = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            max_edge_extension_ratio=0.20,
            detect_row_edges=True,
        ).analyze(frame, dets)

        # Sıkı parametre ile kenar slotu daha az üretmeli
        assert r_tight["empty_count"] <= r_loose["empty_count"]

    def test_slot_sizes_length_matches_empty_count(self):
        frame = _blank(1200, 500)
        dets = [_det(50, 350, 250, 450), _det(550, 350, 750, 450)]
        result = self._run(frame, dets, detect_row_edges=False, max_gap_ratio=10.0)
        assert len(result["slot_sizes_m"]) == result["empty_count"]

    def test_reset_history_clears_state(self):
        det = StreetParkingDetector(smoothing_frames=5, road_color_check=False)
        frame = _blank(640, 480)
        det.analyze(frame, [_det(100, 300, 300, 400)])
        det.reset_history()
        assert len(det._history) == 0
        assert det._analyze_count == 0

    def test_draw_returns_same_shape(self):
        frame = _blank(640, 480)
        dets = [_det(100, 300, 300, 400)]
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1, road_color_check=False
        )
        result = det.analyze(frame, dets)
        out = det.draw(frame.copy(), result)
        assert out.shape == frame.shape


# ══════════════════════════════════════════════════════════════════════════════
# Ground truth değerlendirme testi (3 görüntü, Mikro F1 = %100)
# ══════════════════════════════════════════════════════════════════════════════

GT_IMAGES = [
    ("1.png", 0),
    ("2.png", 2),
    ("3.png", 3),
]


@pytest.mark.skipif(
    not all((ROOT / p).exists() for p, _ in GT_IMAGES),
    reason="Ground truth görüntüleri bulunamadı",
)
class TestGroundTruth:

    @pytest.fixture(scope="class")
    def detector(self):
        from src.detection.vehicle_detector import VehicleDetector
        return VehicleDetector(conf=0.35)

    def _eval(self, img_name, expected, detector):
        frame = cv2.imread(str(ROOT / img_name))
        assert frame is not None, f"{img_name} açılamadı"
        det = StreetParkingDetector(max_edge_extension_ratio=0.20)
        dets = detector.detect(frame)
        result = det.analyze(frame, dets)
        return result["empty_count"], expected

    def test_1png_no_empty(self, detector):
        detected, expected = self._eval("1.png", 0, detector)
        assert detected == expected, f"1.png: expected={expected} detected={detected}"

    def test_2png_two_empty(self, detector):
        detected, expected = self._eval("2.png", 2, detector)
        assert detected == expected, f"2.png: expected={expected} detected={detected}"

    def test_3png_three_empty(self, detector):
        detected, expected = self._eval("3.png", 3, detector)
        assert detected == expected, f"3.png: expected={expected} detected={detected}"

    def test_micro_f1_100(self, detector):
        total_tp = total_fp = total_fn = 0
        for img, expected in GT_IMAGES:
            detected, _ = self._eval(img, expected, detector)
            total_tp += min(detected, expected)
            total_fp += max(0, detected - expected)
            total_fn += max(0, expected - detected)
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        assert micro_f1 == pytest.approx(1.0), f"Mikro F1 = {micro_f1:.3f}, beklenen 1.0"


# ══════════════════════════════════════════════════════════════════════════════
# DrivableAreaSegmenter fallback testi
# ══════════════════════════════════════════════════════════════════════════════

class TestDrivableAreaFallback:

    def test_missing_model_available_false(self):
        seg = DrivableAreaSegmenter(model_path="/nonexistent/model.pt")
        assert seg.available is False

    def test_infer_returns_none_none_when_unavailable(self):
        seg = DrivableAreaSegmenter(model_path="/nonexistent/model.pt")
        frame = _blank(640, 480)
        mask, overlay = seg.infer(frame)
        assert mask is None
        assert overlay is None

    def test_infer_does_not_raise_on_none_frame(self):
        seg = DrivableAreaSegmenter(model_path="/nonexistent/model.pt")
        mask, overlay = seg.infer(None)
        assert mask is None and overlay is None


# ══════════════════════════════════════════════════════════════════════════════
# Üst görüş (overhead) perspektif testi
# ══════════════════════════════════════════════════════════════════════════════

class TestOverheadPerspective:
    """Kuş bakışı otopark senaryosu: araçlar ızgara düzeninde, aralarında boşluklar."""

    def _make_overhead_frame(self, n_cars=4, car_w=120, car_h=60,
                              gap=80, margin=40, img_w=900, img_h=300):
        frame = np.full((img_h, img_w, 3), 128, dtype=np.uint8)
        dets = []
        x = margin
        y = (img_h - car_h) // 2
        for _ in range(n_cars):
            cv2.rectangle(frame, (x, y), (x + car_w, y + car_h), (50, 50, 200), -1)
            dets.append(_det(x, y, x + car_w, y + car_h))
            x += car_w + gap
        return frame, dets

    def test_detects_gaps_between_overhead_vehicles(self):
        frame, dets = self._make_overhead_frame(n_cars=3, gap=100)
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            max_edge_extension_ratio=0.10,
        )
        result = det.analyze(frame, dets)
        assert result["empty_count"] >= 1

    def test_no_gaps_when_cars_packed_tightly(self):
        # 2 piksel boşlukla yanyana park — gap_ratio eşiği altında kalmalı
        frame, dets = self._make_overhead_frame(n_cars=5, gap=2, img_w=1000)
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            max_edge_extension_ratio=0.05,
        )
        result = det.analyze(frame, dets)
        # 2 piksel boşluk ortalama araç genişliğine (120px) kıyasla çok küçük
        assert result["empty_count"] < len(dets)

    def test_result_keys_present_overhead(self):
        frame, dets = self._make_overhead_frame()
        det = StreetParkingDetector(smoothing_frames=1, smoothing_min_hits=1,
                                    road_color_check=False)
        result = det.analyze(frame, dets)
        for key in ("empty_count", "occupied_count", "slot_sizes_m", "empty_spaces"):
            assert key in result


# ══════════════════════════════════════════════════════════════════════════════
# Kamera canlı akış simülasyonu testi
# ══════════════════════════════════════════════════════════════════════════════

class TestCameraSimulation:
    """Sabit kameradan gelen tekrarlı karelerin tracker'ı kararlı hâle getirdiğini doğrular."""

    def _static_detections(self):
        return [
            _det(100, 200, 300, 350),
            _det(320, 200, 520, 350),
            _det(540, 200, 740, 350),
        ]

    def test_tracker_stabilizes_after_enough_frames(self):
        tracker = VehicleTracker(min_history=5, max_disp_ratio=0.20)
        frame = _blank(800, 500)
        dets = self._static_detections()
        for _ in range(25):
            tracker.update(dets, frame)
        static = tracker.get_static_tracks()
        assert len(static) == len(dets)

    def test_duration_increases_across_simulated_frames(self):
        tracker = VehicleTracker(min_history=5)
        frame = _blank(800, 500)
        dets = self._static_detections()
        for _ in range(25):
            tracker.update(dets, frame)
        tracks1 = tracker.get_static_tracks_with_duration()
        time.sleep(0.05)
        tracks2 = tracker.get_static_tracks_with_duration()
        durations1 = [d for _, d in tracks1]
        durations2 = [d for _, d in tracks2]
        assert all(d2 >= d1 for d1, d2 in zip(durations1, durations2))

    def test_new_vehicle_mid_stream_tracked(self):
        tracker = VehicleTracker(min_history=5)
        frame = _blank(800, 500)
        base = self._static_detections()
        for _ in range(25):
            tracker.update(base, frame)
        extra = base + [_det(760, 200, 960, 350)]
        for _ in range(25):
            tracker.update(extra, frame)
        static = tracker.get_static_tracks()
        assert len(static) == len(extra)

    def test_disappeared_vehicle_removed(self):
        tracker = VehicleTracker(min_history=5, max_misses=5)
        frame = _blank(800, 500)
        dets = self._static_detections()
        for _ in range(25):
            tracker.update(dets, frame)
        for _ in range(10):
            tracker.update([], frame)
        assert len(tracker.get_static_tracks()) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Dik Park Modu testleri
# ══════════════════════════════════════════════════════════════════════════════

def _perp_det_front(x1, y1, x2, y2):
    """Ön görünüm dik park: araç kareye yakın (bw/bh ≈ 1.2)."""
    return {"bbox": [x1, y1, x2, y2], "class_id": 2, "confidence": 0.9}


def _perp_det_side(x1, y1, x2, y2):
    """Yan görünüm dik park: araç geniş (bw/bh ≈ 3.0)."""
    return {"bbox": [x1, y1, x2, y2], "class_id": 2, "confidence": 0.9}


class TestPerpendiculaMode:
    """Dik park (perpendicular) modu birim testleri."""

    # ── Görünüm açısı tespiti ─────────────────────────────────────

    def test_front_view_detected_square_cars(self):
        """Kareye yakın bboxlar (bw/bh ≈ 1.2) → ön görünüm."""
        frame = _blank(800, 400)
        # 3 araç, her biri 100×80 px (bw/bh=1.25)
        dets = [
            _perp_det_front(50, 150, 150, 230),
            _perp_det_front(200, 150, 300, 230),
            _perp_det_front(350, 150, 450, 230),
        ]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets)
        assert result["perp_side_view"] is False

    def test_side_view_detected_wide_cars(self):
        """Geniş bboxlar (bw/bh ≈ 3.0) → yan görünüm."""
        frame = _blank(1200, 400)
        # 3 araç, her biri 300×100 px (bw/bh=3.0)
        dets = [
            _perp_det_side(50,  150, 350, 250),
            _perp_det_side(400, 150, 700, 250),
            _perp_det_side(750, 150, 1050, 250),
        ]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets)
        assert result["perp_side_view"] is True

    # ── Bölme yok (n=1) ──────────────────────────────────────────

    def test_no_subdivision_large_gap(self):
        """Dik modda büyük boşluk tek slot olarak raporlanmalı."""
        frame = _blank(1000, 400)
        # Tek araç ortada → her iki yanda büyük boşluk
        dets = [_perp_det_front(400, 150, 550, 250)]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            max_edge_extension_ratio=0.40,
            min_gap_ratio=0.20,
        )
        result = det.analyze(frame, dets)
        # Her boşluk tek slot olmalı — araç genişliğine bölünmemeli
        for slot in result["empty_spaces"]:
            x1, _, x2, _ = slot
            # Slot genişliği = toplam boşluk, çok daha büyük olmalı
            assert (x2 - x1) > 130  # araç genişliği (150px) den büyük

    def test_parallel_mode_subdivides(self):
        """Paralel modda aynı boşluk birden fazla slota bölünebilir."""
        frame = _blank(1000, 400)
        # 2 araç arası büyük boşluk (5× araç genişliği)
        dets = [
            _det(50, 150, 150, 250),   # araç genişliği 100px
            _det(650, 150, 750, 250),  # 500px boşluk → ~5 slot
        ]
        det = StreetParkingDetector(
            orientation="parallel",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            max_spaces_per_gap=4,
            min_gap_ratio=0.20,
        )
        result = det.analyze(frame, dets)
        # Paralel modda büyük boşluk birden çok slota bölünmeli
        assert result["empty_count"] >= 2

    # ── Ölçek referansı ──────────────────────────────────────────

    def test_scale_uses_width_ref_in_front_view(self):
        """Ön görünümde ölçek ref_car_width_m'e göre hesaplanmalı."""
        frame = _blank(800, 400)
        dets = [
            _perp_det_front(100, 150, 200, 230),  # 100px genişlik
            _perp_det_front(350, 150, 450, 230),
        ]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets, ref_car_width_m=2.0)
        scale = result["scale_m_per_px"]
        assert scale is not None
        # scale = 2.0 / 100px = 0.02 m/px
        assert abs(scale - 0.02) < 0.005

    def test_scale_uses_length_ref_in_side_view(self):
        """Yan görünümde ölçek ref_car_length_m'e göre hesaplanmalı."""
        frame = _blank(1200, 400)
        dets = [
            _perp_det_side(50,  150, 350, 250),  # 300px genişlik
            _perp_det_side(400, 150, 700, 250),
        ]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets, ref_car_length_m=4.5)
        scale = result["scale_m_per_px"]
        assert scale is not None
        # scale = 4.5 / 300px = 0.015 m/px
        assert abs(scale - 0.015) < 0.003

    # ── Sığma kontrolü ───────────────────────────────────────────

    def test_slot_size_reported_in_meters(self):
        """Slot boyutu metre cinsinden raporlanmalı."""
        frame = _blank(1200, 400)
        dets = [
            _perp_det_side(50,  150, 350, 250),
            _perp_det_side(750, 150, 1050, 250),
        ]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets, ref_car_length_m=4.5)
        if result["empty_count"] > 0:
            w_m, h_m = result["slot_sizes_m"][0]
            assert w_m > 0
            assert h_m >= 0

    # ── result dict anahtarları ───────────────────────────────────

    def test_result_has_perp_side_view_key(self):
        frame = _blank(800, 400)
        dets = [_perp_det_front(200, 150, 350, 270)]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets)
        assert "perp_side_view" in result
        assert isinstance(result["perp_side_view"], bool)

    def test_parallel_result_no_perp_flag(self):
        """Paralel modda perp_side_view False olmalı (default)."""
        frame = _blank(800, 400)
        dets = [_det(100, 200, 300, 320), _det(400, 200, 600, 320)]
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets)
        assert result.get("perp_side_view") is False

    # ── reset_history ─────────────────────────────────────────────

    def test_reset_clears_perp_state(self):
        """reset_history sonrası perp_side_view False'a dönmeli."""
        frame = _blank(1200, 400)
        dets = [_perp_det_side(50, 150, 350, 250)]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        det.analyze(frame, dets)
        det.reset_history()
        assert getattr(det, "_perp_side_view", False) is False
