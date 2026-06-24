import csv
import time
import math
from datetime import datetime
from pathlib import Path
import threading
import queue

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFrame, QFileDialog, QGridLayout, QMessageBox, QSlider,
    QScrollArea, QDoubleSpinBox, QTabWidget, QDialog
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt, QMetaObject, Q_ARG, pyqtSlot

from src.detection.vehicle_detector import VehicleDetector
from src.ui.alert_system import AlertSystem, AlertLevel, LEVEL_COLORS
from src.detection.parking_space_detector import _EMPTY_KEYWORDS
from src.detection.street_parking_detector import StreetParkingDetector
from src.detection.vehicle_tracker import VehicleTracker
from src.parking.learned_slot_memory import LearnedSlotMemory
from src.parking.occupancy_heatmap import OccupancyHeatmap
from src.detection.drivable_area import DrivableAreaSegmenter
from src.detection.depth_estimator import DepthEstimator
from src.detection.adaptive_slot_detector import AdaptiveSlotDetector
from src.detection.video_stabilizer import VideoStabilizer
from src.geometry import PerspectiveTransformer
from src.geometry import roi as _roi
from src.parking import slot_scoring as _scoring
from src.ui.ipm_dialog import IPMCalibrationDialog
from src.ui.roi_dialog import RoiSelectionDialog
from src.ui import overlays as _ov
from src.parking import ZoneLoader, ParkingAnalyzer
from src.parking import STATUS_AVAILABLE, STATUS_OCCUPIED, STATUS_FORBIDDEN

class AsyncVideoCapture:
    """cv2.VideoCapture wrapper that decodes frames in a background thread to prevent UI lag."""
    def __init__(self, cap):
        self.cap = cap
        self.running = True
        self.lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=1)
        
        # Read the first frame synchronously to ensure immediate availability
        try:
            ret, frame = self.cap.read()
        except Exception:
            ret, frame = False, None
            
        self.last_ret = ret
        self.last_frame = frame
        if ret:
            self.frame_queue.put((ret, frame))
            
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                except Exception:
                    ret, frame = False, None
                
                if not ret:
                    try:
                        self.frame_queue.put((False, None), timeout=1.0)
                    except Exception:
                        pass
                    self.running = False
                    break
                
                # Push to queue. With maxsize=1, this blocks until the previous frame is read,
                # ensuring perfect synchronization without skipping video frames.
                try:
                    self.frame_queue.put((ret, frame), timeout=0.5)
                except Exception:
                    pass
            else:
                time.sleep(0.01)
            time.sleep(0.001)

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def has_new_frame(self):
        return not self.frame_queue.empty()

    def read(self):
        if not self.frame_queue.empty():
            try:
                ret, frame = self.frame_queue.get_nowait()
                with self.lock:
                    self.last_ret = ret
                    self.last_frame = frame
                return ret, frame
            except Exception:
                pass
        with self.lock:
            return self.last_ret, self.last_frame

    def release(self):
        self.running = False
        # Empty queue to unblock any threads waiting to put items
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                break
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def set(self, propId, value):
        with self.lock:
            if self.cap is not None:
                # Clear queue before seeking
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except Exception:
                        break
                res = self.cap.set(propId, value)
                self.last_ret = False
                self.last_frame = None
                return res
        return False

    def get(self, propId):
        with self.lock:
            if self.cap is not None:
                return self.cap.get(propId)
        return 0.0

VEHICLE_CLASSES = {2: "Araba", 3: "Motosiklet", 5: "Otobüs", 7: "Kamyon"}
VEHICLE_COLORS_CV = {
    2: (0, 255, 0),
    3: (0, 165, 255),
    5: (0, 0, 255),
    7: (255, 0, 255),
}
VEHICLE_COLORS_HEX = {
    2: "#00ff00",
    3: "#ffa500",
    5: "#ff4444",
    7: "#ff00ff",
}


def make_section_label(text):
    lbl = QLabel(text)
    lbl.setFont(QFont("Arial", 8, QFont.Bold))
    lbl.setStyleSheet("""
        color: #3b82f6; 
        padding-top: 12px; 
        padding-bottom: 4px; 
        border-bottom: 1px solid #1e293b; 
        margin-bottom: 6px;
    """)
    return lbl


class StatCard(QFrame):
    def __init__(self, icon, label, color):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #1e293b;
                border-radius: 10px;
                border-left: 4px solid {color};
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        left = QVBoxLayout()
        self.count_lbl = QLabel("0")
        self.count_lbl.setFont(QFont("Arial", 18, QFont.Bold))
        self.count_lbl.setStyleSheet(f"color: {color}; border: none;")
        type_lbl = QLabel(f"{icon} {label}")
        type_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; border: none;")
        left.addWidget(self.count_lbl)
        left.addWidget(type_lbl)

        layout.addLayout(left)
        layout.addStretch()

    def set_count(self, n):
        self.count_lbl.setText(str(n))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Parking AI")
        self.setGeometry(100, 100, 1400, 860)
        self.setMinimumSize(1100, 700)

        self.detector = None
        self.cap = None
        self._last_time = time.time()
        self._fps = 0.0
        self.analyzer = None
        self._last_frame = None
        self._rgb_buf = None
        self.auto_detector = None

        self._conf_thresh = 0.20
        self._iou_thresh  = 0.15

        self._street_mode       = False
        self._min_gap_ratio     = 0.30   # yan kamera: park boşluğu ≈ 0.3-2.0× araç genişliği
        self._row_band_ratio    = 0.80   # tek sıra band genişliği
        self._ignore_top_ratio  = 0.20
        # Araç sığma kontrolü için boyutlar (metre)
        self._ref_car_length_m  = 4.5   # ölçek kalibrasyonu için referans araç uzunluğu
        self._user_car_length_m = 4.5   # kullanıcının aracının uzunluğu
        self._perp_mode         = False  # dik park modu
        self._ref_car_width_m   = 2.0   # dik modda ölçek referansı (araç eni)
        self._user_car_width_m  = 2.0   # kullanıcı aracının eni
        self._alert_occ_threshold = 80  # doluluk uyarı eşiği (%)

        self.street_detector: StreetParkingDetector | None = None
        # Performans: YOLO inference frame skipping.
        # Her N karede bir tam pipeline (YOLO + tracker + analiz);
        # ara karelerde son sonuçlar yeniden kullanılır — bbox'lar 1-2 frame
        # eski olabilir ama görsel akıcı kalır. N=3 → ~3x hızlanma.
        self._inference_period  = 3
        self._inference_tick    = 0
        self._last_detections: list = []
        self._last_obstacles:  list = []
        self._last_result: dict | None = None
        self._last_learned_status: list = []
        self._last_static_mask: list | None = None
        # Araç tracker (hareketli ≠ park etmiş ayrımı)
        self.vehicle_tracker    = VehicleTracker(
            history_len=60,
            min_history=5,
            max_disp_ratio=0.45,
            max_misses=3,
        )
        # Long-term öğrenilmiş slot kütüğü
        self.learned_slots      = LearnedSlotMemory()
        # Bir aracı "kalıcı slot" saymak için minimum statik tracker güncellemesi.
        # Frame skipping (period=3) nedeniyle bu değer × period kadar gerçek
        # frame'e karşılık gelir. 30 × 3 = 90 gerçek frame ≈ 3 sn @30 FPS.
        self._learn_min_frames  = 30
        # Zamansal araç olasılığı haritası — slot adaylarını filtreler ve
        # her slot için 0-1 güven skoru üretir.
        self.heatmap            = OccupancyHeatmap()
        self._slot_min_prob     = 0.04  # gerçek slot için minimum olasılık
        # Drivable area segmentasyonu (YOLOPv2). Model varsa yol yüzeyi
        # maskesi klasik LAB maskesinin yerini alır (çok daha doğru).
        # Ağır (~58 ms) → seyrek çalıştır + cache.
        try:
            self.drivable = DrivableAreaSegmenter("models/yolopv2.pt")
        except Exception:
            self.drivable = None
        self._drivable_period   = 12   # her 12 inference frame'de bir (~1.2 sn)
        self._drivable_tick     = 0
        self._last_drivable_mask = None
        self._drivable_lock = threading.Lock()
        self._drivable_thread = None

        # IPM (kuş bakışı) durumu
        self._ipm: PerspectiveTransformer | None = None
        self._ipm_show = False
        self._ipm_is_manual = False
        # Monoküler derinlik (model yoksa graceful → pasif; ilk kullanımda yüklenir)
        self.depth = DepthEstimator(allow_download=False)
        self._depth_filter = False   # derinlikle çapraz-açı slot filtreleme
        self._occlusion_filter = True # görüş engeli (ray-casting) slot filtreleme
        self._last_depth_map = None
        self._depth_overlay = False  # derinlik ısı haritası overlay'i
        self._depth_tick = 0
        self._depth_period = 5       # her 5 karede bir derinlik hesapla (cache)
        # Sesli asistan (Vosk — opsiyonel, model/mikrofon yoksa graceful)
        self._voice_active = False
        self.voice_assistant = None
        self._tts_speaker = None
        # Adaptif çizgi-ızgara modu (çizgi varsa ızgara, yoksa geometri)
        self._adaptive_mode = False
        self._adaptive = AdaptiveSlotDetector()
        self._last_adaptive_result = None
        self._orient_history = []
        self._orientation_is_manual = False
        # Video sabitleme (elde-çekim kaymasını IPM/ızgara için kompanze eder)
        self.stabilizer = VideoStabilizer()
        self._stabilize = False
        # İlgi bölgesi (ROI): None = tüm kare; poligon = sadece o alan işlenir
        self._roi_polygon = None

        self._log_file = None
        self._log_writer = None
        self._logging = False

        self._alerts = AlertSystem(throttle_sec=30.0)
        self._alerts.add_listener(self._on_alert)
        self._alert_dismiss_timer = QTimer(self)
        self._alert_dismiss_timer.setSingleShot(True)
        self._alert_dismiss_timer.timeout.connect(self._dismiss_alert)
        self._frame_count = 0
        
        # Gece Görüşü Modu (Night Vision) durumları
        self._night_vision = False
        self._night_vision_split = False
        self._night_vision_clip = 3.0

        # Otonom Vale Park (AVP) Simülasyonu durumları
        self._sim_active = False
        self._sim_target_idx = -1
        self._sim_car_x = 0.0
        self._sim_car_y = 0.0
        self._sim_car_yaw = 0.0
        self._sim_steering_angle = 0.0
        self._sim_step_name = ""
        self._sim_instruction = ""
        self._sim_path = []
        self._sim_step_idx = 0
        self._sim_points = []
        self._sim_timer = QTimer(self)
        self._sim_timer.timeout.connect(self._tick_simulation)
        self._sim_timer.setInterval(80) # 80ms per tick for smooth animation

        # Ekranda tıklamayı yakalamak için slot koordinatları
        self._schematic_slot_rects = []

        # Juri Sunum ve Analiz Modu durumları
        self._presentation_mode = False
        self._latencies = {
            "yolo": 0.0,
            "lane": 0.0,
            "slot": 0.0,
            "draw": 0.0,
            "total": 0.0,
            "fps": 0.0
        }
        self._fps_last_time = time.time()
        self._fps_frame_count = 0
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        # SLAM Haritalama durumları
        self._slam_active = False
        self._slam_cum_dx = 0.0
        self._slam_cum_dy = 0.0
        self._slam_slots = []

        # Kör Nokta Uyarı Sistemi (BSD) durumları
        self._bsd_active = False
        self._bsd_alert_frame = 0      # last frame that triggered alert
        self._bsd_flash_state = False  # toggles for blinking

        self._build_ui()
        self._load_model()

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Apply global dark slate design system stylesheet
        self.setStyleSheet("""
            QWidget {
                background-color: #090d16;
                color: #f8fafc;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                background-color: transparent;
            }
            QDoubleSpinBox {
                background-color: #0f172a;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px 6px;
                font-size: 11px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 16px;
                border: none;
                background: #1e293b;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background: #334155;
            }
            QSlider::groove:horizontal {
                border: 1px solid #334155;
                height: 4px;
                background: #0f172a;
                margin: 2px 0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #3b82f6;
                border: none;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #60a5fa;
            }
        """)

        self.video_label = QLabel("Kamera / Video Bekleniyor")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color:#0f172a; border-radius:12px; color:#475569; font-size:14px;"
        )
        self.video_label.setMinimumSize(900, 660)

        # Tab wrapping scroll helper
        def make_tab_scroll(tab_layout):
            container = QWidget()
            container.setLayout(tab_layout)
            container.setStyleSheet("background-color: #0f172a;")
            
            scroll = QScrollArea()
            scroll.setWidget(container)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setStyleSheet("""
                QScrollArea { border: none; background: #0f172a; }
                QScrollBar:vertical {
                    width: 4px; background: #0f172a; margin: 0;
                }
                QScrollBar::handle:vertical {
                    background: #334155; border-radius: 2px; min-height: 20px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            """)
            return scroll

        # Double spinbox helper
        def _spin_w(label_text, value, min_v, max_v, callback):
            w = QWidget()
            w.setStyleSheet("background:transparent;")
            row = QHBoxLayout(w)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color:#94a3b8; font-size:11px;")
            lbl.setFixedWidth(140)
            spin = QDoubleSpinBox()
            spin.setRange(min_v, max_v)
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setSuffix(" m")
            spin.setValue(value)
            spin.setFixedWidth(75)
            spin.valueChanged.connect(callback)
            row.addWidget(lbl)
            row.addWidget(spin)
            row.addStretch()
            return w

        def _reprocess():
            if self._last_frame is not None and self.cap is None:
                self._process_and_show(self._last_frame)

        # ── TAB 1: AKIŞ VE BESLEME ──
        feed_layout = QVBoxLayout()
        feed_layout.setSpacing(6)
        feed_layout.setContentsMargins(10, 10, 10, 10)
        
        # Başlık ve FPS
        title_row = QHBoxLayout()
        title = QLabel("Smart Parking AI")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #3b82f6; padding: 2px;")
        title.setAlignment(Qt.AlignCenter)
        title_row.addStretch()
        title_row.addWidget(title)
        title_row.addStretch()
        feed_layout.addLayout(title_row)
        
        fps_row = QHBoxLayout()
        self.fps_lbl = QLabel("FPS: —")
        self.fps_lbl.setStyleSheet("color: #64748b; font-size: 10px;")
        help_btn = QPushButton("ⓘ  Yardım")
        help_btn.setFixedHeight(20)
        help_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #3b82f6; border: none; "
            "font-size: 10px; text-decoration: underline; padding: 0; }"
            "QPushButton:hover { color: #60a5fa; }"
        )
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.clicked.connect(self._show_help)
        fps_row.addWidget(self.fps_lbl)
        fps_row.addStretch()
        fps_row.addWidget(help_btn)
        feed_layout.addLayout(fps_row)
        
        # ── Kaynak ──
        feed_layout.addWidget(make_section_label("KAYNAK SEÇİMİ"))
        src_grid = QGridLayout()
        src_grid.setSpacing(6)
        self.start_btn = self._btn("▶ Kamera", "#10b981", is_action=True)
        self.vid_btn   = self._btn("📂 Video",  "#0284c7", is_action=True)
        self.img_btn   = self._btn("🖼 Resim",  "#7c3aed", is_action=True)
        self.stop_btn  = self._btn("■ Durdur", "#ef4444", is_action=True)
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_camera)
        self.vid_btn.clicked.connect(self.load_video)
        self.img_btn.clicked.connect(self.load_image)
        self.stop_btn.clicked.connect(self.stop_feed)
        src_grid.addWidget(self.start_btn, 0, 0)
        src_grid.addWidget(self.vid_btn,   0, 1)
        src_grid.addWidget(self.img_btn,   1, 0)
        src_grid.addWidget(self.stop_btn,  1, 1)
        feed_layout.addLayout(src_grid)
        
        # ── Araç Tespiti ──
        feed_layout.addWidget(make_section_label("ARAÇ TESPİT DUYARLILIĞI"))
        feed_layout.addLayout(self._slider_row(
            "Conf:", 5, 95, int(self._conf_thresh * 100),
            lambda v: self._set_conf(v)
        ))
        
        # ── Park Durumu ──
        feed_layout.addWidget(make_section_label("PARK DURUMU"))
        self.park_cards = {
            STATUS_AVAILABLE: StatCard("🟢", "Boş",  "#10b981"),
            STATUS_OCCUPIED:  StatCard("🔴", "Dolu", "#ef4444"),
            STATUS_FORBIDDEN: StatCard("⚠️",  "Yasak","#f59e0b"),
        }
        park_grid = QGridLayout()
        park_grid.setSpacing(4)
        for i, card in enumerate(self.park_cards.values()):
            park_grid.addWidget(card, 0, i)
        feed_layout.addLayout(park_grid)
        
        self.occupancy_lbl = QLabel("")
        self.occupancy_lbl.setWordWrap(True)
        self.occupancy_lbl.setStyleSheet(
            "color: #f8fafc; font-size: 11px; font-weight: bold;"
            "background: #1e293b; border-radius: 6px; padding: 6px;"
        )
        self.occupancy_lbl.setAlignment(Qt.AlignCenter)
        feed_layout.addWidget(self.occupancy_lbl)
        
        self.recommendation_lbl = QLabel("")
        self.recommendation_lbl.setWordWrap(True)
        self.recommendation_lbl.setStyleSheet(
            "color: #fbbf24; font-size: 10px; font-weight: bold;"
            "background: #1e293b; border-radius: 6px; padding: 6px;"
        )
        self.recommendation_lbl.setAlignment(Qt.AlignCenter)
        feed_layout.addWidget(self.recommendation_lbl)
        
        # ── Araç Sayıları ──
        feed_layout.addWidget(make_section_label("ARAÇ SAYILARI"))
        counts_grid = QGridLayout()
        counts_grid.setSpacing(4)
        self.stat_cards = {}
        icons = {2: "🚗", 3: "🏍", 5: "🚌", 7: "🚛"}
        for i, (cls_id, name) in enumerate(VEHICLE_CLASSES.items()):
            card = StatCard(icons[cls_id], name, VEHICLE_COLORS_HEX[cls_id])
            self.stat_cards[cls_id] = card
            counts_grid.addWidget(card, i // 2, i % 2)
        feed_layout.addLayout(counts_grid)
        
        # ── Uyarı Eşiği ──
        feed_layout.addWidget(make_section_label("DOLULUK UYARI EŞİĞİ"))
        alert_box = QFrame()
        alert_box.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #334155; border-radius: 8px; }"
        )
        alert_box_layout = QVBoxLayout(alert_box)
        alert_box_layout.setContentsMargins(8, 6, 8, 6)
        alert_box_layout.setSpacing(3)
        occ_row = QHBoxLayout()
        occ_lbl = QLabel("Eşik:")
        occ_lbl.setFixedWidth(45)
        occ_lbl.setStyleSheet("color: #e2e8f0; font-size: 10px; font-weight: bold;")
        occ_slider = QSlider(Qt.Horizontal)
        occ_slider.setRange(10, 95)
        occ_slider.setValue(self._alert_occ_threshold)
        occ_slider.setFixedHeight(18)
        occ_val_lbl = QLabel(f"%{self._alert_occ_threshold}")
        occ_val_lbl.setFixedWidth(45)
        occ_val_lbl.setStyleSheet("color: #e2e8f0; font-size: 10px;")
        def _occ_changed(v):
            occ_val_lbl.setText(f"%{v}")
            self._alert_occ_threshold = v
        occ_slider.valueChanged.connect(_occ_changed)
        occ_row.addWidget(occ_lbl)
        occ_row.addWidget(occ_slider)
        occ_row.addWidget(occ_val_lbl)
        alert_box_layout.addLayout(occ_row)
        alert_box_layout.addWidget(self._info("Doluluk oranı bu eşiği aşınca uyarı tetiklenir."))
        feed_layout.addWidget(alert_box)
        
        # ── Uyarı Barı ──
        alert_row = QHBoxLayout()
        self.alert_bar = QLabel("")
        self.alert_bar.setWordWrap(True)
        self.alert_bar.setAlignment(Qt.AlignCenter)
        self.alert_bar.setStyleSheet(
            "border-radius: 6px; padding: 6px; font-size: 10px; font-weight: bold;"
        )
        self.alert_bar.setVisible(False)
        self._alert_close_btn = QPushButton("x")
        self._alert_close_btn.setFixedSize(18, 18)
        self._alert_close_btn.setStyleSheet(
            "background: #ffffff22; color: #fff; border: none; border-radius: 3px; font-size: 9px;"
        )
        self._alert_close_btn.clicked.connect(self._dismiss_alert)
        self._alert_close_btn.setVisible(False)
        alert_row.addWidget(self.alert_bar, stretch=1)
        alert_row.addWidget(self._alert_close_btn)
        feed_layout.addLayout(alert_row)
        
        # ── Durum / Log ──
        feed_layout.addWidget(make_section_label("DURUM VE GÜNLÜK"))
        self.status_lbl = QLabel("Hazır")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.status_lbl.setStyleSheet("color: #e2e8f0; font-size: 10px; padding: 4px;")
        status_scroll = QScrollArea()
        status_scroll.setWidget(self.status_lbl)
        status_scroll.setWidgetResizable(True)
        status_scroll.setFixedHeight(64)
        status_scroll.setStyleSheet("background: #0f172a; border: 1px solid #334155; border-radius: 6px;")
        feed_layout.addWidget(status_scroll)
        
        btn_row = QHBoxLayout()
        self.snapshot_btn = self._btn("📸 Snapshot", "#b45309", is_action=True)
        self.log_btn      = self._btn("⏺ Log", "#0f766e")
        self.log_btn.setCheckable(True)
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        self.log_btn.clicked.connect(self.toggle_logging)
        btn_row.addWidget(self.snapshot_btn)
        btn_row.addWidget(self.log_btn)
        feed_layout.addLayout(btn_row)
        
        self.eval_btn = self._btn("📊 Değerlendirme Çalıştır", "#7c3aed", is_action=True)
        self.eval_btn.clicked.connect(self._run_evaluation_ui)
        feed_layout.addWidget(self.eval_btn)
        
        feed_layout.addStretch()

        # ── TAB 2: ALGILAMA VE POLİTİKALAR ──
        detect_layout = QVBoxLayout()
        detect_layout.setSpacing(6)
        detect_layout.setContentsMargins(10, 10, 10, 10)
        
        # Otomatik Park Tespiti
        detect_layout.addWidget(make_section_label("OTOMATİK PARK TESPİTİ"))
        self.street_btn = self._btn("🚗  Otomatik Tespiti Aç", "#0e7490")
        self.street_btn.setCheckable(True)
        self.street_btn.clicked.connect(self._toggle_street_mode)
        self.street_btn.setFixedHeight(36)
        detect_layout.addWidget(self.street_btn)
        
        # Otomatik tespit ayarları
        strip_box = QFrame()
        strip_box.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #334155; border-radius: 8px; }"
        )
        strip_layout = QVBoxLayout(strip_box)
        strip_layout.setContentsMargins(8, 8, 8, 8)
        strip_layout.setSpacing(4)
        
        strip_layout.addWidget(self._info("Min Boşluk (iki araç arası):"))
        self._slider_row_into(
            strip_layout, "Boş:", 10, 150, int(self._min_gap_ratio * 100),
            lambda v: self._set_min_gap(v)
        )
        strip_layout.addWidget(self._info("Sıra Bandı (gruplama toleransı):"))
        self._slider_row_into(
            strip_layout, "Sıra:", 30, 150, int(self._row_band_ratio * 100),
            lambda v: self._set_row_tol(v)
        )
        strip_layout.addWidget(self._info("Üst Yoksay (gökyüzü yok sayılır):"))
        self._slider_row_into(
            strip_layout, "Yok:", 0, 60, int(self._ignore_top_ratio * 100),
            lambda v: self._set_ignore_top(v)
        )
        
        # Park yönü seçici
        _s_act = ("background:#3b82f6; color:#fff; border-radius:4px;"
                  " font-size:10px; font-weight:bold; padding:2px 6px; border:none;")
        _s_ina = ("background:#1e293b; color:#94a3b8; border-radius:4px;"
                  " font-size:10px; padding:2px 6px; border:1px solid #334155;")
        
        orient_row = QHBoxLayout()
        orient_lbl = QLabel("Yön:")
        orient_lbl.setStyleSheet("color:#94a3b8; font-size:11px;")
        orient_lbl.setFixedWidth(45)
        
        self._btn_par = QPushButton("Paralel")
        self._btn_par.setFixedHeight(22)
        self._btn_par.setStyleSheet(_s_act)
        
        self._btn_dik = QPushButton("Dik")
        self._btn_dik.setFixedHeight(22)
        self._btn_dik.setStyleSheet(_s_ina)

        def _set_par():
            self._perp_mode = False
            self._btn_par.setStyleSheet(_s_act)
            self._btn_dik.setStyleSheet(_s_ina)
            self._ref_len_w.setVisible(True)
            self._ref_wid_w.setVisible(False)
            self._usr_len_w.setVisible(True)
            self._usr_wid_w.setVisible(False)
            self._view_lbl.setVisible(False)
            if self._street_mode:
                self._rebuild_street_detector()
            if self._last_frame is not None and self.cap is None:
                self._process_and_show(self._last_frame)

        def _set_dik():
            self._perp_mode = True
            self._btn_par.setStyleSheet(_s_ina)
            self._btn_dik.setStyleSheet(_s_act)
            self._ref_len_w.setVisible(False)
            self._ref_wid_w.setVisible(True)
            self._usr_len_w.setVisible(False)
            self._usr_wid_w.setVisible(True)
            self._view_lbl.setText("Görünüm açısı hesaplanıyor...")
            self._view_lbl.setVisible(True)
            if self._street_mode:
                self._rebuild_street_detector()
            if self._last_frame is not None and self.cap is None:
                self._process_and_show(self._last_frame)

        def _manual_set_par():
            self._orientation_is_manual = True
            _set_par()

        def _manual_set_dik():
            self._orientation_is_manual = True
            _set_dik()

        self._btn_par.clicked.connect(_manual_set_par)
        self._btn_dik.clicked.connect(_manual_set_dik)
        self.set_parallel_mode = _set_par
        self.set_perpendicular_mode = _set_dik
        
        orient_row.addWidget(orient_lbl)
        orient_row.addWidget(self._btn_par)
        orient_row.addWidget(self._btn_dik)
        strip_layout.addLayout(orient_row)
        
        self._view_lbl = QLabel("")
        self._view_lbl.setStyleSheet("color:#64748b; font-size:9px; padding-left:2px;")
        self._view_lbl.setVisible(False)
        strip_layout.addWidget(self._view_lbl)
        
        detect_layout.addWidget(strip_box)
        
        # ── Araç Sığma Kontrolü ──
        detect_layout.addWidget(make_section_label("ARAÇ BOYUT SINIRLARI"))
        fit_box = QFrame()
        fit_box.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #334155; border-radius: 8px; }"
        )
        fit_layout = QVBoxLayout(fit_box)
        fit_layout.setContentsMargins(8, 8, 8, 8)
        fit_layout.setSpacing(6)
        
        self._ref_len_w = _spin_w(
            "Ref. araç uzunluğu:", self._ref_car_length_m, 1.0, 8.0,
            lambda v: setattr(self, "_ref_car_length_m", v) or _reprocess())
        self._ref_wid_w = _spin_w(
            "Ref. araç eni:", self._ref_car_width_m, 1.0, 4.0,
            lambda v: setattr(self, "_ref_car_width_m", v) or _reprocess())
        self._usr_len_w = _spin_w(
            "Aracın uzunluğu:", self._user_car_length_m, 1.0, 8.0,
            lambda v: setattr(self, "_user_car_length_m", v) or _reprocess())
        self._usr_wid_w = _spin_w(
            "Aracın eni:", self._user_car_width_m, 1.0, 4.0,
            lambda v: setattr(self, "_user_car_width_m", v) or _reprocess())
            
        fit_layout.addWidget(self._ref_len_w)
        fit_layout.addWidget(self._ref_wid_w)
        fit_layout.addWidget(self._usr_len_w)
        fit_layout.addWidget(self._usr_wid_w)
        self._ref_wid_w.setVisible(False)
        self._usr_wid_w.setVisible(False)
        
        fit_info = QLabel("Yeşil = Sığar  |  Kırmızı = Sığmaz")
        fit_info.setStyleSheet("color:#94a3b8; font-size:10px;")
        fit_layout.addWidget(fit_info)
        detect_layout.addWidget(fit_box)
        
        # ── Adaptif Çizgi-Izgara ──
        detect_layout.addWidget(make_section_label("BOYALI ŞERİT ALGILAMA"))
        self.adaptive_btn = self._btn("Çizgi-Izgara (Adaptif)", "#0e7490")
        self.adaptive_btn.setCheckable(True)
        self.adaptive_btn.clicked.connect(self._toggle_adaptive)
        detect_layout.addWidget(self.adaptive_btn)
        detect_layout.addWidget(self._info(
            "Boyalı şerit çizgileri varsa ızgara tabanlı tespiti etkinleştirir, yoksa geometriye düşer."
        ))
        
        # ── İlgi Bölgesi (ROI) ──
        detect_layout.addWidget(make_section_label("İLGİ BÖLGESİ (ROI)"))
        roi_row = QHBoxLayout()
        self.roi_btn = self._btn("ROI Seç", "#0e7490", is_action=True)
        self.roi_btn.setFixedHeight(28)
        self.roi_auto_btn = self._btn("Oto ROI", "#0e7490", is_action=True)
        self.roi_auto_btn.setFixedHeight(28)
        self.roi_clear_btn = self._btn("Temizle", "#475569", is_action=True)
        self.roi_clear_btn.setFixedHeight(28)
        self.roi_clear_btn.setEnabled(False)
        self.roi_btn.clicked.connect(self._select_roi)
        self.roi_auto_btn.clicked.connect(self._auto_roi)
        self.roi_clear_btn.clicked.connect(self._clear_roi)
        roi_row.addWidget(self.roi_btn)
        roi_row.addWidget(self.roi_auto_btn)
        roi_row.addWidget(self.roi_clear_btn)
        detect_layout.addLayout(roi_row)
        detect_layout.addWidget(self._info(
            "Park alanı poligonla işaretlenerek dış alandaki araç ve gürültüler elenir."
        ))
        
        detect_layout.addStretch()

        # ── TAB 3: GELİŞMİŞ VE GÖRÜNTÜ İŞLEME ──
        adv_layout = QVBoxLayout()
        adv_layout.setSpacing(6)
        adv_layout.setContentsMargins(10, 10, 10, 10)
        
        # Kuş Bakışı (IPM)
        adv_layout.addWidget(make_section_label("PERSPEKTİF DÜZELTME (IPM)"))
        ipm_box = QFrame()
        ipm_box.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #334155; border-radius: 8px; }"
        )
        ipm_layout = QVBoxLayout(ipm_box)
        ipm_layout.setContentsMargins(8, 8, 8, 8)
        ipm_layout.setSpacing(4)
        
        ipm_btn_row = QHBoxLayout()
        self.ipm_calib_btn = self._btn("Kalibre Et", "#7c3aed", is_action=True)
        self.ipm_calib_btn.setFixedHeight(30)
        self.ipm_show_btn = self._btn("Göster", "#475569")
        self.ipm_show_btn.setFixedHeight(30)
        self.ipm_show_btn.setCheckable(True)
        self.ipm_show_btn.setEnabled(False)
        self.ipm_calib_btn.clicked.connect(self._calibrate_ipm)
        self.ipm_show_btn.clicked.connect(self._toggle_ipm_show)
        ipm_btn_row.addWidget(self.ipm_calib_btn)
        ipm_btn_row.addWidget(self.ipm_show_btn)
        ipm_layout.addLayout(ipm_btn_row)
        
        self.ipm_auto_btn = self._btn("Oto IPM (çizgilerden)", "#0e7490", is_action=True)
        self.ipm_auto_btn.setFixedHeight(28)
        self.ipm_auto_btn.clicked.connect(self._auto_calibrate_ipm)
        ipm_layout.addWidget(self.ipm_auto_btn)
        
        self.ipm_status_lbl = QLabel("Kalibre edilmedi")
        self.ipm_status_lbl.setStyleSheet("color:#64748b; font-size:9px;")
        ipm_layout.addWidget(self.ipm_status_lbl)
        adv_layout.addWidget(ipm_box)
        
        # Sabitleme
        adv_layout.addWidget(make_section_label("VİDEO SABİTLEME"))
        self.stab_btn = self._btn("Sabitleme (IPM için)", "#0284c7")
        self.stab_btn.setCheckable(True)
        self.stab_btn.clicked.connect(self._toggle_stabilize)
        adv_layout.addWidget(self.stab_btn)
        adv_layout.addWidget(self._info(
            "Elde çekim titremelerini kompanze eder; IPM ve ızgarayı sabit tutar."
        ))
        
        # Derinlik & Engeller
        adv_layout.addWidget(make_section_label("DERİNLİK VE ENGELLER"))
        depth_row = QHBoxLayout()
        self.depth_btn = self._btn("Derinlik Filtresi", "#7c3aed")
        self.depth_btn.setCheckable(True)
        self.depth_btn.clicked.connect(self._toggle_depth_filter)
        self.depth_overlay_btn = self._btn("Isı Haritası", "#7c3aed")
        self.depth_overlay_btn.setCheckable(True)
        self.depth_overlay_btn.clicked.connect(self._toggle_depth_overlay)
        depth_row.addWidget(self.depth_btn)
        depth_row.addWidget(self.depth_overlay_btn)
        adv_layout.addLayout(depth_row)
        
        depth_ok = self.depth is not None and self.depth.available
        self.depth_status_lbl = QLabel(
            "Derinlik: aktif" if depth_ok else "Derinlik: ilk kullanımda yüklenir")
        self.depth_status_lbl.setStyleSheet("color:#64748b; font-size:9px;")
        adv_layout.addWidget(self.depth_status_lbl)
        
        self.occ_btn = self._btn("Görüş Filtresi AÇIK", "#3b82f6")
        self.occ_btn.setCheckable(True)
        self.occ_btn.setChecked(True)
        self.occ_btn.clicked.connect(self._toggle_occlusion_filter)
        adv_layout.addWidget(self.occ_btn)
        adv_layout.addWidget(self._info(
            "Ray-casting tekniğiyle görüş engeli arkasındaki alanları filtreler."
        ))
        
        # Gece Görüşü
        adv_layout.addWidget(make_section_label("GECE GÖRÜŞÜ (CLAHE)"))
        nv_box = QFrame()
        nv_box.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #334155; border-radius: 8px; }"
        )
        nv_layout = QVBoxLayout(nv_box)
        nv_layout.setContentsMargins(8, 8, 8, 8)
        nv_layout.setSpacing(6)
        
        self.nv_btn = self._btn("Gece Görüşünü Aç", "#2563eb")
        self.nv_btn.setCheckable(True)
        self.nv_btn.clicked.connect(self._toggle_night_vision)
        
        self.nv_split_btn = self._btn("Bölünmüş Ekran (Split)", "#475569")
        self.nv_split_btn.setCheckable(True)
        self.nv_split_btn.setEnabled(False)
        self.nv_split_btn.clicked.connect(self._toggle_nv_split)
        
        nv_btn_layout = QHBoxLayout()
        nv_btn_layout.setSpacing(4)
        nv_btn_layout.addWidget(self.nv_btn)
        nv_btn_layout.addWidget(self.nv_split_btn)
        nv_layout.addLayout(nv_btn_layout)
        
        nv_layout.addWidget(self._info("Kontrast Limit (Hassasiyet):"))
        self.nv_clip_slider = QSlider(Qt.Horizontal)
        self.nv_clip_slider.setRange(10, 80)
        self.nv_clip_slider.setValue(int(self._night_vision_clip * 10))
        self.nv_clip_slider.setFixedHeight(18)
        
        nv_clip_val_lbl = QLabel(f"{self._night_vision_clip:.1f}")
        nv_clip_val_lbl.setFixedWidth(34)
        nv_clip_val_lbl.setStyleSheet("color:#e2e8f0; font-size:10px;")
        
        def _nv_clip_changed(v):
            val = v / 10.0
            nv_clip_val_lbl.setText(f"{val:.1f}")
            self._night_vision_clip = val
            if self._last_frame is not None and self.cap is None:
                self._process_and_show(self._last_frame)
                
        self.nv_clip_slider.valueChanged.connect(_nv_clip_changed)
        
        nv_clip_row = QHBoxLayout()
        nv_clip_row.addWidget(self.nv_clip_slider)
        nv_clip_row.addWidget(nv_clip_val_lbl)
        nv_layout.addLayout(nv_clip_row)
        adv_layout.addWidget(nv_box)
        
        adv_layout.addStretch()

        # ── TAB 4: AKILLI SİSTEMLER VE SUNUM ──
        smart_layout = QVBoxLayout()
        smart_layout.setSpacing(6)
        smart_layout.setContentsMargins(10, 10, 10, 10)
        
        # Sesli Asistan
        smart_layout.addWidget(make_section_label("SESLİ ASİSTAN"))
        voice_box = QFrame()
        voice_box.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #334155; border-radius: 8px; }"
        )
        voice_layout = QVBoxLayout(voice_box)
        voice_layout.setContentsMargins(8, 8, 8, 8)
        voice_layout.setSpacing(6)
        
        self.voice_toggle_btn = self._btn("🎤 Sesli Komut", "#059669")
        self.voice_toggle_btn.setCheckable(True)
        self.voice_toggle_btn.clicked.connect(self._toggle_voice_assistant)
        voice_layout.addWidget(self.voice_toggle_btn)
        
        guide_btn = QPushButton("📋 Sesli Komut Rehberi (Göster)")
        guide_btn.setStyleSheet(
            "QPushButton { background-color: #334155; color: #e2e8f0; font-size: 10px; font-weight: bold; padding: 4px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #475569; }"
        )
        voice_layout.addWidget(guide_btn)
        
        guide_widget = QWidget()
        guide_grid = QGridLayout(guide_widget)
        guide_grid.setContentsMargins(0, 4, 0, 0)
        guide_grid.setSpacing(4)
        
        commands_info = [
            ("🔍 Boş Yer Bul", "bos / en yakin / bos yer"),
            ("🌙 Gece Görüşü", "gece"),
            ("🗺️ Kuş Bakışı", "kus bakisi / harita"),
            ("📐 Izgara Modu", "izgara / adaptif"),
            ("🌡️ Derinlik/Isı", "derinlik / isi"),
            ("📊 Metrik Analiz", "degerlendir"),
            ("🛑 Asistan Kapat", "dur / kapat"),
        ]
        
        for idx, (label, kws) in enumerate(commands_info):
            lbl_cmd = QLabel(label)
            lbl_cmd.setStyleSheet("color: #38bdf8; font-size: 10px; font-weight: bold;")
            lbl_kws = QLabel(f"➔ {kws}")
            lbl_kws.setStyleSheet("color: #94a3b8; font-size: 9px;")
            guide_grid.addWidget(lbl_cmd, idx, 0)
            guide_grid.addWidget(lbl_kws, idx, 1)
            
        guide_widget.setLayout(guide_grid)
        voice_layout.addWidget(guide_widget)
        
        guide_widget.setVisible(False)  # Collapsed by default to save sidebar space
        
        def toggle_guide():
            is_visible = guide_widget.isVisible()
            guide_widget.setVisible(not is_visible)
            guide_btn.setText("📋 Sesli Komut Rehberi (Gizle)" if not is_visible else "📋 Sesli Komut Rehberi (Göster)")
            
        guide_btn.clicked.connect(toggle_guide)
        
        smart_layout.addWidget(voice_box)
        
        # SLAM Haritalama
        smart_layout.addWidget(make_section_label("SLAM HARİTALAMA"))
        slam_box = QFrame()
        slam_box.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #334155; border-radius: 8px; }"
        )
        slam_layout = QVBoxLayout(slam_box)
        slam_layout.setContentsMargins(8, 8, 8, 8)
        slam_layout.setSpacing(6)
        
        self.slam_toggle_btn = self._btn("SLAM Modu AÇ", "#0891b2")
        self.slam_toggle_btn.setCheckable(True)
        self.slam_toggle_btn.clicked.connect(self._toggle_slam_mode)
        slam_layout.addWidget(self.slam_toggle_btn)
        
        self.slam_reset_btn = self._btn("Haritayı Sıfırla", "#ef4444", is_action=True)
        self.slam_reset_btn.clicked.connect(self._reset_slam_map)
        slam_layout.addWidget(self.slam_reset_btn)
        
        slam_desc = QLabel("Araç ilerledikçe dinamik olarak otopark haritasını çıkarır ve biriktirir.")
        slam_desc.setStyleSheet("color:#94a3b8; font-size:9px;")
        slam_desc.setWordWrap(True)
        slam_layout.addWidget(slam_desc)
        smart_layout.addWidget(slam_box)
        
        # Kör Nokta Uyarı BSD
        smart_layout.addWidget(make_section_label("KÖR NOKTA UYARISI (BSD)"))
        bsd_box = QFrame()
        bsd_box.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #334155; border-radius: 8px; }"
        )
        bsd_layout = QVBoxLayout(bsd_box)
        bsd_layout.setContentsMargins(8, 8, 8, 8)
        bsd_layout.setSpacing(6)
        
        self.bsd_toggle_btn = self._btn("BSD Modu AÇ", "#f97316")
        self.bsd_toggle_btn.setCheckable(True)
        self.bsd_toggle_btn.clicked.connect(self._toggle_bsd_mode)
        bsd_layout.addWidget(self.bsd_toggle_btn)
        
        bsd_desc = QLabel("Manevra esnasında kör noktalara giren nesneleri algılar ve uyarır.")
        bsd_desc.setStyleSheet("color:#94a3b8; font-size:9px;")
        bsd_desc.setWordWrap(True)
        bsd_layout.addWidget(bsd_desc)
        smart_layout.addWidget(bsd_box)
        
        # Jüri Sunum
        smart_layout.addWidget(make_section_label("JURİ SUNUM VE DEMO"))
        sunum_box = QFrame()
        sunum_box.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #e11d48; border-radius: 8px; }"
        )
        sunum_layout = QVBoxLayout(sunum_box)
        sunum_layout.setContentsMargins(8, 8, 8, 8)
        sunum_layout.setSpacing(6)
        
        self.sunum_toggle_btn = self._btn("Analiz Paneli AÇ", "#e11d48")
        self.sunum_toggle_btn.setCheckable(True)
        self.sunum_toggle_btn.clicked.connect(self._toggle_presentation_mode)
        sunum_layout.addWidget(self.sunum_toggle_btn)
        
        self.auto_demo_btn = self._btn("Otomatik Demoyu Başlat", "#be123c", is_action=True)
        self.auto_demo_btn.clicked.connect(self._start_auto_demo)
        sunum_layout.addWidget(self.auto_demo_btn)
        smart_layout.addWidget(sunum_box)
        
        smart_layout.addStretch()

        # ── TAB KONTROL ENTEGRASYONU ──
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 0px;
                background-color: #0f172a;
            }
            QTabBar::tab {
                background-color: #1e293b;
                color: #94a3b8;
                padding: 6px 8px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                font-size: 10px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3b82f6;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #334155;
                color: #e2e8f0;
            }
        """)
        
        self.sidebar_tabs.addTab(make_tab_scroll(feed_layout), "📁 Akış")
        self.sidebar_tabs.addTab(make_tab_scroll(detect_layout), "🚗 Tespit")
        self.sidebar_tabs.addTab(make_tab_scroll(adv_layout), "🛠 Gelişmiş")
        self.sidebar_tabs.addTab(make_tab_scroll(smart_layout), "💡 Akıllı")

        # Hidden variables for compatibility
        self.zone_lbl       = QLabel("")
        self.auto_det_lbl   = QLabel("")
        self.auto_det_btn   = QPushButton()
        self.auto_clear_btn = QPushButton()
        self.zone_btn       = QPushButton()

        panel_frame = QFrame()
        panel_frame.setFixedWidth(310)
        panel_frame.setStyleSheet("background-color:#0f172a; border-radius:12px;")
        frame_layout = QVBoxLayout(panel_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.sidebar_tabs)

        # QTabWidget for tabbed main view (Camera vs Schematic Map)
        self.main_tabs = QTabWidget()
        self.main_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 0px;
                background-color: #0f172a;
            }
            QTabBar::tab {
                background-color: #1e293b;
                color: #94a3b8;
                padding: 10px 20px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                font-size: 11px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3b82f6;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #334155;
                color: #e2e8f0;
            }
        """)

        # Tab 1: Kamera Görünümü
        self.main_tabs.addTab(self.video_label, "Kamera Görünümü")

        # Tab 2: 2D Kuş Bakışı Harita
        self.map_label = QLabel("IPM Kalibrasyonu veya Park Tespiti Bekleniyor")
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setStyleSheet(
            "background-color:#0f172a; border-radius:12px; color:#475569; font-size:14px;"
        )
        self.map_label.setMinimumSize(900, 660)
        self.map_label.mousePressEvent = self._on_map_clicked
        self.main_tabs.addTab(self.map_label, "2D Kuş Bakışı Harita")

        root = QHBoxLayout()
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        root.addWidget(self.main_tabs, stretch=1)
        root.addWidget(panel_frame)
        self.setLayout(root)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def _info(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#94a3b8; font-size:9px;")
        lbl.setWordWrap(True)
        return lbl

    def _slider_row_into(self, layout, label_text, lo, hi, init, callback):
        row = self._slider_row(label_text, lo, hi, init, callback)
        layout.addLayout(row)
        return row

    def _btn(self, text, color, height=32, is_action=False):
        b = QPushButton(text)
        b.setFixedHeight(height)
        b.setCursor(Qt.PointingHandCursor)
        if is_action:
            b.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {color};
                    opacity: 0.9;
                }}
                QPushButton:disabled {{
                    background-color: #1e293b;
                    color: #475569;
                }}
            """)
        else:
            b.setStyleSheet(f"""
                QPushButton {{
                    background-color: #1e293b;
                    color: #e2e8f0;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #334155;
                    border-color: #475569;
                }}
                QPushButton:checked {{
                    background-color: {color};
                    color: white;
                    border: 1px solid {color};
                }}
                QPushButton:disabled {{
                    background-color: #0f172a;
                    color: #475569;
                    border-color: #1e293b;
                }}
            """)
        return b

    def _slider_row(self, label_text, lo, hi, init, callback):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(45)
        lbl.setStyleSheet("color: #e2e8f0; font-size: 10px; font-weight: bold;")

        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(init)
        slider.setFixedHeight(18)

        val_lbl = QLabel(f"{init/100:.2f}")
        val_lbl.setFixedWidth(45)
        val_lbl.setStyleSheet("color: #e2e8f0; font-size: 10px;")

        def on_change(v):
            val_lbl.setText(f"{v/100:.2f}")
            callback(v)

        slider.valueChanged.connect(on_change)
        row.addWidget(lbl)
        row.addWidget(slider)
        row.addWidget(val_lbl)
        return row

    def _auto_enable_modes(self):
        """Kamera, video veya resim yüklendiğinde temel modları (Street mode, Adaptive mode, BSD, SLAM) otomatik açar."""
        if hasattr(self, "street_btn") and not self.street_btn.isChecked():
            self.street_btn.setChecked(True)
            self._toggle_street_mode()
        
        if hasattr(self, "adaptive_btn") and not self.adaptive_btn.isChecked():
            self.adaptive_btn.setChecked(True)
            self._toggle_adaptive()
            
        if hasattr(self, "bsd_toggle_btn") and not self.bsd_toggle_btn.isChecked():
            self.bsd_toggle_btn.setChecked(True)
            self._toggle_bsd_mode()

        if hasattr(self, "slam_toggle_btn") and not self.slam_toggle_btn.isChecked():
            self.slam_toggle_btn.setChecked(True)
            self._toggle_slam_mode()

    def _toggle_street_mode(self):
        self._street_mode = self.street_btn.isChecked()
        if self._street_mode:
            self.street_btn.setText("🚗  Otomatik Tespit ACIK")
            self._rebuild_street_detector()
        else:
            self.street_btn.setText("🚗  Otomatik Tespiti Ac")
            self.street_detector = None
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    # ── IPM (kuş bakışı) ──────────────────────────────────────────
    def _calibrate_ipm(self):
        if self._last_frame is None:
            QMessageBox.information(self, "IPM", "Önce bir görüntü/video yükleyin.")
            return
        calib = IPMCalibrationDialog.get_calibration(self._last_frame, self)
        if calib is None:
            return
        points, real_w, real_h = calib
        h, w = self._last_frame.shape[:2]
        try:
            self._ipm = PerspectiveTransformer.from_quad(
                points, out_w=w, out_h=h, real_w_m=real_w, real_h_m=real_h)
            self._ipm_is_default = False
            self._ipm_is_manual = True
        except Exception as e:
            QMessageBox.warning(self, "IPM", f"Kalibrasyon hatası:\n{e}")
            self._ipm = None
            self._ipm_is_manual = False
            return
        mpp = self._ipm.m_per_px
        self.ipm_status_lbl.setText(
            f"Kalibre edildi · {mpp:.4f} m/px" if mpp else "Kalibre edildi")
        # Sabitleme referansını kalibrasyon karesine sabitle: sonraki kareler
        # bu kareye hizalanır, böylece bu homografi geçerli kalır.
        self.stabilizer.set_reference(self._last_frame)
        self._activate_ipm_view()

    def _activate_ipm_view(self):
        """Kalibrasyon sonrası kuş bakışını otomatik aç ve göster."""
        self.ipm_show_btn.setEnabled(True)
        self._ipm_show = True
        self.ipm_show_btn.setChecked(True)
        self.ipm_show_btn.setText("Gizle")
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _auto_calibrate_ipm(self):
        """Yakınsayan çizgilerden otomatik IPM; başarısızsa manuele yönlendir."""
        if self._last_frame is None:
            QMessageBox.information(self, "Oto IPM", "Önce bir görüntü/video yükleyin.")
            return
        from src.geometry.auto_ipm import auto_calibrate, auto_calibrate_from_vehicles
        from src.ui.auto_ipm_dialog import AutoIPMDiagnosticsDialog
        h, w = self._last_frame.shape[:2]
        
        frame_to_use = self._last_frame
        if self._night_vision:
            frame_to_use = self._enhance_low_light(self._last_frame, clip_limit=self._night_vision_clip)

        # 1) Çizgi-yakınsama yöntemi
        tf, diag = auto_calibrate(frame_to_use, out_w=w, out_h=h,
                                 real_w_m=10.0, real_h_m=15.0, return_diagnostics=True)
        
        # 2) Başarısızsa araç-tabanlı yedek (araçları referans alır)
        if tf is None and self.detector is not None:
            dets = self.detector.detect(frame_to_use)
            dets = [d for d in dets if d.get("class_id") in VEHICLE_CLASSES]
            tf, diag = auto_calibrate_from_vehicles(
                dets, frame_to_use.shape, out_w=w, out_h=h,
                real_w_m=10.0, real_h_m=15.0, return_diagnostics=True)
                
        if tf is None:
            QMessageBox.information(
                self, "Oto IPM",
                "Otomatik kalibrasyon başarısız.\n"
                "Düz cepheden çekimde yeterli perspektif ipucu olmayabilir; "
                "açılı/yandan bir kare deneyin veya 'Kalibre Et' ile manuel yapın.")
            return

        # Görsel Önizleme Diyaloğunu Aç
        dlg = AutoIPMDiagnosticsDialog(self._last_frame, tf, diag, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._ipm = tf
            self._ipm_is_default = False
            self._ipm_is_manual = False
            mpp = self._ipm.m_per_px
            self.ipm_status_lbl.setText(
                f"Oto kalibre · {mpp:.4f} m/px" if mpp else "Oto kalibre edildi")
            self.stabilizer.set_reference(self._last_frame)
            self._activate_ipm_view()

    def _toggle_ipm_show(self):
        self._ipm_show = self.ipm_show_btn.isChecked()
        self.ipm_show_btn.setText("Gizle" if self._ipm_show else "Goster")
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _ensure_depth_model(self) -> bool:
        """Derinlik modelini gerektiğinde (ilk kullanımda) yükle. Döner: available."""
        if self.depth is not None and self.depth.available:
            return True
        from PyQt5.QtWidgets import QApplication
        self.depth_status_lbl.setText("Derinlik modeli yükleniyor…")
        QApplication.processEvents()
        try:
            self.depth = DepthEstimator(allow_download=True)
        except Exception:
            self.depth = DepthEstimator(allow_download=False)
        ok = self.depth is not None and self.depth.available
        self.depth_status_lbl.setText(
            "Derinlik: aktif" if ok else "Derinlik modeli yüklenemedi (internet/timm?)")
        return ok

    def _toggle_depth_filter(self):
        want = self.depth_btn.isChecked()
        if want and not self._ensure_depth_model():
            self.depth_btn.setChecked(False)
            return
        self._depth_filter = want
        self.depth_btn.setText("Derinlik AÇIK" if self._depth_filter
                               else "Derinlik Filtresi")
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _toggle_depth_overlay(self):
        want = self.depth_overlay_btn.isChecked()
        if want and not self._ensure_depth_model():
            self.depth_overlay_btn.setChecked(False)
            return
        self._depth_overlay = want
        self.depth_overlay_btn.setText("Isı Haritası AÇIK" if self._depth_overlay
                                       else "Isı Haritası")
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    # ── Sesli Asistan (C1) ────────────────────────────────────────
    def _toggle_voice_assistant(self):
        if not self._voice_active:
            from src.voice import VoiceAssistant, TTSSpeaker
            model_path = "models/vosk-model-small-tr-0.3"
            self.voice_assistant = VoiceAssistant(model_path, self._on_voice_command)
            if not self.voice_assistant.available:
                self.voice_toggle_btn.setChecked(False)
                QMessageBox.information(
                    self, "Sesli Asistan",
                    "Sesli asistan kullanılamıyor.\n"
                    "Gerekli: pip install vosk sounddevice + Türkçe model\n"
                    f"('{model_path}' klasörü) + mikrofon.")
                return
            if not hasattr(self, "_tts_speaker") or self._tts_speaker is None:
                self._tts_speaker = TTSSpeaker()
            self.voice_assistant.start()
            self._voice_active = True
            self.voice_toggle_btn.setText("🎤 Ses KAPAT")
            self.status_lbl.setText("🎤 Sesli asistan açık — komut bekliyor…")
        else:
            if self.voice_assistant:
                self.voice_assistant.stop()
            self._voice_active = False
            self.voice_toggle_btn.setText("🎤 Ses AÇ")
            self.status_lbl.setText("Sesli asistan kapatıldı.")

    def _on_voice_command(self, cmd: str, raw_text: str):
        """Asistan thread'inden gelen komutu Qt ana thread'ine marshal et."""
        QMetaObject.invokeMethod(self, "_execute_voice_cmd",
                                 Qt.QueuedConnection, Q_ARG(str, cmd), Q_ARG(str, raw_text))

    @pyqtSlot(str, str)
    def _execute_voice_cmd(self, cmd: str, raw_text: str):
        """Sesli komutu yalnızca MEVCUT aksiyonlara eşle (ana thread)."""
        from src.voice.voice_assistant import CMD_RESPONSES
        self.status_lbl.setText(f"🎤 Algılanan: '{raw_text}' ➔ Komut: {cmd}")
        if cmd == "find_empty":
            if not self.adaptive_btn.isChecked():
                self.adaptive_btn.setChecked(True)
                self._toggle_adaptive()
            rec_text = self.recommendation_lbl.text().replace("⭐", "").replace("·", ",").strip()
            if rec_text:
                CMD_RESPONSES["find_empty"] = f"Park yerleri analiz edildi. {rec_text}"
            else:
                CMD_RESPONSES["find_empty"] = "Sizin için uygun bir boş park yeri bulunamadı."
        elif cmd == "toggle_adaptive":
            self.adaptive_btn.toggle()
            self._toggle_adaptive()
        elif cmd == "toggle_depth":
            self.depth_overlay_btn.toggle()
            self._toggle_depth_overlay()
        elif cmd == "toggle_night":
            if hasattr(self, "nv_btn"):
                self.nv_btn.toggle()
                self._toggle_night_vision()
        elif cmd == "toggle_ipm":
            if self.ipm_show_btn.isEnabled():
                self.ipm_show_btn.toggle()
                self._toggle_ipm_show()
        elif cmd == "evaluate":
            self._run_evaluation_ui()
        elif cmd == "stop":
            if self._depth_overlay:
                self.depth_overlay_btn.setChecked(False)
                self._toggle_depth_overlay()
            if self._voice_active:
                self.voice_toggle_btn.setChecked(False)
                self._toggle_voice_assistant()
        tts = getattr(self, "_tts_speaker", None)
        if tts and cmd in CMD_RESPONSES:
            tts.speak(CMD_RESPONSES[cmd])

    def _toggle_occlusion_filter(self):
        self._occlusion_filter = self.occ_btn.isChecked()
        self.occ_btn.setText("Görüş Filtresi AÇIK" if self._occlusion_filter
                             else "Görüş Engeli Filtresi")
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _toggle_adaptive(self):
        self._adaptive_mode = self.adaptive_btn.isChecked()
        self.adaptive_btn.setText("Çizgi-Izgara AÇIK" if self._adaptive_mode
                                  else "Çizgi-Izgara (Adaptif)")
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _toggle_stabilize(self):
        self._stabilize = self.stab_btn.isChecked()
        self.stab_btn.setText("Sabitleme AÇIK" if self._stabilize
                              else "Sabitleme (IPM icin)")
        # Referansı mevcut kareye ayarla (varsa)
        if self._stabilize and self._last_frame is not None:
            self.stabilizer.set_reference(self._last_frame)

    def _enhance_low_light(self, frame, clip_limit=3.0):
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            if not hasattr(self, "_clahe") or getattr(self, "_clahe_limit", None) != clip_limit:
                self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                self._clahe_limit = clip_limit
            cl = self._clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced
        except Exception:
            return frame

    def _toggle_night_vision(self):
        self._night_vision = self.nv_btn.isChecked()
        if self._night_vision:
            self.nv_btn.setText("Gece Görüşü AÇIK")
            self.nv_btn.setStyleSheet("background-color: #059669; color: white; border-radius: 8px; font-size: 12px; font-weight: bold;")
            self.nv_split_btn.setEnabled(True)
            self.nv_split_btn.setStyleSheet("background-color: #3b82f6; color: white; border-radius: 8px; font-size: 12px; font-weight: bold;")
        else:
            self.nv_btn.setText("Gece Görüşünü Aç")
            self.nv_btn.setStyleSheet("background-color: #2563eb; color: white; border-radius: 8px; font-size: 12px; font-weight: bold;")
            self.nv_split_btn.setEnabled(False)
            self.nv_split_btn.setStyleSheet("background-color: #334155; color: #64748b; border-radius: 8px; font-size: 12px; font-weight: bold;")
            self._night_vision_split = False
            self.nv_split_btn.setChecked(False)
            self.nv_split_btn.setText("Bölünmüş Ekran (Split)")
            
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _toggle_nv_split(self):
        self._night_vision_split = self.nv_split_btn.isChecked()
        if self._night_vision_split:
            self.nv_split_btn.setText("Bölünmüş Ekran AÇIK")
            self.nv_split_btn.setStyleSheet("background-color: #059669; color: white; border-radius: 8px; font-size: 12px; font-weight: bold;")
        else:
            self.nv_split_btn.setText("Bölünmüş Ekran (Split)")
            self.nv_split_btn.setStyleSheet("background-color: #3b82f6; color: white; border-radius: 8px; font-size: 12px; font-weight: bold;")
            
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _select_roi(self):
        if self._last_frame is None:
            QMessageBox.information(self, "ROI", "Önce bir görüntü/video yükleyin.")
            return
        poly = RoiSelectionDialog.get_roi(self._last_frame, self)
        if poly is None:
            return
        self._roi_polygon = poly
        self.roi_clear_btn.setEnabled(True)
        self.status_lbl.setText(f"ROI ayarlandı ({len(poly)} nokta).")
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _auto_roi(self):
        """Tespit edilen araçların zarfından otomatik ROI üret."""
        if self._last_frame is None:
            QMessageBox.information(self, "Oto ROI", "Önce bir görüntü/video yükleyin.")
            return
        if self.detector is None:
            QMessageBox.information(self, "Oto ROI", "Araç modeli yüklü değil.")
            return

        # 1) Yol maskesi hesaplanmadıysa öncelikle hesapla
        frame_to_use = self._last_frame
        if self._night_vision:
            frame_to_use = self._enhance_low_light(self._last_frame, clip_limit=self._night_vision_clip)

        if self._last_drivable_mask is None and self.drivable is not None and self.drivable.available:
            try:
                da_mask, _ = self.drivable.infer(frame_to_use)
                if da_mask is not None:
                    kx = max(10, int(frame_to_use.shape[1] * 0.018))
                    ky = max(5, int(frame_to_use.shape[0] * 0.010))
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
                    da_mask = cv2.dilate(da_mask, kernel, iterations=1)
                    self._last_drivable_mask = da_mask
            except Exception:
                pass

        dets = self.detector.detect(frame_to_use)
        # 2) Adaptif güven filtresini uygula (yol dışı/hatalı tespitleri eler)
        dets = self._apply_adaptive_confidence_filter(dets, frame_to_use)
        dets = [
            d for d in dets
            if d.get("class_name") in {"car", "motorcycle", "bus", "truck"}
        ]
        poly = _roi.auto_roi_from_detections(dets, self._last_frame.shape)
        if poly is None:
            QMessageBox.information(
                self, "Oto ROI",
                "Araç bulunamadı; otomatik ROI üretilemedi.\n"
                "Manuel 'ROI Sec' ile çizebilirsiniz.")
            return
        self._roi_polygon = poly
        self.roi_clear_btn.setEnabled(True)
        self.status_lbl.setText(f"Oto ROI ({len(poly)} nokta).")
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _clear_roi(self):
        self._roi_polygon = None
        self.roi_clear_btn.setEnabled(False)
        self.status_lbl.setText("ROI temizlendi.")
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _get_corrected_detections(self, frame, detections):
        if not self.street_detector or not self.street_detector.perspective_correction:
            return detections
        if self.street_detector.orientation == "parallel":
            return detections
        import copy
        import math
        corrected = copy.deepcopy(detections)
        h, w = frame.shape[:2]
        f = float(w)  # focal length ≈ image width
        for det in corrected:
            cls_id = det.get("class_id")
            if cls_id not in VEHICLE_CLASSES:
                continue
            if det.get("perspective_corrected", False):
                continue
            
            # Save the raw bbox before cropping
            det["raw_bbox"] = list(det["bbox"])
            
            x1, y1, x2, y2 = map(int, det["bbox"])
            px_w = x2 - x1
            px_h = y2 - y1
            if px_w < 5:
                continue
            cx = (x1 + x2) / 2.0

            cname = VEHICLE_CLASSES.get(cls_id, "car")

            real_len = StreetParkingDetector._real_dim_for(cname, False, self._ref_car_length_m, self._ref_car_width_m)
            real_wid = StreetParkingDetector._real_dim_for(cname, True, self._ref_car_length_m, self._ref_car_width_m)

            # Aracın ön/arka yüz genişliğinin, toplam bbox genişliğine oranını hesapla.
            # Bu oran, aracın bakış açısına göre ne kadar yan profil gösterdiğini belirler.
            #
            # Yöntem: YOLO bbox en-boy oranı (aspect ratio) ile gerçek araç boyutlarını
            # karşılaştır. Araç tam karşıdan görünüyorsa bbox_w/bbox_h ≈ real_wid/real_h.
            # Araç yandan görünüyorsa bbox_w/bbox_h ≈ real_len/real_h.
            # Bu ikisi arasındaki interpolasyon ile "ne kadar yan görünüyor" bulunur.

            # Aspect ratio tabanlı yönelim tahmini
            aspect = px_w / max(1, px_h)
            # Araç yüksekliği referansı: SUV ~1.7m, sedan ~1.5m, ortalama ~1.6m
            real_h = 1.6
            front_aspect = real_wid / real_h   # ~1.25  (ön görünüm)
            side_aspect = real_len / real_h     # ~2.81  (yan görünüm)

            if side_aspect <= front_aspect:
                # Garip durum (çok kısa araç?), kırpma yapma
                det["perspective_corrected"] = True
                continue

            # side_ratio: 0.0 = tam karşıdan, 1.0 = tam yandan
            side_ratio = max(0.0, min(1.0,
                (aspect - front_aspect) / (side_aspect - front_aspect)
            ))

            # Tamamen yandan görünümde (side_ratio > 0.85) bbox kenarları
            # aracın gerçek ön/arka tamponlarıdır → kırpma yapma.
            if side_ratio > 0.85:
                det["perspective_corrected"] = True
                continue

            # Beklenen ön profil piksel genişliği
            expected_front_px = px_w * (1.0 - side_ratio) + px_w * (real_wid / real_len) * side_ratio
            # Daha basit formül: ön yüz oranı
            front_fraction = expected_front_px / px_w if px_w > 0 else 1.0
            front_fraction = max(0.35, min(1.0, front_fraction))  # en az %35, en fazla %100

            # Kırpılacak toplam piksel
            total_crop = int(px_w * (1.0 - front_fraction))

            if total_crop > 2:
                # Kırpma yönü: Aracın merkezinin kamera merkezine göre konumuna göre
                # belirlenir. Araç solda ise yan tarafı sağda kalır → sağdan daha çok kırp.
                # Araç sağda ise yan tarafı solda kalır → soldan daha çok kırp.
                # Ortadaysa simetrik kırp.
                offset_ratio = (cx - w / 2.0) / (w / 2.0)  # -1..+1
                # offset_ratio < 0 → araç solda → sağ taraf yan (sağdan kırp)
                # offset_ratio > 0 → araç sağda → sol taraf yan (soldan kırp)
                # bias: 0.0 = hep sağdan, 1.0 = hep soldan
                bias = max(0.0, min(1.0, 0.5 + offset_ratio * 0.5))
                crop_left = int(total_crop * bias)
                crop_right = total_crop - crop_left

                new_x1 = x1 + crop_left
                new_x2 = x2 - crop_right
                if new_x2 - new_x1 >= 5:
                    x1 = new_x1
                    x2 = new_x2

            det["bbox"] = [float(x1), float(y1), float(x2), float(y2)]
            det["perspective_corrected"] = True
        return corrected

    def _draw_adaptive(self, frame, out, detections, is_inference_frame=True,
                       static_mask=None, external_road_mask=None):
        """Adaptif slot tespitini çalıştır, poligonları çiz, sayımları döndür.

        Çizgi varsa kuş bakışı ızgara, yoksa geometri yöntemi. IPM kalibreyse
        kuş bakışında çalışıp slotları kaynak perspektife geri haritalar.
        Döner: (available, occupied).
        """
        # Araç kutularını de-rotasyon ile düzeltilmiş haliyle çiz
        h, w = frame.shape[:2]
        car_req = self._user_car_width_m if self._perp_mode else self._user_car_length_m
        corrected_detections = self._get_corrected_detections(frame, detections)
        for idx, det in enumerate(corrected_detections):
            cls_id = det.get("class_id")
            if cls_id not in VEHICLE_CLASSES:
                continue
            # Mükerrer/çift kutu çizimini önlemek için statik (park halinde) araçların 
            # etrafına yeşil 2D kutu çizmeyelim, sadece hareketli araçları çizelim.
            is_static = False
            if static_mask is not None and idx < len(static_mask):
                is_static = static_mask[idx]
            if is_static:
                continue
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = VEHICLE_COLORS_CV.get(cls_id, (0, 255, 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if is_inference_frame or not hasattr(self, "_cached_adaptive_data") or self._cached_adaptive_data is None:
            t0 = time.perf_counter()
            res = self._adaptive.analyze(
                frame, corrected_detections,
                ipm=None if getattr(self, "_ipm_is_default", False) else self._ipm,
                static_mask=static_mask,
                external_road_mask=external_road_mask,
                obstacles=self._last_obstacles,
                ref_car_length_m=self._ref_car_length_m,
                ref_car_width_m=self._ref_car_width_m,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._last_adaptive_result = res

            empty_polys = res["empty_polys"]
            sizes_m = res.get("empty_sizes_m") or [None] * len(empty_polys)

            # Derinlik filtresi (çapraz-açı slot elenmesi)
            if self._depth_filter and res["empty_spaces"]:
                parked_boxes = [d["bbox"] for d in corrected_detections if d.get("class_id") in VEHICLE_CLASSES]
                keep = self._depth_keep_indices(frame, res["empty_spaces"], parked_boxes)
                if len(keep) != len(empty_polys):
                    empty_polys = [empty_polys[i] for i in keep]
                    if sizes_m:
                        sizes_m = [sizes_m[i] for i in keep]
                    res["empty_spaces"] = [res["empty_spaces"][i] for i in keep]
                    res["empty_count"] = len(keep)

            # Görüş engeli (occlusion) filtresi
            if res["empty_spaces"]:
                parked_boxes = [d["bbox"] for d in corrected_detections if d.get("class_id") in VEHICLE_CLASSES]
                from src.geometry.roi import filter_occluded_slots
                keep = filter_occluded_slots(res["empty_spaces"], parked_boxes, frame.shape)
                if len(keep) != len(empty_polys):
                    empty_polys = [empty_polys[i] for i in keep]
                    if sizes_m:
                        sizes_m = [sizes_m[i] for i in keep]
                    res["empty_spaces"] = [res["empty_spaces"][i] for i in keep]
                    res["empty_count"] = len(keep)

            car_req = self._user_car_width_m if self._perp_mode else self._user_car_length_m
            fit_flags = []
            for i in range(len(empty_polys)):
                sm = sizes_m[i] if i < len(sizes_m) else None
                fit_flags.append(bool(sm) and sm[0] >= car_req)

            empty_polys_drawn = [p.astype(np.int32).reshape(-1, 1, 2) for p in empty_polys]
            occupied_polys_drawn = [p.astype(np.int32).reshape(-1, 1, 2) for p in res["occupied_polys"]]

            self._cached_adaptive_data = {
                "res": res,
                "empty_polys": empty_polys,
                "empty_polys_drawn": empty_polys_drawn,
                "occupied_polys_drawn": occupied_polys_drawn,
                "sizes_m": sizes_m,
                "fit_flags": fit_flags,
                "elapsed_ms": elapsed_ms,
                "available": res["empty_count"],
                "occupied": res["occupied_count"],
            }
        else:
            cache = self._cached_adaptive_data
            res = cache["res"]
            empty_polys = cache["empty_polys"]
            empty_polys_drawn = cache["empty_polys_drawn"]
            occupied_polys_drawn = cache["occupied_polys_drawn"]
            sizes_m = cache["sizes_m"]
            fit_flags = cache["fit_flags"]
            elapsed_ms = cache["elapsed_ms"]
            available = cache["available"]
            occupied = cache["occupied"]

        COLOR_FIT   = (0, 200, 80)    # yeşil  — boş ve sığar
        COLOR_NOFIT = (0, 165, 255)   # turuncu — boş ama sığmaz
        COLOR_OCC   = (0, 60, 200)    # kırmızı — dolu

        overlay = out.copy()
        for i, poly_d in enumerate(empty_polys_drawn):
            col = COLOR_FIT if fit_flags[i] else (
                COLOR_NOFIT if sizes_m[i] else COLOR_FIT)
            cv2.fillPoly(overlay, [poly_d], col)
        for poly_d in occupied_polys_drawn:
            cv2.fillPoly(overlay, [poly_d], COLOR_OCC)
        cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)

        # Collect labels for each empty slot to draw them in a stacked HUD card format to prevent overlap
        empty_slot_labels = [[] for _ in range(len(empty_polys))]
        
        for i, poly in enumerate(empty_polys):
            sm = sizes_m[i]
            label_color = (0, 220, 80) if (sm is None or fit_flags[i]) else COLOR_NOFIT
            type_prefix = "BOS"

            if sm is None:
                empty_slot_labels[i].append((type_prefix, label_color))
            elif fit_flags[i]:
                empty_slot_labels[i].append((f"{type_prefix} {sm[0]:.1f}m", label_color))
            else:
                empty_slot_labels[i].append((f"SIGMAZ {sm[0]:.1f}m", COLOR_NOFIT))
            _ov.draw_pseudo_3d(out, poly, label_color if (sm is None or fit_flags[i]) else COLOR_NOFIT, lift=0)
            
        for poly_d in occupied_polys_drawn:
            cv2.polylines(out, [poly_d], True, COLOR_OCC, 2)

        # En yakın UYGUN (boş + sığan) slota yönlendirme. Ölçek yoksa tüm boşlar
        # aday kabul edilir (sığma kontrolü yapılamaz).
        h, w = out.shape[:2]
        origin = (w / 2.0, h - 12.0)
        has_size = any(s is not None for s in sizes_m)
        candidates = [p for i, p in enumerate(empty_polys)
                      if (fit_flags[i] or not has_size)]
        near = _ov.nearest_empty(candidates, origin)
        
        near_idx = None
        if near is not None:
            near_i_cand, centroid, dist_px = near
            for idx, poly in enumerate(empty_polys):
                if np.array_equal(poly, candidates[near_i_cand]):
                    near_idx = idx
                    break
            
            label = "EN YAKIN UYGUN"
            if self._ipm is not None and not getattr(self, "_ipm_is_default", False) and self._ipm.m_per_px:
                dist_m = self._ipm.measure_distance_m(origin, centroid)
                if dist_m is not None:
                    label = f"EN YAKIN UYGUN {dist_m:.1f}m"
            elif res.get("scale_m_per_px") is not None:
                y_scales = res.get("y_scales", [])
                s_centroid = self.street_detector.estimate_local_scale(centroid[1], y_scales) if y_scales else res["scale_m_per_px"]
                if s_centroid is not None:
                    z = w * s_centroid
                    x = s_centroid * (centroid[0] - w / 2.0)
                    dist_m = float(np.sqrt(x**2 + z**2))
                    label = f"EN YAKIN UYGUN {dist_m:.1f}m"
            _ov.draw_guidance(out, origin, centroid, label="")
            if near_idx is not None:
                empty_slot_labels[near_idx].append((label, (0, 255, 255)))

        # A4: Çok kriterli en uygun slot önerisi (boş + sığan adaylar)
        cand_slots, cand_idx = [], []
        for i, poly in enumerate(empty_polys):
            if not (fit_flags[i] or not has_size):
                continue
            cc = poly.mean(axis=0)
            wm = sizes_m[i][0] if (i < len(sizes_m) and sizes_m[i]) else None
            cand_slots.append({"cx": float(cc[0]), "cy": float(cc[1]), "width_m": wm})
            cand_idx.append(i)
        best = (_scoring.recommend_best_slot(
            cand_slots, car_req, float(w), origin)
            if cand_slots else None)
        if best is not None:
            bi = cand_idx[best["index"]]
            bpoly = empty_polys_drawn[bi]
            cv2.polylines(out, [bpoly], True, (0, 215, 255), 3, cv2.LINE_AA)
            score_lbl = f"ONERILEN {best['score']}/100"
            empty_slot_labels[bi].append((score_lbl, (0, 215, 255)))
            self.recommendation_lbl.setText(
                f"⭐ ÖNERİLEN: SLOT {bi + 1} · Puan {best['score']}/100 · {best['reason']}")
        else:
            self.recommendation_lbl.setText("Uygun (sığan boş) slot bulunamadı")

        # Draw the labels for all empty slots in a beautiful stacked HUD card format
        for i, poly in enumerate(empty_polys):
            labels = empty_slot_labels[i]
            if not labels:
                continue
            
            c = poly.mean(axis=0).astype(int)
            cx, cy = int(c[0]), int(c[1])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1
            
            # Measure all text lines
            measured = []
            for text, color in labels:
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                measured.append((text, color, tw, th))
            
            max_w = max(item[2] for item in measured)
            total_h = sum(item[3] for item in measured) + (len(measured) - 1) * 5
            
            # Background rectangle dims
            pad_x = 8
            pad_y = 6
            rx1 = cx - max_w // 2 - pad_x
            ry1 = cy - total_h // 2 - pad_y
            rx2 = cx + max_w // 2 + pad_x
            ry2 = cy + total_h // 2 + pad_y
            
            # Keep within frame
            rx1 = max(0, rx1)
            ry1 = max(0, ry1)
            rx2 = min(w - 1, rx2)
            ry2 = min(h - 1, ry2)
            
            # Semi-transparent overlay card
            bg_overlay = out.copy()
            cv2.rectangle(bg_overlay, (rx1, ry1), (rx2, ry2), (15, 17, 26), -1)
            cv2.rectangle(bg_overlay, (rx1, ry1), (rx2, ry2), (51, 65, 85), 1)
            cv2.addWeighted(bg_overlay, 0.82, out, 0.18, 0, out)
            
            # Render labels top-to-bottom
            current_y = ry1 + pad_y
            for text, color, tw, th in reversed(measured):
                tx = cx - tw // 2
                ty = current_y + th
                cv2.putText(out, text, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)
                current_y += th + 5

        badge = ("ADAPTIF: CIZGI-IZGARA" if res["method"] == "line"
                 else "ADAPTIF: GEOMETRI")
        cv2.putText(out, f"{badge}  |  {elapsed_ms:.0f} ms", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Canlı güven göstergesi: slot örtüşme skorlarının ortalaması
        conf = res.get("mean_confidence")
        if conf is not None:
            cv2.putText(out, f"Guven: %{int(conf * 100)}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 2,
                        cv2.LINE_AA)

        # Şematik kuş bakışı mini-harita (sağ üst köşe)
        minimap = _ov.render_minimap(empty_polys, res["occupied_polys"], fit_flags=fit_flags)
        _ov.paste_minimap(out, minimap)

        # Store values for BEV overlay rendering
        self._current_empty_polys = empty_polys
        self._current_occupied_polys = res["occupied_polys"]
        self._current_detections = getattr(self, "_bev_detections", detections)
        self._current_sizes_m = sizes_m
        self._current_fit_flags = fit_flags

        available = res["empty_count"]
        occupied = res["occupied_count"]
        total = available + occupied
        self.park_cards[STATUS_AVAILABLE].set_count(available)
        self.park_cards[STATUS_OCCUPIED].set_count(occupied)
        self.park_cards[STATUS_FORBIDDEN].set_count(0)
        if total > 0:
            pct = int(occupied / total * 100)
            self.occupancy_lbl.setText(
                f"{available} bos · {occupied} dolu  (%{pct})  · {res['method']}")
        else:
            self.occupancy_lbl.setText("Slot tespit edilemedi")
        self._alerts.check_occupancy(available, occupied, self._alert_occ_threshold)

        # 2D Kuş Bakışı Harita güncelle
        if is_inference_frame or not hasattr(self, "_cached_map_pixmap") or self._cached_map_pixmap is None:
            try:
                mw = max(600, self.map_label.width())
                mh = max(400, self.map_label.height())
                schematic_map = _ov.render_full_schematic_map(
                    empty_polys, res["occupied_polys"],
                    sizes_m=sizes_m, fit_flags=fit_flags,
                    detections=getattr(self, "_bev_detections", detections),
                    perp_mode=self._perp_mode,
                    width=mw, height=mh
                )
                rgb_map = cv2.cvtColor(schematic_map, cv2.COLOR_BGR2RGB)
                h_m, w_m, ch_m = rgb_map.shape
                img_m = QImage(rgb_map.data, w_m, h_m, ch_m * w_m, QImage.Format_RGB888)
                self._cached_map_pixmap = QPixmap.fromImage(img_m)
                self.map_label.setPixmap(self._cached_map_pixmap)
            except Exception as e:
                import traceback
                traceback.print_exc()
        elif hasattr(self, "_cached_map_pixmap") and self._cached_map_pixmap is not None:
            self.map_label.setPixmap(self._cached_map_pixmap)

        return available, occupied

    # ── Değerlendirme paneli ──────────────────────────────────────
    def _run_evaluation_ui(self):
        """Sentetik veri setinde nicel değerlendirme çalıştır, sonuçları göster."""
        from PyQt5.QtWidgets import QApplication
        self.eval_btn.setEnabled(False)
        self.eval_btn.setText("Çalışıyor…")
        self.status_lbl.setText("Değerlendirme çalışıyor (120 sahne)…")
        QApplication.processEvents()
        try:
            from src.evaluation.runner import run_synthetic_evaluation
            out_dir = "outputs/evaluation_synthetic"
            dm, cm = run_synthetic_evaluation(n_scenes=120, out_dir=out_dir)
            self._show_evaluation_results(dm, out_dir)
            self.status_lbl.setText(
                f"Değerlendirme bitti · F1={dm.f1:.3f} AP={dm.ap:.3f}")
        except Exception as e:
            QMessageBox.warning(self, "Değerlendirme", f"Hata:\n{e}")
            self.status_lbl.setText("Değerlendirme hatası.")
        finally:
            self.eval_btn.setEnabled(True)
            self.eval_btn.setText("📊 Değerlendirme Çalıştır")

    def _show_evaluation_results(self, dm, out_dir):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea, QWidget
        dlg = QDialog(self)
        dlg.setWindowTitle("Nicel Değerlendirme — 120 Sentetik Sahne")
        dlg.setMinimumSize(560, 620)
        dlg.setStyleSheet("background:#0f172a; color:#e2e8f0;")

        content = QWidget()
        content.setStyleSheet("background:#0f172a;")
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(16, 12, 16, 12)
        vbox.setSpacing(10)

        summary = QLabel(
            f"<b>Tespit-temelli metrikler (120 sahne)</b><br>"
            f"Precision: {dm.precision:.3f} &nbsp; Recall: {dm.recall:.3f}<br>"
            f"F1: {dm.f1:.3f} &nbsp; AP: {dm.ap:.3f}<br>"
            f"TP: {dm.tp} &nbsp; FP: {dm.fp} &nbsp; FN: {dm.fn}")
        summary.setStyleSheet("color:#e2e8f0; font-size:13px;")
        vbox.addWidget(summary)

        for png in ("detection_metrics.png", "confusion_matrix.png"):
            path = Path(out_dir) / png
            if path.exists():
                lbl = QLabel()
                pm = QPixmap(str(path))
                if not pm.isNull():
                    lbl.setPixmap(pm.scaledToWidth(500, Qt.SmoothTransformation))
                vbox.addWidget(lbl)

        note = QLabel(f"Çıktı dosyaları: {out_dir}/ (PNG + CSV)")
        note.setStyleSheet("color:#94a3b8; font-size:10px;")
        vbox.addWidget(note)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none; background:#0f172a;}")

        close_btn = QPushButton("Kapat")
        close_btn.setFixedHeight(34)
        close_btn.setStyleSheet("QPushButton{background:#1e293b; color:#e2e8f0;"
                                "border-radius:6px;} QPushButton:hover{background:#334155;}")
        close_btn.clicked.connect(dlg.accept)

        lay = QVBoxLayout(dlg)
        lay.addWidget(scroll)
        lay.addWidget(close_btn)
        dlg.exec_()

    def _get_crop_bounds(self, frame):
        h_orig, w_orig = frame.shape[:2]
        crop_y_min = 0
        crop_y_max = h_orig
        crop_x_min = 0
        crop_x_max = w_orig

        if hasattr(self, "_ignore_top_ratio") and 0.0 < self._ignore_top_ratio < 0.45:
            crop_y_min = int(self._ignore_top_ratio * h_orig)

        if self._roi_polygon is not None and len(self._roi_polygon) > 0:
            pts = np.array(self._roi_polygon, dtype=np.int32)
            rx_min = max(0, int(pts[:, 0].min()))
            ry_min = max(0, int(pts[:, 1].min()))
            rx_max = min(w_orig, int(pts[:, 0].max()))
            ry_max = min(h_orig, int(pts[:, 1].max()))
            if (rx_max - rx_min) > 100 and (ry_max - ry_min) > 100:
                crop_x_min, crop_y_min = rx_min, ry_min
                crop_x_max, crop_y_max = rx_max, ry_max
        return crop_x_min, crop_y_min, crop_x_max, crop_y_max

    def _depth_keep_indices(self, frame, empty_spaces, parked):
        """Derinlik filtresinden geçen boş-slot indekslerini döndür.

        Çapraz açıda park sırasından çok farklı derinlikteki slotlar (ör. arka
        plandaki asfalt) elenir. Derinlik modeli yoksa/filtre kapalıysa tüm
        indeksler korunur (graceful). Döndürülen indeksler empty_spaces,
        confs ve sizes_m listelerini hizalı filtrelemek için kullanılır.
        """
        all_idx = list(range(len(empty_spaces)))
        if (not self._depth_filter or self.depth is None
                or not self.depth.available or not parked or not empty_spaces):
            return all_idx
            
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = self._get_crop_bounds(frame)
        h_orig, w_orig = frame.shape[:2]
        has_crop = (crop_y_min > 0 or crop_x_min > 0 or crop_y_max < h_orig or crop_x_max < w_orig)
        
        if has_crop:
            cropped_frame = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            depth_map_cropped = self.depth.infer(cropped_frame)
            if depth_map_cropped is None:
                return all_idx
            depth_map = np.zeros((h_orig, w_orig), dtype=np.float32)
            depth_map[crop_y_min:crop_y_max, crop_x_min:crop_x_max] = depth_map_cropped
        else:
            depth_map = self.depth.infer(frame)
            if depth_map is None:
                return all_idx
                
        self._last_depth_map = depth_map
        ref = [self.depth.region_depth(depth_map, b) for b in parked]
        ref = [r for r in ref if r is not None]
        if not ref:
            return all_idx
        ref_d = float(np.median(ref))
        kept = []
        for i, s in enumerate(empty_spaces):
            sd = self.depth.region_depth(depth_map, s)
            if sd is None or abs(sd - ref_d) <= 0.25:
                kept.append(i)
        return kept

    def _rebuild_street_detector(self):
        """Street detector'ı tüm iyileştirilmiş parametrelerle yeniden oluştur."""
        if self._perp_mode:
            self.street_detector = StreetParkingDetector(
                min_gap_ratio=self._min_gap_ratio,
                row_band_ratio=self._row_band_ratio,
                ignore_top_ratio=self._ignore_top_ratio,
                # Dik park optimizasyonları
                bottom_align_tol=0.65,      # ön görüşte dikey hizalama toleranslı
                lateral_split_ratio=4.0,
                max_gap_ratio=4.0,
                max_spaces_per_gap=4,
                max_edge_extension_ratio=0.40,
                frame_edge_margin_ratio=0.01,
                road_center_reject_ratio=0.0,
                road_color_tol_h=40.0,
                road_color_tol_s=90.0,
                road_color_tol_v=90.0,
                orientation="perpendicular",
                perspective_correction=True,
            )
        else:
            self.street_detector = StreetParkingDetector(
                min_gap_ratio=self._min_gap_ratio,
                row_band_ratio=self._row_band_ratio,
                ignore_top_ratio=self._ignore_top_ratio,
                # Paralel park optimizasyonları
                bottom_align_tol=0.35,
                lateral_split_ratio=3.5,
                max_gap_ratio=5.0,
                max_spaces_per_gap=3,
                max_edge_extension_ratio=0.40,
                frame_edge_margin_ratio=0.01,
                road_center_reject_ratio=0.0,
                road_color_tol_h=35.0,
                road_color_tol_s=80.0,
                road_color_tol_v=80.0,
                orientation="parallel",
                perspective_correction=True,
            )
        if hasattr(self, "_adaptive") and self._adaptive is not None:
            self._adaptive.street = self.street_detector

    def _set_min_gap(self, v):
        self._min_gap_ratio = v / 100
        if self._street_mode:
            self._rebuild_street_detector()
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _set_row_tol(self, v):
        self._row_band_ratio = v / 100
        if self._street_mode:
            self._rebuild_street_detector()
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _set_ignore_top(self, v):
        self._ignore_top_ratio = v / 100
        if self._street_mode:
            self._rebuild_street_detector()
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _on_alert(self, alert):
        bg, fg = LEVEL_COLORS[alert.level]
        self.alert_bar.setStyleSheet(
            f"background:{bg}; color:{fg}; border-radius:6px;"
            f" padding:5px 8px; font-size:10px; font-weight:bold;")
        self.alert_bar.setText(alert.message)
        self.alert_bar.setVisible(True)
        self._alert_close_btn.setVisible(True)
        if alert.level != AlertLevel.CRITICAL:
            self._alert_dismiss_timer.start(10_000)
        else:
            self._alert_dismiss_timer.stop()

    def _dismiss_alert(self):
        self.alert_bar.setVisible(False)
        self._alert_close_btn.setVisible(False)
        self._alert_dismiss_timer.stop()

    def _show_help(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea, QWidget
        dlg = QDialog(self)
        dlg.setWindowTitle("Yardım — Smart Parking AI")
        dlg.setMinimumWidth(420)
        dlg.setStyleSheet("background:#0f172a; color:#e2e8f0;")

        content = QWidget()
        content.setStyleSheet("background:#0f172a;")
        vbox = QVBoxLayout(content)
        vbox.setSpacing(10)
        vbox.setContentsMargins(16, 12, 16, 12)

        sections = [
            ("OTOMATİK PARK TESPİTİ", "#0e7490", [
                ("Min Boşluk",
                 "İki araç arasındaki mesafenin, ortalama araç genişliğine oranı. "
                 "Düşük değer: küçük boşlukları da tespit eder. "
                 "Yüksek değer: yalnızca büyük boşlukları gösterir."),
                ("Sıra Band",
                 "Aynı park sırasındaki araçların dikey konum toleransı. "
                 "Düşük değer: araçlar çok hizalı olmak zorunda. "
                 "Yüksek değer: farklı yüksekliklerdeki araçlar aynı sırada sayılır."),
                ("Üst Yoksay",
                 "Görüntünün üst kısmında bu oran kadar alan (gökyüzü, tabelalar) "
                 "tamamen yok sayılır. Yanlış araç tespitini önler."),
            ]),
            ("ARAÇ SIĞMA KONTROLÜ", "#166534", [
                ("Park Yönü (Paralel / Dik)",
                 "Paralel: Araçlar sokak boyunca sıralanmış, yan kamera görüşü. "
                 "Dik: Araçlar yola dik dizilmiş (otopark tipi). "
                 "Sistem görüntüdeki araçların en-boy oranına göre yan/ön görünümü "
                 "otomatik ayırt eder ve bunu etiketle bildirir."),
                ("Ref. Araç Uzunluğu (Paralel mod)",
                 "Görüntüdeki park etmiş araçların gerçek uzunluk tahmini (metre). "
                 "Bu değer piksel → metre ölçeğini belirler. "
                 "Standart otomobil için 4.0–4.8m uygundur."),
                ("Ref. Araç Eni (Dik mod)",
                 "Dik park modunda ölçek referansı olarak kullanılan araç eni (metre). "
                 "Ön görünüm için 1.8–2.0m uygundur. "
                 "Yan görünümde sistem ref. araç uzunluğunu otomatik kullanır."),
                ("Aracın Uzunluğu / Eni",
                 "Sizin aracınızın boyutu (metre). "
                 "Paralel modda uzunluk, dik modda en karşılaştırılır. "
                 "Sığan alanlar YEŞİL, sığmayanlar KIRMIZI gösterilir."),
            ]),
            ("UYARI EŞİĞİ", "#7c3aed", [
                ("Doluluk Eşiği",
                 "Park alanı doluluk yüzdesi bu eşiği aştığında uyarı verilir. "
                 "Örnek: 80 ayarlandıysa park %80 dolduğunda uyarı çıkar. "
                 "Park tamamen dolduğunda eşikten bağımsız olarak kritik uyarı verilir."),
            ]),
            ("GENEL BİLGİ", "#b45309", [
                ("Ölçek Tahmini",
                 "Sistem gerçek mesafeyi bilmez. Görüntüdeki park etmiş araçların "
                 "medyan piksel genişliğini 'Ref. Araç Uzunluğu' ile eşleştirerek "
                 "1 piksel = kaç metre hesaplar. Açı veya uzaklık değiştikçe tahmin sapabilir."),
                ("Park Süresi",
                 "Yalnızca video ve kamera modunda geçerlidir. "
                 "Araç ilk tespit edildiği andan itibaren süre sayılır ve "
                 "bbox üzerinde M:SS formatında gösterilir."),
                ("Snapshot / Log",
                 "Snapshot: o anki analiz sonucunu PNG olarak kaydeder. "
                 "Log: her karedeki tespit sonuçlarını CSV dosyasına yazar."),
            ]),
        ]

        for sec_title, color, items in sections:
            hdr = QLabel(sec_title)
            hdr.setStyleSheet(
                f"color:{color}; font-size:12px; font-weight:bold; "
                "padding:4px 0; border-bottom:1px solid #1e293b;"
            )
            vbox.addWidget(hdr)
            for item_title, desc in items:
                t = QLabel(f"<b style='color:#94a3b8'>{item_title}:</b> {desc}")
                t.setWordWrap(True)
                t.setStyleSheet("color:#cbd5e1; font-size:11px; padding:2px 8px;")
                vbox.addWidget(t)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border:none; background:#0f172a; }"
            "QScrollBar:vertical { width:4px; background:#0f172a; }"
            "QScrollBar::handle:vertical { background:#334155; border-radius:2px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }"
        )

        close_btn = QPushButton("Kapat")
        close_btn.setFixedHeight(36)
        close_btn.setStyleSheet(
            "QPushButton { background:#1e293b; color:#e2e8f0; border-radius:6px; "
            "font-size:12px; border:none; }"
            "QPushButton:hover { background:#334155; }"
        )
        close_btn.clicked.connect(dlg.accept)

        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setContentsMargins(0, 0, 0, 10)
        dlg_layout.addWidget(scroll)
        dlg_layout.addWidget(close_btn)
        dlg.setMinimumHeight(500)
        dlg.exec_()

    def _set_conf(self, v):
        self._conf_thresh = v / 100
        if self.detector:
            self.detector.conf = self._conf_thresh
        if self.auto_detector:
            self.auto_detector.conf = self._conf_thresh
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _set_iou(self, v):
        self._iou_thresh = v / 100
        if self.analyzer:
            self.analyzer.iou_threshold = self._iou_thresh
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    # ── Oto Tespit ────────────────────────────────────────────────
    def _load_auto_detector(self):
        from src.detection.parking_space_detector import ParkingSpaceDetector  # noqa: F811
        path, _ = QFileDialog.getOpenFileName(
            self, "Parking Detector Model Seç", "models/",
            "Model (*.pt)"
        )
        if not path:
            return
        try:
            self.auto_detector = ParkingSpaceDetector(path, conf=self._conf_thresh)
            self.auto_clear_btn.setEnabled(True)
            # Sınıf eşleşmelerini göster
            cmap = self.auto_detector.class_names
            raw_names = self.auto_detector.model.names
            bos_ids  = [f"{k}({raw_names[k]})" for k, v in cmap.items() if v == "BOS"]
            dolu_ids = [f"{k}({raw_names[k]})" for k, v in cmap.items() if v == "DOLU"]
            other_ids = [f"{k}:{v}({raw_names[k]})" for k, v in cmap.items() if v not in ("BOS", "DOLU")]
            info = f"Model: {Path(path).name}\n"
            info += f"BOS  → {bos_ids or 'EŞLEŞMEDİ!'}\n"
            info += f"DOLU → {dolu_ids or 'EŞLEŞMEDİ!'}"
            if other_ids:
                info += f"\nDiğer → {other_ids}"
            self.auto_det_lbl.setText(f"Model: {Path(path).name}")
            self.status_lbl.setText(info)
            if not bos_ids:
                QMessageBox.warning(
                    self, "Sınıf Eşleşme Uyarısı",
                    f"Modelde BOS sınıfı bulunamadı!\n\n"
                    f"Model sınıfları: {dict(raw_names)}\n\n"
                    f"Beklenen anahtar kelimeler: {_EMPTY_KEYWORDS}"
                )
            if self._last_frame is not None and self.cap is None:
                self._process_and_show(self._last_frame)
        except Exception as e:
            self.status_lbl.setText(f"Oto model hatası:\n{e}")

    def _clear_auto_detector(self):
        self.auto_detector = None
        self.auto_det_lbl.setText("Model: yüklenmedi")
        self.auto_clear_btn.setEnabled(False)
        self.status_lbl.setText("Oto tespit modu kapatıldı.")
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    # ── Model ─────────────────────────────────────────────────────
    def _load_model(self):
        self.status_lbl.setText("Model yükleniyor...")
        try:
            self.detector = VehicleDetector(conf=self._conf_thresh)
            self.status_lbl.setText("Model yüklendi.")
        except Exception as e:
            self.status_lbl.setText(f"Model hatası:\n{e}")

    # ── Snapshot ──────────────────────────────────────────────────
    def take_snapshot(self):
        if self._last_frame is None:
            self.status_lbl.setText("Snapshot: görüntü yok.")
            return
        out_dir = Path("outputs/snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = out_dir / f"snapshot_{ts}.jpg"
        # Son işlenmiş kareyi tekrar çizmek yerine mevcut analiz sonucunu kullan
        frame = self._last_frame.copy()
        if self._street_mode and self.street_detector and self._last_result is not None:
            car_dim = self._user_car_width_m if self._perp_mode else self._user_car_length_m
            frame = self.street_detector.draw(
                frame, self._last_result,
                car_length_m=car_dim,
            )
        elif self.analyzer and self.detector:
            dets = self.detector.detect(frame)
            result = self.analyzer.analyze(dets)
            frame = self.analyzer.draw(frame, result, dets)
        cv2.imwrite(str(img_path), frame)
        self.status_lbl.setText(f"Snapshot kaydedildi: {img_path.name}")

    # ── Loglama ───────────────────────────────────────────────────
    def toggle_logging(self):
        if not self._logging:
            out_dir = Path("outputs/metrics")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = out_dir / f"log_{ts}.csv"
            self._log_file = open(log_path, "w", newline="", encoding="utf-8")
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow([
                "frame", "timestamp", "vehicles",
                "available", "occupied", "forbidden_vehicles", "longest_parked_sec"
            ])
            self._frame_count = 0
            self._logging = True
            self.log_btn.setText("⏹  Loglama Durdur")
            self.log_btn.setStyleSheet(self.log_btn.styleSheet().replace("#0f766e", "#dc2626"))
            self.status_lbl.setText(f"Log: {log_path.name}")
        else:
            self._stop_logging()

    def _stop_logging(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._log_writer = None
        self._logging = False
        self.log_btn.setText("⏺  Loglama Başlat")
        self.log_btn.setStyleSheet(self.log_btn.styleSheet().replace("#dc2626", "#0f766e"))
        self.status_lbl.setText("Loglama durduruldu.")

    def _log_frame(self, vehicle_count, available, occupied, forbidden):
        if not self._logging or self._log_writer is None:
            return
        self._frame_count += 1
        durations = [d for _, d in self.vehicle_tracker.get_static_tracks_with_duration(
            min_frames=self._learn_min_frames)]
        longest = round(max(durations), 1) if durations else 0.0
        self._log_writer.writerow([
            self._frame_count,
            datetime.now().strftime("%H:%M:%S.%f")[:-3],
            vehicle_count, available, occupied, forbidden, longest
        ])

    # ── Zone yükleme ──────────────────────────────────────────────
    def _try_load_json(self, json_path: str) -> bool:
        try:
            loader = ZoneLoader(json_path)
            self.analyzer = ParkingAnalyzer(loader, iou_threshold=self._iou_thresh)
            n_park = len(loader.parking_zones)
            n_forb = len(loader.forbidden_zones)
            self.zone_lbl.setText(
                f"Zone: {Path(json_path).name}\n"
                f"{n_park} park · {n_forb} yasak"
            )
            self.status_lbl.setText(f"Zone yüklendi: {n_park + n_forb} bölge")
            return True
        except Exception as e:
            self.status_lbl.setText(f"Zone hatası:\n{e}")
            return False

    def load_zones_from_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Zone için Resim Seç", "data/",
            "Resim (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        json_path = str(Path(path).with_suffix(".json"))
        json_exists = Path(json_path).exists()

        if json_exists:
            msg = QMessageBox(self)
            msg.setWindowTitle("Zone Mevcut")
            msg.setText(f"{Path(json_path).name} zaten var.\nNe yapmak istersiniz?")
            btn_load = msg.addButton("Yükle", QMessageBox.AcceptRole)
            msg.addButton("Annotator ile Düzenle", QMessageBox.RejectRole)
            msg.exec_()
            if msg.clickedButton() == btn_load:
                if self._try_load_json(json_path):
                    ref = cv2.imread(path)
                    if ref is not None:
                        self._process_and_show(ref)
                return
            open_annotator = True
        else:
            reply = QMessageBox.question(
                self, "Zone Bulunamadı",
                f"{Path(json_path).name} bulunamadı.\nAnnotator ile oluşturulsun mu?",
                QMessageBox.Yes | QMessageBox.No
            )
            open_annotator = (reply == QMessageBox.Yes)

        if not open_annotator:
            return
        try:
            from src.parking.zone_annotator import ZoneAnnotator
            load_existing = json_path if json_exists else None
            ZoneAnnotator(path, output_path=json_path, load_path=load_existing).run()
        except Exception as e:
            self.status_lbl.setText(f"Annotator hatası:\n{e}")
            return
        if Path(json_path).exists():
            if self._try_load_json(json_path):
                ref = cv2.imread(path)
                if ref is not None:
                    self._process_and_show(ref)

    def _auto_load_zones(self, source_path: str):
        json_path = str(Path(source_path).with_suffix(".json"))
        if Path(json_path).exists():
            self._try_load_json(json_path)
        else:
            self.analyzer = None
            self.zone_lbl.setText("Zone: yüklenmedi")
            for card in self.park_cards.values():
                card.set_count(0)
            self.occupancy_lbl.setText("")

    # ── Feed yönetimi ─────────────────────────────────────────────
    def _start_feed(self):
        self._last_time = time.time()
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self.vid_btn.setEnabled(False)
        self.img_btn.setEnabled(False)
        if self.street_detector:
            self.street_detector.reset_history()
        self._adaptive.reset()
        self.stabilizer.reset()
        self._alerts._last_fired.clear()
        self.vehicle_tracker.reset()
        self.learned_slots.reset()
        self.heatmap.reset()
        self._drivable_tick = 0
        self._last_drivable_mask = None
        self._inference_tick = 0
        self._last_detections = []
        self._last_obstacles = []
        self._last_result = None
        self._last_learned_status = []
        self._last_static_mask = None
        self.timer.start(15)

    def _downsample_frame_if_needed(self, frame):
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > 960:
            scale = 960.0 / w
            return cv2.resize(frame, (960, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return frame

    def start_camera(self):
        self.stop_feed()
        raw_cap = cv2.VideoCapture(0)
        if not raw_cap.isOpened():
            self.status_lbl.setText("Kamera bulunamadı.")
            self.cap = None
            return
        self.cap = AsyncVideoCapture(raw_cap)
        if self.depth is not None:
            self.depth.reset()
        self.status_lbl.setText("Kamera aktif.")
        ret, first_frame = self.cap.read()
        if ret:
            first_frame = self._downsample_frame_if_needed(first_frame)
            self._run_auto_pipeline(first_frame)
        self._auto_enable_modes()
        self._start_feed()

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Video Seç", "", "Video (*.mp4 *.avi *.mkv *.mov)"
        )
        if not path:
            return
        self.stop_feed()
        raw_cap = cv2.VideoCapture(path)
        if not raw_cap.isOpened():
            self.status_lbl.setText("Video açılamadı.")
            self.cap = None
            return
        self.cap = AsyncVideoCapture(raw_cap)
        if self.depth is not None:
            self.depth.reset()
        self.status_lbl.setText("Video oynatılıyor.")
        self._auto_load_zones(path)
        # İlk kareden otomatik ayarları çıkart
        ret, first_frame = self.cap.read()
        if ret:
            first_frame = self._downsample_frame_if_needed(first_frame)
            self._run_auto_pipeline(first_frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._auto_enable_modes()
        self._start_feed()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Resim Seç", "", "Resim (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        self.stop_feed()
        frame = cv2.imread(path)
        if frame is None:
            self.status_lbl.setText("Resim açılamadı.")
            return
        frame = self._downsample_frame_if_needed(frame)
        self._last_result = None
        self._last_obstacles = []
        self._last_drivable_mask = None
        self._last_detections = []
        self._last_static_mask = None
        self._alerts._last_fired.clear()
        if self.street_detector is not None:
            self.street_detector.reset_history()
        self._adaptive.reset()
        self.stabilizer.reset()
        self._auto_load_zones(path)
        self._last_frame = frame
        self._auto_enable_modes()
        self._run_auto_pipeline(frame)
        self._process_and_show(frame)
        self.status_lbl.setText("Resim yüklendi.")

    def stop_feed(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.vid_btn.setEnabled(True)
        self.img_btn.setEnabled(True)
        self.fps_lbl.setText("FPS: —")
        self.status_lbl.setText("Durduruldu.")
        self._orient_history = []
        self._orientation_is_manual = False

    def closeEvent(self, event):
        self._stop_logging()
        super().closeEvent(event)

    # ── Frame işleme ──────────────────────────────────────────────
    def update_frame(self):
        if self.cap is None:
            return
        if hasattr(self.cap, "has_new_frame") and not self.cap.has_new_frame():
            return
        if getattr(self, "_processing_frame", False):
            return
        self._processing_frame = True
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_feed()
                self.status_lbl.setText("Bitti.")
                return

            frame = self._downsample_frame_if_needed(frame)
            self._last_frame = frame
            self._process_and_show(frame)

            now = time.time()
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(now - self._last_time, 1e-6))
            self._last_time = now
            self.fps_lbl.setText(f"FPS: {self._fps:.1f}")
        finally:
            self._processing_frame = False

    def _update_auto_orientation(self, detections):
        """Dinamik yön tespiti. Araçların en/boy oranlarını biriktirip gürültüden arındırılmış medyan ile yönü belirler."""
        if self._orientation_is_manual:
            return

        veh_dets = [
            d for d in detections
            if d.get("class_name") in {"car", "motorcycle", "bus", "truck"}
        ]
        if not veh_dets:
            return

        for det in veh_dets:
            bbox = det["bbox"]
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            if bh > 0:
                self._orient_history.append(bw / bh)

        # Geçmiş hafızasını son 50 araç oranı ile sınırla (aşırı birikmeyi önler ama kararlıdır)
        if len(self._orient_history) > 50:
            self._orient_history = self._orient_history[-50:]

        if len(self._orient_history) >= 5:
            median_ar = float(np.median(self._orient_history))
            # Paralel vs Dik eşiğini 1.55 olarak güncelledik (daha doğru ayrım)
            if median_ar >= 1.55:
                if not hasattr(self, "_perp_mode") or self._perp_mode:
                    self.set_parallel_mode()
            else:
                if not hasattr(self, "_perp_mode") or not self._perp_mode:
                    self.set_perpendicular_mode()

    def _run_auto_pipeline(self, frame):
        """Kamera/Video/Görsel yüklendiğinde otomatik kalibrasyon, ROI ve yön tespiti yapar."""
        if frame is None or self.detector is None:
            return

        frame_to_use = frame
        if self._night_vision:
            frame_to_use = self._enhance_low_light(frame, clip_limit=self._night_vision_clip)

        # 0) Yol maskesi hesaplanmadıysa öncelikle hesapla
        if self._last_drivable_mask is None and self.drivable is not None and self.drivable.available:
            try:
                da_mask, _ = self.drivable.infer(frame_to_use)
                if da_mask is not None:
                    kx = max(10, int(frame_to_use.shape[1] * 0.018))
                    ky = max(5, int(frame_to_use.shape[0] * 0.010))
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
                    da_mask = cv2.dilate(da_mask, kernel, iterations=1)
                    self._last_drivable_mask = da_mask
            except Exception:
                pass

        # 1) YOLO ile araçları tespit et
        dets = self.detector.detect(frame_to_use)
        # Adaptif güven filtresini uygula (yol dışı/hatalı tespitleri eler)
        dets = self._apply_adaptive_confidence_filter(dets, frame_to_use)
        
        veh_dets = [
            d for d in dets
            if d.get("class_name") in {"car", "motorcycle", "bus", "truck"}
        ]

        # 2) Park Yönü Otomatik Tespiti (Paralel / Dik)
        self._update_auto_orientation(veh_dets)

        # 3) Otomatik ROI Belirleme (Yalnızca statik görsel modunda ve önceden seçilmiş bir ROI yoksa)
        if self.cap is None and (self._roi_polygon is None or len(self._roi_polygon) == 0):
            poly = _roi.auto_roi_from_detections(veh_dets, frame_to_use.shape)
            if poly is not None:
                self._roi_polygon = poly
                self.roi_clear_btn.setEnabled(True)

        # 4) Otomatik IPM Kalibrasyonu (Eğer default/hiç kalibre edilmemişse)
        if self._ipm is None or getattr(self, "_ipm_is_default", False):
            from src.geometry.auto_ipm import auto_calibrate, auto_calibrate_from_vehicles
            h, w = frame_to_use.shape[:2]
            # Önce çizgi yakınsaması ile dene
            tf = auto_calibrate(frame_to_use, out_w=w, out_h=h, real_w_m=10.0, real_h_m=15.0)
            # Başarısızsa araç odaklı otomatik kalibrasyonu dene
            if tf is None:
                tf = auto_calibrate_from_vehicles(veh_dets, frame_to_use.shape, out_w=w, out_h=h, real_w_m=10.0, real_h_m=15.0)
            
            if tf is not None:
                self._ipm = tf
                self._ipm_is_default = False
                self._ipm_is_manual = False
                mpp = self._ipm.m_per_px
                self.ipm_status_lbl.setText(f"Oto kalibre · {mpp:.4f} m/px" if mpp else "Oto kalibre edildi")
                self.stabilizer.set_reference(frame_to_use)
                self._activate_ipm_view()

    def _apply_adaptive_confidence_filter(self, detections, frame):
        if not detections:
            return []

        # 1) Dynamic Confidence Threshold based on max confidence in the scene
        veh_confs = [d["confidence"] for d in detections if d.get("class_name") in {"car", "motorcycle", "bus", "truck"}]
        max_conf = max(veh_confs) if veh_confs else 0.0

        if max_conf > 0.85:
            base_threshold = 0.35
        elif max_conf > 0.70:
            base_threshold = 0.28
        else:
            base_threshold = 0.22

        # 2) Road surface / drivable area constraint
        road_mask = None
        if self._last_drivable_mask is not None:
            road_mask = self._last_drivable_mask
        elif self.street_detector is not None and getattr(self.street_detector, "_last_road_mask", None) is not None:
            road_mask = self.street_detector._last_road_mask

        H, W = frame.shape[:2]
        filtered = []

        # We can also get track history to support temporal boosting in video
        active_tracks = []
        if self.cap is not None and hasattr(self.vehicle_tracker, "tracks"):
            # Get centroids of tracks with history >= 3
            for track_id, track in self.vehicle_tracker.tracks.items():
                if len(track["history"]) >= 3:
                    active_tracks.append(track["history"][-1]) # last box (x1,y1,x2,y2)

        for det in detections:
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            
            # Default threshold
            thresh = base_threshold

            # Drivable area overlap check (base of the car must touch the road)
            if road_mask is not None:
                # Sample the bottom 15% of the bbox
                by1 = int(max(0, y2 - (y2 - y1) * 0.15))
                by2 = int(min(H, y2 + (y2 - y1) * 0.05)) # allow slightly below
                bx1 = int(max(0, x1))
                bx2 = int(min(W, x2))
                if by2 > by1 and bx2 > bx1:
                    road_patch = road_mask[by1:by2, bx1:bx2]
                    road_ratio = np.mean(road_patch > 0) if road_patch.size > 0 else 0.0
                    # If the base of the vehicle is NOT on the road, it's likely a false positive (sky/trees/buildings).
                    # We raise the confidence threshold significantly.
                    if road_ratio < 0.08:
                        thresh = max(thresh, 0.60)

            # Temporal check: if low confidence but matches an existing stable track, keep it.
            is_stable_track = False
            if self.cap is not None and active_tracks:
                for track_box in active_tracks:
                    # Calculate IoU with active track
                    ix1, iy1 = max(x1, track_box[0]), max(y1, track_box[1])
                    ix2, iy2 = min(x2, track_box[2]), min(y2, track_box[3])
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    area_i = iw * ih
                    area_u = (x2 - x1) * (y2 - y1) + (track_box[2] - track_box[0]) * (track_box[3] - track_box[1]) - area_i
                    iou = area_i / float(area_u) if area_u > 0 else 0.0
                    if iou > 0.35:
                        is_stable_track = True
                        break

            # If it's a stable track, we can boost/lower threshold to 0.20
            if is_stable_track:
                thresh = min(thresh, 0.20)

            if conf >= thresh:
                filtered.append(det)

        return filtered

    def _process_and_show(self, frame):
        import time
        self._last_t_start = time.time()
        # Eğer henüz kalibrasyon yapılmadıysa, varsayılan bir homografi oluştur
        if self._ipm is None or getattr(self, "_ipm_is_default", False):
            h_f, w_f = frame.shape[:2]
            if self._ipm is None or self._ipm.out_size != (w_f, h_f):
                try:
                    src_pts = [
                        (int(w_f * 0.35), int(h_f * 0.55)),
                        (int(w_f * 0.65), int(h_f * 0.55)),
                        (w_f, int(h_f * 0.95)),
                        (0, int(h_f * 0.95))
                    ]
                    self._ipm = PerspectiveTransformer.from_quad(
                        src_pts, out_w=w_f, out_h=h_f, real_w_m=10.0, real_h_m=15.0
                    )
                    self._ipm_is_default = True
                    self.ipm_status_lbl.setText("Varsayilan kalibrasyon")
                    self.ipm_show_btn.setEnabled(True)
                except Exception:
                    pass

        counts = {cls_id: 0 for cls_id in VEHICLE_CLASSES}
        detections = []

        # Video sabitleme: etkinse ve video/kamera modundaysa, kareyi referansa
        # hizala. Tespit + IPM + ızgara hizalanmış kare üzerinde çalışır →
        # IPM/ızgara elde-çekim kaymasına rağmen geçerli kalır (graceful).
        if self._stabilize and self.cap is not None:
            frame, _ = self.stabilizer.stabilize(frame)

        # Gece görüşü aydınlatma
        enhanced_frame = frame
        if self._night_vision:
            enhanced_frame = self._enhance_low_light(frame, clip_limit=self._night_vision_clip)

        is_inference_frame = True
        if self.cap is not None:
            is_inference_frame = (self._inference_tick % self._inference_period == 0)

        # Crop bounds computation for speed optimization
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = self._get_crop_bounds(enhanced_frame)
        h_orig, w_orig = enhanced_frame.shape[:2]
        has_crop = (crop_y_min > 0 or crop_x_min > 0 or crop_y_max < h_orig or crop_x_max < w_orig)
        
        if has_crop:
            cropped_frame = enhanced_frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        else:
            cropped_frame = enhanced_frame

        # Inference stratejisi:
        #  - Video/kamera modunda her N frame'de bir YOLO+engel birleşik
        #    çağrı (frame skipping); ara karelerde cache.
        #  - Diğer modlar → her frame YOLO (mevcut davranış).
        if self.detector:
            t_yolo_start = time.time()
            if self.cap is not None:
                if is_inference_frame or not hasattr(self, "_last_counts") or self._last_counts is None:
                    raw_dets = self.detector.detect(cropped_frame)
                    if has_crop:
                        for det in raw_dets:
                            det["bbox"] = [
                                det["bbox"][0] + crop_x_min,
                                det["bbox"][1] + crop_y_min,
                                det["bbox"][2] + crop_x_min,
                                det["bbox"][3] + crop_y_min,
                            ]
                            if "raw_bbox" in det:
                                det["raw_bbox"] = [
                                    det["raw_bbox"][0] + crop_x_min,
                                    det["raw_bbox"][1] + crop_y_min,
                                    det["raw_bbox"][2] + crop_x_min,
                                    det["raw_bbox"][3] + crop_y_min,
                                ]
                    filtered_dets = self._apply_adaptive_confidence_filter(raw_dets, enhanced_frame)
                    self._bev_detections = list(filtered_dets)
                    if self._roi_polygon is not None:
                        detections = _roi.filter_detections(filtered_dets, self._roi_polygon)
                    else:
                        detections = filtered_dets

                    detections = self._get_corrected_detections(enhanced_frame, detections)
                    self._last_detections = detections
                    self._last_bev_detections = self._bev_detections
                    self._last_obstacles  = []

                    counts = {cls_id: 0 for cls_id in VEHICLE_CLASSES}
                    for det in detections:
                        cls_id = det.get("class_id")
                        if cls_id in VEHICLE_CLASSES:
                            counts[cls_id] += 1
                    self._last_counts = counts

                    # Dinamik yön tespiti güncellemesi (her çıkarım karesinde biriktirir)
                    self._update_auto_orientation(self._bev_detections)

                    # Periodic auto-IPM update (every 10 inference frames) to handle non-parallel driving
                    inference_count = self._inference_tick // self._inference_period
                    if (not getattr(self, "_ipm_is_manual", False) and 
                            inference_count > 0 and self._inference_tick % self._inference_period == 0 and inference_count % 10 == 0):
                        from src.geometry.auto_ipm import auto_calibrate, auto_calibrate_from_vehicles
                        h_f, w_f = enhanced_frame.shape[:2]
                        tf = auto_calibrate(enhanced_frame, out_w=w_f, out_h=h_f, real_w_m=10.0, real_h_m=15.0)
                        if tf is None and self._bev_detections:
                            tf = auto_calibrate_from_vehicles(self._bev_detections, enhanced_frame.shape, out_w=w_f, out_h=h_f, real_w_m=10.0, real_h_m=15.0)
                        if tf is not None:
                            self._ipm = tf
                            self._ipm_is_default = False
                            mpp = self._ipm.m_per_px
                            self.ipm_status_lbl.setText(f"Oto kalibre (Dinamik) · {mpp:.4f} m/px" if mpp else "Oto kalibre (Dinamik)")
                else:
                    detections = self._last_detections
                    self._bev_detections = self._last_bev_detections
                    counts = self._last_counts
            else:
                raw_dets = self.detector.detect(cropped_frame)
                if has_crop:
                    for det in raw_dets:
                        det["bbox"] = [
                            det["bbox"][0] + crop_x_min,
                            det["bbox"][1] + crop_y_min,
                            det["bbox"][2] + crop_x_min,
                            det["bbox"][3] + crop_y_min,
                        ]
                        if "raw_bbox" in det:
                            det["raw_bbox"] = [
                                det["raw_bbox"][0] + crop_x_min,
                                det["raw_bbox"][1] + crop_y_min,
                                det["raw_bbox"][2] + crop_x_min,
                                det["raw_bbox"][3] + crop_y_min,
                            ]
                filtered_dets = self._apply_adaptive_confidence_filter(raw_dets, enhanced_frame)
                self._bev_detections = list(filtered_dets)
                if self._roi_polygon is not None:
                    detections = _roi.filter_detections(filtered_dets, self._roi_polygon)
                else:
                    detections = filtered_dets
                
                detections = self._get_corrected_detections(enhanced_frame, detections)
                self._last_obstacles = []
                counts = {cls_id: 0 for cls_id in VEHICLE_CLASSES}
                for det in detections:
                    cls_id = det.get("class_id")
                    if cls_id in VEHICLE_CLASSES:
                        counts[cls_id] += 1
            yolo_ms = (time.time() - t_yolo_start) * 1000.0
            self._latencies["yolo"] = 0.9 * self._latencies["yolo"] + 0.1 * yolo_ms

        available = occupied = forbidden = 0
        
        # Build background frame based on night-vision and split settings
        if self._night_vision and self._night_vision_split:
            h, w = frame.shape[:2]
            mid = w // 2
            bg_frame = frame.copy()
            bg_frame[:, mid:] = enhanced_frame[:, mid:]
            cv2.line(bg_frame, (mid, 0), (mid, h), (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(bg_frame, "ORIJINAL", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(bg_frame, "GECE GORUSU", (mid + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            bg_frame = enhanced_frame if self._night_vision else frame

        out = bg_frame.copy()
        # ROI sınırını çiz ve dışını karart
        if self._roi_polygon is not None:
            out = _roi.draw_roi(out, self._roi_polygon)

        # A1: Monoküler derinlik — ısı haritası overlay + araç mesafe etiketleri.
        # Her _depth_period karede bir hesaplanır (cache ile FPS korunur).
        if ((self._depth_overlay or self._depth_filter)
                and self.depth is not None and self.depth.available):
            self._depth_tick += 1
            if (self._last_depth_map is None
                    or self._depth_tick % self._depth_period == 0):
                self._last_depth_map = self.depth.infer(frame)
            dm = self._last_depth_map
            if dm is not None and self._depth_overlay:
                cmap = self.depth.depth_to_colormap(dm)
                if cmap is not None:
                    cv2.addWeighted(cmap, 0.45, out, 0.55, 0, out)
                for det in detections:
                    cls_id = det.get("class_id")
                    if cls_id not in VEHICLE_CLASSES:
                        continue
                    bbox = det["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    px_w = x2 - x1
                    
                    dist_m = None
                    if px_w > 0:
                        cname = det.get("class_name", "car")
                        ref_w = 2.0
                        if self.street_detector:
                            ref_w = self.street_detector._real_dim_for(cname, True)
                        
                        f = out.shape[1]
                        dx = (cx - f / 2.0) / f
                        phi = np.arctan(abs(dx))
                        
                        if det.get("perspective_corrected", False):
                            expected_dim = ref_w * np.cos(phi)
                        else:
                            ref_l = 4.5
                            if self.street_detector:
                                ref_l = self.street_detector._real_dim_for(cname, False)
                            theta = np.arctan(np.sqrt(dx**2 + ((cy - out.shape[0]/2.0)/f)**2))
                            if self._perp_mode:
                                expected_dim = ref_l * np.cos(theta) + ref_w * np.sin(theta)
                            else:
                                expected_dim = ref_l * np.sin(theta) + ref_w * np.cos(theta)
                        
                        z = f * (expected_dim / px_w)
                        x = (expected_dim / px_w) * (cx - f / 2.0)
                        dist_m = float(np.sqrt(x**2 + z**2))
                        
                    if dist_m is None or not (0.5 <= dist_m <= 120.0):
                        d_val = self.depth.region_depth(dm, bbox)
                        if d_val is not None:
                            dist_m = (1.0 - d_val) * 25.0
                            
                    if dist_m is not None:
                        cv2.putText(out, f"{dist_m:.1f}m", (x1, min(out.shape[0] - 4, y2 + 16)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                                    cv2.LINE_AA)

        if self._street_mode and self.street_detector:
            # Önce araçları normal şekilde çiz (adaptif mod aktif değilse)
            if not self._adaptive_mode:
                for det in detections:
                    cls_id = det.get("class_id")
                    if cls_id not in VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    color = VEHICLE_COLORS_CV.get(cls_id, (0, 255, 0))
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Park süresi — video/kamera modunda statik araçların üzerine yaz
            if self.cap is not None:
                for bbox, dur_sec in self.vehicle_tracker.get_static_tracks_with_duration(
                    min_frames=self._learn_min_frames
                ):
                    x1, y1, x2, y2 = map(int, bbox)
                    mins = int(dur_sec // 60)
                    secs = int(dur_sec % 60)
                    dur_lbl = f"{mins}:{secs:02d}"
                    (tw, th), _ = cv2.getTextSize(
                        dur_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(out,
                                  (x1, y1 - th - 6), (x1 + tw + 6, y1),
                                  (30, 30, 30), -1)
                    cv2.putText(out, dur_lbl, (x1 + 3, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 220, 50), 1, cv2.LINE_AA)

            if self.cap is None:
                static_mask = None
            elif is_inference_frame:
                # Detaylı dt hesabı (video kare süresi ile senkronize park süresi)
                dt = 0.033 # varsayılan 30 FPS
                if self.cap is not None:
                    try:
                        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                        fps_val = video_fps if video_fps > 0 else 30.0
                        is_video_file = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
                        if is_video_file:
                            dt = 1.0 / fps_val
                        else:
                            now = time.time()
                            if not hasattr(self, "_last_tracker_update_time"):
                                self._last_tracker_update_time = now
                            dt = min(1.0, now - self._last_tracker_update_time)
                            self._last_tracker_update_time = now
                    except Exception:
                        pass
                # Frame'i tracker'a geçir → ego-motion (optik akış) düzeltmesi
                static_mask = self.vehicle_tracker.update(detections, frame=enhanced_frame, dt=dt)
                # Kaçırılan ama aktif olan araçları ekleyerek titremeyi (flicker) önle
                missed = self.vehicle_tracker.get_missed_detections()
                if missed:
                    detections = list(detections)
                    detections.extend(missed)
                    static_mask.extend([m["is_static"] for m in missed])
                    
                    # Sayım istatistiklerini güncelle
                    counts = {cls_id: 0 for cls_id in VEHICLE_CLASSES}
                    for det in detections:
                        cls_id = det.get("class_id")
                        if cls_id in VEHICLE_CLASSES:
                            counts[cls_id] += 1
                    self._last_counts = counts
                self._last_static_mask = static_mask
                self._last_detections = detections
            else:
                static_mask = self._last_static_mask
                detections = self._last_detections

            # Drivable area (YOLOPv2) — seyrek hesapla + cache (ağır model) - Asenkron Çalışma
            if (self.drivable is not None and self.drivable.available):
                if (self.cap is None or self._drivable_tick % self._drivable_period == 0
                        or self._last_drivable_mask is None):
                    # Eğer arka planda çalışan aktif bir segmentasyon iş parçacığı yoksa başlat
                    if self._drivable_thread is None or not self._drivable_thread.is_alive():
                        # Parametreleri asenkron iş parçacığına taşımak için kopyala
                        frame_to_infer = enhanced_frame.copy()
                        crop_box = (crop_x_min, crop_y_min, crop_x_max, crop_y_max, h_orig, w_orig) if has_crop else None
                        
                        def _bg_infer():
                            t_start_lane = time.time()
                            try:
                                if crop_box is not None:
                                    cx1, cy1, cx2, cy2, ho, wo = crop_box
                                    cropped = frame_to_infer[cy1:cy2, cx1:cx2]
                                    da_mask_cropped, _ = self.drivable.infer(cropped)
                                    da_mask = np.zeros((ho, wo), dtype=np.uint8)
                                    if da_mask_cropped is not None:
                                        da_mask[cy1:cy2, cx1:cx2] = da_mask_cropped
                                else:
                                    da_mask, _ = self.drivable.infer(frame_to_infer)
                                
                                if da_mask is not None:
                                    kx = max(10, int(da_mask.shape[1] * 0.018))
                                    ky = max(5, int(da_mask.shape[0] * 0.010))
                                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
                                    da_mask = cv2.dilate(da_mask, kernel, iterations=1)
                                
                                with self._drivable_lock:
                                    self._last_drivable_mask = da_mask
                                
                                # Performans telemetrisi güncelle
                                elapsed = (time.time() - t_start_lane) * 1000.0
                                self._latencies["lane"] = 0.9 * self._latencies["lane"] + 0.1 * elapsed
                            except Exception:
                                pass
                                
                        self._drivable_thread = threading.Thread(target=_bg_infer, daemon=True)
                        self._drivable_thread.start()
                self._drivable_tick += 1

            # Adaptif çizgi-ızgara modu: izole dal — çizgi varsa ızgara, yoksa
            # geometri. Mevcut sezgisel yolu tamamen atlar (kendi içinde tam).
            if self._adaptive_mode:
                available, occupied = self._draw_adaptive(
                    enhanced_frame, out, detections, is_inference_frame,
                    static_mask=static_mask,
                    external_road_mask=self._last_drivable_mask,
                )
                for cls_id, card in self.stat_cards.items():
                    card.set_count(counts[cls_id])
                self._log_frame(sum(counts.values()), available, occupied, 0)
                self._show_frame(out)
                if self.cap is not None:
                    self._inference_tick += 1
                return

            # Tracker + analiz + öğrenme: yalnızca inference frame'inde yenile.
            # Ara karelerde cached result/learned_status kullanılır → 3x hız.

            if is_inference_frame or self._last_result is None:
                t_slot_start = time.time()
                result = self.street_detector.analyze(
                    enhanced_frame, detections,
                    obstacles=self._last_obstacles,
                    static_mask=static_mask,
                    external_road_mask=self._last_drivable_mask,
                    ref_car_length_m=self._ref_car_length_m,
                    ref_car_width_m=self._ref_car_width_m,
                    ipm=None if getattr(self, "_ipm_is_default", False) else self._ipm,
                )

                # Occupancy heatmap güncelle (ego-motion ile kaydırılarak)
                if self.cap is not None:
                    vehicle_bboxes = [d["bbox"] for d in detections]
                    ego_dx, ego_dy = self.vehicle_tracker.last_ego_motion
                    self.heatmap.update(enhanced_frame.shape, vehicle_bboxes,
                                        ego_dx=ego_dx, ego_dy=ego_dy)

                learned_status: list = []
                if self.cap is not None:
                    static_tracks = self.vehicle_tracker.get_static_tracks(
                        min_frames=self._learn_min_frames
                    )
                    vehicle_bboxes = [d["bbox"] for d in detections]
                    ego_dx, ego_dy = self.vehicle_tracker.last_ego_motion
                    learned_status = self.learned_slots.update(
                        static_tracks, vehicle_bboxes,
                        road_mask=self._last_drivable_mask,
                        ego_dx=ego_dx,
                        ego_dy=ego_dy,
                    )

                # Heuristic boş slot'lardan öğrenilmiş ile çakışanları çıkar
                from src.detection.street_parking_detector import _bbox_iou as _iou_fn
                learned_bboxes = [s["bbox"] for s in learned_status]
                result["empty_spaces"] = [
                    e for e in result["empty_spaces"]
                    if not any(_iou_fn(e, lb) > 0.40 for lb in learned_bboxes)
                ]

                # Heatmap süzgeci: ısınma sonrası, slot'un altındaki bölgede
                # geçmişte hiç araç görülmediyse (yol ortası, kaldırım) ele.
                # Her geçen slot için 0-1 confidence skoru üret.
                slot_confidences: list[float] = []
                if self.cap is not None and self.heatmap.is_warmed_up:
                    filtered = []
                    for s in result["empty_spaces"]:
                        # Slot çevresinde park sırası araç(lar)ı var mı?
                        # Yolun ortasındaki/hayalî slot'lar 0'a yakın olur.
                        p = self.heatmap.slot_neighborhood_max(
                            s, enhanced_frame.shape, expand=0.6
                        )
                        if p >= self._slot_min_prob:
                            filtered.append(s)
                            slot_confidences.append(min(1.0, p / 0.5))
                    result["empty_spaces"]    = filtered
                    result["slot_confidences"] = slot_confidences
                else:
                    result["slot_confidences"] = [0.5] * len(result["empty_spaces"])

                self._last_result = result
                self._last_learned_status = learned_status
                slot_ms = (time.time() - t_slot_start) * 1000.0
                self._latencies["slot"] = 0.9 * self._latencies["slot"] + 0.1 * slot_ms
            else:
                result = self._last_result
                learned_status = self._last_learned_status

            # Dik mod görünüm açısı etiketi
            if self._perp_mode and result is not None:
                side = result.get("perp_side_view", False)
                self._view_lbl.setText(
                    "Yan gorunum aktif (ref. uzunluk)" if side
                    else "On gorunum aktif (ref. en)"
                )

            # Çizim katmanları — heuristic boş slot'lar zorluk/konum aware
            empty_spaces = result.get("empty_spaces", [])
            confs        = result.get("slot_confidences",
                                      [0.5] * len(empty_spaces))
            sizes_m      = result.get("slot_sizes_m", [])
            difficulties = result.get("slot_difficulties",
                                      [80] * len(empty_spaces))
            # Derinlik filtresi (model + toggle açıksa çapraz-açı slotlarını eler)
            if self._depth_filter and empty_spaces:
                keep = self._depth_keep_indices(
                    enhanced_frame, empty_spaces, result.get("parked", []))
                if len(keep) != len(empty_spaces):
                    empty_spaces = [empty_spaces[i] for i in keep]
                    confs = [confs[i] for i in keep] if confs else confs
                    sizes_m = [sizes_m[i] for i in keep] if sizes_m else sizes_m
                    difficulties = [difficulties[i] for i in keep] if difficulties else difficulties
            scale        = result.get("scale_m_per_px")
            check_fit    = scale is not None and scale > 0

            COLOR_EASY   = (80, 220, 0)      # emerald green BGR
            COLOR_MEDIUM = (0, 165, 255)     # amber orange BGR
            COLOR_HARD   = (0, 0, 240)       # coral red BGR

            if empty_spaces:
                overlay = out.copy()
                for i, s in enumerate(empty_spaces):
                    x1, y1, x2, y2 = map(int, s)
                    score = difficulties[i] if i < len(difficulties) else 80
                    if score >= 75:
                        color = COLOR_EASY
                    elif score >= 45:
                        color = COLOR_MEDIUM
                    else:
                        color = COLOR_HARD
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                avg_c = float(np.mean(confs)) if confs else 0.5
                alpha = 0.18 + 0.27 * avg_c
                cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

                for i, s in enumerate(empty_spaces):
                    x1, y1, x2, y2 = map(int, s)
                    score = difficulties[i] if i < len(difficulties) else 80
                    w_m = sizes_m[i][0] if (i < len(sizes_m) and sizes_m[i][0] > 0) else None

                    # Etiket metnini ve rengini belirle
                    if score >= 75:
                        color = COLOR_EASY
                        label = f"KOLAY {score}%"
                    elif score >= 45:
                        color = COLOR_MEDIUM
                        label = f"ORTA {score}%"
                    else:
                        color = COLOR_HARD
                        # Sığmama durumunu kontrol et
                        dim = self._user_car_width_m if self._perp_mode else self._user_car_length_m
                        if w_m is not None and w_m < dim:
                            label = f"ZOR {score}% (DAR)"
                        else:
                            label = f"ZOR {score}%"

                    if w_m is not None:
                        label += f" ({w_m:.1f}m)"

                    thick = 2 if score < 75 else 3
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
                    font_s = max(0.4, min(0.55, (x2 - x1) / 220))
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_s, 1)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.rectangle(out,
                                  (cx - tw // 2 - 4, cy - th // 2 - 4),
                                  (cx + tw // 2 + 4, cy + th // 2 + 4),
                                  color, -1)
                    cv2.putText(out, label,
                                (cx - tw // 2, cy + th // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, font_s,
                                (255, 255, 255), 1, cv2.LINE_AA)

            # Öğrenilmiş kalıcı slot'lar — yalnızca BOS olanları göster.
            # DOLU slot'lar araç bbox'ı ile zaten temsil edildiği için
            # tekrar çizmek görsel kalabalık yaratıyor.
            COLOR_LEARNED_EMPTY = (255, 200, 0)   # cyan
            learned_empty_slots = [s for s in learned_status if not s["occupied"]]
            if learned_empty_slots:
                overlay = out.copy()
                for s in learned_empty_slots:
                    x1, y1, x2, y2 = map(int, s["bbox"])
                    cv2.rectangle(overlay, (x1, y1), (x2, y2),
                                  COLOR_LEARNED_EMPTY, -1)
                cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
                for s in learned_empty_slots:
                    x1, y1, x2, y2 = map(int, s["bbox"])
                    cv2.rectangle(out, (x1, y1), (x2, y2),
                                  COLOR_LEARNED_EMPTY, 2)
                    # Etiket yalnızca slot yeterince büyükse
                    if (x2 - x1) >= 40 and (y2 - y1) >= 20:
                        cv2.putText(out, "BOS+", (x1 + 4, y2 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    COLOR_LEARNED_EMPTY, 2, cv2.LINE_AA)

            # Engelleri sadece ince konturla; etiket yazma (kalabalık azaltma)
            for ob in self._last_obstacles:
                ox1, oy1, ox2, oy2 = map(int, ob["bbox"])
                cv2.rectangle(out, (ox1, oy1), (ox2, oy2), (60, 60, 220), 1)

            # Sayımlar: öğrenilmiş + dedup'lanmış heuristic
            learned_empty    = sum(1 for s in learned_status if not s["occupied"])
            learned_occupied = sum(1 for s in learned_status if s["occupied"])
            heuristic_empty  = len(result["empty_spaces"])
            # DOLU: tracker'ın gördüğü tüm statik araçlar (heuristic parked
            # zaten learned slot'ları da içerebilir — max ile birleştir)
            available = learned_empty + heuristic_empty
            occupied  = max(result["occupied_count"], learned_occupied)
            total     = available + occupied
            self.park_cards[STATUS_AVAILABLE].set_count(available)
            self.park_cards[STATUS_OCCUPIED].set_count(occupied)
            self.park_cards[STATUS_FORBIDDEN].set_count(0)
            if total > 0:
                pct = int(occupied / total * 100)
                dur_note = "" if self.cap is not None else "  · Park suresi: video modunda"
                self.occupancy_lbl.setText(
                    f"{available} bos · {occupied} dolu  (%{pct}){dur_note}")
            else:
                self.occupancy_lbl.setText("Arac tespit edilemedi")
            self._alerts.check_occupancy(available, occupied, self._alert_occ_threshold)
            if check_fit and empty_spaces:
                dim = self._user_car_width_m if self._perp_mode else self._user_car_length_m
                fit_count = sum(
                    1 for i, s in enumerate(empty_spaces)
                    if i < len(sizes_m) and sizes_m[i][0] >= dim
                )
                self._alerts.check_no_fit(fit_count, len(empty_spaces))
            for cls_id, card in self.stat_cards.items():
                card.set_count(counts[cls_id])

            if not (is_inference_frame or self.cap is None):
                self._log_frame(sum(counts.values()), available, occupied, 0)
                self._show_frame(out)
                if self.cap is not None:
                    self._inference_tick += 1
                return

            # 2D Kuş Bakışı Harita güncelle (Geleneksel mod)
            try:
                trad_empty_polys = []
                for s in empty_spaces:
                    x1, y1, x2, y2 = s
                    trad_empty_polys.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))

                trad_occupied_polys = []
                for s in learned_status:
                    if s["occupied"]:
                        x1, y1, x2, y2 = s["bbox"]
                        trad_occupied_polys.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))

                trad_fit_flags = []
                trad_difficulties = []
                for i in range(len(empty_spaces)):
                    if check_fit and i < len(sizes_m):
                        w_m = sizes_m[i][0]
                        dim = self._user_car_width_m if self._perp_mode else self._user_car_length_m
                        trad_fit_flags.append(w_m >= dim)
                    else:
                        trad_fit_flags.append(True)
                    
                    if i < len(difficulties):
                        trad_difficulties.append(difficulties[i])
                    else:
                        trad_difficulties.append(80)

                self._current_empty_polys = trad_empty_polys
                self._current_occupied_polys = trad_occupied_polys
                self._current_detections = getattr(self, "_bev_detections", detections)
                self._current_sizes_m = sizes_m
                self._current_fit_flags = trad_fit_flags
                self._current_difficulties = trad_difficulties

                # ── A4: Çok Kriterli Slot Önerisi ──────────────────────────
                if empty_spaces and hasattr(self, "occupancy_lbl"):
                    slot_candidates = []
                    frame_w = enhanced_frame.shape[1] if enhanced_frame is not None else 1280
                    for i, s in enumerate(empty_spaces):
                        x1, y1, x2, y2 = s
                        slot_candidates.append({
                            "cx": (x1 + x2) / 2.0,
                            "cy": (y1 + y2) / 2.0,
                            "width_m": sizes_m[i][0] if i < len(sizes_m) and sizes_m[i] else None,
                        })
                    ref_w = self._user_car_width_m if self._perp_mode else self._user_car_length_m
                    rec = _scoring.recommend_best_slot(
                        slot_candidates,
                        ref_width_m=ref_w,
                        map_width=float(frame_w),
                        origin=(frame_w / 2.0, enhanced_frame.shape[0] * 0.9
                                if enhanced_frame is not None else 720),
                    )
                    if rec is not None:
                        self._recommended_slot_idx = rec["index"]
                        rec_txt = (f"⭐ ÖNERİLEN: SLOT {rec['index']+1}"
                                   f"  ·  Puan: {rec['score']}/100"
                                   f"  ·  {rec['reason']}")
                        if hasattr(self, "status_lbl"):
                            self.status_lbl.setText(rec_txt)
                    else:
                        self._recommended_slot_idx = -1
                # ────────────────────────────────────────────────────────────

                # ── SLAM Harita Güncelleme ──────────────────────────────────
                if self._slam_active:
                    ego_dx, ego_dy = self.vehicle_tracker.last_ego_motion
                    self._slam_cum_dx += ego_dx
                    self._slam_cum_dy += ego_dy

                    def _to_global(poly):
                        arr = np.asarray(poly, dtype=np.float32)
                        arr[:, 0] += self._slam_cum_dx
                        arr[:, 1] += self._slam_cum_dy
                        return arr

                    def _centroid(poly):
                        arr = np.asarray(poly, dtype=np.float32)
                        return arr.mean(axis=0)

                    MATCH_DIST = 55.0  # px in global space

                    def _find_match(centroid):
                        for idx, slot in enumerate(self._slam_slots):
                            gc = _centroid(slot["poly_global"])
                            if np.hypot(centroid[0] - gc[0], centroid[1] - gc[1]) < MATCH_DIST:
                                return idx
                        return -1

                    # Merge / add empty slots
                    for i, poly in enumerate(trad_empty_polys):
                        gp = _to_global(poly)
                        gc = _centroid(gp)
                        idx = _find_match(gc)
                        meta = {
                            "poly_global": gp,
                            "occupied": False,
                            "size_m": sizes_m[i] if i < len(sizes_m) else None,
                            "fit": trad_fit_flags[i] if i < len(trad_fit_flags) else True,
                            "difficulty": trad_difficulties[i] if i < len(trad_difficulties) else 80,
                        }
                        if idx == -1:
                            self._slam_slots.append(meta)
                        else:
                            self._slam_slots[idx].update(meta)

                    # Merge / add occupied slots
                    for poly in trad_occupied_polys:
                        gp = _to_global(poly)
                        gc = _centroid(gp)
                        idx = _find_match(gc)
                        if idx == -1:
                            self._slam_slots.append({"poly_global": gp, "occupied": True,
                                                      "size_m": None, "fit": False, "difficulty": 0})
                        else:
                            self._slam_slots[idx]["occupied"] = True

                    # Project global slots back to current frame coordinates
                    def _to_local(poly_global):
                        arr = np.asarray(poly_global, dtype=np.float32).copy()
                        arr[:, 0] -= self._slam_cum_dx
                        arr[:, 1] -= self._slam_cum_dy
                        return arr

                    slam_empty = []
                    slam_empty_fits = []
                    slam_empty_diffs = []
                    slam_occupied = []
                    for slot in self._slam_slots:
                        local = _to_local(slot["poly_global"])
                        if slot["occupied"]:
                            slam_occupied.append(local)
                        else:
                            slam_empty.append(local)
                            slam_empty_fits.append(slot.get("fit", True))
                            slam_empty_diffs.append(slot.get("difficulty", 80))

                    self._current_empty_polys = slam_empty
                    self._current_occupied_polys = slam_occupied
                    self._current_fit_flags = slam_empty_fits
                    self._current_difficulties = slam_empty_diffs
                # ────────────────────────────────────────────────────────────

                if is_inference_frame or self._sim_active:
                    self._update_schematic_map_ui()
            except Exception:
                pass

            self._log_frame(sum(counts.values()), available, occupied, 0)
            self._show_frame(out)
            return

        if self.auto_detector:
            auto_dets = self.auto_detector.detect(enhanced_frame)
            out = self.auto_detector.draw(out, auto_dets)
            # Araç kutularını üzerine çiz (etiketsiz)
            for det in detections:
                cls_id = det.get("class_id")
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(out, (x1, y1), (x2, y2),
                              VEHICLE_COLORS_CV.get(cls_id, (0, 255, 0)), 2)
            available = sum(1 for d in auto_dets if d["status"] == "BOS")
            occupied  = sum(1 for d in auto_dets if d["status"] == "DOLU")
            # Model boş yer tespit edemiyorsa zone sistemiyle tamamla
            if available == 0 and self.analyzer:
                zone_result = self.analyzer.analyze(detections)
                available = zone_result.available
                occupied  = max(occupied, zone_result.occupied)
            # Yasaklı zone'lar zone sistemiyle de gösterilebilir
            if self.analyzer:
                zone_result = self.analyzer.analyze(detections)
                forbidden = zone_result.forbidden_vehicles
                for zs in zone_result.zone_statuses:
                    if zs.status == STATUS_FORBIDDEN:
                        pts = zs.zone.polygon
                        overlay = out.copy()
                        cv2.fillPoly(overlay, [pts], (0, 0, 200))
                        cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)
                        cv2.polylines(out, [pts], True, (0, 0, 200), 2)
                        cx = int(pts[:, 0].mean())
                        cy = int(pts[:, 1].mean())
                        cv2.putText(out, "YASAK", (cx - 25, cy + 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2, cv2.LINE_AA)
            total = available + occupied
            self.park_cards[STATUS_AVAILABLE].set_count(available)
            self.park_cards[STATUS_OCCUPIED].set_count(occupied)
            self.park_cards[STATUS_FORBIDDEN].set_count(forbidden)
            if total > 0:
                pct = int(occupied / total * 100)
                self.occupancy_lbl.setText(f"{occupied}/{total} slot dolu  (%{pct})")
            else:
                self.occupancy_lbl.setText("")
            self._alerts.check_occupancy(available, occupied, self._alert_occ_threshold)
            self._alerts.check_forbidden(forbidden)
        elif self.analyzer:
            result = self.analyzer.analyze(detections)
            out = self.analyzer.draw(out, result, detections)
            available = result.available
            occupied  = result.occupied
            forbidden = result.forbidden_vehicles
            total     = result.total_parking
            self.park_cards[STATUS_AVAILABLE].set_count(available)
            self.park_cards[STATUS_OCCUPIED].set_count(occupied)
            self.park_cards[STATUS_FORBIDDEN].set_count(forbidden)
            if total > 0:
                pct = int(occupied / total * 100)
                self.occupancy_lbl.setText(f"{occupied}/{total} slot dolu  (%{pct})")
            else:
                self.occupancy_lbl.setText("")
            self._alerts.check_occupancy(available, occupied, self._alert_occ_threshold)
            self._alerts.check_forbidden(forbidden)
        else:
            for card in self.park_cards.values():
                card.set_count(0)
            self.occupancy_lbl.setText("")
            for det in detections:
                cls_id = det.get("class_id")
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, det["bbox"])
                conf = det["confidence"]
                color = VEHICLE_COLORS_CV.get(cls_id, (0, 255, 0))
                label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for cls_id, card in self.stat_cards.items():
            card.set_count(counts[cls_id])

        self._log_frame(sum(counts.values()), available, occupied, forbidden)

        self._show_frame(out)
        if self.cap is not None:
            self._inference_tick += 1

    def _show_frame(self, out):
        import time
        t_draw_start = time.time()
        # IPM açıksa görselleştirmeyi kuş bakışına çevir (analiz orijinalde kalır)
        if self._ipm_show and self._ipm is not None:
            try:
                # Orijinal temiz görüntüyü büküyoruz (üzerine yamulmuş yazılar çizilmeden önce)
                clean_frame = self._last_frame.copy()
                if self._stabilize and self.cap is not None:
                    clean_frame, _ = self.stabilizer.stabilize(clean_frame)

                bev_out = self._ipm.warp_image(clean_frame)

                # Bükülmüş temiz kuş bakışı görüntü üzerine kutuları, slotları ve yazıları dik çizeriz
                if hasattr(self, "_current_empty_polys"):
                    _ov.draw_bev_overlays(
                        bev_out, self._ipm,
                        self._current_empty_polys,
                        self._current_occupied_polys,
                        self._current_detections,
                        sizes_m=getattr(self, "_current_sizes_m", None),
                        fit_flags=getattr(self, "_current_fit_flags", None),
                        difficulties=getattr(self, "_current_difficulties", None)
                    )
                out = bev_out
            except Exception:
                pass

        # Performance & FPS telemetry update
        if hasattr(self, "_latencies"):
            t_end = time.time()
            draw_ms = (t_end - t_draw_start) * 1000.0
            self._latencies["draw"] = 0.9 * self._latencies["draw"] + 0.1 * draw_ms

            t_start = getattr(self, "_last_t_start", None)
            if t_start is None:
                t_start = t_end - 0.03
            total_ms = (t_end - t_start) * 1000.0
            self._latencies["total"] = 0.9 * self._latencies["total"] + 0.1 * total_ms

            self._fps_frame_count += 1
            elapsed = t_end - self._fps_last_time
            if elapsed >= 1.0:
                self._latencies["fps"] = self._fps_frame_count / elapsed
                print(f"[PERF] FPS: {self._latencies['fps']:.1f} | YOLO: {self._latencies['yolo']:.1f}ms | Lane: {self._latencies['lane']:.1f}ms | Slot: {self._latencies['slot']:.1f}ms | Draw: {self._latencies['draw']:.1f}ms | Total: {self._latencies['total']:.1f}ms")
                self._fps_frame_count = 0
                self._fps_last_time = t_end

            # Render presentation mode HUD overlay
            if self._presentation_mode:
                self._draw_performance_hud(out)

            # Render BSD blind-spot warning overlay
            if self._bsd_active:
                self._draw_bsd_overlay(out)

        # Performance optimization: Scale image using OpenCV instead of Qt QPixmap.scaled()
        lbl_w = self.video_label.width()
        lbl_h = self.video_label.height()
        h0, w0 = out.shape[:2]
        if lbl_w > 0 and lbl_h > 0 and h0 > 0 and w0 > 0:
            scale_f = min(lbl_w / w0, lbl_h / h0)
            nw = int(w0 * scale_f)
            nh = int(h0 * scale_f)
            out_resized = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_NEAREST)
        else:
            out_resized = out

        self._rgb_buf = cv2.cvtColor(out_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = self._rgb_buf.shape
        img = QImage(self._rgb_buf.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def _update_schematic_map_ui(self):
        if not hasattr(self, "_current_empty_polys"):
            return
        try:
            mw = max(600, self.map_label.width())
            mh = max(400, self.map_label.height())
            schematic_map = _ov.render_full_schematic_map(
                self._current_empty_polys, self._current_occupied_polys,
                sizes_m=self._current_sizes_m, fit_flags=self._current_fit_flags,
                detections=self._current_detections,
                perp_mode=self._perp_mode,
                width=mw, height=mh,
                difficulties=self._current_difficulties,
                sim_active=self._sim_active,
                sim_car_x=self._sim_car_x,
                sim_car_y=self._sim_car_y,
                sim_car_yaw=self._sim_car_yaw,
                sim_target_idx=self._sim_target_idx,
                sim_instruction=self._sim_instruction,
                sim_steering_angle=self._sim_steering_angle,
                sim_step_name=self._sim_step_name,
                sim_path=self._sim_path,
                night_vision=self._night_vision
            )
            rgb_map = cv2.cvtColor(schematic_map, cv2.COLOR_BGR2RGB)
            h_m, w_m, ch_m = rgb_map.shape
            img_m = QImage(rgb_map.data, w_m, h_m, ch_m * w_m, QImage.Format_RGB888)
            self.map_label.setPixmap(QPixmap.fromImage(img_m))
        except Exception as e:
            import traceback
            traceback.print_exc()

    def _get_schematic_slots(self, width, height):
        empty_polys = getattr(self, "_current_empty_polys", [])
        occupied_polys = getattr(self, "_current_occupied_polys", [])
        sizes_m = getattr(self, "_current_sizes_m", None)
        fit_flags = getattr(self, "_current_fit_flags", None)
        difficulties = getattr(self, "_current_difficulties", None)
        detections = getattr(self, "_current_detections", None)
        perp_mode = self._perp_mode

        slots_list = []
        for i, p in enumerate(empty_polys):
            cx = float(np.asarray(p)[:, 0].mean())
            sz = sizes_m[i] if (sizes_m is not None and i < len(sizes_m)) else None
            fit = fit_flags[i] if (fit_flags is not None and i < len(fit_flags)) else True
            diff = difficulties[i] if (difficulties is not None and i < len(difficulties)) else None
            slots_list.append((cx, False, sz, fit, diff, i, "BOS"))

        occupied_xs = []
        for i, p in enumerate(occupied_polys):
            cx = float(np.asarray(p)[:, 0].mean())
            slots_list.append((cx, True, None, False, "DOLU", i, "DOLU"))
            occupied_xs.append(cx)

        VEHICLE_CLASSES = {2, 3, 5, 7}
        if detections:
            for det in detections:
                cls_id = det.get("class_id")
                if cls_id not in VEHICLE_CLASSES:
                    continue
                bbox = det["bbox"]
                cx = (bbox[0] + bbox[2]) / 2.0
                is_duplicate = False
                for ox in occupied_xs:
                    if abs(cx - ox) < 40:
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue
                overlapping_empty_idx = -1
                for idx, item in enumerate(slots_list):
                    if not item[1]: # Bos slot
                        if abs(cx - item[0]) < 35:
                            overlapping_empty_idx = idx
                            break
                if overlapping_empty_idx != -1:
                    slots_list[overlapping_empty_idx] = (cx, True, None, False, "DOLU", slots_list[overlapping_empty_idx][5], "DOLU")
                else:
                    slots_list.append((cx, True, None, False, "ARAC", -1, "ARAC"))

        slots_list.sort(key=lambda t: t[0])
        return slots_list

    def _calculate_start_x(self, slots_list, start_x, slot_w, gap, w, scale):
        n = len(slots_list)
        if n == 0:
            return w // 2
        camera_w = 1280
        all_x = []
        for item in slots_list:
            all_x.append(item[0])
        if all_x:
            camera_w = max(camera_w, max(all_x))
        
        ego_x_camera = camera_w / 2.0
        if n <= 1:
            return w // 2
        else:
            camera_x_coords = [item[0] for item in slots_list]
            schematic_x_coords = [start_x + idx * (slot_w + gap) + slot_w // 2 for idx in range(n)]
            if ego_x_camera <= camera_x_coords[0]:
                return int(schematic_x_coords[0] - (camera_x_coords[0] - ego_x_camera) * (slot_w + gap) / max(10.0, camera_x_coords[0]))
            elif ego_x_camera >= camera_x_coords[-1]:
                return int(schematic_x_coords[-1] + (ego_x_camera - camera_x_coords[-1]) * (slot_w + gap) / max(10.0, camera_w - camera_x_coords[-1]))
            else:
                return int(np.interp(ego_x_camera, camera_x_coords, schematic_x_coords))

    def _on_map_clicked(self, event):
        if event.button() != Qt.LeftButton:
            return
            
        w = max(600, self.map_label.width())
        h = max(400, self.map_label.height())
        click_x = event.pos().x()
        click_y = event.pos().y()

        slots_list = self._get_schematic_slots(w, h)
        n = len(slots_list)
        if n == 0:
            return

        perp_mode = self._perp_mode
        if perp_mode:
            slot_w_base, slot_h_base, gap_base = 90, 140, 12
        else:
            slot_w_base, slot_h_base, gap_base = 150, 80, 12

        max_draw_w = w - 80
        total_w_base = n * slot_w_base + (n - 1) * gap_base
        scale = max_draw_w / total_w_base if total_w_base > max_draw_w else 1.0

        slot_w = int(slot_w_base * scale)
        slot_h = int(slot_h_base * scale)
        gap = int(gap_base * scale)

        total_w = n * slot_w + (n - 1) * gap
        start_x = (w - total_w) // 2
        road_y = h // 2 - 10
        slot_y = road_y - slot_h - 4

        # Find which slot was clicked
        clicked_idx = -1
        for i in range(n):
            x1 = start_x + i * (slot_w + gap)
            x2 = x1 + slot_w
            y1 = slot_y
            y2 = slot_y + slot_h
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                clicked_idx = i
                break

        if clicked_idx != -1:
            cx, is_occupied, sz, fit, extra, orig_idx, slot_type = slots_list[clicked_idx]
            if is_occupied:
                self.status_lbl.setText("Hata: Secilen park alani dolu! Lutfen bos bir slot secin.")
            else:
                self._start_avp_simulation(clicked_idx, slots_list, w, h, scale)

    def _start_avp_simulation(self, target_idx, slots_list, w, h, scale):
        # 1) Stop any previous simulation
        self._sim_timer.stop()
        
        # 2) Set up simulation variables
        self._sim_active = True
        self._sim_target_idx = target_idx
        self._sim_step_idx = 0
        
        perp_mode = self._perp_mode
        n = len(slots_list)
        if perp_mode:
            slot_w_base, slot_h_base, gap_base = 90, 140, 12
        else:
            slot_w_base, slot_h_base, gap_base = 150, 80, 12

        slot_w = int(slot_w_base * scale)
        slot_h = int(slot_h_base * scale)
        gap = int(gap_base * scale)
        total_w = n * slot_w + (n - 1) * gap
        start_x = (w - total_w) // 2
        road_y = h // 2 - 10
        road_h = 160
        slot_y = road_y - slot_h - 4

        start_x_val = self._calculate_start_x(slots_list, start_x, slot_w, gap, w, scale)
        start_y_val = road_y + road_h // 2 + 25

        tx = start_x + target_idx * (slot_w + gap) + slot_w // 2
        ty = slot_y + slot_h // 2

        # 3) Generate points based on mode with smooth, realistic yaw and steering profiles
        points = []
        if perp_mode:
            # Perpendicular Path (Dik Park)
            align_x = tx + int(1.4 * slot_w)
            
            # Phase 1: Forward Alignment
            steps_1 = 15
            for k in range(steps_1):
                t = k / (steps_1 - 1)
                x = start_x_val + (align_x - start_x_val) * t
                points.append((x, start_y_val, 0.0, 0.0, "Adim 1/4: Hizalanma", "Park yerinin onunden gecerek ileriye dogru hizalanin."))
                
            # Phase 2: Turn and Reverse (Circular Arc / Bezier)
            steps_2 = 25
            for k in range(steps_2):
                t = k / (steps_2 - 1)
                x = (1-t)**2 * align_x + 2*(1-t)*t * (align_x - 10) + t**2 * tx
                y = (1-t)**2 * start_y_val + 2*(1-t)*t * start_y_val + t**2 * (road_y + 45)
                # Smooth S-curve (cubic) interpolation from 0.0 to 90.0 degrees
                yaw = 90.0 * (t ** 2 * (3 - 2 * t))
                steering = 35.0 * (1.0 - t)
                points.append((x, y, yaw, steering, "Adim 2/4: Geri Manevra", "Direksiyonu saga kirip yavasca geri gelin."))
                
            # Phase 3: Straight Back-in
            steps_3 = 15
            for k in range(steps_3):
                t = k / (steps_3 - 1)
                y = (road_y + 45) + (ty - (road_y + 45)) * t
                points.append((tx, y, 90.0, 0.0, "Adim 3/4: Parka Giris", "Direksiyonu duzeltip geri gelerek otoparka girin."))
                
            # Phase 4: Completed
            for _ in range(8):
                points.append((tx, ty, 90.0, 0.0, "Adim 4/4: Tamamlandi", "Park islemi basariyla tamamlandi. El freni cekildi."))
        else:
            # Parallel Path (Paralel Park)
            align_x = tx + int(1.1 * slot_w)
            align_y = start_y_val
            
            # Phase 1: Forward Alignment
            steps_1 = 15
            for k in range(steps_1):
                t = k / (steps_1 - 1)
                x = start_x_val + (align_x - start_x_val) * t
                points.append((x, align_y, 0.0, 0.0, "Adim 1/5: Hizalanma", "Yandaki aracin hizasina kadar duz ilerleyin ve durun."))
                
            # Phase 2: Reverse Angle
            steps_2 = 18
            for k in range(steps_2):
                t = k / (steps_2 - 1)
                x = align_x + (tx + int(0.5 * slot_w) - align_x) * t
                y = start_y_val + (road_y + 35 - start_y_val) * t
                # Smooth S-curve transition from 0.0 to 30.0 degrees
                yaw = 30.0 * (t ** 2 * (3 - 2 * t))
                points.append((x, y, yaw, 35.0, "Adim 2/5: Saga Manevra", "Direksiyonu tam saga kirin ve geri gelin (45 derece aciya kadar)."))
                
            # Phase 3: Reverse Counter-steer
            steps_3 = 20
            for k in range(steps_3):
                t = k / (steps_3 - 1)
                x = (tx + int(0.5 * slot_w)) + (tx - (tx + int(0.5 * slot_w))) * t
                y = (road_y + 35) + (ty + 5 - (road_y + 35)) * t
                # Smooth S-curve transition from 30.0 to 0.0 degrees
                yaw = 30.0 * (1.0 - (t ** 2 * (3 - 2 * t)))
                points.append((x, y, yaw, -35.0, "Adim 3/5: Sola Manevra", "Direksiyonu tam sola kirin ve yavasca geri gelmeye devam edin."))
                
            # Phase 4: Center Forward
            steps_4 = 10
            for k in range(steps_4):
                t = k / (steps_4 - 1)
                x = tx + (tx + int(0.05 * slot_w) - tx) * t
                y = (ty + 5) + (ty - (ty + 5)) * t
                points.append((x, y, 0.0, 0.0, "Adim 4/5: Duzeltme", "Direksiyonu duzeltip hafifce one dogru araci ortalayin."))
                
            # Phase 5: Completed
            for _ in range(8):
                points.append((tx + int(0.05 * slot_w), ty, 0.0, 0.0, "Adim 5/5: Tamamlandi", "Park islemi basariyla tamamlandi. El freni cekildi."))

        self._sim_points = points
        self._sim_path = [(pt[0], pt[1]) for pt in points]
        
        # Start timer
        self._sim_timer.start()
        
    def _tick_simulation(self):
        if not self._sim_active or self._sim_step_idx >= len(self._sim_points):
            self._sim_timer.stop()
            return
            
        pt = self._sim_points[self._sim_step_idx]
        self._sim_car_x = pt[0]
        self._sim_car_y = pt[1]
        self._sim_car_yaw = pt[2]
        self._sim_steering_angle = pt[3]
        self._sim_step_name = pt[4]
        self._sim_instruction = pt[5]
        
        self._sim_step_idx += 1
        
        self._update_schematic_map_ui()

    def _toggle_presentation_mode(self):
        self._presentation_mode = self.sunum_toggle_btn.isChecked()
        if self._presentation_mode:
            self.sunum_toggle_btn.setText("Analiz Paneli KAPAT")
            self.sunum_toggle_btn.setStyleSheet("background-color:#e11d48; color:white; font-weight:bold; border-radius:8px;")
        else:
            self.sunum_toggle_btn.setText("Analiz Paneli AÇ")
            self.sunum_toggle_btn.setStyleSheet("background-color:#475569; color:white; font-weight:bold; border-radius:8px;")
        
        if self._last_frame is not None:
            self._show_frame(self._last_frame)

    def _start_auto_demo(self):
        if not hasattr(self, "_current_empty_polys") or not self._current_empty_polys:
            QMessageBox.information(self, "Otomatik Demo", "Algılanmış boş park yeri bulunmuyor. Lütfen video oynatıldığından emin olun.")
            return
            
        best_idx = -1
        max_score = -1
        for i, score in enumerate(self._current_difficulties):
            is_fit = self._current_fit_flags[i] if i < len(self._current_fit_flags) else True
            if is_fit and score > max_score:
                max_score = score
                best_idx = i
                
        if best_idx == -1:
            best_idx = 0
            
        # AVP çizim alanının genişlik ve yüksekliğini al
        w = self.avp_draw_area.width()
        h = self.avp_draw_area.height()
        if w <= 0 or h <= 0:
            w, h = 800, 400

        slots_list = self._get_schematic_slots(w, h)
        n = len(slots_list)
        if n == 0:
            return

        perp_mode = self._perp_mode
        if perp_mode:
            slot_w_base, slot_h_base, gap_base = 90, 140, 12
        else:
            slot_w_base, slot_h_base, gap_base = 150, 80, 12

        max_draw_w = w - 80
        total_w_base = n * slot_w_base + (n - 1) * gap_base
        scale = max_draw_w / total_w_base if total_w_base > max_draw_w else 1.0

        target_idx = -1
        for idx, item in enumerate(slots_list):
            is_occupied = item[1]
            orig_idx = item[5]
            if not is_occupied and orig_idx == best_idx:
                target_idx = idx
                break

        if target_idx == -1:
            for idx, item in enumerate(slots_list):
                if not item[1]:
                    target_idx = idx
                    break

        if target_idx != -1:
            self._start_avp_simulation(target_idx, slots_list, w, h, scale)
            self.main_tabs.setCurrentIndex(1)
            self.status_lbl.setText(f"Otomatik Demo Başlatıldı: Slot {best_idx + 1} (Zorluk: {self._current_difficulties[best_idx]}%)")

    def _toggle_slam_mode(self):
        self._slam_active = self.slam_toggle_btn.isChecked()
        if self._slam_active:
            self.slam_toggle_btn.setText("SLAM Modu KAPAT")
            self.slam_toggle_btn.setStyleSheet(
                "background-color:#0891b2; color:white; font-weight:bold; border-radius:8px;"
            )
            # Reset cumulative offset so mapping starts fresh from current position
            self._slam_cum_dx = 0.0
            self._slam_cum_dy = 0.0
            self.status_lbl.setText("SLAM Haritalama Modu AÇık — Araç ilerledikçe harita birikir.")
        else:
            self.slam_toggle_btn.setText("SLAM Modu AÇ")
            self.slam_toggle_btn.setStyleSheet(
                "background-color:#475569; color:white; font-weight:bold; border-radius:8px;"
            )
            self.status_lbl.setText("SLAM Haritalama Modu KAPANDI.")

    def _reset_slam_map(self):
        self._slam_slots.clear()
        self._slam_cum_dx = 0.0
        self._slam_cum_dy = 0.0
        self._update_schematic_map_ui()
        self.status_lbl.setText("SLAM haritası sıfırlandı — Harita artık tamamen boş.")

    def _toggle_bsd_mode(self):
        self._bsd_active = self.bsd_toggle_btn.isChecked()
        if self._bsd_active:
            self.bsd_toggle_btn.setText("BSD Modu KAPAT")
            self.bsd_toggle_btn.setStyleSheet(
                "background-color:#ea580c; color:white; font-weight:bold; border-radius:8px;"
            )
            self.status_lbl.setText("BSD Kör Nokta Uyarı Sistemi AÇık — Yaklaşan nesneler izleniyor.")
        else:
            self.bsd_toggle_btn.setText("BSD Modu AÇ")
            self.bsd_toggle_btn.setStyleSheet(
                "background-color:#475569; color:white; font-weight:bold; border-radius:8px;"
            )
            self._bsd_alert_frame = 0
            self.status_lbl.setText("BSD Kör Nokta Uyarı Sistemi KAPANDI.")

    def _draw_bsd_overlay(self, out):
        """Kör Nokta Uyarı Sistemi: Arka/yan kör bölgelere giren yayalar,
        bisikletçiler veya motosikletler için ayna köşesi uyarı paneli çizer."""
        h, w = out.shape[:2]

        # Sınıflar: 0=yaya, 1=bisiklet, 3=motosiklet
        BSD_CLASSES = {0, 1, 3}
        # Kör bölge: alt %40 ve kamera kenarlarına yakın (sol %35, sağ %35)
        danger_zone_y_start = int(h * 0.60)
        danger_zone_lx_end  = int(w * 0.35)
        danger_zone_rx_start = int(w * 0.65)

        detections = getattr(self, "_current_detections", None) or []
        threat_left = False
        threat_right = False

        for det in detections:
            cls_id = det.get("class_id")
            if cls_id not in BSD_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, det["bbox"])
            cx_det = (x1 + x2) // 2
            cy_det = (y1 + y2) // 2

            if cy_det < danger_zone_y_start:
                continue  # Not in the rear blind zone

            if cx_det <= danger_zone_lx_end:
                threat_left = True
            if cx_det >= danger_zone_rx_start:
                threat_right = True

        if not (threat_left or threat_right):
            self._bsd_flash_state = False
            self._bsd_alert_frame = 0
            return

        # Toggle blink state
        self._bsd_alert_frame += 1
        self._bsd_flash_state = (self._bsd_alert_frame // 2) % 2 == 0

        # Beep once per new alert trigger
        if self._bsd_alert_frame == 1:
            from PyQt5.QtWidgets import QApplication
            QApplication.beep()

        if not self._bsd_flash_state:
            return  # Blink off — skip drawing

        alert_color_bgr  = (0, 80, 255)   # Amber orange in BGR
        border_color_bgr = (0, 40, 200)
        mirror_h = int(h * 0.28)
        mirror_w = int(w * 0.12)
        mirror_y = int(h * 0.60)

        # Left mirror warning panel
        if threat_left:
            lx1, ly1 = 10, mirror_y
            lx2, ly2 = lx1 + mirror_w, mirror_y + mirror_h
            sub = out[ly1:ly2, lx1:lx2]
            overlay = sub.copy()
            cv2.rectangle(overlay, (0, 0), (mirror_w, mirror_h), alert_color_bgr, -1)
            cv2.addWeighted(overlay, 0.55, sub, 0.45, 0, sub)
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), border_color_bgr, 3, cv2.LINE_AA)
            cv2.putText(out, "!", (lx1 + mirror_w // 2 - 6, ly1 + mirror_h // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(out, "KOR NOKTA", (lx1 + 4, ly2 - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(out, "SOL TARAF", (lx1 + 4, ly2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        # Right mirror warning panel
        if threat_right:
            rx2, ry1 = w - 10, mirror_y
            rx1, ry2 = rx2 - mirror_w, mirror_y + mirror_h
            sub = out[ry1:ry2, rx1:rx2]
            overlay = sub.copy()
            cv2.rectangle(overlay, (0, 0), (mirror_w, mirror_h), alert_color_bgr, -1)
            cv2.addWeighted(overlay, 0.55, sub, 0.45, 0, sub)
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), border_color_bgr, 3, cv2.LINE_AA)
            cv2.putText(out, "!", (rx1 + mirror_w // 2 - 6, ry1 + mirror_h // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(out, "KOR NOKTA", (rx1 + 4, ry2 - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(out, "SAG TARAF", (rx1 + 4, ry2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        # Central top warning banner
        banner_txt = "DİKKAT: KÖR NOKTA UYARISI"
        (tw, th), _ = cv2.getTextSize(banner_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bx = w // 2 - tw // 2 - 10
        by = h - 55
        cv2.rectangle(out, (bx - 6, by - th - 6), (bx + tw + 16, by + 8), (0, 40, 200), -1)
        cv2.rectangle(out, (bx - 6, by - th - 6), (bx + tw + 16, by + 8), border_color_bgr, 2, cv2.LINE_AA)
        cv2.putText(out, banner_txt, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        self.status_lbl.setText("⚠️  KÖR NOKTA UYARISI — Yaklaşan nesne tespit edildi!")

    def _draw_performance_hud(self, out):
        h, w = out.shape[:2]
        card_w = 260
        card_h = 175
        x1_card = w - card_w - 20
        y1_card = 20
        x2_card = w - 20
        y2_card = y1_card + card_h
        
        overlay = out.copy()
        cv2.rectangle(overlay, (x1_card, y1_card), (x2_card, y2_card), (30, 22, 15), -1)
        cv2.rectangle(overlay, (x1_card, y1_card), (x2_card, y2_card), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        
        cv2.putText(out, "JURI SUNUM VE ANALIZ MODU", (x1_card + 12, y1_card + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(out, (x1_card + 10, y1_card + 30), (x2_card - 10, y1_card + 30), (70, 70, 70), 1)
        
        fps = self._latencies.get("fps", 0.0)
        total = self._latencies.get("total", 0.0)
        cv2.putText(out, f"FPS: {fps:.1f}", (x1_card + 12, y1_card + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"Toplam Gecikme: {total:.1f} ms", (x1_card + 12, y1_card + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        
        metrics = [
            ("YOLO Det", self._latencies.get("yolo", 0.0), (0, 255, 0)),
            ("Lane Seg", self._latencies.get("lane", 0.0), (255, 0, 0)),
            ("Slot Eval", self._latencies.get("slot", 0.0), (0, 255, 255)),
            ("UI Render", self._latencies.get("draw", 0.0), (0, 150, 255))
        ]
        
        bar_y = y1_card + 84
        for label, val, bar_color in metrics:
            cv2.putText(out, f"{label}: {val:.1f} ms", (x1_card + 12, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
            
            bar_x = x1_card + 100
            bar_w_max = 140
            cv2.rectangle(out, (bar_x, bar_y - 8), (bar_x + bar_w_max, bar_y - 2), (50, 50, 50), -1)
            
            val_clamped = max(0.0, min(val, 100.0))
            filled_w = int(bar_w_max * (val_clamped / 100.0))
            if filled_w > 0:
                cv2.rectangle(out, (bar_x, bar_y - 8), (bar_x + filled_w, bar_y - 2), bar_color, -1)
            
            bar_y += 18
            bar_w_max = 140
            cv2.rectangle(out, (bar_x, bar_y - 8), (bar_x + bar_w_max, bar_y - 2), (50, 50, 50), -1)
            
            val_clamped = max(0.0, min(val, 100.0))
            filled_w = int(bar_w_max * (val_clamped / 100.0))
            if filled_w > 0:
                cv2.rectangle(out, (bar_x, bar_y - 8), (bar_x + filled_w, bar_y - 2), bar_color, -1)
            
            bar_y += 18
