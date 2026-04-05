import time
import cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFrame, QFileDialog, QGridLayout
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

from src.detection.vehicle_detector import VehicleDetector

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
    lbl.setFont(QFont("Arial", 9, QFont.Bold))
    lbl.setStyleSheet("color: #94a3b8; padding-top: 10px;")
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
        self.setGeometry(100, 100, 1400, 820)
        self.setMinimumSize(1100, 700)

        self.detector = None
        self.cap = None
        self._last_time = time.time()
        self._fps = 0.0

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        # ── Video alanı ──────────────────────────────────────────
        self.video_label = QLabel("Video / Kamera Bekleniyor")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #0f172a;
            border-radius: 12px;
            color: #475569;
            font-size: 14px;
        """)
        self.video_label.setMinimumSize(860, 640)

        # ── Sağ panel ─────────────────────────────────────────────
        panel = QVBoxLayout()
        panel.setSpacing(8)
        panel.setContentsMargins(12, 16, 12, 16)

        # Başlık
        title = QLabel("Smart Parking AI")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #3b82f6;")
        title.setAlignment(Qt.AlignCenter)
        panel.addWidget(title)

        subtitle = QLabel("YOLOv8 · Araç Tespiti")
        subtitle.setStyleSheet("color: #64748b; font-size: 10px;")
        subtitle.setAlignment(Qt.AlignCenter)
        panel.addWidget(subtitle)

        # Kontroller
        panel.addWidget(make_section_label("KONTROL"))

        self.start_btn = self._btn("▶  Kamera Başlat", "#2563eb")
        self.stop_btn  = self._btn("■  Durdur",        "#dc2626")
        self.img_btn   = self._btn("🖼  Resim Yükle",   "#7c3aed")
        self.vid_btn   = self._btn("📂  Video Yükle",   "#0891b2")

        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_feed)
        self.img_btn.clicked.connect(self.load_image)
        self.vid_btn.clicked.connect(self.load_video)

        for b in (self.start_btn, self.stop_btn, self.img_btn, self.vid_btn):
            panel.addWidget(b)

        # İstatistikler
        panel.addWidget(make_section_label("TESPİT"))

        self.stat_cards = {}
        icons = {2: "🚗", 3: "🏍", 5: "🚌", 7: "🚛"}
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, (cls_id, name) in enumerate(VEHICLE_CLASSES.items()):
            card = StatCard(icons[cls_id], name, VEHICLE_COLORS_HEX[cls_id])
            self.stat_cards[cls_id] = card
            grid.addWidget(card, i // 2, i % 2)
        panel.addLayout(grid)

        # Durum
        panel.addWidget(make_section_label("DURUM"))

        self.status_lbl = QLabel("Hazır")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color: #e2e8f0; font-size: 12px;")
        panel.addWidget(self.status_lbl)

        self.fps_lbl = QLabel("FPS: —")
        self.fps_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        panel.addWidget(self.fps_lbl)

        panel.addStretch()

        panel_frame = QFrame()
        panel_frame.setFixedWidth(280)
        panel_frame.setStyleSheet("background-color: #0f172a; border-radius: 12px;")
        panel_frame.setLayout(panel)

        # ── Ana layout ────────────────────────────────────────────
        root = QHBoxLayout()
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        root.addWidget(self.video_label, stretch=1)
        root.addWidget(panel_frame)
        self.setLayout(root)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def _btn(self, text, color):
        b = QPushButton(text)
        b.setFixedHeight(38)
        b.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 8px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
            QPushButton:disabled {{ background-color: #334155; color: #64748b; }}
        """)
        return b

    # ── Model ─────────────────────────────────────────────────────
    def _load_model(self):
        self.status_lbl.setText("Model yükleniyor...")
        try:
            self.detector = VehicleDetector()
            self.status_lbl.setText("Model yüklendi.")
        except Exception as e:
            self.status_lbl.setText(f"Model hatası:\n{e}")

    # ── Feed yönetimi ─────────────────────────────────────────────
    def _start_feed(self):
        self._last_time = time.time()
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self.vid_btn.setEnabled(False)
        self.img_btn.setEnabled(False)
        self.timer.start(30)

    def start_camera(self):
        self.stop_feed()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_lbl.setText("Kamera bulunamadı.")
            self.cap = None
            return
        self.status_lbl.setText("Kamera aktif.")
        self._start_feed()

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Video Seç", "", "Video (*.mp4 *.avi *.mkv *.mov)"
        )
        if not path:
            return
        self.stop_feed()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.status_lbl.setText("Video açılamadı.")
            self.cap = None
            return
        self.status_lbl.setText("Video oynatılıyor.")
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

    # ── Frame işleme ──────────────────────────────────────────────
    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_feed()
            self.status_lbl.setText("Bitti.")
            return

        self._process_and_show(frame)

        # FPS
        now = time.time()
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(now - self._last_time, 1e-6))
        self._last_time = now
        self.fps_lbl.setText(f"FPS: {self._fps:.1f}")

    def _process_and_show(self, frame):
        counts = {cls_id: 0 for cls_id in VEHICLE_CLASSES}

        if self.detector:
            detections = self.detector.detect(frame)
            for det in detections:
                cls_id = det.get("class_id")
                if cls_id not in VEHICLE_CLASSES:
                    continue
                counts[cls_id] += 1
                x1, y1, x2, y2 = map(int, det["bbox"])
                conf = det["confidence"]
                color = VEHICLE_COLORS_CV[cls_id]
                label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for cls_id, card in self.stat_cards.items():
            card.set_count(counts[cls_id])

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
