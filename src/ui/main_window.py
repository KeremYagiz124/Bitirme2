import cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFrame, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("🚗 Smart Parking AI")
        self.setGeometry(100, 100, 1400, 800)

        # 🔥 MODEL (başta yüklenmez)
        self.model = None

        # 🎥 VIDEO
        self.video_label = QLabel("Camera / Video Feed")
        self.video_label.setObjectName("video")
        self.video_label.setFixedSize(900, 650)
        self.video_label.setAlignment(Qt.AlignCenter)

        # 🔘 BUTTONS
        self.start_btn = QPushButton("▶ Kamera")
        self.stop_btn = QPushButton("■ Durdur")
        self.load_btn = QPushButton("📂 Video Yükle")

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.load_btn.clicked.connect(self.load_video)

        # 📊 INFO
        self.info_label = QLabel("Durum: Hazır")

        # 📦 PANEL
        panel = QVBoxLayout()
        panel.setSpacing(15)
        panel.addWidget(self.start_btn)
        panel.addWidget(self.stop_btn)
        panel.addWidget(self.load_btn)
        panel.addSpacing(20)
        panel.addWidget(self.info_label)

        panel_frame = QFrame()
        panel_frame.setLayout(panel)

        # 🧱 LAYOUT
        layout = QHBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(panel_frame)

        self.setLayout(layout)

        self.timer = QTimer()
        self.cap = None

    # 🎥 Kamera
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # 📂 Video yükle
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Video Seç", "", "Video Files (*.mp4 *.avi)"
        )

        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)

    # ⛔ Durdur
    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()

    # 🔥 YOLO (güvenli yükleme)
    def load_model(self):
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt")
                self.info_label.setText("Model yüklendi ✅")
            except Exception as e:
                self.info_label.setText("Model yüklenemedi ❌")
                print(e)

    # 🎯 Frame işle
    def update_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # 🔥 model ilk frame’de yüklenir
        if self.model is None:
            self.load_model()

        car_count = 0

        if self.model:
            results = self.model(frame)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])

                    if cls == 2:  # 🚗 car
                        car_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame, "CAR", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        self.info_label.setText(f"🚗 Araç Sayısı: {car_count}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))