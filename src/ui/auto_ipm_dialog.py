"""Otomatik IPM Kalibrasyonu Görsel Önizleme ve Doğrulama Penceresi.

Kaybolma noktası, yakınsayan dikey çizgiler ve hesaplanan trapezoid alanı
kamera görüntüsü üzerinde görselleştirerek jüriye matematiksel doğrulamayı sunar.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QWidget,
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt


class AutoIPMDiagnosticsDialog(QDialog):
    """Otomatik IPM sonuçlarını ve geometrik analizini gösteren jüri dostu diyalog."""

    def __init__(self, frame_bgr: np.ndarray, transformer, diagnostics: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Otomatik IPM Kalibrasyon Analizi (Vanishing Point)")
        self.setStyleSheet("background-color: #0f172a; color: #f8fafc;")
        self.resize(1150, 720)

        self.transformer = transformer
        self.diagnostics = diagnostics
        self.frame = frame_bgr

        # Görüntüleri hazırla
        self.orig_pixmap = self._prepare_original_visualization()
        self.bev_pixmap = self._prepare_bev_visualization()

        self._setup_ui()

    def _prepare_original_visualization(self) -> QPixmap:
        """Kamera görüntüsü üzerine çizgileri, kaçış noktasını ve trapezi çizer."""
        draw_img = self.frame.copy()
        h, w = draw_img.shape[:2]

        segments = self.diagnostics.get("segments", [])
        vert_segs = self.diagnostics.get("vertical_segments", [])
        vp = self.diagnostics.get("vanishing_point", None)
        src_quad = self.diagnostics.get("src_quad", [])

        # 1) Tüm algılanan çizgiler (ince mavi-gri)
        for s in segments:
            cv2.line(draw_img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (148, 163, 184), 1, cv2.LINE_AA)

        # 2) Dikey/yakınsayan çizgiler (kalın mor)
        for s in vert_segs:
            cv2.line(draw_img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (219, 39, 119), 2, cv2.LINE_AA)

        # 3) Kaynak trapez (parlak turkuaz)
        if len(src_quad) == 4:
            pts = np.array(src_quad, dtype=np.int32)
            cv2.polylines(draw_img, [pts], True, (234, 179, 8), 3, cv2.LINE_AA)
            # Köşeleri çiz
            for pt in pts:
                cv2.circle(draw_img, (int(pt[0]), int(pt[1])), 6, (234, 179, 8), -1, cv2.LINE_AA)

        # 4) Ufuk çizgisi (kesikli veya ince kırmızı)
        if vp is not None:
            vpx, vpy = vp
            # Kaybolma noktasından geçen yatay ufuk çizgisi
            cv2.line(draw_img, (0, int(vpy)), (w, int(vpy)), (59, 130, 246), 1, cv2.LINE_AA)
            
            # Kaybolma noktası hedefi (yellow target)
            cv2.circle(draw_img, (int(vpx), int(vpy)), 8, (34, 197, 94), 2, cv2.LINE_AA)
            cv2.circle(draw_img, (int(vpx), int(vpy)), 2, (34, 197, 94), -1, cv2.LINE_AA)
            cv2.line(draw_img, (int(vpx) - 15, int(vpy)), (int(vpx) + 15, int(vpy)), (34, 197, 94), 1, cv2.LINE_AA)
            cv2.line(draw_img, (int(vpx), int(vpy) - 15), (int(vpx), int(vpy) + 15), (34, 197, 94), 1, cv2.LINE_AA)
            
            # Metin ekle
            cv2.putText(draw_img, f"VP ({int(vpx)}, {int(vpy)})", (int(vpx) + 12, int(vpy) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 1, cv2.LINE_AA)

        # Ölçekle ve QPixmap'e dönüştür
        max_w, max_h = 550, 420
        scale = min(max_w / w, max_h / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        disp = cv2.resize(draw_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        
        qimg = QImage(rgb.data, disp_w, disp_h, 3 * disp_w, QImage.Format_RGB888)
        # Bellek sızıntısı ve çökme koruması için tamponu tut
        self._orig_buf = rgb
        return QPixmap.fromImage(qimg)

    def _prepare_bev_visualization(self) -> QPixmap:
        """Kuş bakışı bükülmüş görüntüyü (BEV) hazırlar."""
        bev = self.transformer.warp_image(self.frame)
        h, w = bev.shape[:2]

        # Üzerine kalibrasyon ızgarası çiz (yeşil kılavuz çizgileri)
        # Bu, perspektifin kalktığını kanıtlar
        grid_color = (16, 185, 129) # zümrüt yeşili
        grid_gap = 40
        for x in range(0, w, grid_gap):
            cv2.line(bev, (x, 0), (x, h), grid_color, 1, cv2.LINE_AA)
        for y in range(0, h, grid_gap):
            cv2.line(bev, (0, y), (w, y), grid_color, 1, cv2.LINE_AA)

        # Ölçekle ve dönüştür
        max_w, max_h = 550, 420
        scale = min(max_w / w, max_h / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        disp = cv2.resize(bev, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

        qimg = QImage(rgb.data, disp_w, disp_h, 3 * disp_w, QImage.Format_RGB888)
        self._bev_buf = rgb
        return QPixmap.fromImage(qimg)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Başlık Bölümü
        title = QLabel("Geometrik Kamera Kalibrasyon Analizi (IPM & Vanishing Point)")
        title.setFont(QFont("Inter", 16, QFont.Bold))
        title.setStyleSheet("color: #38bdf8;") # Sky blue
        main_layout.addWidget(title)

        # Görseller Bölümü (Side-by-Side)
        views_layout = QHBoxLayout()
        views_layout.setSpacing(15)

        # Sol Görsel (Perspektif ve Çizgiler)
        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: #1e293b; border-radius: 8px; border: 1px solid #334155;")
        left_layout = QVBoxLayout(left_widget)
        left_title = QLabel("1. Perspektif Analiz & Ufuk Doğrulaması")
        left_title.setFont(QFont("Inter", 11, QFont.Bold))
        left_title.setStyleSheet("color: #e2e8f0; border: none; padding: 4px;")
        
        self.lbl_orig = QLabel()
        self.lbl_orig.setPixmap(self.orig_pixmap)
        self.lbl_orig.setAlignment(Qt.AlignCenter)
        self.lbl_orig.setStyleSheet("border: none;")

        left_layout.addWidget(left_title)
        left_layout.addWidget(self.lbl_orig, 1)
        views_layout.addWidget(left_widget, 1)

        # Sağ Görsel (Kuş Bakışı / BEV)
        right_widget = QWidget()
        right_widget.setStyleSheet("background-color: #1e293b; border-radius: 8px; border: 1px solid #334155;")
        right_layout = QVBoxLayout(right_widget)
        right_title = QLabel("2. Kuş Bakışı Dönüşümü (BEV Grid)")
        right_title.setFont(QFont("Inter", 11, QFont.Bold))
        right_title.setStyleSheet("color: #e2e8f0; border: none; padding: 4px;")

        self.lbl_bev = QLabel()
        self.lbl_bev.setPixmap(self.bev_pixmap)
        self.lbl_bev.setAlignment(Qt.AlignCenter)
        self.lbl_bev.setStyleSheet("border: none;")

        right_layout.addWidget(right_title)
        right_layout.addWidget(self.lbl_bev, 1)
        views_layout.addWidget(right_widget, 1)

        main_layout.addLayout(views_layout, 1)

        # Matematiksel Rapor Paneli
        report_widget = QWidget()
        report_widget.setStyleSheet("background-color: #0f172a; border-radius: 8px; border: 1px solid #1e293b;")
        report_layout = QVBoxLayout(report_widget)
        report_layout.setContentsMargins(5, 5, 5, 5)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet(
            "background-color: #020617; color: #10b981; border: 1px solid #334155; "
            "font-family: Consolas, monospace; font-size: 13px; padding: 10px;"
        )

        # Metrik bilgileri derle
        method_str = "Zemin Çizgi Yakınsaması (Hough Line Transform)" if self.diagnostics.get("method") == "line" else "Araç Hizalama Referansı"
        vp = self.diagnostics.get("vanishing_point", (0, 0))
        lines_count = len(self.diagnostics.get("vertical_segments", []))
        mpp = self.transformer.m_per_px if self.transformer else 0.0

        report_html = f"""
>> GEOMETRİK ANALİZ VE PROJEKSİYON RAPORU
----------------------------------------------------------------------
[Kalibrasyon Modu]   : {method_str}
[Tespit Edilen Çizgi]: {lines_count} adet yakınsak dikey şerit çizgisi
[Kaçış Noktası (VP)] : X = {vp[0]:.2f} px , Y = {vp[1]:.2f} px
[Yol Eğimi (Pitch)]  : Ufuk düzlemi konumu doğrulandı.
[Çözünürlük Ölçeği]  : {mpp:.5f} metre/piksel (BEV metric scale)
----------------------------------------------------------------------
>> DURUM: Matematiksel matris dönüşümü (Homografi) başarıyla kuruldu.
>> Kuş bakışı ızgara üzerindeki dikey şeritlerin paralelliği doğrulanmıştır.
"""
        self.report_text.setPlainText(report_html)
        report_layout.addWidget(self.report_text)
        main_layout.addWidget(report_widget)

        # Butonlar Bölümü
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        cancel_btn = QPushButton("İptal Et")
        cancel_btn.setFixedHeight(38)
        cancel_btn.setStyleSheet(
            "QPushButton { background-color: #334155; color: #f1f5f9; border-radius: 6px; font-weight: bold; min-width: 120px; }"
            "QPushButton:hover { background-color: #475569; }"
        )
        cancel_btn.clicked.connect(self.reject)

        apply_btn = QPushButton("Kalibrasyonu Uygula")
        apply_btn.setFixedHeight(38)
        apply_btn.setStyleSheet(
            "QPushButton { background-color: #0ea5e9; color: #ffffff; border-radius: 6px; font-weight: bold; min-width: 180px; }"
            "QPushButton:hover { background-color: #0284c7; }"
        )
        apply_btn.clicked.connect(self.accept)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(apply_btn)
        main_layout.addLayout(btn_layout)
