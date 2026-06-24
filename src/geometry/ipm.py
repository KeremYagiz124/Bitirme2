"""Inverse Perspective Mapping (IPM) — kuş bakışı dönüşümü.

Çapraz açılı kamera görüntülerinde aynı hizadaki farklı derinlikteki nesneler
üst üste biner ve 2B ölçüm yaklaşık kalır. IPM, zemin düzlemini bir homografi
ile kuş bakışına (bird's eye view) çevirir; bu görünümde mesafeler doğrusaldır
ve gerçek metrik ölçüm yapılabilir.

Kalibrasyon: kullanıcı görüntü üzerinde zemindeki bir dikdörtgenin (ör. park
alanı sınırı) 4 köşesini işaretler. Bu 4 nokta, çıktı dikdörtgeninin köşelerine
eşlenerek homografi (3x3) hesaplanır.

Gerçek metrik: işaretlenen dikdörtgenin gerçek-dünya genişlik/yüksekliği
(metre) verilirse, kuş bakışı görüntüde m/px ölçeği sabittir ve her yerde
geçerlidir (perspektif bozulması olmadığı için).
"""

from __future__ import annotations

import cv2
import numpy as np


class PerspectiveTransformer:
    """Zemin düzlemi homografisi ile perspektif/kuş bakışı dönüşümü."""

    def __init__(self, H: np.ndarray | None = None,
                 out_size: tuple[int, int] | None = None,
                 m_per_px: float | None = None):
        self.H = H                       # 3x3 homografi (kaynak → kuş bakışı)
        self.out_size = out_size         # (genişlik, yükseklik) piksel
        self.m_per_px = m_per_px         # kuş bakışı sabit ölçeği (metre/piksel)
        self._H_inv = None               # kuş bakışı → kaynak (lazy)

    # ── Ters dönüşüm (kuş bakışı → kaynak görüntü) ───────────────────────────

    @property
    def H_inv(self) -> np.ndarray | None:
        if self._H_inv is None and self.H is not None:
            self._H_inv = np.linalg.inv(self.H)
        return self._H_inv

    def inverse_transform_points(self, points) -> np.ndarray:
        """Kuş bakışı nokta(lar)ı kaynak görüntü koordinatına geri taşı."""
        if self.H is None:
            raise RuntimeError("Önce kalibrasyon yapılmalı")
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H_inv.astype(np.float32))
        return out.reshape(-1, 2)

    def inverse_transform_quad(self, bbox) -> np.ndarray:
        """Kuş bakışı eksen-hizalı bbox'ın 4 köşesini kaynak görüntüye taşı.

        Kaynakta dörtgen (perspektif) olur; çizim için poligon olarak kullanılır.
        Döner: (4,2) köşe dizisi (sol-üst, sağ-üst, sağ-alt, sol-alt).
        """
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return self.inverse_transform_points(corners)

    # ── Kalibrasyon ──────────────────────────────────────────────────────────

    @classmethod
    def from_quad(cls, src_pts, out_w: int, out_h: int,
                  real_w_m: float | None = None,
                  real_h_m: float | None = None,
                  dst_pts: np.ndarray | list | None = None) -> "PerspectiveTransformer":
        """Zemindeki 4 köşeden homografi kur.

        src_pts: kaynak görüntüde 4 nokta [(x,y), ...] sırası:
                 sol-üst, sağ-üst, sağ-alt, sol-alt
        out_w, out_h: kuş bakışı çıktı boyutu (piksel)
        real_w_m, real_h_m: dikdörtgenin gerçek genişlik/yükseklik (metre)
                            → verilirse m_per_px hesaplanır.
        dst_pts: kuş bakışı görüntüdeki hedef 4 nokta [(x,y), ...].
                 Belirtilmezse çıktı görüntüsünün tam köşeleri kullanılır.
        """
        src = np.asarray(src_pts, dtype=np.float32)
        if src.shape != (4, 2):
            raise ValueError("src_pts tam olarak 4 (x,y) nokta olmalı")
        if dst_pts is not None:
            dst = np.asarray(dst_pts, dtype=np.float32)
        else:
            dst = np.array([[0, 0], [out_w - 1, 0],
                            [out_w - 1, out_h - 1], [0, out_h - 1]],
                           dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)

        m_per_px = None
        if real_w_m is not None and real_h_m is not None:
            if dst_pts is not None:
                dst_w = float(np.linalg.norm(dst[1] - dst[0]))
                dst_h = float(np.linalg.norm(dst[2] - dst[1]))
            else:
                dst_w = out_w
                dst_h = out_h
            sx = real_w_m / dst_w
            sy = real_h_m / dst_h
            m_per_px = float((sx + sy) / 2.0)
        return cls(H=H, out_size=(out_w, out_h), m_per_px=m_per_px)

    # ── Dönüşümler ───────────────────────────────────────────────────────────

    def warp_image(self, frame: np.ndarray) -> np.ndarray:
        """Görüntüyü kuş bakışına çevir."""
        if self.H is None or self.out_size is None:
            raise RuntimeError("Önce kalibrasyon yapılmalı (from_quad)")
        return cv2.warpPerspective(frame, self.H, self.out_size)

    def transform_points(self, points) -> np.ndarray:
        """Nokta(lar)ı homografi ile kuş bakışına taşı. Döner: (N,2) array."""
        if self.H is None:
            raise RuntimeError("Önce kalibrasyon yapılmalı")
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def transform_box(self, bbox, ref_car_length_m: float = 4.5) -> tuple[float, float, float, float]:
        """Araç 2B kutusunu kuş bakışı (BEV) düzlemine saptırarak aktarır.

        Boyut distorsiyonunu (yükseklik hatasını) önlemek için yalnızca zeminle
        temas eden alt köşeleri homografi ile taşır. Derinlik boyutunu (boyunu)
        ise metrik araç uzunluğuna (ref_car_length_m) göre BEV plane üzerinde kurgular.
        """
        x1, y1, x2, y2 = bbox
        # Sadece zemin/tekerlek temas noktalarını (alt köşeler) projekte et
        bottom_corners = [(x1, y2), (x2, y2)]
        tp = self.transform_points(bottom_corners)
        bx1, by1 = tp[0]
        bx2, by2 = tp[1]

        # Kuş bakışında aracın genişliği
        xs = [bx1, bx2]
        min_x, max_x = min(xs), max(xs)

        # Araç boyunu (dikey eksen) metrik ölçeğe göre BEV üzerinde oluştur.
        # Kamera öne/aşağı baktığı için aracın gövdesi BEV düzleminde yukarı (uzaklaşan yöne) uzanır.
        # Metrik ölçek (m_per_px) kalibre edilmişse gerçek araç uzunluğunu kullan.
        # Kalibre edilmemişse piksel bazlı varsayılan bir oran (ör. genişliğin 2.2 katı) kullan.
        if self.m_per_px and self.m_per_px > 0:
            length_px = ref_car_length_m / self.m_per_px
        else:
            # Yedek: genişliğe göre makul bir oran (boy ≈ 2.2 * en)
            width_px = abs(bx2 - bx1)
            length_px = width_px * 2.2

        min_y = min(by1, by2) - length_px
        max_y = max(by1, by2)

        return float(min_x), float(min_y), float(max_x), float(max_y)

    # ── Metrik ölçüm ─────────────────────────────────────────────────────────

    def measure_distance_m(self, p1, p2) -> float | None:
        """Kaynak görüntüdeki iki nokta arası gerçek mesafe (metre).

        Noktalar kuş bakışına taşınır, Öklid mesafesi m_per_px ile çarpılır.
        m_per_px kalibre edilmemişse None döner.
        """
        if self.m_per_px is None:
            return None
        tp = self.transform_points([p1, p2])
        d_px = float(np.hypot(tp[1, 0] - tp[0, 0], tp[1, 1] - tp[0, 1]))
        return d_px * self.m_per_px

    def box_size_m(self, bbox) -> tuple[float, float] | None:
        """Bbox'ın kuş bakışındaki gerçek (genişlik_m, yükseklik_m) boyutu."""
        if self.m_per_px is None:
            return None
        bx1, by1, bx2, by2 = self.transform_box(bbox)
        return ((bx2 - bx1) * self.m_per_px, (by2 - by1) * self.m_per_px)
