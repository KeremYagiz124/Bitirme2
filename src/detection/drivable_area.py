"""Sürülebilir alan (drivable area) + şerit çizgisi segmentasyonu.

YOLOPv2 (TorchScript, BDD100K ile eğitilmiş) ile yol yüzeyini ve şerit
çizgilerini segment eder. Sürücü-kamera domaininde eğitildiği için bizim
senaryomuza doğrudan uyar — kaldırım, refüj, bina gibi yol-dışı alanları
otomatik dışlar.

Model ağır (~58 ms @1050 Ti) olduğundan main_window seyrek çağırır + cache'ler;
sahne yavaş değiştiği için her ~1 sn'de bir yenilemek yeterlidir.

Model dosyası yoksa `available=False` döner → sistem klasik LAB road mask'e
düşer (graceful degradation).
"""

import os

import cv2
import numpy as np


class DrivableAreaSegmenter:
    INPUT_H = 384
    INPUT_W = 640

    def __init__(self, model_path: str = "models/yolopv2.pt", device: str | None = None):
        self.available = False
        self.model = None
        self.device = "cpu"
        self._torch = None
        if not os.path.exists(model_path):
            return
        try:
            import torch
            self._torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            self.available = True
        except Exception:
            self.available = False
            self.model = None

    @staticmethod
    def _letterbox(im, new_shape=(384, 640), color=114):
        """Aspect-ratio koruyan resize + padding (YOLO ailesi standardı)."""
        h, w = im.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        im_r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = new_shape[1] - nw, new_shape[0] - nh
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        out = cv2.copyMakeBorder(im_r, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(color, color, color))
        return out, (left, top)

    def _unletterbox(self, mask, pad, out_shape):
        """Letterbox padding'i kaldır → orijinal frame boyutuna döndür."""
        px, py = pad
        mh, mw = mask.shape
        crop = mask[py:mh - py if py else mh, px:mw - px if px else mw]
        return cv2.resize(crop, (out_shape[1], out_shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    def infer(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns (drivable_mask, lane_mask) — orijinal boyutta uint8 0/255.

        Model yoksa (None, None) döner. Letterbox preprocessing kullanır —
        aspect-ratio bozulması segmentasyon kalitesini ciddi düşürür.
        """
        if not self.available or self.model is None:
            return None, None
        torch = self._torch
        h0, w0 = frame.shape[:2]
        lb, pad = self._letterbox(frame)
        img = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(t)
        # out: (det, drivable[1,2,H,W], lane[1,1,H,W])
        da_mask = out[1].argmax(1).squeeze().to("cpu").numpy().astype(np.uint8) * 255
        ll_t = out[2].squeeze().to("cpu")
        ll_mask = (torch.sigmoid(ll_t).numpy() > 0.5).astype(np.uint8) * 255

        da_full = self._unletterbox(da_mask, pad, (h0, w0))
        ll_full = self._unletterbox(ll_mask, pad, (h0, w0))
        return da_full, ll_full
