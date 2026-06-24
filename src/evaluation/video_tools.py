"""Video araçları — offline değerlendirme için kare çıkarma.

Canlı sistemden bağımsız: rapor/doğruluk ölçümü amacıyla videodan eşit
aralıklı kareler çıkarır. Kullanıcı bu kareleri boş/dolu etiketler
(data/ground_truth.json formatında), sonra runner ile metrik üretilir.
"""

from __future__ import annotations

from pathlib import Path

import cv2


def extract_frames(video_path: str, out_dir: str, count: int = 20,
                   prefix: str = "frame") -> list[str]:
    """Videodan eşit aralıklı `count` kare çıkar, PNG olarak kaydet.

    Döner: kaydedilen dosya yollarının listesi.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = []
    try:
        if total <= 0:
            # kare sayısı bilinmiyor: sırayla oku
            idx = 0
            while len(saved) < count:
                ret, frame = cap.read()
                if not ret:
                    break
                p = out_dir / f"{prefix}_{idx:04d}.png"
                cv2.imwrite(str(p), frame)
                saved.append(str(p))
                idx += 1
            return saved
        step = max(1, total // count)
        for i in range(count):
            pos = min(total - 1, i * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue
            p = out_dir / f"{prefix}_{pos:05d}.png"
            cv2.imwrite(str(p), frame)
            saved.append(str(p))
    finally:
        cap.release()
    return saved
