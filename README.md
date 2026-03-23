# Park Detection AI

Kamera görüntülerinden araç tespiti ve park uygunluğu analizi için yapay zeka tabanlı sistem.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Baseline Test
```bash
# Resim üzerinde test
python scripts/baseline_test.py --source path/to/image.jpg

# Video üzerinde test
python scripts/baseline_test.py --source path/to/video.mp4

# Webcam
python scripts/baseline_test.py --source 0
```

## Proje Yapısı

```
├── configs/          # Konfigürasyon dosyaları
├── data/             # Veri setleri (git'e eklenmez)
├── models/           # Model ağırlıkları
├── notebooks/        # Jupyter notebook'lar
├── outputs/          # Çıktılar (görselleştirme, metrikler)
├── scripts/          # Çalıştırılabilir scriptler
├── src/
│   ├── detection/    # YOLOv8 araç tespiti
│   ├── parking/      # OpenCV park alanı analizi
│   ├── classification/ # MobileNetV2 sınıflandırma
│   └── utils/        # Yardımcı fonksiyonlar
└── tests/            # Testler
```

## Çıktı Sınıfları

- **Park Edilebilir**: Araç park edebilir
- **Park Edilemez**: Yasak bölge / çizgi üstü
- **Park Alanı Dolu**: Slot dolu

## Metrikler

Accuracy, Precision, Recall, F1-Score, mAP — hedef: %85+
