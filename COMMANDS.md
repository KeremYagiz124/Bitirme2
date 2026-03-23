# Çalıştırma Komutları

## Araç Tespiti (YOLOv8 Baseline)

```bash
# Tek resim üzerinde araç tespiti
python scripts/baseline_test.py --source data/raw/resim.jpg

# Video üzerinde araç tespiti
python scripts/baseline_test.py --source data/raw/video.mp4

# Webcam ile canlı araç tespiti
python scripts/baseline_test.py --source 0

# Farklı model boyutu (n/s/m/l/x — büyüdükçe daha doğru ama yavaş)
python scripts/baseline_test.py --source data/raw/resim.jpg --model yolov8s.pt

# Confidence threshold değiştir (0.0-1.0, düşüklerse daha fazla tespit)
python scripts/baseline_test.py --source data/raw/resim.jpg --conf 0.3
```

Çıktı: `outputs/visualizations/baseline_<dosyaadi>.jpg`

---

## Park Alanı Etiketleme (Zone Annotator)

```bash
# Yeni etiketleme başlat
python scripts/annotate_zones.py --image data/raw/otopark.jpg

# Kaydedilen dosyayı düzenlemeye devam et
python scripts/annotate_zones.py --image data/raw/otopark.jpg --load data/raw/otopark.json

# Çıktıyı farklı konuma kaydet
python scripts/annotate_zones.py --image data/raw/otopark.jpg --output data/annotations/lot1.json
```

### Kontroller (pencere açıkken)
| Tuş | İşlev |
|-----|-------|
| Sol tık | Nokta ekle |
| ENTER veya S | Mevcut poligonu PARK ALANI olarak kaydet |
| F | Mevcut poligonu YASAK BÖLGE olarak kaydet |
| Z | Son noktayı geri al |
| C | Mevcut poligonu temizle |
| D | Son kaydedilen zonu sil |
| Q veya ESC | Kaydet ve çık |

Çıktı: Resimle aynı dizinde `.json` (ya da `--output` ile belirtilen yol)

---

## Kurulum

```bash
pip install -r requirements.txt
```
