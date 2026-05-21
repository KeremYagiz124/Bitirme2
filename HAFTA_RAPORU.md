# Haftalık Çalışma Raporu — Hafta 5 (19–22 Mayıs 2026)

## Proje
Kamera Görüntülerinden Akıllı Park Yeri Tespiti  
Stack: Python · YOLOv8 · YOLOPv2 · OpenCV · PyTorch · PyQt5

---

## Yapılan Çalışmalar

### 1. Lateral Row Split + Road Center Rejection

**Sorun:** Karşı şeritten gelen araçlar park şeridiyle aynı satıra atanıyor, yol ortasında sahte boş alan oluşuyordu.

**Yapılan:**
- Araç tespitlerini y-koordinatına göre kümeleme mantığı güçlendirildi; sol/sağ şerit birleşimi engellendi.
- `road_center_reject_ratio` parametresiyle yol merkezi slotları elendi.
- `_detect_in_row` içinde yatay merkez kontrolü eklendi.

---

### 2. Doğruluk Metrikleri Altyapısı

**Oluşturulan dosyalar:**
- `data/ground_truth/street_gt.json` — 3 test görüntüsü için beklenen boş/dolu alan sayıları
- `data/raw/araba2.json`, `data/raw/sample_bus.json` — bölge bazlı expected etiketleri eklendi
- `scripts/evaluate_street.py` — sokak modu için sayı tabanlı precision/recall/F1 hesabı
- `scripts/evaluate_all.py` — her iki modu birleştiren ana değerlendirme scripti

**Sonuçlar:**

| Metrik | Değer |
|--------|-------|
| Sabit Kamera Doğruluğu | %100.0 (7/7 bölge) |
| Sokak Modu Mikro-F1 | %66.7 |
| Genel Skor (ortalama) | %83.3 |

---

### 3. Bildirim / Uyarı Sistemi

**Oluşturulan:** `src/ui/alert_system.py`

- `AlertSystem` sınıfı: throttle (30 sn), maksimum geçmiş (50 kayıt), dinleyici zinciri
- Üç seviye: INFO (mavi) · WARNING (sarı) · CRITICAL (kırmızı)
- Hazır kontrol metodları: `check_occupancy`, `check_forbidden`, `check_no_fit`

**Tetikleme koşulları:**

| Kod | Seviye | Koşul |
|-----|--------|-------|
| `park_full` | CRITICAL | Hiç boş alan kalmadı |
| `park_high` | WARNING | Doluluk ≥ %80 |
| `forbidden_park` | WARNING | Yasak bölgede araç var |
| `no_fit` | INFO | Araç hiçbir boş alana sığmıyor |

**main_window.py entegrasyonu:**
- Renkli uyarı barı (alert_bar) UI'a eklendi
- CRITICAL uyarılar otomatik kapanmaz; diğerleri 10 sn sonra kapanır
- Sabit kamera ve sokak modu her ikisine de bağlandı

---

### 4. Stres Testleri

`tests/test_stress.py` oluşturuldu — 27 test, 27 geçti.

Kapsam:
- AlertSystem: throttle, geçmiş sınırı, dinleyici hata yutma, tüm check metodları
- ParkingAnalyzer: 0 tespit, tam dolu, 50 üst üste tespit, bölgesiz, tekrarlı çağrı
- StreetParkingDetector: boş frame, tek araç, 20 araç, 32×32 küçük frame, 4K büyük frame, kenar araçlar, reset_history tutarlılığı

---

### 5. README / Kullanıcı Kılavuzu

`README.md` komple yeniden yazıldı:
- Kurulum adımları (venv + GPU/CPU seçeneği)
- Sabit kamera ve sokak modu kullanım kılavuzu
- Bölge JSON formatı
- Araç sığma kontrolü kullanımı
- Bildirim sistemi açıklaması
- Değerlendirme scriptleri kullanımı
- Test çalıştırma
- Proje klasör yapısı
- Performans tablosu

---

### 6. Mimari Diyagram

README içine Mermaid akış şeması eklendi:

```
Görüntü/Video → VehicleDetector (YOLOv8)
                      ↓
              ┌───────┴───────┐
         Sabit Kamera    Sokak Modu
         ZoneLoader      StreetParkingDetector ← DrivableAreaDetector (YOLOPv2)
         ParkingAnalyzer
              └───────┬───────┘
                  AlertSystem
                  Araç Sığma Kontrolü
                      ↓
                  MainWindow (PyQt5)
```

---

### 7. Conf Değişince Boş Alan Güncelleme

**Sorun:** Conf slider'ı değiştirildiğinde yeni tespit edilen araçlar slot listesini güncellemiyordu.

**Düzeltme:**
- `_set_conf` içinde `street_detector.reset_history()` çağrısı eklendi
- Statik görüntüde conf değişince history temizlenerek yeniden analiz yapılıyor

---

### 8. Filtre Kaçıran Araç / Slot Çakışması Düzeltmesi

**Sorun:** `_filter_candidates` bazı araçları (çok küçük bbox, frame kenarında) eliyordu; bu araçların bulunduğu bölgede hâlâ boş slot gösteriliyordu.

**Düzeltme:**
- `analyze()` içinde raw slot listesi tüm ham tespitlerle çakışma kontrolünden geçiriliyor
- Slot genişliğinin %35'inden fazlası herhangi bir araç bbox'ıyla örtüşüyorsa slot eleniyor

---

## Değiştirilen / Oluşturulan Dosyalar

| Dosya | İşlem |
|-------|-------|
| `src/detection/street_parking_detector.py` | Lateral row split, road center rejection, slot-araç çakışma filtresi, conf reset |
| `src/ui/alert_system.py` | Yeni oluşturuldu |
| `src/ui/main_window.py` | Alert entegrasyonu, conf reset, slot çakışma düzeltmesi |
| `scripts/evaluate_all.py` | Yeni oluşturuldu |
| `scripts/evaluate_street.py` | Yeni oluşturuldu |
| `data/ground_truth/street_gt.json` | Yeni oluşturuldu |
| `data/raw/araba2.json` | Expected etiketleri eklendi |
| `data/raw/sample_bus.json` | Expected etiketi eklendi |
| `tests/test_stress.py` | Yeni oluşturuldu |
| `README.md` | Komple yeniden yazıldı |

---

## Sonuç

Bu hafta sistemin güvenilirlik ve olgunluk katmanları tamamlandı: doğruluk ölçümü, otomatik uyarı sistemi, stres testleri, eksiksiz dokümantasyon ve iki kritik tespit hatası giderildi. Sistem artık demo ve sunum için hazır durumda.
