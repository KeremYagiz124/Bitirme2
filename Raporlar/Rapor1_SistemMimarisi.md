# 3. Sistem Mimarisi

Bu bölümde geliştirilen sistemin nasıl çalıştığı, hangi parçalardan oluştuğu ve bu parçaların birbiriyle nasıl etkileşime girdiği anlatılmaktadır.

---

## 3.1 Genel Bakış

Geliştirilen sistem temelde şu soruyu yanıtlamaktadır: "Bu kamera görüntüsündeki araç, orada park edebilir mi?"

Bu soruyu yanıtlamak için sırayla dört işlem gerçekleştirilmektedir: görüntüdeki araçlar bulunmakta, park alanı bilgisi yüklenmekte, araçların bu bölgelerle örtüşüp örtüşmediği hesaplanmakta ve karar ekranda gösterilmektedir. Bu yapıya **pipeline** adı verilmektedir.

```
Kamera / Video / Resim
          ↓
   [YOLOv8 ile Araç Tespiti]
   → Bounding box, güven skoru, araç tipi
          ↓
   [Park Alanı Poligonları Yükleniyor]
   → Park edilebilir ve yasak bölgeler
          ↓
   [IoU Hesabı]
   → Her araç için bölge örtüşme oranı
          ↓
   [Karar Motoru]
   → Park Edilebilir / Park Alanı Dolu / Park Edilemez
          ↓
   [PyQt5 Arayüzü]
   → Gerçek zamanlı görselleştirme
```

---

## 3.2 Klasör Yapısı

```
Bitirme2/
├── src/
│   ├── detection/
│   │   └── vehicle_detector.py   ← YOLOv8 ile araç tespiti
│   ├── parking/
│   │   ├── zone_annotator.py     ← Park alanı çizme aracı
│   │   └── zone_loader.py        ← Alan yükleme + IoU hesabı
│   ├── ui/
│   │   └── main_window.py        ← PyQt5 arayüz
│   └── main.py
├── scripts/
│   ├── annotate_zones.py
│   ├── baseline_test.py
│   ├── synthetic_data_generator.py
│   └── fine_tune_yolo.py
├── data/
│   ├── raw/
│   └── synthetic/
│       ├── images/train/         ← %80 eğitim
│       ├── images/val/           ← %20 doğrulama
│       └── data.yaml
├── configs/
│   └── config.yaml
└── outputs/
    └── visualizations/
```

---

## 3.3 Bileşen 1: Araç Tespiti (VehicleDetector)

**Dosya:** [src/detection/vehicle_detector.py](../src/detection/vehicle_detector.py)

Görüntü YOLOv8 modeline verilmekte, model her araç için bounding box ve güven skoru döndürmekte, yalnızca araç sınıfları filtrelenmektedir.

### COCO mu, Fine-Tuned Model mi?

Sistem iki farklı modelle çalışabilmektedir. COCO önceden eğitilmiş model 80 nesneyi tanımakta, araç sınıfları sonradan filtrelenmektedir (ID: 2, 3, 5, 7). Fine-tuned model ise yalnızca 4 araç sınıfını 0'dan başlayan ID'lerle tanımaktadır. Sistem model adına bakarak doğru ID haritasını otomatik seçmektedir:

```
Model adında "fine_tuned" geçiyor mu?
    Evet → class_map = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
    Hayır → class_map = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
```

---

## 3.4 Bileşen 2: Park Alanı Etiketleme (ZoneAnnotator)

**Dosya:** [src/parking/zone_annotator.py](../src/parking/zone_annotator.py)

Bir fotoğraf açılmakta, fareyle köşe noktaları işaretlenmekte ve program bu noktalardan poligon oluşturmaktadır. Kaydedilen bilgi JSON formatında dosyaya yazılmaktadır.

| Tuş | İşlev |
|-----|-------|
| Sol tık | Nokta ekle |
| ENTER / S | Park alanı kaydet |
| F | Yasak bölge kaydet |
| Z | Son noktayı geri al |
| C | Çizimi temizle |
| D | Son bölgeyi sil |
| Q / ESC | Kaydet ve çık |

Literatürde otomatik park yeri tespiti çalışmaları mevcuttur (APSD-OC [9]); ancak her park alanının şekli ve çizgi düzeni farklıdır. Manuel çizim bir kez yapılmakta, sonrasında sistem otomatik çalışmaktadır. Parquery [11] gibi ticari sistemler de benzer yaklaşımı benimsemektedir.

---

## 3.5 Bileşen 3: IoU Hesabı (ZoneLoader)

**Dosya:** [src/parking/zone_loader.py](../src/parking/zone_loader.py)

IoU (Intersection over Union), iki şeklin örtüşme oranını 0-1 arası bir değerle ifade etmektedir:

```
        [Kesişim Alanı]
IoU = ───────────────────────
       [Birleşim Alanı]
```

Park alanları poligon şeklinde olduğundan piksel maskesi yaklaşımı kullanılmaktadır: araç bounding box ve park poligonu ayrı piksel maskelerine çizilmekte, mantıksal AND ile kesişim, OR ile birleşim alanı hesaplanmaktadır. IoU 0.3 eşiğini aşarsa araç o bölgeyle ilişkilendirilmektedir. Eşiğin 0.5 yerine 0.3 seçilmesinin nedeni kamera açısı ve poligon çizim toleransıdır.

---

## 3.6 Bileşen 4: Karar Motoru

```
Bir araç için:
  1. Yasak bölgelerle IoU > 0.3 → "PARK EDİLEMEZ"
  2. Park alanlarıyla IoU > 0.3 → "PARK ALANI DOLU"
  3. Eşleşme yok → "Serbest alan"

Park alanı için araç yoksa → "PARK EDİLEBİLİR"
```

Mevcut çalışmaların büyük çoğunluğu yalnızca "dolu/boş" ikili sınıflandırması yapmaktadır. Geliştirilen altyapıda üçüncü sınıf (yasak bölge) da desteklenmektedir; bu özellik ilerleyen aşamada arayüze entegre edilmesi planlanmaktadır.

---

## 3.7 Bileşen 5: Sentetik Veri Üretici

**Dosya:** [scripts/synthetic_data_generator.py](../scripts/synthetic_data_generator.py)

Her görüntüde gri bir zemin üzerine 1-5 arası rastgele araç yerleştirilmektedir. Her araç tipi farklı renk ve basit detaylarla temsil edilmektedir. YOLO formatında etiket dosyaları otomatik oluşturulmaktadır. 1000 görüntünün 800'ü eğitime, 200'ü doğrulamaya ayrılmıştır.

---

## 3.8 Bileşen 6: YOLOv8 Fine-Tuning

**Dosya:** [scripts/fine_tune_yolo.py](../scripts/fine_tune_yolo.py)

COCO ile eğitilmiş YOLOv8'in öğrendiği genel özellikler korunarak yalnızca park senaryosuna özgü ayrıntılar yeniden öğretilmektedir. Eğitim parametreleri:

| Parametre | Değer |
|-----------|-------|
| Epoch | 30 |
| Batch size | 8 |
| Görüntü boyutu | 640×640 |
| Early stopping | 10 epoch |
| Device | GPU (CUDA) / CPU otomatik |

---

## 3.9 Bileşen 7: Arayüz (MainWindow)

**Dosya:** [src/ui/main_window.py](../src/ui/main_window.py)

PyQt5 ile geliştirilen arayüz kamera, video ve resim kaynaklarını desteklemektedir. Sol tarafta gerçek zamanlı video alanı, sağ tarafta araç tipi sayaçları, FPS göstergesi ve kontrol butonları yer almaktadır. QTimer her 30 ms'de bir kare okuyarak arayüzü bloklamadan güncellemektedir.

| Araç Tipi | Çerçeve Rengi |
|-----------|--------------|
| Araba | Yeşil |
| Motosiklet | Turuncu |
| Otobüs | Kırmızı |
| Kamyon | Mor |

---

## 3.10 Tasarım Kararları

**Neden YOLOv8n?** GTX 1050 Ti Mobile 4GB VRAM'de büyük modeller gerçek zamanlı çalışamamaktadır. YOLOv8n 30+ FPS ile en uygun denge noktasıdır.

**Neden IoU eşiği 0.3?** Kamera açısı ve poligon çizim hassasiyeti nedeniyle 0.5 çok katı kalmaktadır. Gerçek testlerde 0.3 daha güvenilir sonuç vermektedir [7].

**Neden piksel maskesi?** Park alanları poligon şeklindedir; standart IoU formülü yalnızca dikdörtgenler için çalışmaktadır.

**Timer mimarisi:** Timer başlangıçta bir kez bağlanmakta, yalnızca `start()` ve `stop()` ile kontrol edilmektedir. Çoklu bağlantı her karede katlanarak çalışmaya yol açmaktaydı.
