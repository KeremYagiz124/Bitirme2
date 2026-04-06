# 3. Sistem Mimarisi

Bu bölümde geliştirilen sistemin nasıl çalıştığı, hangi parçalardan oluştuğu ve bu parçaların birbiriyle nasıl etkileşime girdiği anlatılmaktadır. Teknik detaylar olabildiğince sade bir dille aktarılmıştır.

---

## 3.1 Genel Bakış

Geliştirilen sistem temelde şu soruyu yanıtlamaktadır: "Bu kamera görüntüsündeki araç, orada park edebilir mi?"

Bu soruyu yanıtlamak için sırayla dört işlem gerçekleştirilmektedir:

1. Görüntüdeki araçlar bulunuyor
2. Park alanları ve yasak bölgeler bilgisi yükleniyor
3. Araçların bu bölgelerle örtüşüp örtüşmediği hesaplanıyor
4. Karar veriliyor ve ekranda gösteriliyor

Bu dört adım birbirini takip eden bir zincir oluşturmaktadır. Bu yapıya **pipeline** adı verilmektedir.

```
Kamera / Video / Resim
          ↓
   [YOLOv8 ile Araç Tespiti]
   → Bounding box koordinatları
   → Güven skoru (confidence)
   → Araç tipi (araba/otobüs/kamyon/motosiklet)
          ↓
   [Park Alanı Poligonları Yükleniyor]
   → Park edilebilir bölgeler
   → Yasak bölgeler
          ↓
   [IoU Hesabı]
   → Her araç için hangi bölgeyle ne kadar örtüşüyor?
          ↓
   [Karar Motoru]
   → Park Edilebilir / Park Alanı Dolu / Park Edilemez
          ↓
   [PyQt5 Arayüzü]
   → Gerçek zamanlı görselleştirme
```

---

## 3.2 Klasör Yapısı

Proje birden fazla kişi tarafından geliştirildiği için dosyalar mantıklı klasörlere ayrılmıştır. Her klasörün tek bir sorumluluğu bulunmaktadır:

```
Bitirme2/
│
├── src/                        ← Projenin kalbi, asıl kod burada
│   ├── detection/
│   │   └── vehicle_detector.py ← YOLOv8 ile araç tespiti
│   ├── parking/
│   │   ├── zone_annotator.py   ← Park alanı çizme aracı
│   │   └── zone_loader.py      ← Çizilen alanları yükle + IoU hesapla
│   ├── ui/
│   │   └── main_window.py      ← PyQt5 görsel arayüz
│   └── main.py                 ← Uygulamayı başlatan giriş noktası
│
├── scripts/                    ← Tek seferlik çalıştırılan araçlar
│   ├── annotate_zones.py       ← Park alanı etiketleme aracını başlat
│   ├── baseline_test.py        ← Komut satırından hızlı test
│   ├── synthetic_data_generator.py ← Sentetik eğitim verisi üret
│   └── fine_tune_yolo.py       ← YOLOv8 fine-tuning
│
├── data/
│   ├── raw/                    ← Test için gerçek görüntüler
│   └── synthetic/              ← Üretilen sentetik eğitim verisi
│       ├── images/train/       ← %80 eğitim
│       ├── images/val/         ← %20 doğrulama
│       └── data.yaml           ← YOLO eğitim konfigürasyonu
│
├── configs/
│   └── config.yaml             ← Tüm ayarlar tek yerde
│
└── outputs/
    └── visualizations/         ← Tespit sonucu görseller
```

---

## 3.3 Bileşen 1: Araç Tespiti (VehicleDetector)

**Dosya:** [src/detection/vehicle_detector.py](../src/detection/vehicle_detector.py)

Bu bileşen sistemin gözüdür. Bir görüntü (ya da video karesi) alıyor ve içindeki araçları buluyor.

### Ne Yapıyor?

- Görüntü YOLOv8 modeline verilmektedir
- Model her araç için bir bounding box (çerçeve) ve güven skoru döndürmektedir
- Yalnızca araç sınıfları filtrelenmektedir (COCO ID: araba=2, motosiklet=3, otobüs=5, kamyon=7)
- Sonuçlar liste halinde döndürülmektedir

### Ne Döndürüyor?

Her tespit için şu bilgiler geliyor:
```
{
  "bbox": [x1, y1, x2, y2],   ← sol-üst ve sağ-alt köşe koordinatları
  "confidence": 0.93,          ← modelin emin olma oranı (0-1)
  "class_id": 2,               ← araç tipi numarası
  "class_name": "car"          ← araç tipi adı
}
```

### COCO mu, Fine-Tuned Model mi?

Sistem iki farklı modelle çalışabilmektedir:

- **COCO önceden eğitilmiş model (yolov8n.pt):** 80 farklı nesneyi tanımaktadır. Yalnızca araç sınıfları filtrelenmektedir. Kurulumdan sonra doğrudan çalışmaktadır.
- **Fine-tuned model:** Sentetik veriyle eğitilmiş özel model. 4 sınıf (araba, motosiklet, otobüs, kamyon), 0'dan başlayan ID'ler. Sistem model adına bakarak hangisinin kullanılacağına otomatik karar vermektedir.

```
Model adında "fine_tuned" geçiyor mu?
    Evet → class_map = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
    Hayır → class_map = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
```

Bu ayrımı yapmak önemliydi çünkü iki model farklı ID sistemleri kullanıyor. Bunu atlamak, fine-tuning sonrasında hiçbir şey tespit edilememesine yol açardı.

---

## 3.4 Bileşen 2: Park Alanı Etiketleme (ZoneAnnotator)

**Dosya:** [src/parking/zone_annotator.py](../src/parking/zone_annotator.py)

Bu bileşen, sisteme "nerede park edilebilir, nerede edilemez?" bilgisini tanımlamak için kullanılmaktadır.

### Nasıl Çalışıyor?

Bir fotoğraf açılmakta, fareyle tıklanarak park alanının veya yasak bölgenin köşe noktaları işaretlenmektedir. Program bu noktalardan bir poligon oluşturmaktadır. Bu işlem her park alanı için tekrarlanmakta, ardından `Q` tuşuyla kaydedilmektedir.

Kaydedilen bilgi JSON formatında bir dosyaya yazılmaktadır:

```json
{
  "image": "data/raw/otopark.jpg",
  "zones": [
    {
      "id": 1,
      "type": "parking",
      "points": [[120, 200], [250, 200], [250, 320], [120, 320]]
    },
    {
      "id": 2,
      "type": "forbidden",
      "points": [[300, 150], [420, 150], [420, 280], [300, 280]]
    }
  ]
}
```

### Klavye Kontrolleri

| Tuş | İşlev |
|-----|-------|
| Sol tık | Noktayı ekle |
| ENTER veya S | Park alanı olarak kaydet |
| F | Yasak bölge olarak kaydet |
| Z | Son noktayı geri al |
| C | Mevcut çizimi temizle |
| D | Son kaydedilen bölgeyi sil |
| Q veya ESC | Kaydet ve çık |

### Neden Manuel Çizim?

Literatürde (APSD-OC [9], PakLoc gibi) otomatik park yeri tespiti çalışmaları mevcuttur; ancak bunlar sabit ve standart park alanları için iyi çalışmaktadır. Her park alanının şekli farklı olup çizgiler farklı açılarda bulunmakta, bazı yerlerde çizgi bulunmamaktadır. Manuel çizim zahmetli görünse de bir kez yapılmakta ve sonrasında sistem otomatik olarak çalışmaktadır. Parquery gibi ticari sistemler de benzer yaklaşımı benimsemektedir.

---

## 3.5 Bileşen 3: Park Alanı Yükleme ve IoU Hesabı (ZoneLoader)

**Dosya:** [src/parking/zone_loader.py](../src/parking/zone_loader.py)

Bu bileşen iki iş yapıyor: kayıtlı park alanlarını yüklüyor ve bir araçla örtüşüp örtüşmediğini hesaplıyor.

### IoU Nedir?

IoU (Intersection over Union), iki şeklin ne kadar örtüştüğünü 0-1 arası bir sayıyla ifade ediyor.

```
        [Kesişim Alanı]
IoU = ───────────────────────
       [Birleşim Alanı]
```

Değer 1'e ne kadar yakınsa iki şekil o kadar örtüşüyor. 0 ise hiç örtüşme yok.

Pratikte şöyle çalışıyor:

- Araç bounding box'u bir dikdörtgen
- Park alanı poligonu herhangi bir çokgen
- İkisinin örtüşme oranı hesaplanıyor
- Bu oran 0.3'ün üzerindeyse araç bu park alanıyla ilişkili kabul edilmektedir

```
IoU > 0.3 → Araç bu park alanında
IoU < 0.3 → Araç bu park alanıyla ilgili değil
```

Neden 0.5 değil de 0.3? Çünkü araç poligonu tam olarak park alanının içinde olmayabilir; kapı açılmış, biraz taşmış ya da kamera açısı sebebiyle görsel olarak kaymış olabilir. 0.3 daha toleranslı bir eşik değeri.

### Nasıl Hesaplıyor?

Piksel tabanlı hesaplama yapılıyor:
1. Araç bounding box'u büyüklüğünde boş bir maske oluşturuluyor, dikdörtgen çiziliyor
2. Park poligonu aynı büyüklükte boş bir maskede çiziliyor
3. İki maskenin mantıksal AND'i → kesişim alanı
4. İki maskenin mantıksal OR'u → birleşim alanı
5. Kesişim / Birleşim = IoU

---

## 3.6 Bileşen 4: Karar Motoru

Araç tespiti ve IoU hesabından gelen bilgiler bir araya getirilerek nihai karar üretiliyor. Mantık şu şekilde:

```
Bir araç için:

1. Yasak bölgelerle IoU kontrol et
   → IoU > 0.3 → "PARK EDİLEMEZ" (ihlal)

2. Park alanlarıyla IoU kontrol et
   → IoU > 0.3 → "PARK ALANI DOLU"

3. Hiçbir bölgeyle eşleşme yok
   → "Araç serbest alanda" (bilgi amaçlı)

Park alanı kontrolü:
   → Hiçbir araç yoksa → "PARK EDİLEBİLİR"
```

Bu karar yapısı literatürdeki çalışmaların büyük çoğunluğundan farklıdır. Mevcut çalışmalar yalnızca "dolu / boş" ikili sınıflandırması yapmaktadır. Geliştirilen altyapıda "dolu / boş / yasak bölge" ayrımı yapılabilmektedir; bu özellik ilerleyen aşamada arayüze entegre edilmesi planlanmaktadır.

---

## 3.7 Bileşen 5: Sentetik Veri Üretici

**Dosya:** [scripts/synthetic_data_generator.py](../scripts/synthetic_data_generator.py)

Modeli park senaryosuna özel eğitmek için etiketli veriye ihtiyacımız var. Gerçek park alanı görüntüsünü etiketlemek çok zaman alıyor. Bunun yerine bilgisayarda otomatik görüntüler üretiyoruz.

### Ne Üretiyor?

Her görüntüde:
- Gri tonlu bir yol zemini oluşturuluyor
- Rastgele 1-5 araç yerleştiriliyor (araba, otobüs, kamyon, motosiklet)
- Her araç farklı boyut ve konumda, belirli bir ölçek değişkeniyle (%80-%150 arası)
- Her araç tipi için farklı renk ve basit detay çiziliyor
- Otomatik olarak YOLO formatında etiket dosyası oluşturuluyor

```
Araba: kırmızı dikdörtgen + cam detayı
Motosiklet: mavi dikdörtgen + gidon detayı
Otobüs: yeşil dikdörtgen + yan pencereler
Kamyon: sarı dikdörtgen + kargo bölmesi
```

### YOLO Format Etiketi

YOLO, her nesne için bu formatta bir satır bekliyor:
```
<class_id> <x_merkez> <y_merkez> <genişlik> <yükseklik>
```
Tüm değerler görüntü boyutuna göre normalize edilmiş (0-1 arası). Bu etiketi de otomatik üretiyoruz.

### Train / Val Ayrımı

Üretilen verinin %80'i eğitim (%80 train), %20'si doğrulama (val) için ayrılıyor. Bu ayrım önemli; eğer aynı veriyi hem eğitim hem test için kullansaydık, model "ezber" yapıyor gibi görünürdü ama gerçekte bir şey öğrenmemiş olurdu.

```
1000 görüntü üretildi
   → 800 görüntü: data/synthetic/images/train/
   → 200 görüntü: data/synthetic/images/val/
```

---

## 3.8 Bileşen 6: YOLOv8 Fine-Tuning

**Dosya:** [scripts/fine_tune_yolo.py](../scripts/fine_tune_yolo.py)

Transfer learning, önceden büyük bir veriyle eğitilmiş bir modelin alınarak farklı bir probleme uyarlanması işlemidir. Sıfırdan eğitim yapılmak yerine mevcut ağırlıklardan başlanmaktadır.

### Neden Fine-Tuning?

COCO ile eğitilmiş YOLOv8, 80 farklı nesneyi tanımaktadır. Projede yalnızca 4 araç tipi (araba, otobüs, kamyon, motosiklet) park senaryosunda tespit edilmek istenmektedir. Fine-tuning ile:
- Modelin öğrendiği genel özellikler korunmaktadır (kenarlar, şekiller, dokular)
- Yalnızca park senaryosuna özgü ayrıntılar yeniden öğretilmektedir
- Bunun için çok daha az veri ve süre yeterli olmaktadır

### Eğitim Parametreleri

| Parametre | Değer | Neden? |
|-----------|-------|--------|
| Epoch | 30 | Küçük veri seti için yeterli |
| Batch size | 8 | 4GB VRAM'e uygun |
| Görüntü boyutu | 640×640 | YOLOv8 standardı |
| Optimizer | AdamW (otomatik) | YOLOv8 en iyi seçiyor |
| Early stopping | 10 epoch | İyileşme durduğunda otomatik bitirir |
| Device | GPU varsa CUDA, yoksa CPU | Otomatik seçiyor |

### Eğitim Süreci

```
Epoch 1 / 30
   → Her batch için:
       - Görüntüler modele veriliyor
       - Model tahmin yapıyor
       - Hata hesaplanıyor (loss)
       - Model ağırlıkları güncelleniyor
   → Epoch sonunda validasyon yapılıyor
   → En iyi model "best.pt" olarak kaydediliyor
```

---

## 3.9 Bileşen 7: Arayüz (MainWindow)

**Dosya:** [src/ui/main_window.py](../src/ui/main_window.py)

Tüm bu bileşenleri bir araya getiren görsel arayüz PyQt5 ile yapıldı. Kullanıcı teknik detaylarla uğraşmadan sistemi çalıştırabiliyor.

### Arayüzde Ne Var?

**Sol taraf — Video alanı:**
Kamera görüntüsü veya yüklenen video/resim burada gösteriliyor. Tespit edilen araçların etrafına renkli çerçeveler çiziliyor, yanında araç tipi ve güven skoru yazıyor.

| Araç Tipi | Çerçeve Rengi |
|-----------|--------------|
| Araba | Yeşil |
| Motosiklet | Turuncu |
| Otobüs | Kırmızı |
| Kamyon | Mor |

**Sağ taraf — Kontrol paneli:**
- ▶ Kamera Başlat / ■ Durdur / 📂 Video Yükle / 🖼 Resim Yükle butonları
- Her araç tipi için ayrı sayaç kartları (kaç araba, kaç otobüs vb.)
- FPS göstergesi (saniyede kaç kare işleniyor)
- Durum mesajı (model yüklendi, video oynatılıyor vb.)

### Timer Mimarisi

Arayüz "freeze" olmaması için şöyle çalışıyor:

- QTimer her 30 milisaniyede bir tetikleniyor (~33 FPS hedef)
- Her tetiklemede bir kare okunuyor
- Kare YOLOv8'e gönderiliyor, sonuçlar işleniyor
- Arayüz güncelleniyor

Bu döngü boyunca butonlar çalışmaya devam ediyor çünkü QTimer arayüzü bloklamıyor.

---

## 3.10 Veri Akışı: Uçtan Uca Örnek

Bir kullanıcı video yüklediğinde arka planda neler oluyor, adım adım:

```
1. Kullanıcı "📂 Video Yükle" butonuna tıklıyor
       ↓
2. OpenCV video dosyasını açıyor (cv2.VideoCapture)
       ↓
3. QTimer her 30ms'de bir kare okuyor
       ↓
4. VehicleDetector.detect(frame) çağrılıyor
       → YOLOv8 modeli kareyi işliyor
       → Araç tespitleri liste olarak dönüyor
       ↓
5. (Eğer zone dosyası yüklüyse) ZoneLoader.find_zone(bbox)
       → Her araç için IoU hesaplanıyor
       → En yüksek IoU'ya sahip park alanı eşleştiriliyor
       ↓
6. Karar motoru kararı üretiyor
       → Park Edilebilir / Park Alanı Dolu / Park Edilemez
       ↓
7. OpenCV ile araçların etrafına renkli çerçeve çiziliyor
       ↓
8. PyQt5 görüntüyü ekrana yansıtıyor
       ↓
9. Sayaç kartları ve FPS değeri güncelleniyor
```

Bu döngü video bitene ya da kullanıcı "Durdur" diyene kadar devam ediyor.

---

## 3.11 Tasarım Kararları

Projeyi geliştirirken aldığımız bazı kararlar ve gerekçeleri:

**Neden YOLOv8n, daha büyük model değil?**
GTX 1050 Ti Mobile 4GB VRAM'e sahip. YOLOv8m veya YOLOv8l çok daha yavaş çalışırdı ya da bellek yetersizliği hatası verirdi. n (nano) versiyonu gerçek zamanlı çalışma için en uygun denge noktası.

**Neden IoU için 0.3 eşiği?**
Kamera açısı, araç boyutu ve poligon çizim hassasiyeti göz önüne alındığında 0.5 çok katı bir eşik. Gerçek testlerde 0.3'ün daha güvenilir sonuç verdiği gözlemlenmiştir. da Luz vd. [7] de benzer şekilde esnek IoU eşikleri kullanmaktadır.

**Neden piksel maskesi ile IoU, formül ile değil?**
Park alanları dikdörtgen değil, poligon şeklinde. Standart IoU formülü sadece dikdörtgenler için işliyor. Piksel maskesi yaklaşımı herhangi bir poligon şekliyle çalışıyor; hesaplama biraz daha ağır ama doğru sonuç veriyor.

**Neden timer tek bir yerde bağlandı?**
Başlangıçta `timer.timeout.connect(update_frame)` her buton tıklamasında çağrılıyordu. Bu, fonksiyonun birden fazla kez bağlanmasına ve her karede birden fazla çalışmasına yol açıyordu. Timer artık başlangıçta bir kez bağlanıyor, sadece `start()` ve `stop()` ile kontrol ediliyor.
