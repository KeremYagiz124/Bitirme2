# 5. Uygulama ve Bulgular

Bu bölümde projenin şu ana kadar hangi aşamalardan geçtiği ve elde edilen sonuçlar aktarılmaktadır. Proje hâlâ devam ettiği için bazı metrik sonuçları ilerleyen haftalarda güncellenecektir.

---

## 5.1 Geliştirme Süreci

Proje haftalar halinde planlanarak ilerlemiştir. Her hafta belirli bir hedefe odaklanılmıştır.

### Hafta 1 — Temel Kurulum ve Baseline

İlk haftada projenin iskeletini kurulmuştur:

- Klasör yapısı ve konfigürasyon dosyaları oluşturuldu
- Gerekli kütüphaneler kuruldu (YOLOv8, OpenCV, PyTorch, PyQt5 vb.)
- `VehicleDetector` sınıfı yazıldı
- YOLOv8n modeli COCO ağırlıklarıyla ilk kez çalıştırıldı
- Park alanı etiketleme aracı (`ZoneAnnotator`) geliştirildi
- Komut satırından çalışan test betiği yazıldı (`baseline_test.py`)

Bu haftanın en önemli çıktısı: sistem gerçekten çalışıyor mu? Cevap olumlu. İlk testte model bir otobüs görüntüsünde **%93 güven skoru** ile başarılı tespit yaptı.

### Hafta 2 — Sentetik Veri ve Arayüz

İkinci haftada iki büyük geliştirme yapıldı:

- Sentetik veri üreteci yazıldı ve 1000 görüntü üretildi
- YOLOv8 fine-tuning betiği hazırlandı
- PyQt5 tabanlı görsel arayüz geliştirildi
- Araç tespiti sonuçları arayüze entegre edildi
- Her araç tipi için ayrı renk ve sayaç eklendi
- FPS göstergesi eklendi
- Resim yükleme özelliği eklendi

---

## 5.2 Baseline Sistem Sonuçları

COCO ağırlıklarıyla çalışan YOLOv8n modeli [2], park senaryosuna özel herhangi bir eğitim yapılmadan test edilmiştir.

### Test Koşulları

- Model: YOLOv8n (COCO önceden eğitilmiş)
- Confidence eşiği: 0.5
- IoU eşiği: 0.45
- Test görüntüsü: Farklı açı ve koşullardaki araç görüntüleri

### Gözlemlenen Sonuçlar

Model araçları genel olarak başarıyla tespit etti. Güven skorları genellikle %80 ile %95 arasında ölçüldü. Özellikle net görünürlükte ve iyi ışıkta çekilen görüntülerde tespit kalitesi yüksekti.

Bazı gözlemler:
- Büyük araçlar (otobüs, kamyon) daha güvenilir tespit edildi
- Kısmen kapalı (örtülmüş) araçlarda güven skoru düştü
- Kamera açısına göre tespit kalitesi değişti
- Motosikletler küçük olduğu için zaman zaman kaçırıldı

Bu sonuçlar beklentilerle örtüşüyor. COCO ile eğitilmiş model park senaryosuna özel bir beklenti taşımıyor; fine-tuning sonrasında bu sonuçların iyileşmesi bekleniyor.

---

## 5.3 Sentetik Veri Üretimi

Modeli park senaryosuna uyarlamak için sentetik eğitim verisi üretilmiştir.

### Üretim Parametreleri

- Toplam görüntü: 1000
- Eğitim seti: 800 görüntü (%80)
- Doğrulama seti: 200 görüntü (%20)
- Görüntü boyutu: 640×480 piksel
- Görüntü başına araç sayısı: 1-5 (rastgele)
- Boyut değişkeni: ±%50 (rastgele ölçek)

### Araç Temsilleri

Her araç tipi basit renkli dikdörtgen ve detaylarla temsil edilmektedir:

| Araç Tipi | Temel Boyut | Renk | Detay |
|-----------|-------------|------|-------|
| Araba | 80×40 px | Kırmızı | Cam alanı |
| Motosiklet | 30×60 px | Mavi | Gidon çizgisi |
| Otobüs | 120×50 px | Yeşil | Yan pencereler |
| Kamyon | 100×45 px | Sarı | Kargo bölmesi |

### Etiket Formatı

Her görüntü için bir `.txt` etiketi otomatik olarak oluşturulmuştur. YOLO formatı kullanılmıştır:

```
<class_id> <x_merkez> <y_merkez> <genişlik> <yükseklik>
```

Tüm değerler normalize edilmiş (0-1 arası). Örnek bir satır:

```
0 0.421875 0.354167 0.125000 0.083333
```

Bu satır: araba sınıfı (0), görüntünün ortasına yakın konumda.

---

## 5.4 Görsel Arayüz

Geliştirilen PyQt5 arayüzü üç ana işlevi desteklemektedir: kamera akışı, video oynatma ve tek resim analizi.

### Arayüz Özellikleri

- Karanlık tema (Fusion stili)
- Gerçek zamanlı araç sayaçları (araba, motosiklet, otobüs, kamyon için ayrı)
- Her araç tipi farklı renk ile işaretlenmektedir
- FPS göstergesi
- Durum çubuğu (model yüklendi, video oynatılıyor vb.)
- Video bittiğinde otomatik durdurma
- Butonlar aktif/pasif durumuna göre görsel olarak değişmektedir

### Performans

GTX 1050 Ti Mobile donanımında YOLOv8n ile ölçülen yaklaşık FPS değerleri:

| Kaynak | Çözünürlük | Yaklaşık FPS |
|--------|-----------|--------------|
| Video (720p) | 1280×720 | ~18-22 |
| Video (480p) | 854×480 | ~25-30 |
| Resim | Değişken | Anlık |

Not: PyTorch GPU sürücü uyumsuzluğu nedeniyle CPU modunda çalışıyor. GPU aktif olsaydı bu değerlerin 2-3 kat artması beklenirdi.

---

## 5.5 Park Alanı Etiketleme Aracı

Geliştirilen `ZoneAnnotator` aracı, herhangi bir park alanı fotoğrafı üzerinde poligon çizmeyi sağlıyor.

### Çalışma Prensibi

1. Fotoğraf açılıyor
2. Mouse ile köşe noktaları tıklanıyor
3. En az 3 nokta sonrası ENTER ile park alanı, F ile yasak bölge olarak kaydediliyor
4. Tüm bölgeler JSON dosyasına yazılıyor

### JSON Çıktı Yapısı

```json
{
  "image": "data/raw/otopark.jpg",
  "zones": [
    {
      "id": 1,
      "type": "parking",
      "points": [[145, 210], [278, 208], [281, 335], [142, 338]]
    },
    {
      "id": 2,
      "type": "forbidden",
      "points": [[310, 155], [425, 153], [428, 195], [308, 197]]
    }
  ]
}
```

Bu dosya daha sonra `ZoneLoader` tarafından okunarak IoU hesabında kullanılıyor.

---

## 5.6 IoU Tabanlı Doluluk Analizi

Park alanı JSON dosyası yüklendiğinde sistem araç tespiti sonuçlarını bu alanlarla karşılaştırıyor.

### Hesaplama Örneği

Bir araç için bounding box koordinatları: `[120, 200, 280, 360]`
Park alanı poligonu: `[[130, 195], [275, 195], [275, 365], [130, 365]]`

Piksel maske yöntemiyle hesaplanan IoU: **0.87**

Bu değer 0.3 eşiğini geçtiği için sistem bu park alanını **"Dolu"** olarak işaretliyor.

Başka bir araç için: yasak bölge poligonuyla IoU: **0.52** → **"Park Edilemez"** kararı.

### Karar Tablosu

| Durum | IoU (park alanı) | IoU (yasak bölge) | Karar |
|-------|-----------------|------------------|-------|
| Park alanı dolu | > 0.3 | < 0.3 | Park Alanı Dolu |
| Yasak ihlali | < 0.3 | > 0.3 | Park Edilemez |
| İki bölgeyle de yok | < 0.3 | < 0.3 | Serbest alan |
| Park alanı boş | Araç yok | — | Park Edilebilir |

---

## 5.7 Mevcut Sistem Sınırlamaları

Dürüst bir değerlendirme için sistemin şu anki eksikliklerini de belirtiyoruz:

**Araç takibi yok:** Geçen bir araç da tespit ediliyor ve geçici olarak "Dolu" kararı verilebiliyor. ByteTrack entegrasyonu ile bu çözülecek.

**Gece ve yağmur testi yapılmadı:** Sistem farklı hava koşullarında henüz test edilmedi. Veri artırma teknikleriyle (parlaklık değişimi, bulanıklaştırma) bu kısmen telafi edilecek.

**Sentetik veri gerçek değil:** Üretilen görüntüler oldukça basit. Gerçek kamera görüntüleriyle karşılaştırıldığında doku, ışık ve perspektif farklılıkları var. Bu farkı kapatmak için PKLot veri setiyle ek eğitim planlanıyor.

**GPU sorunu:** PyTorch sürücü uyumsuzluğu nedeniyle CPU modunda çalışılmaktadır. GPU aktif olsaydı hem eğitim hem de çıkarım (inference) çok daha hızlı olurdu.

---

## 5.8 Planlanan Ölçümler

İlerleyen haftalarda aşağıdaki metrikler hesaplanacak ve rapora eklenecek:

| Metrik | Açıklama | Hedef |
|--------|----------|-------|
| mAP@0.5 | IoU=0.5'te ortalama kesinlik | > 0.80 |
| Precision | Doğru pozitif / Tüm pozitif tahminler | > 0.85 |
| Recall | Doğru pozitif / Tüm gerçek pozitifler | > 0.80 |
| F1-Score | Precision ve Recall dengesi | > 0.82 |
| FPS (GPU) | Saniyede işlenen kare sayısı | > 25 |
| Confusion Matrix | 3 sınıf arası karışıklık oranları | — |

Ayrıca ablasyon çalışması yapılacak: "IoU tek başına" ile "IoU + MobileNetV2 doğrulama katmanı" karşılaştırılacak. Bu, ikinci doğrulama katmanının gerçekten katkı sağlayıp sağlamadığını gösterecek.
