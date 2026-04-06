# 4. Kullanılan Teknolojiler

Bu bölümde projede kullandığımız araçları, kütüphaneleri ve bunları neden seçtiğimizi anlatılmaktadır. Her teknoloji için alternatiflerle kıyaslama da yapılmıştır.

---

## 4.1 Programlama Dili: Python

Proje Python ile geliştirilmiştir. Bunun birkaç nedeni bulunmaktadır:

Yapay zeka ve görüntü işleme alanındaki neredeyse tüm popüler kütüphaneler Python için yazılmıştır. PyTorch, OpenCV, Ultralytics ve scikit-learn Python'da en iyi desteğe sahiptir. Farklı bir dil tercih edilseydi bu kütüphanelerin büyük çoğunluğundan yararlanmak mümkün olmayacaktı.

Bunun yanı sıra Python'un sözdizimi sade ve okunması kolaydır. Birden fazla kişinin üzerinde çalıştığı bir projede bu özellik önemli bir avantaj sağlamaktadır.

**Versiyon:** Python 3.12

---

## 4.2 Nesne Tespiti: YOLOv8 (Ultralytics)

Projenin en kritik bileşeni. Görüntüdeki araçları bulan model.

### YOLO Nedir?

YOLO "You Only Look Once" kelimelerinin baş harflerinden oluşuyor. Adı da mantığını anlatıyor: görüntüye bir kez bakıyor ve aynı anda hem "ne var?" hem de "nerede?" sorularını cevaplıyor. Eski yöntemler önce bölge önerisi yapıp sonra sınıflandırıyordu (iki aşama). YOLO bunu tek adımda yapıyor, bu yüzden çok daha hızlı.

### Neden YOLOv8?

| Özellik | YOLOv8 | Faster R-CNN | SSD |
|---------|--------|-------------|-----|
| Hız | Çok yüksek | Düşük | Yüksek |
| Doğruluk | Yüksek | Çok yüksek | Orta |
| Küçük nesne | Orta | İyi | Zayıf |
| Kullanım kolaylığı | Çok kolay | Zor | Orta |
| Gerçek zamanlı | Evet | Hayır | Evet |

Faster R-CNN doğrulukta YOLOv8'i geçebiliyor ama gerçek zamanlı çalışmıyor. Bizim için hız kritik olduğundan YOLOv8 doğal seçim oldu. Literatürdeki güncel çalışmalar da (da Luz vd., 2024) aynı modeli park tespitinde kullanıyor [7].

### Neden Nano (n) Versiyonu?

YOLOv8'in beş boyutu var: n (nano), s (small), m (medium), l (large), x (extra large). Büyüdükçe daha doğru ama daha yavaş ve daha fazla bellek istiyor.

| Model | mAP | Hız (ms) | Parametre |
|-------|-----|----------|-----------|
| YOLOv8n | 37.3 | 0.80 | 3.2M |
| YOLOv8s | 44.9 | 1.20 | 11.2M |
| YOLOv8m | 50.2 | 3.53 | 25.9M |
| YOLOv8l | 52.9 | 6.61 | 43.7M |

GTX 1050 Ti Mobile 4GB VRAM'e sahip. YOLOv8m ve üzeri bu donanımda gerçek zamanlı çalışmıyor. YOLOv8n ise 30+ FPS'i rahatlıkla veriyor.

### Transfer Learning

YOLOv8n modelini Microsoft COCO veri setiyle önceden eğitilmiş haliyle kullanılmıştır [2]. COCO, 330.000'den fazla görüntü ve 80 farklı nesne sınıfı içeriyor [14]. Bu devasa veriyle öğrenilen genel özellikler (kenarlar, şekiller, dokular) modelimize aktarıldı. Biz bunun üzerine kendi sentetik verilerimizle sadece araç tespiti için ince ayar (fine-tuning) yapılmıştır.

---

## 4.3 Görüntü İşleme: OpenCV

OpenCV (Open Source Computer Vision Library), görüntü ve video işleme için endüstri standardı kütüphane [15]. 1999'dan beri geliştiriliyor ve bugün hâlâ en yaygın kullanılan görüntü işleme aracı.

### Projede Nerede Kullanılıyor?

**Kamera ve video okuma:**
```python
cap = cv2.VideoCapture(0)        # Kameradan oku
cap = cv2.VideoCapture("video.mp4")  # Videodan oku
ret, frame = cap.read()          # Bir kare al
```

**Görüntü üzerine çizim:**
Araçların etrafındaki renkli çerçeveler, metin etiketleri ve poligon çizimleri OpenCV ile gerçekleştirilmektedir.

**IoU hesabı için piksel maskesi:**
Poligon ile bounding box arasındaki örtüşme oranını hesaplamak için piksel bazlı maske oluşturulmaktadır. OpenCV'nin `fillPoly` ve `rectangle` fonksiyonları bu işlemi sağlamaktadır.

**Park alanı etiketleme aracı:**
Mouse tıklamalarının yakalanması, poligon çizimi ve klavye girişi OpenCV penceresi üzerinden yürütülmektedir.

**Neden PIL/Pillow değil?**
PIL daha basit görüntü işlemler için iyi ama video okuma ve gerçek zamanlı işlem için OpenCV çok daha hızlı ve kapsamlı.

---

## 4.4 Derin Öğrenme Altyapısı: PyTorch

YOLOv8'in altında çalışan derin öğrenme framework'ü PyTorch. Model ağırlıklarını yüklemek, hesaplamaları yapmak, GPU'ya aktarmak gibi işlemlerin tamamı PyTorch üzerinden gerçekleşiyor.

PyTorch doğrudan kod yazılarak kullanılmamaktadır; Ultralytics bu katmanı soyutlamaktadır. Ancak PyTorch yüklü olmadan sistem çalışamamaktadır.

**Neden TensorFlow değil?**
YOLOv8'in resmi implementasyonu PyTorch tabanlı [16]. TensorFlow kullansaydık ya başka bir YOLO versiyonu kullanmak zorunda kalırdık ya da performans kayıpları yaşardık. Ayrıca son yıllarda araştırma topluluğu büyük ölçüde PyTorch'a geçti.

**Versiyon:** PyTorch 2.2.2 (CPU — GPU sürücü uyumsuzluğu nedeniyle)

---

## 4.5 Görsel Arayüz: PyQt5

Kullanıcının sistemi kolayca kullanabilmesi için bir masaüstü uygulaması gerekiyordu. Bu arayüzü PyQt5 ile geliştirilmiştir.

### PyQt5 Nedir?

Qt, C++ ile yazılmış çok güçlü bir arayüz framework'ü. PyQt5 ise Python'dan Qt'yi kullanmamızı sağlayan bir köprü. Profesyonel masaüstü uygulamalar yapmak için kullanılıyor.

### Neden PyQt5?

**Tkinter ile kıyaslama:** Tkinter Python'a dahil geliyor ama görsel olarak çok sınırlı, özelleştirmesi zor ve gerçek zamanlı video gösterimi için yavaş kalıyor.

**PySimpleGUI ile kıyaslama:** Daha basit ama karmaşık layout yönetimi zor.

**PyQt5'in avantajları:**
- QTimer ile gerçek zamanlı video döngüsü arayüzü bloklamadan çalışıyor
- QLabel ile görüntü gösterimi çok hızlı
- Fusion teması ile karanlık arayüz kolay yapılabiliyor
- Düğme, etiket, grid, çerçeve gibi tüm bileşenler hazır geliyor

### Arayüzde Kullanılan Başlıca Bileşenler

| Bileşen | Kullanım Amacı |
|---------|----------------|
| QMainWindow / QWidget | Ana pencere |
| QLabel | Video görüntüsü + metin |
| QPushButton | Kontrol butonları |
| QVBoxLayout / QHBoxLayout | Dikey ve yatay düzen |
| QGridLayout | Araç sayaç kartları |
| QFrame | Panel çerçeveleri |
| QTimer | Her 30ms'de kare güncelleme |
| QFileDialog | Dosya seçim penceresi |
| QImage / QPixmap | OpenCV frame'ini Qt'ye dönüştürme |

---

## 4.6 Model Değerlendirme: scikit-learn

Modelin ne kadar iyi çalıştığını ölçmek için scikit-learn kütüphanesini kullanılmaktadır.

### Ne İşe Yarıyor?

Eğitim ve test sonuçlarından şu metrikler hesaplanmaktadır:

**Accuracy (Doğruluk):** Tüm tahminlerin ne kadarının doğru olduğunu ölçmektedir.

**Precision (Kesinlik):** "Dolu" olarak tahmin edilenlerin ne kadarının gerçekten dolu olduğunu göstermektedir.

**Recall (Duyarlılık):** Gerçekten dolu olanların ne kadarının doğru tespit edilebildiğini ölçmektedir.

**F1-Score:** Precision ve Recall'un dengeli ortalamasıdır; ikisini tek bir değerde özetlemektedir.

**Confusion Matrix (Karışıklık Matrisi):** Hangi sınıfın hangisiyle karıştırıldığını ortaya koymaktadır.

**K-Fold Cross Validation:** Veriyi K parçaya bölüp her seferinde farklı bir parçayı test için kullanılmaktadır. Bu yöntem modelin gerçek performansını daha güvenilir şekilde ölçüyor; şansa bağlı iyi sonuç alma ihtimalini azaltıyor.

### Neden Bu Metrikler Önemli?

Yalnızca "accuracy" değerine bakmak yanıltıcı olabilmektedir. Örneğin veri setinin %90'ı aynı sınıfta ise, model tüm örnekleri o sınıfa atasa bile %90 doğruluk elde etmektedir; ancak bu model hiçbir işe yaramamaktadır. Precision, Recall ve F1-Score bu tür sorunları açığa çıkarmaktadır.

---

## 4.7 Veri İşleme: NumPy ve Pandas

**NumPy:** Sayısal hesaplamalar için kullanılmaktadır. IoU hesabında piksel maskeleri NumPy dizileri olarak işlenmektedir. Görüntü verileri de temelde NumPy dizilerinden oluşmaktadır (her piksel bir sayıdır).

**Pandas:** Eğitim loglarını ve metrik sonuçlarını tablolar halinde düzenlemek ve analiz etmek için kullanılmaktadır. Birden fazla epoch'taki loss değerlerinin karşılaştırılması ve sonuçların CSV'ye kaydedilmesi bu kapsamda gerçekleştirilmektedir.

---

## 4.8 Grafik ve Görselleştirme: Matplotlib ve Seaborn

Eğitim sürecini ve model performansını görsel olarak sunmak için bu iki kütüphane kullanılıyor.

**Matplotlib:** Loss eğrileri, mAP grafikleri ve precision-recall eğrilerinin çizilmesinde kullanılmaktadır.

**Seaborn:** Confusion matrix'in ısı haritası (heatmap) olarak görselleştirilmesinde kullanılmaktadır; hangi sınıfın hangisiyle karıştırıldığı renkli bir tablo biçiminde sunulmaktadır.

---

## 4.9 Konfigürasyon: PyYAML

Tüm ayarlar `configs/config.yaml` dosyasında tutulmaktadır. Model boyutu, confidence eşiği, IoU eşiği, sınıf isimleri ve train/val/test oranları gibi parametreler burada tanımlanmaktadır.

```yaml
model:
  name: yolov8n
  confidence_threshold: 0.5
  iou_threshold: 0.45

classes:
  vehicle_ids: [2, 3, 5, 7]
  names: [car, motorcycle, bus, truck]

training:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

Bu yaklaşımın avantajı, bir parametre değiştirilmek istendiğinde koda dokunulmasına gerek kalmadan yalnızca bu dosyanın düzenlenmesinin yeterli olmasıdır.

---

## 4.10 Teknoloji Özet Tablosu

| Teknoloji | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| Python | 3.12 | Ana programlama dili |
| Ultralytics YOLOv8 | 8.4.24 | Araç tespiti (nesne dedektörü) |
| PyTorch | 2.2.2 | YOLOv8 derin öğrenme altyapısı |
| OpenCV | 4.x | Görüntü işleme, video okuma, çizim |
| PyQt5 | 5.15+ | Masaüstü arayüzü |
| scikit-learn | 1.3+ | Model değerlendirme metrikleri |
| NumPy | 1.26.4 | Sayısal hesaplama, piksel maskesi |
| Pandas | 2.0+ | Veri analizi, log kayıtları |
| Matplotlib | 3.7+ | Loss ve metrik grafikleri |
| Seaborn | 0.12+ | Confusion matrix görselleştirme |
| PyYAML | 6.0+ | Konfigürasyon dosyası okuma |

---

## 4.11 Geliştirme Ortamı

Proje Windows 10 üzerinde geliştirilmiştir. Ekip üyeleri farklı bilgisayarlar kullandığından göreceli yol yönetimi ve ortak konfigürasyon yapısı oluşturulmuştur. Versiyon kontrolü için Git ve GitHub kullanılmaktadır.

**Donanım:**
- CPU: Intel Core i7-7700HQ
- GPU: NVIDIA GTX 1050 Ti Mobile (4GB VRAM)
- RAM: 16GB

Bu donanım sınırlı olsa da YOLOv8n gerçek zamanlı çalışabildiğinden proje için yeterli olmaktadır. Fine-tuning GPU desteğiyle başarıyla tamamlanmıştır.
