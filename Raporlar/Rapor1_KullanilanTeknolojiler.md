# 4. Kullanılan Teknolojiler

Bu bölümde projede kullanılan araçlar, kütüphaneler ve bunların seçilme gerekçeleri açıklanmaktadır.

---

## 4.1 Programlama Dili: Python 3.12

PyTorch, OpenCV, Ultralytics ve scikit-learn gibi yapay zeka ve görüntü işleme kütüphaneleri Python'da en iyi desteğe sahiptir. Birden fazla kişinin çalıştığı projede sözdiziminin sade ve okunabilir olması da tercih nedeni olmuştur.

---

## 4.2 Nesne Tespiti: YOLOv8 (Ultralytics)

YOLO "You Only Look Once" kelimelerinin baş harflerinden oluşmaktadır. Görüntüye tek seferinde bakarak hem nesnenin ne olduğunu hem de nerede olduğunu aynı anda tespit etmektedir. Bu sayede iki aşamalı yöntemlere (Faster R-CNN gibi) kıyasla çok daha hızlı çalışmaktadır.

| Özellik | YOLOv8 | Faster R-CNN | SSD |
|---------|--------|-------------|-----|
| Hız | Çok yüksek | Düşük | Yüksek |
| Doğruluk | Yüksek | Çok yüksek | Orta |
| Gerçek zamanlı | Evet | Hayır | Evet |
| Kullanım kolaylığı | Çok kolay | Zor | Orta |

Faster R-CNN doğrulukta öne geçebilmektedir; ancak gerçek zamanlı çalışmamaktadır. Literatürdeki güncel çalışmalar da park tespitinde YOLOv8 kullanmaktadır [7].

### Neden Nano (n) Versiyonu?

GTX 1050 Ti Mobile 4GB VRAM'de YOLOv8m ve üzeri modeller gerçek zamanlı çalışamamaktadır. YOLOv8n ise 30+ FPS'i rahatlıkla vermektedir.

| Model | mAP | Hız (ms) | Parametre |
|-------|-----|----------|-----------|
| YOLOv8n | 37.3 | 0.80 | 3.2M |
| YOLOv8s | 44.9 | 1.20 | 11.2M |
| YOLOv8m | 50.2 | 3.53 | 25.9M |

### Transfer Learning

Microsoft COCO veri setiyle [14] önceden eğitilmiş YOLOv8n modeli transfer learning yöntemiyle kullanılmıştır. COCO 330.000'den fazla görüntü ve 80 sınıf içermektedir. Bu ağırlıklar üzerine sentetik veriyle araç tespitine özgü ince ayar (fine-tuning) yapılmıştır [2].

---

## 4.3 Görüntü İşleme: OpenCV [15]

Kamera/video okuma, görüntü üzerine çizim, piksel maskesi oluşturma ve park alanı etiketleme aracındaki mouse etkileşimi OpenCV ile gerçekleştirilmektedir. PIL/Pillow video okuma ve gerçek zamanlı işlem için OpenCV kadar kapsamlı değildir.

---

## 4.4 Derin Öğrenme Altyapısı: PyTorch 2.2.2 [16]

YOLOv8'in resmi implementasyonu PyTorch tabanlıdır. Doğrudan kullanılmamakta; Ultralytics bu katmanı soyutlamaktadır. Ancak PyTorch yüklü olmadan sistem çalışamamaktadır. TensorFlow tercih edilseydi farklı bir YOLO versiyonu kullanmak gerekirdi.

---

## 4.5 Görsel Arayüz: PyQt5 5.15+

Qt, C++ ile yazılmış güçlü bir arayüz framework'üdür. PyQt5 Python'dan Qt'yi kullanmayı sağlamaktadır. Tkinter gerçek zamanlı video gösterimi için yavaş kalmakta, PySimpleGUI ise karmaşık layout yönetiminde yetersiz kalmaktadır. PyQt5'in temel avantajları:

- QTimer ile gerçek zamanlı video döngüsü arayüzü bloklamadan çalışmaktadır
- Fusion temasıyla karanlık arayüz kolayca uygulanmaktadır

| Bileşen | Kullanım Amacı |
|---------|----------------|
| QMainWindow / QWidget | Ana pencere |
| QLabel | Video görüntüsü + metin |
| QPushButton | Kontrol butonları |
| QVBoxLayout / QHBoxLayout / QGridLayout | Düzen yönetimi |
| QTimer | Her 30ms'de kare güncelleme |
| QImage / QPixmap | OpenCV frame'ini Qt'ye dönüştürme |

---

## 4.6 Model Değerlendirme: scikit-learn 1.3+

Eğitim ve test sonuçlarından Accuracy, Precision, Recall, F1-Score ve Confusion Matrix hesaplanmaktadır. Yalnızca accuracy değerine bakmak yanıltıcı olabilmektedir; veri setinin büyük çoğunluğu tek bir sınıfa aitse model tüm örnekleri o sınıfa atasaydı bile yüksek doğruluk elde edebilirdi. Precision, Recall ve F1-Score bu tür sorunları açığa çıkarmaktadır. K-Fold Cross Validation ile modelin gerçek performansı daha güvenilir biçimde ölçülmektedir.

---

## 4.7 Veri İşleme ve Görselleştirme

**NumPy 1.26.4:** IoU hesabında piksel maskeleri NumPy dizileri olarak işlenmektedir.

**Pandas 2.0+:** Eğitim loglarını ve metrik sonuçlarını tablo halinde düzenlemek için kullanılmaktadır.

**Matplotlib 3.7+:** Loss eğrileri ve mAP grafikleri çizilmektedir.

**Seaborn 0.12+:** Confusion matrix ısı haritası olarak görselleştirilmektedir.

---

## 4.8 Konfigürasyon: PyYAML 6.0+

Tüm ayarlar `configs/config.yaml` dosyasında tutulmaktadır. Parametre değiştirmek için koda dokunmaya gerek yoktur.

---

## 4.9 Teknoloji Özet Tablosu

| Teknoloji | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| Python | 3.12 | Ana programlama dili |
| Ultralytics YOLOv8 | 8.4.24 | Araç tespiti |
| PyTorch | 2.2.2 | YOLOv8 derin öğrenme altyapısı |
| OpenCV | 4.x | Görüntü işleme, video okuma |
| PyQt5 | 5.15+ | Masaüstü arayüzü |
| scikit-learn | 1.3+ | Model değerlendirme metrikleri |
| NumPy | 1.26.4 | Sayısal hesaplama |
| Pandas | 2.0+ | Veri analizi, log kayıtları |
| Matplotlib | 3.7+ | Grafik çizimi |
| Seaborn | 0.12+ | Confusion matrix görselleştirme |
| PyYAML | 6.0+ | Konfigürasyon dosyası |

---

## 4.10 Geliştirme Ortamı

Proje Windows 10 üzerinde geliştirilmiştir. Versiyon kontrolü için Git ve GitHub kullanılmaktadır.

**Donanım:**
- CPU: Intel Core i7-7700HQ
- GPU: NVIDIA GTX 1050 Ti Mobile (4GB VRAM)
- RAM: 16GB

YOLOv8n bu donanımda gerçek zamanlı çalışabilmektedir. Fine-tuning GPU desteğiyle başarıyla tamamlanmıştır.
