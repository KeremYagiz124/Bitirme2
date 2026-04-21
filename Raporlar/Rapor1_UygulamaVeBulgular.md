# 5. Uygulama ve Bulgular

Bu bölümde projenin şu ana kadar hangi aşamalardan geçtiği ve elde edilen sonuçlar aktarılmaktadır. Proje hâlâ devam ettiği için bazı metrik sonuçları ilerleyen haftalarda güncellenecektir.

---

## 5.1 Geliştirme Süreci

**Hafta 1 — Temel Kurulum ve Baseline:** Klasör yapısı oluşturulmuş, kütüphaneler kurulmuş, `VehicleDetector` sınıfı yazılmış ve YOLOv8n ilk kez COCO ağırlıklarıyla çalıştırılmıştır. Park alanı etiketleme aracı (`ZoneAnnotator`) geliştirilmiş, komut satırı test betiği tamamlanmıştır. İlk testte model bir otobüs görüntüsünde %93 güven skoru ile başarılı tespit yapmıştır.

**Hafta 2 — Sentetik Veri ve Arayüz:** 1000 görüntülük sentetik veri üretilmiş, YOLOv8 fine-tuning betiği hazırlanmış ve PyQt5 tabanlı görsel arayüz geliştirilmiştir. Araç tespiti sonuçları arayüze entegre edilmiş; araç tipi sayaçları, renk kodlaması ve FPS göstergesi eklenmiştir.

---

## 5.2 Baseline Sistem Sonuçları

COCO ağırlıklarıyla çalışan YOLOv8n modeli [2] park senaryosuna özel herhangi bir eğitim yapılmadan test edilmiştir. Güven skorları genellikle %80-95 arasında ölçülmüştür. Büyük araçlar (otobüs, kamyon) daha güvenilir tespit edilmiş; kısmen kapalı araçlarda ve olumsuz kamera açılarında güven skoru düşmüştür. Motosikletler küçük olduğu için zaman zaman kaçırılmıştır.

---

## 5.3 Sentetik Veri Üretimi

| Parametre | Değer |
|-----------|-------|
| Toplam görüntü | 1000 |
| Eğitim seti | 800 (%80) |
| Doğrulama seti | 200 (%20) |
| Görüntü boyutu | 640×480 px |
| Araç/görüntü | 1-5 (rastgele) |

Her araç tipi basit renkli dikdörtgen ve detaylarla temsil edilmektedir. YOLO formatında normalize edilmiş etiket dosyaları otomatik oluşturulmaktadır.

---

## 5.4 Fine-Tuning Sonuçları

Fine-tuning GPU desteğiyle tamamlanmıştır. Early stopping 40. epoch'ta devreye girmiştir.

| Metrik | Değer |
|--------|-------|
| mAP@0.5 | %99.5 |
| En iyi model | `runs/detect/models/fine_tuned/yolov8_fine_tuned2/weights/best.pt` |

---

## 5.5 Görsel Arayüz

Geliştirilen PyQt5 arayüzü kamera akışı, video oynatma ve tek resim analizini desteklemektedir. Karanlık tema (Fusion stili), gerçek zamanlı araç sayaçları, FPS göstergesi ve durum çubuğu içermektedir.

GTX 1050 Ti Mobile donanımında ölçülen yaklaşık FPS değerleri:

| Kaynak | Çözünürlük | Yaklaşık FPS |
|--------|-----------|--------------|
| Video (720p) | 1280×720 | ~18-22 |
| Video (480p) | 854×480 | ~25-30 |
| Resim | Değişken | Anlık |

Not: GPU sürücü uyumsuzluğu giderilerek GPU desteği aktif hale getirilmiştir. Değerler GPU ile ölçülmüştür.

---

## 5.6 IoU Tabanlı Doluluk Analizi

Park alanı JSON dosyası yüklendiğinde sistem araç tespiti sonuçlarını bu alanlarla karşılaştırmaktadır.

| Durum | IoU (park) | IoU (yasak) | Karar |
|-------|-----------|-------------|-------|
| Park alanı dolu | > 0.3 | < 0.3 | Park Alanı Dolu |
| Yasak ihlali | < 0.3 | > 0.3 | Park Edilemez |
| Eşleşme yok | < 0.3 | < 0.3 | Serbest alan |
| Park alanı boş | Araç yok | — | Park Edilebilir |

IoU pipeline kodu çalışır durumdadır; arayüze entegrasyonu ilerleyen aşamada tamamlanması planlanmaktadır.

---

## 5.7 Mevcut Sistem Sınırlamaları

**Araç takibi yok:** Geçen bir araç geçici olarak "Dolu" kararı tetikleyebilmektedir. ByteTrack entegrasyonu planlanmaktadır.

**Gece ve yağmur testi yapılmadı:** Veri artırma teknikleriyle (parlaklık değişimi, bulanıklaştırma) bu kısmen telafi edilecektir.

**Sentetik veri gerçek değil:** Gerçek kamera görüntüleriyle doku ve ışık farklılıkları mevcuttur. PKLot veri setiyle [8] ek eğitim planlanmaktadır.

---

## 5.8 Planlanan Ölçümler

| Metrik | Açıklama | Hedef |
|--------|----------|-------|
| Precision | Doğru pozitif / Tüm pozitif tahminler | > 0.85 |
| Recall | Doğru pozitif / Tüm gerçek pozitifler | > 0.80 |
| F1-Score | Precision ve Recall dengesi | > 0.82 |
| FPS | Saniyede işlenen kare sayısı | > 25 |
| Confusion Matrix | Sınıflar arası karışıklık oranları | — |
