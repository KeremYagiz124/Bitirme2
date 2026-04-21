# 6. SONUÇ VE GELECEK ÇALIŞMALAR

## 6.1 Genel Değerlendirme

Bu projede kamera görüntülerinden araç tespiti ve park uygunluğu analizi yapan yapay zeka tabanlı bir sistem geliştirilmesi hedeflenmiştir. İlk iki haftada gerçekleştirilen çalışmalar şu şekilde özetlenebilir:

- YOLOv8 tabanlı araç tespit sistemi çalışır duruma getirilmiştir.
- Park alanı etiketleme aracı (ZoneAnnotator) tamamlanmıştır.
- IoU tabanlı doluluk analizi altyapısı hazırlanmıştır.
- 1000 görüntülük sentetik veri üretilmiş ve fine-tuning tamamlanmıştır (mAP@0.5: %99.5).
- PyQt5 arayüzü tamamlanmış; kamera, video ve resim üzerinde gerçek zamanlı analiz yapılabilmektedir.

Başlangıçta belirlenen hipotez — "YOLOv8 ile poligon tabanlı park analizinin birleştirilmesi, park uygunluğunu üç sınıfta %85 ve üzeri doğrulukla sınıflandırmak için yeterlidir" — henüz tam olarak test edilememiştir. Kurulan altyapı bu testi mümkün kılmaktadır.

---

## 6.2 Literatürle Karşılaştırma

2017'deki mAlexNet [3] ve 2019'daki CarNet [4] gibi çalışmalara kıyasla çok daha güncel bir model (YOLOv8) kullanılmaktadır. Söz konusu çalışmalar yalnızca "dolu/boş" kararı üretirken geliştirilen projede "yasak bölge" sınıfı da eklenmiştir.

da Luz vd.'nin 2024 çalışması [7] mevcut projeye en yakın referanstır; YOLOv8 + ROI analizi kombinasyonu büyük ölçüde örtüşmektedir. Ticari tarafta ise Parquery [11] ile aynı felsefe paylaşılmaktadır: mevcut kameraya yapay zeka ekle, ek altyapı gerektirme.

---

## 6.3 Karşılaşılan Zorluklar

**PyTorch/NumPy sürüm uyumsuzluğu:** PyTorch 2.10 kurulduğunda DLL hatası alınmıştır. PyTorch 2.2.2 ve NumPy 1.26.4 kombinasyonuna geçilerek sorun giderilmiştir.

**GPU kullanılamaması:** Başlangıçta sürücü uyumsuzluğu nedeniyle CPU kullanılmıştır. CUDA 12.1 destekli PyTorch kurulumunun ardından GPU aktif hale getirilmiştir.

**Class ID uyumsuzluğu:** COCO modelinde araç ID'leri 2, 3, 5, 7 iken fine-tuned modelde 0, 1, 2, 3'tür. VehicleDetector'a model adına göre otomatik seçim eklenerek sorun çözülmüştür.

**Timer çoklu bağlantı:** Arayüzde her buton tıklamasında timer'a yeni bağlantı eklenmekteydi. Timer başta bir kez bağlanacak şekilde yeniden düzenlenmiştir.

---

## 6.4 Gelecek Çalışmalar

**Yakın vadeli (Hafta 3-4):**
- IoU pipeline arayüze entegre edilecektir.
- Üç sınıflı çıktı ekrana yansıtılacaktır.
- Gerçek kamera görüntüleriyle sistem test edilecektir.
- PKLot veri seti [8] ile baseline karşılaştırması gerçekleştirilecektir.

**Orta vadeli (Hafta 5-6):**
- MobileNetV2 [6] ikinci doğrulama katmanı eklenecektir.
- Ablasyon çalışması yapılacaktır: "Yalnızca IoU" ile "IoU + MobileNetV2" karşılaştırılacaktır.
- Confusion matrix, F1-Score ve mAP metrikleri hesaplanacaktır.
- Veri artırma teknikleri uygulanacaktır.

**İleri vadeli (Hafta 7-8):**
- ByteTrack [10] araç takibi entegre edilecektir.
- FPS ve gecikme ölçümleri gerçekleştirilecektir.
- Farklı kamera açılarında sistem test edilecektir.
- Final demo videosu hazırlanacaktır.

---

## KAYNAKLAR

[1] Spherical Insights, (2024), Smart Parking Systems Market Report, https://www.sphericalinsights.com

[2] Jocher, G. v.d., (2023), "Ultralytics YOLOv8", https://github.com/ultralytics/ultralytics

[3] Amato, G. v.d., (2017), "Deep learning for decentralized parking lot occupancy detection", Expert Systems with Applications, Vol.72, pp.327-334.

[4] Nurullayev, S. ve Lee, S.W., (2019), "Generalized parking occupancy analysis based on dilated convolutional neural network", Sensors, Vol.19, No.2, s.277.

[5] Almeida, P.R.L. v.d., (2015), "PKLot – A robust dataset for parking lot classification", Expert Systems with Applications, Vol.42, No.11, pp.4937-4949.

[6] Sandler, M. v.d., (2018), "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018, Salt Lake City, pp.4510-4520.

[7] da Luz, G.P.C.P. v.d., (2024), "Smart parking with pixel-wise ROI selection for vehicle detection using YOLOv8, YOLOv9, YOLOv10, and YOLOv11", arXiv:2412.01983.

[8] Yuldashev, B. v.d., (2023), "Parking lot occupancy detection with improved MobileNetV3", Sensors, Vol.23, No.18.

[9] Grbić, R. ve Koch, T., (2023), "APSD-OC: Automatic parking space detection and occupancy classification", Expert Systems with Applications.

[10] Zhang, Y. v.d., (2022), "ByteTrack: Multi-object tracking by associating every detection box", ECCV 2022, Tel Aviv, pp.1-21.

[11] Parquery AG, (2024), Parquery Parking Intelligence, https://parquery.com

[12] Quercus Technologies, (2024), BirdWatch & QAI Systems, https://www.quercus-technologies.com

[13] Xie, X. v.d., (2017), "Real-time illegal parking detection system based on deep learning", arXiv:1710.02546.

[14] Lin, T.Y. v.d., (2014), "Microsoft COCO: Common Objects in Context", ECCV 2014, Zurich, pp.740-755.

[15] OpenCV Documentation, (2024), https://opencv.org

[16] PyTorch Documentation, (2024), https://pytorch.org
