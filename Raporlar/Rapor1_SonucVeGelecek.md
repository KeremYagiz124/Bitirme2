# 6. SONUÇ VE GELECEK ÇALIŞMALAR

## 6.1 Genel Değerlendirme

Bu projede kamera görüntülerinden araç tespiti ve park uygunluğu analizi yapan bir yapay zeka sistemi geliştirilmesi hedeflenmiş; ilk iki haftada kurulan temel, projenin geri kalanı için sağlam bir zemin oluşturmuştur.

Şu ana kadar gerçekleştirilen çalışmalar şu şekilde özetlenebilir:

- YOLOv8 tabanlı araç tespit sistemi çalışır duruma getirilmiştir. Model araba, otobüs, kamyon ve motosikleti gerçek zamanlı olarak tespit etmektedir.
- Park alanı etiketleme aracı tamamlanmıştır. Herhangi bir park alanı fotoğrafı üzerinde poligon çizilerek park ve yasak bölgeler tanımlanabilmektedir.
- IoU tabanlı doluluk analizi için altyapı hazırlanmıştır. Araç bounding box ile park poligonu arasındaki örtüşme hesaplanabilmektedir.
- Sentetik veri üreteci yazılmış ve 1000 görüntü üretilmiştir. Veriler %80 eğitim, %20 doğrulama olarak ayrılmıştır.
- Fine-tuning altyapısı hazırlanmıştır. YOLOv8'i üretilen veriyle yeniden eğitme betiği çalışır durumdadır.
- PyQt5 arayüzü tamamlanmıştır. Kamera, video ve resim üzerinde gerçek zamanlı analiz gerçekleştirilebilmektedir.

Başlangıçta belirlenen hipotez hatırlandığında — "YOLOv8 ile poligon tabanlı park analizinin birleştirilmesi, park uygunluğunu üç sınıfta %85 ve üzeri doğrulukla sınıflandırmak için yeterlidir" — bu hipotez henüz tam olarak test edilememiştir. Ancak kurulan altyapı bu testi gerçekleştirmeyi mümkün kılmaktadır.

---

## 6.2 Literatürle Karşılaştırma

Gerçekleştirilen çalışmalar literatürdeki benzer çalışmalarla kıyaslandığında şu tablo ortaya çıkmaktadır:

Geliştirilen sistem, 2017'deki mAlexNet [3] ve 2019'daki CarNet [4] gibi çalışmalara kıyasla çok daha güncel bir model (YOLOv8) kullanmaktadır. Söz konusu çalışmalarda yalnızca "dolu/boş" kararı üretilirken bu projede "yasak bölge ihlali" sınıfı da eklenmiştir.

da Luz vd.'nin 2024 çalışması [7] mevcut projeye en yakın referans niteliğindedir. Bu çalışmada da YOLOv8 ile ROI analizi bir arada kullanılmaktadır. Geliştirilen sistemdeki fark, yasak bölge tespiti ve üç sınıflı karar motorunun eklenmesidir.

Parquery [11] gibi ticari sistemlerle aynı felsefe paylaşılmaktadır: mevcut kameraya yapay zeka ekle, ek altyapı gerektirme.

---

## 6.3 Karşılaşılan Zorluklar

Proje boyunca aşağıdaki teknik sorunlarla karşılaşılmıştır:

**PyTorch ve NumPy sürüm uyumsuzluğu:** PyTorch 2.10 kurulduğunda DLL hatası alınmıştır. NumPy 2.x ile PyTorch 2.2.x sürümleri çakışmıştır. PyTorch 2.2.2 ve NumPy 1.26.4 kombinasyonuna geçilerek sorun giderilmiştir.

**GPU kullanılamaması:** Başlangıçta sürücü uyumsuzluğu nedeniyle GPU yerine CPU kullanılmıştır. CUDA 12.1 destekli PyTorch kurulumunun ardından GPU aktif hale getirilmiştir.

**Class ID uyumsuzluğu:** COCO modelinde araç class ID'leri 2, 3, 5, 7 iken sentetik veriyle eğitilen modelde 0, 1, 2, 3 olarak belirlenmiştir. Bu durum, fine-tuned modelin hiçbir şey tespit edememesine yol açabilecekti. VehicleDetector sınıfına model adına göre otomatik seçim eklenerek sorun çözülmüştür.

**Timer çoklu bağlantı:** Arayüzde kamera ve video butonları her tıklamada timer'a yeni bir bağlantı ekliyordu. Bu durum her karede işlemin katlanarak çalışmasına neden olmaktaydı. Timer başta bir kez bağlanacak şekilde yeniden düzenlenmiştir.

---

## 6.4 Gelecek Çalışmalar

Projenin önümüzdeki haftalarında yapılması planlanan geliştirmeler aşağıda sıralanmıştır:

**Yakın vadeli (Hafta 3-4):**
- IoU pipeline'ı arayüze tam entegre edilecektir. Şu an araç tespiti görselleştirilmekte ancak park alanı kararı ekrana yansıtılmamaktadır.
- Kural tabanlı karar motoru yazılacak ve üç sınıflı çıktı üretilecektir.
- Okul otoparkından gerçek kamera görüntüsü çekilip sistem test edilecektir.
- PKLot veri seti [8] indirilerek baseline karşılaştırması gerçekleştirilecektir.

**Orta vadeli (Hafta 5-6):**
- MobileNetV2 [6] ikinci doğrulama katmanı eklenecektir.
- Ablasyon çalışması kapsamında "Yalnızca IoU" ile "IoU + MobileNetV2" karşılaştırılacaktır.
- Confusion matrix, F1-Score, mAP metrikleri hesaplanacaktır.
- Veri artırma teknikleri uygulanacaktır: parlaklık değişimi, bulanıklaştırma, döndürme.

**İleri vadeli (Hafta 7-8):**
- ByteTrack [10] araç takibi entegre edilecektir. Geçen araç ile park edilmiş araç ayırt edilebilecektir.
- FPS ve gecikme ölçümleri gerçekleştirilecektir.
- Farklı kamera açılarında sistem test edilecektir.
- Final demo videosu hazırlanacaktır.

---

## 6.5 Projenin Potansiyeli

Bu proje akademik bir çalışma olarak başlamıştır; ancak benzer sistemlerin ticari hayata geçtiği görülmektedir. Parquery [11], Quercus [12] gibi şirketler aynı fikri milyarlarca dolarlık bir pazara dönüştürmüştür.

Geliştirilen sistem şu an sınırlı bir donanımda çalışmakta ve birkaç eksikliği bulunmaktadır. Ancak temel mimari doğru kurulmuştur: mevcut kamera + yapay zeka + yazılım. Bu üçlü, zeminde sensör kazımadan, yüksek altyapı maliyeti olmadan akıllı park yönetiminin gerçekleştirilebileceğini göstermektedir.

İleride bu sistem Jetson Nano gibi düşük güçlü bir edge cihaza taşınabilir, birden fazla kamerayla çalışacak şekilde genişletilebilir ve gerçek zamanlı uyarı sistemiyle entegre edilebilir.

---

## 6.6 Özet

Bu raporda aşağıdaki konular ele alınmıştır:

- Kentsel park sorunu ve mevcut çözümlerin yetersizlikleri tanımlanmıştır.
- Literatürdeki akademik çalışmalar ve ticari sistemler incelenmiş, projenin bu ekosistem içindeki yeri belirlenmiştir.
- Sistemin mimarisi, bileşenleri ve bunların birbirleriyle nasıl çalıştığı açıklanmıştır.
- Kullanılan teknolojiler ve seçim gerekçeleri ortaya konmuştur.
- Şimdiye kadar gerçekleştirilen çalışmalar ve elde edilen ilk sonuçlar paylaşılmıştır.

Proje hâlâ devam etmektedir. Önümüzdeki haftalarda IoU pipeline entegrasyonu, gerçek kamera testi ve model metriklerinin ölçülmesi öncelikli hedefler arasında yer almaktadır.

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
