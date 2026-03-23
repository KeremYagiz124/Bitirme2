PROJE ADI:
Kamera Görüntülerinden Araç Tespiti ve Park Uygunluğu Analizi için Yapay Zeka Tabanlı Sistem

PROJE ÖZETİ:
Düzeltilmiş Metin (kaynakça listesi aynı kalıyor, sadece atıflar güncellendi):
Bu proje, kentsel alanlarda giderek büyüyen park denetimi sorununa teknolojik bir çözüm sunmak amacıyla, kamera görüntülerini gerçek zamanlı olarak analiz eden yapay zeka tabanlı bir sistem geliştirmeyi hedeflemektedir. Artan araç sayısı ve yetersiz park altyapısı nedeniyle şehir merkezlerinde sıkça karşılaşılan yasadışı park sorunları, hem trafik akışını bozmakta hem de güvenlik risklerine yol açmaktadır. Mevcut denetim yöntemleri büyük ölçüde manuel kontrole dayandığından hız ve otomasyon açısından yetersiz kalmaktadır. Geliştirilen bu sistem, söz konusu sorunu insan müdahalesine ihtiyaç duymadan, kamera görüntüsünden hareketle tamamen otomatik biçimde çözmeyi amaçlamaktadır.
Sistem üç temel bileşenden oluşmaktadır:
1.	YOLOv8 mimarisi [1] ile gerçek zamanlı araç tespiti: Derin öğrenme tabanlı YOLOv8 modeli, kamera görüntüsü üzerindeki araçları yüksek hız ve doğrulukla tespit etmektedir [2]. COCO veri seti [3] üzerinde önceden eğitilmiş ağırlıklar, transfer learning yöntemiyle projeye özgü veri setine uyarlanacak; her araç için konum koordinatları ve güven skoru üretilecektir.
2.	OpenCV [4] tabanlı görüntü işleme ile yol çizgileri, park alanları ve yasaklı bölgelerin otomatik olarak belirlenmesi: Görüntü üzerinde Gaussian Blur ile gürültü azaltma, Canny Edge Detection [5] ile kenar tespiti ve Hough Line Transform [6] ile şerit/çizgi analizi uygulanacaktır. Perspektif dönüşümü ile park alanı sınırları poligon tespiti yöntemiyle koordinat bazında belirlenecektir. Tespit edilen araç konumu ile park alanı poligonu arasındaki örtüşme oranı IoU (Intersection over Union) analizi ile hesaplanacaktır.
3.	Kural tabanlı ve CNN destekli karar mekanizması ile park uygunluğu sınıflandırması: Nesne tespiti ve görüntü işleme çıktıları bir kural motoru üzerinden birleştirilerek nihai sınıflandırma kararı üretilmektedir. Kararın güvenilirliğini artırmak amacıyla MobileNetV2 [7] tabanlı yardımcı bir CNN sınıflandırıcısı, sonuçları ikinci bir katmanda doğrulayacaktır.
Sistem, bu üç bileşenin çıktısını birleştirerek üç çıktı sınıfından birini üretmektedir: 'Park Edilebilir', 'Park Edilemez' ve 'Park Alanı Dolu'. Eğitim süreci için COCO Dataset [3], UA-DETRAC [8] ve PKLot [9] gibi açık kaynaklı veri setleri kullanılacak; veri artırma teknikleriyle eğitim seti çeşitlendirilerek modelin farklı ışık koşulları ve kamera açılarına karşı dayanıklılığı artırılacaktır. Model performansı; Accuracy, Precision, Recall, F1-Score ve mAP (mean Average Precision) metrikleri ile değerlendirilecek, karışıklık matrisi ve k-katlı çapraz doğrulama analizleriyle desteklenecektir. Son olarak sistem, simülasyon ortamında video veri setleri üzerinde gerçek zamanlı performans açısından test edildikten sonra, gelecekte mevcut trafik kamerası altyapılarına entegre edilebilecek ve akıllı şehir uygulamalarında kullanılabilecek ölçeklenebilir bir yapı sunması hedeflenmektedir.


Projeye Yön Verecek Araştırma Sorusu Ve/Veya Hipotez:
Kentsel alanlarda trafik kamerası görüntülerinden hareketle, bir aracın bulunduğu konumda park etmesinin uygun olup olmadığı; derin öğrenme tabanlı nesne tespiti ve görüntü işleme tekniklerinin birlikte kullanımıyla, insan müdahalesine ihtiyaç duyulmaksızın, gerçek zamanlı ve güvenilir biçimde otomatik olarak belirlenebilir mi?
Alt Araştırma Soruları:
1.	YOLOv8 tabanlı nesne tespit modeli, farklı ışık koşulları, hava şartları ve kalabalık trafik senaryolarında araçları ne ölçüde doğru ve tutarlı biçimde tespit edebilir?
2.	Canny kenar tespiti, Hough dönüşümü ve perspektif dönüşümü gibi görüntü işleme teknikleri, gerçek ortam kamera görüntülerinde yol çizgilerini, park alanlarını ve yasaklı bölgeleri yeterli hassasiyetle ve güvenilir biçimde belirleyebilir mi?
3.	Geliştirilen park uygunluğu sınıflandırıcısı, farklı kamera açıları ve çözünürlük düzeylerinde %85 ve üzeri doğruluk ve F1-Score değerine ulaşabilir mi?
4.	YOLOv8 ile elde edilen bounding box koordinatları ve görüntü işlemeden elde edilen park alanı poligonları arasında gerçekleştirilen IoU analizi, aracın park konumunu yeterli doğrulukta tespit etmek için tek başına yeterli midir, yoksa ek bir sınıflandırma katmanına ihtiyaç duyulmakta mıdır?
YOLOv8 tabanlı nesne tespiti ile OpenCV tabanlı görüntü işleme tekniklerinin birlikte kullanılması ve bu iki bileşenden elde edilen çıktıların kural tabanlı bir karar motoru aracılığıyla birleştirilmesi; park uygunluğunu 'Park Edilebilir', 'Park Edilemez' ve 'Park Alanı Dolu' şeklinde üç sınıfta, %85 ve üzeri doğruluk oranıyla sınıflandıran, gerçek zamanlı çalışabilen bir sistem ortaya koymak için yeterlidir.


Projede Kullanılacak Yöntem Ve Metotlar:
1. VERİ TOPLAMA VE HAZIRLAMA
Açık kaynaklı veri setleri: COCO Dataset [3], UA-DETRAC [8] ve PKLot [9]
Etiketleme: araç, park alanı, yasak bölge (YOLO formatı: .txt bbox koordinatları)
Veri artırma: döndürme, çevirme, parlaklık/kontrast değişimi
2. ÖN İŞLEME
Yeniden boyutlandırma, normalizasyon (OpenCV [6])
Gürültü azaltma: Gaussian Blur, Median Filter
Kenar tespiti: Canny Edge Detection
Çizgi/şerit tespiti: Hough Line Transform
Renk uzayı dönüşümü (BGR→HSV)
3. ARAÇ TESPİTİ
YOLOv8 [1] (Ultralytics) modelinin seçimi ve yapılandırması; model, YOLO mimarisinin önceki sürümleri [10] üzerine inşa edilmiş güncel bir versiyondur
COCO [3] pre-trained ağırlıklardan fine-tuning (transfer learning)
Bounding box tespiti ve güven skoru filtrelemesi
PyTorch [11] altyapısı kullanılarak model eğitimi ve çıkarsama (inference) gerçekleştirilecektir
4. PARK ALANI ANALİZİ
Perspektif dönüşümü ile üstten görünüm [5]
Poligon tespiti ile park koordinatı belirleme
IoU (Intersection over Union) analizi
5. SINIFLANDIRMA VE KARAR MEKANİZMASI
Kural tabanlı karar motoru: Park Edilebilir / Park Edilemez / Park Alanı Dolu
MobileNetV2 [7] CNN ile sonuç doğrulaması
6. MODEL DEĞERLENDİRME
Eğitim/Doğrulama/Test: %70 / %15 / %15
Metrikler: Accuracy, Precision, Recall, F1-Score, mAP
Karışıklık matrisi analizi ve k-fold cross-validation
Ablasyon çalışması: IoU tek başına vs. IoU + MobileNetV2 [7] karşılaştırması
7. SİMÜLASYON VE TEST
Video veri setleri üzerinde simülasyon testi
Gerçek zamanlı performans analizi (FPS, latency)
Farklı ışık koşulları ve kamera açılarında performans karşılaştırması


Projede Faydalanılacak Kaynaklar:
1.  Jocher G. et al. (2023). Ultralytics YOLOv8.
    https://github.com/ultralytics/ultralytics
2.  Redmon J., Divvala S., Girshick R., Farhadi A. (2016). You Only Look Once: Unified,
    Real-Time Object Detection. CVPR. https://arxiv.org/abs/1506.02640
3.  Lin T.Y. et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.
    https://cocodataset.org
4.  OpenCV Documentation. https://opencv.org
5.  Gonzalez R.C. & Woods R.E. (2018). Digital Image Processing (4th ed.). Pearson.
6.  Szeliski R. (2010). Computer Vision: Algorithms and Applications. Springer.
7.  Sandler M., Howard A., Zhu M., Zhmoginov A., Chen L.C. (2018). MobileNetV2:
    Inverted Residuals and Linear Bottlenecks. CVPR.
8.  Wen L. et al. (2015). UA-DETRAC: A New Benchmark and Protocol for
    Multi-Object Detection and Tracking. arXiv:1511.04136.
9.  de Almeida P.R.L. et al. (2015). PKLot – A robust dataset for parking lot
    classification. Expert Systems with Applications, 42(11), 4937–4949.
--- (metinde doğrudan atıf yapılmayan kaynaklar) ---
10. Bochkovskiy A., Wang C.Y., Liao H.Y.M. (2020). YOLOv4: Optimal Speed and
    Accuracy of Object Detection. arXiv:2004.10934.
11. PyTorch Documentation. https://pytorch.org

