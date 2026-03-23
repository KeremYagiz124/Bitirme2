İÇİNDEKİLER

İÇİNDEKİLER	1
1.GİRİŞ	2
2.PROBLEM TANIMI	2
3.PROJENİN AMACI	3
4.LİTERATÜR ÖZETİ	3
5.KULLANILACAK TEKNOLOJİLER	4
6.SİSTEM TASARIMI	5
6.1. Görüntü Alma	5
6.2. Ön İşleme	5
6.3. Araç Tespiti	5
6.4. Park Alanı Analizi	5
6.5. Karar Mekanizması	5
7.VERİ SETİ HAZIRLAMA	6
8.MODEL EĞİTİMİ	6
9.BEKLENEN SONUÇLAR	7
10.KAYNAKÇA	5
 
1. GİRİŞ
Kentleşmenin artmasıyla birlikte araç sayısında ciddi bir yükseliş meydana gelmiştir. Bu durum özellikle şehir merkezlerinde park alanlarının yetersiz kalmasına ve trafik düzeninin bozulmasına neden olmaktadır. Yol kenarlarına yapılan hatalı parklar trafik akışını olumsuz etkileyebilmekte ve güvenlik sorunlarına yol açabilmektedir.

Bu projede, simülasyon ortamında kamera görüntülerini analiz ederek bir aracın bulunduğu konumda park etmesinin uygun olup olmadığını otomatik olarak belirleyen bir sistem geliştirilmesi amaçlanmaktadır. Sistem, görüntü işleme teknikleri ve makine öğrenmesi algoritmaları kullanarak park alanlarını, yol çizgilerini ve araç konumlarını analiz edecektir.

2. PROBLEM TANIMI
Mevcut şehir altyapılarında park denetimi genellikle insan kontrolüne veya sabit işaretlemelere dayanmaktadır. Bu yöntemler hem zaman alıcıdır hem de otomasyon seviyesi düşüktür.

Temel problem:
•	Kamera görüntüsünden araçları tespit etmek
•	Park alanlarının veya yasak bölgelerin belirlenmesi
•	Aracın bulunduğu konumun park için uygun olup olmadığını belirlemek

Bu problem bilgisayarla görme ve sınıflandırma teknikleri ile çözülebilecek bir görüntü analiz problemidir.




3. PROJENİN AMACI
Bu çalışmanın amacı:
•	Kamera görüntülerinden araçları otomatik olarak tespit eden
•	Park alanlarını veya yasak bölgeleri algılayan
•	Aracın park etmesinin uygun olup olmadığını sınıflandıran bir yapay zeka tabanlı sistem geliştirmektir.

4. LİTERATÜR ÖZETİ
Bilgisayarla görme alanında araç tespiti ve park alanı analizi üzerine birçok çalışma yapılmıştır.

Örneğin:

•	YOLO tabanlı nesne tespit modelleri araç algılama için yaygın olarak kullanılmaktadır.
•	CNN tabanlı sınıflandırma modelleri park alanlarının dolu veya boş olduğunu belirlemek için kullanılmaktadır.
•	Akıllı otopark sistemleri şehir içi trafik yönetimi için geliştirilen önemli uygulamalardandır.

Bu projede benzer yaklaşımlar incelenerek uygun bir model seçilecektir.





5. KULLANILACAK TEKNOLOJİLER
Projede aşağıdaki araç ve teknolojilerin kullanılması planlanmaktadır:
Programlama Dili
•	Python

Görüntü İşleme
•	OpenCV

Makine Öğrenmesi / Derin Öğrenme
•	TensorFlow veya PyTorch
•	YOLO (You Only Look Once) Nesne Tespit Modeli

Veri Seti
•	Açık kaynak trafik ve araç veri setleri
•	Gerekirse gerçek görüntülerden oluşturulan veri seti (sonrasında simülasyona entegre edilecek.)








6. SİSTEM TASARIMI
Sistem aşağıdaki aşamalardan oluşacaktır.

6.1. Görüntü Alma
•	Trafik kamerası, cep telefonu ve/veya veri seti görüntülerinin alınması

6.2. Ön İşleme
•	Görüntü boyutlandırma
•	Gürültü azaltma
•	Kenar ve çizgi tespiti

6.3. Araç Tespiti
•	YOLO modeli kullanılarak araçların tespiti

6.4. Park Alanı Analizi
•	Yol çizgileri ve park alanlarının belirlenmesi
•	Araç konumunun analiz edilmesi

6.5. Karar Mekanizması
•	Bu sürecin aşamalar ile sistemi kurduktan sonra üç çıktı sonucu elde edilmeli:
•	Park edilebilir
•	Park edilemez
•	Park alanı dolu


7. VERİ SETİ HAZIRLAMA
Modelin eğitilebilmesi için etiketlenmiş görüntüler gerekmektedir.
Veri seti şu şekilde oluşturulacaktır:
•	Araç bulunan görüntüler
•	Park alanı görüntüleri
•	Yasak park bölgeleri

Her görüntü aşağıdaki etiketlerle işaretlenecektir:
•	araç
•	park alanı
•	yasak bölge

8. MODEL EĞİTİMİ
•	Makine öğrenmesi modeli aşağıdaki adımlarla eğitilecektir:
•	Veri setinin hazırlanması
•	Eğitim ve test verilerinin ayrılması
•	Modelin eğitilmesi
•	Performans ölçümü
•	Değerlendirme metrikleri:
•	Accuracy
•	Precision
•	Recall
•	F1-score




9. BEKLENEN SONUÇLAR
•	Kamera görüntülerinden araçları doğru şekilde tespit etmesi
•	Park alanlarını analiz edebilmesi
•	Park uygunluğunu yüksek doğrulukla belirleyebilmesi beklenmektedir.
•	Bu sistem gelecekte akıllı şehir uygulamalarında kullanılabilecek potansiyele sahiptir.

10. KAYNAKÇA
1.	Digital Image Processing — Rafael C. Gonzalez ve Richard E. Woods, Pearson Education, 2018.
2.	Computer Vision: Algorithms and Applications — Richard Szeliski, Springer, 2010.
3.	OpenCV kütüphanesi resmi dokümantasyonu, Open Source Computer Vision Library, https://opencv.org
4.	PyTorch resmi dokümantasyonu, https://pytorch.org
5.	TensorFlow resmi dokümantasyonu, https://www.tensorflow.org
6.	You Only Look Once: Unified, Real-Time Object Detection — Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.
7.	COCO Dataset — Microsoft, https://cocodataset.org
8.	ImageNet Classification with Deep Convolutional Neural Networks — Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton, Advances in Neural Information Processing Systems, 2012.
9.	YOLO resmi proje sayfası, https://pjreddie.com/darknet/yolo/
