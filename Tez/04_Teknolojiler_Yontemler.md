3. KULLANILAN TEKNOLOJİLER VE YÖNTEMLER
================================================================

Bu bölümde, geliştirilen sistemin dayandığı programlama dili, kütüphaneler, derin öğrenme modelleri ve temel görüntü işleme yöntemleri tanıtılmaktadır. Her teknoloji, sistemdeki somut kullanım amacıyla birlikte ele alınmaktadır.


3.1 Programlama Dili: Python

Sistemin tamamı, bilimsel hesaplama ve yapay zekâ uygulamalarında yaygın olarak tercih edilen Python programlama dili [26] ile geliştirilmiştir. Python'un zengin kütüphane ekosistemi, derin öğrenme, görüntü işleme ve arayüz geliştirmenin tek bir teknoloji yığını altında bütünleştirilmesini sağlamıştır. Projede modüler bir yapı benimsenmiş; algılama, geometri, park mantığı, arayüz, sesli asistan ve değerlendirme bileşenleri ayrı paketler hâlinde düzenlenmiştir.


3.2 Derin Öğrenme Altyapısı: PyTorch

Derin öğrenme modellerinin yüklenmesi ve çıkarsaması için PyTorch kütüphanesi [27] kullanılmıştır. Hem nesne tespit modeli hem de monoküler derinlik ve sürülebilir alan modelleri PyTorch üzerinde çalışmaktadır. Sistem, donanımda CUDA destekli bir GPU bulunduğunda otomatik olarak GPU'yu kullanmakta, bunun mümkün olmadığı durumlarda CPU'ya düşmektedir. GPU mevcut olduğunda yarı kesinlik (FP16) ile çıkarım yapılarak hız artırılmaktadır.


3.3 Nesne Tespiti: YOLOv8

Araç tespiti, gerçek zamanlı nesne tespiti için geliştirilmiş YOLOv8 modeliyle [3] gerçekleştirilmektedir. Hesaplama yükünün düşük tutulması ve sınırlı bir GPU üzerinde dahi gerçek zamanlı çalışılabilmesi için modelin en hafif sürümü olan YOLOv8n tercih edilmiştir. Model, COCO veri seti [28] üzerinde önceden eğitilmiş ağırlıklarla kullanılmakta; bu veri setindeki otomobil, motosiklet, otobüs ve kamyon sınıflarına ait tespitler süzülerek yalnızca araçlar değerlendirilmektedir.

Sistemde YOLOv8'in kendi sınıf-içi tespit bastırması (NMS) yetersiz kaldığı bir durum ayrıca ele alınmıştır: aynı araç bazen birden çok sınıfta (örneğin hem otomobil hem kamyon) tespit edilerek çift sayılabilmektedir. Bu sorunu gidermek için, tespitler sınıf-bağımsız bir bastırma adımından geçirilmekte; yüksek güven skorlu kutu korunurken, onunla yüksek oranda örtüşen düşük güvenli kutular elenir. Ek olarak, görüntü alanına oranla çok küçük kalan "kamyon" tespitleri sezgisel bir kuralla yeniden otomobil olarak sınıflandırılarak yanlış sınıflandırmaların önüne geçilmektedir. Sistem, projeye özgü sentetik veriyle ince ayar yapılmış modellerin (0-tabanlı sınıf düzeniyle) yüklenmesini de desteklemektedir.


3.4 Görüntü İşleme: OpenCV

Görüntü işleme işlemlerinin tamamı OpenCV kütüphanesi [29] ile gerçekleştirilmektedir. Renk uzayı dönüşümleri, eşikleme, morfolojik işlemler, perspektif dönüşümü ve çizim işlemleri bu kütüphane üzerinden yürütülmektedir. Sistemde OpenCV'nin sağladığı birkaç temel yöntem öne çıkmaktadır.

Kenar tespiti için Canny algoritması [30] kullanılmaktadır; bu yöntem, park şeritlerinin kenarlarını belirginleştirmek amacıyla çizgi tespit aşamasında uygulanır. Park çizgilerinin doğrusal segmentlere dönüştürülmesinde olasılıksal Hough dönüşümü [6] kullanılmaktadır. Düşük ışık ve gece koşullarında görüntü kontrastını iyileştirmek için ise Kontrast Sınırlı Uyarlamalı Histogram Eşitleme (CLAHE) yöntemi [31] uygulanmaktadır; bu yöntem, görüntüyü yerel bölgelere ayırarak her bölgenin kontrastını ayrı ayrı dengeler ve böylece aşırı parlama oluşturmadan karanlık bölgeleri belirginleştirir.


3.5 Sayısal Hesaplama: NumPy

Tüm dizi ve matris işlemleri, vektörel ölçüm hesapları ve geometrik dönüşümler NumPy kütüphanesi [32] ile gerçekleştirilmektedir. Slot skorlama, ölçek kestirimi ve homografi hesapları gibi sayısal yoğun işlemler saf NumPy ile yazılarak dış bağımlılık en aza indirilmiş ve bu bileşenlerin bağımsız test edilebilirliği sağlanmıştır.


3.6 Görsel Arayüz: PyQt5

Kullanıcı arayüzü, Qt çatısının Python bağlayıcısı olan PyQt5 [33] ile geliştirilmiştir. Arayüz; canlı görüntü ve video oynatımı, denetim panelleri, kuş bakışı ve şematik harita görünümleri ile sesli asistan denetimini tek bir masaüstü uygulamasında bir araya getirmektedir. Ağır görüntü işleme adımları, arayüzün yanıt vermesini engellememesi için ayrı iş parçacıklarında yürütülmektedir.


3.7 Monoküler Derinlik Kestirimi: MiDaS

Tek kameradan derinlik bilgisi elde etmek için MiDaS monoküler derinlik kestirim modeli [34] kullanılmaktadır. Stereo kamera veya LiDAR gerektirmeden, her piksel için göreli bir derinlik haritası üretilmektedir. Üretilen derinlik haritası, zamansal bir üstel hareketli ortalama ile 0–1 aralığına normalize edilerek kareler arası kararlılık sağlanır. Derinlik bilgisi sistemde iki amaçla kullanılır: aynı yatay hizada görünmesine rağmen farklı uzaklıkta bulunan nesnelerin ayırt edilmesi ve ters perspektif dönüşümünün çapraz açı belirsizliğini azaltacak biçimde tamamlanması. Model, ağır olduğundan ve indirme gerektirdiğinden isteğe bağlıdır; yüklenemediğinde sistem derinlik bilgisi olmadan zarif biçimde çalışmaya devam eder.


3.8 Sesli Etkileşim: Vosk ve Sinirsel Metinden Sese Dönüşüm

Sistem, sürücüyle çift yönlü sesli etkileşim kurabilmektedir. Sürücünün sesli komutları, çevrimdışı çalışan Vosk konuşma tanıma araç takımı [35] ile metne dönüştürülmekte; internet bağlantısı gerektirmeden, anahtar kelime eşleştirmesiyle ilgili komuta haritalanmaktadır. Sistemin sesli yanıtları ise, doğal ve anlaşılır bir Türkçe ses üreten sinirsel metinden sese dönüşüm hizmeti (edge-tts) [36] ile seslendirilmekte; çevrimdışı durumlar için işletim sistemi tabanlı bir yedek seslendirme motoruna düşülmektedir. Bu sayede sürücü, gözünü yoldan ayırmadan sistemle etkileşebilmektedir.


3.9 Sürülebilir Alan Segmentasyonu: YOLOPv2

Boş yer analizinin yalnızca gerçek yol yüzeyiyle sınırlı kalması ve kaldırım, refüj veya bina gibi yol-dışı alanların dışlanması için, sürücü-kamera alanında eğitilmiş YOLOPv2 panoptik sürüş algısı modeli [37] kullanılmaktadır. Model, yol yüzeyini ve şerit çizgilerini segment ederek güvenilir bir yol maskesi üretir. Hesaplama maliyeti nedeniyle bu maske seyrek aralıklarla güncellenip önbelleğe alınır; sahne yavaş değiştiğinden bu yaklaşım gerçek zamanlı çalışmayı korumaktadır. Model bulunmadığında sistem, klasik renk uzayı tabanlı bir yol maskesine düşmektedir.


3.10 Ters Perspektif Dönüşümü ve Homografi

Çapraz açılı kamera görüntülerinde, aynı hizadaki farklı derinlikteki nesneler üst üste binmekte ve iki boyutlu ölçüm yalnızca yaklaşık kalmaktadır. Bu sorunu gidermek için zemin düzlemine ait bir homografi [38] kullanılarak görüntü kuş bakışı (bird's eye view) görünümüne dönüştürülmektedir. Kuş bakışı görünümde mesafeler doğrusal hâle geldiğinden, kalibrasyon sırasında verilen gerçek-dünya boyutları yardımıyla metre/piksel ölçeği sabitlenmekte ve park yerlerinin gerçek metrik boyutları ölçülebilmektedir. Homografi, ya kullanıcının zemindeki bir dikdörtgenin dört köşesini işaretlediği manuel kalibrasyonla ya da derinlik yönünde yakınsayan çizgilerin kaybolma noktasının kestirildiği otomatik kalibrasyonla kurulmaktadır.


3.11 Değerlendirme ve Test Yığını

Sistemin başarımını ölçmek için ayrı bir değerlendirme altyapısı geliştirilmiştir. Kesinlik (Precision), duyarlılık (Recall), F1 skoru ve ortalama isabet (AP) gibi metrikler hesaplanmakta; sonuçların görselleştirilmesinde Matplotlib kütüphanesi [39] kullanılmaktadır. Değerlendirme altyapısı ayrıca, algoritmanın geometrik doğruluğunu nesne tespitinden bağımsız ölçmeye yarayan bir sentetik senaryo üreteci ile bileşenlerin katkısını tek tek ölçen bir ablasyon çerçevesi içermektedir. Yazılımın doğruluğu ve kararlılığı, pytest çatısı [40] ile yazılmış kapsamlı bir otomatik test kümesiyle sürekli olarak doğrulanmaktadır.
