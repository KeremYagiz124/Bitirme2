4. SİSTEM TASARIMI VE MİMARİSİ
================================================================

Bu bölümde, geliştirilen sistemin uçtan uca işlem hattı ve bu hattı oluşturan bileşenler ayrıntılı olarak ele alınmaktadır. Sistem, ardışık ve birbirini besleyen modüllerden oluşan bir boru hattı (pipeline) olarak tasarlanmıştır; her modül belirli bir alt görevden sorumludur ve bir sonraki modüle girdi sağlar.


4.1 Genel İşlem Hattı

Sistemin işlem hattı, kamera veya video girdisinin alınmasıyla başlar ve sürücüye sunulan yönlendirme ile sona erer. Temel akış şu adımlardan oluşur:

  Kamera / Video Girdisi
     ↓
  Gece Görüşü İyileştirme (düşük ışıkta kontrast düzeltme)
     ↓
  YOLOv8 Araç Tespiti  →  Sınıf-bağımsız bastırma (çift tespit eleme)
     ↓
  Araç Takibi (ego-hareket telafisi, statik/hareketli ayrımı)
     ↓
  Adaptif Boş-Yer Tespiti
     ├─ Çizgi varsa → Izgara tabanlı (Hough + füzyon + zamansal oylama)
     └─ Çizgi yoksa → Geometri tabanlı (boşluk analizi)
     ↓
  Ters Perspektif Dönüşümü (+ otomatik kalibrasyon)  →  Metrik ölçüm
     ↓
  Sığma Kontrolü  +  Çok-Kriterli Slot Öneri Motoru
     ↓
  Arayüz: Canlı Görüntü + 2B Şematik Harita + Yönlendirme + Sesli Asistan

Bu mimarinin önemli bir tasarım ilkesi, ağır ve isteğe bağlı bileşenlerin (derinlik, sürülebilir alan, sesli asistan gibi) bulunmadığı durumlarda sistemin çökmeden, daha temel bir davranışa geçerek çalışmaya devam etmesidir. Bu zarif bozunma (graceful degradation) ilkesi, sistemin farklı donanım ve koşullarda kararlı çalışmasını sağlamaktadır.


4.2 Gece Görüşü İyileştirme

İşlem hattının ilk adımı, düşük ışık koşullarında görüntü kalitesini artırmaktır. Bunun için görüntü, parlaklık (L) ve renk (a, b) bileşenlerinin ayrıştığı LAB renk uzayına dönüştürülmekte ve yalnızca parlaklık bileşenine Kontrast Sınırlı Uyarlamalı Histogram Eşitleme (CLAHE) [31] uygulanmaktadır. Bu yöntem, görüntüyü yerel bölgelere ayırarak her bölgenin kontrastını ayrı dengelediğinden, aşırı parlama oluşturmadan karanlık bölgeleri belirginleştirir ve sonraki tespit adımlarının başarımını artırır.


4.3 Araç Tespiti ve Sınıf-Bağımsız Bastırma

İyileştirilen görüntü, araç tespiti için YOLOv8 modeline [3] verilmektedir. Model yalnızca otomobil, motosiklet, otobüs ve kamyon sınıflarındaki nesneleri döndürecek biçimde süzülür. Her tespit için sınırlayıcı kutu, güven skoru ve sınıf bilgisi üretilir. YOLOv8'in kendi bastırma adımı sınıf-içi çalıştığından, aynı araç bazen birden çok sınıfta tespit edilip çift sayılabilmektedir; bu durumu gidermek için tespitler, yüksek güvenli kutunun korunup onunla yüksek oranda örtüşen düşük güvenli kutuların elendiği sınıf-bağımsız bir bastırma adımından geçirilir. Böylece her fiziksel araç tek bir tespitle temsil edilir.


4.4 Araç Takibi

Park alanından yalnızca geçen araçların yanlışlıkla park sırasına dahil edilmemesi için, tespit edilen araçlar kareler boyunca takip edilmektedir. Takip, ardışık karelerdeki kutuların örtüşme (IoU) oranına göre eşleştirilmesiyle yapılır; her araç için merkez konum geçmişi tutulur.

İki sınırlayıcı kutu A ve B için örtüşme oranı (IoU) şu biçimde tanımlanır:

    IoU(A, B) = Alan(A ∩ B) / Alan(A ∪ B)                              (4.1)

Burada pay iki kutunun kesişiminin, payda ise birleşiminin alanıdır; IoU değeri 0 (örtüşme yok) ile 1 (tam çakışma) arasında değişir.

Sürücü kamerası gibi hareketli kameralarda, kameranın kendi hareketi araçların sahte yer değiştirmesi olarak görünür ve park etmiş araçların hareketli sanılmasına yol açabilir. Bu sorunu çözmek için ego-hareket telafisi uygulanmaktadır: kameranın global hareketi, araç kutularının dışında kalan bölgelerden seçilen köşe noktalarının seyrek optik akış [41] ile takip edilmesiyle kestirilir. Köşe noktaları Shi-Tomasi yöntemiyle [42] belirlenir ve araç bölgeleri maskelenerek yalnızca arka planın hareketi ölçülür; tüm vektörlerin medyanı alınarak kameranın yer değiştirmesi elde edilir. Bu yer değiştirme, araç geçmişlerinden çıkarılarak araçların gerçek hareketi yalıtılır.

Bir aracın statik (park etmiş) sayılması için, son karelerdeki konum geçmişinin yer değiştirmesi, araç boyutuna oranlı bir eşiğin altında kalmalıdır; bu hesapta uç değerler budanarak tespit gürültüsüne karşı dayanıklılık sağlanır. Yalnızca yeterince uzun süredir görülen ve statik olan araçlar park sırası oluşturur. Ayrıca her aracın park süresi izlenmekte ve YOLO'nun anlık olarak kaçırdığı araçlar, takip tarafından kısa süre canlı tutularak tespit kararlılığı artırılmaktadır.


4.5 Adaptif Boş-Yer Tespiti

Sistemin özgün yanlarından biri, boş yer tespitini sahnenin özelliklerine göre uyarlamasıdır. Her karede sahnede yeterli ve güvenilir boyalı şerit bulunup bulunmadığına bakılır ve buna göre iki yöntemden biri seçilir. Anlık geçişlerin yol açacağı titremeyi önlemek için, mod seçimi bir geçmiş penceresi üzerinden histerezisle yumuşatılır: çizgi modu yalnızca yeterli karede tutarlı biçimde şerit görüldüğünde devreye girer, şerit güveni düştüğünde geometri moduna geri dönülür.

4.5.1 Çizgi Tabanlı Izgara Yöntemi

Boyalı şeritlerin bulunduğu otoparklarda, zemindeki şeritlerden park ızgarası çıkarılır. Şeritler önce HSV renk uzayında beyaz ve sarı boya maskeleriyle vurgulanır; bu maske, asfalt dokusunun ürettiği yanıltıcı kenarları eleyerek gerçek boya işaretlerini hedefler. Ardından Canny [30] kenarları ile birleştirilen maske üzerinde olasılıksal Hough dönüşümü [6] uygulanarak çizgi segmentleri bulunur. Segmentler dikey ve yatay olarak ayrılır, konumları bir boyutlu kümelemeyle birleştirilir ve yoğunluk profilinin ağırlık merkezine göre alt-piksel hassasiyetle iyileştirilir. Çizgi konumları kareler arasında zamansal füzyonla kararlı hâle getirilir; elde edilen ızgara hücrelerinin her biri bir slot oluşturur. Slotların doluluğu, araç kutularıyla örtüşme oranına göre belirlenir ve sonuç zamansal oylama ile yumuşatılır: her slot kareler arasında eşleştirilip son karelerin oy çoğunluğuna göre boş/dolu durumu kararlı hâle getirilir. Bu yöntem, perspektifin kalktığı kuş bakışı görünümde en yüksek doğruluğu verir.

4.5.2 Geometri Tabanlı Boşluk Analizi

Boyalı şeridin bulunmadığı yol kenarı ve düzensiz park alanlarında, boş yerler araçların geometrik dizilişinden çıkarılır. Statik araçlar, alt kenarlarının düşey konumuna göre park sıralarına kümelenir; aynı düşey hizada olup aralarında yol genişliği kadar boşluk bulunan sıralar ayrı şeritlere bölünür. Sistem, park düzenini paralel ve dik olarak ayırt eder; dik park durumunda araçların yandan mı yoksa önden mi görüldüğü, kutu en-boy oranlarının medyanına göre belirlenir. Komşu araçlar arasındaki ve sıra uçlarındaki boşluklar değerlendirilerek bir araç sığacak genişlikteki aralıklar boş yer olarak işaretlenir. Perspektif bozulmasını telafi etmek için araç genişlik ve yükseklikleri, yatay konuma göre doğrusal bir model ile kestirilir; böylece sahnenin farklı bölgelerindeki ölçek değişimi hesaba katılır. Aday boş yerler ayrıca yol yüzeyi denetiminden geçirilir: sürülebilir alan maskesi (veya klasik renk tabanlı yol maskesi) dışındaki, çim veya kaldırım gibi bölgelere düşen adaylar elenir; engel denetimi ile araç-dışı engellerin bulunduğu alanlar da reddedilir.


4.6 Ters Perspektif Dönüşümü ve Otomatik Kalibrasyon

Çapraz açılı görüntülerde metrik ölçümün doğru yapılabilmesi için zemin düzlemi bir homografi [38] ile kuş bakışı görünüme dönüştürülür. Homografi iki yolla kurulabilir. Manuel kalibrasyonda kullanıcı, zemindeki bir dikdörtgenin (örneğin bir park yeri sınırının) dört köşesini işaretler ve dikdörtgenin gerçek-dünya boyutlarını girer. Otomatik kalibrasyonda ise, derinlik yönünde uzanıp kuş bakışında paralelleşmesi gereken çizgiler tespit edilir; bu çizgilerin en küçük kareler yöntemiyle bulunan kesişim noktası (kaybolma noktası) kullanılarak bir yamuk çıkarılır ve bu yamuk bir dikdörtgene eşlenerek homografi hesaplanır. Yeterli çizgi bulunamadığında sistem, araç tabanlı bir yedek kalibrasyona veya manuel kalibrasyona düşer.

Bir zemin noktasının (x, y) kuş bakışı düzlemindeki karşılığı (x′, y′), 3×3 homografi matrisi H ile homojen koordinatlarda hesaplanır:

    s · [x′  y′  1]ᵀ = H · [x  y  1]ᵀ                                  (4.2)

Burada H kalibrasyonla elde edilen homografi matrisi, s ise homojen koordinatları normalize eden ölçekleme katsayısıdır.


4.7 Gerçek Metrik Ölçüm ve Ölçek Kestirimi

Kuş bakışı görünümde perspektif bozulması ortadan kalktığından, metre/piksel ölçeği görüntünün her yerinde sabittir. Kalibrasyon sırasında verilen gerçek boyutlardan bu ölçek hesaplanır ve park yerlerinin gerçek metrik boyutları ölçülebilir. Ölçeğin kalibrasyon olmadan da kestirilebilmesi için sistem, sınıfı bilinen araçların (otomobil, otobüs vb.) bilinen gerçek boyutlarını referans alarak görüntüdeki piksel boyutlarından ölçek çıkarımı yapar. Araç kutuları kuş bakışı düzlemine aktarılırken, yükseklik kaynaklı bozulmayı önlemek amacıyla yalnızca zeminle temas eden alt köşeler projekte edilir; aracın derinlik boyutu ise metrik araç uzunluğu üzerinden kurgulanır.

Kalibrasyon dikdörtgeninin gerçek genişliği G ve yüksekliği Y (metre), bunların kuş bakışındaki piksel karşılıkları g ve y ise, sabit metre/piksel ölçeği şu şekilde hesaplanır:

    s_mp = ½ · ( G ⁄ g + Y ⁄ y )                                       (4.3)

Bu ölçek görüntünün her yerinde sabit olduğundan, kuş bakışındaki herhangi iki nokta arasındaki piksel uzaklığı s_mp ile çarpılarak gerçek mesafe (metre) elde edilir; aynı yolla park yerlerinin gerçek genişliği hesaplanır.


4.8 Sığma Kontrolü

Boş bir yerin yalnızca var olması değil, sürücünün aracının o yere sığması da önemlidir. Metrik ölçüm sayesinde her boş yerin gerçek genişliği hesaplanır ve sürücünün aracının boyutlarıyla karşılaştırılır. Boş yerin genişliği, araç genişliğine güvenli bir manevra payı eklenecek biçimde yeterliyse yer "sığar", aksi hâlde "sığmaz" olarak işaretlenir ve kullanıcıya metre cinsinden alan genişliğiyle birlikte sunulur.


4.9 Çok Kriterli Slot Öneri Motoru

Sistem, boş yerleri yalnızca listelemek yerine, sürücü için en uygun olanı çok kriterli bir skorla önerir. Her boş yer için dört ölçüt değerlendirilir: manevra zorluğu (yerin genişliğinin araç enine oranı), yakınlık (kamera/sürücü referansına piksel mesafe), genişlik payı (rahat sığma marjı) ve çıkışa yakınlık.

Manevra zorluğu skoru, boş yerin genişliği w ile referans araç genişliği w_ref (metre) arasındaki orana göre hesaplanır:

    Z = sınırla( 50 + 120 · (w − w_ref) ⁄ w_ref ,  0 ,  100 )          (4.4)

Yüksek Z değeri daha geniş ve kolay manevralı bir yeri gösterir. Dört ölçüt 0–1 aralığına normalize edilip ağırlıklı olarak toplanarak nihai slot skoru elde edilir:

    S = 100 · ( 0,40 · z₁ + 0,25 · z₂ + 0,20 · z₃ + 0,15 · z₄ )        (4.5)

Burada z₁ zorluk (Eşitlik 4.4'ten normalize edilmiş), z₂ yakınlık, z₃ genişlik payı ve z₄ çıkışa yakınlık bileşenidir. Üretilen 0–100 aralığındaki skorlar arasından en yüksek S değerine sahip yer sürücüye önerilir. Ayrıca önerinin gerekçesi ("kolay manevra", "geniş slot", "yakın mesafe", "çıkışa yakın" gibi) insan-okunur bir metinle açıklanır. Bu bileşen, salt doluluk bilgisinin ötesine geçerek sisteme bir karar destek niteliği kazandırır.


4.10 Öğrenilen Slot Belleği ve Şematik Park Haritası

Sistem, gerçek park yerlerinin konumlarını zamanla öğrenen bir bellek bileşeni içerir. Bir araç yeterince uzun süre aynı konumda statik kaldığında, kapladığı alan kalıcı bir park yeri olarak kaydedilir; sonraki karelerde bu yerin durumu, mevcut araç kutularıyla örtüşme karşılaştırması yapılarak güncellenir. Böylece geometrik kestirimle elde edilen ilk tahmin, gözleme dayalı olarak zamanla kalibre edilir. Park alanının genel durumu ayrıca iki boyutlu bir şematik harita (dijital ikiz) üzerinde gösterilir; bu harita, boş ve dolu yerleri kuşbakışı bir düzende özetleyerek sürücüye sahnenin bütününü sunar.


4.11 Monoküler Derinlik Entegrasyonu

Çapraz açıda aynı yatay hizada görünmesine rağmen farklı uzaklıkta bulunan nesnelerin ayrıştırılabilmesi için sisteme isteğe bağlı bir monoküler derinlik kestirimi [34] bileşeni eklenmiştir. Etkinleştirildiğinde, üretilen göreli derinlik haritası yardımıyla iki bölgenin aynı derinlik düzleminde olup olmadığı denetlenebilir ve bu bilgi, ters perspektif dönüşümünü tamamlayarak boşluk analizinin güvenilirliğini destekleyebilir. Bu bileşen ağır bir model gerektirdiğinden varsayılan olarak devre dışıdır ve yalnızca istendiğinde çalıştırılır; bulunmadığı durumda sistem derinlik bilgisi olmadan çalışmaya devam eder.


4.12 Sesli Asistan

Sistem, sürücüyle çift yönlü sesli etkileşim kurar. Sürücünün sesli komutları çevrimdışı çalışan Vosk konuşma tanıma araç takımıyla [35] metne dönüştürülür ve anahtar kelime eşleştirmesiyle yalnızca uygulamada gerçekten karşılığı bulunan eylemlere ("boş yer bul", "kuş bakışı", "gece görüşü", "değerlendir" gibi) haritalanır. Komutun yürütülmesinin ardından sistem, doğal bir Türkçe sesle seslendirilen sinirsel metinden sese dönüşüm [36] ile sözlü bir geri bildirim verir; çevrimdışı durumlar için işletim sistemi tabanlı bir yedek seslendirme motoruna düşülür. Bu sayede sürücü, gözünü yoldan ayırmadan sistemle etkileşebilir.


4.13 Tasarım Kararları ve Gerekçeleri

Sistemin mimarisi birkaç temel ilke üzerine kurulmuştur. Birincisi, zarif bozunmadır: derinlik, sürülebilir alan ve sesli asistan gibi ağır veya isteğe bağlı bileşenler bulunmadığında sistem hata vermek yerine daha temel bir davranışa geçer. İkincisi, çekirdek karar bileşenlerinin (slot skorlama, ölçek kestirimi, geometri hesapları) dış bağımlılık olmadan saf sayısal kodla yazılmasıdır; bu, bileşenlerin bağımsız ve güvenilir biçimde test edilebilmesini sağlar. Üçüncüsü, gerçek zamanlı çalışmanın korunmasıdır: ağır modeller seyrek aralıklarla çağrılıp önbelleğe alınır, görüntü işleme adımları arayüzü bloke etmemek için ayrı iş parçacıklarında yürütülür ve sahnenin yavaş değiştiği varsayımından yararlanılır.
