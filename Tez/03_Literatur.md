2. LİTERATÜR ARAŞTIRMASI
================================================================

Bu bölümde, kamera tabanlı park yeri tespiti alanında bugüne kadar yapılmış akademik çalışmalar, kullanılan yöntemler ve veri setleri ile piyasadaki ticari sistemler incelenmekte; ardından geliştirilen sistemin literatürdeki yeri ortaya konmaktadır. Çalışmaya yön veren temel yaklaşım, sıfırdan özgün bir yöntem aramak yerine alandaki köklü çözümleri iyi kavrayıp onların üzerine inşa etmektir.


2.1 Problemin Tanımı

Park yeri yönetimi problemi, literatürde genellikle üç alt göreve ayrılmaktadır: park yeri konumunun tespiti (görüntüde hangi bölgelerin park yeri olduğu), doluluk sınıflandırması (her park yerinin dolu mu boş mu olduğu) ve araç sayımı. Bilgisayarlı görü tabanlı park yönetimi üzerine yapılan sistematik bir derleme, bu alt görevlerin ve kullanılan kamu veri setlerinin alanın çekirdeğini oluşturduğunu; dinamik fiyatlandırma, sürücü yönlendirme ve otomatik denetim gibi uygulamaların bu temel üzerine kurulduğunu ortaya koymaktadır [4]. Bu tezde geliştirilen sistem, söz konusu alt görevlerin tamamını ele almakta; araçları tespit etmekte, park yerlerini belirlemekte ve doluluk ile uygunluk kararını üretmektedir.


2.2 Derin Öğrenme Öncesi Yöntemler

Derin öğrenme yaygınlaşmadan önce park doluluğu tespiti, büyük ölçüde el yapımı özniteliklere ve klasik görüntü işleme tekniklerine dayanıyordu. Öne çıkan üç temel yaklaşım şunlardır:

Arka plan çıkarma (background subtraction) yönteminde, sabit kameranın gördüğü "boş park yeri" görüntüsü referans alınır; sonraki karelerde bu referanstan farklılaşan piksel bölgeleri araç olarak kabul edilir. Yöntem kavramsal olarak basit olsa da ışık değişimlerine, gölgelere ve hava koşullarına karşı son derece hassastır.

İkinci yaklaşımda, görüntüden Yönlü Gradyan Histogramları (HOG) gibi el yapımı öznitelikler çıkarılır ve bunlar Destek Vektör Makinesi (SVM) gibi bir sınıflandırıcıya verilir [5]. HOG el yapımı bir öznitelik çıkarıcı olduğundan, farklı açı ve ışık koşullarında tutarsız davranma eğilimindedir.

Üçüncü yaklaşımda, park çizgileri Hough dönüşümü ile tespit edilerek her bölgenin doluluğu denetlenir [6]. Hough dönüşümü kontrollü ortamlarda iyi çalışsa da çizgiler aşınmış, kirli veya gölgeli olduğunda başarısız olmaktadır. İlginç bir biçimde, Hough dönüşümü gibi klasik teknikler güncel sistemlerde tamamen terk edilmemiş; bu tezde olduğu gibi derin öğrenme bileşenleriyle birlikte, çizgi tespiti amacıyla hâlâ etkin biçimde kullanılmaktadır.

Bu yöntemlerin ortak sınırı, kontrollü koşullarda makul sonuçlar verirken gerçek dünya senaryolarına geçildiğinde performanslarının belirgin biçimde düşmesidir.


2.3 Derin Öğrenme ile Gelen İki Ana Paradigma

Derin öğrenmenin bu alana girmesiyle birbirinden ayrılan iki temel paradigma ortaya çıkmıştır; bu iki yaklaşım arasındaki seçim, sistemin tüm mimarisini belirlemektedir.

Birinci paradigma, bölge sınıflandırmadır (patch-based classification). Bu yaklaşımda her park yerine karşılık gelen küçük bir görüntü kırpılır ve bir evrişimli sinir ağı ile "dolu" ya da "boş" olarak sınıflandırılır. Yöntem yüksek doğruluk verse de, park yeri konumlarının önceden bilinmesini gerektirir; kamera açısı değiştiğinde veya kamera söküldüğünde tüm park yeri koordinatlarının yeniden etiketlenmesi gerekir.

İkinci paradigma, nesne tespiti ve örtüşme analizidir. Bu yaklaşımda bir nesne dedektörü tüm görüntüyü işler; tespit edilen araçların konumları ile önceden tanımlı park yeri sınırları arasındaki örtüşme (IoU) hesaplanarak doluluk kararı verilir. Bu yöntem daha esnektir, çünkü park yeri konumları bir kez tanımlanır ve sonraki karelerdeki araç tespitleri otomatik olarak karşılaştırılır. Bu tezde geliştirilen sistem temel olarak ikinci paradigmaya dayanmakla birlikte, sabit poligon tanımına bağımlı kalmak yerine boş yerleri araçların geometrik dizilişinden ve park şeritlerinden adaptif biçimde çıkararak bu yaklaşımı genişletmektedir.


2.4 Literatürdeki Önemli Çalışmalar

2.4.1 PKLot Veri Seti ve Erken CNN Uygulamaları (2015)

PKLot veri setinin yayımlanması, alanda bir dönüm noktası olmuştur [7]. Bu çalışmada her park yeri küçük bir görüntü olarak kırpılmış ve bir evrişimli sinir ağıyla dolu/boş olarak sınıflandırılmıştır. Güneşli havada %99'a yakın, yağmurlu havada %95 dolayında doğruluk elde edilmiş; böylece derin öğrenmenin klasik yöntemlere üstünlüğü açıkça gösterilmiştir. PKLot, alandaki standart karşılaştırma veri seti olma özelliğini hâlâ korumaktadır.

2.4.2 mAlexNet ve Akıllı Kamerada Tespit (Amato vd., 2017)

Amato ve arkadaşları, akıllı kameraların kendi üzerinde çalışabilecek kadar hafif, AlexNet'e kıyasla çok daha küçük bir evrişimli ağ olan mAlexNet'i tasarlamış ve CNRPark-EXT veri setini yayımlamışlardır [8]. Çalışma, kamera üzerinde dağıtık işleme (edge computing) yaklaşımının bu alandaki öncülerinden biri olmuştur.

2.4.3 CarNet — Genişletilmiş Evrişim (Nurullayev ve Lee, 2019)

Standart evrişimli ağların küçük nesneleri kaçırma eğilimini gidermek amacıyla Nurullayev ve Lee, genişletilmiş evrişim (dilated convolution) tabanlı CarNet mimarisini önermiştir [9]. Her park yeri için 54×32 piksellik girdi kullanan model, hem PKLot hem de CNRPark-EXT üzerinde önceki yöntemleri geride bırakarak %97'nin üzerinde ortalama doğruluğa ulaşmıştır.

2.4.4 Geliştirilmiş MobileNetV3 ve Dikkat Mekanizması (Yuldashev vd., 2023)

Yuldashev ve arkadaşları, hem hafif hem de yüksek doğruluklu bir model elde etmek için MobileNetV3 mimarisini dikkat mekanizmaları (CBAM) ve verimli evrişim bloklarıyla geliştirmiştir [10]. PKLot alt kümelerinde %99'un üzerinde doğruluk bildiren çalışma, 2023 itibarıyla alandaki en yüksek sınıflandırma doğruluklarından birini ortaya koymuştur.

2.4.5 APSD-OC — Otomatik Park Yeri Tespiti (Grbić ve Koch, 2023)

Park yeri koordinatlarının manuel etiketlenmesi sorununa sistematik bir çözüm getiren APSD-OC, iki aşamalı bir yapı önermiştir [11]. İlk aşamada YOLOv5 ile tespit edilen araçların merkezleri bir homografi matrisi aracılığıyla kuş bakışı görünüme dönüştürülerek kümelenmekte ve park yeri sınırları otomatik belirlenmektedir; ikinci aşamada her park yeri ResNet34 tabanlı bir sınıflandırıcıya verilmektedir. Bu çalışma, homografi tabanlı kuş bakışı dönüşümünün park yeri analizinde kullanımına dair önemli bir referanstır.

2.4.6 PakLoc ve PakSta — İş Yükünün Azaltılması (Nguyen ve Sartipi, 2024)

Nguyen ve Sartipi, park yerlerini otomatik konumlandıran (PakLoc) ve doluluk durumunu belirleyen (PakSta) iki bileşenli bir sistem önermiş; PKLot üzerinde yaptıkları çalışmada insan iş yükünü yaklaşık %94 oranında azalttıklarını göstermişlerdir [12]. Bu çalışma, otomasyonun sağladığı iş yükü azalmasını sayısal olarak ortaya koyması bakımından öne çıkmaktadır.

2.4.7 YOLO Sürümlerinin Karşılaştırılması ve Piksel Tabanlı ROI (da Luz vd., 2024)

da Luz ve arkadaşları, kendi oluşturdukları veri seti üzerinde YOLOv8, YOLOv9, YOLOv10 ve YOLOv11 sürümlerini karşılaştırmış ve tespit sonrası her park yerinin piksel düzeyinde doluluk oranını hesaplayan piksel tabanlı bir ilgi bölgesi (ROI) tekniği önermiştir [13]. Bu çalışma, YOLO ile araç tespiti ve bölgesel doluluk analizini birleştirmesi bakımından bu teze en yakın referanslardan biridir.

2.4.8 Gerçek Zamanlı Yasadışı Park Tespiti (Xie vd., 2017)

Xie ve arkadaşları, SSD tabanlı bir dedektörü araç tespiti için optimize ederek gerçek zamanlı bir yasadışı park tespit sistemi geliştirmiş; %99 doğruluk ve 25 FPS hız bildirmiştir [14]. Çalışma, tespit edilen araçların konum takibi ile ihlal süresinin hesaplanabileceğini göstermesi açısından önemlidir.


2.5 Geçici Araç ve Park Edilmiş Araç Ayrımı

Park alanından yalnızca geçen bir araç da nesne dedektörü tarafından tespit edilebilir ve sistemi yanlış "dolu" kararına yönlendirebilir. Bu pratik sorunun çözümü, ardışık karelerde aynı aracın takip edilmesinden geçer: park edilmiş araç ardışık karelerde aynı konumda kalırken, geçen araç birkaç kare sonra sahneyi terk eder.

Bu amaçla geliştirilen başlıca takip algoritmaları şunlardır. SORT, Kalman filtresi ve Macar algoritmasını birleştiren hızlı bir yöntemdir; ancak örtüşme durumlarında araç kimliğini kaybedebilir [15]. DeepSORT, SORT'a derin öğrenme tabanlı görsel benzerlik özniteliği ekleyerek örtüşen araçlarda kimlik tutarlılığını artırır [16]. ByteTrack ise düşük güven skorlu tespitleri de ikincil bir eşleştirme adımında değerlendirerek kimlik kayıplarını önemli ölçüde azaltır [17]. Bu algoritmaların park ihlali takibinde kullanımına örnek olarak, YOLOv8 ile takip algoritmalarını birleştirerek park süresi ihlallerini saptayan bir çalışma gösterilebilir [18]. Bu tezde geliştirilen sistemde de, araçların ardışık karelerde takip edilmesi ve hareketli araçların park sırasından ayrıştırılması için ego-hareket telafili bir takip bileşeni kullanılmaktadır.


2.6 YOLO Sürümleri ve Araç Tespiti

YOLO (You Only Look Once) ailesi, nesne tespitini tek geçişli bir regresyon problemi olarak ele alarak gerçek zamanlı tespitin önünü açmıştır [19]. Sonraki sürümler, doğruluk ve hız dengesini sürekli iyileştirmiştir. Park tespiti bağlamında yapılan karşılaştırmalar, daha büyük modellerin (ör. YOLOv9e) COCO üzerinde daha yüksek ortalama isabet (mAP) sunduğunu, buna karşılık küçük modellerin (ör. YOLOv8n) çok daha düşük gecikme ile kaynak kısıtlı cihazlarda gerçek zamanlı çalışabildiğini göstermektedir [13]. Bu tezde, sınırlı bir GPU üzerinde gerçek zamanlı çalışabilmesi nedeniyle hafif YOLOv8n modeli [3] tercih edilmiştir.


2.7 Benchmark Veri Setleri

Alandaki çalışmaların büyük çoğunluğu, standart karşılaştırma veri setleri üzerinde değerlendirme yapmaktadır. PKLot, üç farklı park alanından çekilmiş yaklaşık 12.400 görüntü ve 695.900 etiketli park yeri örneği içerir; güneşli, bulutlu ve yağmurlu hava koşullarını kapsar [7]. CNRPark-EXT, dokuz farklı kamera açısından çekilmiş yaklaşık 150.000 etiketli görüntü ile farklı aydınlatma, gölge ve kısmi örtülme senaryolarını içerir ve modelin genelleme kapasitesini ölçmek için daha zorlu bir veri seti sunar [8]. PS2.0 ise surround-view kameralardan elde edilmiş 12.165 görüntü içerir; ancak yalnızca park yeri tespitini kapsar, doluluk bilgisi sunmaz [20].


2.8 Sensör Tabanlı ve Kamera Tabanlı Sistemler

Park doluluğu tespitinde sensör tabanlı ve kamera tabanlı olmak üzere iki temel yaklaşım yarışmaktadır. Manyetik, ultrasonik ve döngü bobini gibi sensörler park yeri başına yüksek doğruluk sunar; ancak her park yeri için ayrı cihaz gerektirdiklerinden kurulum ve bakım maliyetleri yüksektir, ölçeklenmeleri güçtür. Döngü bobinleri ayrıca yol yüzeyinin kazılmasını gerektirdiğinden son derece müdahalecidir.

Kamera tabanlı sistemler ise tek bir kamera ile birden fazla park yerini izleyebilir ve çoğu durumda mevcut güvenlik kamerası altyapısıyla çalışabilir. Bu sistemlerin hava ve ışık koşullarına karşı daha hassas olması, yazılımsal iyileştirmelerle kısmen telafi edilebilmektedir. Kamera tabanlı yaklaşımın gelecekteki en önemli üstünlüklerinden biri elektrikli araç uyumudur: hafif gövdeleri nedeniyle manyetik alanı yalnızca sınırlı ölçüde etkileyen elektrikli araçlar manyetik sensörlerin güvenilirliğini düşürürken, kamera tabanlı sistemler aracın türünden bağımsız olarak çalışmaya devam eder.


2.9 Piyasadaki Ticari Sistemler

Akademik çalışmaların yanı sıra, park yönetimi alanında olgunlaşmış ticari sistemler de bulunmaktadır. ETH Zürih çıkışlı Parquery, mevcut kameralara bir yapay zekâ katmanı ekleyerek ek altyapı gerektirmeden araç tespiti, doluluk analizi ve ihlal tespiti yapmakta; tek bir görüntüde yüzlerce park yerini izleyebilmektedir [2]. Quercus Technologies, zemine gömülü sensör kullanmadan çalışan "sanal döngü" teknolojisi ve plaka tanıma yetenekleriyle öne çıkmakta; 60'tan fazla ülkede on binin üzerinde kuruluma sahiptir [21]. Bosch, araç sensörlerinden toplanan veriyle oluşturulan topluluk tabanlı park haritası, otomatik vale park ve kamera üzerinde işleme yapan edge yapay zekâ kameraları olmak üzere birden çok yaklaşımı bir arada yürütmektedir [22]. Metropolis, bilgisayarlı görü tabanlı temassız ödeme sistemleriyle aracı plakasından tanıyıp otomatik ücretlendirme yapmaktadır [23]. Hikvision, çok sayıda kameranın görüntüsünü merkezi bir yapay zekâ modeliyle işleyerek park doluluk haritası çıkarmakta ve plaka tanıma ile bütünleşik çalışmaktadır [24]. Dahua ise WizMind yapay zekâ platformuyla araç türü, rengi ve plakasını eş zamanlı tespit edebilmektedir [25].

Bu sistemlerin tamamında ortak olan teknik akış, bu tezde geliştirilen sistemle büyük ölçüde örtüşmektedir: kamera görüntüsü, araç tespiti, bölge analizi ve doluluk kararı. Farklılıklar ölçek, donanım kalitesi ve plaka tanıma veya dinamik fiyatlandırma gibi ek katmanlarda ortaya çıkmaktadır.


2.10 Projenin Literatürdeki Yeri

Yapılan inceleme, geliştirilen sistemin literatürdeki konumunu net biçimde ortaya koymaktadır. Sistem, alandaki güncel yaklaşımlarla aynı temel akışı benimsemekle birlikte, birkaç yönüyle ayrışmaktadır.

Güçlü yönler:

• Güncel ve hafif bir nesne tespit modeli (YOLOv8n) [3] kullanılarak gerçek zamanlı çalışma hedeflenmiştir.

• Literatürdeki çalışmaların çoğu yalnızca dolu/boş ikili sınıflandırması yaparken, geliştirilen sistem boş yerleri araç dizilişinden ve park şeritlerinden adaptif biçimde çıkarmakta; ayrıca ters perspektif dönüşümü ile park yerlerinin gerçek metrik boyutlarını ölçerek aracın sığıp sığamayacağını denetlemektedir. APSD-OC [11] gibi çalışmalarda kullanılan homografi tabanlı kuş bakışı dönüşümü, bu sistemde metrik ölçüm ve sığma denetimi amacıyla genişletilerek kullanılmıştır.

• Sistem, geçici araçları park edilmiş araçlardan ayırmak için ego-hareket telafili bir araç takibi içermektedir; bu yönüyle ByteTrack [17] gibi takip odaklı çalışmaların ele aldığı sorunu pratik biçimde adreslemektedir.

• Boş yer tespitinin ötesine geçilerek, sürücüye en uygun yeri öneren çok kriterli bir karar mekanizması ve gece görüşü, monoküler derinlik kestirimi ve çevrimdışı sesli asistan gibi sürücü-asistanı özellikleri bütünleşik biçimde sunulmaktadır.

• Açık otopark alanlarından ziyade yol kenarı (on-street) park senaryolarına da odaklanılması, literatürde görece az çalışılmış bir alana katkı sağlamaktadır.

Geliştirmeye açık yönler:

• Sistem henüz PKLot veya CNRPark-EXT gibi standart benchmark veri setleri üzerinde geniş ölçekli olarak değerlendirilmemiştir.

• Boş/dolu kararı ağırlıklı olarak geometrik ve örtüşme tabanlı yöntemlere dayanmaktadır; kendi verisiyle eğitilen bir doluluk sınıflandırıcısı (CNN) henüz devreye alınmamıştır.

• Gece ve yağmur gibi zorlu koşullardaki başarım, gündüz koşullarına kıyasla sınırlıdır ve ileriye dönük çalışma olarak ele alınmaktadır.

Sonuç olarak geliştirilen sisteme en yakın akademik referans, YOLO ile araç tespiti ve bölgesel doluluk analizini birleştiren da Luz vd. (2024) [13] çalışmasıdır; ticari referans olarak ise mevcut kameraya yapay zekâ ekleyerek ek altyapı gerektirmeyen Parquery [2] benimsenmiştir. Otomatik park yeri tespiti için APSD-OC [11] ve araç takibi için ByteTrack [17], ileriye dönük geliştirme açısından yol gösterici çalışmalar olarak değerlendirilmektedir.
