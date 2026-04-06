# 2. Literatür Araştırması

Hocamızın yönlendirmesiyle bu alanda sıfırdan bir şey icat etmeye çalışmak yerine, literatürde neyin yapıldığını iyi anlayıp onun üzerine inşa etmeyi hedefledik. Bu bölümde park yeri tespit sistemleri konusunda şimdiye kadar yapılmış akademik çalışmalar, kullanılan veri setleri, izlenen yöntemler ve piyasadaki ticari ürünler detaylıca incelendi.

---

## 2.1 Problem Nasıl Tanımlanıyor?

Literatür bu problemi genellikle üç alt göreve bölüyor:

1. **Park yeri konumu tespiti** — Görüntüde hangi bölgeler park yeri? (Otomatik ya da manuel olarak sınırlar belirlenir.)
2. **Doluluk sınıflandırması** — Her park yeri dolu mu, boş mu? (İkili sınıflandırma)
3. **Araç sayımı** — Toplam kaç araç var? (Daha basit versiyon)

Bu üç görev, akıllı park sistemlerinin çekirdek bileşenlerini oluşturmaktadır. Dinamik fiyatlandırma, sürücü yönlendirme ve otomatik yönetim gibi uygulamalar bu temel üzerine inşa edilmektedir. Geliştirilen projede bu üç görevin tamamı ele alınmaktadır: araç tespiti yapılmakta, park alanı sınırları tanımlanmakta ve doluluk kararı verilmektedir.

---

## 2.2 Tarihsel Gelişim: Derin Öğrenmeden Öncesi

Derin öğrenme yaygınlaşmadan önce sistemler üç temel tekniğe dayanıyordu:

**a) Arka plan çıkarma (Background Subtraction):**
Kamera sabit olduğunda "boş park yeri" görüntüsü referans olarak alınır. Sonraki karelerde bu referanstan farklılaşan piksel bölgeleri araç olarak kabul edilir. Yöntemin mantığı basit ve anlaşılır, ama ışık değişimlerine son derece hassas. Sabah ile öğleden sonrasının farklı gün ışığı açısı bile sistemi yanıltabiliyor. Gölgeler, yansımalar ve hava koşulları bu yöntemi pratikte çok kırılgan yapıyor.

**b) HOG + SVM:**
Her görüntüden kenar ve yoğunluk bilgisi çıkarılıyor (HOG — Histogram of Oriented Gradients), ardından bu özellikler bir makine öğrenmesi sınıflandırıcısı olan SVM'e veriliyor. HOG el yapımı bir özellik çıkarıcı olduğu için farklı açı ve ışık koşullarında tutarsız davranıyor. Orta düzeyde doğruluk sunsa da büyük ölçekli sistemler için yavaş.

**c) Çizgi Tespiti (Hough Transform):**
Park çizgileri Hough dönüşümü algoritmasıyla tespit ediliyor, her bölgenin dolu/boş olduğu kontrol ediliyor. Kontrollü ortamlarda iyi çalışıyor ama çizgiler aşınmış, kirli, karlı ya da gölgeli olduğunda tamamen başarısız oluyor. Gerçek dünya park alanlarında bu koşullar son derece yaygın.

Bu üç yöntemin ortak sorunu: kontrollü, sabit koşullarda makul sonuçlar veriyorlar ama gerçek dünya senaryolarına geçildiğinde performansları ciddi ölçüde düşüyor.

---

## 2.3 Derin Öğrenme ile Gelen İki Ana Yaklaşım

Derin öğrenme bu alana girince iki farklı paradigma ortaya çıktı. İkisi arasındaki seçim, sistemin tüm mimarisini belirliyor.

### Yaklaşım A — Bölge Sınıflandırma (Patch-Based Classification)

Bu yaklaşımda her park yerine karşılık gelen küçük bir görüntü (patch) kırpılır ve bir CNN ile "dolu" veya "boş" olarak sınıflandırılır.

```
Tam park alanı görüntüsü
         ↓
  [Manuel/Otomatik bbox tespiti]
         ↓
  Her slot için patch kırpma
         ↓
  CNN → Dolu / Boş
```

Bu yaklaşım PKLot ve CNRPark gibi veri setleriyle çok iyi sonuçlar verdi. Ama kritik bir kısıtı var: kamera açısı değiştiğinde ya da kamera söküldüğünde tüm park yeri koordinatlarının yeniden etiketlenmesi gerekiyor. 200 park yerli 5 ayrı parkı yönetmek durumundaki bir operatör için bu, ciddi bir iş yükü demek.

**Bu yaklaşımın sınırları:**
- Her kamera değişikliğinde yeniden etiketleme
- Farklı açılara genelleme zor
- Eğitim için büyük miktarda etiketli veri gerekiyor
- Sadece dolu/boş sınıflandırması yapıyor; ihlal türü belirlenemiyor

### Yaklaşım B — Nesne Tespiti + IoU (Object Detection-Based)

Nesne dedektörü tüm görüntüyü işler; tespit edilen araçların konumları ile önceden bilinen park yeri sınırları IoU (Intersection over Union) hesaplanarak karşılaştırılır.

```
Kamera Görüntüsü
       ↓
[YOLOv8 ile Araç Tespiti]
       ↓
[Park Yeri Poligonlarıyla IoU Hesabı]
       ↓
[IoU > 0.5 → Dolu | IoU < 0.5 → Boş]
       ↓
[Doluluk Kararı + İhlal Tespiti]
```

Bu yaklaşım daha esnek çünkü park yeri konumları bir kez tanımlanıyor, sonraki her karedeki araç tespitleri otomatik karşılaştırılıyor. Ayrıca yasak bölge poligonu da tanımlanabiliyor; böylece ihlal tespiti de mümkün oluyor.

**Geliştirilen proje bu ikinci yaklaşımı uygulamaktadır:** YOLOv8 ile araç tespiti yapılmakta, park alanları fareyle tıklanarak poligon olarak tanımlanmakta, ardından IoU analizi ile doluluk kararı verilmektedir.

---

## 2.4 Literatürdeki Önemli Çalışmalar

### 2.4.1 AlexNet ile Park Doluluk Tespiti (2015)

**Problem:** PKLot veri setinin yayınlanmasıyla birlikte bu veri seti üzerinde AlexNet mimarisi uygulandı [5]. Çalışma, o dönem için oldukça güçlü sonuçlar verdi.

**Nasıl yaptılar:** Her park slotunu küçük bir görüntü olarak kestiler, AlexNet'e "dolu" veya "boş" olarak sınıflandırdılar. Güneşli havada %99'a yakın, yağmurlu havada %95 civarı doğruluk elde ettiler.

**Önemi:** Bu çalışma, derin öğrenmenin park tespitinde klasik yöntemlere kıyasla ne kadar üstün olduğunu ilk kez net biçimde gösterdi ve PKLot veri setini alandaki standart benchmark haline getirdi.

**Kısıtı:** AlexNet büyük bir model; akıllı kamerada çalıştırmak için çok ağır.

---

### 2.4.2 mAlexNet — Dağıtık Park Tespiti (Amato vd., 2017)

**Problem:** AlexNet iyi sonuçlar veriyordu ama her park kamerasında bir sunucu çalıştırmak mümkün değil. Daha hafif bir model gerekiyordu.

**Nasıl yaptılar:** Amato ve ekibi, AlexNet'e kıyasla üç kat daha küçük özel bir CNN tasarladı — buna mAlexNet dediler [3]. Aynı zamanda CNRPark veri setini yayınladılar. Model, kameranın kendi içinde çalışabilecek kadar hafif tasarlandı.

**Sonuçlar:** AUC (Alan Under Curve) değerleri 0.92-0.99 arasında ölçüldü. AlexNet'in yaklaşık yarısı büyüklüğünde bir modelle benzer doğruluk elde ettiler.

**Önemi:** Akıllı kamerada çalışan ilk ciddi park tespiti sistemi. "Edge computing" yaklaşımının bu alandaki öncüsü.

---

### 2.4.3 CarNet — Dilate CNN ile Park Tespiti (Nurullayev & Lee, 2019)

**Problem:** Standart CNN'ler küçük nesneleri kaçırma eğiliminde [4]. Park yerleri de sahnede görece küçük nesneler.

**Nasıl yaptılar:** Dilate edilmiş evrişimli sinir ağları (dilated convolution) kullandılar. Bu teknik, filtrenin "boşluklu" hareket etmesini sağlıyor; böylece daha büyük bir alanı daha az hesapla görüyor. Her park slotu için 54×32 piksellik RGB görüntü girişi kullandılar.

**Sonuçlar:** Hem PKLot hem CNRPark-EXT üzerinde AlexNet ve mAlexNet'i geride bıraktılar. Ortalama doğruluk %97'nin üzerine çıktı.

**Önemi:** Dilated convolution'ın park tespitindeki etkinliğini gösterdi; küçük nesne tespitinde önemli bir adım.

---

### 2.4.4 Geliştirilmiş MobileNetV3 + CBAM (Yuldashev vd., 2023)

**Problem:** Yüksek doğruluklu modeller ağır; düşük donanımlı sistemlerde çalışmıyor. Hem hafif hem doğru bir model gerekiyor [8].

**Nasıl yaptılar:** MobileNetV3 mimarisini üç modifikasyonla geliştirdiler:
1. ReLU6 aktivasyonu yerine Leaky-ReLU6 kullandılar — negatif değerleri tamamen kesmek yerine küçük bir sızıntı bıraktılar.
2. Squeeze-Excitation (SE) modülü yerine CBAM (Convolutional Block Attention Module) kullandılar — hem kanal hem de uzamsal dikkat mekanizması eklendi.
3. Standart depthwise separable convolution yerine Blueprint Separable Convolution (BSConv) kullandılar — daha az parametre, benzer performans.

**Sonuçlar:** PKLot alt setlerinde %99.68, CNRPark-EXT'de %97.69 doğruluk. Önceki en iyi model olan CarNet'in %97.03'ünü geçtiler. Bu, 2023 itibarıyla literatürdeki en yüksek sınıflandırma doğruluklarından biri.

**Önemi:** Hafif ama yüksek performanslı bir model isteyenler için güçlü bir referans. Dikkat mekanizmalarının park tespitindeki etkinliğini kanıtladı.

---

### 2.4.5 APSD-OC — Otomatik Park Yeri Tespiti (Grbić & Koch, 2023)

**Problem:** Tüm çalışmalarda park yeri koordinatları manuel olarak etiketleniyor [9]. Bu, büyük ölçekli sistemlerde ciddi iş yükü yaratıyor. Kamera değişince her şeyi yeniden yapmak gerekiyor.

**Nasıl yaptılar:** Tamamen otomatik bir iki aşamalı sistem geliştirdiler:
- **1. Aşama (Park Yeri Tespiti):** YOLOv5 tüm araçları tespit ediyor → her tespitinin merkezi bir homografi matrisi aracılığıyla kuş bakışı görünüme dönüştürülüyor → bu noktalar kümelenerek park yeri sınırları belirleniyor.
- **2. Aşama (Doluluk Sınıflandırması):** Belirlenen her park yeri ResNet34 tabanlı bir sınıflandırıcıya veriliyor → dolu/boş kararı üretiliyor.

**Sonuçlar:** PKLot ve CNRPark üzerinde yüksek doğruluk elde ettiler. Ama asıl önemli sonuç şu: insan iş yükü önemli ölçüde azaldı.

**Önemi:** Manuel etiketleme sorununa ilk sistematik çözüm. Geliştirilen sistemde park alanları hâlâ manuel olarak çizilmektedir. Bu çalışma, ileride otomatik hale getirmek için bir yol haritası niteliği taşımaktadır.

---

### 2.4.6 PakLoc + PakSta — İş Yükünü %94 Azaltmak (2024)

**Problem:** Manuel etiketleme problemi hâlâ çözülmüş sayılmıyor. APSD-OC otomasyonu getirdi ama pratik iş yükü azalması ne kadar?

**Nasıl yaptılar:** PakLoc (Park Locator) ve PakSta (Park Status) adında iki ayrı sistem geliştirdiler. PakLoc, park alanlarını otomatik tespit ediyor; PakSta ise doluluk durumunu belirliyor. PKLot veri seti üzerinde kapsamlı bir ablasyon çalışması yaptılar.

**Sonuçlar:** İnsan iş yükünü %94.25 oranında azalttıklarını kanıtladılar. Bu rakam, büyük ölçekli sistemler için son derece anlamlı.

**Önemi:** "Otomasyonun gerçekten ne kadar iş yükü azalttığını" sayısal olarak ortaya koyan ilk çalışma.

---

### 2.4.7 YOLOv8/v9/v10/v11 Karşılaştırması (da Luz vd., 2024)

**Problem:** YOLO'nun hangi versiyonu park tespiti için en uygun [7]? Ve pixel bazlı ROI seçimi performansı nasıl etkiliyor?

**Nasıl yaptılar:** 3.484 görüntüden oluşan kendi veri setlerini oluşturdular ve YOLOv8, YOLOv9, YOLOv10, YOLOv11 modellerini karşılaştırdılar. Bunun yanı sıra pixel bazlı ROI (Region of Interest) seçimi adlı bir post-processing tekniği önerdiler: tespit sonrası her park slotunun piksel düzeyinde doluluk oranı hesaplanıyor.

**Sonuçlar:** YOLOv9e, COCO veri setinde 55.6 mAP ile en yüksek tespit doğruluğunu verdi. Pixel bazlı ROI tekniği eklendikten sonra %99.68 balanced accuracy elde ettiler. Edge cihazlarda da test ettiler.

**Önemi:** Bu çalışma, geliştirilen projeye en yakın referans niteliğindedir. Aynı yaklaşım (YOLO + ROI analizi) temel alınmış; ilerleyen aşamalarda MobileNetV2 doğrulama katmanı eklenmesi planlanmaktadır.

---

### 2.4.8 Gerçek Zamanlı Yasadışı Park Tespiti (Xie vd., 2017)

**Problem:** Yasadışı park tespiti için mevcut sistemlerin büyük çoğunluğu yavaş çalışıyor ya da gerçek zamanlı değil [13].

**Nasıl yaptılar:** SSD (Single Shot MultiBox Detector) algoritmasını optimize ettiler. SSD'nin aspect ratio (en-boy oranı) parametrelerini araç tespit için ayarladılar. Tespit edilen araçlar için konum takibi yapıp ihlal süresini de hesapladılar.

**Sonuçlar:** %99 doğruluk, 25 FPS gerçek zamanlı çalışma.

**Önemi:** Yasadışı park tespiti alanında o dönem için en yüksek doğruluk elde edilmiştir. Geliştirilen projede de benzer şekilde yasak bölge altyapısı kurulmuştur; temel fark olarak SSD yerine daha güncel YOLOv8 modeli tercih edilmiştir.

---

### 2.4.9 YOLOv7 ile Gerçek Zamanlı Park İzleme (2025)

**Problem:** Araç tespiti güzel çalışıyor, peki bu bilgiyi nasıl yönetim sistemine entegre ederiz?

**Nasıl yaptılar:** YOLOv7'yi bir web arayüzüne bağladılar. Flask (Python web framework), OpenCV ve SQLite kullanarak gerçek zamanlı veri kaydı yapan bir sistem kurdu. Araç tespiti sonuçları veritabanına işleniyor, web panelinden görüntülenebiliyor.

**Sonuçlar:** Farklı çevre koşullarında ortalama %94.23 doğruluk. Sistem gerçek zamanlı çalışıyor.

**Önemi:** Sadece tespit değil, yönetim entegrasyonu da başarılı. Geliştirilen sistem şu aşamada masaüstü arayüzüne odaklanmaktadır; ilerleyen aşamada veritabanı entegrasyonu için bu çalışma referans alınabilir.

---

## 2.5 Önemli Bir Sorun: Geçici Araç mı, Park Edilmiş Araç mı?

Literatürün çoğu zaman atlayıp geçtiği ama pratikte son derece kritik bir problem var: park alanından sadece geçen bir araç da nesne dedektörü tarafından tespit edilebilir ve sistemi yanlış "dolu" kararına yönlendirebilir.

Bu problemi çözmek için **araç takibi (tracking)** kullanılıyor. Ardışık karelerde aynı araç takip ediliyor:
- Hareketsiz park edilmiş bir araç → ardışık karelerde aynı konumda, sürekli yüksek IoU
- Geçen bir araç → birkaç kare sonra sahneyi terk ediyor, düşük IoU tutarlılığı

Bu alanda öne çıkan algoritmalar:

**SORT:** Kalman filtresi + Macar algoritması. Hızlı ama örtüşme durumlarında araç kimliğini kaybediyor. Basit senaryolar için yeterli.

**DeepSORT:** SORT'a derin öğrenme tabanlı görsel benzerlik özelliği ekledi. Her araca görsel bir "imza" çıkarılıyor; hem konum hem de görünüş bilgisiyle eşleştirme yapılıyor. Örtüşen araçlarda daha güvenilir.

**ByteTrack (ECCV 2022):** Önemli bir yenilik getirdi: düşük güven skorlu tespitleri de işleme dahil ediyor [10]. Standart yaklaşımlar bu tespitleri atar; oysa örtülmüş bir araç çoğunlukla düşük güven skoru alır ama tamamen arka plan değildir. ByteTrack bu bilgiyi ikinci bir eşleştirme adımında kullanarak ID kayıplarını ciddi ölçüde azaltıyor.

**Önerilen kombinasyon:** Park ihlali tespiti için literatür **YOLOv8 + ByteTrack** veya **YOLOv8 + DeepSORT** kombinasyonlarını öneriyor. Bizim sistemimizde henüz tracking yok; bu ilerleyen haftalarda eklenmesi planlanan bir özellik.

---

## 2.6 YOLO Versiyonları Karşılaştırması

| Model | mAP (COCO) | Hız | Küçük Nesne | Park Tespitine Uygunluk |
|-------|------------|-----|-------------|------------------------|
| YOLOv8n | ~37.3 | En hızlı | Zayıf | Edge cihazlar, düşük GPU |
| YOLOv8m | ~50.2 | Orta | Orta | Standart GPU |
| YOLOv9e | 55.6 | Yavaş | İyi | Yüksek doğruluk gerektiğinde |
| YOLOv11n | ~39.5 | En hızlı | Orta+ | Edge + doğruluk dengesi |

Biz projemizde **YOLOv8n** kullanılmaktadır. Gerekçesi: GTX 1050 Ti Mobile gibi sınırlı bir GPU'da gerçek zamanlı çalışabilmesi için hız kritik. YOLOv8n bu modeller arasında en düşük hesaplama yüküne sahip ve gerçek zamanlı tespit için en uygun seçenek. Daha yüksek doğruluk gerekirse ilerleyen aşamada YOLOv8s veya YOLOv8m'e geçilebilir.

---

## 2.7 Benchmark Veri Setleri

Literatürdeki neredeyse tüm çalışmalar iki ana veri seti üzerinde test yapıyor. Bu veri setleri alandaki standart kıyaslama araçları haline geldi.

### PKLot (UFPR, Brezilya, 2015)

Üç farklı park alanından (PUCPR, UFPR04, UFPR05) çekilmiş 12.417 tam park alanı görüntüsü ve yaklaşık 695.900 etiketli park yeri örneği içeriyor. Güneşli, bulutlu ve yağmurlu olmak üzere üç farklı hava koşulunu kapsıyor. Her park yeri hem döndürülmüş dikdörtgen hem de kontur formatında XML olarak etiketlenmiş. Alandaki standart benchmark veri seti olma özelliğini hâlâ koruyor.

**Neden önemli:** Bu veri seti üzerinde çalışan model, standart test koşullarında diğer çalışmalarla doğrudan karşılaştırılabiliyor. Doğruluk iddialarının güvenilir olması için PKLot üzerinde test şart.

### CNRPark-EXT (CNR Pisa, İtalya, 2017)

9 farklı kamera açısından çekilmiş yaklaşık 150.000 etiketli patch görüntüsü içeriyor. 164 park slotunu farklı mevsimler ve koşullar altında kapsıyor. Gölgeler, kısmi örtülmeler, farklı mevsimler ve yılın farklı saatleri mevcut. Generalizasyon testi için daha zorlu bir veri seti.

**Neden önemli:** Farklı kameralar ve farklı koşullar modelin genelleme kapasitesini ölçüyor. Sadece bir veri setinde iyi sonuç vermek yeterli değil — CNRPark-EXT'deki performans modelin gerçek dünyada ne kadar tutarlı olduğunu gösteriyor.

### PS2.0 (Tongji Üniversitesi, Çin, 2018)

600×600 piksel çözünürlüğünde 12.165 surround-view görüntü içeriyor. İç mekan, açık hava gün ışığı, sokak lambası, gölge, yağmur ve eğimli zemin olmak üzere altı farklı test kategorisi var. Ama önemli bir eksik: sadece park yeri tespitini kapsıyor, doluluk bilgisi sunmuyor.

| Veri Seti | Görüntü | Park Slotu | Hava Koşulu | Gece | Doluluk |
|-----------|---------|-----------|-------------|------|---------|
| PKLot | 12.417 | 695.900 | Var | Yok | Var |
| CNRPark-EXT | ~150.000 patch | 164 | Var | Var | Var |
| PS2.0 | 12.165 | — | Var | Var | Yok |

---

## 2.8 Sensör Tabanlı vs Kamera Tabanlı Sistemler

Piyasada iki farklı yaklaşım yarışıyor ve literatür bu konuyu sıkça tartışıyor.

### Sensör Tabanlı Sistemler

**Manyetik sensörler:** Araç üzerinden geçtiğinde dünya manyetik alanında oluşan değişimi algılıyor. Her park slotu için ayrı sensör gerekiyor, yüksek doğruluk (%98-99). Ama zemine gömme, kablolama ve bakım masrafları yüksek.

**Ultrasonik sensörler:** Yüksek frekanslı ses dalgaları ile mesafe ölçüyor. Kurulumu daha kolay ama yağmur ve rüzgarda güvenilirlik düşüyor.

**Döngü bobinleri (Loop Coil):** Yola gömülen metal spiral bobin, araç metalini tespit ediyor. Çok yüksek doğruluk (%98-99) ama kurulum yol kazma gerektiriyor — en pahalı ve en invaziv yöntem.

### Kamera Tabanlı Sistemler

Tek kamera birden fazla park yerini izleyebiliyor. Ek altyapı gerektirmiyor, mevcut güvenlik kameralarıyla çalışabiliyor. Hava koşullarına karşı daha hassas ama yazılım iyileştirmeleriyle bu kısmen telafi edilebiliyor.

**Kritik avantaj — Elektrikli Araç Uyumu:**
Manyetik sensörlerin gelecekteki en büyük kısıtı elektrikli araçlar. Elektrikli araçlar çok hafif yapıları ve verimli gövde tasarımları nedeniyle manyetik alana çok az etki ediyor. Bu durum, yalnızca manyetometreye dayalı park sensörlerini EV'ler karşısında yetersiz kılıyor. Kamera tabanlı sistemler bu sorunu hiç yaşamıyor — araç elektrikli mi dizel mi olduğu fark etmiyor, görüntüde görülüyor.

| Kriter | Ultrasonik | Manyetik | Döngü Bobin | Kamera (CV) |
|--------|-----------|---------|-------------|-------------|
| Doğruluk | %95-98 | %90-97 | %98-99 | %92-99 |
| Kurulum | Kolay | Kolay | Zor (yol kazıma) | Orta |
| Bakım | Düşük | Düşük | Orta | Orta |
| Kapsam | 1 yer/sensör | 1 yer/sensör | 1-2 yer | Çoklu yer |
| Hava koşulları | Orta | İyi | İyi | Zayıf-Orta |
| EV uyumu | İyi | Zayıf | İyi | Mükemmel |
| Ölçeklenebilirlik | Düşük | Düşük | Çok Düşük | Yüksek |
| Birim maliyet | Düşük | Düşük | Yüksek | Paylaşımlı (düşük) |

---

## 2.9 Piyasadaki Ticari Sistemler

Akademik çalışmaların yanı sıra park yönetimi alanında faaliyet gösteren ticari sistemler de mevcuttur. Bu sistemlerin incelenmesi, geliştirilen projenin gerçek dünya uygulamalarıyla nasıl ilişkilendirildiğini ortaya koymak açısından önem taşımaktadır.

### Parquery (İsviçre, 2014 — ETH Zürich Çıkışlı)

ETH Zürich araştırmacıları tarafından kurulan Parquery [11], mevcut kameralardan alınan görüntüleri yapay zeka ile işleyerek araçları tespit etmektedir. Özel donanım gerektirmemekte; var olan kamera altyapısına bir yazılım katmanı eklenerek sisteme dahil edilebilmektedir.

Yalnızca doluluk tespiti değil; yasak bölgede park, engelli parkının işgali, süreli park süresinin aşılması ve şarj istasyonunun yanlış kullanımı gibi ihlaller de gerçek zamanlı olarak tespit edilebilmektedir. Bu özellikler açısından Parquery, geliştirilen projeyle en çok örtüşen ticari sistem konumundadır.

- **Ölçek:** 30'dan fazla ülke, 100'den fazla aktif kurulum
- **İddia edilen doğruluk:** %99
- **İş modeli:** SaaS (yazılım hizmet aboneliği), park yeri başına fiyatlandırma
- **Teknik yaklaşım:** Kamera tabanlı, ek altyapı gerektirmez

### Quercus Technologies (İspanya, 25+ Yıl)

Quercus Technologies [12], "sanal döngü" adını verdikleri teknoloji ile zemine gömülü sensör kullanmaksızın araç varlığını tespit etmektedir. Sistem, gölge ve ani ışık değişimi gibi dış etkenleri minimize edecek biçimde tasarlanmıştır. Şirketin geliştirdiği QAI (Quercus AI) platformu aynı anda birden fazla park alanında plaka tanıma ve doluluk tespiti yapabilmektedir.

- **Ölçek:** 60'tan fazla ülkede 11.000'den fazla kurulum
- **Plaka tanıma:** 150'den fazla ülke plakasını desteklemekte, %99 ve üzeri doğruluk
- **Teknik yaklaşım:** Kamera tabanlı görüntü işleme + isteğe bağlı sensör entegrasyonu
- **Öne çıkan özellik:** Gece ve kötü hava koşullarında yüksek performans

### Metropolis (ABD, 2012)

Metropolis, otopark sektörünün en yüksek fonlamasını almış şirketi olma özelliğini taşımaktadır (1,7 milyar dolar). Bilgisayarlı görme tabanlı ödeme sistemleri geliştirmektedir: araç plakası kamerayla tanınmakta ve otomatik olarak ücretlendirme gerçekleştirilmektedir. Bu sayede gişe, ödeme makinesi veya kart okuyucuya ihtiyaç kalmamaktadır.

- **Ölçek:** ABD'de büyük otopark operatörleriyle yaygın entegrasyon
- **Teknik yaklaşım:** Plaka tanıma (LPR — License Plate Recognition)
- **İş modeli:** Otopark operatörleriyle gelir paylaşımı
- **Öne çıkan özellik:** Temassız ödeme deneyimi, insan operatörü ortadan kaldırma

### ParkHub + Smarking (ABD)

ParkHub ve Smarking, büyük etkinlik mekanları ve stadyumlara yönelik SaaS tabanlı park yönetimi çözümleri sunmaktadır. 2.500'den fazla park lokasyonunu kapsayan platformda gerçek zamanlı veri erişimi, dinamik fiyatlandırma ve dijital izin yönetimi temel özellikler arasında yer almaktadır.

- **Ölçek:** 2.500'den fazla lokasyon
- **Teknik yaklaşım:** IoT sensörleri + veri analitiği
- **İş modeli:** SaaS aboneliği
- **Öne çıkan özellik:** Talebe dayalı dinamik fiyatlandırma (konser, maç vb. etkinliklerde anlık fiyat ayarı)

### IEM (Almanya) — Intelligent Parking Solutions

IEM, Avrupa'nın en köklü park yönetimi şirketlerinden biridir. Hem sensör tabanlı hem de kamera tabanlı çözümler sunmaktadır. Özellikle büyük alışveriş merkezi ve hastane otoparkları için kapsamlı sistemler kurulmaktadır. Şehir genelinde merkezi yönetim paneli üzerinden tüm park alanlarının anlık durumu izlenebilmektedir.

- **Teknik yaklaşım:** Ultrasonik sensör + kamera hibrit
- **Öne çıkan özellik:** Merkezi yönetim paneli, şehir ölçekli entegrasyon

### Bosch (Almanya) — Çok Katmanlı Park Çözümleri

Bosch, park yönetimi alanında birbirinden farklı üç yaklaşımla faaliyet yürütmektedir.

**Community-Based Parking:** Araçların sensörlerinden ve akıllı şehir altyapısından toplanan veriler bulut tabanlı yapay zeka platformuna aktarılmaktadır. Bu verilerden gerçek zamanlı dijital park haritası oluşturulmakta; sürücüler müsait park yeri bilgisine navigasyon uygulamaları üzerinden erişebilmektedir.

**Automated Valet Parking (AVP):** Stuttgart Havalimanı ve Almanya'daki 15 farklı otogarajda hayata geçirilmiştir. Araç, sürücüsüz olarak otopark girişinden boş alana kadar yönlendirilmektedir. Sistem LiDAR, kamera ve radar sensörlerini bir arada kullanmaktadır.

**INTEOX Kameralar:** Edge AI (uç bilişim) mimarisiyle çalışan bu kameralar, görüntü işlemeyi kamera üzerinde gerçekleştirmektedir. Detroit M-1 koridorunda konuşlandırılmış olup gerçek zamanlı park durumu tespiti yapılmaktadır. Buluta yalnızca sonuç verisi gönderilmekte; bu sayede bant genişliği ve gecikme sorunları en aza indirilmektedir.

- **Teknik yaklaşım:** Sensör füzyonu + kamera + edge AI
- **Öne çıkan özellik:** Otomatik vale park, edge AI kamera altyapısı
- **Ölçek:** Almanya genelinde 15+ otogaraj, ABD Detroit koridoru

### Hikvision (Çin) — Büyük Ölçekli AI Kamera Sistemleri

Hikvision, dünya genelinde en yaygın kullanılan güvenlik kamerası üreticileri arasında yer almakta ve park yönetimi için özelleştirilmiş yapay zeka çözümleri sunmaktadır.

**Guanlan Büyük AI Modeli:** Çok sayıda kameranın ürettiği görüntüyü merkezi yapay zeka modeliyle analiz etmekte ve park doluluk haritası çıkarmaktadır. DeepinViewX serisi kameralar ile yüksek hassasiyetli araç tespiti ve plaka tanıma yapılmaktadır.

**Irida Labs İş Birliği:** Avrupa merkezli Irida Labs ile gerçekleştirilen ortak projede uç bilişim (edge vision AI) mimarisi kullanılmaktadır. Hikvision'ın kamera donanımı ile Irida Labs'ın görüntü işleme yazılımı birleştirilmekte; görüntü analizi kamera üzerinde tamamlanmaktadır.

- **Teknik yaklaşım:** Kamera tabanlı derin öğrenme + ANPR (plaka tanıma)
- **Öne çıkan özellik:** Büyük ölçekli merkezi AI modeli, edge + merkezi hibrit mimari

### Dahua Technology (Çin) — WizMind AI Platformu

Dahua Technology, WizMind yapay zeka platformunu park yönetimi alanında aktif olarak kullanmaktadır.

Danimarka'daki havalimanı otopark projesinde WizMind kameralar konuşlandırılmıştır. Sistem araç türünü, rengini ve plakasını eş zamanlı olarak tespit edebilmekte; %95 ve üzeri plaka tanıma doğruluğu bildirilmektedir. Boş ve dolu park yerleri anlık olarak izlenmekte, veriler merkezi yönetim sistemine aktarılmaktadır.

- **Teknik yaklaşım:** Kamera tabanlı + ANPR
- **Öne çıkan özellik:** Araç türü + renk + plaka eş zamanlı tespiti, %95+ plaka doğruluğu
- **Referans kurulum:** Danimarka havalimanı otoparkı

### Frogparking (Yeni Zelanda) — Hibrit Sensör Çözümü

Frogparking, açık hava park alanları için ultrasonik sensör ile kamera teknolojisini birleştiren hibrit bir yaklaşım benimsemektedir. Kapalı otoparklar yerine açık hava parkında karşılaşılan hava koşulları ve ışık değişkenliği sorunlarını gidermek amacıyla çift kaynaklı algılama tercih edilmektedir.

- **Teknik yaklaşım:** Ultrasonik sensör + kamera hibrit
- **Öne çıkan özellik:** Açık hava park alanları için optimize edilmiş, hava koşullarına dayanıklı
- **Ölçek:** Yeni Zelanda ve Avustralya merkezli, uluslararası expansion sürecinde

### Ticari Sistemlerin Karşılaştırması

| Sistem | Ülke | Yaklaşım | Plaka Tanıma | İhlal Tespiti | Ek Donanım |
|--------|------|---------|-------------|--------------|------------|
| Parquery | İsviçre | Kamera (AI) | Hayır | Evet | Hayır |
| Quercus | İspanya | Kamera + Sensör | Evet | Kısmi | İsteğe bağlı |
| Metropolis | ABD | Kamera (LPR) | Evet | Hayır | Hayır |
| ParkHub | ABD | IoT Sensör | Hayır | Hayır | Evet |
| IEM | Almanya | Sensör + Kamera | Kısmi | Hayır | Evet |
| Bosch | Almanya | Sensör + Edge AI | Evet | Hayır | Evet (AVP) |
| Hikvision | Çin | Kamera (ANPR+AI) | Evet | Kısmi | Hayır |
| Dahua | Çin | Kamera (WizMind) | Evet | Hayır | Hayır |
| Frogparking | Yeni Zelanda | Sensör + Kamera | Hayır | Hayır | Evet |
| **Geliştirilen Proje** | Türkiye | Kamera (YOLOv8) | Hayır | Altyapı hazır | Hayır |

---

## 2.10 Projenin Literatürdeki Yeri

Tüm bu çalışmalar incelendikten sonra geliştirilen projenin literatürdeki konumu net biçimde ortaya konabilmektedir.

### Güçlü Yönler

**Güncel model kullanılmaktadır:** 2017-2021 arasındaki çalışmalar AlexNet, VGG, mAlexNet gibi daha eski modellerle yürütülmüştür. Geliştirilen projede YOLOv8 kullanılmaktadır; bu model söz konusu alternatiflerin büyük çoğunluğundan daha hızlı ve daha doğrudur.

**Poligon tabanlı park alanı analizi altyapısı geliştirilmiştir:** Literatürdeki çalışmaların büyük çoğunluğu yalnızca "dolu/boş" ikili sınıflandırma yapmaktadır. Geliştirilen projede park alanları ve yasak bölgeler poligon olarak tanımlanabilmekte; IoU hesabıyla doluluk analizi yapılabilmektedir. Üç sınıflı çıktı (park edilebilir / dolu / yasak) ilerleyen aşamada arayüze entegre edilmesi planlanmaktadır.

**Sentetik veri üretilebilmektedir:** Etiketli gerçek veri ihtiyacını azaltmak için otomatik sentetik veri üretimi yapılmaktadır. Bu yaklaşım hem iş yükünü azaltmakta hem de veri çeşitliliğini artırmaktadır.

**Ek donanım gerektirmemektedir:** Mevcut kamera altyapısıyla çalışılabilmekte; Parquery gibi ticari sistemlerle aynı felsefe benimsenmiştir.

### Zayıf Yönler

**PKLot ile kapsamlı karşılaştırma yapılmamıştır:** Literatürdeki standart benchmark üzerinde henüz test gerçekleştirilmemiştir.

**Araç takibi bulunmamaktadır:** Geçen araç ile park edilmiş araç henüz ayırt edilememektedir. ByteTrack entegrasyonu ilerleyen aşamalarda planlanmaktadır.

**Park alanları manuel tanımlanmaktadır:** APSD-OC ve PakLoc gibi çalışmalar bunu otomatik gerçekleştirmektedir; geliştirilen projede henüz el ile çizim yöntemi kullanılmaktadır.

**Gece ve yağmur testi gerçekleştirilmemiştir:** Literatürdeki çalışmaların bir kısmı farklı hava koşullarında test yapmaktadır; bu testler henüz gerçekleştirilememiştir.

### Temel Alınan Çalışmalar

Geliştirilen projeye en yakın referans **da Luz vd. (2024)** [7] çalışmasıdır; YOLOv8 + ROI analizi kombinasyonu bu proje ile büyük ölçüde örtüşmektedir.

Ticari referans olarak **Parquery** [11] baz alınmaktadır: mevcut kameraya yapay zeka ekleyerek ek altyapı gerektirmeden analiz yapılması hedeflenmektedir.

Gelecekteki geliştirme için ise **APSD-OC** otomatik park yeri tespiti yaklaşımı ve **ByteTrack** araç takip algoritması referans noktalarımız.

---

## 2.11 Literatür Özeti Tablosu

| Çalışma | Yıl | Yöntem | Veri Seti | Doğruluk | Bizimle Fark |
|---------|-----|--------|-----------|----------|-------------|
| AlexNet park uyarlaması | 2015 | CNN patch | PKLot | ~%95 | Eski model, sadece dolu/boş |
| mAlexNet (Amato vd.) | 2017 | Küçük CNN | CNRPark | ~%96 | Hafif ama sınırlı |
| CarNet (Nurullayev & Lee) | 2019 | Dilated CNN | PKLot+CNR | ~%97 | Daha iyi ama büyük |
| MobileNetV3+CBAM (Yuldashev vd.) | 2023 | Lightweight CNN | PKLot+CNR | %98.01 | Sınıflandırma odaklı, ihlal yok |
| APSD-OC (Grbić & Koch) | 2023 | YOLOv5+ResNet34 | PKLot+CNR | Yüksek | Otomatik slot tespiti var |
| da Luz vd. | 2024 | YOLOv8-v11+ROI | Özel | %99.68 | En yakın referans, ihlal yok |
| **Bizim Sistemimiz** | 2025 | YOLOv8+IoU+Kural | Sentetik+Gerçek | Ölçülecek | 3 sınıf + ihlal tespiti |

---

Tüm kaynaklar raporun sonundaki Kaynaklar bölümünde listelenmiştir.
