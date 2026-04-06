# 1. GİRİŞ

## 1.1 Projenin Konusu ve Amacı

Dünya genelinde kentsel nüfus hızla artmaya devam etmekte ve buna paralel olarak şehirlerdeki araç sayısı da her geçen yıl yükselmektedir. Türkiye İstatistik Kurumu verilerine göre Türkiye'deki kayıtlı araç sayısı 2023 yılında 27 milyonu aşmıştır. Bu artış, zaten yetersiz olan park altyapısını daha da büyük bir baskı altına sokmaktadır.

Özellikle şehir merkezlerinde yasadışı park, günlük hayatın olağan bir parçası haline gelmiştir. Kaldırıma park edilen araçlar yayaların ve engellilerin geçişini engellemektedir. Çift sıra park edenler trafiği tıkamakta, kavşak köşelerine park edenler görüş açısını kapatarak kaza riskini artırmaktadır. Yangın çıkışlarının ve acil servis yollarının önünü kapatan araçlar ise can güvenliğini tehdit etmektedir.

Bu sorunla başa çıkmak için kullanılan mevcut yöntemler büyük ölçüde trafik polisinin ya da güvenlik görevlilerinin sahaya çıkıp araçları tek tek kontrol etmesine dayanmaktadır. Bu yaklaşımın üç temel sorunu bulunmaktadır: birincisi, yalnızca görevlinin orada bulunduğu anlarda etkili olmaktadır; ikincisi, çok sayıda personel gerektirdiği için maliyetlidir; üçüncüsü, insan hatasına açık olduğundan tutarsız uygulamaya yol açabilmektedir.

Bu projede söz konusu soruna tamamen otomatik ve ölçeklenebilir bir çözüm geliştirilmesi hedeflenmiştir. Geliştirilen sistem; bilgisayar kamerası, telefon kamerası veya harici bir kameradan alınan görüntüleri yapay zeka ile analiz ederek araçları tespit etmekte ve aracın bulunduğu konumun park kurallarına uygun olup olmadığını gerçek zamanlı olarak değerlendirmektedir.

---

## 1.2 Neden Bu Problem Önemli?

Park sorunu yüzeysel bakıldığında basit bir "düzensizlik" meselesi gibi görünebilir. Oysa etkisi çok daha derine inmektedir.

**Trafik akışı üzerindeki etkisi:** Çift sıra ya da yasak bölgeye park, trafik akışını bozarak ortalama seyahat sürelerini ciddi biçimde uzatmaktadır. Texas A&M Ulaştırma Enstitüsü'nün araştırmalarına göre trafik sıkışıklığı ABD'de yılda 87 milyar dolarlık kayba yol açmaktadır; bu kaybın önemli bir bölümü yasadışı park kaynaklı darboğazlardan kaynaklanmaktadır.

**Acil müdahale gecikmeleri:** İtfaiye ya da ambulans araçlarının yasadışı park nedeniyle geçemediği senaryolar teorik değildir. Araştırmalar, acil servislerin kentsel alanlarda ortalama müdahale süresinin yasadışı park nedeniyle %15-20 oranında uzayabildiğini göstermektedir.

**Ekonomik kayıp:** Yasadışı park cezaları ve bu cezaların tahsil edilememesi büyük bir sorun oluşturmaktadır. İngiltere'de her yıl tahsil edilemeyen park cezalarının toplamı yüz milyonlarca pound düzeyinde seyretmektedir. Otomatik tespit sistemleri bu kayıpları ciddi ölçüde azaltabilir.

**Çevre etkisi:** Araştırmalar, park yeri arayan araçların şehir trafiğinin yaklaşık %30'unu oluşturduğunu ve bu boşta dolaşmanın gereksiz yakıt tüketimine ve emisyona yol açtığını ortaya koymaktadır. Doluluk bilgisinin anlık paylaşılması bu problemi azaltabilir.

---

## 1.3 Motivasyon

Bu projenin gerçekleştirilmesinin teknik ve ekonomik gerekçeleri bulunmaktadır:

**Pazar büyük ve büyümektedir:** Küresel akıllı park sistemleri pazarının 2024 yılında 7.85 milyar dolar büyüklüğünde olduğu tahmin edilmektedir. 2035'e kadar yılda yaklaşık %17-18 büyümesi öngörülmekte ve pazar değerinin 47 milyar dolara ulaşması beklenmektedir [1].

**Teknoloji artık buna hazırdır:** 2015 yılında ilk YOLO modeli yayınlandığında gerçek zamanlı nesne tespiti yalnızca yüksek uçlu GPU'larda mümkündü. Bugün YOLOv8, orta sınıf bir dizüstü bilgisayarda 30+ FPS ile çalışabilmektedir. Hesaplama maliyeti önemli ölçüde düşmüştür.

**Mevcut sistemler pahalı ve kırılgandır:** Zemine gömülen manyetik sensörler veya döngü bobinleri yüksek kurulum maliyeti gerektirmektedir. Yol yüzeyi değiştiğinde ya da sensör arızalandığında sistemin tamamı devre dışı kalmaktadır. Kamera tabanlı bir yaklaşım aynı bilgiyi çok daha esnek ve düşük maliyetle üretebilmektedir.

**Elektrikli araçlar yeni bir problem oluşturmaktadır:** Manyetik sensörler, araçların kütlesi nedeniyle ortaya çıkan manyetik alan bozulmasını okuyarak çalışmaktadır. Ancak elektrikli araçlar hafif yapıları ve gövde tasarımları nedeniyle manyetik alana çok az etki etmektedir. Bu durum, yalnızca manyetometreye dayanan sistemleri giderek daha yetersiz kılmaktadır. Kamera tabanlı sistemler bu kısıtı yaşamamaktadır.

**Transfer learning ile az veriyle güçlü model oluşturulabilmektedir:** COCO gibi devasa veri setleri üzerinde önceden eğitilmiş modeller, transfer learning yöntemiyle ilgili probleme uyarlanabilmektedir. Bu yaklaşım, akademik bir proje için de uygulanabilir hale gelmektedir.

---

## 1.4 Sistemin Genel Yapısı

Geliştirilen sistem üç ana bileşenden oluşmakta ve bu bileşenler sıralı bir pipeline şeklinde çalışmaktadır:

**Bileşen 1 — Araç Tespiti (YOLOv8)**
Kameradan gelen görüntü üzerinde araçlar tespit edilmektedir. Sistem araba, otobüs, kamyon ve motosiklet sınıflarını ayrı ayrı tanıyabilmektedir. Her tespit için bounding box koordinatları ve güven skoru hesaplanmaktadır. Microsoft COCO veri setiyle önceden eğitilmiş YOLOv8n modeli kullanılmaktadır [2].

**Bileşen 2 — Park Alanı Analizi (OpenCV + IoU)**
Park alanları ve yasak bölgeler sisteme poligon çizilerek bir kez tanıtılmaktadır. Ardından araç tespitinden gelen bounding box ile park alanı poligonu arasındaki örtüşme oranı hesaplanmaktadır. Bu hesaplama IoU (Intersection over Union) yöntemiyle gerçekleştirilmektedir. IoU > 0.3 eşiğini geçen araçlar için ilgili park yeri "dolu" kabul edilmektedir.

**Bileşen 3 — Karar Mekanizması (Kural Motoru)**
IoU analizi sonucuna ve park alanının özelliğine bağlı olarak üç sınıftan biri üretilmektedir:
- **Park Edilebilir:** Araç yok, alan uygun
- **Park Alanı Dolu:** Alan dolu, kurallara uygun araç mevcut
- **Park Edilemez:** Araç yasak bölgedeyse ya da uygunsuz konumdaysa

```
Kamera Görüntüsü
        ↓
[YOLOv8 — Araç Tespiti]
   Bounding Box + Confidence Score
        ↓
[Park Alanı Poligonlarıyla IoU Hesabı]
   IoU > 0.3 → Dolu | IoU < 0.3 → Boş
        ↓
[Kural Motoru]
   Normal Alan + Dolu → "Park Alanı Dolu"
   Yasak Bölge + Araç Var → "Park Edilemez"
   Normal Alan + Boş → "Park Edilebilir"
        ↓
[PyQt5 Arayüzü — Gerçek Zamanlı Görselleştirme]
```

---

## 1.5 Araştırma Sorusu ve Hipotez

**Temel araştırma sorusu:**
Kentsel alanlarda kamera görüntülerinden hareketle, bir aracın bulunduğu konumda park etmesinin uygun olup olmadığı; derin öğrenme tabanlı nesne tespiti ve görüntü işleme tekniklerinin birlikte kullanımıyla, insan müdahalesine ihtiyaç duyulmaksızın, gerçek zamanlı ve güvenilir biçimde otomatik olarak belirlenebilir mi?

**Alt sorular:**
1. YOLOv8 tabanlı nesne tespit modeli, farklı ışık koşulları ve trafik yoğunluğu senaryolarında araçları ne ölçüde doğru ve tutarlı biçimde tespit edebilmektedir?
2. Geliştirilen park uygunluğu sınıflandırıcısı, farklı kamera açılarında %85 ve üzeri doğruluk değerine ulaşabilmekte midir?
3. IoU analizi tek başına park uygunluğunu belirlemek için yeterli midir, yoksa ek bir sınıflandırma katmanına ihtiyaç duyulmakta mıdır?
4. Sentetik veriyle eğitilen model gerçek kamera görüntülerine ne kadar iyi genelleyebilmektedir?

**Hipotez:**
YOLOv8 tabanlı nesne tespiti ile poligon tabanlı park alanı analizinin kural motoru aracılığıyla birleştirilmesinin; park uygunluğunu "Park Edilebilir", "Park Edilemez" ve "Park Alanı Dolu" şeklinde üç sınıfta, %85 ve üzeri doğrulukla, gerçek zamanlı olarak sınıflandırmak için yeterli olduğu öngörülmektedir.

---

## 1.6 Projenin Kapsamı

**Kapsam içinde:**
- Sabit kamera görüntüsünden araç tespiti (araba, otobüs, kamyon, motosiklet)
- Poligon tabanlı park alanı ve yasak bölge tanımlama aracı
- IoU tabanlı doluluk analizi
- 3 sınıflı park uygunluğu kararı
- Sentetik veri üretimi ve YOLOv8 fine-tuning
- Gerçek zamanlı grafik arayüz (PyQt5)
- Performans metrikleri: Accuracy, Precision, Recall, F1-Score, mAP

**Kapsam dışında:**
- Plaka tanıma ve araç kimliği
- Otomatik ceza kesme veya bildirim sistemi
- Mobil uygulama
- Gece görüş kamerası entegrasyonu
- Araç takibi (tracking) — ilerleyen versiyonlar için planlanmıştır

---

## 1.7 Projenin Özgün Katkısı

Literatürde bu alanda pek çok çalışma mevcut olmasına rağmen geliştirilen projenin kendine özgü birkaç yönü bulunmaktadır:

**Birden fazla ihlal türü aynı anda işlenmektedir:** Literatürdeki çalışmaların büyük çoğunluğu yalnızca "dolu mu / boş mu" sorusuna yanıt vermektedir. Geliştirilen sistemde buna ek olarak yasak bölge ihlali de tespit edilmekte ve üç sınıflı bir çıktı üretilmektedir.

**Sentetik veri + gerçek kamera hibrit yaklaşımı benimsenmiştir:** Sistem hem otomatik üretilen sentetik veriyle eğitilmekte hem de gerçek kamera görüntüsü üzerinde test edilebilmektedir. Bu sayede etiketli gerçek veri ihtiyacı azalmaktadır.

**Mevcut kameralarla çalışılabilmektedir:** Ek altyapı gerektirmeyen sistem, park alanlarının bir kez poligonla tanımlanması ardından tamamen otomatik çalışmaktadır.

**Düşük donanımda çalışılabilmektedir:** GTX 1050 Ti gibi sınırlı bir GPU'da bile gerçek zamanlı çalışabilen YOLOv8n tercih edilmiştir.

---

## 1.8 Raporun Yapısı

Bu rapor şu şekilde düzenlenmiştir:

- **Bölüm 2 — Literatür Araştırması:** Akademik çalışmalar, benchmark veri setleri, ticari sistemler ve projenin bu ekosistem içindeki yeri incelenmektedir.
- **Bölüm 3 — Sistem Mimarisi:** Geliştirilen sistemin genel yapısı, bileşenler arası veri akışı ve tasarım kararları açıklanmaktadır.
- **Bölüm 4 — Kullanılan Teknolojiler:** Seçilen araç ve kütüphanelerin gerekçeleri ve alternatifleriyle karşılaştırması verilmektedir.
- **Bölüm 5 — Uygulama ve Bulgular:** Yapılan çalışmalar, ekran görüntüleri ve ölçüm sonuçlarıyla aktarılmaktadır.
- **Bölüm 6 — Sonuç ve Gelecek Çalışmalar:** Elde edilen bulgular değerlendirilmekte, sınırlamalar tartışılmakta ve ileriye dönük geliştirme önerileri sunulmaktadır.
