7. SONUÇ VE GELECEK ÇALIŞMALAR
================================================================

7.1 Genel Değerlendirme

Bu tez kapsamında, tek bir kamera görüntüsünden araçları tespit eden, park yerlerinin doluluğunu ve uygunluğunu gerçek zamanlı çözümleyen ve sürücüyü en uygun boş yere yönlendiren, yapay zekâ tabanlı bütünleşik bir sistem geliştirilmiştir. Sistem; ek donanım gerektirmeden, yalnızca kamera görüntüsünden çalışacak biçimde tasarlanmış ve bu yönüyle düşük maliyetli, ölçeklenebilir bir çözüm hedefi gerçekleştirilmiştir.

Geliştirilen sistemin temel yeteneği olan boş/dolu yer ayrımı, sentetik senaryolarda yüksek kesinlikle (1.00) ve dengeli bir başarımla (F1 = 0.94) gerçekleştirilmiştir. Sistem, salt doluluk tespitinin ötesine geçerek, ters perspektif dönüşümü ile park yerlerinin gerçek metrik boyutlarını ölçmekte, aracın o yere sığıp sığamayacağını denetlemekte ve çok kriterli bir karar motoruyla en uygun yeri önermektedir. Buna ek olarak araç takibi, gece görüşü iyileştirme, monoküler derinlik kestirimi ve çift yönlü çevrimdışı sesli asistan gibi sürücü-asistanı özellikleri tek bir uygulamada bütünleştirilmiştir.

Böylece tezin başında ortaya konan araştırma soruları olumlu biçimde yanıtlanmıştır: tek bir kameradan ek donanım olmadan park doluluğu güvenilir biçimde belirlenebilmekte; çizgili ve çizgisiz alanların her ikisinde de çalışan adaptif bir tespit gerçekleştirilebilmekte; metrik ölçümle sığma denetimi yapılabilmekte ve sürücüye en uygun yeri öneren çok kriterli bir karar mekanizması tasarlanabilmektedir.


7.2 Literatürle Karşılaştırma

Geliştirilen sistem, literatürdeki güncel çalışmalarla aynı temel akışı (kamera, araç tespiti, bölge analizi, doluluk kararı) paylaşmakla birlikte, birkaç yönüyle ayrışmaktadır. Literatürdeki çalışmaların büyük çoğunluğu yalnızca dolu/boş ikili sınıflandırması yaparken, bu sistem boş yerleri adaptif biçimde çıkarmakta ve gerçek metrik ölçümle sığma denetimi sunmaktadır. En yakın akademik referans olan da Luz vd. (2024) [13] çalışması YOLO ile bölgesel doluluk analizini birleştirir; bu sistem ise aynı temeli metrik ölçüm, sığma denetimi ve çok kriterli yönlendirme ile genişletmektedir. Ticari sistemlerden Parquery [2] ile aynı "mevcut kameraya yapay zekâ ekleme" felsefesi benimsenmiştir. Otomatik park yeri tespiti açısından APSD-OC [11] yaklaşımı, araç takibi açısından ise ByteTrack [17] çalışması, sistemin ileriye dönük gelişimi için referans noktaları olarak değerlendirilmiştir.


7.3 Karşılaşılan Zorluklar

Geliştirme sürecinde çeşitli teknik zorluklarla karşılaşılmıştır. Çapraz açılı görüntülerde metrik ölçümün doğru yapılabilmesi, ters perspektif dönüşümü ve ölçek kestiriminin dikkatli biçimde kurgulanmasını gerektirmiştir. Hareketli kameralarda park etmiş araçların hareketli sanılması sorunu, ego-hareket telafisi ile çözülmüştür. Boyalı şeridin bulunduğu ve bulunmadığı alanlar arasında kararlı geçiş yapabilmek için adaptif mod seçiminin histerezisle yumuşatılması gerekmiştir. Ayrıca, araçların yandan görüldüğü dik park senaryolarında boş yer sınırlarının doğru belirlenebilmesi için perspektif düzeltme mantığının bu özel duruma uyarlanması gerekmiştir. Tüm bu zorluklar, sistemin kararlılığını koruyan çözümlerle ele alınmıştır.


7.4 Gelecek Çalışmalar

Geliştirilen sistem, çeşitli yönlerden ilerletilmeye açıktır. Öncelikli gelecek çalışmalar şunlardır:

• Öğrenme tabanlı doluluk sınıflandırması: Mevcut sistemde boş/dolu kararı ağırlıklı olarak geometrik ve örtüşme tabanlı yöntemlere dayanmaktadır. Kendi verisiyle eğitilen hafif bir evrişimli sinir ağı doluluk sınıflandırıcısı — örneğin MobileNetV2 [45] gibi verimli bir mimari — eklenerek, özellikle gece ve zorlu hava koşullarında doğruluğun artırılması hedeflenmektedir.

• Gelişmiş araç takibi: Geçici ve park edilmiş araçların ayrımını daha da güçlendirmek için ByteTrack [17] gibi güncel bir çok-nesneli takip algoritması entegre edilebilir.

• Standart veri setleri üzerinde değerlendirme: Sistemin PKLot [7] gibi standart benchmark veri setleri üzerinde geniş ölçekli olarak test edilmesi ve literatürdeki çalışmalarla doğrudan karşılaştırılması planlanmaktadır.

• Çoklu kamera füzyonu: Birden çok kameranın görüntüsü birleştirilerek tüm park alanının tek bir bütünleşik haritada izlenmesi sağlanabilir.

• İşlem hattının hızlandırılması: Modellerin niceleme (quantization) ve optimizasyon teknikleriyle hızlandırılarak gömülü (edge) cihazlarda daha yüksek kare hızında çalıştırılması hedeflenmektedir.


7.5 Kapanış

Sonuç olarak bu tez, yalnızca bir kamera kullanarak, ek altyapı gerektirmeden park yeri analizi, gerçek metrik ölçüm ve sürücü yönlendirmesi gerçekleştiren bütünleşik bir sistem ortaya koymuştur. Elde edilen sonuçlar, mevcut kamera altyapısıyla çalışan düşük maliyetli ve ölçeklenebilir bir akıllı park çözümünün uygulanabilir olduğunu göstermektedir. Önerilen gelecek çalışmalarla birlikte sistemin, gerçek dünya koşullarındaki başarımının daha da artırılması ve akıllı şehir uygulamalarına entegre edilebilecek olgunluğa ulaştırılması mümkündür.


7.6 Ekip Katkı Beyanı
----------------------------------------------------------------

Bu bitirme çalışmasının fikir aşamasından yazılım gerçekleştirimine ve tez yazımına kadar olan tüm süreçlerinde, ekip üyelerinin katkı payları ve üstlendikleri sorumluluklar uluslararası CRediT (Contributor Roles Taxonomy) standartları çerçevesinde aşağıdaki tabloda ayrıntılı olarak sunulmuştur.

Tablo 7.1. Ekip üyelerinin görev ve katkı dağılımı

| Çalışma Alanı / Görev | Kerem Yağız KARAKAŞ | Ahmet EKŞİOĞLU | Ezgi KIRNAPÇI |
| :--- | :---: | :---: | :---: |
| **Kavramsallaştırma ve Tasarım** <br> (Fikir geliştirme, mimari tasarım) | %100 | - | - |
| **Tez Metni Yazımı (Original Draft)** <br> (Giriş, Literatür, Mimari, Sonuç) | %100 | - | - |
| **Yazılım Geliştirme (Core Software)** <br> (İşlem hattı, IPM geometrisi, GUI) | %90 | %5 | %5 |
| **Deneysel Değerlendirme & Test** <br> (Sentetik veri üreteci, ablasyon) | %100 | - | - |
| **Görselleştirme** <br> (Sistem şemaları, grafik ve çizimler) | %100 | - | - |
| **Kaynak Araştırması & Kaynakça** <br> (Literatür taraması ve tasnifi) | %100 | - | - |

**Detaylı Açıklamalar:**
* **Kerem Yağız KARAKAŞ:** Projenin fikir babası olup, sistem mimarisini ve genel akışı tasarlamıştır. Yapay zekâ modellerinin (YOLOv8, MiDaS, YOLOPv2) entegrasyonu, ters perspektif dönüşümü (IPM), metrik ölçüm algoritmaları, çok kriterli öneri motoru, sesli asistan bileşenleri ve PyQt5 kullanıcı arayüzünün tamamını içeren çekirdek yazılım mimarisini geliştirmiştir (%90 yazılım katkısı). Tez metninin (tüm bölümler), sentetik test yığınının, performans analizlerinin ve grafiklerin hazırlanmasını tek başına üstlenmiştir.
* **Ahmet EKŞİOĞLU & Ezgi KIRNAPÇI:** Yazılım geliştirme sürecinin başlangıç aşamasında yardımcı kütüphanelerin araştırılması, veri yükleme modüllerinin kurulması ve test senaryolarının altyapı hazırlığı gibi destekleyici yazılım görevlerinde yer almışlardır (%10 yazılım katkısı).

