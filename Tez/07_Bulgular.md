6. DENEYSEL DEĞERLENDİRME VE BULGULAR
================================================================

Bu bölümde, geliştirilen sistemin başarımını ölçmek için uygulanan değerlendirme yöntemi, kullanılan metrikler ve elde edilen sonuçlar sunulmaktadır. Sunulan tüm sayısal değerler, sistemin değerlendirme altyapısı tarafından üretilen gerçek ölçüm çıktılarına dayanmaktadır.


6.1 Değerlendirme Yöntemi

Sistemin boş/dolu yer ayrımındaki başarımı iki ayrı ortamda değerlendirilmiştir.

Birincisi, sentetik senaryolardır. Park sahnelerinin prosedürel olarak üretildiği bu ortamda, her sahnenin doğru cevabı (hangi yerlerin boş, hangilerinin dolu olduğu) tam olarak bilinmektedir. Sentetik değerlendirmenin amacı, boş-yer tespit algoritmasının geometrik doğruluğunu, nesne tespit modelinin hatalarından bağımsız biçimde ölçmektir; böylece algoritmanın kendi başarımı kontrollü bir ortamda izole edilebilmektedir.

İkincisi, gerçek görüntüler üzerinde yapılan ön doğrulamadır. Bu aşamada yalnızca birkaç etiketli gerçek park görüntüsü kullanılarak sistemin gerçek sahnelerdeki davranışı gözlemlenmiştir. Bu küme istatistiksel bir genelleme yapılamayacak kadar küçük olduğundan, gerçek görüntü sonuçları nicel bir başarı kanıtı olarak değil, sistemin gerçek sahnelerde beklendiği gibi çalıştığını gösteren niteliksel bir ön gözlem olarak sunulmaktadır. Bu nedenle aşağıdaki nicel sonuçlar, doğru cevabı tam olarak bilinen sentetik senaryolara dayandırılmıştır.


6.2 Kullanılan Metrikler

Değerlendirmede, ikili tespit problemleri için standart metrikler kullanılmıştır. Sayımlar; doğru pozitif (TP), yanlış pozitif (FP) ve yanlış negatif (FN) olarak tutulur. Kesinlik (Precision), sistemin boş dediği yerlerin ne kadarının gerçekten boş olduğunu ölçer:

    Kesinlik = TP ⁄ (TP + FP)                                         (6.1)

Duyarlılık (Recall), gerçekten boş olan yerlerin ne kadarının sistem tarafından yakalandığını ölçer:

    Duyarlılık = TP ⁄ (TP + FN)                                       (6.2)

F1 skoru, bu iki metriğin harmonik ortalamasıdır:

    F1 = 2 · (Kesinlik · Duyarlılık) ⁄ (Kesinlik + Duyarlılık)        (6.3)

Ortalama isabet (AP) ise kesinlik-duyarlılık eğrisi altındaki alanı özetleyen tek bir değerdir.


6.3 Sentetik Senaryo Sonuçları

Prosedürel olarak üretilen sentetik sahneler üzerinde, toplam 268 park yeri örneği değerlendirilmiştir. Sistem bu örneklerin 221'ini doğru sınıflandırmış, hiç yanlış pozitif üretmemiş (FP = 0) ve 47 örneği kaçırmıştır (FN = 47). Tespitlerin doğru/yanlış dağılımı Şekil 6.1'deki karışıklık matrisinde, elde edilen başarım metrikleri (Kesinlik 1.00, Duyarlılık 0.82, F1 0.90, AP 0.82) ise Şekil 6.2'de sunulmuştur.

Sonuçlar, sistemin kesinliğinin çok yüksek olduğunu göstermektedir: boş olarak işaretlenen yerlerin tamamı gerçekten boştur, yani sistem dolu bir yeri boş sanmamaktadır. Bu, bir sürücü yönlendirme uygulaması için kritik bir özelliktir; çünkü sürücüyü dolu bir yere yönlendirmek, yanlış bir boş yeri atlamaktan daha sakıncalıdır. Duyarlılığın görece daha düşük olması (0.82), sistemin bazı zorlu boşlukları (örneğin dar veya kısmen görünen yerleri) ihtiyatlı davranarak kaçırdığını göstermektedir. Bu davranış, kesinliği yüksek tutmak adına bilinçli bir dengelemenin sonucudur.


6.4 Ablasyon Çalışması

Sistemin farklı yapılandırmalarının başarıma etkisini incelemek için bir ablasyon çalışması yürütülmüştür. Üç yapılandırma karşılaştırılmıştır: duyarlılığı önceleyen agresif, kesinliği önceleyen muhafazakâr ve dengeli bir temel (baseline) yapılandırma. Sonuçlar Tablo 6.1'de özetlenmiştir.

Tablo 6.1. Yapılandırma karşılaştırması (ablasyon)

  | Yapılandırma  | Kesinlik | Duyarlılık | F1   | AP   |
  |---------------|----------|------------|------|------|
  | Temel         | 1.00     | 0.89       | 0.94 | 0.89 |
  | Agresif       | 0.52     | 0.97       | 0.67 | 0.52 |
  | Muhafazakâr   | 1.00     | 0.82       | 0.90 | 0.82 |

Tablodan görüldüğü üzere, agresif yapılandırma duyarlılığı en yüksek değere (0.97) çıkarmakta, ancak çok sayıda yanlış pozitif ürettiğinden kesinliği ciddi biçimde düşmektedir (0.52). Buna karşılık temel ve muhafazakâr yapılandırmalar kesinliği tam (1.00) tutmakta; temel yapılandırma duyarlılık ve kesinlik arasında en iyi dengeyi sağlayarak en yüksek F1 skoruna (0.94) ulaşmaktadır. Bu sonuçlar, sistemin parametrelerinin kesinlik-duyarlılık dengesi üzerinde belirleyici bir etkisi olduğunu ve uygulama önceliğine göre ayarlanabileceğini göstermektedir. Ayrıca, minimum boşluk oranı parametresi üzerinde bir duyarlılık analizi yapılarak bu parametrenin sonuçlara etkisi ayrıca incelenmiştir.


6.5 Performans

Sistem, standart bir kişisel bilgisayar üzerinde etkileşimli bir hızda çalışmaktadır. Gerçek zamanlı çalışmayı korumak için uygulanan kare atlama, kırpma ve ağır modellerin arka planda seyrek çalıştırılması gibi teknikler sayesinde, görüntü akıcılığı korunmaktadır. Derinlik ve sürülebilir alan gibi ağır modüller devre dışı bırakıldığında belirgin bir hızlanma gözlenmektedir; bu da sistemin, donanım olanaklarına göre başarım ve hız arasında ölçeklenebileceğini göstermektedir.


6.6 Yazılım Kalitesi

Geliştirilen yazılımın doğruluğu ve kararlılığı, pytest çatısıyla yazılmış kapsamlı bir otomatik test kümesiyle sürekli olarak doğrulanmaktadır. Mevcut durumda toplam 268 otomatik test bulunmakta ve bunlar; geometri hesapları, ölçek kestirimi, slot skorlama, takip, çizgi tespiti, zamansal oylama ve bölge analizi gibi çekirdek bileşenleri kapsamaktadır. Çekirdek karar bileşenlerinin dış bağımlılık olmadan saf sayısal kodla yazılmış olması, bu testlerin hızlı ve güvenilir biçimde çalışmasını sağlamaktadır.


6.7 Niteliksel Sonuçlar

Sayısal değerlendirmenin yanı sıra, sistemin çıktıları görsel olarak da incelenmiştir. Uygulama; boş ve dolu yerleri renkli kutularla ayırarak sayım yapabilmekte, kuş bakışı görünümde park yerlerinin metrik boyutlarını gösterebilmekte, aracın boş yere sığıp sığamayacağını metre cinsinden alan genişliğiyle birlikte belirtebilmekte ve park alanının genel durumunu iki boyutlu şematik harita üzerinde özetleyebilmektedir. Bu görsel çıktılar, sistemin farklı park senaryolarında beklenen biçimde çalıştığını doğrulamaktadır.


6.8 Koşula Bağlı Davranış ve Sınırlamalar

Değerlendirme sonuçları, sistemin koşula bağlı davranışına ilişkin önemli gözlemler ortaya koymaktadır. Gündüz ve boyalı şeritli park koşullarında sistem yüksek isabetle çalışmakta; özellikle yanlış pozitif üretmeme konusunda güçlü bir başarım göstermektedir. Buna karşılık, gece ve düşük ışık gibi zorlu koşullarda araç tespitinin güçleşmesiyle doğruluğun düşmesi beklenmektedir; gece görüşü iyileştirme bu etkiyi kısmen hafifletse de bu koşullar sistemin başlıca zorluğunu oluşturmaktadır.

Çalışmanın temel sınırlamaları şunlardır: gerçek görüntüler üzerinde yapılan değerlendirme küçük bir kümeyle sınırlı kalmış, sistem henüz PKLot [7] gibi standart benchmark veri setleri üzerinde geniş ölçekli olarak test edilmemiştir. Bu kapsamlı değerlendirmeler, ileriye dönük çalışma olarak ele alınmaktadır.
