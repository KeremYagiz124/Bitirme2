5. GERÇEKLEŞTİRİM
================================================================

Bu bölümde, önceki bölümde tasarımı anlatılan sistemin yazılımsal olarak nasıl gerçekleştirildiği; gerçek zamanlı çalışmayı sağlayan iş parçacığı yönetimi, kararlılık teknikleri, perspektif düzeltme ve kullanıcı arayüzü ayrıntılarıyla ele alınmaktadır.


5.1 Yazılım Mimarisi ve Modülerlik

Sistem, sorumlulukları ayrılmış paketlerden oluşan modüler bir yapıda geliştirilmiştir. Algılama bileşenleri araç tespiti, takip, çizgi tespiti, geometri tabanlı boşluk analizi, derinlik ve sürülebilir alan modüllerini; geometri paketi ters perspektif dönüşümü ve kalibrasyonu; park paketi slot skorlama, öğrenilen bellek ve doluluk analizini; sesli asistan paketi konuşma tanıma ve seslendirmeyi; değerlendirme paketi ise metrik, sentetik veri ve ablasyon araçlarını barındırır. Tüm bu bileşenler, kullanıcı arayüzünü oluşturan ana pencere tarafından bir araya getirilerek tek bir uygulamada düzenlenmektedir. Bu ayrışma, her bileşenin bağımsız olarak geliştirilip test edilebilmesini sağlamaktadır.


5.2 Gerçek Zamanlı İşlem Hattı ve İş Parçacığı Yönetimi

Gerçek zamanlı çalışmanın korunması için işlem hattında birkaç önemli teknik uygulanmıştır. Video kareleri, arayüzü bloke etmemek amacıyla ayrı bir arka plan iş parçacığında çözülmektedir; çözülen kareler en fazla bir kare tutan bir kuyruk üzerinden ana işleme aktarılır. Bu tasarım, arayüzün akıcı kalmasını sağlamanın yanı sıra, duraklatılmış videoda konum değiştirme (seek) sırasında oluşabilecek çözücü hatalarını da önlemektedir: konum değiştirildiğinde kuyruk temizlenip okuma ertelenerek kararlı bir geçiş sağlanır.

Araç tespiti, görüntü işleme hattının en maliyetli adımıdır. Bu nedenle video ve kamera modunda, YOLO çıkarımı her karede değil, belirli bir kare aralığında bir çalıştırılır; ara karelerde önceki çıkarım sonuçları önbellekten kullanılır (kare atlama). Böylece görüntü akıcılığı korunurken hesaplama yükü belirgin biçimde azaltılır. Ek olarak, ilgi bölgesi tanımlandığında görüntü yalnızca o bölgeyle sınırlanarak (kırpma) işlenir ve tespit koordinatları sonradan tam çerçeveye geri taşınır; bu da gereksiz hesaplamayı ortadan kaldırır.

Hesaplama açısından ağır olan sürülebilir alan segmentasyonu, ana işlemeyi yavaşlatmaması için ayrı bir arka plan iş parçacığında yürütülür ve sonucu önbelleğe alınır; sahne yavaş değiştiğinden bu maske yalnızca belirli aralıklarla yenilenir. Benzer biçimde, otomatik ters perspektif kalibrasyonu da belirli sayıda çıkarım karesinde bir güncellenerek kameranın yönelimindeki değişimlere uyum sağlar.


5.3 Kararlılık Teknikleri

Canlı görüntüde tespitlerin kare-kare titremesini azaltmak için birden çok kararlılık tekniği bir arada kullanılmaktadır. Çizgi tabanlı slot tespitinde, slotların doluluk durumu zamansal oylama ile yumuşatılır; her slot kareler arasında eşleştirilip son karelerin oy çoğunluğuna göre kararlı hâle getirilir.

Elde-çekim kameralarda görüntünün titremesi, tek seferlik ters perspektif kalibrasyonunu ve kare-arası eşleştirmeleri geçersiz kılabilir. Bu durumu telafi etmek için isteğe bağlı, kullanıcı tarafından etkinleştirilen bir video sabitleme bileşeni geliştirilmiştir: bileşen açıkken her kare, bir referans kareye ORB öznitelikleri [43] çıkarılıp eşleştirilerek ve aykırı eşleşmeler RANSAC [44] ile elenip bir homografi kestirilerek hizalanır. Böylece arka planda ters perspektif dönüşümü ve ızgara hizalaması geçerli kalır; bu adım yalnızca görsel bir iyileştirme değil, sistemin geometrik doğruluğunu koruyan teknik bir önlemdir. Yeterli öznitelik veya eşleşme bulunamadığında kare olduğu gibi geçirilerek sistemin kararlılığı korunur.

Kullanıcının gereksiz biçimde sık uyarılmaması için uyarı sistemi de bir kısıtlama (throttle) mekanizmasıyla çalışır; aynı türden uyarılar belirli bir süre içinde yalnızca bir kez gösterilir.


5.4 Perspektif Düzeltme ve Yan Görüş Ele Alımı

Çapraz açılı görüntülerde araç sınırlayıcı kutuları, aracın yan profilini de içerecek biçimde gerçek ön/arka genişliğinden daha geniş çıkabilir; bu durum komşu boş yerlerin hatalı biçimde dolu sayılmasına yol açabilir. Bunu önlemek için, tespit edilen araç kutuları aracın görünüm açısına göre düzeltilir: kutunun en-boy oranından aracın ne kadar yandan göründüğü kestirilir ve fazlalık yan bölge kırpılarak ön profil genişliğine indirgenir.

Bu düzeltmenin kendisinin sorun yaratabildiği bir durum özellikle ele alınmıştır. Araç tamamen yandan görüldüğünde, sınırlayıcı kutunun kenarları zaten aracın gerçek ön ve arka tamponlarına karşılık gelir; bu durumda kırpma uygulanması, boş yer kutularının aracın gerçek sınırlarının içine doğru kaymasına neden olur. Bu nedenle, aracın neredeyse tamamen yandan göründüğü tespit edildiğinde kırpma adımı atlanmakta ve kutu olduğu gibi korunmaktadır. Bu iyileştirme, yandan görülen dik park senaryolarında boş yer sınırlarının doğru belirlenmesini sağlamıştır.


5.5 Kullanıcı Arayüzü

Kullanıcı arayüzü, sistemin tüm yeteneklerini tek bir masaüstü uygulamasında bir araya getirir. Ana görünümde canlı görüntü veya video, tespit edilen araçlar ve boş/dolu yerlerle birlikte gösterilir; boş yerler yeşil, dolu yerler kırmızı renkle ayırt edilir. Yan panellerde gece görüşü, kuş bakışı görünüm, derinlik modu, ilgi bölgesi seçimi ve kalibrasyon gibi denetimler bulunur. Park alanının genel durumu, iki boyutlu bir şematik harita üzerinde kuşbakışı olarak da özetlenir.

Sesli komutlar ayrı bir iş parçacığında dinlendiğinden, tanınan bir komut arayüzün ana iş parçacığına güvenli biçimde aktarılarak yürütülür; bu, arayüz çatısının iş parçacığı güvenliği gereksinimini karşılar. Komut yürütüldükten sonra kullanıcıya sesli bir geri bildirim verilir. Kullanıcı, aynı işlemleri hem fare ve klavye hem de sesle gerçekleştirebilir.


5.6 Yapılandırma ve Genişletilebilirlik

Sistemin davranışını belirleyen eşik ve parametreler (güven skoru eşiği, minimum boşluk oranı, kare atlama aralığı, uyarı süresi gibi) yapılandırılabilir biçimde tasarlanmıştır. İsteğe bağlı ağır modeller (derinlik, sürülebilir alan) bulunmadığında sistemin temel davranışa geçmesi sayesinde, uygulama farklı donanım koşullarında çalışabilmektedir. Modüler yapı, ileride yeni bir doluluk sınıflandırıcısı veya farklı bir takip algoritması gibi bileşenlerin sisteme görece kolay eklenebilmesine olanak tanımaktadır.
