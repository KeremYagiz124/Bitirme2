1. GİRİŞ
================================================================

1.1 Problemin Tanımı ve Önemi

Şehirleşmenin hızlanması ve özel araç sahipliğinin sürekli artması, kentsel park yönetimini günümüzün en belirgin ulaşım sorunlarından biri hâline getirmiştir. Sürücüler, özellikle yoğun saatlerde boş bir park yeri bulabilmek için kayda değer bir süre harcamakta; bu arayış hem gereksiz yakıt tüketimine ve karbon emisyonuna hem de şehir içi trafiğin daha da yoğunlaşmasına neden olmaktadır. Boş yer arayan araçların oluşturduğu ek trafik, yalnızca bireysel zaman kaybıyla sınırlı kalmamakta, tüm ulaşım ağının verimliliğini düşürmektedir.

Bu ihtiyaç, akıllı park sistemleri pazarının dünya genelinde hızla büyümesini beraberinde getirmiştir. Akıllı park sistemleri pazarının önümüzdeki on yıl boyunca yüksek bir bileşik büyüme oranıyla genişlemesi ve birkaç katına ulaşması beklenmektedir [1]. Bu büyüme, park doluluğunu otomatik olarak algılayan, sürücüyü yönlendiren ve park alanlarını verimli yöneten teknolojilere yönelik talebin ne denli güçlü olduğunu açıkça ortaya koymaktadır.

Park doluluğunun otomatik olarak belirlenmesi; dinamik fiyatlandırma, gerçek zamanlı sürücü yönlendirme, ihlal denetimi ve şehir ölçekli park planlaması gibi birçok uygulamanın temelini oluşturmaktadır. Dolayısıyla bu problemin güvenilir, düşük maliyetli ve ölçeklenebilir biçimde çözülmesi, hem bireysel sürücüler hem de şehir yönetimleri açısından doğrudan fayda sağlamaktadır.


1.2 Motivasyon

Park doluluğunu algılamak için yaygın olarak kullanılan yöntemlerden biri, her park yerine gömülü sensör (manyetik, ultrasonik veya döngü bobini) yerleştirmektir. Bu sensörler yüksek doğruluk sunsa da; park yeri başına bir cihaz gerektirdiklerinden kurulum, kablolama ve bakım maliyetleri hızla artmakta, sistem büyüdükçe ölçeklenmesi güçleşmektedir. Ayrıca yola gömülen döngü bobinleri gibi çözümler, kurulum sırasında yol yüzeyinin kazılmasını gerektirdiğinden son derece müdahaleci ve pahalıdır.

Bu sensör tabanlı yaklaşımların önemli bir kısıtı da elektrikli araçların yaygınlaşmasıyla ortaya çıkmaktadır. Hafif gövde yapıları ve verimli tasarımları nedeniyle elektrikli araçlar, manyetik alanı yalnızca sınırlı ölçüde etkiler; bu durum, yalnızca manyetometreye dayalı park sensörlerinin güvenilirliğini düşürmektedir. Buna karşılık kamera tabanlı sistemler, aracın türünden bağımsız olarak görüntüde araç görüldüğü sürece çalıştığından bu sorunu yaşamaz.

Kamera tabanlı yaklaşımın bir diğer üstünlüğü, tek bir kameranın aynı anda çok sayıda park yerini izleyebilmesi ve çoğu durumda mevcut güvenlik kamerası altyapısının yeniden kullanılabilmesidir. Mevcut kameralara bir yapay zekâ yazılım katmanı ekleyerek ek donanım olmaksızın park analizi yapan ticari sistemlerin varlığı, bu yaklaşımın endüstriyel olarak da uygulanabilir olduğunu göstermektedir [2]. Bu tezde geliştirilen sistem de aynı felsefeyi benimsemekte; ek altyapı gerektirmeden yalnızca kamera görüntüsünden çalışan, düşük maliyetli ve taşınabilir bir çözüm hedeflemektedir.


1.3 Amaç ve Kapsam

Bu tezin amacı, tek bir kamera görüntüsünden araçları tespit eden, park yerlerinin doluluğunu ve uygunluğunu gerçek zamanlı çözümleyen ve sürücüyü en uygun boş yere yönlendiren, yapay zekâ tabanlı bütünleşik bir sistem geliştirmektir. Sistem, araç tespiti için derin öğrenme tabanlı, gerçek zamanlı çalışan YOLOv8 nesne tespit modelini [3] kullanmaktadır.

Geliştirilen sistemin kapsamı şu temel yeteneklerden oluşmaktadır:

• Araç tespiti ve sahnedeki araçların takibi,
• Boş park yerlerinin, çizgi durumuna göre ızgara veya geometri tabanlı çalışan adaptif bir yöntemle belirlenmesi,
• Ters perspektif dönüşümü ile kuş bakışı görünüm elde edilmesi ve park yerlerinin gerçek metrik boyutlarının ölçülmesi,
• Aracın boş yere sığıp sığamayacağının ölçülen boyutlar üzerinden denetlenmesi,
• Çok kriterli bir karar motoru ile sürücüye en uygun boş yerin önerilmesi,
• Gece görüşü iyileştirme, monoküler derinlik kestirimi ve çift yönlü çevrimdışı sesli asistan gibi sürücü-asistanı özelliklerinin sunulması.

Çalışma, masaüstü bir uygulama üzerinde, görüntü ve video kaynakları ile gerçek zamanlıya yakın hızda yürütülecek biçimde tasarlanmıştır. Sistemin gece ve düşük ışık gibi zorlu koşullardaki doğruluğunun daha da artırılması ve standart karşılaştırma veri setleri üzerinde geniş ölçekli değerlendirilmesi, ileriye dönük çalışma olarak konumlandırılmıştır.


1.4 Araştırma Soruları

Bu tez, aşağıdaki araştırma sorularına yanıt aramaktadır:

1. Tek bir kamera görüntüsünden, ek donanım kullanmadan, park yerlerinin doluluğu gerçek zamanlı ve güvenilir biçimde belirlenebilir mi?

2. Boyalı şeritlerin bulunduğu ve bulunmadığı park alanlarının her ikisinde de çalışabilen, duruma göre yöntem değiştiren adaptif bir tespit yaklaşımı gerçekleştirilebilir mi?

3. Ters perspektif dönüşümü kullanılarak, park yerlerinin gerçek metrik boyutları yeterli doğrulukta ölçülebilir ve bir aracın boş yere sığıp sığamayacağı otomatik olarak denetlenebilir mi?

4. Boş yer tespitinin ötesinde, sürücüye en uygun yeri öneren çok kriterli bir karar mekanizması tasarlanabilir mi?


1.5 Özgün Katkılar

Literatürdeki çalışmaların büyük çoğunluğu, park yeri analizini yalnızca "dolu/boş" ikili sınıflandırması olarak ele almaktadır. Bu tezde geliştirilen sistem, söz konusu temel yeteneği aşan ve birbirini tamamlayan birden çok katkıyı tek bir gerçek zamanlı işlem hattında birleştirmektedir:

• Adaptif tespit: Sistem, sahnede boyalı şerit bulunup bulunmadığına göre ızgara tabanlı ve geometri tabanlı yöntemler arasında otomatik olarak geçiş yaparak, hem çizgili otoparklarda hem de çizgisiz yol kenarı park alanlarında çalışabilmektedir.

• Gerçek metrik ölçüm ve sığma denetimi: Ters perspektif dönüşümü sayesinde park yerlerinin gerçek boyutları ölçülmekte ve aracın o yere sığıp sığamayacağı denetlenmektedir. Bu yetenek, salt doluluk bilgisinin ötesine geçerek doğrudan sürücüye yönelik somut bir karar sunmaktadır.

• Çok kriterli yönlendirme: Boş yerler arasından, manevra zorluğu, yakınlık ve uygunluk gibi ölçütler birlikte değerlendirilerek en uygun olan sürücüye önerilmektedir.

• Bütünleşik sürücü asistanı: Araç takibi, gece görüşü iyileştirme, monoküler derinlik kestirimi ve çift yönlü çevrimdışı sesli asistan tek bir uygulamada bir araya getirilerek bütüncül bir kullanıcı deneyimi sağlanmaktadır.


1.6 Tezin Organizasyonu

Tezin geri kalanı şu şekilde düzenlenmiştir. İkinci bölümde, park yeri tespiti alanındaki akademik çalışmalar, kullanılan yöntemler, veri setleri ve piyasadaki ticari sistemler incelenerek geliştirilen sistemin literatürdeki yeri ortaya konmaktadır. Üçüncü bölümde, projede kullanılan teknolojiler ve temel yöntemler açıklanmaktadır. Dördüncü bölümde, sistemin mimarisi ve bileşenleri ayrıntılı olarak ele alınmaktadır. Beşinci bölümde, sistemin yazılımsal gerçekleştirimine ilişkin ayrıntılar sunulmaktadır. Altıncı bölümde, deneysel değerlendirme yöntemi ve elde edilen bulgular paylaşılmaktadır. Yedinci bölümde ise genel değerlendirme yapılarak gelecek çalışmalar tartışılmaktadır.
