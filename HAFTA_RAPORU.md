Proje Adı: Kamera Görüntülerinden Araç Tespiti ve Park Uygunluğu Analizi için Yapay Zeka Tabanlı Sistem
Rapor 7


1. DİK PARK İÇİN DESTEK EKLENDİ

Otopark doluluğunu görüntüden tespit eden çalışmalar literatürde derin öğrenme ile yapılmaktadır [1]. Bu proje de aynı yaklaşımı temel almakta, ancak farklı park düzenlerini desteklemeyi hedeflemektedir.

Sistem bu rapora kadar yalnızca paralel parkı tanıyordu, yani araçların yol kenarına yan yana dizildiği sokak parkını. Bu hafta dik park desteği de eklendi. Dik park, araçların yola dik şekilde park ettiği otoparklardaki düzendir.

Bunun için araç tespitinde kullanılan YOLO [2] sisteminin bulduğu araç kutularını dik park mantığına göre yeniden değerlendiren bir yapı kuruldu. Araç tespit modülüne "yön" adında bir ayar eklendi: paralel ya da dik. Kullanıcı arayüzdeki butondan hangi park türünü analiz etmek istediğini seçmekte, sistem de buna göre çalışmaktadır.

Paralel modda araçlar yatay (geniş) görünür, bu nedenle çok ince kutular araç olarak sayılmaz. Dik modda ise araçlar hem önden hem yandan görünebildiği için bu filtre gevşetildi; böylece dik duran araçlar da gözden kaçırılmaz.


2. FOTOĞRAFIN AÇISI OTOMATİK OLARAK BELİRLENİYOR

Dik park fotoğrafları çekim açısına göre iki türlü gelebilmektedir: araçların önüne bakan açı (ön görünüm) ya da yanına bakan açı (yan görünüm). Bu iki durum farklı hesaplama gerektirdiği için sistemin açıyı kendi kendine belirlemesi gerekmekteydi.

Bu sorun şöyle çözüldü: fotoğraftaki araç kutularının ortalama en-boy oranına bakılmaktadır. Kutular genişse (oran 1.6'dan büyük) araçlara yandan bakıldığı, kutular kareye yakınsa araçlara önden bakıldığı anlaşılmaktadır. Karar sistem tarafından otomatik verilmekte ve sonuç arayüzde "Yan gorunum aktif" veya "On gorunum aktif" yazısıyla kullanıcıya gösterilmektedir. Böylece kullanıcı sistemin neye göre ölçüm yaptığını görebilmektedir.


3. GERÇEK BOYUT DOĞRU ŞEKİLDE HESAPLANIYOR

Sistem bir alanın kaç metre olduğunu doğrudan bilemez; bu değer, fotoğraftaki park etmiş araçların piksel boyutunun bilinen gerçek bir ölçüyle kıyaslanmasıyla tahmin edilmektedir. Kullanılan ölçü, fotoğrafın açısına göre değişmektedir:

- Paralel parkta ve dik-yan görünümde: araç uzunluğu (yaklaşık 4.5 metre) referans alınır, çünkü bu açılarda araç kutusunun genişliği aracın uzunluğuna denk gelir.
- Dik-ön görünümde: araç eni (yaklaşık 2.0 metre) referans alınır, çünkü önden bakıldığında araç kutusunun genişliği aracın enine denk gelir.

Önemli bir düzeltme olarak boş alanların parçalara bölünmesi uygulamasından vazgeçildi. Önceki sürümde büyük bir boş alan araç eni kadar parçalara bölünüyor ve her parça ayrı ayrı "sığmaz" olarak gösteriliyordu. Artık dik modda boş alan tek parça olarak ölçülmekte ve kullanıcının aracının o alana girip giremeyeceğine toplam genişliğe göre karar verilmektedir.


4. ARAYÜZ KULLANIMI KOLAYLAŞTIRILDI

Araç boyutu giriş kutuları seçilen moda göre değiştirildi. Paralel moddayken kullanıcı "araç uzunluğu" girmekte, dik moda geçildiğinde bu kutu gizlenip yerine "araç eni" kutusu çıkmaktadır. Böylece kullanıcıya yalnızca o an işine yarayan bilgi gösterilmekte, karışıklık önlenmektedir.

Boş alanların ekrana çizilmesi, etiketlerin yazılması ve fotoğrafların kaydedilmesi gibi tüm görüntü işlemlerinde OpenCV [3] kütüphanesi kullanıldı. Ayrıca aracın sığıp sığmadığının kontrol edildiği tüm yerler (ekrandaki çizim, etiket yazısı, uyarı mesajı ve kaydedilen fotoğraf) tek tip hale getirildi: dik modda araç eni, paralel modda araç uzunluğu kullanılmaktadır. Bu sayede hangi ekrana bakılırsa bakılsın aynı sonuç görülmektedir.


5. YENİ TESTLER YAZILDI

Eklenen dik park özelliğinin doğru çalıştığından emin olmak için test paketine on yeni test eklendi. Bu testler şunları kontrol etmektedir:

- Sistem fotoğrafın açısını (ön mü yan mı) doğru belirliyor mu
- Dik modda boş alan tek parça mı kalıyor, paralel modda parçalara bölünüyor mu
- Gerçek boyut hesabında doğru referans (uzunluk ya da en) kullanılıyor mu
- Yeni fotoğraf yüklendiğinde sistem önceki durumu sıfırlıyor mu

Toplam test sayısı 51'den 61'e çıktı ve testlerin tamamı başarıyla geçmektedir.


6. SİSTEMİN BİR SINIRI: ÇOK EĞİK AÇILAR

Sistem fotoğraf üzerindeki araç kutuları üzerinden çalışmakta, yani iki boyutlu bir analiz yapmaktadır. Kamera tam karşıdan ya da yukarıdan bakmadığında, çok eğik açılarda, aynı hizada ama farklı uzaklıkta duran nesneler görüntüde üst üste binebilmektedir. Bu durumda boş alan ölçümü tam doğru olmamakta, yaklaşık kalmaktadır.

Bu sorunun tamamen çözülmesi için fotoğrafı kuş bakışına çeviren bir dönüşüm ya da her pikselin uzaklığını tahmin eden ayrı bir derinlik modeli gerekmektedir. Her ikisi de kamera kalibrasyonu veya ağır ek modeller gerektirdiğinden bu projenin kapsamı dışındadır. Sistem düz ya da yukarıdan çekilmiş fotoğraflarda yüksek doğrulukla, çok eğik açılı fotoğraflarda ise yaklaşık sonuçla çalışmaktadır. Bu durum bir eksiklik olarak değil, bilinen bir çalışma koşulu olarak not edilmiştir.


Rapor Özeti

Bu hafta sisteme dik park desteği eklendi. Artık hem paralel hem dik park alanları analiz edilebilmektedir. Sistem fotoğrafın açısını kendisi belirlemekte, gerçek boyutu doğru referansla hesaplamakta ve boş alanı bölmeden tek parça değerlendirmektedir. Arayüz seçilen moda göre sadeleştirildi, on yeni testle birlikte test sayısı 61'e ulaştı. Çok eğik açılı fotoğraflardaki ölçüm sınırı da raporda açıkça belirtildi.


Kullanılan Kaynaklar
[1] Amato, G. ve diğerleri (2017). Deep Learning for Decentralized Parking Lot Occupancy Detection. Expert Systems with Applications, 72, 327–334.
[2] Jocher, G., Chaurasia, A. ve Qiu, J. (2023). Ultralytics YOLOv8 https://github.com/ultralytics/ultralytics
[3] Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools, 25(11), 120–125.
