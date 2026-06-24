POSTER İÇERİĞİ — Resmi 50×70 PowerPoint Şablonuna Yapıştırmak İçin
================================================================
Her bölüm aşağıda hazır metin olarak verildi. PowerPoint şablonundaki ilgili
kutuya doğrudan kopyala-yapıştır yapabilirsin. [KÖŞELİ PARANTEZ] içindekiler
senin dolduracağın kişisel/kurumsal bilgilerdir.


────────────────────────────────────────────────────────────────
PROJE BAŞLIĞI
────────────────────────────────────────────────────────────────
Kamera Görüntülerinden Araç Tespiti ve Park Uygunluğu Analizi için
Yapay Zekâ Tabanlı Sistem

(Kısa alternatif: "Akıllı Park Asistanı: Tek Kameradan Gerçek-Zamanlı Park
Yeri Analizi ve Sürücü Yönlendirme Sistemi")


────────────────────────────────────────────────────────────────
ÖĞRENCİLER / DANIŞMAN  (şablonun üst kısmı)
────────────────────────────────────────────────────────────────
Proje Öğrencisi: Kerem Yağız KARAKAŞ (202213709064)
                 Bilgisayar Mühendisliği — 4. Sınıf
Proje Öğrencisi: Ahmet EKŞİOĞLU (202113709025)
                 Bilgisayar Mühendisliği — 4. Sınıf
Proje Öğrencisi: Ezgi KIRNAPÇI (202113709041)
                 Bilgisayar Mühendisliği — 4. Sınıf
Danışman (Akademik): Prof. Dr. Selçuk KAVUT — Mühendislik Fakültesi
Proje ID: BIT51

(Not: Öğrenci numaralarını istemezsen posterde sadece ad-soyad bırakabilirsin.)


────────────────────────────────────────────────────────────────
PROJE ÖZETİ  (~135 kelime)
────────────────────────────────────────────────────────────────
Bu projede, tek bir kameradan araçları tespit eden ve park alanlarının
uygunluğunu gerçek zamanlı analiz eden yapay zekâ tabanlı bir sistem
geliştirilmiştir. Araç tespiti YOLOv8 ile yapılmış; boş yerler, boyalı şerit
varsa ızgara tabanlı, yoksa geometri tabanlı adaptif yöntemle belirlenmiştir.
Ters Perspektif Dönüşümü (IPM) ile görüntü kuş bakışına çevrilerek çapraz açı
bozulması giderilmiş ve park yerlerinin gerçek metrik boyutları ölçülmüştür.
Sistem, sürücünün aracına uygun en yakın boş yeri çok kriterli bir skorla
önerir; gece görüşü iyileştirme, monoküler derinlik ve çevrimdışı sesli komut
gibi sürücü-asistanı özellikleri içerir. Yöntem, etiketli test görüntüleri ve
sentetik senaryolarla değerlendirilmiş; gündüz ve çizgili park koşullarında
boş/dolu ayrımını yüksek isabetle gerçekleştirmiştir. Sistem, standart bir
bilgisayarda etkileşimli hızda (yaklaşık gerçek-zamanlı) çalışmakta; gece ve
düşük ışık gibi zorlu koşullarda doğruluğun artırılması gelecek çalışma olarak
belirlenmiştir. Geliştirilen çözüm, ek sensör altyapısı gerektirmeden yalnızca
kamera ile düşük maliyetli ve taşınabilir bir akıllı park sistemi sunar.

Anahtar kelimeler: Araç tespiti, Park yeri analizi, Derin öğrenme,
Bilgisayarlı görü, ADAS


────────────────────────────────────────────────────────────────
GİRİŞ / MOTİVASYON
────────────────────────────────────────────────────────────────
Şehir içinde boş park yeri arama; zaman, yakıt ve karbon emisyonu kaybına yol
açar. Mevcut akıllı otopark sistemleri çoğunlukla her yere yerleştirilen
sensörlere dayanır; bu kurulum pahalı ve bakımı zordur. Bu projede ek donanım
gerektirmeden yalnızca kamera ile çalışan, düşük maliyetli ve taşınabilir bir
çözüm hedeflenmiştir.

Literatürde otopark doluluk tespiti çalışmaları bulunsa da; gerçek metrik ölçüm,
sürücüye uygun-yer önerisi ve çizgili/çizgisiz alanlarda çalışan adaptif tespitin
tek bir gerçek-zamanlı sistemde birleştirilmesi bir boşluktur. Bu proje, bu üç
yeteneği bir araya getirerek hem akademik hem pratik katkı sunar.


────────────────────────────────────────────────────────────────
SİSTEM MİMARİSİ  (Görsel: mimari akış diyagramı — sistem_mimarisi.png)
────────────────────────────────────────────────────────────────
Sistem, ardışık modüllerden oluşan bir işlem hattıdır:

Kamera / Video Girişi
   ↓
Gece Görüşü İyileştirme (CLAHE düşük-ışık kontrastı)
   ↓
YOLOv8 Araç Tespiti  →  Sınıf-bağımsız NMS (çift tespit eleme)
   ↓
Araç Takibi (ego-hareket telafili)
   ↓
Adaptif Boş-Yer Tespiti
   ├─ Çizgi varsa → Izgara tabanlı (Hough + zamansal füzyon)
   └─ Çizgi yoksa → Geometri tabanlı (boşluk analizi)
   ↓
IPM Kuş Bakışı Dönüşümü (+ otomatik kalibrasyon)  →  Gerçek metrik ölçüm
   ↓
Sığma Kontrolü  +  Çok-Kriterli Slot Öneri Motoru
   ↓
Arayüz: Canlı Görüntü + 2B Şematik Harita + Yönlendirme + Sesli Asistan

(Not: Bu akışın temiz görseli sistem_mimarisi.png dosyasında; "SİSTEM MİMARİSİ"
kutusuna bu PNG yerleştirilecek.)


────────────────────────────────────────────────────────────────
MATERYAL VE YÖNTEM
────────────────────────────────────────────────────────────────
Geliştirme: Python, PyQt5 arayüz, OpenCV ve NumPy görüntü işleme. Temel
yöntemler:

• Araç Tespiti: YOLOv8 ile gerçek-zamanlı tespit; sınıfa göre gerçek boyut
  ataması ve sınıf-bağımsız NMS ile çift-tespit eleme.

• Adaptif Boş-Yer Tespiti: Şerit varsa ızgara tabanlı (Hough + alt-piksel
  hassaslaştırma + zamansal füzyon); şerit yoksa geometri tabanlı boşluk analizi.

• Ters Perspektif Dönüşümü (IPM): Zemin homografisiyle kuş bakışı; manuel
  4-nokta veya kaybolma-noktası/araç tabanlı otomatik kalibrasyon. Kuş bakışında
  ölçek sabit olduğundan gerçek metrik ölçüm yapılır.

• Kararlılık: Zamansal oylama ve video sabitleme ile canlı görüntüde tutarlılık.

• Sürücü Asistanı: Çok-kriterli slot skoru, monoküler derinlik (MiDaS), çevrimdışı
  sesli komut (Vosk) ve gece görüşü.

• Değerlendirme: metrik kütüphanesi (P/R/F1/mAP), sentetik veri üreteci ve
  ablation çerçevesi; 211 otomatik test.


────────────────────────────────────────────────────────────────
KULLANILAN TEKNOLOJİLER  (logolar + isim)
────────────────────────────────────────────────────────────────
Python · PyTorch / Ultralytics YOLOv8 · OpenCV · PyQt5 · NumPy ·
MiDaS (derinlik) · Vosk (sesli komut) · Matplotlib · pytest

(Bu kutuya yukarıdaki teknolojilerin logoları + isimleri yerleştirilecek.)


────────────────────────────────────────────────────────────────
BULGULAR  (Görseller: uygulama ekran görüntüleri)
────────────────────────────────────────────────────────────────
Sistemin temel yeteneklerini gösteren uygulama görüntüleri:

• Boş/dolu slot tespiti — yeşil (boş) ve kırmızı (dolu) kutular, slot sayımı
• Kuş bakışı (IPM) modu — gerçek metrik ölçümle slot boyutları
• Sığma kontrolü — SIGAR/SIGMAZ etiketi + metre cinsinden alan genişliği
• 2B Şematik Park Haritası — dijital ikiz görünümü
• Gece görüşü — CLAHE öncesi/sonrası karşılaştırma

(Görseller — en net, gündüz, geniş açı çekimler tercih edilmeli; jürinin
 uzaktan rahat görebileceği 2-3 ekran görüntüsü yan yana / alt alta.
 Değerlendirme grafikleri ve metrik tablosu tez metninde yer alacak.)


────────────────────────────────────────────────────────────────
SONUÇ VE ÖNERİLER
────────────────────────────────────────────────────────────────
Tek kameradan, ek donanım gerektirmeden park yeri analizi, metrik ölçüm ve
sürücü yönlendirmesi gerçekleştirilmiştir. Adaptif tespit yaklaşımı sayesinde
sistem hem çizgili otoparklarda hem de çizgisiz yol kenarı park alanlarında
çalışabilmektedir. Gündüz ve çizgili koşullarda boş/dolu ayrımı yüksek isabetle
yapılmakta; sistem etkileşimli hızda çalışmaktadır.

İleriye dönük öneriler:
• Kendi verisiyle eğitilen doluluk sınıflandırıcı (CNN) ile gece/zor koşullarda
  doğruluğun artırılması,
• İşlem hattının hızlandırılması ile yüksek kare hızına ulaşılması,
• Çoklu kamera füzyonu ile tüm otoparkın tek haritada birleştirilmesi,
• Gerçek veri seti (PKLot) üzerinde ince ayar ve karşılaştırmalı değerlendirme.


────────────────────────────────────────────────────────────────
KAZANIMLAR
────────────────────────────────────────────────────────────────
• Bilgisayarlı görü ve derin öğrenme (YOLO) uygulama deneyimi
• Geometrik bilgisayarlı görü: homografi / ters perspektif dönüşümü
• Gerçek-zamanlı sistem tasarımı ve performans optimizasyonu
• Yazılım mühendisliği disiplini: modüler mimari ve 211 otomatik test
• Bilimsel değerlendirme: metrik, ablation ve veri seti tabanlı analiz


────────────────────────────────────────────────────────────────
GÖRSEL YERLEŞİM PLANI  (hangi görsel hangi bölüme)
────────────────────────────────────────────────────────────────
SİSTEM MİMARİSİ kutusu:
  • sistem_mimarisi.png  (yeni ürettiğim akış diyagramı — sol büyük kutu)

BULGULAR kutusu (2-3 görsel, yan yana / alt alta):
  • Boş/dolu slot tespiti çalışırken — renkli kutular + sayım
  • Kuş bakışı (IPM) modu + metrik slot boyutu
  • Sığma kontrolü (SIGAR/SIGMAZ + metre) veya Park Haritası görünümü
  Not: En net, gündüz, geniş açı çekimleri seç. ground truth çıktıları
       da kullanılabilir: 2_out.png, 3_out.png.
  (Değerlendirme grafikleri — P/R/F1, confusion matrix, ablation — tez metnine.)

KULLANILAN TEKNOLOJİLER kutusu:
  • Python, YOLO/Ultralytics, OpenCV, PyQt5, NumPy, MiDaS, Vosk logoları

GENEL TASARIM İPUÇLARI:
  • Görsel ağırlıklı olsun; metni kısa madde/cümlelerle ver.
  • Başlık ve banner'ları şablondaki gibi koru.
  • Renk: koyu lacivert + vurgular (yeşil=boş, kırmızı=dolu) — sistemle tutarlı.
  • Yazı boyutu: başlıklar büyük, gövde en az 24-28 pt (uzaktan okunmalı).


────────────────────────────────────────────────────────────────
SENİN DOLDURMAN / SEÇMEN GEREKENLER
────────────────────────────────────────────────────────────────
1. Öğrenci ad-soyad(ları) + bölüm/sınıf
2. Danışman ünvan + ad-soyad + fakülte
3. Proje ID (BIT__)
4. Bulgular/Giriş için 1-2 UI ekran görüntüsü (en iyi gündüz kareleri)
5. Teknoloji logoları (internetten temin)

