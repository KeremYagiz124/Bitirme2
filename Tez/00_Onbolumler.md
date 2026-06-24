TEZ ÖN BÖLÜMLERİ
================================================================
Bu dosya, tezin ön kısımlarını içerir: İçindekiler, Kısaltma Listesi, Şekil
Listesi ve Tablo Listesi. Kılavuz (Ek 1) uyarınca nihai belgedeki sıralama
şöyledir: Dış Kapak → İç Kapak → İçindekiler → Kısaltma Listesi → Şekil Listesi
→ Tablo Listesi → Önsöz → Özet → Abstract → Giriş → Ana Metin → Sonuç →
Kaynaklar. Ön sayfalar İçindekiler "ii" olacak şekilde Romen rakamlarıyla
numaralandırılır; sayfa numaraları belge Word'e aktarılırken doldurulacaktır.


────────────────────────────────────────────────────────────────
İÇİNDEKİLER
────────────────────────────────────────────────────────────────
                                                                  Sayfa

İÇİNDEKİLER ............................................... ii
KISALTMA LİSTESİ ......................................... iv
ŞEKİL LİSTESİ ............................................ v
TABLO LİSTESİ ............................................ vi
ÖNSÖZ .................................................... vii
ÖZET ..................................................... viii
ABSTRACT ................................................. ix

1. GİRİŞ
   1.1 Problemin Tanımı ve Önemi
   1.2 Motivasyon
   1.3 Amaç ve Kapsam
   1.4 Araştırma Soruları
   1.5 Özgün Katkılar
   1.6 Tezin Organizasyonu

2. LİTERATÜR ARAŞTIRMASI
   2.1 Problemin Tanımı
   2.2 Derin Öğrenme Öncesi Yöntemler
   2.3 Derin Öğrenme ile Gelen İki Ana Paradigma
   2.4 Literatürdeki Önemli Çalışmalar
   2.5 Geçici Araç ve Park Edilmiş Araç Ayrımı
   2.6 YOLO Sürümleri ve Araç Tespiti
   2.7 Benchmark Veri Setleri
   2.8 Sensör Tabanlı ve Kamera Tabanlı Sistemler
   2.9 Piyasadaki Ticari Sistemler
   2.10 Projenin Literatürdeki Yeri

3. KULLANILAN TEKNOLOJİLER VE YÖNTEMLER
   3.1 Programlama Dili: Python
   3.2 Derin Öğrenme Altyapısı: PyTorch
   3.3 Nesne Tespiti: YOLOv8
   3.4 Görüntü İşleme: OpenCV
   3.5 Sayısal Hesaplama: NumPy
   3.6 Görsel Arayüz: PyQt5
   3.7 Monoküler Derinlik Kestirimi: MiDaS
   3.8 Sesli Etkileşim: Vosk ve Sinirsel Metinden Sese Dönüşüm
   3.9 Sürülebilir Alan Segmentasyonu: YOLOPv2
   3.10 Ters Perspektif Dönüşümü ve Homografi
   3.11 Değerlendirme ve Test Yığını

4. SİSTEM TASARIMI VE MİMARİSİ
   4.1 Genel İşlem Hattı
   4.2 Gece Görüşü İyileştirme
   4.3 Araç Tespiti ve Sınıf-Bağımsız Bastırma
   4.4 Araç Takibi
   4.5 Adaptif Boş-Yer Tespiti
   4.6 Ters Perspektif Dönüşümü ve Otomatik Kalibrasyon
   4.7 Gerçek Metrik Ölçüm ve Ölçek Kestirimi
   4.8 Sığma Kontrolü
   4.9 Çok Kriterli Slot Öneri Motoru
   4.10 Öğrenilen Slot Belleği ve Şematik Park Haritası
   4.11 Monoküler Derinlik Entegrasyonu
   4.12 Sesli Asistan
   4.13 Tasarım Kararları ve Gerekçeleri

5. GERÇEKLEŞTİRİM
   5.1 Yazılım Mimarisi ve Modülerlik
   5.2 Gerçek Zamanlı İşlem Hattı ve İş Parçacığı Yönetimi
   5.3 Kararlılık Teknikleri
   5.4 Perspektif Düzeltme ve Yan Görüş Ele Alımı
   5.5 Kullanıcı Arayüzü
   5.6 Yapılandırma ve Genişletilebilirlik

6. DENEYSEL DEĞERLENDİRME VE BULGULAR
   6.1 Değerlendirme Yöntemi
   6.2 Kullanılan Metrikler
   6.3 Sentetik Senaryo Sonuçları
   6.4 Ablasyon Çalışması
   6.5 Performans
   6.6 Yazılım Kalitesi
   6.7 Niteliksel Sonuçlar
   6.8 Koşula Bağlı Davranış ve Sınırlamalar

7. SONUÇ VE GELECEK ÇALIŞMALAR
   7.1 Genel Değerlendirme
   7.2 Literatürle Karşılaştırma
   7.3 Karşılaşılan Zorluklar
   7.4 Gelecek Çalışmalar
   7.5 Kapanış

KAYNAKLAR


────────────────────────────────────────────────────────────────
SİMGELER LİSTESİ
────────────────────────────────────────────────────────────────

H       : 3x3 Boyutunda Homografi Matrisi
s       : Homojen Koordinat Normalizasyon Ölçek Katsayısı
s_mp    : Metrik Piksel Ölçeği (metre/piksel)
G       : Kalibrasyon Dikdörtgeninin Gerçek Genişliği (metre)
Y       : Kalibrasyon Dikdörtgeninin Gerçek Yüksekliği (metre)
g       : Kuş Bakışı Görüntüdeki Kalibrasyon Genişliği (piksel)
y       : Kuş Bakışı Görüntüdeki Kalibrasyon Yüksekliği (piksel)
w       : Park Slotunun Gerçek Metrik Genişliği (metre)
w_ref   : Referans Alınan Standart Araç Genişliği (metre)
Z       : Park Manevra Zorluğu Skoru (0-100)
S       : Çok Kriterli Slot Öneri Skoru (0-100)
z_1     : Normalize Edilmiş Manevra Zorluğu Kriteri
z_2     : Normalize Edilmiş Mesafe Yakınlığı Kriteri
z_3     : Normalize Edilmiş Genişlik Marjı Kriteri
z_4     : Normalize Edilmiş Çıkışa Yakınlık Kriteri
A, B    : Takip veya Karşılaştırma Yapılan Sınırlayıcı Kutular (Bounding Boxes)

────────────────────────────────────────────────────────────────
KISALTMA LİSTESİ
────────────────────────────────────────────────────────────────

ADAS   : Gelişmiş Sürücü Destek Sistemleri (Advanced Driver Assistance Systems)
AP     : Ortalama İsabet (Average Precision)
BEV    : Kuş Bakışı Görünüm (Bird's Eye View)
CLAHE  : Kontrast Sınırlı Uyarlamalı Histogram Eşitleme
CNN    : Evrişimli Sinir Ağı (Convolutional Neural Network)
COCO   : Common Objects in Context (veri seti)
CPU    : Merkezi İşlem Birimi (Central Processing Unit)
F1     : F1 Skoru (kesinlik ve duyarlılığın harmonik ortalaması)
FN     : Yanlış Negatif (False Negative)
FP     : Yanlış Pozitif (False Positive)
FPS    : Saniyedeki Kare Sayısı (Frames Per Second)
GPU    : Grafik İşlem Birimi (Graphics Processing Unit)
HOG    : Yönlü Gradyan Histogramları (Histogram of Oriented Gradients)
HSV    : Renk-Doygunluk-Parlaklık renk uzayı (Hue-Saturation-Value)
IoU    : Kesişimin Birleşime Oranı (Intersection over Union)
IPM    : Ters Perspektif Dönüşümü (Inverse Perspective Mapping)
LAB    : Parlaklık-renk (L*a*b*) renk uzayı
LK     : Lucas-Kanade (optik akış yöntemi)
mAP    : Ortalama İsabet Ortalaması (mean Average Precision)
MiDaS  : Monoküler derinlik kestirim modeli
NMS    : Maksimum-Olmayan Bastırma (Non-Maximum Suppression)
ORB    : Oriented FAST and Rotated BRIEF (öznitelik tanımlayıcı)
RANSAC : Rastgele Örnek Uzlaşması (Random Sample Consensus)
ROI    : İlgi Bölgesi (Region of Interest)
SSD    : Single Shot MultiBox Detector
STT    : Konuşmadan Metne (Speech to Text)
SVM    : Destek Vektör Makinesi (Support Vector Machine)
TP     : Doğru Pozitif (True Positive)
TTS    : Metinden Sese (Text to Speech)
YOLO   : You Only Look Once (nesne tespit mimarisi)


────────────────────────────────────────────────────────────────
ŞEKİL LİSTESİ
────────────────────────────────────────────────────────────────
(Görseller nihai belge hazırlanırken ilgili bölümlere yerleştirilecektir.)

   Şekil 3.1. Sürülebilir alan (yol) maskesi
   Şekil 3.2. Gece görüşü: CLAHE öncesi/sonrası
   Şekil 4.1. Sistemin uçtan uca işlem hattı
   Şekil 4.2. YOLOv8 ile araç tespiti
   Şekil 4.3. Geometri tabanlı boş/dolu yer tespiti
   Şekil 4.4. Kuş bakışı (IPM) görünümü ve metrik ölçüm
   Şekil 4.5. Sığma kontrolü (sığar/sığmaz ve alan genişliği)
   Şekil 5.1. Uygulama arayüzünün genel görünümü
   Şekil 6.1. Sentetik değerlendirme karışıklık matrisi
   Şekil 6.2. Sentetik değerlendirme metrik çubuk grafiği


────────────────────────────────────────────────────────────────
TABLO LİSTESİ
────────────────────────────────────────────────────────────────

Tablo 6.1. Yapılandırma karşılaştırması (ablasyon)
