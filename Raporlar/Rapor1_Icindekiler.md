# İÇİNDEKİLER

(Bu dosyayı Word'e kopyalayıp sayfa numaralarını kendin gir)

---

ÖZET ........................................................ i

ABSTRACT .................................................... ii

ÖNSÖZ ....................................................... iii

İÇİNDEKİLER ................................................. iv

ŞEKİL LİSTESİ ............................................... v

TABLO LİSTESİ ............................................... vi

KISALTMALAR VE SİMGELER LİSTESİ ............................ vii

---

1. GİRİŞ .................................................... 1

   1.1 Projenin Konusu ve Amacı ............................... 1

   1.2 Neden Bu Problem Önemli? ............................... 2

   1.3 Motivasyon ............................................. 3

   1.4 Sistemin Genel Yapısı .................................. 4

   1.5 Araştırma Sorusu ve Hipotez ............................ 5

   1.6 Projenin Kapsamı ....................................... 6

   1.7 Projenin Özgün Katkısı ................................. 7

   1.8 Raporun Yapısı ......................................... 7

2. LİTERATÜR ARAŞTIRMASI .................................... 8

   2.1 Problem Nasıl Tanımlanıyor? ............................ 8

   2.2 Tarihsel Gelişim: Derin Öğrenmeden Öncesi .............. 8

   2.3 Derin Öğrenme ile Gelen İki Ana Yaklaşım ............... 9

   2.4 Literatürdeki Önemli Çalışmalar ....................... 10

   2.5 Önemli Bir Sorun: Geçici Araç mı, Park Edilmiş Araç mı? 13

   2.6 YOLO Versiyonları Karşılaştırması ..................... 14

   2.7 Benchmark Veri Setleri ................................ 14

   2.8 Sensör Tabanlı vs Kamera Tabanlı Sistemler ............ 15

   2.9 Piyasadaki Ticari Sistemler ........................... 16

   2.10 Projemizin Literatürdeki Yeri ........................ 17

   2.11 Literatür Özeti Tablosu .............................. 18

3. SİSTEM MİMARİSİ .......................................... 19

   3.1 Genel Bakış ........................................... 19

   3.2 Klasör Yapısı ......................................... 20

   3.3 Bileşen 1: Araç Tespiti (VehicleDetector) ............. 21

   3.4 Bileşen 2: Park Alanı Etiketleme (ZoneAnnotator) ..... 22

   3.5 Bileşen 3: Park Alanı Yükleme ve IoU Hesabı .......... 23

   3.6 Bileşen 4: Karar Motoru ............................... 24

   3.7 Bileşen 5: Sentetik Veri Üretici ...................... 24

   3.8 Bileşen 6: YOLOv8 Fine-Tuning ........................ 25

   3.9 Bileşen 7: Arayüz (MainWindow) ....................... 26

   3.10 Veri Akışı: Uçtan Uca Örnek ......................... 27

   3.11 Tasarım Kararları .................................... 27

4. KULLANILAN TEKNOLOJİLER .................................. 28

   4.1 Programlama Dili: Python .............................. 28

   4.2 Nesne Tespiti: YOLOv8 (Ultralytics) .................. 28

   4.3 Görüntü İşleme: OpenCV ................................ 30

   4.4 Derin Öğrenme Altyapısı: PyTorch ...................... 31

   4.5 Görsel Arayüz: PyQt5 ................................. 31

   4.6 Model Değerlendirme: scikit-learn ..................... 32

   4.7 Veri İşleme: NumPy ve Pandas .......................... 33

   4.8 Grafik ve Görselleştirme: Matplotlib ve Seaborn ....... 33

   4.9 Konfigürasyon: PyYAML ................................. 34

   4.10 Teknoloji Özet Tablosu ............................... 34

   4.11 Geliştirme Ortamı .................................... 35

5. UYGULAMA VE BULGULAR ..................................... 36

   5.1 Geliştirme Süreci ..................................... 36

   5.2 Baseline Sistem Sonuçları ............................. 37

   5.3 Sentetik Veri Üretimi ................................. 38

   5.4 Görsel Arayüz ......................................... 39

   5.5 Park Alanı Etiketleme Aracı ........................... 40

   5.6 IoU Tabanlı Doluluk Analizi ........................... 41

   5.7 Mevcut Sistem Sınırlamaları ........................... 42

   5.8 Planlanan Ölçümler .................................... 43

6. SONUÇ VE GELECEK ÇALIŞMALAR ............................. 44

   6.1 Genel Değerlendirme ................................... 44

   6.2 Literatürle Karşılaştırma ............................. 45

   6.3 Karşılaşılan Zorluklar ................................ 45

   6.4 Gelecek Çalışmalar .................................... 46

   6.5 Projenin Potansiyeli .................................. 47

   6.6 Özet .................................................. 47

KAYNAKLAR ................................................... 48
