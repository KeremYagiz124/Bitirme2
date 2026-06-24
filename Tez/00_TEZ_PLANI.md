TEZ PLANI VE YAZIM PROGRAMI
================================================================
Bu dosya, bitirme tezinin bölüm yapısını, yazım sırasını, atıf yönetimini ve
kalite kurallarını tanımlar. Her bölüm metni `Tez/` klasörü altında ayrı bir
dosyada hazırlanır. Atıflar, metin içinde ortaya çıkış sırasına göre numaralanır
([1], [2], [3] ...) ve tüm kaynaklar `KAYNAKCA.md` dosyasında toplanır.


────────────────────────────────────────────────────────────────
1. YAZIM SIRASI (atıf numaralarının sıralı çıkması için)
────────────────────────────────────────────────────────────────
Atıfların metinde sırayla ([1], [2], ...) görünmesi gerektiğinden, bölümler
NİHAİ BELGE SIRASINA göre yazılır. Özet ve Abstract atıf içermez (akademik
konvansiyon); böylece ilk atıf Giriş bölümünde başlar ve numaralandırma bozulmaz.

Yazım/üretim sırası:
  00. Plan (bu dosya)                          → 00_TEZ_PLANI.md          [bitti]
  01. Özet ve Abstract (atıfsız)               → 01_Ozet_Abstract.md
  02. Giriş                                    → 02_Giris.md
  03. Literatür Araştırması                    → 03_Literatur.md
  04. Kullanılan Teknolojiler ve Yöntemler     → 04_Teknolojiler_Yontemler.md
  05. Sistem Tasarımı ve Mimarisi              → 05_Sistem_Mimarisi.md
  06. Gerçekleştirim (Uygulama Detayları)      → 06_Gerceklestirim.md
  07. Deneysel Değerlendirme ve Bulgular       → 07_Bulgular.md
  08. Sonuç ve Gelecek Çalışmalar              → 08_Sonuc.md
  09. Kaynakça (sürekli güncellenir)           → KAYNAKCA.md

Not: Atıf numaraları, bölümler bu sırayla yazıldıkça KAYNAKCA.md'ye eklenir. Her bölüm yazılırken bir sonraki boş atıf numarasından devam edilir.


────────────────────────────────────────────────────────────────
2. ATIF YÖNETİMİ (kritik kural)
────────────────────────────────────────────────────────────────
• Atıf biçimi: "... YOLOv8 mimarisi [1] kullanılmıştır." (köşeli parantez, cümle
  içinde, ilgili teknoloji/çalışma adının hemen ardından).
• Numaralandırma: metinde İLK kez geçtiği yerde numara alır; sonraki geçişlerde
  aynı numara tekrar kullanılır (yeni numara verilmez).
• Sıra: [1] ilk atıf, [2] ikinci atıf ... atlamasız ve geri sıçramasız ilerler.
• Her yeni atıf, aşağıdaki "ATIF SİCİLİ"ne ve KAYNAKCA.md'ye eş zamanlı işlenir.

ATIF SİCİLİ (kullanıldıkça doldurulur — bir sonraki numara: [26])
  [1] Spherical Insights — akıllı park pazarı raporu     — Giriş 1.1
  [2] Parquery — kamera tabanlı ticari sistem            — Giriş 1.2 / Lit 2.9
  [3] YOLOv8 / Ultralytics (Jocher vd.)                  — Giriş 1.3 / Lit 2.6
  [4] de Almeida vd. 2022 — sistematik derleme           — Lit 2.1
  [5] Dalal & Triggs 2005 — HOG                          — Lit 2.2
  [6] Duda & Hart 1972 — Hough dönüşümü                  — Lit 2.2
  [7] de Almeida vd. 2015 — PKLot                        — Lit 2.4.1 / 2.7
  [8] Amato vd. 2017 — mAlexNet / CNRPark-EXT            — Lit 2.4.2 / 2.7
  [9] Nurullayev & Lee 2019 — CarNet                     — Lit 2.4.3
  [10] Yuldashev vd. 2023 — MobileNetV3+CBAM             — Lit 2.4.4
  [11] Grbić & Koch 2023 — APSD-OC                       — Lit 2.4.5 / 2.10
  [12] Nguyen & Sartipi 2024 — PakLoc/PakSta             — Lit 2.4.6
  [13] da Luz vd. 2024 — YOLOv8-v11 + piksel ROI         — Lit 2.4.7 / 2.6 / 2.10
  [14] Xie vd. 2017 — yasadışı park (SSD)                — Lit 2.4.8
  [15] Bewley vd. 2016 — SORT                            — Lit 2.5
  [16] Wojke vd. 2017 — DeepSORT                         — Lit 2.5
  [17] Zhang vd. 2022 — ByteTrack                        — Lit 2.5 / 2.10
  [18] Sharma vd. 2023 — YOLOv8 + takip, süre ihlali     — Lit 2.5
  [19] Redmon vd. 2016 — YOLO (You Only Look Once)       — Lit 2.6
  [20] Zhang vd. 2018 — PS2.0 veri seti                  — Lit 2.7
  [21] Quercus Technologies                              — Lit 2.9
  [22] Bosch Mobility                                    — Lit 2.9
  [23] Metropolis                                        — Lit 2.9
  [24] Hikvision                                         — Lit 2.9
  [25] Dahua Technology                                  — Lit 2.9

  [26] Python                                            — Tek 3.1
  [27] PyTorch (Paszke vd.)                              — Tek 3.2
  [28] COCO (Lin vd.)                                    — Tek 3.3
  [29] OpenCV (Bradski)                                  — Tek 3.4
  [30] Canny 1986 — kenar tespiti                        — Tek 3.4
  [31] Zuiderveld 1994 — CLAHE                           — Tek 3.4
  [32] NumPy (Harris vd.)                                — Tek 3.5
  [33] PyQt5 (Riverbank)                                 — Tek 3.6
  [34] Ranftl vd. 2022 — MiDaS                           — Tek 3.7
  [35] Vosk (Alpha Cephei)                               — Tek 3.8
  [36] edge-tts (Microsoft)                              — Tek 3.8
  [37] Han vd. 2022 — YOLOPv2                            — Tek 3.9
  [38] Hartley & Zisserman 2004 — homografi              — Tek 3.10
  [39] Hunter 2007 — Matplotlib                          — Tek 3.11
  [40] pytest (Krekel vd.)                               — Tek 3.11

  [41] Lucas & Kanade 1981 — optik akış (ego-hareket)    — Mim 4.4
  [42] Shi & Tomasi 1994 — köşe tespiti                   — Mim 4.4
  (Mimaride tekrar kullanılanlar: [3] YOLOv8, [6] Hough, [30] Canny,
   [31] CLAHE, [34] MiDaS, [35] Vosk, [36] edge-tts, [38] homografi.)

  [43] Rublee vd. 2011 — ORB öznitelik                   — Ger 5.3
  [44] Fischler & Bolles 1981 — RANSAC                    — Ger 5.3

  07_Bulgular: yeni atıf YOK (yalnızca [7] PKLot tekrar kullanıldı).
  Gerçek değerler kullanıldı (outputs/): sentetik 221 TP / 0 FP / 47 FN →
  P=1.00, R=0.82, F1=0.90, AP=0.82; ablasyon Temel F1=0.94 / Agresif F1=0.67 /
  Muhafazakâr F1=0.90; otomatik test sayısı 268 (pytest --collect-only).
  NOT: POSTER'da 211 yazıyor → güncel değil (268). Kullanıcıya bildirildi.

  [45] Sandler vd. 2018 — MobileNetV2 (gelecek çalışma)  — Sonuç 7.4
  (Sonuç'ta tekrar kullanılanlar: [2] Parquery, [7] PKLot, [11] APSD-OC,
   [13] da Luz, [17] ByteTrack.)

  TÜM İÇERİK BÖLÜMLERİ TAMAM. Toplam atıf: 45.
  Sıradaki: önbölümler (İçindekiler, Şekil/Tablo Listesi, Kısaltmalar) +
  tüm tezin atıf sırası/tutarlılık denetimi.


────────────────────────────────────────────────────────────────
3. SİSTEMİN GERÇEK KAPSAMI (tez bu güncel duruma göre yazılır)
────────────────────────────────────────────────────────────────
Önemli: Eski raporlar (Raporlar/Rapor1_*) sistemin erken bir sürümünü (manuel
poligon + IoU, takip yok) anlatır. Tez, kod tabanındaki GÜNCEL ve gelişmiş
sistemi yansıtacaktır. Doğrulanmış modüller (src/ altında mevcut):

Algılama (src/detection/):
  • vehicle_detector.py     — YOLOv8 araç tespiti, sınıf-bağımsız NMS
  • vehicle_tracker.py      — ego-hareket telafili takip (optik akış), statik/
                              hareketli araç ayrımı, süre takibi
  • drivable_area.py        — YOLOPv2 sürülebilir alan (yol) maskesi
  • parking_line_detector.py— Hough tabanlı park şeridi tespiti, ızgara çıkarımı
  • grid_fusion.py          — şerit konumlarının zamansal füzyonu
  • temporal_voter.py       — slot doluluk kararının zamansal oylaması
  • street_parking_detector.py — geometri tabanlı boş yer (paralel/dik) analizi
  • adaptive_slot_detector.py  — çizgi varsa ızgara, yoksa geometri (adaptif)
  • depth_estimator.py      — MiDaS monoküler derinlik
  • video_stabilizer.py     — video sabitleme

Park mantığı (src/parking/):
  • parking_analyzer.py     — analiz orkestrasyonu
  • slot_scoring.py         — çok kriterli slot skoru / öneri motoru
  • learned_slot_memory.py  — öğrenilen slot belleği (zamansal kalıcılık)
  • occupancy_heatmap.py    — doluluk ısı haritası
  • zone_annotator.py / zone_loader.py — bölge etiketleme/yükleme

Geometri (src/geometry/):
  • ipm.py                  — ters perspektif dönüşümü (kuş bakışı), metrik ölçek
  • auto_ipm.py             — otomatik IPM kalibrasyonu
  • roi.py                  — ilgi bölgesi

Arayüz (src/ui/):
  • main_window.py          — PyQt5 ana pencere, canlı işlem hattı
  • overlays.py             — çizim katmanları
  • alert_system.py         — uyarı sistemi
  • ipm_dialog.py / auto_ipm_dialog.py / roi_dialog.py — kalibrasyon diyalogları

Sesli asistan (src/voice/):
  • voice_assistant.py      — Vosk çevrimdışı sesli komut + edge-tts sesli yanıt

Değerlendirme (src/evaluation/):
  • metrics.py, synthetic.py, ablation.py, datasets.py, runner.py, plots.py,
    video_tools.py — metrik (P/R/F1/mAP), sentetik veri, ablasyon çerçevesi

Eğitim (src/training/):
  • finetune.py             — YOLOv8 ince ayar

Test: 268 otomatik test (tests/).


────────────────────────────────────────────────────────────────
4. BÖLÜM BÖLÜM İÇERİK PLANI
────────────────────────────────────────────────────────────────

01. ÖZET ve ABSTRACT (atıfsız)
   • Türkçe özet (~200-250 kelime): problem, yöntem, başlıca bileşenler, sonuç.
   • İngilizce abstract: birebir karşılık.
   • Anahtar kelimeler / keywords.

02. GİRİŞ
   2.1 Problemin Tanımı ve Önemi (kentsel park sorunu, maliyet/emisyon)
   2.2 Motivasyon (sensör maliyeti, mevcut kamerayla çözüm, EV uyumu)
   2.3 Amaç ve Kapsam
   2.4 Araştırma Soruları
   2.5 Özgün Katkılar (adaptif tespit + metrik ölçüm + sürücü yönlendirme +
       çevrimdışı sesli asistan birleşimi)
   2.6 Tezin Organizasyonu
   → İlk atıflar burada açılır (örn. pazar büyüklüğü, YOLOv8).

03. LİTERATÜR ARAŞTIRMASI
   3.1 Problemin Alt Görevleri (konum, doluluk, sayım)
   3.2 Derin Öğrenme Öncesi Yöntemler (background subtraction, HOG+SVM, Hough)
   3.3 İki Ana Paradigma (patch sınıflandırma vs nesne tespiti + IoU)
   3.4 Önemli Akademik Çalışmalar (mAlexNet, CarNet, MobileNetV3+CBAM, APSD-OC,
       PakLoc, da Luz vd.)
   3.5 Geçici/Park Edilmiş Araç Ayrımı ve Takip (SORT, DeepSORT, ByteTrack)
   3.6 YOLO Sürümleri Karşılaştırması
   3.7 Benchmark Veri Setleri (PKLot, CNRPark-EXT, PS2.0)
   3.8 Sensör vs Kamera Tabanlı Sistemler (EV sorunu)
   3.9 Ticari Sistemler (Parquery, Quercus, Bosch, Metropolis, Hikvision, Dahua)
   3.10 Projenin Literatürdeki Yeri ve Boşluk Analizi

04. KULLANILAN TEKNOLOJİLER VE YÖNTEMLER
   4.1 Python, PyTorch
   4.2 YOLOv8 (Ultralytics) — mimari, neden YOLOv8n
   4.3 OpenCV — görüntü işleme temelleri (CLAHE, Hough, renk uzayları)
   4.4 NumPy
   4.5 PyQt5 arayüz
   4.6 MiDaS monoküler derinlik
   4.7 Vosk (sesli komut) + edge-tts (sesli yanıt)
   4.8 Ters Perspektif Dönüşümü (IPM) — homografi matematiği
   4.9 Değerlendirme yığını (metrik, sentetik veri, ablasyon, pytest)

05. SİSTEM TASARIMI VE MİMARİSİ
   5.1 Genel İşlem Hattı (uçtan uca akış)
   5.2 Gece Görüşü İyileştirme (CLAHE)
   5.3 Araç Tespiti ve Sınıf-Bağımsız NMS
   5.4 Araç Takibi (ego-hareket telafisi, statik/hareketli ayrımı)
   5.5 Adaptif Boş-Yer Tespiti
       5.5.1 Çizgi tabanlı ızgara (Hough + füzyon + zamansal oylama)
       5.5.2 Geometri tabanlı boşluk analizi (paralel/dik, perspektif uyumu)
   5.6 IPM Kuş Bakışı ve Otomatik Kalibrasyon
   5.7 Gerçek Metrik Ölçüm ve Ölçek Kestirimi
   5.8 Sığma Kontrolü (araç-slot uygunluğu)
   5.9 Çok Kriterli Slot Öneri Motoru
   5.10 Öğrenilen Slot Belleği ve 2B Şematik Harita
   5.11 Monoküler Derinlik Entegrasyonu
   5.12 Sesli Asistan (çift yönlü etkileşim)
   5.13 Tasarım Kararları ve Gerekçeleri

06. GERÇEKLEŞTİRİM (UYGULAMA DETAYLARI)
   6.1 Yazılım Mimarisi ve Modülerlik
   6.2 Gerçek-Zamanlı İşlem Hattı ve İş Parçacığı Yönetimi (AsyncVideoCapture)
   6.3 Kararlılık Teknikleri (zamansal oylama, video sabitleme, throttle)
   6.4 Perspektif Düzeltme ve Yan Görüş Ele Alımı
   6.5 Kullanıcı Arayüzü (paneller, kontroller, harita)
   6.6 Yapılandırma ve Genişletilebilirlik

07. DENEYSEL DEĞERLENDİRME VE BULGULAR
   7.1 Değerlendirme Yöntemi (sentetik senaryolar + ön doğrulama)
   7.2 Metrikler (Precision/Recall/F1/AP)
   7.3 Sentetik Senaryo Sonuçları
   7.4 Ablasyon Çalışması
   7.5 Performans (etkileşimli hız, modül etkisi)
   7.6 Yazılım Kalitesi (268 test)
   7.7 Niteliksel Sonuçlar (uygulama görüntüleri)
   7.8 Koşula Bağlı Davranış ve Sınırlamalar (gece/düşük ışık)

08. SONUÇ VE GELECEK ÇALIŞMALAR
   8.1 Genel Değerlendirme
   8.2 Literatürle Karşılaştırma
   8.3 Karşılaşılan Zorluklar
   8.4 Gelecek Çalışmalar (CNN doluluk sınıflandırıcı, ByteTrack, çoklu kamera,
       PKLot ince ayar)
   8.5 Kapanış

09. KAYNAKÇA
   • IEEE benzeri biçim, metindeki ilk geçiş sırasına göre numaralı.


────────────────────────────────────────────────────────────────
5. KALİTE KURALLARI
────────────────────────────────────────────────────────────────
• Dil: akıcı, akademik, profesyonel Türkçe. Birinci çoğul ("geliştirdik",
  "kullanılmıştır") tutarlı kullanılır.
• Doğruluk: yalnızca kod tabanında GERÇEKTEN VAR OLAN özellikler iddia edilir.
  Olmayan sonuç/metrik uydurulmaz; ölçülmemiş şeyler "planlanmaktadır" denir.
• Tutarlılık: terim birliği (slot, boş yer, doluluk, IPM, kuş bakışı).
• Atıflar sıralı ve eksiksiz; her atıfın KAYNAKCA'da karşılığı bulunur.
• Her bölüm kendi içinde tutarlı; bölümler arası tekrar minimumda.
• Sayısal iddialar (268 test, metrikler) gerçek değerlerle verilir.
