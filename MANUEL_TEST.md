MANUEL TEST REHBERİ — Tüm Eklenen Özellikler
Tarih: 2026-06-05

Bu dosya, projeye eklenen HER özelliği adım adım manuel test etmen için
hazırlandı. Sırayla git, hiçbir maddeyi atlama. Her maddede: NE YAPILACAK +
BEKLENEN SONUÇ yazılı.


════════════════════════════════════════════════════════════════════
0. OTOMATİK TESTLER (önce bunu çalıştır)
════════════════════════════════════════════════════════════════════
Komut:
    python -m pytest tests/test_system.py -q

Beklenen: 148 passed (hepsi yeşil). Bir tane bile kırmızı olmamalı.
Bu, tüm modüllerin (metrik, IPM, çizgi tespiti, adaptif, oylama, derinlik,
overlay, fine-tuning dönüşümü, değerlendirme) birim düzeyinde sağlam olduğunu
gösterir.


════════════════════════════════════════════════════════════════════
1. UYGULAMAYI BAŞLATMA
════════════════════════════════════════════════════════════════════
Komut (Windows):
    python run.py    (veya ana giriş neyse)

Beklenen: "Smart Parking AI" penceresi açılır. Sağda yan panel, solda video
alanı. Durum çubuğunda "Model yüklendi." görünür.


════════════════════════════════════════════════════════════════════
2. KAYNAK YÜKLEME
════════════════════════════════════════════════════════════════════
2.1 "🖼 Resim" → bir park fotoğrafı seç.
    Beklenen: Görüntü solda görünür, araçlar yeşil kutuyla işaretlenir.
2.2 "📂 Video" → otopark videonu seç.
    Beklenen: Video oynar, araçlar canlı kutulanır, sağ üstte FPS artar.
2.3 "■ Durdur".
    Beklenen: Video durur, FPS "—" olur.


════════════════════════════════════════════════════════════════════
3. OTOMATİK PARK TESPİTİ (paralel / dik)
════════════════════════════════════════════════════════════════════
3.1 "🚗 Otomatik Tespiti Ac" butonuna bas.
    Beklenen: Buton "Otomatik Tespit ACIK" olur; boş alanlar yeşil işaretlenir.
3.2 "Yön" satırında "Dik" butonuna bas.
    Beklenen: Buton mavi/aktif olur; altta "On gorunum aktif (ref. en)" veya
    "Yan gorunum aktif (ref. uzunluk)" yazısı çıkar (videonun açısına göre).
3.3 "Paralel"e geri bas.
    Beklenen: Görünüm etiketi gizlenir, paralel mod parametreleri devreye girer.
3.4 Sliderlar: "Bos" (min boşluk), "Sir" (sıra band), "Yok" (üst yoksay).
    Her birini oynat.
    Beklenen: Değer sağda güncellenir; fotoğraf modunda tespit anında yenilenir.


════════════════════════════════════════════════════════════════════
4. ARAÇ SIĞMA KONTROLÜ
════════════════════════════════════════════════════════════════════
4.1 Paralel modda: "Ref. araç uzunluğu" ve "Aracın uzunluğu" kutuları görünür.
    "Aracın uzunluğu"nu büyüt/küçült.
    Beklenen: Boş slot etiketleri "SIGAR/SIGMAZ Xm" olarak renk değiştirir
    (yeşil sığar, kırmızı sığmaz).
4.2 Dik moda geç: "Ref. araç eni" ve "Aracın eni" kutuları görünür, uzunluk
    kutuları gizlenir.
    Beklenen: Sığma kararı artık araç enine göre verilir.


════════════════════════════════════════════════════════════════════
5. UYARI EŞİĞİ
════════════════════════════════════════════════════════════════════
5.1 "UYARI EŞİĞİ" kutusunda "Dol" sliderını %20 gibi düşük bir değere çek.
5.2 Doluluğu eşiği aşan bir görüntü/video yükle.
    Beklenen: Üstte sarı/kırmızı uyarı barı çıkar ("Park alani %X dolu...").
    10 sn sonra otomatik kapanır (kritik değilse). "x" ile elle kapatılır.


════════════════════════════════════════════════════════════════════
6. IPM — KUŞ BAKIŞI (yeni)
════════════════════════════════════════════════════════════════════
6.1 Bir görüntü/video yükle. Sağ panelde "KUS BAKISI (IPM)" kutusunu bul.
6.2 "Kalibre Et" butonuna bas.
    Beklenen: Ayrı bir pencere açılır, görüntü tam boyutta görünür.
6.3 Zemindeki bir dikdörtgenin 4 köşesini SIRAYLA tıkla:
    1 sol-üst, 2 sağ-üst, 3 sağ-alt, 4 sol-alt.
    Beklenen: Her tık yeşil numaralı nokta + aralarına mavi çizgi.  
6.4 "Gercek genislik (m)" ve "Gercek yukseklik (m)" değerlerini gir (ör. 10 / 5).
6.5 "Onayla".
    Beklenen: Pencere kapanır; panelde "Kalibre edildi · 0.0XXX m/px" yazar;
    "Goster" butonu aktifleşir.
6.6 "Goster"e bas.
    Beklenen: Görüntü kuş bakışına döner (üstten bakış). "Gizle"ye basınca
    normale döner.
6.7 (Hata kontrolü) Hiç görüntü yokken "Kalibre Et" → "Önce görüntü yükleyin"
    uyarısı çıkmalı. 4 noktadan az işaretleyip "Onayla" → "Tam 4 nokta" uyarısı.


════════════════════════════════════════════════════════════════════
7. DERİNLİK FİLTRESİ (yeni — model yoksa pasif)
════════════════════════════════════════════════════════════════════
7.1 IPM kutusunda "Derinlik Filtresi" butonuna bak.
    Beklenen (model yoksa): Buton PASİF (gri), altında "Derinlik modeli yok
    (pasif)" yazar. Bu normaldir; MiDaS modeli indirilince otomatik aktifleşir.
    (Model varsa buton tıklanır, "Derinlik AÇIK" olur ve çapraz-açı slotları
    elenir.)


════════════════════════════════════════════════════════════════════
8. ADAPTİF ÇİZGİ-IZGARA (yeni — ana özellik)
════════════════════════════════════════════════════════════════════
8.1 Çizgili (boyalı şeritli) bir otopark görüntüsü/videosu yükle.
    (En iyi sonuç için önce IPM Kalibre Et — kuş bakışında çizgiler netleşir.)
8.2 "Çizgi-Izgara (Adaptif)" butonuna bas.
    Beklenen: Buton "Çizgi-Izgara AÇIK" olur. Görüntüde:
      - Boş slotlar YEŞİL sahte-3B kutu ("BOS" etiketli)
      - Dolu slotlar KIRMIZI çerçeve
      - Sol üstte "ADAPTIF: CIZGI-IZGARA | N ms" rozeti
      - Sarı "EN YAKIN" oku en yakın boş slotu gösterir
        (IPM kalibreyse "EN YAKIN X.Xm" mesafe ile)
      - Sağ panelde Boş/Dolu sayıları güncellenir
8.3 Çizgisiz bir alan (yol kenarı) görüntüsü yükle, adaptif açıkken.
    Beklenen: Rozet "ADAPTIF: GEOMETRI" olur — sistem otomatik geometri
    yöntemine düşer (çizgi olmayan senaryo da çalışır).
8.4 Videoda birkaç saniye izle.
    Beklenen: Slot durumları titremez (zamansal oylama yumuşatır).
8.5 Yeni görüntü yükle.
    Beklenen: Adaptif geçmiş sıfırlanır (eski slotlar taşınmaz).


════════════════════════════════════════════════════════════════════
9. DEĞERLENDİRME PANELİ (yeni)
════════════════════════════════════════════════════════════════════
9.1 "📊 Değerlendirme Çalıştır" butonuna bas.
    Beklenen: Buton "Çalışıyor…" olur; birkaç saniye sonra bir pencere açılır:
      - Üstte Precision/Recall/F1/AP sayıları (120 sentetik sahne)
      - Altında 2 grafik (metrik çubukları + karışıklık matrisi)
    Durum çubuğunda "Değerlendirme bitti · F1=0.9XX" yazar.
    Çıktılar: outputs/evaluation_synthetic/ (PNG + CSV).


════════════════════════════════════════════════════════════════════
10. SNAPSHOT / LOG
════════════════════════════════════════════════════════════════════
10.1 "📸 Snapshot".
     Beklenen: outputs/snapshots/ altına o anki analizli kare PNG kaydedilir.
10.2 "⏺ Log" (video/kamera modunda) → biraz bekle → tekrar bas.
     Beklenen: outputs/metrics/ altına CSV log (frame, doluluk, park süresi).


════════════════════════════════════════════════════════════════════
11. KOMUT SATIRI ARAÇLARI (UI dışı, rapor için)
════════════════════════════════════════════════════════════════════
11.1 Sentetik değerlendirme (120 sahne):
     python -m src.evaluation.runner --mode synthetic --scenes 120
     Beklenen: P/R/F1/AP yazdırılır; outputs/evaluation_synthetic/ dolar.

11.2 Yöntem karşılaştırma + ablation + duyarlılık:
     python -m src.evaluation.ablation --scenes 120
     Beklenen: Baseline/Agresif/Muhafazakar F1'leri + ablation + min_gap
     duyarlılık grafiği; outputs/ablation/ dolar.

11.3 Gerçek görüntü değerlendirmesi (3 örnek görüntü):
     python -m src.evaluation.runner --gt data/ground_truth.json --images .
     Beklenen: Micro-F1 yazdırılır; outputs/evaluation/ dolar.
     (YOLO modeli yükler, biraz sürer.)

11.4 Videodan kare çıkarma (offline etiketleme için):
     Python içinden:
       from src.evaluation.video_tools import extract_frames
       extract_frames("VIDEO_YOLU.mp4", "data/frames", count=20)
     Beklenen: data/frames/ altına 20 kare PNG.


════════════════════════════════════════════════════════════════════
11b. İKİNCİ FAZ ÖZELLİKLERİ (yeni — adaptif iyileştirmeler)
════════════════════════════════════════════════════════════════════
Not: Çizgi renk segmentasyonu, alt-piksel hassaslaştırma ve ızgara füzyonu
arka planda otomatik çalışır (ayrı buton yok); etkilerini adaptif modda
(çizgi-ızgara) gözlersin: çizgiler daha sağlam bulunur, slotlar titremez.

11b.1 Otomatik IPM:
  "Oto IPM (cizgilerden)" butonuna bas. Önce çizgi-yakınsama, başarısızsa
  araç-tabanlı yedek denenir. Başarılıysa "Oto kalibre · X m/px" yazar ve
  kuş bakışı OTOMATİK açılır. İkisi de başarısızsa açılı kare/manuel öneren
  uyarı çıkar. Not: Düz cepheden tek-derinlik karede otomatik çalışmaz
  (perspektif ipucu yok) — açılı/yandan kare veya manuel kalibrasyon gerekir.

11b.1b Oto ROI:
  "Oto ROI" butonuna bas. Beklenen: Araçların çevresini saran ilgi bölgesi
  otomatik oluşur (dışı kararır). Araç yoksa uyarı verir.

11b.1c Sığma kontrolü (adaptif + IPM):
  IPM kalibreyken Çizgi-Izgara aç. Beklenen: Yeşil = boş ve aracın sığar
  ("SIGAR X.Xm"), turuncu = boş ama sığmaz ("SIGMAZ X.Xm"), kırmızı = dolu.
  "EN YAKIN UYGUN Xm" oku yalnızca sığan boş slota gider.

11b.2 Video sabitleme:
  Video oynarken "Sabitleme (IPM icin)" butonuna bas.
  Beklenen: "Sabitleme AÇIK" olur. Elde-çekim titremesi varsa görüntü daha
  sabit durur; IPM/ızgara kaymadan yerinde kalır. (Yeterli özellik yoksa kareyi
  değiştirmez — sorun değil.)

11b.3 ROI (ilgi bölgesi):
  "ROI Sec" → açılan pencerede otopark alanının köşelerini sırayla tıkla (3+),
  "Onayla". Beklenen: Poligon dışı kararır, dışındaki araç tespitleri elenir.
  "Temizle" ile ROI kalkar.

11b.4 Adaptif modda yeni görseller:
  "Çizgi-Izgara (Adaptif)" açıkken:
    - Sağ üstte ŞEMATİK MİNİ-HARİTA (yeşil boş / kırmızı dolu, numaralı hücreler)
    - Sol üstte "Guven: %X" canlı güven göstergesi
    - "EN YAKIN Xm" yönlendirme oku + "N ms" performans rozeti (faz 1)


════════════════════════════════════════════════════════════════════
12. BİLİNEN DAVRANIŞLAR (hata değil)
════════════════════════════════════════════════════════════════════
- Derinlik modeli (MiDaS) indirilmediği için Derinlik Filtresi pasiftir.
  İstersen sonra modeli ekleriz, otomatik aktifleşir.
- IPM kalibrasyonu manueldir (4 köşe). Kamera çok oynarsa yeniden kalibre et.
- Çizgi yoksa adaptif mod otomatik geometriye düşer — beklenen davranış.
- Headless ortamda (sunucu) torch+Qt çakışması olabilir; normal Windows
  masaüstünde sorun yoktur.


════════════════════════════════════════════════════════════════════
13. EKLENEN DOSYALAR (referans)
════════════════════════════════════════════════════════════════════
Çekirdek:
  src/detection/parking_line_detector.py   — çizgi/ızgara tespiti
  src/detection/adaptive_slot_detector.py  — adaptif seçici (çizgi/geometri)
  src/detection/temporal_voter.py          — zamansal oylama
  src/detection/depth_estimator.py         — monoküler derinlik (opsiyonel)
  src/geometry/ipm.py                       — kuş bakışı homografi + ters
  src/ui/ipm_dialog.py                      — IPM 4-nokta kalibrasyon penceresi
  src/ui/overlays.py                        — en yakın slot oku + AR 3B
Değerlendirme:
  src/evaluation/metrics.py, plots.py, runner.py, synthetic.py,
  datasets.py (PKLot), ablation.py, video_tools.py
Eğitim:
  src/training/finetune.py                  — PKLot→YOLO + fine-tuning
Testler: tests/test_system.py (148 test)
