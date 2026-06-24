GELECEK FİKİRLER — Henüz Yapılmayan Opsiyonel Geliştirmeler
Tarih: 2026-06-05

Bu dosya, beyin fırtınalarında konuşulan ama henüz BAŞLANMAYAN fikirleri tutar.
Her madde: NE + NEDEN (doğruluk/jüri/problem) + BAĞIMLILIK + TAHMİNİ EMEK.
Senaryo kısıtı: tek dik otopark videosu, araç yok, başta etiket yok.

Etiket notu: Doğruluk/değerlendirme/ablation/CNN gerektiren her madde için
önce videodan kare çıkarıp boş/dolu etiketlemek gerekir (araç hazır:
src/evaluation/video_tools.py + data/ground_truth.json formatı).


════════════════════════════════════════════════════════════════════
A. EĞİTİMSİZ — HEMEN YAPILABİLİR (en yüksek getiri)
════════════════════════════════════════════════════════════════════

A1. Otomatik IPM (araçları cetvel olarak kullan)
  NE: Manuel 4-nokta yerine, sahnedeki araçların bilinen gerçek boyutlarını
      (VEHICLE_REAL_DIMS) referans alıp RANSAC ile zemin düzlemi homografisini
      otomatik hesapla.
  NEDEN: Manuel kalibrasyon adımını kaldırır; kamera oynasa bile her karede
      yeniden kestirilebilir; "sistem kendi kalibre ediyor" jüriye etkili.
  BAĞIMLILIK: Yok (mevcut YOLO tespitleri + VEHICLE_REAL_DIMS yeter).
  EMEK: Orta. RANSAC + en az 2-3 araç footprint'i gerekir.

A2. Çarpık / hatalı park tespiti
  NE: Aracın çizgiyi taşması veya açılı park etmesini yakala; "kötü park"
      uyarısı.
  NEDEN: Pratik, ilişkilendirilebilir; günlük kullanım değeri yüksek.
  BAĞIMLILIK: Park çizgisi tespiti (var) + araç bbox/yönelimi.
  EMEK: Düşük-orta. Slot ile araç hizası karşılaştırması.

A3. Geometrik tutarlılık — genişletme
  NE: Mevcut boyut tutarlılık filtresine ek olarak slot aralık düzenliliği,
      hizalama ve sıra sürekliliği denetimleri.
  NEDEN: Sahte pozitifleri daha da düşürür (bedava doğruluk).
  BAĞIMLILIK: Yok. (Temel sürüm zaten var: filter_by_size_consistency.)
  EMEK: Düşük.

A4. Belirsizlik / güven kalibrasyonu
  NE: Her slot için güven skoru; eşik altında "emin değilim" diyip çekimser
      kal (abstain). Doluluk örtüşme skoru zaten üretiliyor (classify_slots).
  NEDEN: Olgun ML davranışı; jüri "hata payını biliyorlar" der.
  BAĞIMLILIK: Yok. EMEK: Düşük.


════════════════════════════════════════════════════════════════════
B. JÜRİ-CEZBEDEN DEMO ÖZELLİKLERİ
════════════════════════════════════════════════════════════════════

B1. Metrik 3B AR overlay (mevcut pseudo-3B'nin gelişmişi)
  NE: IPM ile gerçek zemin düzlemi bilindiğinden slotları doğru yükseklikli
      3B kutular olarak çiz (şu an sabit-piksel sahte-3B).
  NEDEN: Görsel olarak çarpıcı, kuş bakışı çalışmanın doğal devamı.
  BAĞIMLILIK: IPM kalibrasyon. EMEK: Orta.

B2. Doğal dil / sesli sorgu
  NE: "En yakın boş yer nerede?" → sistem yanıtlar (metin/ses).
  NEDEN: Gösteriş; küçük entegrasyon, büyük etki.
  BAĞIMLILIK: Küçük bir komut ayrıştırıcı; en yakın slot zaten hesaplanıyor.
  EMEK: Düşük-orta.

B3. Gelişmiş performans paneli
  NE: Modül başına gecikme (tespit / IPM / çizgi / çizim) ayrı ayrı + FPS
      grafiği. (Şu an sadece toplam ms rozeti var.)
  NEDEN: Mühendislik olgunluğu; akademik "verimlilik analizi".
  BAĞIMLILIK: Yok. EMEK: Düşük.


════════════════════════════════════════════════════════════════════
C. AKADEMİK AĞIRLIK (etiket/veri gerektirir)
════════════════════════════════════════════════════════════════════

C1. Kendi-veri doluluk CNN'i
  NE: Senin videondan birkaç yüz slot kırpıntısını boş/dolu etiketle, küçük
      bir CNN eğit; her hücreyi sınıflandır.
  NEDEN: Projeyi "sezgisel"den "derin öğrenme"ye taşır; domain farkı olmaz
      (kendi verin). Ana fikrin 4. adımı.
  BAĞIMLILIK: Slot kırpıntısı etiketleme (manuel) + hafif eğitim (CPU yeter).
  EMEK: Orta-yüksek (çoğu etiketleme).

C2. IPM / derinlik katkısını ölçmek (kendi modüllerimizin ablation'ı)
  NE: Etiketli karelerde IPM açık/kapalı (ve derinlik açık/kapalı) doğruluğu
      karşılaştır → "IPM metrik hatayı %X düşürüyor".
  NEDEN: "Katkımızı sayısal kanıtladık" — jüri çok sever.
  BAĞIMLILIK: Etiketli kareler. Altyapı hazır: run_adaptive_evaluation(ipm=...).
  EMEK: Düşük (etiketten sonra).

C3. Fine-tuned YOLO (PKLot) gerçekten eğit
  NE: Hazır pipeline (src/training/finetune.py) ile PKLot üzerinde park-özel
      model eğit; öncesi/sonrası mAP.
  NEDEN: "Script yazdık" → "modelimiz var". Akademik anlatı.
  BAĞIMLILIK: PKLot indir (~1.5GB) + GPU (CPU çok yavaş) + domain farkı riski.
  EMEK: Yüksek. (Tek video için getirisi düşük; öncelik düşük.)

C4. PKLot doluluk CNN'i (literatür baseline)
  NE: PKLot'ta CNN eğit (Amato/Nurullayev ~%99) ve videona uygula.
  NEDEN: Yayınlanmış baseline ile kıyas.
  BAĞIMLILIK: PKLot indir + eğitim; havadan→telefon domain farkı.
  EMEK: Yüksek. (C1 "kendi-veri" versiyonu daha gerçekçi.)

C5. Çapraz veri seti genelleme testi
  NE: Bir sette öğren, başka sette test et.
  NEDEN: Genelleme gücü — güçlü akademik nokta.
  BAĞIMLILIK: En az 2 veri seti (PKLot + CNRPark). Tek video ile uygulanamaz.
  EMEK: Yüksek. (Şimdilik kapsam dışı.)


════════════════════════════════════════════════════════════════════
D. TAHMİN / ANALİTİK (zaman serisi)
════════════════════════════════════════════════════════════════════

D1. Doluluk tahmini + güven aralığı
  NE: Geçmiş doluluktan "10 dk sonra %85 dolu olacak" + belirsizlik bandı.
  NEDEN: Tahminsel zeka; jüriyi etkiler.
  BAĞIMLILIK: Uzun video / zaman içinde değişen doluluk. Tek kısa videoda sınırlı.
  EMEK: Orta.

D2. Slot bazında devir hızı / dwell-time analizi
  NE: Hangi slot en çok kullanılıyor, ortalama kalış süresi, doluluk döngüsü.
  NEDEN: İşletme/gelir perspektifi; olgunluk.
  BAĞIMLILIK: Park süresi takibi (var) + slot kimliği (adaptif slotlar).
  EMEK: Orta.

D3. Terk edilmiş araç / anomali tespiti
  NE: Çok uzun süre hiç hareket etmeyen araç veya hiç boşalmayan slot uyarısı.
  NEDEN: Gerçek-dünya faydası.
  BAĞIMLILIK: Park süresi takibi (var). EMEK: Düşük-orta.


════════════════════════════════════════════════════════════════════
E. SENARYO KISITINDAN DOLAYI ŞİMDİLİK UYGUN OLMAYANLAR
════════════════════════════════════════════════════════════════════
(Araç yok / tek video / tek seans olduğu için ertelendi.)

E1. Denetimsiz slot haritası öğrenme (otoparkı izleyerek kroki çıkarma):
    Birden çok park olayı / uzun gözlem gerekir.
E2. Giriş-çıkış sayımı + re-ID: Araç akışı / uzun video gerekir.
E3. Çoklu kamera füzyonu: Birden çok kamera gerekir.
E4. Engelli / EV / rezerve sembol tanıma: SADECE videondaki otoparkta o
    semboller gerçekten varsa anlamlı — videonu kontrol et, varsa A grubuna alınır.


════════════════════════════════════════════════════════════════════
F. İKİNCİ TUR BRAINSTORM — OPSİYONELE ALINANLAR
════════════════════════════════════════════════════════════════════
(Kullanıcı kararı: şimdilik yapılmayacak, bir kenarda dursun.)

F1. Yan yana karşılaştırma görünümü
  NE: Aynı karede solda sezgisel geometri, sağda adaptif çizgi-ızgara; canlı.
  NEDEN: Katkıyı görsel kanıtlar ("eski yöntem kaçırıyor, bizimki yakalıyor").
  BAĞIMLILIK: Her iki yöntem de hazır. EMEK: Orta. Karar: opsiyonel.

F2. Kenar / kısmi slot işleme
  NE: Kare kenarında yarım kalan slotları doğru say.
  NEDEN: Pratik doğruluk düzeltmesi.
  BAĞIMLILIK: Yok. Karar: opsiyonel (yaklaşım belirsiz, ölçümle netleşir).

F3. Oklüzyon akıl yürütme
  NE: Bir araç diğerini kısmen kapattığında, ızgara önbilgisiyle "tespit zayıf
      ama bu slot dolu" çıkarımı yap.
  NEDEN: Kalabalık sahnede doğruluk. BAĞIMLILIK: Izgara + tespit güveni.
  EMEK: Orta. Karar: opsiyonel.

F4. Uygulama-içi etiketleme aracı
  NE: Bir karede slotları tıklayıp boş/dolu etiketle, ground truth kaydet.
  NEDEN: Etiket darboğazını çözer; değerlendirme/ablation/CNN'in önünü açar.
  BAĞIMLILIK: Yok. EMEK: Orta-yüksek (uğraştırır). Karar: opsiyonel.

F5. Bozulmaya karşı sağlamlık analizi
  NE: Kendi karelerine bulanıklık/gürültü/parlaklık ekleyip doğruluğun düşüşünü
      ölç (sağlamlık eğrisi). Hava verisi olmadan robustness analizi.
  NEDEN: Akademik puan. BAĞIMLILIK: Etiketli kareler + birkaç görüntü.
  EMEK: Orta. Karar: opsiyonel (görüntü azken sona bırakıldı).

F6. Zamansal kararlılık metriği
  NE: Oylamanın titremeyi ne kadar azalttığını sayısallaştır. Açıklama:
      Video boyunca her slotun durumu kare-kare ne sıklıkla değişiyor sayılır
      (flip oranı). Oylama AÇIK ve KAPALI iken bu oran karşılaştırılır;
      "oylama ardışık-kare durum değişimini %X azalttı" denir. Modülün
      faydasının nesnel kanıtı.
  NEDEN: Kendi katkımızı ölçmek; jüriye somut sayı.
  BAĞIMLILIK: Video + oylama açık/kapalı koşturma. EMEK: Düşük.
  Karar: opsiyonel.


════════════════════════════════════════════════════════════════════
ÖNERİLEN SIRADAKI TUR
════════════════════════════════════════════════════════════════════
Eğitimsiz + yüksek etki: A1 (otomatik IPM) → A2 (çarpık park) → A4 (güven
kalibrasyonu) → B1 (metrik 3B). Sonra etiketleme yapılırsa C2 (IPM katkı
ablation) ve C1 (kendi-veri CNN). C3/C4/C5/E grubu, kaynak (PKLot/GPU/çoklu
kamera) bulunursa.
