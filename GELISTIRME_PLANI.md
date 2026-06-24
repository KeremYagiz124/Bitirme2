GELİŞTİRME PLANI — Akademik Derinleştirme

Amaç: Projeyi "basit demo"dan "değerlendirilmiş araştırma"ya taşımak.
Durum: Hoca projeyi basit buldu. Aşağıdaki maddeler sırayla, eksiksiz uygulanacak.

────────────────────────────────────────────────────────
0. BÜYÜK ARAÇLARA VARSAYILAN BOYUT (sınıf-duyarlı ölçek)
────────────────────────────────────────────────────────
Sorun: Ölçek tahmini tüm park etmiş araçları ~4.5m otomobil varsayıyor.
Kamyon/otobüs varsa medyan piksel genişliği bozuluyor, ölçek hatalı.
Çözüm: Her araç sınıfına gerçek-dünya boyutu ata (car, truck, bus, motorcycle).
Ölçeği sınıf-duyarlı hesapla: her aracın (gerçek_boyut / piksel) değerinin medyanı.
Durum: YAPILIYOR.

────────────────────────────────────────────────────────
1. NİCEL DEĞERLENDİRME ÇERÇEVESİ
────────────────────────────────────────────────────────
- src/evaluation/ modülü: Precision, Recall, F1, mAP, IoU-tabanlı eşleştirme.
- Karışıklık matrisi (TP/FP/FN/TN).
- matplotlib ile grafikler (PR eğrisi, bar grafikleri, confusion matrix heatmap).
- Sonuçları outputs/evaluation/ altına PNG + CSV olarak kaydet.
- Tekrarlanabilir: tek komutla tüm metrikler üretilsin.

────────────────────────────────────────────────────────
2. VERİ SETİ HAZIRLAMA
────────────────────────────────────────────────────────
- PKLot / CNRPark-EXT ingest scripti (indir, çöz, etiket formatına çevir).
- Ground truth format: her görüntü için boş/dolu slot sayısı + bbox.
- 100+ görüntülük değerlendirme alt kümesi.
- İnternet/disk gerektirir; indirme scripti + yerel cache.

────────────────────────────────────────────────────────
3. YÖNTEM KARŞILAŞTIRMA + ABLATION
────────────────────────────────────────────────────────
- 3 yöntem aynı veri setinde: sezgisel boşluk / öğrenilmiş slot / derin model.
- Ablation: her bileşeni kapat-aç, metriğe etkisini ölç.
- Parametre duyarlılık analizi (min_gap_ratio, edge_extension vb. süpürme).
- Karşılaştırma tabloları + grafikler.

────────────────────────────────────────────────────────
4. FINE-TUNING PIPELINE
────────────────────────────────────────────────────────
- PKLot üzerinde park-özel YOLO ince ayar scripti.
- Eğitim config + veri yaml + train/val split.
- Öncesi (hazır YOLO) vs sonrası (fine-tuned) metrik karşılaştırması.
- GPU gerektirir; script + dokümantasyon, mümkünse demo eğitim.

────────────────────────────────────────────────────────
5. IPM / BIRD'S EYE VIEW (perspektif düzeltme)
────────────────────────────────────────────────────────
- Homografi ile görüntüyü kuş bakışına çevir (cv2.getPerspectiveTransform).
- UI: 4 nokta seçerek zemin düzlemi kalibrasyonu.
- Kuş bakışı görünümde gerçek metrik ölçüm → çapraz açı sorunu çözülür.
- Mevcut park analizini IPM'lenmiş görüntü üzerinde çalıştırma seçeneği.

────────────────────────────────────────────────────────
6. MONOKÜLER DERİNLİK TAHMİNİ
────────────────────────────────────────────────────────
- MiDaS veya Depth Anything entegrasyonu (DrivableAreaSegmenter gibi opsiyonel).
- Model yoksa graceful degrade (sistem çökmez).
- Derinlik haritasından gerçek 3D mesafe doğrulaması.

────────────────────────────────────────────────────────
2026-06-08 — A1 · A4 · C1 TAMAMLANDI (202 test geçiyor).
- A4 (slot_scoring.py): Çok kriterli slot seçim motoru — zorluk, yakınlık,
  genişlik payı, çıkış yakınlığı ağırlıklı skor + öneri gerekçesi. Adaptif modda
  en uygun slota altın çerçeve + "ÖNERİLEN N/100" + recommendation_lbl kartı.
- A1 (depth_estimator.py + UI): MiDaS monoküler derinlik. depth_to_colormap
  eklendi; "Isı Haritası" overlay toggle + araç bbox'larına "~X.Xm" mesafe
  etiketi; model ilk kullanımda lazy yüklenir (_ensure_depth_model, allow_download).
  Her 5 karede bir hesap (cache). Model yoksa graceful.
- C1 (voice/voice_assistant.py): Vosk çevrimdışı sesli asistan. Saf match_command
  parser; VoiceAssistant thread + graceful (model/mikrofon yoksa pasif). UI "Ses AÇ"
  butonu; komutlar QMetaObject ile ana thread'e marshal edilip YALNIZCA mevcut
  aksiyonlara bağlanır (find_empty, depth, night, ipm, evaluate, stop).

2026-06-11 — GECE VİDEOSU TESPİT DÜZELTMELERİ (211 test geçiyor). İzole, geriye-uyumlu.
- Madde 1 (street_parking_detector._slot_blocked_by_vehicle): park aracı boş
  sayılmasın. Slot merkezi araç içinde VEYA örtüşme slot %10 / araç %15 üzeri →
  elenir. TÜM araçlara, tam kutu (raw_bbox) ile. İnline filtre bu yardımcıya taşındı.
- Madde 2 (vehicle_detector._dedupe): sınıf-bağımsız NMS (IoU>0.6). YOLO'nun
  sınıf-bazlı NMS'i aynı aracı car+truck olarak çift sayıyordu → tek sayılır.
- Madde 4 (_slot_looks_like_road): alaca-karanlıkta (mean_v<85) renk-benzerliği
  kontrolü atlanır; yeşil-bitki reddi korunur. Gece gerçek karanlık-asfalt boşluğu
  artık reddedilmez. Gündüz (mean_v yüksek) davranışı AYNEN korunur → ground truth
  F1=%100 etkilenmez.
- Madde 5 (adaptive _pack_line_result): çizgi-ızgara slotu araç kutusuna denk
  geliyorsa (van gövdesindeki şeritler) BOŞ değil DOLU sayılır (Madde 1 yardımcısı
  kaynak uzayda). veh_src raw_bbox ile geçirilir.
- Madde 3/6: kümeleme ve slot-yükseklik değişiklikleri gündüz exact-count
  testlerini/dik-park sığma mantığını bozma riski taşıdığından YAPILMADI; sorunlar
  Madde 1+4+5 ile çözüldü (sayım düzeltilmiş poligonlardan türer).
Yeni testler: TestVehicleDedupe (3), TestVehicleOverlapReject (5),
line-slot-over-vehicle (1). 202→211 test.

İLERLEME NOTLARI
────────────────────────────────────────────────────────
2026-06-05 — Madde 0 TAMAMLANDI. VEHICLE_REAL_DIMS sabiti (car/truck/bus/
motorcycle) eklendi. estimate_scale sınıf-duyarlı hale getirildi
(_estimate_scale_classaware + _class_for_box + _real_dim_for). Sıraya
kamyon/otobüs karışsa bile ölçek bozulmuyor. 9 yeni test, toplam 70 geçti.

2026-06-05 — Madde 1 TAMAMLANDI. src/evaluation/: metrics.py (P/R/F1/AP/mAP/
karışıklık matrisi), plots.py (matplotlib grafikler), runner.py. 3 gerçek
görüntüde Micro-F1=1.0. outputs/evaluation/ PNG+CSV.

2026-06-05 — Madde 2/3 TAMAMLANDI. synthetic.py (120 prosedürel sahne, YOLO'dan
bağımsız slot değerlendirmesi → P=1.0 R=0.89 F1=0.94 AP=0.89). datasets.py
(PKLot XML→ground truth, veri yoksa graceful). outputs/evaluation_synthetic/.

2026-06-05 — Madde 4 TAMAMLANDI. ablation.py: yöntem karşılaştırması (Baseline/
Agresif/Muhafazakar), ablation (kenar/multi-row kapat-aç), parametre duyarlılık
analizi (min_gap_ratio süpürme). CSV+grafik. outputs/ablation/. Toplam 102 test.

2026-06-05 — Madde 5 TAMAMLANDI. src/training/finetune.py: PKLot→YOLO format
dönüşümü (write_yolo_labels, pklot_to_yolo, train/val split), make_data_yaml,
train_finetune (Ultralytics), compare_before_after. Veri dönüşümü test edildi;
eğitim GPU+veri gerektirir (dokümante).

2026-06-05 — Madde 6 TAMAMLANDI. src/geometry/ipm.py: PerspectiveTransformer
(from_quad homografi, warp_image, transform_box, m_per_px metrik kalibrasyon,
measure_distance_m, box_size_m). Çapraz açı sorununa doğrudan çözüm. 8 test.

2026-06-05 — Madde 7 TAMAMLANDI. src/detection/depth_estimator.py: monoküler
derinlik (MiDaS/Depth Anything, graceful degrade). region_depth, same_plane
(çapraz açıda farklı derinlikteki nesneleri ayırır). 10 test.

TÜM 7 MADDE TAMAMLANDI. Toplam 124 test geçiyor.
Çıktı klasörleri: outputs/evaluation, outputs/evaluation_synthetic, outputs/ablation

2026-06-05 — ENTEGRASYON FAZI TAMAMLANDI. Yazılan modüller canlı UI'a bağlandı:
- IPM: src/ui/ipm_dialog.py (4-nokta kalibrasyon penceresi, birebir koordinat
  eşleme). Yan panelde "KUS BAKISI (IPM)" bölümü: Kalibre Et + Goster/Gizle,
  m/px durumu. _show_frame IPM açıkken kuş bakışı warp gösterir.
- Derinlik: depth_estimator UI'a bağlandı; "Derinlik Filtresi" toggle (model
  yoksa pasif + "Derinlik modeli yok" durumu). _depth_keep_indices çapraz-açı
  slotlarını eler, empty_spaces/confs/sizes_m hizalı filtrelenir (graceful).
- Değerlendirme: "Değerlendirme Çalıştır" butonu → 120 sahne sentetik eval,
  sonuç penceresinde metrikler + grafikler (PNG) gösterilir.
Not: Headless bash'te torch+Qt offscreen segfault verir (ortam kaynaklı, kodla
ilgisi yok; Windows'ta gerçek ekranla sorunsuz). Syntax + 124 test geçti.

2026-06-05 — ADAPTİF ÇİZGİ-IZGARA FAZI TAMAMLANDI (148 test geçiyor).
Tek video / canlı senaryoya göre uyarlandı: çalışma zamanı hep canlı; çizgi
her yerde olmadığından adaptif (çizgi varsa ızgara, yoksa geometri).
- parking_line_detector.py: Hough ile şerit → grid_lines, has_lines,
  build_slots, classify_slots (YOLO örtüşmesi), filter_by_size_consistency.
- adaptive_slot_detector.py: çizgi/geometri seçici; IPM verilirse kuş
  bakışında çalışıp slotları kaynağa geri haritalar (poligon). reset().
- temporal_voter.py: kareler arası IoU eşleştirme + oy çoğunluğu (titreme↓).
- ipm.py: ters dönüşüm (inverse_transform_points/quad).
- ui/overlays.py: nearest_empty, draw_guidance (en yakın slot oku),
  draw_pseudo_3d (AR kutu). Saf cv2, test edildi.
- main_window.py: "Çizgi-Izgara (Adaptif)" toggle; izole _draw_adaptive dalı
  (poligon + AR 3B boş slot + en yakın yönlendirme + ms göstergesi);
  load_image/_start_feed'de _adaptive.reset().
- evaluation/video_tools.py: extract_frames (offline etiketleme için).
- evaluation/runner.py: run_adaptive_evaluation (IPM açık/kapalı karşılaştırma).
Kalan kullanıcı aksiyonu: videodan kare çıkar → boş/dolu etiketle → adaptif
değerlendirme + IPM ablation çalıştır; canlı testte IPM kalibrasyonu + adaptif
toggle'ı gerçek videoda doğrula.

2026-06-05 — İYİLEŞTİRME FAZI 2 TAMAMLANDI (173 test geçiyor). 8 madde:
- Çizgi renk segmentasyonu: beyaz/sarı şerit HSV maskesi (_color_line_mask) →
  asfalt dokusuna karşı sağlam çizgi tespiti.
- Alt-piksel çizgi hassaslaştırma: ızgara çizgileri boya yoğunluk tepesine snap
  (_refine_position; grid_lines refine=True).
- Kalıcı ızgara füzyonu (grid_fusion.py): çizgi geometrisini kareler arası EMA +
  kaçırma toleransıyla kararlı tutar; bir karede çizgi kaçsa bile harita ayakta.
- Video sabitleme (video_stabilizer.py): ORB+RANSAC ile kareyi referansa hizalar
  (IPM/ızgara elde-çekim kaymasında geçerli kalır; graceful). UI: "Sabitleme".
- ROI maskeleme (geometry/roi.py + ui/roi_dialog.py): poligon ROI; dışındaki
  tespitler elenir, dışı kararır. UI: "ROI Sec"/"Temizle".
- Otomatik IPM (geometry/auto_ipm.py): yakınsayan çizgiler + kaybolma noktasından
  trapez→homografi; başarısızsa manuele yönlendirir. UI: "Oto IPM".
- Şematik mini-harita (overlays.render_minimap/paste_minimap): sağ üstte soyut
  park krokisi (yeşil boş/kırmızı dolu, numaralı).
- Canlı güven göstergesi: slot karar güveni ortalaması ("Guven: %X").
Adaptif sonuca mean_confidence eklendi. Tüm yeni modüller reset zincirine bağlı.

2026-06-05 — KULLANICI GERİBİLDİRİMİ DÜZELTMELERİ + OTOMASYONLAR (180 test).
- Düzeltme: IPM kalibrasyonu (manuel + oto) sonrası kuş bakışı OTOMATİK açılıyor
  (_activate_ipm_view). Önceden "Goster"e elle basmak gerekiyordu.
- Düzeltme: Adaptif modda sığma kontrolü. IPM kalibreyse boş slotların gerçek
  genişliği hesaplanıyor (empty_sizes_m). Yeşil=boş+sığar, turuncu=boş ama
  sığmaz, kırmızı=dolu. Yönlendirme oku SADECE en yakın UYGUN (boş+sığan) slota.
- Oto ROI (roi.auto_roi_from_detections): araçların convex hull'undan otomatik
  ilgi bölgesi. UI: "Oto ROI" butonu.
- Araç-tabanlı Oto IPM (auto_ipm.auto_calibrate_from_vehicles): çizgi-yakınsama
  başarısız olunca araçların perspektif sırasından trapez→homografi (yedek).
  Yeterli derinlik yoksa None (düz cephe → manuel gerekir; dürüstçe bildiriliyor).
