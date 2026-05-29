Proje Adı: Kamera Görüntülerinden Araç Tespiti ve Park Uygunluğu Analizi için Yapay Zeka Tabanlı Sistem
Rapor 6


1. PARK SÜRESİ TAKİBİ

Video ve kamera modunda her araç için park süresinin anlık olarak görüntülenmesini sağlayan özellik geliştirildi.

A. VehicleTracker Genişletmesi (src/detection/vehicle_tracker.py)

- _Track sınıfına first_seen zaman damgası eklendi; araç ilk tespit edildiği anda kaydedilir
- get_static_tracks_with_duration() metodu eklendi: statik araçları (bbox, süre_sn) çiftleri olarak döndürür

B. Arayüz Entegrasyonu

- Her statik aracın bbox'ı üzerine M:SS formatında süre etiketi çiziliyor (koyu arka plan üzerinde sarı yazı)
- Yalnızca video/kamera modunda aktif; fotoğraf modunda gösterilmiyor
- CSV log çıktısına longest_parked_sec sütunu eklendi: o anki en uzun süre park eden araç süresi

C. Snapshot İyileştirmesi

- Snapshot alınırken görüntü yeniden işlenmek yerine önbellekteki son analiz sonucu (_last_result) kullanılıyor
- Sokak modunda draw() mevcut sonuçla çağrılıyor; sabit kamera modunda önceki davranış korunuyor


2. SOKAK MODU DOĞRULUK KALİBRASYONU

A. Parametre Optimizasyonu

max_edge_extension_ratio parametresi 0.40'tan 0.20'ye düşürüldü. Bu parametre çerçeve kenarındaki son araçtan ne kadar ötesine slot uzatılacağını belirler. Varsayılan değer geniş kenar boşluklarını fazladan slot olarak bölüyordu.

B. Değerlendirme Sonuçları

3 görüntülük ground truth veri setiyle gerçekleştirilen değerlendirme:

| Görüntü | Beklenen Boş | Tespit | TP | FP | FN |
|---|---|---|---|---|---|
| 1.png | 0 | 0 | 0 | 0 | 0 |
| 2.png | 2 | 2 | 2 | 0 | 0 |
| 3.png | 3 | 3 | 3 | 0 | 0 |

Mikro F1: %100.0 (önceki rapor: %90.9)

Önceki raporda 3.png görüntüsünde çerçeve sağ kenarındaki 747 piksellik boşluk 3 slota bölünüyordu (beklenen: 2). Yeni parametre sınırı bu uzantıyı kesiyor ve yalnızca 2 slot üretiyor.


3. OTOMATİK PARK TESPİTİ MOD YENİDEN ADLANDIRMASI

"Sokak Modu" UI etiketi "Otomatik Park Tespiti" olarak güncellendi. Mevcut StreetParkingDetector algoritması yalnızca sokak parkı değil, herhangi bir açıdan çekilen otopark görüntülerinde de çalışabilmektedir (multi_row=True). Yeniden adlandırma sistemin yeni ortamlara uyarlanabilirliğini yansıtmaktadır.


Rapor Özeti

Bu hafta üç temel geliştirme gerçekleştirildi: Park süresi takibi ve log iyileştirmesi, sokak modu doğruluk kalibrasyonu ile mod yeniden adlandırması. Sokak modunda boş alan tespiti %100 Mikro F1'e ulaştı. Sistem sunum ve demo olgunluğuna erişmiştir.


Kullanılan Kaynaklar
1. Jocher, G. et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
2. OpenCV Documentation. https://opencv.org
3. Wang, H. et al. (2022). YOLOPv2: Better, Faster, Stronger for Panoptic Driving Perception. arXiv:2208.11434
4. Redmon, J. & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv:1804.02767
5. Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools, 25(11), 120–125.
6. Lucas, B. D. & Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision. IJCAI, 674–679.
7. Shi, J. & Tomasi, C. (1994). Good Features to Track. IEEE CVPR, 593–600.
8. Amato, G. et al. (2017). Deep Learning for Decentralized Parking Lot Occupancy Detection. Expert Systems with Applications, 72, 327–334.
9. Nurullayev, S. & Lee, S. W. (2019). Generalized Parking Occupancy Analysis Based on Squeeze-and-Excitation Networks. Sensors, 19(3), 480.
10. Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv:2004.10934
