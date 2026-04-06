# ÖZET

Kentsel alanlarda araç sayısının hızla artması, park altyapısının yetersiz kalmasına ve park yönetiminin giderek daha karmaşık bir hal almasına yol açmaktadır. Mevcut sistemler park alanlarının doluluk durumunu ve yasak bölge ihlallerini gerçek zamanlı olarak izleyememekte; denetim büyük ölçüde manuel kontrole dayanmaktadır.

Bu projede yapay zeka tabanlı bir araç tespit ve park analizi sistemi geliştirilmiştir. Sistem iki temel işlevi yerine getirmektedir: YOLOv8 derin öğrenme modeli ile kamera görüntüsünden gerçek zamanlı araç tespiti ve poligon tabanlı bölge etiketleme aracıyla tanımlanmış park alanlarının IoU (Intersection over Union) hesabıyla doluluk analizi.

Proje kapsamında Microsoft COCO veri setiyle önceden eğitilmiş YOLOv8n modeli transfer learning yöntemiyle kullanılmıştır. Park senaryosuna özel eğitim için 1000 adet sentetik görüntü otomatik olarak üretilmiş ve YOLO formatında etiketlenmiştir. Park alanları, geliştirilen poligon tabanlı etiketleme aracıyla görsel olarak tanımlanabilmektedir. PyQt5 çerçevesiyle gerçek zamanlı görselleştirme sağlayan bir masaüstü arayüzü geliştirilmiştir.

İlk testlerde YOLOv8n modeli araçları %80-95 güven skoru aralığında başarıyla tespit etmiştir. Sentetik veri üzerinde fine-tuning sonrasında mAP@0.5 değeri %99.5'e ulaşmıştır. GPU desteğiyle 720p çözünürlükte 18-22 FPS gerçek zamanlı performans elde edilmiştir. Sistem bilgisayar kamerası, telefon kamerası veya harici kamera ile çalışabilmektedir.

Çalışmanın devam eden aşamalarında IoU pipeline'ının arayüze entegrasyonu ve gerçek kamera görüntüleriyle test planlanmaktadır.

**Anahtar Kelimeler:** Park tespiti, YOLOv8, nesne tespiti, IoU analizi, derin öğrenme, görüntü işleme, transfer learning
