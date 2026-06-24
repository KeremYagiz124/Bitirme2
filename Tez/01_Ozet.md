ÖZET
================================================================

KAMERA GÖRÜNTÜLERİNDEN ARAÇ TESPİTİ VE PARK UYGUNLUĞU ANALİZİ İÇİN YAPAY ZEKÂ TABANLI SİSTEM

Kerem Yağız KARAKAŞ, Ahmet EKŞİOĞLU, Ezgi KIRNAPÇI

(Danışman: Prof. Dr. Selçuk KAVUT)

Balıkesir, 2026

Kentsel alanlarda artan araç yoğunluğu ve yetersiz park altyapısı; sürücülerin boş park yeri ararken zaman, yakıt ve karbon emisyonu kaybetmesine yol açmakta, trafik akışını olumsuz etkilemektedir. Mevcut akıllı park sistemlerinin önemli bir bölümü her park yerine yerleştirilen gömülü sensörlere dayanır; bu yaklaşım yüksek kurulum ve bakım maliyeti getirmekte, elektrikli araçların yaygınlaşması ile manyetik sensörlerde güvenilirlik sorunları doğurmaktadır. Bu tez kapsamında, ek donanım gerektirmeden yalnızca tek bir kamera görüntüsünden çalışan, park yerlerinin doluluğunu ve uygunluğunu gerçek zamanlı çözümleyen yapay zekâ tabanlı bir sistem geliştirilmiştir. Sistem, araçları derin öğrenme tabanlı bir nesne tespit modeliyle saptamakta; boş park yerlerini, boyalı şerit bulunduğunda ızgara tabanlı, bulunmadığında geometri tabanlı çalışan adaptif bir yöntemle belirlemektedir. Ters perspektif dönüşümü ile görüntü kuş bakışına çevrilerek çapraz açı bozulması giderilmekte ve park yerlerinin gerçek metrik boyutları ölçülmektedir. Ölçülen boyutlar üzerinden aracın boş yere sığıp sığamayacağı denetlenmekte; çok kriterli bir karar motoru sürücüye en uygun boş yeri önermektedir. Sisteme ayrıca araç takibi, gece görüşü iyileştirme, monoküler derinlik kestirimi ve çift yönlü çalışan çevrimdışı sesli asistan gibi sürücü-asistanı özellikleri entegre edilmiştir. Geliştirilen yöntem, etiketli test görüntüleri ve sentetik senaryolarla değerlendirilmiş; gündüz ve çizgili park koşullarında boş/dolu ayrımını yüksek isabetle gerçekleştirmiştir. Sonuçlar, mevcut kamera altyapısıyla çalışan düşük maliyetli ve ölçeklenebilir bir akıllı park çözümünün uygulanabilirliğini ortaya koymaktadır.

ANAHTAR KELİMELER: Araç Tespiti, Park Yeri Analizi, Derin Öğrenme, Bilgisayarlı Görü, Ters Perspektif Dönüşümü, Sürücü Destek Sistemleri.


================================================================
ABSTRACT
================================================================

AN ARTIFICIAL INTELLIGENCE BASED SYSTEM FOR VEHICLE DETECTION AND PARKING SUITABILITY ANALYSIS FROM CAMERA IMAGES

Kerem Yağız KARAKAŞ, Ahmet EKŞİOĞLU, Ezgi KIRNAPÇI

(Supervisor: Prof. Dr. Selçuk KAVUT)

Balıkesir, 2026

The increasing vehicle density and insufficient parking infrastructure in urban areas cause drivers to waste time, fuel and carbon emissions while searching for an empty parking space, and negatively affect traffic flow. A significant portion of existing smart parking systems rely on embedded sensors placed at each parking space; this approach incurs high installation and maintenance costs, and introduces reliability problems for magnetic sensors as electric vehicles become widespread. In this thesis, an artificial intelligence based system that operates solely from a single camera image without requiring any additional hardware, and that analyses the occupancy and suitability of parking spaces in real time, has been developed. The system detects vehicles with a deep learning based object detection model, and determines empty parking spaces with an adaptive method that operates in a grid based manner when painted lane markings are present, and in a geometry based manner when they are absent. By means of inverse perspective mapping, the image is transformed into a bird's eye view, eliminating oblique angle distortion and enabling the measurement of the real metric dimensions of parking spaces. Based on the measured dimensions, it is checked whether the vehicle fits into the empty space, and a multi criteria decision engine recommends the most suitable space to the driver. The system is further integrated with driver assistance features such as vehicle tracking, night vision enhancement, monocular depth estimation and a bidirectional offline voice assistant. The developed method has been evaluated with labelled test images and synthetic scenarios, and has distinguished empty and occupied spaces with high accuracy under daytime and lined parking conditions. The results demonstrate the feasibility of a low cost and scalable smart parking solution that operates with the existing camera infrastructure.

KEYWORDS: Vehicle Detection, Parking Space Analysis, Deep Learning, Computer Vision, Inverse Perspective Mapping, Driver Assistance Systems.
