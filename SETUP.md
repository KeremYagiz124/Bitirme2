# Kurulum Kılavuzu - Çok Kişili Geliştirme

Bu proje 3+ kişi tarafından aynı anda geliştirilebilmesi için tasarlanmıştır.

## 1. Başlangıç Kurulumu (Her Geliştirici)

### 1.1 Depoyu Klonlayın
```bash
git clone https://github.com/your-username/Bitirme2.git
cd Bitirme2
```

### 1.2 Python Ortamını Oluşturun
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 1.3 Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

## 2. Konfigürasyon (Kullanıcı Spesifik)

### 2.1 Yerel Konfigürasyon Oluşturun

`configs/config.local.yaml` dosyası oluşturun (GIT'e yüklenmeyecek):

```yaml
# configs/config.local.yaml
# Bu dosya her kullanıcıda farklı olabilir

data:
  synthetic_dir: "./data/synthetic"  # Yerel veri dizini
  raw_dir: "./data/raw"

model:
  yolo:
    cache_dir: "./models/cache"  # Modelin cache'i

# Opsiyonel: İş makinesi spesifik ayarlar
ui:
  default_window_width: 1280
  default_window_height: 720
```

**Not:** `config.local.yaml` `.gitignore`'da yer aldığı için sadece lokal makinenizde kalır.

## 3. Veri Dizinleri Ayarı

Her kullanıcı kendi lokal ortamında bu dizinleri oluşturabilir:

```
Bitirme2/
├── data/
│   ├── raw/              # Orijinal veri setleri
│   ├── synthetic/        # Üretilen sentetik veriler
│   └── processed/        # İşlenmiş veriler
├── models/
│   ├── cache/            # YOLO model cache'i
│   └── fine_tuned/       # Eğitilmiş modeller
└── runs/                 # Antrenman çıktıları
```

## 4. Kod Yazarken Dikkat Edilecekler

### ✅ YAPILMALI:
- **Relative paths kullanın:** `./data/`, `./models/` gibi
- **Config kullanın:** `from src.utils.config import config`
- **Lokal dosyalar:** `configs/config.local.yaml`'a yazın

### ❌ YAPILMAMALI:
- Absolute paths: `C:\Users\eksi\...` ❌
- Hardcoded paths ❌
- Kullanıcı adı barındıran paths ❌

## 5. Kod Örneği

```python
# ✅ Doğru
from src.utils.config import config

model_dir = config.get_path("model.yolo.cache_dir")
data_dir = config.get_path("data.synthetic_dir")

model_size = config.get("model.yolo.model_size")
```

```python
# ❌ Yanlış
model_dir = Path("C") / "Users" / "eksi" / "data"  # Absolute path
data_dir = "C:\Users\eksi\Documents\data"  # User-specific path
```

## 6. Git Workflow

### Yeni bir özellik başlarken:
```bash
git pull origin main
git checkout -b feature/your-feature
# Kodunuzu geliştirin
git add .
git commit -m "feat: açıklama"
git push origin feature/your-feature
# Pull request oluşturun
```

### Çatışmaları çözme:
Birden fazla kişi aynı dosyada çalışıyorsa:
```bash
git pull origin main
# Çatışmaları manuel düzeltip
git add .
git commit -m "resolve: merge conflicts"
git push
```

## 7. IDE Ayarları (VS Code - Opsiyonel)

`.vscode/settings.json` dosyası **lokal** kalacak (gitignore'da):
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.formatting.provider": "black"
}
```

## 8. Sorun Giderme

**Q: "ModuleNotFoundError: No module named 'src'"**
```bash
# Python dosyalarını proje kökünden çalıştırın
python run.py
```

**Q: Veri dosyaları yüklü değil**
```bash
# Verileri lokal indirin
python scripts/download_datasets.py
```

**Q: Bağımlılık problemleri**
```bash
pip install --upgrade -r requirements.txt
```

## 9. Çalışma Planı (Task Dağılımı)

Her hafta aşağıdaki görevler paralel yapılabilir:

| Görev | Sorumlu | Dizin |
|-------|---------|-------|
| YOLOv8 Fine-tuning | Person A | `scripts/`, `models/` |
| Park Area Detection | Person B | `src/parking/` |
| UI/Visualization | Person C | `src/ui/` |

---

**Son güncelleme:** April 2026