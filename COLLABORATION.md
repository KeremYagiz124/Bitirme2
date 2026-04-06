# GitHub Collaboration Guidelines

3 kişi tarafından aynı projeye katkı yapanlar için kurallar.

## 1. Repository Setup

Her kişi **bir kez** yapmalı:

```bash
git clone <repo-url>
cd Bitirme2
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configuration

Her kişi kendi lokal config'i oluşturmalı (`gitignore`'a alındığından güvenli):

```bash
cp configs/config.example.yaml configs/config.local.yaml
# Kendi makinenize göre config.local.yaml'ı düzenleyin
```

**config.local.yaml örneği:**
```yaml
# configs/config.local.yaml
training:
  device: "cuda"  # Eğer GPU varsa
  batch_size: 32   # Kendi makinenize göre ayarlayın

data:
  raw_dir: "/fast_drive/data/raw"  # Lokal hızlı disk
```

## 3. Branching Strategy

```
main (production - hiçbir kod direkt push yapılmaz)
  ├── feature/parking-detection  (Person A)
  ├── feature/ui-improvements    (Person B)
  └── feature/training            (Person C)
```

### Yeni bir feature başlarken:

```bash
# Ana branch'i güncelleyin
git pull origin main

# Yeni branch oluşturun
git checkout -b feature/your-feature-name

# Kodunuzu geliştirin
git add .
git commit -m "feat: açıklamalar"
git push origin feature/your-feature-name

# GitHub'da Pull Request açın
```

## 4. Pull Request Workflow

1. **Açıklaması yazılı PR oluşturun**
   - Ne yaptığınızı açıklayın
   - İlgili issue varsa reference'ı ekleyin

2. **Gözden geçiriş bekleyin**
   - En az 1 kişi review'lemeli
   - Önerilen değişiklikleri yapın

3. **Merge etmeden önce çatışmaları çözün**
   ```bash
   git fetch origin
   git rebase origin/main
   # Çatışmaları manuel çöz
   git add .
   git rebase --continue
   git push -f origin feature/your-feature-name
   ```

4. **Merge edin**
   - Branch'inizi delete edin

## 5. Çatışma Çözme

**Eğer birden fazla kişi aynı dosyaya yazıyorsa:**

```bash
# Pull'ü deneyin
git pull origin main
# Çatışma varsa, çatışmalı dosyaları açıp manual düzeltin
# Sonra:
git add .
git commit -m "resolve: merge conflicts with main"
git push
```

## 6. Yapılmadık (DO NOT):

- ❌ `main`'e direkt push yapmayın
- ❌ `git force push` yapmayın (feature branch hariç)
- ❌ Mutlak user path'leri commitlemeyin
- ❌ Model dosyaları (.pt, .pth) commitlemeyin
- ❌ Kişisel IDE ayarlarını commitlemeyin

## 7. Yapılacak (DO):

- ✅ Descriptive commit messages yazın
- ✅ REL ATIVE paths kullanın
- ✅ `.gitignore` kurallarına uyun
- ✅ PR açmadan önce `git pull origin main` yapın
- ✅ Büyük dosyaları `.gitignore`'a ekleyin

## 8. Commit Message Format

```
feat: yeni özellik açıklaması
fix: hata düzeltmesi
refactor: kod reorganizasyonu
docs: dokümantasyon
test: test ekleme

Örnek:
feat: YOLOv8 fine-tuning script eklendi
  - Sentetik veriye uyarlama
  - Validation metrikleri
  - Save checkpoint her 10 epoch

Closes #123
```

## 9. İşbölümü (Week 2 example)

| Görev | Person | Branch |
|-------|--------|--------|
| YOLOv8 fine-tuning | Person A | `feature/yolo-finetune` |
| Parking zone detection | Person B | `feature/parking-zones` |
| UI improvements | Person C | `feature/ui-v2` |

Her branc'h paralel geliştirilir, sonra PR ile merge'lenir.

## 10. Emergency (Acil Durum)

**Eğer main'de sorun varsa:**

```bash
# Problemi gör
git log --oneline main

# Önceki commit'e dön
git revert <commit-id>

# Yeni PR'da fix'le
git checkout -b fix/emergency-issue
```

---

**İhtiyaç duyarsan:** Slack/Discord'da yazabilirsin veya Issue açabilirsin.