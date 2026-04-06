# 3-Person Team Setup Checklist

3 kişi projeyi başlattıklarında kontrol listesi.

## ✅ Before First Push (Her kişi yapmalı)

### Person A:
- [ ] Repo klonlandı
- [ ] Python venv oluşturuldu
- [ ] `pip install -r requirements.txt` çalıştırıldı
- [ ] `configs/config.local.yaml` oluşturuldu
- [ ] `python examples_multi_user_config.py` çalıştırıldı (test)
- [ ] `python run.py` ile UI başlatıldı ✓ (hatasız)
- [ ] VS Code `.gitignore` ayarlandı (Settings: exclude patterns)

### Person B:
- [ ] (A ile aynı adımlar)

### Person C:
- [ ] (A ile aynı adımlar)

## ✅ Development Start

### Week 1 - Setup & Planning:
- [ ] `SETUP.md` tüm kişiler tarafından okundu
- [ ] `COLLABORATION.md` tüm kişiler tarafından okundu
- [ ] Branching strategy anlaşıldı
- [ ] İşbölümü kararı alındı

### Week 2 - Task Distribution:

**Person A - Synthetic Data & Fine-tuning:**
- [ ] `scripts/synthetic_data_generator.py` test edildi
- [ ] `scripts/fine_tune_yolo.py` test edildi
- [ ] Feature branch açıldı: `feature/yolo-finetuning`
- [ ] İlk commit yapıldı
- [ ] PR oluşturuldu

**Person B - YOLOv8 Detection & Parking Zones:**
- [ ] Parking zone detection modülü başlatıldı
- [ ] `src/parking/zone_detector.py` oluşturuluyor
- [ ] Feature branch: `feature/parking-zones`
- [ ] Initial commit yapıldı

**Person C - UI & Visualization:**
- [ ] `src/ui/main_window.py` review'lendi
- [ ] Model detection visualization ekleniyor
- [ ] Feature branch: `feature/ui-improvements`
- [ ] Initial commit yapıldı

## ✅ Daily Workflow

### Her şiftin başında:
```bash
git pull origin main
```

### Gün sonunda:
```bash
git add .
git commit -m "feat: açıklama"
git push origin feature/your-branch
```

### PR hazırken:
- [ ] `git pull origin main` yapıldı (conflict kontrol)
- [ ] Commit messages temiz
- [ ] `.gitignore` kurallarına uyuldu
- [ ] No absolute paths
- [ ] Tests pass (eğer varsa)

## ✅ Before Main Merge

- [ ] Gözden geçiriş talebinde bulunuldu
- [ ] Review'ci değişiklikleri onayladı
- [ ] Çatışmalar çözüldü (eğer varsa)
- [ ] CI/CD pass'lerini kontrol et (eğer kuruluysa)

## ✅ After Merge

- [ ] Branch lokal silindi: `git branch -D feature/...`
- [ ] Branch remote silindi
- [ ] `git pull origin main` ile senkronize et
- [ ] Diğer kişiler bilgilendirildi

## ⚠️ Troubleshooting

**"ModuleNotFoundError: No module named 'src'"**
```bash
python run.py  # (project root'dan)
```

**"Merge conflict"**
```bash
git pull origin main
# Dosyaları düzelt
git add .
git commit -m "resolve: merge conflicts"
git push
```

**"Committed'e .pt dosyası gitti"**
```bash
git rm --cached models/*.pt
git commit -m "remove: committed model files"
```

**"Config.local.yaml gittrack'lenmiş"**
```bash
git rm --cached configs/config.local.yaml
git commit -m "fix: remove tracked config.local.yaml"
```

## 📞 Communication

- **Büyük değişiklikler:** PR'da açıkla
- **Çatışma olabilir:** Slack'te haber ver
- **Blocked edildim:** Issue aç veya haber ver

---

**Last updated:** April 2026