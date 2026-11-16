# CLAUDE.md — xG-Vision (CNN ile görüntüden xG)

Bu dosya Claude Code için proje bağlamıdır. Her oturumda önce bunu oku.

## Çalışma şekli
- Kullanıcıyla **Türkçe** konuş. Kod içi yorumlar, docstring'ler ve commit mesajları İngilizce.
- Proje **adım adım** ilerliyor. Bir adım bitmeden sonrakine geçme. Her adımın sonunda sonuçları
  (metrikler, grafikler) kullanıcıya göster ve birlikte karar ver.
- Önemli bir tasarım seçimi varsa (model mimarisi, loss, görüntü boyutu, kanal ekleme/çıkarma...)
  seçenekleri ve önerini kısaca yaz, kullanıcıya sor.
- Ortam: yerel Windows, proje klasörü `Masaüstü\xG-Vision`, sanal ortam `.venv` (Python 3.13,
  PyTorch 2.11 + CUDA 12.8). GPU: **RTX 4050 Laptop, 6 GB VRAM**, yani batch size ve model boyutunu buna göre seç
  (gerekirse AMP / mixed precision kullan). Gerekirse Google Colab da kullanılabilir.
- Repo: https://github.com/Huseyinyagmur/xG-Vision (dal: `main`).
- Portföy projesi: kod okunaklı, modüler ve tekrarlanabilir olmalı (seed'ler sabit, yollar config'te).

## Git kuralları
- Sık commit atma. **Anlamlı bir iş birimi bittiğinde ve çalıştığı doğrulandığında** commit at
  (ör. "dataset pipeline works", "baseline model + training loop"). Her küçük düzenleme için commit yok.
- Mesaj formatı: Conventional Commits, İngilizce, kısa (`feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`).
- Bir yol haritası adımı tamamlanıp kullanıcı onayladığında `push` et.
- `data/`, model ağırlıkları (`*.pt`, `*.pth`) ve büyük dosyalar **asla** commit edilmez (`.gitignore`'da).
  Küçük sonuç grafikleri (`results/plots/*.png`) commit edilebilir, README'de gösterilecek.
- Force push, history rewrite yapma.

## Proje
Şut anındaki tüm oyuncu konumları (StatsBomb "freeze frame") kuşbakışı çok kanallı bir saha
tensörüne çevrilir. Bir CNN bu tensörden şutun gol olma olasılığını (xG) tahmin eder. Model,
StatsBomb'un kendi xG'si ve basit bir baseline ile kıyaslanır.

Kullanıcının önceki projesi (yapı referansı): https://github.com/Huseyinyagmur/AI-Surface-Defect-Inspection
(PyTorch, ResNet18 transfer learning, `src/` altında modüler dosyalar).

### Yol haritası
1. ✅ **Veri:** StatsBomb şutları + freeze frame → temizlik → çok kanallı tensör → görselleştirme.
2. ✅ **Model:** sıfırdan küçük bir CNN ile ön eğitimli bir modeli (ör. ResNet18, ilk conv 6 kanala
   uyarlanmış) karşılaştır. Eğitim döngüsü, early stopping, en iyi modeli kaydetme.
3. ⏭️ **Değerlendirme:** maç bazında test, log loss, Brier, ROC-AUC, kalibrasyon grafiği; StatsBomb xG
   ve baseline (sadece mesafe + açı ile lojistik regresyon) ile karşılaştırma; ilginç test pozisyonları; Grad-CAM.
4. **Mantık testleri:** şutçuyu yaklaştır/uzaklaştır, araya savunmacı koy, kaleciyi çıkar → xG beklenen
   yönde değişiyor mu? Şutçuyu ceza sahasının her noktasına koyup xG ısı haritası.
5. **İnteraktif taktik tahtası:** Streamlit veya Gradio, oyuncuları sürükle-bırak → anlık xG.
6. (İsteğe bağlı) Yayın karesi → YOLO ile oyuncu tespiti → saha anahtar noktaları + homografi → kuşbakışı → xG.

### Değerlendirme ilkeleri
- Sınıflar dengesiz (~%10 gol). Ana metrikler: **log loss, Brier, ROC-AUC, kalibrasyon**. Accuracy ana metrik değil.
- xG bir olasılıktır, kalibrasyon önemli. Class weight / oversampling kalibrasyonu bozar. Kullanılırsa
  sonradan kalibre et ya da düz BCE tercih et.
- Bölme **maç bazında** yapılır, aynı maçın şutları farklı bölümlerde olamaz (`split` sütunu sabit, değiştirme).

## Mevcut durum (2. adım tamamlandı)

1. adım `0eff913` ile push edildi. 2. adım commit'leri: `06d6435` (CLAUDE.md), `9484875` (render
hızlandırma), `3439ffc` (modeller + eğitim döngüsü) ve 2. adımın README/CLAUDE.md commit'i.
README İngilizce ve portföy odaklı; her adım bitince README'nin yol haritası ve sonuçlar bölümü güncellenmeli.

### Dosyalar
```
src/config.py         # yollar, saha geometrisi, kanallar, split, seed + eğitim ayarları (MODEL_CONFIGS, BN_RECAL_BATCHES...)
src/data_download.py  # StatsBomb indir -> şutlar -> temizlik -> maç bazlı split -> data/shots.pkl
src/pitch_tensor.py   # (şutçu, freeze frame) -> (6, 160, 120) float32 tensör
src/dataset.py        # load_shots(split), ShotDataset -> (image, extra, y, idx), PhaseBatchSampler
src/features.py       # extra_features(df): vücut bölgesi one-hot [head, foot, other]
src/models.py         # SmallCNN, ResNet18xG, XGHead, build_model(name), load_checkpoint(path)
src/train.py          # eğitim döngüsü, precise BN, early stopping, checkpoint, log, eğri grafiği
src/evaluate.py       # compute_metrics, metrics_by_group (cinsiyet), reference_predictions; `python -m src.evaluate`
src/utils.py          # seed_everything, get_device, tee_output (log dosyası)
src/visualize.py      # saha çizimleri, kanal grafikleri, veri özeti; `python -m src.visualize`
tests/                # render = eski referans, mirror_y = satır çevirme, modeller, sampler (11 test)
results/plots/        # veri grafikleri + training_curves_small_cnn.png, training_curves_resnet18.png
models/<run>/         # best.pt, last.pt, history.csv, config.json, train.log, val_predictions.csv (git'te yok)
```

### Çalıştırma
```bash
pip install -r requirements.txt
python -m src.data_download     # ~4.000 maç, data/cache/ altında maç başına önbellek, tekrar çalıştırılabilir
python -m src.visualize         # grafikleri results/plots/ altına yazar
python -m src.evaluate          # StatsBomb xG + sabit tahmin metrikleri (val, test)
python -m src.train --model small_cnn | resnet18 | resnet18_nomaxpool   [--subset N --epochs E]
python -m src.train --bench-loader --bench-workers 0 6                  # DataLoader hızı
python -m pytest tests -q       # shots.pkl yoksa gerçek veri testleri atlanır
```
Veri GitHub raw üzerinden çekiliyor (`STATSBOMB_RAW`). statsbombpy kullanılmıyor (daha yavaş).

### Veri tablosu (`data/shots.pkl`, her satır bir şut)
Sütunlar: `shot_id, match_id, competition, season, gender, match_date, period, minute, team, player,
x, y, shot_type, body_part, technique, outcome, statsbomb_xg, first_time, one_on_one, under_pressure,
freeze_frame (np.ndarray N×4: [x, y, teammate, is_gk]), n_players, has_gk, is_goal, distance, angle, split`.

- Penaltılar, freeze frame'i olmayan şutlar ve x < 60 olan şutlar çıkarıldı.
- Tablodaki şut türleri: Open Play 95.160, Free Kick 4.443, Corner 33, Kick Off 3.
  `load_shots()` sadece `TRAIN_SHOT_TYPES` (varsayılan `("Open Play",)`) döndürür; diğerleri tabloda kalır, eğitime girmez.
- `distance` (kale merkezine yarda), `angle` (kale ağzının görüş açısı, radyan) → baseline için hazır.
- `body_part` değerleri (açık oyun): Right Foot 49.898, Left Foot 28.926, Head 16.036, Other 300.

### Tensör tasarımı
- Koordinatlar: StatsBomb 120×80 yarda. Hücum her zaman x = 120'deki kaleye doğru. Direkler y = 36 ve 44.
- Bölge: hücum yarı sahası, x ∈ [60, 120], y ∈ [0, 80]. 0,5 yd/px → **H = 160 (y), W = 120 (x)**.
- **Satır 0 = y = 0 = mplsoccer StatsBomb sahasının üst kenarı.** Çizimde `origin="upper"`,
  `extent=[X_MIN, X_MAX, Y_MAX, Y_MIN]` kullan (ilk denemede bu yüzden görüntüler ters çıkmıştı).
- Kanallar: 0 şutçu, 1 takım arkadaşları, 2 rakip saha oyuncuları, 3 rakip kaleci (σ = 1 yd Gauss lekeleri,
  üst üste binince toplanır), 4 şut üçgeni (şutçu → iki direk, ikili maske), 5 sabit kaleye-mesafe haritası
  (CoordConv benzeri konum bilgisi).
- Tensörler diske yazılmıyor, `ShotDataset` içinde anlık çiziliyor: **~1,2 ms/örnek** (tek çekirdek).
  Gauss lekeleri oyuncu etrafında ±4σ pencerede, şut üçgeni sınır kutusunda hesaplanıyor (eski tam ızgara
  sürümü ~10–14 ms'ydi; testlerde referans olarak duruyor, fark < 1e-3). Tümünü saklamak ~44 GB tutardı.
- Şut üçgeninde tam kenar üstündeki pikseller içeride sayılır (`EDGE_EPS`); böylece aynalama birebir tutarlı.
- Augmentation: y → 80 − y aynalama (kale simetrik). Test: aynalanmış şutun tensörü = orijinalin satırları çevrilmiş hali.
  Vücut bölgesinde sol/sağ ayak birleşik (`foot`), bu yüzden extra girdi aynalamadan etkilenmez.

### Model ve eğitim (2. adım)
- API: `forward(image, extra) -> logit`; `extra` = vücut bölgesi one-hot (3). GAP → concat(extra) →
  Linear(·, 64) → ReLU → Dropout 0,3 → Linear(64, 1). Çıkış bias'ı logit(eğitim gol oranı) ile başlar.
- **SmallCNN:** 5 × [conv-BN-ReLU ×2 + maxpool], 16-32-64-128-256 kanal, 1,20 M parametre. (İlk denemede
  32 ile başlıyordu: tam çözünürlükte 4,9 GB VRAM, iki kat yavaş. Kullanıcı 16'yı onayladı.)
- **ResNet18xG:** ImageNet ağırlıkları; conv1 6 kanal (RGB filtre ortalaması × 3/6). 11,2 M parametre.
  LR: conv1 + head 1e-3, ön eğitimli katmanlar 1e-4. `resnet18_nomaxpool` varyantı config'te var, eğitilmedi.
- Düz BCE (class weight yok), AdamW (wd 1e-4), ReduceLROnPlateau (×0,5, patience 2), bf16 autocast,
  batch 256, early stopping val log loss (patience 5, en fazla 40 epoch).
- **Precise BN:** her val'den önce BN istatistikleri sabit ağırlıklarla 40 eğitim batch'i üzerinden yeniden
  hesaplanır (`recalibrate_bn`). Neden: BN'nin biriken (momentum) istatistikleri hızla değişen ağırlıkların
  gerisinde kalıyordu; kısa testte `eval()` modunda ortalama tahmin 0,033 (eğitim alt kümesinde de aynı,
  yani ezberleme değil), BN batch istatistikleriyle 0,111, gerçek oran 0,100. Düzeltmeden sonra iki mod da ~0,09.
- **Veri yükleme (Windows'a özel dersler):** worker'lar spawn ile her seferinde sıfırdan başlıyor (~10 sn).
  Bu yüzden train+val tek bir kalıcı DataLoader'da (`ShotLoader` + `PhaseBatchSampler`, fazlar: train / bn / val),
  num_workers = 6; görüntüler worker'dan float16 gelir, GPU'da float32'ye çevrilir; worker'larda BLAS/OMP tek
  iş parçacığı (yoksa OpenBLAS bellek hatası, Windows hata 1455). 8+ worker veya iki ayrı kalıcı loader
  bellek ayırma payını aşıyor. torchvision ve matplotlib geç yükleniyor (worker'lar train.py'yi yeniden import eder).
- Aynalama kararını ana süreçteki sampler verir → augmentation worker sayısından bağımsız, tekrarlanabilir.
- Hız (RTX 4050 Laptop): ilk epoch ~105 sn (worker başlatma + cuDNN autotune), sonra ~40–45 sn/epoch.
  Tepe VRAM: SmallCNN 2,56 GB, ResNet18 1,62 GB, maxpool'suz ResNet18 ~3,4 GB (sentetik ölçüm).
- Ağır işler (oyun vb.) açıkken eğitim çok yavaşlıyor / bellek hatası veriyor; ölçümden önce kontrol et.
- stdin'den (`python - <<EOF`) worker'lı DataLoader çalıştırılamaz (spawn ana dosyayı bulamaz); betik dosyası kullan.

### Sonuçlar (16 Kasım 2026'da yerelde indirilen veri)
- 3.961 maç, **95.160 açık oyun şutu**, gol oranı **%10,4**.
- Split (lig bazında tabakalı): train 66.754 / val 14.240 / test 14.166 şut.
- Erkek ~62 bin, kadın ~33 bin şut. Kalecisi olmayan freeze frame: ~%0,1.
- Mesafeye göre gol oranı ile StatsBomb xG ortalaması neredeyse örtüşüyor (veri doğru okunmuş).
- **Geçilmesi gereken çıta, StatsBomb xG (test):** log loss **0,2778**, Brier **0,0799**, ROC-AUC **0,8065**.
- Sabit tahmin (eğitim gol oranı), test: log loss 0,3478, Brier 0,0983.
- StatsBomb verisi zaman zaman güncelleniyor; yeniden indirince rakamlar çok az değişebilir.
  Bu yüzden karşılaştırmalar hep aynı `shots.pkl` üzerinden yapılmalı.

**2. adım, val (14.240 şut, gol oranı 0,1033). Model seçimi val'de yapıldı → CNN'ler burada hafif avantajlı:**

| Model | Log loss | Brier | ROC-AUC | Ort. xG | En iyi / son epoch | Süre |
|---|---|---|---|---|---|---|
| SmallCNN | **0,2656** | **0,0765** | **0,8108** | 0,109 | 9 / 14 | 10,8 dk |
| ResNet18 (maxpool'lu) | 0,2690 | 0,0772 | 0,8060 | 0,099 | 5 / 10 | 8,3 dk |
| StatsBomb xG | 0,2629 | 0,0753 | 0,8138 | 0,100 | – | – |
| Sabit tahmin | 0,3323 | 0,0926 | 0,5000 | 0,103 | – | – |

- Cinsiyete göre (val): SmallCNN erkek 0,2617 / 0,0755 / 0,819, kadın 0,2728 / 0,0782 / 0,796;
  ResNet18 erkek 0,2660 / 0,0765 / 0,812, kadın 0,2746 / 0,0785 / 0,795 (log loss / Brier / AUC).
- SmallCNN düzenli iyileşiyor, epoch 9'dan sonra düzleşiyor. ResNet18 epoch 5'ten sonra ezberliyor
  (train loss 0,228'e iniyor, val 0,294'e çıkıyor). Ön eğitim burada avantaj sağlamadı.
- Maxpool'suz ResNet18 henüz eğitilmedi; sonuçlara bakıp kullanıcıyla karar verilecek.

### Kullanıcının onayladığı kararlar
1. **Vücut bölgesi → skaler girdi.** CNN görüntüden özellik çıkarır; global pooling sonrası vektöre
   vücut bölgesi eklenip sonra tam bağlantılı katmana verilir. Kodlama: `head / foot / other` one-hot
   (sol/sağ ayak ayrımı yapılmaz, böylece y-aynalama augmentation'ı geçerli kalır). Model API'si
   `forward(image, extra_features)` biçiminde; ileride `first_time`, `under_pressure` gibi
   ek skalerler aynı yoldan eklenebilir (eklemeden önce kullanıcıya sor).
2. **Kadın ve erkek maçları birlikte eğitilir.** Değerlendirmede metrikler toplam + erkek/kadın ayrı raporlanır.
3. **Sadece açık oyun şutları** (`TRAIN_SHOT_TYPES = ("Open Play",)`). Frikikler tabloda kalır, eğitime girmez.
4. 2. adımda onaylananlar: render hızlandırma (A seçeneği), dataset dönüşü `(image, extra, y, idx)`,
   README'de 6 adımlık yol haritası, float16 aktarım, tek iş parçacıklı BLAS, num_workers = 6,
   SmallCNN 16-32-64-128-256, precise BN, ResNet18 önce maxpool'lu.

### 3. adım için notlar
- Test metrikleri `src/evaluate.py` içindeki `compute_metrics` / `metrics_by_group` ile hesaplanmalı (CNN'ler,
  StatsBomb xG, sabit tahmin ve henüz yazılmamış mesafe + açı lojistik regresyonu aynı fonksiyonla).
- Checkpoint'ler precise-BN istatistiklerini içerir; `load_checkpoint` ile yüklenip doğrudan `eval()`'de kullanılabilir.
- Test tahmininde aynalama TTA (orijinal + aynalanmış ortalaması) denenebilir; eklemeden önce kullanıcıya sor.
