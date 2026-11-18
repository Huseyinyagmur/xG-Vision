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
3. ✅ **Değerlendirme:** maç bazında test, log loss, Brier, ROC-AUC, kalibrasyon grafiği; StatsBomb xG
   ve baseline (sadece mesafe + açı ile lojistik regresyon) ile karşılaştırma; ilginç test pozisyonları; Grad-CAM.
4. ✅ **Mantık testleri:** şutçuyu yaklaştır/uzaklaştır, araya savunmacı koy, kaleciyi çıkar → xG beklenen
   yönde değişiyor mu? Şutçuyu ceza sahasının her noktasına koyup xG ısı haritası.
5. ✅ **İnteraktif taktik tahtası:** Streamlit veya Gradio, oyuncuları sürükle-bırak → anlık xG.
6. (İsteğe bağlı) Yayın karesi → YOLO ile oyuncu tespiti → saha anahtar noktaları + homografi → kuşbakışı → xG.

### Değerlendirme ilkeleri
- Sınıflar dengesiz (~%10 gol). Ana metrikler: **log loss, Brier, ROC-AUC, kalibrasyon**. Accuracy ana metrik değil.
- xG bir olasılıktır, kalibrasyon önemli. Class weight / oversampling kalibrasyonu bozar. Kullanılırsa
  sonradan kalibre et ya da düz BCE tercih et.
- Bölme **maç bazında** yapılır, aynı maçın şutları farklı bölümlerde olamaz (`split` sütunu sabit, değiştirme).

## Mevcut durum (5. adım tamamlandı; HF Hub yüklemesi + push kullanıcı onayını bekliyor)

1. adım `0eff913` ile push edildi. 2. adım commit'leri: `06d6435` (CLAUDE.md), `9484875` (render
hızlandırma), `3439ffc` (modeller + eğitim döngüsü), `f7f3115` (README/CLAUDE.md, push edildi). 3. adım: `6d72da3` (push edildi). 4. adım: `083d3e5` (push edildi).
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
src/baselines.py      # lojistik regresyon: mesafe + açı (+ vücut bölgesi); `python -m src.baselines`
src/evaluate.py       # metrikler, maç bazlı bootstrap, kalibrasyon kutuları, rapor; `python -m src.evaluate --split val|test`
src/gradcam.py        # GradCAM (hook'larla), smallcnn_layer(model, blok)
src/interpret.py      # --compare-layers (val), --split test: CNN ile StatsBomb'un en çok ayrıştığı şutlar + Grad-CAM
src/perturb.py        # Shot(x, y, ff, body) + saf düzenlemeler (şutçu taşı, savunmacı ekle, kaleci çıkar/kaydır...), OOD işaretleri
src/sanity.py         # 4. adım mantık testleri (tüm test şutları), tanılama tabloları, grafikler; `--split test`
src/heatmap.py        # şutçu ızgarası üzerinde xG ısı haritaları (kaleci tek / kalabalık / kontra), CNN vs LogReg
src/board.py          # taktik tahtası mantığı (Streamlit'siz): px <-> yd, TTA'lı XGModel, LogRegJSON, uyarılar
src/export_assets.py  # app/assets/scenarios.json (boş + 5 kural bazlı test pozisyonu), logreg.json
app/                  # streamlit_app.py + kendi bileşenimiz board.js/board.css (components v2), assets/
scripts/ui_smoke.py   # isteğe bağlı Playwright testi (sistem Edge'i), pytest dışında
src/utils.py          # seed_everything, get_device, tee_output, free_memory_gb, pick_num_workers, is_memory_error
src/visualize.py      # saha çizimleri, kanal grafikleri, veri özeti; `python -m src.visualize`
tests/                # render, aynalama, modeller, sampler, ağırlıklı AUC / bootstrap / kalibrasyon, Grad-CAM
results/              # metrics_{val,test}.md|csv, interesting_test.md
results/plots/        # veri, eğitim eğrileri, calibration_*, gradcam_layers_val, interesting_cnn_*_test
models/<run>/         # best.pt, last.pt, history.csv, config.json, train.log, val_predictions.csv (git'te yok)
models/predictions_{val,test}.csv  # tüm modellerin + TTA + baseline'ların şut bazlı tahminleri (git'te yok)
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
- **Kullanıcı kuralı:** kullanıcı iş sırasında oyun oynayabilir (Valorant ~5 GB RAM + GPU). DataLoader'lı işlerden önce
  boş belleği kontrol et; boş RAM veya commit < 6 GB ise num_workers = 2 (`pick_num_workers`). Bellek hatasında
  worker sayısını düşürüp tekrar dene (`sanity.predict_logits` bunu otomatik yapıyor). Süre ölçülmeyen adımlarda yavaşlık sorun değil.
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

### 3. adım: değerlendirme (17 Kasım 2026)
- Ek eğitimler (gece kuyruğu, ~71 dk): small_cnn seed 43/44, small_cnn_nodist (mesafe kanalı yok,
  `channels=(0..4)`) seed 42/43/44, resnet18_scratch (ImageNet'siz) seed 42. Seed ≠ 42 → `models/<model>_s<seed>/`.
- Seçimler val'de sabitlendi: `EVAL_MODELS` (4 CNN), `GRADCAM_LAYER = 2` (20×15). Test **bir kez** çalıştırıldı.
  Testi yeniden çalıştırıp ona göre seçim yapma.
- Bootstrap: maç bazında, B = 2000, seed 2026. Eşleştirilmiş farklar seed-42 modelleriyle; çok seed'li
  modellerde ayrı ortalama ± std tablosu. TTA (aynalama) yalnızca ek satır.

**Test (14.166 şut, 597 maç, gol oranı 0,1105), log loss / Brier / AUC:**

| Model | Log loss | Brier | AUC |
|---|---|---|---|
| StatsBomb xG | **0,2778** | **0,0799** | **0,8065** |
| SmallCNN (seed 42) | 0,2826 | 0,0813 | 0,7966 |
| SmallCNN 3 seed | 0,2824 ± 0,0006 | 0,0813 | 0,7983 ± 0,0015 |
| SmallCNN mesafesiz (seed 42) | 0,2830 | 0,0818 | 0,7974 |
| ResNet18 ön eğitimsiz | 0,2832 | 0,0814 | 0,7966 |
| ResNet18 ImageNet | 0,2872 | 0,0826 | 0,7962 |
| LogReg mesafe + açı + vücut | 0,3044 | 0,0879 | 0,7533 |
| LogReg mesafe + açı | 0,3131 | 0,0903 | 0,7281 |
| Sabit | 0,3478 | 0,0983 | 0,5000 |

- SmallCNN − StatsBomb: log loss +0,0047 [+0,0023, +0,0071], AUC −0,0099 → **StatsBomb anlamlı biçimde daha iyi**
  (erkek ve kadında ayrı ayrı da). SmallCNN − LogReg(b): −0,0218 log loss, +0,043 AUC.
- ImageNet − ön eğitimsiz ResNet18: +0,0039 [+0,0013, +0,0066] (ön eğitimli daha kötü; fark 0,003'ün üstünde
  ama tek seed ve farklı LR). README'de "ön eğitim fayda sağlamadı" diye yazıldı, üstünlük iddiası yok.
- Mesafe kanalı ablasyonu: −0,0005 [−0,0023, +0,0015] → anlamlı fark yok.
- ECE (test): SmallCNN 0,009, ResNet18 0,011, StatsBomb 0,012, LogReg 0,015. TTA: SmallCNN 0,2826 → 0,2822.
- CNN'in StatsBomb'dan çok yüksek dediği şutlarda vole/yarım vole/aşırtma oranı yüksek (CNN teknik bilgisini görmüyor).
- Grad-CAM sınırlı açıklayıcı: GAP yüzünden boş alanları da vurguluyor, yalnızca pozitif kanıt. README'de not edildi.

### 4. adım: mantık testleri ve ısı haritaları (17 Kasım 2026)
- Model SmallCNN seed 42, test split'inin tamamı. Adım 3 yd; |Δp| < 0,001 "değişmedi". OOD: kalecisiz şut,
  25+ yd kafa, sadece kaleci senaryosu (tablolarda ayrı). Kullanıcı kararları: 3c betimleyici; 2c ölçütü
  |Δ2c| < |Δ2a| ve < |Δ2b|; 3b şut çizgisine dik, çizgiden uzağa; ızgara x 78–119, y 10–70; 2 gerçek pozisyon.
- 3b'de kaleci kale çizgisinin arkasına taşacaksa şut atlanıyor (kırpma yok; 3.303 şut).
- Sonuçlar (beklenen yön oranı, OOD hariç): 1a %85, 1b **%67**, 2a %99,6, 2b **%64**, 3a [OOD] %99, 3b %89,
  4a/4b %100. 2c ölçütü %81. 3c: xG çoğunlukla artıyor (+0,024). Aynalama: ort. |Δp| 0,012, en fazla 0,15.
- Beklenmedik (gerçek): (1) geri adım xG'yi artırabiliyor; savunmacıdan uzaklaşmaya bağlı (mesafe/baskı takası,
  büyüklüğü fazla). (2) Yarı yoldaki savunmacı kaleciden 3–10 yd öndeyse xG'yi sıkça artırıyor. (3) Kale çizgisi
  kenar artefaktı: x = 119'da 28 yd açıkta bile xG ~0,45 (eğitimde oradan 112 şut). (4) Sol/sağ asimetri.
- Isı haritası: kalabalık pozisyonda ceza sahası noktalarının %96'sında xG düşüyor (ort. −0,13).

### 5. adım: taktik tahtası (MVP, 18 Kasım 2026)
- streamlit-drawable-canvas 0.13 denendi: sürükleme çalışıyor ama taşıma modu yalnızca tuvalin kendi ✎ düğmesiyle
  açılıyor (Python'dan ayarlanamıyor). Kullanıcı kararı: **kendi bileşenimiz** (`st.components.v2`, derleme yok).
- Dikey saha, kale üstte, y = 0 solda (VerticalPitch ile doğrulandı); 8 px/yd, görünüm x 60–122 → 640 × 496.
  Tahta duyarlı: en fazla 640 px, oran sabit (dar sütunda kesilme hatası böyle çözüldü).
- xG her zaman TTA; LogReg (b) yanında; "son hamlenin etkisi" Δ; uyarılar: kale çizgisi kenarı (x ≥ 118,5,
  direk dışı), kalecisiz, sadece kaleci, 25+ yd kafa. Mod: Move / Add (+ tür seçimi); çift tık / sağ tık siler.
- JS bileşeni oyuncuları kendi tutar, yalnızca `version` değişince (senaryo yükle / sıfırla) Python'dan alır.
- Testler: yön testi (senaryo → tuval → geri → render_shot aynı), Playwright: sürükleme mesafesi, ekleme, silme,
  kesilmeme (1500 ve 1100 px). Görsel kontrol: plot_freeze_frame ile aynı diziliş (90° sola döndürülmüş).
- MVP push edildi (`4750546`). Kullanıcı geri bildirimiyle düzeltilenler: vücut bölgesi + senaryo seçicileri ana
  panelde, xG etiketinde vücut bölgesi; varsayılan senaryo `counter`; kenar uyarısında xG "—" (değer üstü çizili);
  tekil/çoğul; tahta + panel tek yatay container'da (en fazla 640 + 340 px, dar ekranda tahta küçülür).
- **Hata (bulundu, düzeltildi):** vücut bölgesi radyosu anahtarsızdı ve `index`'i her çalıştırmada `ss.body`'den
  hesaplanıyordu; Streamlit bunu yeni widget sayıp tıklamaları kaybediyor, bayat değer geri dönüyordu (kullanıcının
  gördüğü LogReg 0,151 → 0,047 = ayak → kafa). Kural: widget'lar `key` ile; değer yalnızca callback'te değişir.
  Bileşen durumu `st.session_state["board"]`'dan bileşen çizilmeden ÖNCE okunuyor (harita güncel kalsın diye).
- **Model bulgusu (README limitation 5):** kaleciyi öne almak xG'yi artırıyor (boş saha 0,19 → 0,49; test setinde
  8+ yd şutların %90'ından fazlasında; ort. 0,072 → 0,110 @ 4 yd). Veride de aynı ilişki var (12–18 yd: kaleci çizgide
  %5,9, 6–10 yd önde %29 gol) → karıştırıcı. Tahtada kaleci 3+ yd öndeyse bilgi notu çıkıyor.
- Isı haritası katmanı: `board.xg_map` (TTA, 128'lik parçalar), karekök renk ölçeği, düşük xG saydam,
  kenar hücreleri gri; şutçu dışındaki oyunculara göre `st.cache_data`. 2 yd ~1,5 sn, 1 yd ~5 sn (yerel GPU).
- `ae1934f` (yalnızca ekran görüntüsü yeniden adlandırma, yanlışlıkla) + `2f9503b` (harita + düzeltmeler), push edildi.

### 5. adım: yayın kararları (18 Kasım 2026)
- HF hesabı (huseyinyagmur) PRO değil; HF belgelerine göre Docker/Gradio Space ücretli plan istiyor →
  **kullanıcı kararı: canlı demo yok.** Sadece HF Hub model deposu `huseyinyagmur/xg-vision-smallcnn`
  (`small_cnn.pt` = models/small_cnn/best.pt, model kartı `deploy/hf_model/README.md`, logo, kalibrasyon grafiği).
  Yükleme: `python scripts/publish_model.py --upload` (mevcut `hf auth` girişi; token'ı asla yazdırma/dosyaya yazma).
- **StatsBomb Public Data User Agreement:** 1.2.1 veriyi dağıtmak yasak → `app/assets/scenarios.json` artık yalnızca
  match_id/shot_id tutuyor; konumlar açılışta StatsBomb'un reposundan çekiliyor (`board.load_scenarios`,
  data/cache'e önbellek). Eski sürüm git geçmişinde duruyor (history rewrite yok). 1.2.2 ticari kullanım yok.
  1.4 yayınlarda StatsBomb logosu: `assets/statsbomb_logo.png` (open-data reposundaki resmi PNG, 600 px).
- Kod MIT (`LICENSE`); ağırlıklar "other" lisans, StatsBomb sözleşmesine bağlı, ticari olmayan kullanım.
- CPU ölçümü (2 thread): tek xG ~10 ms, harita 2 yd ~27 sn, 1 yd ~87 sn → CPU'da seçenekler 3 yd (varsayılan) /
  2 yd; GPU'da 2 yd / 1 yd. `XG_DEVICE=cpu` ile CPU zorlanabilir. Ağırlık yerelde yoksa HF Hub'dan iner.

### 5. adım için notlar
- Taktik tahtası `perturb.Shot` + `render_shot` + `load_checkpoint` ile kurulabilir; kenar artefaktı nedeniyle
  x ≥ 118,5 ve direk dışı konumlar için arayüzde uyarı göstermeyi kullanıcıya öner.
- **Olası v2 (ertelendi, kullanıcı kararı):** technique + first_time skaler girdi. StatsBomb ile farkın olası kaynağı.
  Testten sonra eklendiği için sonuçları **post-hoc** olarak raporlanacak (ana test sonuçlarının yerine geçmez).
- **Olası v2:** tensör tuvalini kale çizgisinin arkasına uzatmak (ör. x ≤ 124), kale çizgisi kenar artefaktını kökten
  çözmek için. Yeniden eğitim gerekir; sonuçları post-hoc raporlanır.
