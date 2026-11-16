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
2. ⏭️ **Model:** sıfırdan küçük bir CNN ile ön eğitimli bir modeli (ör. ResNet18, ilk conv 6 kanala
   uyarlanmış) karşılaştır. Eğitim döngüsü, early stopping, en iyi modeli kaydetme.
3. **Değerlendirme:** maç bazında test, log loss, Brier, ROC-AUC, kalibrasyon grafiği; StatsBomb xG
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

## Mevcut durum (1. adım tamamlandı)

1. adım `0eff913` commit'iyle main'e push edildi. README İngilizce ve portföy odaklı yazıldı;
2. adımdan itibaren her adım bitince README'nin yol haritası ve sonuçlar bölümü güncellenmeli.

### Dosyalar
```
src/config.py         # yollar, saha geometrisi, çözünürlük, kanallar, split oranları, seed
src/data_download.py  # StatsBomb indir -> şutlar -> temizlik -> maç bazlı split -> data/shots.pkl
src/pitch_tensor.py   # (şutçu, freeze frame) -> (6, 160, 120) float32 tensör
src/dataset.py        # load_shots(split), ShotDataset (tensörü anlık çizer, augment=True -> y aynalama)
src/visualize.py      # saha çizimleri, kanal grafikleri, veri özeti; `python -m src.visualize`
results/plots/        # data_overview.png, freeze_frame_examples.png, example_channels.png
```

### Çalıştırma
```bash
pip install -r requirements.txt
python -m src.data_download     # ~4.000 maç, data/cache/ altında maç başına önbellek, tekrar çalıştırılabilir
python -m src.visualize         # grafikleri results/plots/ altına yazar
```
Veri GitHub raw üzerinden çekiliyor (`STATSBOMB_RAW`). statsbombpy kullanılmıyor (daha yavaş).

### Veri tablosu (`data/shots.pkl`, her satır bir şut)
Sütunlar: `shot_id, match_id, competition, season, gender, match_date, period, minute, team, player,
x, y, shot_type, body_part, technique, outcome, statsbomb_xg, first_time, one_on_one, under_pressure,
freeze_frame (np.ndarray N×4: [x, y, teammate, is_gk]), n_players, has_gk, is_goal, distance, angle, split`.

- Penaltılar, freeze frame'i olmayan şutlar ve x < 60 olan şutlar çıkarıldı.
- `load_shots()` sadece `TRAIN_SHOT_TYPES` (varsayılan `("Open Play",)`) döndürür. Frikikler tabloda var ama filtreleniyor.
- `distance` (kale merkezine yarda), `angle` (kale ağzının görüş açısı, radyan) → baseline için hazır.

### Tensör tasarımı
- Koordinatlar: StatsBomb 120×80 yarda. Hücum her zaman x = 120'deki kaleye doğru. Direkler y = 36 ve 44.
- Bölge: hücum yarı sahası, x ∈ [60, 120], y ∈ [0, 80]. 0,5 yd/px → **H = 160 (y), W = 120 (x)**.
- **Satır 0 = y = 0 = mplsoccer StatsBomb sahasının üst kenarı.** Çizimde `origin="upper"`,
  `extent=[X_MIN, X_MAX, Y_MAX, Y_MIN]` kullan (ilk denemede bu yüzden görüntüler ters çıkmıştı).
- Kanallar: 0 şutçu, 1 takım arkadaşları, 2 rakip saha oyuncuları, 3 rakip kaleci (σ = 1 yd Gauss lekeleri,
  üst üste binince toplanır), 4 şut üçgeni (şutçu → iki direk, ikili maske), 5 sabit kaleye-mesafe haritası
  (CoordConv benzeri konum bilgisi).
- Tensörler diske yazılmıyor, `ShotDataset` içinde anlık çiziliyor (~3,6 ms/örnek). Tümünü saklamak ~44 GB tutardı.
- Augmentation: y → 80 − y aynalama geçerli (kale simetrik). Vücut bölgesi girdi olarak eklenirse sol/sağ ayak da takas edilmeli.

### Sonuçlar (16 Kasım 2026'da yerelde indirilen veri)
- 3.961 maç, **95.160 açık oyun şutu**, gol oranı **%10,4**.
- Split (lig bazında tabakalı): train 66.754 / val 14.240 / test 14.166 şut.
- Erkek ~62 bin, kadın ~33 bin şut. Kalecisi olmayan freeze frame: ~%0,1.
- Mesafeye göre gol oranı ile StatsBomb xG ortalaması neredeyse örtüşüyor (veri doğru okunmuş).
- **Geçilmesi gereken çıta, StatsBomb xG (test):** log loss **0,2778**, Brier **0,0799**, ROC-AUC **0,8065**.
- Sabit tahmin (eğitim gol oranı), test: log loss ≈ 0,347, Brier ≈ 0,098.
- StatsBomb verisi zaman zaman güncelleniyor; yeniden indirince rakamlar çok az değişebilir.
  Bu yüzden karşılaştırmalar hep aynı `shots.pkl` üzerinden yapılmalı.

### Kullanıcının onayladığı kararlar (2. adım için geçerli)
1. **Vücut bölgesi → skaler girdi.** CNN görüntüden özellik çıkarır; global pooling sonrası vektöre
   vücut bölgesi eklenip sonra tam bağlantılı katmana verilir. Kodlama: `head / foot / other` one-hot
   (sol/sağ ayak ayrımı yapılmaz, böylece y-aynalama augmentation'ı geçerli kalır). Model API'si
   `forward(image, extra_features)` biçiminde olsun; ileride `first_time`, `under_pressure` gibi
   ek skalerler aynı yoldan eklenebilir (eklemeden önce kullanıcıya sor).
2. **Kadın ve erkek maçları birlikte eğitilir.** Değerlendirmede metrikler toplam + erkek/kadın ayrı raporlanır.
3. **Sadece açık oyun şutları** (`TRAIN_SHOT_TYPES = ("Open Play",)`). Frikikler tabloda kalır, eğitime girmez.

### Küçük yapılacaklar
- StatsBomb xG ve sabit tahmin metriklerini yazdıran bir fonksiyon `src/evaluate.py` içinde olmalı
  (şu an bu sayılar ayrıca hesaplandı, kodda yok). 3. adımda bütün modeller aynı fonksiyonla ölçülecek.
