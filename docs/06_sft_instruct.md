# Tahap 6 — SFT: IDK-1-Instruct

**Tujuan:** Fine-tune IDK-1 base agar bisa follow instruksi bahasa Indonesia.  
**Input:** IDK-1 base checkpoint terbaik (dari PT step 100k)  
**Output:** `IDK-1-Instruct` — published ke HuggingFace

---

## Format Data SFT

Instruction tuning pakai format chat sederhana:

```json
{
  "instruction": "Jelaskan apa itu fotosintesis dengan bahasa sederhana.",
  "input": "",
  "output": "Fotosintesis adalah proses yang dilakukan tumbuhan untuk membuat makanannya sendiri..."
}
```

Template prompt yang dipakai saat training:

```
### Instruksi:
{instruction}

### Input:
{input}

### Respons:
{output}
```

---

## Dataset

**Indo Alpaca** — Alpaca dataset yang ditranslate ke bahasa Indonesia.  
Tersedia di HuggingFace: `yahma/alpaca-cleaned` (versi cleaned) lalu translate, atau cari yang udah jadi Indo.

Alternatif tambahan:
- `Ichsan/indonesian-alpaca` — Indo Alpaca yang sudah jadi
- GPT-generated synthetic Indo instruction data (generate sendiri pakai API)

**Target jumlah:** 10k-50k instruction pairs. Cukup untuk SFT di 100M.

---

## Training Config SFT

```python
# Berbeda dari PT — jauh lebih kecil
LR          = 2e-5     # lebih kecil dari PT (jangan overwrite terlalu keras)
EPOCHS      = 3        # bukan steps, tapi epochs (dataset kecil)
BATCH_SIZE  = 4
GRAD_ACCUM  = 4        # effective batch = 16
WARMUP      = 100

# Load dari best PT checkpoint
CKPT_PATH   = "path/to/idk1_base_best.pt"
```

SFT jauh lebih cepat dari PT — dataset kecil, LR kecil, beberapa epoch saja.

---

## Estimasi Waktu SFT

- 50k instruction pairs × 3 epochs = 150k samples
- Dengan batch 16: ~9,375 steps
- Di 2x T4: **< 2 jam**

Jauh lebih ringan dari PT.

---

## Evaluasi SFT

Test dengan prompt Indo berbagai kategori:

```python
test_prompts = [
    "Buatkan saya resep nasi goreng sederhana.",
    "Jelaskan perbedaan antara AI dan machine learning.",
    "Tolong buatkan email formal untuk melamar pekerjaan.",
    "Apa ibu kota dari provinsi Jawa Barat?",
    "Bagaimana cara belajar programming dari nol?",
]
```

Yang dinilai:
- Relevan sama instruksi?
- Bahasa Indonesia natural?
- Tidak ada repetition loop?
- Format sesuai (kalau diminta list, berikan list)?

---

## Publish ke HuggingFace

```
idk-ai/IDK-1         ← base model (PT)
idk-ai/IDK-1-Instruct ← setelah SFT
```

Model card include:
- Arsitektur (~100M LLaMA-style + tweaks)
- Training data (Wikipedia ID + CulturaX cleaned)
- SFT data (Indo Alpaca)
- Val loss curve
- Sample conversations
- Known limitations (masih kecil, bukan untuk task kompleks)
