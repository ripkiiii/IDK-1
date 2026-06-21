# Tahap 2 — Data Pipeline

**Tujuan:** Prepare training data yang bersih → binary `.bin` format.

---

## Problem di DFD-1

Output DFD-1 step 50k bagus secara grammar, tapi **severe repetition loops**. Penyebabnya: data training banyak pola repetitif dari web:

```
# Contoh noise di CulturaX:
"Beranda > Kategori > Sub-kategori > Artikel"
"Home | About | Contact | Privacy Policy"
"Halaman 1 2 3 4 5 ... Next"
"Tags: berita, indonesia, terkini, update"
```

Pola navigasi web ini muncul ribuan kali di corpus → model belajar untuk loop.

---

## Rencana Cleaning

### Filter yang harus ditambah:

1. **Deduplikasi** — exact dan near-duplicate removal
2. **Navigation pattern filter** — buang baris dengan pola `X > Y > Z` atau `A | B | C`
3. **Short line filter** — buang baris < 20 karakter (mostly noise)
4. **Repetition filter** — detect dan buang dokumen yang banyak ngulang phrase yang sama
5. **Quality score** — perplexity filter (buang teks dengan perplexity sangat rendah/tinggi)

---

## Data Sources

| Source | Size | Kualitas |
|--------|------|----------|
| Wikipedia ID | ~500MB | ⭐⭐⭐⭐⭐ Tinggi, ensiklopedi |
| CulturaX ID | ~4GB | ⭐⭐⭐ Medium, web crawl |
| Oscar ID (opsional) | ~2GB | ⭐⭐ Harus dicleaning keras |

**Rekomendasi:** Wikipedia ID + CulturaX yang sudah di-filter. Lebih kecil tapi jauh lebih bersih.

---

## Target Token Count

- DFD-1: 4.5B tokens (banyak noise)
- IDK-1: target **2-3B tokens bersih** > 4.5B tokens kotor

Quality >> Quantity untuk SLM.

---

## Output

```
data/
├── train.bin   ← token IDs dalam binary format
└── val.bin     ← validation set (~1% dari train)
```

Format: `uint16` array (karena vocab_size=40k, masuk di 16-bit)
