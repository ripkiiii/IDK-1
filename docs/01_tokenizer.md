# Tahap 1 — Tokenizer

**Tujuan:** Bikin tokenizer BPE custom untuk bahasa Indonesia.

---

## Bisa Reuse dari DFD-1?

**Ya.** Tokenizer DFD-1 sudah:
- BPE 40k vocab
- Trained dari Wikipedia ID + CulturaX
- Token ratio 1.45 token/kata (bagus untuk Indo)
- Ada `<s>` dan `</s>` token

Tidak perlu train ulang. Cukup upload ulang ke Kaggle dataset baru dengan nama `idk1-tokenizer`.

---

## Kalau mau train ulang (opsional)

Alasan train ulang tokenizer:
- Mau tambah domain-specific vocab (tech, coding terms)
- Mau ubah vocab size (misal 32k biar lebih ringan)
- Mau lebih banyak data training tokenizer

Tools: HuggingFace `tokenizers` library, BPE algorithm.

---

## Output yang dibutuhkan

```
tokenizer/
└── tokenizer.json   ← file utama, semua ada di sini
```

---

## Rekomendasi

**Reuse DFD-1 tokenizer.** Hemat 1-2 Kaggle session. IDK-1 adalah model baru, bukan tokenizer baru.

Kalau nanti mau fine-tune untuk coding (IDK-1-Code), baru consider retrain tokenizer dengan tambahan code tokens.
