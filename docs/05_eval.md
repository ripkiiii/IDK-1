# Tahap 5 — Evaluasi & Inference

**Tujuan:** Test model hasil training, generate teks, assess kualitas.

---

## Apa yang Dievaluasi?

### Kuantitatif
- **Val Loss** — metric utama training (sudah ada di training loop)
- **Perplexity** = `exp(val_loss)` — lebih intuitif: "seberapa 'kaget' model sama teks baru?"

```
Loss 3.0 → Perplexity ~20
Loss 2.5 → Perplexity ~12
Loss 2.0 → Perplexity ~7
```

Makin rendah makin bagus (model lebih "familiar" dengan teks Indo).

### Kualitatif
Generate teks dari berbagai prompt dan nilai:
1. Grammar benar?
2. Relevan sama prompt?
3. Ada repetition loop?
4. Koheren antar kalimat?

---

## Test Prompts

```python
prompts = [
    # Faktual
    "Indonesia adalah negara kepulauan yang",
    "Presiden pertama Indonesia adalah",
    
    # Cerita
    "Hari itu, seorang pemuda dari Bandung",
    
    # Teknis
    "Kecerdasan buatan bekerja dengan cara",
    
    # Completion
    "Bahasa Indonesia merupakan bahasa persatuan yang",
]
```

---

## Temperature Testing

```python
temperatures = [0.5, 0.7, 0.8, 1.0, 1.2]
# 0.5 = deterministic, predictable
# 0.8 = sweet spot untuk Indonesian text
# 1.2 = lebih creative, bisa lebih random
```

---

## Kapan Model "Bagus Cukup"?

Untuk showcase/portfolio:
- Val loss < 3.0
- Tidak ada major repetition loop
- Text secara umum terasa "Indonesian"

Untuk produksi (nanti): perlu fine-tuning di atas ini.

---

## Publish ke HuggingFace

```bash
# Setelah eval bagus:
huggingface-cli upload idk-ai/IDK-1 ./checkpoints/best.pt
```

Model card akan include:
- Arsitektur (~100M LLaMA-style + RoPE theta 500k + logit soft-capping)
- Training data (Wikipedia ID + CulturaX cleaned)
- Val loss curve
- Sample generations
- Known limitations
