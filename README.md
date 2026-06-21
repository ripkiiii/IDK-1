# IDK-1

**Brand:** I Don't Know (IDK)  
**Model:** IDK-1 — Indonesian Language Model ~100M  
**Umbrella:** Deflated

> "We don't know everything. We just build anyway."

---

## Kenapa 100M?

- Cukup besar untuk output yang meaningful
- Kaggle free tier: ~1 minggu untuk full 100k steps
- Speed of iteration > size

## Arsitektur

LLaMA-style decoder-only transformer dengan dua modern tweak:
- `rope_theta = 500_000` (LLaMA-3 style, better long-context generalization)
- Logit soft-capping: `30.0 * tanh(logits / 30.0)` (Gemma 2 style, training lebih stabil)

---

## Roadmap

```
PT: IDK-1 base (100M, 100k steps)
    └── SFT: IDK-1-Instruct (instruction tuning, Indo)
```

| Tahap | File | Status |
|-------|------|--------|
| 1. Tokenizer | — reuse dari DFD-1 | ✅ |
| 2. Data Pipeline | `notebooks/02_data_pipeline.ipynb` | ✅ 2.64B tokens |
| 3. Arsitektur | `notebooks/03_architecture.ipynb` | ✅ 106.24M params |
| 4. Pre-training | `notebooks/04_pretrain.ipynb` | 🟡 step ~9600/100k, val=7.80 |
| 5. Evaluasi base | `notebooks/05_eval.ipynb` | ⏳ |
| 6. SFT Instruct | `notebooks/06_sft_instruct.ipynb` | ⏳ |

---

## Target

- PT val loss: ~2.5-3.0 di step 100k
- SFT: IDK-1-Instruct bisa follow instruksi bahasa Indo
- Publish ke HuggingFace: `idk-ai/IDK-1` + `idk-ai/IDK-1-Instruct`
- Demo di deflated.xyz
