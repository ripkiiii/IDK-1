# IDK-1

**Brand:** I Don't Know (IDK)  
**Model:** IDK-1 — Indonesian SLM ~100M params  
**Umbrella:** [Deflated](https://deflated.xyz)

> "We don't know everything. We just build anyway."

---

## Kenapa 100M?

- Cukup besar untuk output yang meaningful
- Kaggle free tier: ~2 sesi untuk full 30k steps
- Speed of iteration > size

## Arsitektur

LLaMA-style decoder-only transformer dengan dua modern tweak:
- `rope_theta = 500_000` (LLaMA-3 style, better long-context generalization)
- Logit soft-capping: `30.0 * tanh(logits / 30.0)` (Gemma 2 style, training lebih stabil)

**Config:** dim=768, n_layers=12, n_heads=12, n_kv_heads=4 (GQA), ffn_dim=2048, vocab=40k, max_seq=1024

---

## Roadmap

```
PT: IDK-1 base (100M, 30k steps, ~1B tokens)
    └── SFT: IDK-1-Instruct (instruction tuning, Indo)
        └── DPO: alignment, 200-500 preference pairs
```

| Tahap | File | Status |
|-------|------|--------|
| 1. Tokenizer | — reuse dari DFD-1 | ✅ |
| 2. Data Pipeline | `notebooks/02_data_pipeline.ipynb` | ✅ 2.64B tokens, dataset: `ripkii/idk1-data-new` |
| 3. Arsitektur | `notebooks/03_architecture.ipynb` | ✅ 106.24M params verified |
| 4. Pre-training | `notebooks/04_pretrain.ipynb` | 🟡 RUNNING — 30k steps, 12.6k tok/s, loss turun 10.76→6.11 di step 500 |
| 5. Evaluasi base | `notebooks/05_eval.ipynb` | ⏳ |
| 6. SFT Instruct | `notebooks/06_sft_instruct.ipynb` | ⏳ 1,390 pairs ready (`idk1_instruct_1k.jsonl`) |

---

## Data

- **Wikipedia ID**: 373,337 docs (56.1% kept), 0.16B tokens
- **CulturaX ID**: 4,920,349 docs (98.4% kept), 2.51B tokens
- **Total**: 2.64B tokens, train.bin 5.27GB

Aggressive cleaning: deduplicate, filter noise, round-robin interleave (no shuffle bug).

---

## Target

- PT val loss: ~3.0-4.0 di step 30k
- SFT: IDK-1-Instruct bisa follow instruksi bahasa Indo
- Publish ke HuggingFace: `idk-ai/IDK-1` + `idk-ai/IDK-1-Instruct`
- Demo di deflated.xyz
