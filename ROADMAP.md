# IDK-1 Research Roadmap
> by Deflated / ripkiiii — last updated 2026-06-26

---

## Visi

Bangun Indonesian SLM ecosystem dari scratch:
- Model yang bisa dibandingkan (dense vs MoE)
- Pipeline alignment lengkap (SFT → DPO)
- Benchmark Indo buatan sendiri
- Semua reproducible di commodity hardware (Kaggle T4)

---

## Phase 1 — Pre-Training (🔜 NEXT)

### IDK-1 Dense (baseline)
- **Arsitektur:** LLaMA-style, 106M params, dim=768, 12 layers, GQA
- **Data:** 2.64B tokens (Wikipedia ID + CulturaX ID), clean pipeline
- **Target:** 100k steps (~60 jam, 2 akun Kaggle alternating)
- **Schedule:** step 0→20k (Rifky) → 20k→40k (Raisa) → ... → 100k
- **Status:** ⏳ nunggu Kaggle quota reset ~2026-06-27

### IDK-1-MoE (experiment)
- **Arsitektur:** IDK-1 + FFN diganti Mixture of Experts
  - 8 expert per layer, top-2 routing
  - ~200M total params, ~50M aktif per token
  - Auxiliary load balancing loss
- **Data:** sama persis dengan IDK-1 dense
- **Target:** 100k steps, train paralel di akun Raisa setelah sesi 1
- **Status:** ❌ belum dimulai — sketch arsitektur dulu

**Deliverable Phase 1:**
- `best_dense.pt` + `best_moe.pt`
- Val loss curve comparison
- tok/s benchmark di T4

---

## Phase 2 — Instruction Tuning / SFT (setelah Phase 1)

### IDK-1-Instruct
- **Data:** `idk1_instruct_1k.jsonl` — 1390 pairs, 15 topik
- **Method:** Full fine-tune (model kecil, ga perlu LoRA)
- **Format:** ChatML
- **Target:** ~3-5 epoch, monitor val loss

### IDK-1-MoE-Instruct
- SFT dengan data yang sama di atas MoE base
- Compare: apakah MoE lebih baik di instruction following?

**Deliverable Phase 2:**
- 2 model instruct: dense + MoE
- Sample output comparison (qualitative)

---

## Phase 3 — Alignment / DPO Mini (setelah Phase 2)

### Preference Dataset Mini
- Generate 2 response per prompt dari IDK-1-Instruct
- Manual label: pilih yang lebih baik (chosen vs rejected)
- Target: 200-500 preference pairs
- Tool: bisa pake script sederhana atau Argilla lokal

### DPO Mini
- **Method:** Direct Preference Optimization — tanpa reward model
- **Library:** TRL `DPOTrainer` atau implement manual
- **Base:** IDK-1-Instruct (dense) sebagai starting point
- **Target:** lihat apakah response lebih coherent + less hallucination

**Deliverable Phase 3:**
- `IDK-1-DPO` model
- Before/after comparison: 10 prompt benchmark

---

## Phase 4 — Benchmark Indo Mini

### Dataset Benchmark
Bikin benchmark evaluasi Indo sendiri — 200 soal, 5 kategori:

| Kategori | Jumlah | Contoh |
|----------|--------|--------|
| Faktual Indo | 50 | "Siapa proklamator Indonesia?" |
| Reasoning | 40 | soal logika, matematika sederhana |
| Bahasa | 40 | grammar, sinonim, antonim |
| Instruksi | 40 | ikutin instruksi multi-step |
| Safety | 30 | tolak request berbahaya |

### Eval semua model:
- IDK-1 Dense base
- IDK-1-MoE base
- IDK-1-Instruct
- IDK-1-MoE-Instruct
- IDK-1-DPO
- Bonus: bandingkan sama Qwen2.5-0.5B sebagai external baseline

**Deliverable Phase 4:**
- `indo_bench_mini.jsonl` — publish ke HuggingFace
- Leaderboard tabel: semua model head-to-head

---

## Phase 5 — Deployment & Publication

### HuggingFace Release
- `idk-ai/IDK-1` — base model
- `idk-ai/IDK-1-Instruct` — chat model
- `idk-ai/IDK-1-MoE` — MoE experiment
- `idk-ai/indo-bench-mini` — benchmark dataset

### chat.deflated.xyz
- Backend: HF Inference API atau HF Spaces (Gradio)
- Model: IDK-1-Instruct atau IDK-1-DPO
- UI: simple chat interface

### Paper
- **Judul:** "IDK-1: Training Indonesian Small Language Models from Scratch — Dense, MoE, and Alignment on Commodity Hardware"
- **Target venue:** ArXiv → ACL/COLING/EACL
- **Sections:** Data, Architecture (Dense vs MoE), SFT, DPO, Benchmark, Analysis

---

## Timeline Estimasi

```
Juni 27-28   : Phase 1 — PT Dense (Kaggle quota reset)
Juli 1-7     : Phase 1 — PT MoE (paralel di akun Raisa)
Juli 8-10    : Phase 2 — SFT kedua model
Juli 11-13   : Phase 3 — DPO Mini + preference dataset
Juli 14-17   : Phase 4 — Benchmark Indo Mini
Juli 18-20   : Phase 5 — Deploy + paper draft finalize
```

---

## Compute Budget

| Resource | Quota | Dipakai untuk |
|----------|-------|---------------|
| Kaggle Rifky | 30h/minggu | PT Dense sesi 1,3,5 |
| Kaggle Raisa | 30h/minggu | PT MoE / Dense sesi 2,4 |
| SageMaker Studio Lab | 4h/hari | backup darurat |
| HF Spaces (free) | unlimited CPU | deployment |

---

## Success Metrics

- [ ] IDK-1 Dense val loss < 3.0 di 100k steps
- [ ] IDK-1-MoE val loss lebih rendah dari Dense (atau comparable dengan lebih sedikit active params)
- [ ] IDK-1-Instruct bisa jawab 80%+ prompt dasar dengan koheren
- [ ] DPO Mini improve qualitative score vs Instruct baseline
- [ ] Benchmark Indo Mini dipublish ke HF
- [ ] chat.deflated.xyz live

---

*"Just Build."*
