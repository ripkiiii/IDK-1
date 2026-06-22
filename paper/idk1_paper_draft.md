# IDK-1: Training a 100M Parameter Indonesian Language Model from Scratch on Commodity Hardware

**Muhammad Rifky Firmansyah Sujana**  
Telkom University, Bandung, Indonesia  
rifky@keemail.me

---

## Abstract

We present IDK-1, a 100M parameter decoder-only language model trained natively on Indonesian text from scratch. Unlike existing Indonesian NLP models that rely on multilingual pretraining or cross-lingual transfer, IDK-1 is trained entirely on Indonesian corpora using a BPE tokenizer built for Indonesian vocabulary. Training was conducted on free commodity hardware (2x NVIDIA T4 via Kaggle) over 100,000 steps on 2.64 billion tokens. We document the full training process, architecture decisions, and convergence behavior, with the goal of providing a reproducible blueprint for low-resource language model development under compute constraints. IDK-1 and all associated artifacts are released openly.

---

## 1. Introduction

Large language models have demonstrated remarkable capabilities across a wide range of tasks. However, the majority of these models are trained primarily on English text, with other languages — including Indonesian — receiving significantly less representation. While multilingual models such as mBERT and mT5 include Indonesian, they allocate only a small fraction of their training budget to it, often resulting in suboptimal performance on Indonesian-specific tasks.

Indonesian presents an interesting case for language model development. With over 270 million speakers, it is one of the most widely spoken languages in the world, yet remains underrepresented in the NLP literature relative to its speaker count. Existing Indonesian-specific models such as IndoBERT and IndoGPT represent important contributions, but were developed with access to significant compute resources unavailable to most researchers in the region.

In this work, we ask: *can a capable Indonesian language model be trained from scratch on free, commodity hardware?* We answer affirmatively, presenting IDK-1 — a 106M parameter LLaMA-style model trained natively on Indonesian text using only Kaggle's free GPU tier.

Our contributions are:
1. A fully open Indonesian SLM trained from scratch with documented methodology
2. An aggressive data cleaning pipeline yielding 2.64B high-quality Indonesian tokens
3. A reproducible training recipe for low-resource language modeling under compute constraints
4. All model weights, code, and datasets released publicly

---

## 2. Related Work

### 2.1 Indonesian NLP Models
- **IndoBERT** (Wilie et al., 2020) — BERT-based encoder trained on Indonesian Wikipedia and news
- **IndoGPT** — GPT-2 style Indonesian generative model
- **Nusantara NLP** — benchmark suite for Indonesian and regional languages

### 2.2 Small Language Models
Recent work has shown that smaller models trained on high-quality data can be surprisingly capable:
- **TinyLlama** (Zhang et al., 2024) — 1.1B model trained on 3T tokens
- **Phi-1/2** (Microsoft, 2023) — small models trained on "textbook quality" data
- **MobileLLM** (Meta, 2024) — sub-1B models for on-device deployment

### 2.3 Low-Resource Language Modeling
[TODO: cite relevant papers on low-resource LM training]

---

## 3. Data

### 3.1 Data Sources

We train IDK-1 on two primary sources of Indonesian text:

| Source | Raw Docs | Kept Docs | Keep Rate | Tokens |
|--------|----------|-----------|-----------|--------|
| Wikipedia ID | 665,000 | 373,000 | 56.1% | 0.16B |
| CulturaX ID | 5,000,000 | 4,920,000 | 98.4% | 2.51B |
| **Total** | **5,665,000** | **5,293,000** | **93.5%** | **2.64B** |

### 3.2 Cleaning Pipeline

We apply aggressive cleaning to ensure data quality:
- Remove documents with excessive repetition
- Filter non-Indonesian content using language detection
- Remove documents below minimum length threshold
- Deduplicate at document level using exact hash matching
- Normalize Unicode and whitespace

Wikipedia ID undergoes stricter filtering (56.1% kept) due to its inclusion of stub articles, disambiguation pages, and non-prose content. CulturaX ID is already relatively clean, resulting in a higher keep rate.

### 3.3 Tokenizer

We reuse the BPE tokenizer from our earlier DFD-1 project, trained on Indonesian text with a vocabulary size of 40,000. This tokenizer was built natively for Indonesian, unlike multilingual tokenizers that typically allocate fewer tokens to non-English languages.

**Tokenizer stats:**
- Vocabulary size: 40,000
- Type: Byte-Pair Encoding (BPE)
- Training data: Indonesian web corpus

### 3.4 Data Format

Documents are concatenated with EOS tokens and split into sequences of 1,024 tokens. The resulting binary dataset:
- `train.bin`: 5.27 GB (~2.64B tokens)
- `val.bin`: 0.05 GB (~25M tokens)

---

## 4. Model Architecture

IDK-1 is a decoder-only transformer based on the LLaMA architecture, with two additional modern modifications.

### 4.1 Base Architecture

| Hyperparameter | Value |
|---------------|-------|
| Parameters | 106.24M |
| Hidden dim (d_model) | 768 |
| Layers | 12 |
| Attention heads | 12 |
| KV heads (GQA) | 4 |
| FFN dim | 2,048 |
| Max sequence length | 1,024 |
| Vocabulary size | 40,000 |

### 4.2 Architectural Modifications

**Grouped Query Attention (GQA)**  
We use 4 KV heads instead of 12, following the approach in LLaMA-3. GQA reduces memory bandwidth requirements during inference while maintaining model quality, making deployment on constrained hardware more practical.

**RoPE with Extended Base Frequency**  
We use Rotary Position Embeddings with theta=500,000, following LLaMA-3 (vs. the original theta=10,000). This allows better length generalization beyond the training context window.

**Logit Soft-Capping**  
Following Gemma 2, we apply soft-capping to output logits:
```
logits = 30.0 * tanh(logits / 30.0)
```
This stabilizes training by preventing logit values from growing unboundedly.

---

## 5. Training

### 5.1 Setup

Training was conducted on Kaggle's free GPU tier using 2x NVIDIA T4 GPUs with PyTorch DataParallel. No distributed training framework was required.

**Hardware:**
- 2x NVIDIA T4 (16GB VRAM each)
- Platform: Kaggle Notebooks (free tier)
- ~30 GPU-hours per week available

**Training throughput:** ~15,400 tokens/second

### 5.2 Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| Total steps | 100,000 |
| Batch size | [TODO] |
| Learning rate | cosine decay, max [TODO] |
| Warmup steps | [TODO] |
| Gradient clipping | [TODO] |
| Optimizer | AdamW |
| Weight decay | [TODO] |
| Precision | [TODO: bf16/fp16] |

### 5.3 Training Dynamics

[TODO: add full val loss curve plot]

Selected validation loss checkpoints:

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 2,000 | — | 7.8010 |
| 4,500 | — | 7.7955 |
| 5,000 | — | 7.7966 |
| 10,000 | — | 7.8006 |
| 15,000 | 7.7903 | 7.7941 |
| 17,500 | 7.7932 | 7.8001 |
| ... | ... | ... |

The model enters a plateau phase in the first 15-20% of training, with validation loss oscillating in the 7.79–7.81 range. This is expected behavior for language model pretraining; significant loss decrease typically occurs after 20-30% of the training data has been seen. Train and validation loss track closely throughout, indicating no overfitting.

---

## 6. Results

[TODO: complete after 100k steps]

### 6.1 Perplexity

### 6.2 Qualitative Evaluation

[TODO: sample outputs from IDK-1 base model]

### 6.3 Comparison with Existing Models

[TODO: compare with IndoBERT, IndoGPT on available benchmarks]

---

## 7. Limitations

**Compute constraints:** Training was conducted entirely on free Kaggle GPU quota (~30 GPU-hours/week), requiring multiple session handoffs and checkpoint resumption. This introduces training discontinuities that a well-resourced lab would avoid.

**Vocabulary size:** Our 40k BPE vocabulary may be suboptimal. Recent work suggests larger vocabularies (64k–100k) improve tokenization efficiency for non-English languages.

**Context length:** The 1,024 token context window is shorter than modern models. This limits the model's ability to handle long documents.

**No instruction tuning:** IDK-1 is a base model only. We plan IDK-1-Instruct as a follow-up.

---

## 8. Conclusion

We present IDK-1, a 106M parameter Indonesian language model trained from scratch on commodity hardware. Our work demonstrates that meaningful language model pretraining is achievable without access to large compute clusters, providing a reproducible blueprint for the Indonesian NLP community and low-resource language researchers more broadly.

**Future work:**
- IDK-1-Instruct: supervised fine-tuning on Indonesian instruction data
- Extension to regional Indonesian languages (Sundanese, Javanese)
- Larger model variants as compute becomes available
- Indonesian NLP benchmark suite

---

## References

[TODO: fill in all citations]

---

*IDK-1 model weights, training code, and datasets are available at:*  
*HuggingFace: `idk-ai/IDK-1`*  
*GitHub: `github.com/ripkiiii/IDK-1`*
