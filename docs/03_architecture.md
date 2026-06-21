# Tahap 3 — Arsitektur IDK-1

**Tujuan:** Define config ~100M parameter, verify forward pass.

---

## Config IDK-1 (~100M)

```python
@dataclass
class IDK1Config:
    vocab_size: int = 40_000     # reuse tokenizer DFD-1
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4          # GQA
    ffn_dim: int = 2048          # SwiGLU
    max_seq_len: int = 1024
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 500_000  # ← LLaMA-3 style (dari 10_000)
```

**Estimasi params: ~100-106M** (diverifikasi di notebook)

---

## Modern Tweaks

### 1. RoPE theta = 500_000
LLaMA-3 pakai ini. Satu angka, impact signifikan untuk generalization di sequence yang lebih panjang dari training. Tidak ada cost tambahan.

```python
# Before (DFD-1):
rope_theta: float = 10_000.0

# After (IDK-1):
rope_theta: float = 500_000.0
```

### 2. Logit soft-capping (Gemma 2)
Cegah logit explode di awal training. Training lebih stabil, terutama di 100M ke bawah.

```python
# Di forward pass, sebelum return:
def forward(self, idx):
    ...
    logits = self.lm_head(self.norm(x))
    logits = 30.0 * torch.tanh(logits / 30.0)  # ← tambah ini
    return logits
```

---

## Perbandingan DFD-1 vs IDK-1

| | DFD-1 | IDK-1 |
|---|---|---|
| Params | ~500M | ~100M |
| dim | 1280 | 768 |
| Layers | 24 | 12 |
| Heads | 16 | 12 |
| KV heads | 8 | 4 |
| seq_len | 2048 | 1024 |
| rope_theta | 10,000 | 500,000 |
| Logit cap | ❌ | ✅ |
| Est. speed | 5.8k tok/s | ~15k tok/s |
| Est. full cycle | 3 minggu | ~1 minggu |

---

## Verifikasi di Notebook

1. Instantiate model
2. Hitung params (target: 100-110M)
3. Forward pass dummy input → cek shape `[B, T, vocab_size]`
4. Cek logit range sebelum/sesudah soft-capping

---

## Base untuk SFT

IDK-1 base (PT) akan jadi pondasi untuk:
- `IDK-1-Instruct` — SFT instruction following Indo
