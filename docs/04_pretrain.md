# Tahap 4 — Pre-training

**Tujuan:** Train IDK-1 base dari scratch sampai 100k steps.

---

## Training Config

```python
SEQ_LEN     = 512
BATCH_SIZE  = 8
GRAD_ACCUM  = 8           # effective batch = 64 seq
# Effective tokens/step = 64 × 512 = 32,768

LR          = 3e-4
MIN_LR      = 3e-5
WARMUP      = 1000
MAX_STEPS   = 100_000

dtype       = torch.float16   # T4 tidak support bfloat16
```

**Total training tokens:** 32,768 × 100,000 = **~3.3B tokens**

---

## Estimasi Waktu (~100M)

| Hardware | Speed | 100k steps |
|----------|-------|------------|
| 2x T4 Kaggle | ~15k tok/s | ~60 jam ≈ 8-9 session |
| 1x T4 | ~8k tok/s | ~113 jam ≈ 15-16 session |

Kaggle 30 jam/minggu → **~2 minggu** kalau 2x T4.

---

## Checkpoint Strategy

```
Simpan setiap 2500 steps:
├── step_002500.pt
├── step_005000.pt
├── ...
├── latest.pt  → symlink ke checkpoint terakhir
└── best.pt    → checkpoint dengan val_loss terendah
```

Upload ke Kaggle Dataset baru setelah tiap session.

---

## Target Loss

```
Step 10k:  loss ~4.0-4.5
Step 30k:  loss ~3.5-4.0
Step 60k:  loss ~3.0-3.5
Step 100k: loss ~2.5-3.0  ← target PT selesai
```

---

## Perbedaan dari DFD-1

1. Data lebih bersih → kurangi repetition
2. Model lebih kecil → training lebih cepat
3. `rope_theta=500_000` → better generalization
4. Logit soft-capping → training lebih stabil
5. Checkpoint naming jelas: `step_XXXXX.pt`, bukan symlink ambigu

---

## Setelah PT Selesai

IDK-1 base → lanjut ke `docs/06_sft_instruct.md`
