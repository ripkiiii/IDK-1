# Training Config Reference

---

## PT Config (Pre-training)

```python
# Model
VOCAB_SIZE   = 40_000
DIM          = 768
N_LAYERS     = 12
N_HEADS      = 12
N_KV_HEADS   = 4
FFN_DIM      = 2048
MAX_SEQ_LEN  = 1024
ROPE_THETA   = 500_000    # LLaMA-3 style
LOGIT_CAP    = 30.0       # Gemma 2 style

# Training
SEQ_LEN      = 512
BATCH_SIZE   = 8
GRAD_ACCUM   = 8           # effective batch = 64 seq = 32,768 tokens
LR           = 3e-4
MIN_LR       = 3e-5
WARMUP       = 1_000
MAX_STEPS    = 100_000
EVAL_EVERY   = 500
SAVE_EVERY   = 2_500
DTYPE        = torch.float16

# Paths (Kaggle)
DATA_PATH    = "/kaggle/input/datasets/ripkii/idk1-data/train.bin"
VAL_PATH     = "/kaggle/input/datasets/ripkii/idk1-data/val.bin"
CKPT_DIR     = "/kaggle/working/checkpoints"
TOKENIZER    = "/kaggle/input/idk1-tokenizer/tokenizer.json"
```

---

## SFT Config (Instruction Tuning)

```python
LR           = 2e-5
EPOCHS       = 3
BATCH_SIZE   = 4
GRAD_ACCUM   = 4
WARMUP       = 100
DTYPE        = torch.float16

# Load dari best PT checkpoint
BASE_CKPT    = "/kaggle/input/idk1-checkpoint/best.pt"
DATA_PATH    = "/kaggle/input/idk1-instruct-data/train.json"
```

---

## Kaggle Dataset Names

| Dataset | Isi |
|---------|-----|
| `idk1-tokenizer` | `tokenizer.json` (reuse dari DFD-1) |
| `idk1-data` | `train.bin` + `val.bin` (PT data) |
| `idk1-checkpoint-XXXXX` | checkpoint tiap upload |
| `idk1-instruct-data` | `train.json` (SFT instruction pairs) |
