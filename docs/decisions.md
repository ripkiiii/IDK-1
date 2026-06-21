# Design Decisions — IDK-1

Keputusan teknis dan alasannya.

---

## Kenapa 100M bukan 125M atau 50M?

- 50M: output quality terlalu jelek
- 100M: sweet spot — seminggu di Kaggle, quality cukup untuk demo + SFT
- 125M: 2 minggu, tidak worth extra waktu vs 100M
- 500M (DFD-1): terlalu lambat, tidak bisa selesaikan full cycle

---

## Kenapa Arsitektur LLaMA-style + dua tweak?

"Semua arsitektur modern perform within ~1% satu sama lain di 70-100M. Depth-width ratio dan training recipe lebih penting dari pilihan arsitektur." — HuggingFace research.

Mamba lebih modern tapi implementasi lebih tricky. Priority sekarang adalah **selesaikan full cycle**, bukan experiment arsitektur.

Dua tweak dipilih karena:
- `rope_theta=500_000` → 1 angka, proven di LLaMA-3
- Logit soft-capping → 1 baris, proven di Gemma 2, training lebih stabil

---

## Kenapa Reuse Tokenizer DFD-1?

Tokenizer BPE 40k vocab sudah good enough untuk Indo. Hemat 1-2 Kaggle session.

---

## Kenapa Data Cleaning Aggressive?

Root cause repetition loop di DFD-1: training data penuh navigation breadcrumbs dan boilerplate web. 2B token bersih > 4.5B token kotor.

---

## SFT: Kenapa Instruction Tuning?

Setelah PT selesai, SFT ke instruction following karena:
- Dataset tersedia (Indo Alpaca, translated)
- Hasilnya bisa di-demo langsung
- Paling mudah dijelaskan di portfolio/interview
- Base untuk SFT berikutnya (summarization, QA) kalau mau

---

## Kenapa Bukan Bikin Arsitektur Sendiri?

Bikin arsitektur baru yang genuinely better = PhD-level research. Tujuan sekarang: selesaikan full cycle, publish, belajar. Eksperimen arsitektur = IDK-2 nanti.

---

## Apakah IDK-1 Berguna untuk dCode?

Tidak langsung. IDK-1 = base model bahasa Indo. dCode butuh coding assistant.
Plan nanti: fine-tune Qwen2.5-Coder untuk dCode (beda project). IDK-1 = porto/research.
