# Assignment 1 — QANet (Training Guide)

**COMP5329 Deep Learning — University of Sydney, Semester 1 2026**

This repository contains a repaired QANet assignment codebase plus additional fixes for:
- evaluation decoding,
- checkpoint selection,
- preprocessing / embedding coverage,
- normalization / encoder stability,
- Colab RAM usage.

The main goal of this README is to explain **how to retrain correctly** so that the latest fixes actually take effect, and how to use a high-memory Colab setup to push for better EM.

---

## 1. What was fixed

Key fixes already included in this repo:

- **joint best-span decoding** instead of independent start/end argmax;
- **best-checkpoint saving** by dev EM/F1 instead of effectively keeping only the last checkpoint;
- **checkpoint-aware evaluation** so evaluation reuses saved model config;
- **memory-friendlier dataset loading** for full-data Colab runs;
- **channel-wise LayerNorm** for QANet tensors `[B, C, L]`;
- **stronger encoder feed-forward network**;
- **gradient accumulation support**;
- **scheduler alignment with gradient accumulation**;
- **better embedding coverage during preprocessing**:
  - case variants like `Apple` / `apple` are matched more reliably against GloVe;
  - remaining missing words no longer collapse into a single identical OOV vector.

Because of the preprocessing fixes, **you must rebuild `_data/` outputs before doing a serious retrain**.

---

## 2. Important: how to make the new fixes actually take effect

If you only `git pull` and continue training from old cached files, the new fixes may **not** be reflected.

Before a fresh full retrain, delete old generated files:

### Delete old preprocessing outputs
- `_data/train.npz`
- `_data/dev.npz`
- `_data/word_emb.json`
- `_data/char_emb.json`
- `_data/word2idx.json`
- `_data/char2idx.json`
- `_data/train_eval.json`
- `_data/dev_eval.json`
- `_data/dev_meta.json`

### Delete old training outputs
- `_model_full/`
- `_log_full/`

Then:
1. run full-data preprocessing again;
2. retrain from scratch;
3. evaluate the new best checkpoint.

---

## 3. Recommended way to use this repo

## Option A — Recommended: use `assignment1.ipynb` on Colab

Main notebook:
- `assignment1.ipynb`

### Step 0. Use GPU runtime
On Colab:
- **Runtime -> Change runtime type -> GPU**

Do not use CPU runtime for a serious full-data training run.

### Step 1. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2. Clone / update repo

```python
import os
REPO_URL = "https://github.com/2fool/Assignment1_2026_fixed.git"
PROJECT_ROOT = "/content/drive/MyDrive/Assignment1_2026_fixed"

if not os.path.exists(PROJECT_ROOT):
    !git clone {REPO_URL} {PROJECT_ROOT}
else:
    print("Already cloned")
```

### Step 3. Install dependencies

```python
!pip install -r /content/drive/MyDrive/Assignment1_2026_fixed/requirements.txt -q
!python -m spacy download en_core_web_sm
```

### Step 4. Open `assignment1.ipynb` and run all sections in order
The notebook now includes:
- full-data download;
- cache cleanup before preprocessing;
- preprocessing rebuild;
- preprocessing statistics inspection;
- tiny-subset overfit sanity check;
- checkpoint/log cleanup before retraining;
- training;
- evaluation.

---

## Option B — Run manually

### 1) Download full data

```python
from Tools.download import download
download(data_dir="_data")
```

This downloads:
- SQuAD `train-v1.1.json`
- SQuAD `dev-v1.1.json`
- GloVe `glove.840B.300d.txt`
- spaCy model dependency

### 2) Clear old generated files before preprocessing

```python
from pathlib import Path

for path in [
    "_data/train.npz", "_data/dev.npz",
    "_data/word_emb.json", "_data/char_emb.json",
    "_data/word2idx.json", "_data/char2idx.json",
    "_data/train_eval.json", "_data/dev_eval.json", "_data/dev_meta.json",
]:
    p = Path(path)
    if p.exists():
        p.unlink()
```

### 3) Rebuild preprocessing outputs

```python
from Tools.preproc import preprocess

preprocess(
    train_file="_data/squad/train-v1.1.json",
    dev_file="_data/squad/dev-v1.1.json",
    glove_word_file="_data/glove/glove.840B.300d.txt",
    target_dir="_data",
    para_limit=400,
    ques_limit=50,
)
```

### 3.5. Before the long run: inspect preprocess stats, check label upper bound, and run a tiny overfit check

After rebuilding `_data/`, check:
- `_data/preprocess_stats.json`
- how many word embeddings were matched from pretrained GloVe vs randomly initialized
- how many train/dev records were actually built after filtering

Then check the *label upper bound*:

```python
from Tools.data_diagnostics import label_upper_bound

train_label_upper = label_upper_bound("_data/train.npz", "_data/train_eval.json", limit=2000)
dev_label_upper = label_upper_bound("_data/dev.npz", "_data/dev_eval.json", limit=2000)

print(train_label_upper)
print(dev_label_upper)
```

Interpretation:
- if the decoded gold spans already have very low EM/F1 against the official answers, then preprocessing labels are misaligned;
- in that case, a long training run will not fix the problem.

Then run the tiny-subset overfit check before the expensive full run:

After rebuilding `_data/`, check:
- `_data/preprocess_stats.json`
- how many word embeddings were matched from pretrained GloVe vs randomly initialized
- how many train/dev records were actually built after filtering

Then run the tiny-subset overfit check before the expensive full run:

```python
from TrainTools.overfit_check import overfit_check

overfit_results = overfit_check(
    train_npz="_data/train.npz",
    word_emb_json="_data/word_emb.json",
    char_emb_json="_data/char_emb.json",
    train_eval_json="_data/train_eval.json",
    subset_size=64,
    save_dir="_overfit_check",
    log_dir="_overfit_log",
    num_steps=400,
    checkpoint=100,
    batch_size=16,
    learning_rate=3e-3,
)
```

Expected behavior:
- F1 / EM should rise sharply on this tiny subset;
- if they stay near random, do **not** launch the long full-data run yet.

### 4) Clear old checkpoints/logs before a fresh retrain

```python
from pathlib import Path
import shutil

for path in ["_model_full", "_log_full"]:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)

Path("_model_full").mkdir(exist_ok=True)
Path("_log_full").mkdir(exist_ok=True)
```

### 5) Train

#### Preferred configuration for a high-memory GPU runtime

```python
from TrainTools.train import train

results = train(
    train_npz="_data/train.npz",
    dev_npz="_data/dev.npz",
    word_emb_json="_data/word_emb.json",
    char_emb_json="_data/char_emb.json",
    train_eval_json="_data/train_eval.json",
    dev_eval_json="_data/dev_eval.json",
    save_dir="_model_full",
    log_dir="_log_full",
    num_steps=20000,
    checkpoint=2000,
    batch_size=32,
    grad_accum_steps=1,
    seed=42,
    optimizer_name="adam",
    scheduler_name="cosine",
    learning_rate=1e-3,
    loss_name="qa_nll",
    norm_name="layer_norm",
    test_num_batches=-1,
    max_answer_len=30,
)
```

#### Fallback configuration if batch size 32 runs out of memory

```python
results = train(
    train_npz="_data/train.npz",
    dev_npz="_data/dev.npz",
    word_emb_json="_data/word_emb.json",
    char_emb_json="_data/char_emb.json",
    train_eval_json="_data/train_eval.json",
    dev_eval_json="_data/dev_eval.json",
    save_dir="_model_full",
    log_dir="_log_full",
    num_steps=22000,
    checkpoint=2000,
    batch_size=16,
    grad_accum_steps=1,
    seed=42,
    optimizer_name="adam",
    scheduler_name="cosine",
    learning_rate=1e-3,
    loss_name="qa_nll",
    norm_name="layer_norm",
    test_num_batches=-1,
    max_answer_len=30,
)
```

Reason:
- `batch_size=32` is preferred on a very large-memory Colab runtime because it exposes much more data per step and reaches multiple epochs faster;
- if memory is tighter than expected, fall back to `batch_size=16`;
- use `grad_accum_steps>1` only when memory becomes the bottleneck again.

### 6) Evaluate

```python
from EvaluateTools.evaluate import evaluate

metrics = evaluate(
    dev_npz="_data/dev.npz",
    word_emb_json="_data/word_emb.json",
    char_emb_json="_data/char_emb.json",
    dev_eval_json="_data/dev_eval.json",
    save_dir="_model_full",
    log_dir="_log_full",
    ckpt_name="model.pt",
    test_num_batches=-1,
    max_answer_len=30,
)
```

---

## 4. Recommended high-resource training setup

For the current large-memory Colab runtime, the recommended aggressive configuration is:
- `batch_size=32`
- `grad_accum_steps=1`
- `num_steps=20000`
- `checkpoint=2000`
- `optimizer_name="adam"`
- `scheduler_name="cosine"`
- `learning_rate=1e-3`
- `test_num_batches=-1`
- `max_answer_len=30`

If `batch_size=32` unexpectedly runs out of memory, fall back to:
- `batch_size=16`
- `grad_accum_steps=1`
- `num_steps=22000`

If the model is still improving late in training, try:
- `num_steps=24000` or `30000`

If training is unstable, try:
- `learning_rate=5e-4`

---

## 5. What to expect

- Metrics are printed on a **0–100 percentage scale**.
  - Example: `F1 4.7%` means `4.7 percent`, not `4.7 out of 1`.
- A final result around **F1 4.9% / EM 1.3%** is still abnormally low for a repaired QA system.
- If results remain that low after following the full retrain process above, do **not** continue from the same checkpoint; instead verify:
  - you rebuilt preprocessing outputs;
  - you are not evaluating stale checkpoints;
  - you are using the latest `main` branch.

---

## 6. Smoke test mode

If you only need a fast pipeline check, you can temporarily switch back to:
- `download_mini()`
- `train-mini.json`
- `glove.mini.txt`
- `num_steps=20` or `200`

But that mode is only for:
- dependency checks;
- notebook executability checks;
- smoke tests.

It is **not** the recommended setup for trying to recover EM.

---

## 7. Notes

- Full-data SQuAD is memory-heavy on Colab.
- This repo reduces RAM pressure, but full-data training is still much heavier than mini-data smoke tests.
- If local and Colab results disagree, trust a clean Colab rerun using the same configuration and rebuilt `_data/` outputs.
