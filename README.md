# Assignment 1 — QANet (High-EM Ready Repo)

**COMP5329 Deep Learning — University of Sydney, Semester 1 2026**

This repository is a repaired QANet assignment codebase that has been further adjusted for **higher-quality training**, not just notebook executability.

---

## 1. What was changed

To address the low-EM issue from the previous mini-data / short-run setup, the following changes were made.

### Code changes

#### `EvaluateTools/eval_utils.py`
- changed evaluation decoding from **independent start/end argmax** to **joint best-span decoding**;
- enforces valid answer spans with:
  - `start <= end`
  - bounded `max_answer_len`

This is important because extractive QA should choose the **best valid span**, not two unrelated positions.

#### `TrainTools/train.py`
- changed checkpoint logic to save the **best checkpoint by dev performance** instead of effectively keeping only the last one;
- added `max_answer_len` into the training/evaluation flow;
- improved model-selection behavior for long training runs.

#### `EvaluateTools/evaluate.py`
- changed evaluation loading so it can reuse the **architecture config stored inside the checkpoint**;
- this reduces mismatch risk between training-time config and evaluation-time config.

### Notebook changes

#### `assignment1.ipynb`
- switched the default workflow from **mini data** to **full SQuAD v1.1**;
- changed the default training recipe to a more realistic high-EM setup:
  - full `train-v1.1.json`
  - full `glove.840B.300d.txt`
  - `optimizer_name="adam"`
  - `scheduler_name="cosine"`
  - `num_steps=12000`
  - `batch_size=16`
  - `test_num_batches=-1`

### Documentation changes

#### `README.md`
- rewritten to explain:
  - what was changed;
  - why previous EM could stay low;
  - how users should train the updated repo;
  - how to run smoke tests vs real training.

#### `requirements.txt`
- cleaned duplicated entries.

---

## 2. Why previous EM could be low

If you train with:
- `train-mini.json`
- `glove.mini.txt`
- `num_steps=200`

then low EM is expected.

That setup is useful only for a **pipeline smoke test**.
It is **not** a serious configuration for trying to reach a high Exact Match score.

If the goal is to improve EM substantially, use:
- **full** `train-v1.1.json`
- **full** `dev-v1.1.json`
- **full** `glove.840B.300d.txt`
- **thousands of training steps**
- **full dev-set evaluation** for checkpoint selection

---

## 3. How users should use this repo

## Option A — Recommended: use the notebook on Colab

Main notebook:
- `assignment1.ipynb`

### Step 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2. Clone the repo to Drive

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

### Step 4. Open and run `assignment1.ipynb`

The notebook is already configured for:
- full-data download;
- full-data preprocessing;
- long training;
- best-checkpoint evaluation.

---

## Option B — Run the pipeline manually

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

### 2) Preprocess full data

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

### 3) Train

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
    num_steps=12000,
    checkpoint=1000,
    batch_size=16,
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

### 4) Evaluate the best checkpoint

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

## 4. Recommended training defaults

Current recommended starting recipe:
- full SQuAD v1.1 train/dev
- full GloVe 840B 300d
- `optimizer_name="adam"`
- `scheduler_name="cosine"`
- `batch_size=16`
- `num_steps=12000`
- `checkpoint=1000`
- `test_num_batches=-1`
- `max_answer_len=30`

If Colab memory is not enough:
- reduce `batch_size` from `16` to `8`

If the model is still improving late in training:
- increase `num_steps` to `16000` or `20000`

If training is unstable:
- try `learning_rate=5e-4`

---

## 5. Smoke test mode

If a user only wants to verify that the pipeline runs end-to-end quickly, they can temporarily switch back to:
- `download_mini()`
- `train-mini.json`
- `glove.mini.txt`
- `num_steps=20` or `200`

But that mode should only be used for:
- dependency checks;
- notebook executability checks;
- pipeline smoke tests.

It should **not** be used as the final training setup for EM comparison.

---

## 6. Notes

- Metrics in this repo are returned on a **0–100 scale**, not 0–1.
- Final EM depends mostly on:
  - training data scale,
  - training duration,
  - checkpoint selection,
  - and decoding correctness.
- If local and Colab results differ, trust the clean Colab rerun using the same configuration.
