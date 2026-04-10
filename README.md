# Assignment 1 — QANet (High-EM Ready Repo)

**COMP5329 Deep Learning — University of Sydney, Semester 1 2026**

This repository is a repaired QANet assignment codebase that has been further adjusted for **real training quality**, not just notebook executability.

## What changed in this revision

To address the low-EM feedback from the mini-data / short-run setup, this repo now includes:

- **joint span decoding** during evaluation (`start <= end` and bounded answer length), instead of independent start/end argmax;
- **best-checkpoint saving by dev EM/F1**, instead of effectively keeping only the last checkpoint;
- **checkpoint-aware evaluation**, so `evaluate()` reuses the architecture config stored in the checkpoint;
- a rewritten **`assignment1.ipynb`** configured for **full SQuAD v1.1 training**;
- an updated **README** with a recommended high-EM training recipe.

## Important practical conclusion

If you train with:
- `train-mini.json`
- `glove.mini.txt`
- `num_steps=200`

then low EM is expected. That setup is suitable only for a **pipeline smoke test**.

If your goal is to push EM much higher, use:
- **full** `train-v1.1.json`
- **full** `glove.840B.300d.txt`
- **thousands of training steps**
- **Adam** and a stable scheduler
- **full dev-set evaluation** for model selection

## Main notebook

The main notebook is:
- `assignment1.ipynb`

It is now configured for **full-data training** by default.

## Recommended training recipe

The default high-EM notebook recipe uses:

- full SQuAD v1.1 train/dev
- full GloVe 840B 300d
- `optimizer_name="adam"`
- `scheduler_name="cosine"`
- `batch_size=16`
- `num_steps=12000`
- `checkpoint=1000`
- `test_num_batches=-1`
- `max_answer_len=30`

If Colab memory is tight, reduce:
- `batch_size` from `16` to `8`

If results plateau too early, try:
- `num_steps=16000` or `20000`
- `learning_rate=5e-4`

## Quick start on Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import os
REPO_URL = "https://github.com/2fool/Assignment1_2026_fixed.git"
PROJECT_ROOT = "/content/drive/MyDrive/Assignment1_2026_fixed"

if not os.path.exists(PROJECT_ROOT):
    !git clone {REPO_URL} {PROJECT_ROOT}
else:
    print("Already cloned")
```

```python
!pip install -r /content/drive/MyDrive/Assignment1_2026_fixed/requirements.txt -q
!python -m spacy download en_core_web_sm
```

Then open and run `assignment1.ipynb`.

## Full-data pipeline summary

### 1. Download full data
The notebook now calls:

```python
from Tools.download import download
download(data_dir="_data")
```

This downloads:
- SQuAD `train-v1.1.json`
- SQuAD `dev-v1.1.json`
- GloVe `glove.840B.300d.txt`
- the required spaCy model

### 2. Preprocess full data

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

### 3. Train

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

### 4. Evaluate the best checkpoint

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

## Smoke test mode

If you only want to verify that the pipeline runs end-to-end quickly, you can temporarily switch back to:
- `download_mini()`
- `train-mini.json`
- `glove.mini.txt`
- `num_steps=20` or `200`

But do **not** treat that configuration as a serious benchmark for EM.

## Notes

- The evaluation metric returned by this codebase is on a **0–100 scale**, not 0–1.
- Final EM quality depends mostly on **training data scale**, **training time**, and **model selection**, not just on “whether the notebook runs”.
- If local and Colab results differ, trust the clean Colab rerun with the same data and checkpoint settings.
