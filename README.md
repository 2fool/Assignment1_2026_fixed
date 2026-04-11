# Assignment 1 — QANet (Dual-Notebook Guide)

**COMP5329 Deep Learning — University of Sydney, Semester 1 2026**

This repository contains a repaired QANet assignment codebase plus additional fixes for:
- training / evaluation pipeline correctness,
- label and preprocessing diagnostics,
- checkpoint loading / saving,
- QANet encoder stability,
- Colab rerun reliability.

## Which notebook should you use?

This repo now keeps **two notebook workflows** so you can both **finish the assignment safely** and **optionally chase stronger final metrics**.

### 1) `assignment1.ipynb` — **submission-safe notebook**
This is the notebook you should treat as the **main grading notebook**.

It follows the original project protocol more closely:
- **train:** `train-mini.json`
- **dev/eval:** `dev-v1.1.json`
- **word vectors:** `glove.mini.txt`

Use this notebook for:
- tutor-facing reruns,
- controlled experiments,
- report figures / tables,
- the safest end-to-end submission path.

### 2) `assignment1_fullrun.ipynb` — **optional high-score notebook**
This notebook is for an additional heavier retrain after the submission notebook is already working.

It uses:
- **train:** `train-v1.1.json`
- **dev/eval:** `dev-v1.1.json`
- **word vectors:** `glove.840B.300d.txt`

Use this notebook only if you want:
- stronger final EM / F1,
- an extra full-data checkpoint to report as an extended result.

---

## What was fixed in the codebase?

Key repairs already included in this repository:
- correct QA loss usage,
- correct checkpoint schema and evaluation reload,
- improved best-span decoding,
- corrected normalization behavior for `[B, C, L]` tensors,
- corrected QANet encoder residual / normalization ordering,
- stronger preprocessing / embedding coverage diagnostics,
- tiny-subset overfit sanity check,
- Colab-friendly rerun behavior.

Because preprocessing and model logic changed, you should **rebuild `_data/` outputs** before any serious retraining run.

---

## Colab quick start

### Step 0 — Use GPU runtime
On Colab:
- **Runtime -> Change runtime type -> GPU**

### Step 1 — Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2 — Clone or reuse the repo
```python
import os
REPO_URL = "https://github.com/2fool/Assignment1_2026_fixed.git"
PROJECT_ROOT = "/content/drive/MyDrive/Assignment1_2026_fixed"

if not os.path.exists(PROJECT_ROOT):
    !git clone {REPO_URL} {PROJECT_ROOT}
else:
    print("Already cloned")
```

### Step 3 — Install dependencies
```python
!pip install -r /content/drive/MyDrive/Assignment1_2026_fixed/requirements.txt -q
!python -m spacy download en_core_web_sm
```

### Step 4 — Open the notebook you want
- For the main assignment workflow: **`assignment1.ipynb`**
- For an optional stronger full-data retrain: **`assignment1_fullrun.ipynb`**

Run cells **from top to bottom** after a clean runtime restart.

---

## Recommended workflow

### Workflow A — Main assignment / report workflow
Use **`assignment1.ipynb`**.

It already includes:
- repo sync,
- preprocessing rebuild,
- preprocessing statistics checks,
- label upper-bound sanity checks,
- tiny-subset overfit sanity checks,
- training,
- final evaluation.

This is the safest notebook for:
- reproducibility,
- experiments,
- submission executability.

### Workflow B — Optional stronger final checkpoint
Use **`assignment1_fullrun.ipynb`** only after Workflow A is healthy.

That notebook keeps the same sanity philosophy but switches to a **full-data training run**.

---

## Dataset protocol summary

### Submission-safe notebook (`assignment1.ipynb`)
- `download_mini()`
- `_data/squad/train-mini.json`
- `_data/squad/dev-v1.1.json`
- `_data/glove/glove.mini.txt`

### Full-data notebook (`assignment1_fullrun.ipynb`)
- `download()`
- `_data/squad/train-v1.1.json`
- `_data/squad/dev-v1.1.json`
- `_data/glove/glove.840B.300d.txt`

---

## Important sanity gates

Both notebooks are designed to stop early if major assumptions fail.

### 1) Preprocess stats sanity
Checks that:
- enough records were built,
- embedding coverage is not suspiciously low.

### 2) Label upper bound sanity
Decodes stored gold `(y1, y2)` spans back into text and compares them to the official answers.

If this upper bound is poor, long training is not trustworthy.

### 3) Tiny-subset overfit sanity
This is **not** the final assignment dataset.
It is a debugging gate that asks whether the model can memorize a tiny real subset.

If this fails, the model implementation is still broken and you should not continue into long training.

---

## Output directories

### Submission-safe notebook
- model dir: `_model_mini/`
- log dir: `_log_mini/`

### Full-data notebook
- model dir: `_model_full/`
- log dir: `_log_full/`

---

## Suggested reporting strategy

A strong and safe submission usually looks like this:

1. Use **`assignment1.ipynb`** for the main report experiments.
2. Run your controlled comparisons there (optimizer / scheduler / normalization, etc.).
3. If time and resources permit, add one extra **full-data result** from `assignment1_fullrun.ipynb` as an extended stronger checkpoint.

That gives you:
- a notebook tutors can rerun reliably,
- experiments that are easier to compare cleanly,
- optional stronger final EM/F1 from the full-data path.

---

## Practical notes

- If local and Colab results disagree, trust a **clean Colab rerun** with rebuilt `_data/` outputs.
- If the sync cell does not show the expected latest files, **restart the runtime** and rerun from the top.
- If you switch between the two notebooks, remember that they intentionally use **different datasets and output directories**.
- Full-data training is much heavier than the submission-safe mini-data workflow.

---

## Repository structure highlights

- `assignment1.ipynb` — main submission notebook
- `assignment1_fullrun.ipynb` — optional full-data notebook
- `TrainTools/overfit_check.py` — tiny-subset overfit gate
- `Tools/data_diagnostics.py` — label upper-bound diagnostics
- `TrainTools/train.py` — training entry point
- `EvaluateTools/evaluate.py` — checkpoint evaluation entry point

