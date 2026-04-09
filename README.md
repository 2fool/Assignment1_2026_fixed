# Assignment 1 — QANet (Fixed Submission Repo)

**COMP5329 Deep Learning — University of Sydney, Semester 1 2026**

This repository is a repaired, submission-oriented version of the provided Assignment 1 QANet codebase.

It is intended to help a group:
- run the pipeline locally,
- validate executability on Google Colab,
- record bugs and experiments for the report,
- and prepare the final Google Drive submission required by the assignment PDF.

## Primary entrypoint

The grading-facing notebook is:
- `assignment1.ipynb`

Tutors are expected to run the notebook directly, so notebook executability is a hard requirement.

## Recommended workflow

### 1. Local-first development
Use your local machine for:
- code inspection and debugging,
- experiment iteration,
- report writing,
- and keeping a clean git history.

### 2. Colab validation
Use Colab for:
- clean-environment validation,
- notebook executability checks,
- and optional longer training if local compute is limited.

### 3. Final submission
Upload the **entire corrected repository** to Google Drive and place the public sharing link on the **first page of the report**.

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

## Documentation bundle

See the `docs/` folder for the submission support materials:
- `docs/01_总说明与运行部署指南.md`
- `docs/02_Bug清单与报告素材.md`
- `docs/03_实验设计与记录模板.md`
- `docs/04_最终提交Checklist.md`
- `docs/05_报告目录与每节写什么.md`
- `docs/06_三组实验可直接运行命令.md`

## Remaining work before submission

The codebase is repaired and the training/evaluation pipeline is runnable, but the assignment is **not fully complete until the items below are finished**.

### Please use Colab to finish these remaining steps
Colab should be treated as the final validation environment because tutors are expected to run the notebook directly.

### Step 1 — Re-run the final notebook in a clean Colab runtime
Use a fresh runtime and make sure:
- `PROJECT_ROOT` points to `/content/drive/MyDrive/Assignment1_2026_fixed`
- `cwd` points to this repo
- no older `Assignment1_2026` path remains in `sys.path`

### Step 2 — Complete at least 3 experiments
Use the ready-made commands in:
- `docs/06_三组实验可直接运行命令.md`

Recommended experiments:
- Optimizer: `sgd` vs `adam`
- Normalization: `layer_norm` vs `group_norm`
- Scheduler: `step` vs `cosine`

### Step 3 — Save and summarize all experiment results
For each experiment, record:
- hypothesis
- configuration
- best F1 / EM
- final evaluation metrics
- short analysis

Use:
- `docs/03_实验设计与记录模板.md`

### Step 4 — Write the final PDF report
Use:
- `docs/05_报告目录与每节写什么.md`
- `docs/02_Bug清单与报告素材.md`

The report must include:
- Introduction
- complete debugging analysis
- at least 3 experiments
- quantitative results
- analytical discussion
- conclusion

### Step 5 — Upload the final full repository to Google Drive
The final Google Drive repo should include at least:
- source code
- `assignment1.ipynb`
- `requirements.txt`
- docs and experiment records if desired

### Step 6 — Put the public Google Drive link on the first page of the report
The assignment PDF explicitly requires:
- the full corrected repository uploaded to Google Drive
- a public access link
- that link placed prominently on page 1 of the report

### Step 7 — Do one final executability check in Colab
Before submission, verify again in Colab that:
- dependencies install cleanly
- preprocessing runs
- training runs
- evaluation runs
- notebook completes without manual patching

## Notes

- Use this repo as the **only final baseline**. Do not mix it with older broken copies.
- For stable checkpoint saving, prefer the notebook/default training configuration in this repo.
- If Colab imported an older copy before, restart the runtime and ensure `cwd` and `sys.path[0]` point to this repo.
- If your local machine and Colab results disagree, trust the clean Colab validation result for final submission readiness.
