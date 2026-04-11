"""
Tiny-subset overfit check for QANet.

This is intended as a preflight sanity test before launching a long full-data
run. If the model cannot memorize a very small subset, the issue is likely in
the data pipeline or implementation rather than in hyperparameters.
"""

import os
import tempfile

import numpy as np
import ujson as json

from TrainTools.train import train
from EvaluateTools.evaluate import evaluate


def _slice_npz(src_npz: str, dst_npz: str, limit: int):
    data = np.load(src_npz)
    payload = {k: data[k][:limit] for k in data.files}
    np.savez(dst_npz, **payload)


def _slice_eval(src_eval_json: str, dst_eval_json: str, ids):
    with open(src_eval_json, "r") as f:
        eval_data = json.load(f)
    sliced = {str(i): eval_data[str(i)] for i in ids}
    with open(dst_eval_json, "w") as f:
        json.dump(sliced, f)


def overfit_check(
    train_npz="_data/train.npz",
    word_emb_json="_data/word_emb.json",
    char_emb_json="_data/char_emb.json",
    train_eval_json="_data/train_eval.json",
    subset_size: int = 64,
    save_dir: str = "_overfit_check",
    log_dir: str = "_overfit_log",
    num_steps: int = 400,
    checkpoint: int = 100,
    batch_size: int = 16,
    seed: int = 42,
    learning_rate: float = 1e-3,
    **train_kwargs,
):
    """
    Train/evaluate on the same tiny subset.

    Expected behavior:
      - F1 / EM should rise sharply.
      - If they stay near random, something is likely still broken.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="qanet_overfit_") as tmp:
        sub_npz = os.path.join(tmp, "subset.npz")
        sub_eval = os.path.join(tmp, "subset_eval.json")

        full = np.load(train_npz)
        ids = full["ids"][:subset_size].tolist()
        _slice_npz(train_npz, sub_npz, subset_size)
        _slice_eval(train_eval_json, sub_eval, ids)

        results = train(
            train_npz=sub_npz,
            dev_npz=sub_npz,
            word_emb_json=word_emb_json,
            char_emb_json=char_emb_json,
            train_eval_json=sub_eval,
            dev_eval_json=sub_eval,
            save_dir=save_dir,
            log_dir=log_dir,
            num_steps=num_steps,
            checkpoint=checkpoint,
            batch_size=batch_size,
            seed=seed,
            optimizer_name="adam",
            scheduler_name="lambda",
            learning_rate=learning_rate,
            loss_name="qa_nll",
            norm_name="layer_norm",
            test_num_batches=-1,
            max_answer_len=30,
            early_stop=100,
            **train_kwargs,
        )

        metrics = evaluate(
            dev_npz=sub_npz,
            word_emb_json=word_emb_json,
            char_emb_json=char_emb_json,
            dev_eval_json=sub_eval,
            save_dir=save_dir,
            log_dir=log_dir,
            ckpt_name="model.pt",
            test_num_batches=-1,
            max_answer_len=30,
        )

    return {
        "train_results": results,
        "eval_metrics": metrics,
    }
