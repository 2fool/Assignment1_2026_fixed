"""
Dataset diagnostics for SQuAD preprocessing outputs.

The most important check here is the *label upper bound*:
decode the stored gold `(y1, y2)` spans back into answer text and compare that
text against the ground-truth answer list in `*_eval.json`.

If this upper bound is low, then training will fail regardless of model or
hyperparameters because the supervision itself is misaligned.
"""

import numpy as np
import ujson as json

from EvaluateTools.eval_utils import squad_evaluate


def label_upper_bound(record_npz: str, eval_json: str, limit: int = None):
    data = np.load(record_npz)
    with open(eval_json, "r") as f:
        eval_data = json.load(f)

    ids = data["ids"]
    y1s = data["y1s"]
    y2s = data["y2s"]
    if limit is not None:
        ids = ids[:limit]
        y1s = y1s[:limit]
        y2s = y2s[:limit]

    answer_dict = {}
    mismatches = []

    for qid, y1, y2 in zip(ids.tolist(), y1s.tolist(), y2s.tolist()):
        item = eval_data[str(qid)]
        spans = item["spans"]
        context = item["context"]
        if y1 >= len(spans) or y2 >= len(spans) or y1 > y2:
            pred = ""
        else:
            pred = context[spans[y1][0]:spans[y2][1]]
        answer_dict[str(qid)] = pred

        golds = item["answers"]
        if pred not in golds:
            mismatches.append({
                "id": int(qid),
                "pred_from_labels": pred,
                "gold_answers": golds[:3],
                "y1": int(y1),
                "y2": int(y2),
            })

    metrics = squad_evaluate(eval_data, answer_dict)
    return {
        "count": len(answer_dict),
        "f1": metrics["f1"],
        "exact_match": metrics["exact_match"],
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:10],
    }
