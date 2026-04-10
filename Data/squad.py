import os

import numpy as np
import torch
from torch.utils.data import Dataset


def require_file(path: str, hint: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}\nHint: {hint}")


def sanity_check_cache(args):
    require_file(args.train_npz, "Run preproc.py and place train.npz under data/")
    require_file(args.dev_npz, "Run preproc.py and place dev.npz under data/")
    require_file(args.word_emb_json, "Run preproc.py to generate word_emb.json")
    require_file(args.char_emb_json, "Run preproc.py to generate char_emb.json")
    require_file(args.train_eval_json, "Run preproc.py to generate train_eval.json")
    require_file(args.dev_eval_json, "Run preproc.py to generate dev_eval.json")

    data = np.load(args.train_npz)
    for k in ["context_idxs", "context_char_idxs", "ques_idxs", "ques_char_idxs", "y1s", "y2s", "ids"]:
        if k not in data:
            raise KeyError(f"{args.train_npz} missing key: {k}")
    if data["ids"].shape[0] <= 0:
        raise ValueError("train.npz has zero samples.")
    if (data["y1s"] > data["y2s"]).any():
        raise ValueError("Found y1 > y2 in train.npz (span start > end).")


class SQuADDataset(Dataset):
    def __init__(self, npz_file: str):
        super().__init__()
        # Memory-map arrays instead of materialising the full split as int64
        # tensors up front. This reduces host RAM usage substantially on Colab
        # for full-data SQuAD training.
        data = np.load(npz_file, mmap_mode="r")

        self.context_idxs      = data["context_idxs"]
        self.context_char_idxs = data["context_char_idxs"]
        self.ques_idxs         = data["ques_idxs"]
        self.ques_char_idxs    = data["ques_char_idxs"]
        self.y1s               = data["y1s"]
        self.y2s               = data["y2s"]
        self.ids               = data["ids"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(np.asarray(self.context_idxs[idx])).long(),
            torch.from_numpy(np.asarray(self.context_char_idxs[idx])).long(),
            torch.from_numpy(np.asarray(self.ques_idxs[idx])).long(),
            torch.from_numpy(np.asarray(self.ques_char_idxs[idx])).long(),
            torch.tensor(int(self.y1s[idx]), dtype=torch.long),
            torch.tensor(int(self.y2s[idx]), dtype=torch.long),
            torch.tensor(int(self.ids[idx]), dtype=torch.long),
        )
