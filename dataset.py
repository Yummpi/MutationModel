import torch
from torch.utils.data import Dataset
from functools import lru_cache

# small in-RAM cache of recently used mutant tensors
@lru_cache(maxsize=256)
def _load_pt(path: str):
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        t = t["emb"]
    return t


class MutationDataset(Dataset):
    def __init__(self, pairs, window=8):
        self.pairs = pairs
        self.window = window

        wild = torch.load(self.pairs[0]["wild"], map_location="cpu")
        if isinstance(wild, dict):
            wild = wild["emb"]
        self.wild = wild  # [L, D]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if idx % 500 == 0:
            print(f"accessed sample {idx}", flush=True)

        p = self.pairs[idx]
        mut = _load_pt(p["mut"])

        delta = mut - self.wild  # [L, D]

        pos = int(p["pos"])
        w = self.window
        i0 = max(0, pos - w)
        i1 = min(delta.shape[0], pos + w + 1)

        x = delta[i0:i1]

        target_len = 2 * w + 1
        if x.shape[0] < target_len:
            pad = torch.zeros(target_len - x.shape[0], x.shape[1], dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)

        y = torch.tensor(p["label"], dtype=torch.float32)
        return x, y
