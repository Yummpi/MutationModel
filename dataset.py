import torch
from torch.utils.data import Dataset

class MutationDataset(Dataset):
    def __init__(self, pairs, window=8):
        self.pairs = pairs
        self.window = window

        wild = torch.load(self.pairs[0]["wild"])
        if isinstance(wild, dict):
            wild = wild["emb"]
        self.wild = wild

        self.mut_cache = []
        for p in self.pairs:
            m = torch.load(p["mut"])
            if isinstance(m, dict):
                m = m["emb"]
            self.mut_cache.append(m)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mut = self.mut_cache[idx]
        delta = mut - self.wild

        pos = int(self.pairs[idx]["pos"])
        w = self.window
        i0 = max(0, pos - w)
        i1 = min(delta.shape[0], pos + w + 1)

        x = delta[i0:i1]

        target_len = 2 * w + 1
        if x.shape[0] < target_len:
            pad = torch.zeros(target_len - x.shape[0], x.shape[1], dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)

        y = torch.tensor(self.pairs[idx]["label"], dtype=torch.float32)
        return x, y

