import torch
from torch.utils.data import Dataset

class MutationDataset(Dataset):
    def __init__(self, pairs, window=8):
        self.pairs = pairs
        self.window = window

    def __getitem__(self, idx):
        wild = torch.load(self.pairs[idx]["wild"])
        mut  = torch.load(self.pairs[idx]["mut"])
        pos  = int(self.pairs[idx]["pos"])
        label = torch.tensor(self.pairs[idx]["label"], dtype=torch.float32)

        # handle tensor vs dict
        if isinstance(wild, dict):
            wild = wild["emb"]
        if isinstance(mut, dict):
            mut = mut["emb"]

        delta = mut - wild  # [L, D]

        i0 = max(0, pos - self.window)
        i1 = min(delta.shape[0], pos + self.window + 1)

        x = delta[i0:i1]

        # pad to fixed length (2*window + 1)
        target_len = 2 * self.window + 1
        if x.shape[0] < target_len:
            pad = torch.zeros(target_len - x.shape[0], x.shape[1])
            x = torch.cat([x, pad], dim=0)

        return x, label

    def __len__(self):
        return len(self.pairs)
