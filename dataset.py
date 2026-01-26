import torch

class MutationDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, window=8):
        self.pairs = pairs
        self.window = window

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        wild = torch.load(self.pairs[idx]["wild"])
        mut  = torch.load(self.pairs[idx]["mut"])

        # handle both formats: dict {"emb": tensor} or tensor directly
        if isinstance(wild, dict): wild = wild["emb"]
        if isinstance(mut,  dict): mut  = mut["emb"]

        delta = mut - wild  # [L, D] typically

        pos = int(self.pairs[idx]["pos"])
        w = self.window
        start = max(0, pos - w)
        end   = min(delta.shape[0], pos + w + 1)

        x = delta[start:end]

        # pad to fixed length (2w+1)
        target_len = 2*w + 1
        if x.shape[0] < target_len:
            pad = torch.zeros(target_len - x.shape[0], x.shape[1])
            x = torch.cat([x, pad], dim=0)

        y = torch.tensor([float(self.pairs[idx]["label"])], dtype=torch.float32)
        return x, y
