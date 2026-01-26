import torch
from torch.utils.data import Dataset

class MutationDataset(Dataset):
    def __init__(self, pairs, window=8):
        self.pairs = pairs
        self.window = window

    def __getitem__(self, idx):
        wild = torch.load(self.pairs[idx]["wild"])
        mut = torch.load(self.pairs[idx]["mut"])
        pos = self.pairs[idx]["pos"]
        label = torch.tensor(self.pairs[idx]["label"], dtype=torch.float32)

        delta = mut["emb"] - wild["emb"]

        i0 = max(0, pos-self.window)
        i1 = min(len(delta), pos+self.window+1)

        x = delta[i0:i1]
        return x, label

    def __len__(self):
        return len(self.pairs)
