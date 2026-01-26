import torch
from torch.utils.data import DataLoader
from mutation_model import MutationEffectTransformer
from dataset import MutationDataset
import os

pairs = torch.load("data/pairs.pt")
ds = MutationDataset(pairs, window=8)
dl = DataLoader(ds, batch_size=8, shuffle=True)

embed_dim = ds[0][0].shape[-1]
model = MutationEffectTransformer(embed_dim).cuda()

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = torch.nn.MSELoss()

os.makedirs("models", exist_ok=True)

for epoch in range(30):
    total = 0
    for x,y in dl:
        x,y = x.cuda(), y.cuda()
        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    print(f"Epoch {epoch}: loss={total/len(dl):.4f}")
    torch.save(model.state_dict(), f"models/epoch_{epoch}.pt")
