import torch
from torch.utils.data import DataLoader
from mutation_model import MutationEffectTransformer
from dataset import MutationDataset

pairs = torch.load("data/pairs.pt")
ds = MutationDataset(pairs, window=8)
dl = DataLoader(ds, batch_size=8, shuffle=True)

model = MutationEffectTransformer(embed_dim=ds[0][0].shape[-1]).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = torch.nn.MSELoss()

for epoch in range(40):
    epoch_loss = 0
    for x,y in dl:
        x,y = x.cuda(), y.cuda()
        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch}: loss={epoch_loss/len(dl):.4f}")
    torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")
