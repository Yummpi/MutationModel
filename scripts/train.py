import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from torch.utils.data import DataLoader

from mutation_model import MutationEffectTransformer
from dataset import MutationDataset

def main():
    pairs = torch.load("data/pairs.pt")
    ds = MutationDataset(pairs, window=8)
    dl = DataLoader(ds, batch_size=14, shuffle=True, num_workers=0)
    print("pairs:", len(pairs))
    print("batches:", len(dl))

    embed_dim = ds[0][0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MutationEffectTransformer(embed_dim).to(device)
<<<<<<< HEAD
    ckpt = "models/epoch_1.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
=======
>>>>>>> 3f1ab4f384448123aef716d220f4467bf4a6f775
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.MSELoss()

    os.makedirs("models", exist_ok=True)

<<<<<<< HEAD
    for epoch in range(2, 30):
=======
    for epoch in range(30):
>>>>>>> 3f1ab4f384448123aef716d220f4467bf4a6f775
        total = 0.0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device).view(-1, 1)   # force [B,1]
            pred = model(x)
            loss = loss_fn(pred, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch}: loss={total/len(dl):.4f}", flush=True)
        torch.save(model.state_dict(), f"models/epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
