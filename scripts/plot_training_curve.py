import os
import re
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import MutationDataset
from mutation_model import MutationEffectTransformer


def get_epoch(path: str) -> int:
    m = re.search(r"epoch_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = torch.load("data/pairs.pt", map_location="cpu")
    ds = MutationDataset(pairs, window=8)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    embed_dim = ds[0][0].shape[-1]
    model = MutationEffectTransformer(embed_dim).to(device)
    loss_fn = torch.nn.MSELoss()

    ckpts = [f for f in os.listdir("models") if f.startswith("epoch_") and f.endswith(".pt")]
    ckpts = sorted(ckpts, key=lambda x: get_epoch(x))

    epochs = []
    losses = []

    for ck in ckpts:
        ep = get_epoch(ck)
        path = os.path.join("models", ck)

        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        total = 0.0
        n = 0
        with torch.no_grad():
            for x, y in dl:
                x = x.to(device)
                y = y.to(device).view(-1, 1)
                pred = model(x)
                loss = loss_fn(pred, y)
                total += loss.item()
                n += 1

        avg = total / max(n, 1)
        print(f"epoch {ep}: loss={avg:.4f}")
        epochs.append(ep)
        losses.append(avg)

    plt.plot(epochs, losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training loss by checkpoint")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

