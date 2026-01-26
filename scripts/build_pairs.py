import torch
import pandas as pd
import os

df = pd.read_csv("data/raw/assay/normalized.csv")

pairs = []

for _, row in df.iterrows():
    mut = row["mut"]
    label = float(row["label"])

    pos = int(mut[1:-1]) - 1

    pairs.append({
        "wild": "data/embeddings/wild.pt",
        "mut": f"data/embeddings/{mut}.pt",
        "pos": pos,
        "label": label
    })

os.makedirs("data", exist_ok=True)
torch.save(pairs, "data/pairs.pt")

print(f"Saved {len(pairs)} pairs to data/pairs.pt")
