import torch, csv, os

pairs = []

with open("data/raw/GFP_dataset.csv") as f:
    r = csv.reader(f)
    next(r)
    for mut, score in r:
        pos = int(mut[1:-1]) - 1
        pairs.append({
            "wild": "data/embeddings/wild.pt",
            "mut": f"data/embeddings/{mut}.pt",
            "pos": pos,
            "label": float(score)
        })

torch.save(pairs, "data/pairs.pt")
print("Saved data/pairs.pt")
