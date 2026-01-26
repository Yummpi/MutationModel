import torch
import esm
from pathlib import Path

device = "cuda"

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

seq_dir = Path("data/sequences")
out_dir = Path("data/embeddings")
out_dir.mkdir(parents=True, exist_ok=True)

for fasta in seq_dir.glob("*.fasta"):
    name = fasta.stem
    with open(fasta) as f:
        seq = "".join(f.read().splitlines()[1:])

    batch = [("protein", seq)]
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[33])
        emb = out["representations"][33][0].cpu()

    torch.save(emb, out_dir / f"{name}.pt")
    print(f"Saved {name}.pt")
