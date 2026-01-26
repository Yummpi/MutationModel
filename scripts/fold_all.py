import torch
import esm
from pathlib import Path
from Bio import SeqIO

device = "cuda" if torch.cuda.is_available() else "cpu"

model = esm.pretrained.esmfold_v1()
model = model.eval().to(device)

seq_dir = Path("data/sequences")
out_dir = Path("data/embeddings")
out_dir.mkdir(parents=True, exist_ok=True)

for fasta in seq_dir.glob("*.fasta"):
    name = fasta.stem
    with open(fasta) as f:
        seq = "".join(f.read().splitlines()[1:])

    with torch.no_grad():
        output = model.infer_pdb(seq)

    # extract per-residue embeddings
    with torch.no_grad():
        emb = model.embed_tokens(seq).cpu()

    torch.save(emb, out_dir / f"{name}.pt")
    print(f"Saved {name}.pt")
