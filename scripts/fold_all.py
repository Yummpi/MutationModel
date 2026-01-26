import torch, glob, os
from app import run_esmfold   # import your existing fold function

os.makedirs("data/embeddings", exist_ok=True)

# Fold wild-type
wt_seq = open("data/raw/wild.fasta").read().splitlines()[1]
torch.save(run_esmfold(wt_seq), "data/embeddings/wild.pt")

# Fold mutants
for file in glob.glob("data/raw/seqs/*.fasta"):
    seq = open(file).read().splitlines()[1]
    data = run_esmfold(seq)
    name = os.path.basename(file).replace(".fasta",".pt")
    torch.save(data, f"data/embeddings/{name}")
    print("Saved", name)
