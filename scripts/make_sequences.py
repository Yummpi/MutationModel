import pandas as pd
from pathlib import Path

# load wild sequence
with open("data/raw/wild.fasta") as f:
    wt = "".join(f.read().splitlines()[1:])

df = pd.read_csv("data/raw/assay/normalized.csv").sample(n=5000, random_state=0)

out = Path("data/sequences")
out.mkdir(parents=True, exist_ok=True)

# save wild
with open(out / "wild.fasta", "w") as f:
    f.write(">wild\n" + wt + "\n")

for m in df["mut"]:
    ref, pos, alt = m[0], int(m[1:-1]) - 1, m[-1]
    seq = wt[:pos] + alt + wt[pos+1:]
    with open(out / f"{m}.fasta", "w") as f:
        f.write(f">{m}\n{seq}\n")
