from Bio import SeqIO
import csv, os

wt = str(next(SeqIO.parse("data/raw/wild.fasta", "fasta")).seq)
os.makedirs("data/raw/seqs", exist_ok=True)

def apply_mutation(seq, mut):
    wt_aa = mut[0]
    pos = int(mut[1:-1]) - 1
    new_aa = mut[-1]
    assert seq[pos] == wt_aa
    return seq[:pos] + new_aa + seq[pos+1:], pos

with open("data/raw/GFP_dataset.csv") as f:
    r = csv.reader(f)
    next(r)
    for mut, score in r:
        new_seq, pos = apply_mutation(wt, mut)
        with open(f"data/raw/seqs/{mut}.fasta","w") as out:
            out.write(f">{mut}\n{new_seq}\n")
