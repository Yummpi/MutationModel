import pandas as pd

df = pd.read_csv("data/raw/assay/assay.csv")

df = df.rename(columns={
    "mutant": "mut",
    "DMS_score": "label"
})

df = df[["mut", "label"]]
df.to_csv("data/raw/assay/normalized.csv", index=False)

print("normalized.csv created")
