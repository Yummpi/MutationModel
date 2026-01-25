mutation = st.sidebar.text_input("Mutation (e.g. A123V)", "A50V")

def parse_mutation(m):
    wt = m[0]
    pos = int(m[1:-1]) - 1
    mut = m[-1]
    return wt, pos, mut

def apply_mutation(seq, pos, aa):
    return seq[:pos] + aa + seq[pos+1:]

import torch
import esm

@st.cache_resource
def load_esmfold():
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    return model

def fold_local(seq):
    model = load_esmfold()
    with torch.no_grad():
        out = model.infer(seq)
    return out

def extract_features(wild, mut, pos, window=8):
    w_emb = wild["representations"][36]
    m_emb = mut["representations"][36]

    delta = m_emb - w_emb
    i0 = max(0, pos-window)
    i1 = min(len(delta), pos+window+1)
    return delta[i0:i1].unsqueeze(0)

import torch.nn as nn

class MutationEffectModel(nn.Module):
    def __init__(self, embed_dim, hidden=256, layers=4, heads=8):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden*4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        return self.head(x.mean(dim=1)).squeeze(-1)

from mutation_model import MutationEffectModel

@st.cache_resource
def load_mut_model(embed_dim):
    model = MutationEffectModel(embed_dim)
    model.load_state_dict(torch.load("mutation_ai.pt", map_location="cpu"))
    model.eval()
    return model

wt, pos, aa = parse_mutation(mutation)
mut_seq = apply_mutation(txt, pos, aa)

wild = fold_local(txt)
mut = fold_local(mut_seq)

x = extract_features(wild, mut, pos)
embed_dim = x.shape[-1]
model = load_mut_model(embed_dim)

with torch.no_grad():
    score = model(x).item()

st.subheader("Mutation Effect Score")
st.metric("Predicted Impact", round(score, 4))

from torch.utils.data import DataLoader
from mutation_model import MutationEffectModel
from dataset import MutationDataset

pairs = torch.load("pairs.pt")
ds = MutationDataset(pairs)
dl = DataLoader(ds, batch_size=8, shuffle=True)

embed_dim = ds[0][0].shape[-1]
model = MutationEffectModel(embed_dim).cuda()

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = torch.nn.MSELoss()

for epoch in range(30):
    for x,y in dl:
        x,y = x.cuda(), y.cuda()
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(epoch, loss.item())

torch.save(model.state_dict(), "mutation_ai.pt")

{
 "wild": "wild.pt",
 "mut": "mut.pt",
 "pos": 123,
 "label": -1.7   # ΔΔG, fitness change, pathogenicity score, etc
}

