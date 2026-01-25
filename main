import streamlit as st
st.set_page_config(page_title="Mutation Model")
st.title("Mutation Model")

from stmol import showmol
import py3Dmol
import requests
import biotite.structure.io as bsio

#st.set_page_config(layout = 'wide')
st.sidebar.title('ESMFold')
st.sidebar.write('[*ESMFold*](https://esmatlas.com/about) is an end-to-end single sequence protein structure predictor based on the ESM-2 language model. For more information, read the [research article](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2) and the [news article](https://www.nature.com/articles/d41586-022-03539-1) published in *Nature*.')

# stmol
def render_mol(pdb):
    pdbview = py3Dmol.view()
    pdbview.addModel(pdb,'pdb')
    pdbview.setStyle({'cartoon':{'color':'spectrum'}})
    pdbview.setBackgroundColor('white')#('0xeeeeee')
    pdbview.zoomTo()
    pdbview.zoom(2, 800)
    pdbview.spin(True)
    showmol(pdbview, height = 500,width=800)

# Protein sequence input
DEFAULT_SEQ = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
txt = st.sidebar.text_area('Input sequence', DEFAULT_SEQ, height=275)

# ESMfold
def update(sequence=txt):
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    response = requests.post('https://api.esmatlas.com/foldSequence/v1/pdb/', headers=headers, data=sequence)
    name = sequence[:3] + sequence[-3:]
    pdb_string = response.content.decode('utf-8')

    with open('predicted.pdb', 'w') as f:
        f.write(pdb_string)

    struct = bsio.load_structure('predicted.pdb', extra_fields=["b_factor"])
    b_value = round(struct.b_factor.mean(), 4)

    # Display protein structure
    st.subheader('Visualization of predicted protein structure')
    render_mol(pdb_string)

    # plDDT value is stored in the B-factor field
    st.subheader('plDDT')
    st.write('plDDT is a per-residue estimate of the confidence in prediction on a scale from 0-100.')
    st.info(f'plDDT: {b_value}')

    st.download_button(
        label="Download PDB",
        data=pdb_string,
        file_name='predicted.pdb',
        mime='text/plain',
    )

predict = st.sidebar.button('Predict', on_click=update)


if not predict:
    st.warning('👈 Enter protein sequence data!')
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

import torch
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

