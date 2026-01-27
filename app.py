import streamlit as st
from stmol import showmol
import py3Dmol
import requests
import biotite.structure.io as bsio
import torch
from mutation_model import MutationEffectTransformer
import os

WEIGHTS = "models/epoch_18.pt"  # set to the checkpoint you want

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
if os.path.exists(WEIGHTS):
    model = MutationEffectTransformer(embed_dim=1024).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

# ---- ESMFold ----
def update(sequence: str):
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(
        "https://api.esmatlas.com/foldSequence/v1/pdb/",
        headers=headers,
        data=sequence,
        timeout=120,
    )
    r.raise_for_status()
    pdb_string = r.content.decode("utf-8")

    with open("predicted.pdb", "w", encoding="utf-8") as f:
        f.write(pdb_string)

    return pdb_string

def render_mol(pdb):
    pdbview = py3Dmol.view()
    pdbview.addModel(pdb, "pdb")
    pdbview.setStyle({"cartoon": {"color": "spectrum"}})
    pdbview.setBackgroundColor("white")
    pdbview.zoomTo()
    pdbview.zoom(2, 800)
    pdbview.spin(True)
    showmol(pdbview, height=500, width=800)

DEFAULT_SEQ = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
txt = st.sidebar.text_area("Input sequence", DEFAULT_SEQ, height=275)

predict = st.sidebar.button("Predict")
pdb_string = None

if predict:
    pdb_string = update(txt)

if pdb_string is not None:
    st.subheader("Visualization of predicted protein structure")
    render_mol(pdb_string)

    try:
        struct = bsio.load_structure("predicted.pdb", extra_fields=["b_factor"])
        b_value = float(struct.b_factor.mean())
    except Exception:
        b_value = None

    st.subheader("plDDT")
    st.write("plDDT is a per-residue estimate of confidence from 0–100.")
    st.info(f"plDDT: {b_value if b_value is not None else 'unavailable'}")

    st.download_button(
        label="Download PDB",
        data=pdb_string,
        file_name="predicted.pdb",
        mime="text/plain",
    )
else:
    st.warning("Enter protein sequence data and press Predict.")

mutation = st.sidebar.text_input("Mutation (e.g. A123V)", "A50V")

import re

def parse_mutation(m: str):
    m = m.strip().upper()
    if not re.match(r"^[A-Z]\d+[A-Z]$", m):
        return None
    wt = m[0]
    pos = int(m[1:-1]) - 1
    aa2 = m[-1]
    return wt, pos, aa2

def aa_to_idx(a: str) -> int:
    # 20 standard amino acids
    aas = "ACDEFGHIKLMNPQRSTVWY"
    return aas.index(a)

if model is None:
    st.error("Trained model not found. Train first or set WEIGHTS to an existing file.")
    st.stop()

st.subheader("Mutation effect prediction")

pm = parse_mutation(mutation)
if pm is None:
    st.error("Mutation format must look like A123V")
    st.stop()

wt_aa, pos0, mut_aa = pm

wild = torch.load("data/embeddings/wild.pt", map_location="cpu")
mut_path = f"data/embeddings/{mutation.upper()}.pt"

if isinstance(wild, dict):
    wild = wild["emb"]

if not os.path.exists(mut_path):
    st.error(f"Missing embedding file: {mut_path}")
    st.stop()

mut_emb = torch.load(mut_path, map_location="cpu")
if isinstance(mut_emb, dict):
    mut_emb = mut_emb["emb"]

delta = mut_emb - wild  # [L, D]
w = 8
i0 = max(0, pos0 - w)
i1 = min(delta.shape[0], pos0 + w + 1)
x = delta[i0:i1]
target_len = 2*w + 1
if x.shape[0] < target_len:
    pad = torch.zeros(target_len - x.shape[0], x.shape[1], dtype=x.dtype)
    x = torch.cat([x, pad], dim=0)

with torch.no_grad():
    score = model(x.unsqueeze(0).to(device)).item()

st.metric("Predicted effect score", f"{score:.4f}")





