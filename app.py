import os
import re

import streamlit as st
from stmol import showmol
import py3Dmol
import requests
import biotite.structure.io as bsio
import torch

from mutation_model import MutationEffectTransformer
from embedder import load_esm2, embed_sequence, get_cached_embedding, validate_sequence

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="MutationModel", layout="wide")

WEIGHTS = "models/epoch_14.pt"   # you said you want epoch 14
WINDOW = 8                      # must match training window you used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load trained mutation model
# -----------------------
if not os.path.exists(WEIGHTS):
    st.error(f"Missing weights file: {WEIGHTS}")
    st.stop()

model = MutationEffectTransformer(embed_dim=1280).to(DEVICE)
state = torch.load(WEIGHTS, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# -----------------------
# Cache ESM2 (for embeddings on-the-fly)
# -----------------------
@st.cache_resource
def get_esm2(device_str: str):
    dev = torch.device(device_str)
    return load_esm2(dev)

device_str = "cuda" if torch.cuda.is_available() else "cpu"
esm2_model, batch_converter = get_esm2(device_str)

# -----------------------
# Helpers
# -----------------------
def render_mol(pdb: str):
    pdbview = py3Dmol.view()
    pdbview.addModel(pdb, "pdb")
    pdbview.setStyle({"cartoon": {"color": "spectrum"}})
    pdbview.setBackgroundColor("white")
    pdbview.zoomTo()
    pdbview.zoom(2, 800)
    pdbview.spin(True)
    showmol(pdbview, height=500, width=800)

def fold_sequence(sequence: str) -> str:
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(
        "https://api.esmatlas.com/foldSequence/v1/pdb/",
        headers=headers,
        data=sequence,
        timeout=120,
    )
    r.raise_for_status()
    pdb_string = r.content.decode("utf-8")
    return pdb_string

def apply_mutation(seq: str, mut: str):
    m = mut.strip().upper()
    if not re.match(r"^[A-Z]\d+[A-Z]$", m):
        return None
    wt = m[0]
    pos1 = int(m[1:-1])  # 1-based
    aa2 = m[-1]
    if pos1 < 1 or pos1 > len(seq):
        return None
    if seq[pos1 - 1] != wt:
        return None
    return seq[:pos1 - 1] + aa2 + seq[pos1:], pos1 - 1

def fixed_window_delta(wild_emb: torch.Tensor, mut_emb: torch.Tensor, pos0: int, window: int):
    # embeddings are [L, D]
    delta = mut_emb - wild_emb
    i0 = max(0, pos0 - window)
    i1 = min(delta.shape[0], pos0 + window + 1)
    x = delta[i0:i1]
    target_len = 2 * window + 1
    if x.shape[0] < target_len:
        pad = torch.zeros(target_len - x.shape[0], x.shape[1], dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    return x  # [target_len, D]

# -----------------------
# Sidebar inputs
# -----------------------
st.sidebar.title("Protein Tools")

DEFAULT_SEQ = (
    "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
)
txt = st.sidebar.text_area("Input sequence", DEFAULT_SEQ, height=220)

mutation = st.sidebar.text_input("Mutation (e.g. A123V)", "A50V")

colA, colB = st.sidebar.columns(2)
do_fold = colA.button("Predict structure")
do_score = colB.button("Score mutation")

# -----------------------
# Main: validation
# -----------------------
seq = validate_sequence(txt)
if seq is None:
    st.error("Sequence must be amino-acid letters only (ACDEFGHIKLMNPQRSTVWY).")
    st.stop()

# -----------------------
# Structure prediction UI
# -----------------------
if do_fold:
    try:
        pdb_string = fold_sequence(seq)
        st.subheader("Visualization of predicted protein structure")
        render_mol(pdb_string)

        # Try to compute mean b-factor (plDDT proxy) from the returned PDB text
        with open("predicted.pdb", "w", encoding="utf-8") as f:
            f.write(pdb_string)

        try:
            struct = bsio.load_structure("predicted.pdb", extra_fields=["b_factor"])
            b_value = float(struct.b_factor.mean())
        except Exception:
            b_value = None

        st.subheader("plDDT")
        st.write("plDDT is an estimate of confidence from 0–100.")
        st.info(f"plDDT: {b_value if b_value is not None else 'unavailable'}")

        st.download_button(
            label="Download PDB",
            data=pdb_string,
            file_name="predicted.pdb",
            mime="text/plain",
        )
    except Exception as e:
        st.error("Structure prediction failed.")
        st.exception(e)

# -----------------------
# Mutation scoring UI (ESM2 embeddings on-the-fly + caching)
# -----------------------
if do_score:
    try:
        out = apply_mutation(seq, mutation)
        if out is None:
            st.error("Invalid mutation for this sequence (format/position/wild-type mismatch).")
            st.stop()

        mut_seq, pos0 = out

        os.makedirs("data/cache", exist_ok=True)
        wild_cache = get_cached_embedding(seq, cache_dir="data/cache")
        mut_cache = get_cached_embedding(mut_seq, cache_dir="data/cache")

        if os.path.exists(wild_cache):
            wild_emb = torch.load(wild_cache, map_location="cpu")
        else:
            wild_emb = embed_sequence(seq, esm2_model, batch_converter, DEVICE)
            torch.save(wild_emb, wild_cache)

        if os.path.exists(mut_cache):
            mut_emb = torch.load(mut_cache, map_location="cpu")
        else:
            mut_emb = embed_sequence(mut_seq, esm2_model, batch_converter, DEVICE)
            torch.save(mut_emb, mut_cache)

        # Drop special tokens if embedder returns [CLS] ... [EOS]
        if wild_emb.ndim == 2 and wild_emb.shape[0] >= len(seq) + 2:
            wild_emb = wild_emb[1:-1]
            mut_emb = mut_emb[1:-1]

        x = fixed_window_delta(wild_emb, mut_emb, pos0, WINDOW)

        with torch.no_grad():
            score = model(x.unsqueeze(0).to(DEVICE)).item()

        st.subheader("Mutation effect prediction")
        st.metric("Predicted effect score", f"{score:.4f}")
        st.caption(f"Embedding cache: {os.path.basename(wild_cache)} | {os.path.basename(mut_cache)}")

    except Exception as e:
        st.error("Mutation scoring failed.")
        st.exception(e)

