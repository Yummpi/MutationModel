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
WEIGHTS = "models/epoch_14.pt"   # keep only epoch_14 in repo
EMBED_DIM = 1280                # matches your checkpoint (256 x 1280 input_proj)
WINDOW = 8
CACHE_DIR = "data/cache"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Load trained mutation model
# -----------------------
if not os.path.exists(WEIGHTS):
    st.error(f"Missing weights file: {WEIGHTS}")
    st.stop()

model = MutationEffectTransformer(embed_dim=EMBED_DIM).to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()


# -----------------------
# Cache ESM2 for embeddings
# -----------------------
@st.cache_resource
def get_esm2(device_str: str):
    dev = torch.device(device_str)
    return load_esm2(dev)

device_str = "cuda" if torch.cuda.is_available() else "cpu"
esm2_model, batch_converter = get_esm2(device_str)


# -----------------------
# UI helpers
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
    pos = int(m[1:-1])  # 1-based
    aa2 = m[-1]
    if pos < 1 or pos > len(seq):
        return None
    if seq[pos - 1] != wt:
        return None
    return seq[:pos - 1] + aa2 + seq[pos:]


# -----------------------
# Sidebar: input
# -----------------------
st.sidebar.title("ESMFold + Mutation Scoring")

DEFAULT_SEQ = (
    "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
)
txt = st.sidebar.text_area("Input sequence", DEFAULT_SEQ, height=275)

predict_btn = st.sidebar.button("Predict structure")
st.sidebar.markdown("---")
mutation = st.sidebar.text_input("Mutation (e.g. A123V)", "A50V")
score_btn = st.sidebar.button("Score mutation")


# -----------------------
# Main: folding + display
# -----------------------
pdb_string = None
seq = validate_sequence(txt)

if predict_btn:
    if seq is None:
        st.error("Sequence must be amino-acid letters only (ACDEFGHIKLMNPQRSTVWY).")
        st.stop()

    pdb_string = fold_sequence(seq)

    st.subheader("Visualization of predicted protein structure")
    render_mol(pdb_string)

    try:
        # parse directly from the string; no need to write predicted.pdb
        with open("predicted.pdb", "w", encoding="utf-8") as f:
            f.write(pdb_string)
        struct = bsio.load_structure("predicted.pdb", extra_fields=["b_factor"])
        b_value = float(struct.b_factor.mean())
    except Exception:
        b_value = None

    st.subheader("plDDT")
    st.write("plDDT is an estimate of prediction confidence from 0–100.")
    st.info(f"plDDT: {b_value if b_value is not None else 'unavailable'}")

    st.download_button(
        label="Download PDB",
        data=pdb_string,
        file_name="predicted.pdb",
        mime="text/plain",
    )
else:
    st.info("Enter a sequence, then click Predict structure.")


# -----------------------
# Main: mutation scoring
# -----------------------
if score_btn:
    if seq is None:
        st.error("Sequence must be amino-acid letters only (ACDEFGHIKLMNPQRSTVWY).")
        st.stop()

    mut_seq = apply_mutation(seq, mutation)
    if mut_seq is None:
        st.error("Invalid mutation for this sequence (format/position/wild-type mismatch).")
        st.stop()

    os.makedirs(CACHE_DIR, exist_ok=True)

    wild_cache = get_cached_embedding(seq, cache_dir=CACHE_DIR)
    mut_cache = get_cached_embedding(mut_seq, cache_dir=CACHE_DIR)

    if os.path.exists(wild_cache):
        wild_emb = torch.load(wild_cache, map_location="cpu")
    else:
        wild_emb = embed_sequence(seq, esm2_model, batch_converter, device)
        torch.save(wild_emb, wild_cache)

    if os.path.exists(mut_cache):
        mut_emb = torch.load(mut_cache, map_location="cpu")
    else:
        mut_emb = embed_sequence(mut_seq, esm2_model, batch_converter, device)
        torch.save(mut_emb, mut_cache)

    # Drop special tokens if present: [CLS] ... [EOS]
    if wild_emb.shape[0] == mut_emb.shape[0] and wild_emb.shape[0] > len(seq):
        wild_emb = wild_emb[1:-1]
        mut_emb = mut_emb[1:-1]

    pos0 = int(mutation[1:-1]) - 1  # 0-based
    delta = mut_emb - wild_emb      # [L, D]

    i0 = max(0, pos0 - WINDOW)
    i1 = min(delta.shape[0], pos0 + WINDOW + 1)
    x = delta[i0:i1]

    target_len = 2 * WINDOW + 1
    if x.shape[0] < target_len:
        pad = torch.zeros(target_len - x.shape[0], x.shape[1], dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)

    with torch.no_grad():
        score = model(x.unsqueeze(0).to(device)).item()

    st.subheader("Mutation effect prediction")
    st.metric("Predicted effect score", f"{score:.4f}")
    st.caption(f"Cached embeddings: {os.path.basename(wild_cache)}, {os.path.basename(mut_cache)}")



