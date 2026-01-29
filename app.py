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
from weights import ensure_weights

from pathlib import Path
import traceback

st.set_page_config(page_title="MutationModel", layout="wide")

def crashbox(e: Exception):
    st.error(str(e))
    st.code(traceback.format_exc())
    raise e

st.sidebar.write("WEIGHTS_URL set:", bool(os.getenv("WEIGHTS_URL")))
p = Path("models/epoch_14.pt")
st.sidebar.write("weights exists:", p.exists())
if p.exists():
    st.sidebar.write("weights bytes:", p.stat().st_size)

EMBED_DIM = 1280

@st.cache_resource(show_spinner=False)
def load_model():
    ckpt = ensure_weights()
    model = MutationEffectTransformer(embed_dim=EMBED_DIM)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    crashbox(e)

# -----------------------
# Config
# -----------------------

DEFAULT_SEQ = (
    "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
)

# -----------------------
# ESM2 cache (for embeddings used in mutation scoring)
# -----------------------
@st.cache_resource
def get_esm2(device_str: str):
    dev = torch.device(device_str)
    return load_esm2(dev)

if score_btn:
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
    with open("predicted.pdb", "w", encoding="utf-8") as f:
        f.write(pdb_string)
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
# Sidebar UI
# -----------------------
st.sidebar.title("ESMFold")
st.sidebar.write(
    "[*ESMFold*](https://esmatlas.com/about) folds a protein sequence into a 3D structure "
    "using the ESM Atlas API."
)

txt = st.sidebar.text_area("Input sequence", DEFAULT_SEQ, height=275)

predict_btn = st.sidebar.button("Predict structure")
pdb_string = None

# -----------------------
# Step 3: predict -> store pdb_string
# -----------------------
if predict_btn:
    seq_for_fold = validate_sequence(txt)
    if seq_for_fold is None:
        st.error("Sequence must be amino-acid letters only (ACDEFGHIKLMNPQRSTVWY).")
        st.stop()
    pdb_string = fold_sequence(seq_for_fold)

# -----------------------
# Step 4: render only if pdb_string exists
# -----------------------
if pdb_string is not None:
    st.subheader("Visualization of predicted protein structure")
    render_mol(pdb_string)

    try:
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
    st.info("Enter protein sequence data and press Predict structure.")

# -----------------------
# Mutation scoring
# -----------------------
st.sidebar.markdown("---")
mutation = st.sidebar.text_input("Mutation (e.g. A123V)", "A50V")
score_btn = st.sidebar.button("Score mutation")

if score_btn:
    seq = validate_sequence(txt)
    if seq is None:
        st.error("Sequence must be amino-acid letters only (ACDEFGHIKLMNPQRSTVWY).")
        st.stop()

    mut_seq = apply_mutation(seq, mutation)
    if mut_seq is None:
        st.error("Invalid mutation for this sequence (format/position/wild-type mismatch).")
        st.stop()

    os.makedirs("data/cache", exist_ok=True)

    wild_cache = get_cached_embedding(seq, cache_dir="data/cache")
    mut_cache = get_cached_embedding(mut_seq, cache_dir="data/cache")

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

    # drop [CLS]/[EOS]
    wild_emb = wild_emb[1:-1]
    mut_emb = mut_emb[1:-1]

    pos0 = int(mutation[1:-1]) - 1
    delta = mut_emb - wild_emb

    w = 8
    i0 = max(0, pos0 - w)
    i1 = min(delta.shape[0], pos0 + w + 1)
    x = delta[i0:i1]

    target_len = 2 * w + 1
    if x.shape[0] < target_len:
        pad = torch.zeros(target_len - x.shape[0], x.shape[1], dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)

    x = x.float().to(device)
    with torch.no_grad():
        score = model(x.unsqueeze(0).to(device)).item()

    st.subheader("Mutation effect prediction")
    st.metric("Predicted effect score", f"{score:.4f}")
    st.caption(
        f"Cached embeddings: {os.path.basename(wild_cache)}, {os.path.basename(mut_cache)}"
    )
