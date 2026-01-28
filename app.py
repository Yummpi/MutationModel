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

# MUST be before any other Streamlit UI call
st.set_page_config(page_title="MutationModel", layout="wide")

WEIGHTS = "models/epoch_14.pt"
WINDOW = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Utilities
# -----------------------
def unwrap_emb(obj):
    # Accept tensor OR dict {"emb": tensor}
    if isinstance(obj, dict):
        if "emb" in obj:
            return obj["emb"]
        # fall back: first tensor-like value
        for v in obj.values():
            if torch.is_tensor(v):
                return v
        raise RuntimeError("Loaded embedding dict has no tensor payload.")
    if torch.is_tensor(obj):
        return obj
    raise RuntimeError(f"Loaded embedding is not a tensor or dict: {type(obj)}")

def normalize_to_LD(emb, seq_len: int):
    """
    Convert ESM2 embedding output to [L, D] aligned to the input sequence.
    Handles:
      - [L, D] already
      - [L+2, D] with special tokens (drop first/last)
      - Any mismatch -> hard error (better than silently wrong scores)
    """
    emb = unwrap_emb(emb)

    if emb.ndim != 2:
        raise RuntimeError(f"Embedding must be 2D [T, D], got shape {tuple(emb.shape)}")

    T, D = emb.shape
    if T == seq_len:
        return emb
    if T == seq_len + 2:
        return emb[1:-1]
    raise RuntimeError(f"Embedding length mismatch: got T={T}, expected {seq_len} or {seq_len+2}")

def fixed_window_delta(wild_LD: torch.Tensor, mut_LD: torch.Tensor, pos0: int, window: int):
    delta = mut_LD - wild_LD  # [L, D]
    i0 = max(0, pos0 - window)
    i1 = min(delta.shape[0], pos0 + window + 1)
    x = delta[i0:i1]
    target_len = 2 * window + 1
    if x.shape[0] < target_len:
        pad = torch.zeros(target_len - x.shape[0], x.shape[1], dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    return x  # [target_len, D]

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
    return r.content.decode("utf-8")

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

# -----------------------
# Load trained mutation model
# -----------------------
if not os.path.exists(WEIGHTS):
    st.error(f"Missing weights file: {WEIGHTS}")
    st.stop()

try:
    model = MutationEffectTransformer(embed_dim=1280).to(DEVICE)
    state = torch.load(WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
except Exception as e:
    st.error("Failed to load model weights.")
    st.exception(e)
    st.stop()

# -----------------------
# Cache ESM2 model
# -----------------------
@st.cache_resource
def get_esm2(device_str: str):
    dev = torch.device(device_str)
    return load_esm2(dev)

device_str = "cuda" if torch.cuda.is_available() else "cpu"
try:
    esm2_model, batch_converter = get_esm2(device_str)
except Exception as e:
    st.error("Failed to load ESM2 (embedding model).")
    st.exception(e)
    st.stop()

# -----------------------
# Sidebar UI
# -----------------------
st.sidebar.title("Protein Tools")

DEFAULT_SEQ = (
    "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
)
txt = st.sidebar.text_area("Input sequence", DEFAULT_SEQ, height=220)
mutation = st.sidebar.text_input("Mutation (e.g. A123V)", "A50V")

c1, c2 = st.sidebar.columns(2)
do_fold = c1.button("Predict structure")
do_score = c2.button("Score mutation")

# -----------------------
# Validate sequence once
# -----------------------
seq = validate_sequence(txt)
if seq is None:
    st.error("Sequence must be amino-acid letters only (ACDEFGHIKLMNPQRSTVWY).")
    st.stop()

# -----------------------
# Structure prediction
# -----------------------
if do_fold:
    try:
        pdb_string = fold_sequence(seq)

        # Only write file after successful fold
        with open("predicted.pdb", "w", encoding="utf-8") as f:
            f.write(pdb_string)

        st.subheader("Visualization of predicted protein structure")
        render_mol(pdb_string)

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
# Mutation scoring
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

        # Path or string support
        wild_cache = str(wild_cache)
        mut_cache = str(mut_cache)

        if os.path.exists(wild_cache):
            wild_raw = torch.load(wild_cache, map_location="cpu")
        else:
            wild_raw = embed_sequence(seq, esm2_model, batch_converter, DEVICE)
            torch.save(wild_raw, wild_cache)

        if os.path.exists(mut_cache):
            mut_raw = torch.load(mut_cache, map_location="cpu")
        else:
            mut_raw = embed_sequence(mut_seq, esm2_model, batch_converter, DEVICE)
            torch.save(mut_raw, mut_cache)

        wild_LD = normalize_to_LD(wild_raw, len(seq))
        mut_LD = normalize_to_LD(mut_raw, len(seq))

        x = fixed_window_delta(wild_LD, mut_LD, pos0, WINDOW)

        with torch.no_grad():
            score = model(x.unsqueeze(0).to(DEVICE)).item()

        st.subheader("Mutation effect prediction")
        st.metric("Predicted effect score", f"{score:.4f}")
        st.caption(f"Embedding cache: {os.path.basename(wild_cache)} | {os.path.basename(mut_cache)}")

    except Exception as e:
        st.error("Mutation scoring failed.")
        st.exception(e)


