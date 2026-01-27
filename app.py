import os
import re
import urllib.request

import streamlit as st
from stmol import showmol
import py3Dmol
import requests
import biotite.structure.io as bsio
import torch

from mutation_model import MutationEffectTransformer
from embedder import load_esm2, embed_sequence, get_cached_embedding, validate_sequence

WEIGHTS = "models/epoch_14.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MutationEffectTransformer(embed_dim=1024).to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()

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
        
try:
    struct = bsio.load_structure('predicted.pdb', extra_fields=["b_factor"])
    b_value = float(struct.b_factor.mean())
except:
    b_value = None

    # Display protein structure
    st.subheader('Visualization of predicted protein structure')
    render_mol(pdb_string)

    # plDDT value is stored in the B-factor field
    st.subheader('plDDT')
    st.write('plDDT is a per-residue estimate of the confidence in prediction on a scale from 0-100.')
    st.info(f'plDDT: {b_value}')
=======
import os
import urllib.request

EPOCHS = list(range(30))
USE_EPOCH = 14

# Example: direct-download base URL where epoch_0.pt ... epoch_29.pt live
BASE_URL = "PUT_BASE_URL_HERE"

os.makedirs("models", exist_ok=True)

for e in EPOCHS:
    path = f"models/epoch_{e}.pt"
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, WEIGHTS)
    return True

ensure_weight()
    
ensure_models()

WEIGHTS = f"models/epoch_{USE_EPOCH}.pt"

# -----------------------
# Load trained mutation model (epoch 14 stored in repo)
# -----------------------
WEIGHTS = "models/epoch_14.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(WEIGHTS):
    st.error(f"Missing weights file: {WEIGHTS}")
    st.stop()

model = MutationEffectTransformer(embed_dim=1280).to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()


# -----------------------
# Cache ESM2 for embeddings (used for on-the-fly scoring)
# -----------------------
@st.cache_resource
def get_esm2(device_str: str):
    dev = torch.device(device_str)
    return load_esm2(dev)

device_str = "cuda" if torch.cuda.is_available() else "cpu"
esm2_model, batch_converter = get_esm2(device_str)


# -----------------------
# UI: ESMFold
# -----------------------
st.sidebar.title("ESMFold")
st.sidebar.write(
    "[*ESMFold*](https://esmatlas.com/about) folds a protein sequence into a 3D structure "
    "using the ESM Atlas API."
)

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

DEFAULT_SEQ = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
txt = st.sidebar.text_area("Input sequence", DEFAULT_SEQ, height=275)

predict_btn = st.sidebar.button("Predict structure")
pdb_string = None

if predict_btn:
    seq_for_fold = validate_sequence(txt)
    if seq_for_fold is None:
        st.error("Sequence must be amino-acid letters only (ACDEFGHIKLMNPQRSTVWY).")
        st.stop()
    pdb_string = fold_sequence(seq_for_fold)

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
# Mutation scoring (on-the-fly embeddings + caching)
# -----------------------
st.sidebar.markdown("---")
mutation = st.sidebar.text_input("Mutation (e.g. A123V)", "A50V")
score_btn = st.sidebar.button("Score mutation")

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

if score_btn:
    seq = validate_sequence(txt)
    if seq is None:
        st.error("Sequence must be amino-acid letters only (ACDEFGHIKLMNPQRSTVWY).")
        st.stop()

    mut_seq = apply_mutation(seq, mutation)
    if mut_seq is None:
        st.error("Invalid mutation for this sequence (format/position/wild-type mismatch).")
        st.stop()

    # Cache paths (one file per sequence)
    wild_cache = get_cached_embedding(seq, cache_dir="data/cache")
    mut_cache = get_cached_embedding(mut_seq, cache_dir="data/cache")

    # Load or compute embeddings
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

    # ESM2 embeddings include special tokens -> drop [CLS]/[EOS]
    wild_emb = wild_emb[1:-1]  # [L, D]
    mut_emb = mut_emb[1:-1]    # [L, D]

    pos0 = int(mutation[1:-1]) - 1  # 0-based index
    delta = mut_emb - wild_emb

    w = 8
    i0 = max(0, pos0 - w)
    i1 = min(delta.shape[0], pos0 + w + 1)
    x = delta[i0:i1]

    target_len = 2 * w + 1
    if x.shape[0] < target_len:
        pad = torch.zeros(target_len - x.shape[0], x.shape[1], dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)

    with torch.no_grad():
        score = model(x.unsqueeze(0).to(device)).item()

    st.subheader("Mutation effect prediction")
    st.metric("Predicted effect score", f"{score:.4f}")
    st.caption(f"Cached embeddings: {os.path.basename(wild_cache)}, {os.path.basename(mut_cache)}")



