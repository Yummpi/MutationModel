import os, hashlib
from pathlib import Path
import torch
import esm  # provided by fair-esm
if not hasattr(esm, "pretrained"):
    raise RuntimeError(
        "Wrong 'esm' installed. Remove pip package 'esm' and install 'fair-esm==2.0.0'."
    )

AA20 = set("ACDEFGHIKLMNPQRSTVWY")

def _hash_seq(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:16]

@torch.inference_mode()
def load_esm2(device):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    bc = alphabet.get_batch_converter()
    return model, bc

@torch.inference_mode()
def embed_sequence(sequence: str, model, batch_converter, device):
    data = [("seq", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[33], return_contacts=False)
    reps = out["representations"][33][0].detach().cpu()  # (L+2, 1280)
    return reps

def get_cached_embedding(seq: str, cache_dir="data/cache") -> str:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = _hash_seq(seq)
    return str(Path(cache_dir) / f"{key}.pt")

def validate_sequence(seq: str):
    s = seq.strip().upper()
    if not s or any(c not in AA20 for c in s):
        return None
    if len(s) > 1200:
        return None
    return s
