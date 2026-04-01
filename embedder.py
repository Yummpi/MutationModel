import hashlib
from pathlib import Path

import torch
import esm


def load_esm2(device):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


@torch.inference_mode()
def embed_sequence(sequence: str, model, batch_converter, device):
    data = [("seq", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[33], return_contacts=False)
    reps = out["representations"][33][0].detach().cpu()   # (L+2, 1280)
    return reps


def get_cached_embedding(sequence: str, cache_dir="data/cache"):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(sequence.encode("utf-8")).hexdigest()
    return str(Path(cache_dir) / f"{key}.pt")


def validate_sequence(seq: str):
    seq = seq.strip().upper()
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    if not seq or any(c not in allowed for c in seq):
        return None
    return seq
