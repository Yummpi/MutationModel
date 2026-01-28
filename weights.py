import os
from pathlib import Path
import requests

MODEL_PATH = Path("models/epoch_14.pt")

def _looks_like_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        if path.stat().st_size < 1024 * 1024:
            head = path.read_text(errors="ignore")[:200]
            return ("git-lfs" in head) and ("oid sha256" in head)
        return False
    except Exception:
        return False

def ensure_weights() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and not _looks_like_lfs_pointer(MODEL_PATH) and MODEL_PATH.stat().st_size >= 1024 * 1024:
        return MODEL_PATH

    url = os.getenv("WEIGHTS_URL", "").strip()
    if not url:
        raise RuntimeError("Missing WEIGHTS_URL. Add it in Streamlit Cloud Secrets.")

    tmp = MODEL_PATH.with_suffix(".download")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp.replace(MODEL_PATH)

    if _looks_like_lfs_pointer(MODEL_PATH) or MODEL_PATH.stat().st_size < 1024 * 1024:
        raise RuntimeError("Downloaded weights look invalid. Check WEIGHTS_URL target.")

    return MODEL_PATH
