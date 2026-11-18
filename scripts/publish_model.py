"""Publish the SmallCNN weights and the model card to the Hugging Face Hub.

    python scripts/publish_model.py            # dry run: lists what would be uploaded where
    python scripts/publish_model.py --upload   # creates the public model repo (if needed) and uploads

Uses the existing `hf auth login`; no token is read, written or printed here.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

FILES = {                                    # path in the HF repo -> local source
    C.HF_MODEL_FILE: C.MODELS_DIR / C.BOARD_MODEL / "best.pt",
    "README.md": ROOT / "deploy" / "hf_model" / "README.md",
    "statsbomb_logo.png": C.LOGO_PATH,
    "calibration_test.png": C.PLOTS_DIR / "calibration_test.png",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="actually create the repo and upload")
    args = ap.parse_args()

    missing = [str(p) for p in FILES.values() if not p.exists()]
    if missing:
        print("missing:", *missing, sep="\n  ")
        return 1
    print(f"target: https://huggingface.co/{C.HF_MODEL_REPO} (public model repo)")
    for dst, src in FILES.items():
        print(f"  {dst:24s} <- {src.relative_to(ROOT)}  ({src.stat().st_size / 1024:,.0f} KB, sha256 {sha256(src)})")
    if not args.upload:
        print("dry run; add --upload to publish")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(C.HF_MODEL_REPO, repo_type="model", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for dst, src in FILES.items():
            shutil.copy2(src, Path(tmp) / dst)
        info = api.upload_folder(folder_path=tmp, repo_id=C.HF_MODEL_REPO, repo_type="model",
                                 commit_message="Upload SmallCNN (seed 42) weights and model card")
    print("uploaded:", info.commit_url if hasattr(info, "commit_url") else info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
