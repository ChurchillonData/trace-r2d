"""
Download the SBERT model weights needed for semantic embeddings.

Run this once before python -m recsys.run.  The model weights (~90 MB) are
only needed while computing item and user embeddings.  semantic.build()
deletes the weights file automatically once embeddings are cached to disk.

Usage:
    python setup_sbert.py
"""

from __future__ import annotations

import os
import ssl

# Allow outgoing network requests for this download only.
os.environ.pop("HF_HUB_OFFLINE", None)

# Work around SSL certificate check failures on this machine.
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001

from sentence_transformers import SentenceTransformer  # noqa: E402

MODEL_HF_ID = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_DIR = "models/all-MiniLM-L6-v2"

print(f"Downloading {MODEL_HF_ID} (~90 MB) …")
model = SentenceTransformer(MODEL_HF_ID)
model.save(LOCAL_DIR)
print(f"Saved to {LOCAL_DIR}")
print("You can now run: python -m recsys.run --config configs/default.yaml")
