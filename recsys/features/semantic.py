"""
Semantic features: Sentence-BERT item/user embeddings and cosine scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm


def _collect_item_texts(df: pd.DataFrame, train_idx: list[int]) -> dict[int, str]:
    """Return one representative review per item from training rows.

    Using first() rather than joining all reviews keeps the text dict ~37x
    smaller in RAM, which matters with a 5M-row dataset.
    """
    sub = df.loc[train_idx, ["item", "review"]]
    return sub.groupby("item")["review"].first().to_dict()


def _embed_texts(texts: list[str], cfg: dict) -> np.ndarray:
    """Encode texts with SBERT in small chunks to cap peak RAM."""
    import os
    from sentence_transformers import SentenceTransformer

    # Force offline so SentenceTransformer uses the local copy and skips
    # the HuggingFace version check, which fails with SSL errors on this machine.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model_name = cfg["features"]["semantic"]["model_name"]
    batch_size = cfg["features"]["semantic"]["batch_size"]
    chunk_size = cfg["features"]["semantic"].get("encoding_chunk_size", 5000)
    model = SentenceTransformer(model_name)
    n_chunks = max(1, (len(texts) + chunk_size - 1) // chunk_size)
    print(f"  encoding {len(texts):,} texts ({n_chunks} chunk(s), batch={batch_size}) …")
    parts = []
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        vecs = model.encode(
            chunk, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        ).astype(np.float32)
        parts.append(vecs)
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def embed_items(
    df: pd.DataFrame, train_idx: list[int], cfg: dict, cache_path: Path
) -> np.ndarray:
    """Return item embedding matrix (n_items, dim), loading cache if available."""
    if cache_path.exists() and cfg["features"]["semantic"]["cache"]:
        print(f"  loading cached item embeddings from {cache_path}")
        return np.load(cache_path)
    n_items = int(df["item"].max()) + 1
    item_texts = _collect_item_texts(df, train_idx)
    items_sorted = sorted(item_texts.keys())
    vecs = _embed_texts([item_texts[i] for i in items_sorted], cfg)
    embeddings = np.zeros((n_items, vecs.shape[1]), dtype=np.float32)
    for item_id, vec in zip(items_sorted, vecs):
        embeddings[item_id] = vec
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"  saved item embeddings to {cache_path}")
    return embeddings


def build_user_embeddings(
    df: pd.DataFrame,
    row_idx: list[int],
    item_embs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Build user embeddings by averaging highly-rated item embeddings."""
    n_users = int(df["user"].max()) + 1
    n_dim = item_embs.shape[1]
    user_embs = np.zeros((n_users, n_dim), dtype=np.float32)
    sub = df.loc[row_idx]
    for user, grp in sub[sub["rating"] >= threshold].groupby("user"):
        vecs = item_embs[grp["item"].values]
        valid = vecs[np.any(vecs != 0, axis=1)]
        if len(valid) > 0:
            user_embs[int(user)] = valid.mean(axis=0)
    return user_embs


def cosine_scores(
    user_embs: np.ndarray,
    item_embs: np.ndarray,
    df: pd.DataFrame,
    row_idx: list[int],
) -> pd.DataFrame:
    """Compute cosine similarity scores for (user, item) pairs in 50K-row batches."""
    sub = df.loc[row_idx, ["user", "item"]]
    users = sub["user"].values
    items = sub["item"].values
    out = np.empty(len(sub), dtype=np.float32)
    for i in range(0, len(sub), 50_000):
        u = user_embs[users[i:i + 50_000]]
        v = item_embs[items[i:i + 50_000]]
        u_n = np.linalg.norm(u, axis=1, keepdims=True)
        v_n = np.linalg.norm(v, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            u_s = np.where(u_n > 0, u / u_n, 0.0)
            v_s = np.where(v_n > 0, v / v_n, 0.0)
        out[i:i + 50_000] = (u_s * v_s).sum(axis=1)
    return sub.assign(semantic_score=out)


def oof_semantic_scores(
    df: pd.DataFrame,
    train_idx: list[int],
    item_embs: np.ndarray,
    cfg: dict,
) -> pd.DataFrame:
    """OOF semantic scores: user embeddings exclude validation-fold items."""
    n_folds = cfg["folds"]["n_folds"]
    seed = cfg["seed"]
    # Use retrieval threshold (liked items) for user profiles, not evaluation threshold.
    threshold = float(cfg["retrieval"].get("rating_threshold", 4))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    arr = np.array(train_idx)
    parts: list[pd.DataFrame] = []
    for tr_pos, va_pos in tqdm(kf.split(arr), total=n_folds, desc="  semantic folds"):
        user_embs = build_user_embeddings(df, arr[tr_pos].tolist(), item_embs, threshold)
        parts.append(cosine_scores(user_embs, item_embs, df, arr[va_pos].tolist()))
    return pd.concat(parts, ignore_index=True)


def _delete_model_weights(model_name: str) -> None:
    """Remove weight files after embeddings are cached to free disk space."""
    model_dir = Path(model_name)
    if not model_dir.is_dir():
        return
    freed = 0
    for fname in ("pytorch_model.bin", "model.safetensors", "tf_model.h5"):
        p = model_dir / fname
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
    if freed:
        print(f"  [cleanup] deleted model weights from {model_dir} ({freed / 1_000_000:.0f} MB freed)")


def build(cfg: dict) -> None:
    """Write item embeddings, user embeddings, and semantic score files."""
    import gc

    paths = cfg["paths"]
    feat_dir = Path(paths["features_dir"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    oof_path = feat_dir / "semantic_oof.parquet"
    test_path = feat_dir / "semantic_test.parquet"
    item_emb_path = feat_dir / "item_embeddings.npy"
    user_emb_path = feat_dir / "user_embeddings.npy"
    if oof_path.exists() and test_path.exists():
        print("[semantic] artifacts found, skipping.")
        return
    split = json.load(open(paths["split"]))
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    # Use retrieval threshold so user profiles include all liked items (>=4), not just >=5.
    threshold = float(cfg["retrieval"].get("rating_threshold", 4))

    # Phase 1: embeddings — load only item + review to keep RAM low.
    print("[semantic] computing item embeddings …")
    df_text = pd.read_parquet(paths["clean"], columns=["item", "review"])
    item_embs = embed_items(df_text, train_idx, cfg, item_emb_path)
    del df_text
    gc.collect()

    # Phase 2: scoring — no review column needed.
    df = pd.read_parquet(paths["clean"], columns=["user", "item", "rating"])
    print("[semantic] building user embeddings (full training partition) …")
    user_embs = build_user_embeddings(df, train_idx, item_embs, threshold)
    np.save(user_emb_path, user_embs)
    print("[semantic] generating OOF semantic scores …")
    oof = oof_semantic_scores(df, train_idx, item_embs, cfg)
    oof.to_parquet(oof_path, index=False)
    print("[semantic] scoring test partition …")
    test_df = cosine_scores(user_embs, item_embs, df, test_idx)
    test_df.to_parquet(test_path, index=False)
    print(f"[semantic] done.  oof={len(oof):,}  test={len(test_df):,}")
