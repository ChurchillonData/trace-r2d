"""
Data layer for the two-stage recommender.

This module owns the download, cleaning, and the one coordinated
train/test split.  No other module may define what is train or test.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from recsys.eda import run_eda


def _make_src_iter(local, src: str, cat: str):
    import gzip
    if local is not None:
        opener = gzip.open if str(local).endswith(".gz") else open
        return (json.loads(l) for l in opener(local, "rt", encoding="utf-8") if l.strip())
    import datasets as hf
    for ext, fmt in [(".jsonl.gz", "json"), (".parquet", "parquet"), (".jsonl", "json")]:
        try:
            return hf.load_dataset(fmt, data_files={"full": f"hf://datasets/{src}/raw_review_{cat}/*{ext}"}, split="full", streaming=True)
        except Exception as e:
            print(f"  [{ext}] failed: {e}")
    return iter([])


def download_data(cfg: dict, raw_path: Path) -> None:
    """Stream sample_size rows from a local file or HuggingFace and write parquet."""
    import ssl
    import requests
    import urllib3
    import pyarrow as pa, pyarrow.parquet as pq
    # Machine has SSL issues with HuggingFace; bypass for both stdlib and requests/urllib3.
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["CURL_CA_BUNDLE"] = ""
    urllib3.disable_warnings()
    _orig = requests.Session.send
    def _no_verify(self, *a, **kw): kw["verify"] = False; return _orig(self, *a, **kw)
    requests.Session.send = _no_verify  # type: ignore[method-assign]
    n, cat, src = cfg["data"]["sample_size"], cfg["data"]["category"], cfg["data"]["source"]
    print(f"  source: {src} / raw_review_{cat}  sample_size: {n:,}")
    local = next((p for p in [Path(f"data/raw/{cat}.jsonl.gz"), Path(f"data/raw/{cat}.jsonl")] if p.exists()), None)
    src_iter = _make_src_iter(local, src, cat)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    S = pa.schema([("user", pa.string()), ("item", pa.string()), ("rating", pa.float64()), ("review", pa.string())])
    mk = lambda r: {"user": str(r.get("user_id", r.get("reviewerID",""))), "item": str(r.get("parent_asin", r.get("asin",""))), "rating": float(r.get("rating", r.get("overall",0.0))), "review": str(r.get("text", r.get("reviewText","")) or r.get("title",""))}
    w, buf, total = pq.ParquetWriter(str(raw_path), S), [], 0
    for rec in tqdm(src_iter, total=n, desc="  rows"):
        buf.append(mk(rec))
        if len(buf) >= 100_000 or total + len(buf) >= n:
            w.write_table(pa.Table.from_pylist(buf, schema=S)); total += len(buf); buf = []
        if total >= n: break
    if buf: w.write_table(pa.Table.from_pylist(buf, schema=S)); total += len(buf)
    w.close()
    if total == 0: raise RuntimeError(f"No rows loaded.  Place {cat} reviews at data/raw/{cat}.jsonl.gz")
    print(f"  saved {total:,} rows to {raw_path}")


def _clean_text(text: str) -> str:
    """Lowercase and keep only alphanumeric chars and spaces."""
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()


def clean_data(raw_path: Path) -> pd.DataFrame:
    """Load raw parquet, remove bad rows, and add review_clean column."""
    df = pd.read_parquet(raw_path)
    df = df[(df["user"].str.len() > 0) & (df["item"].str.len() > 0)]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating", "review"]).copy()
    df = df.drop_duplicates(subset=["user", "item"])
    df["review_clean"] = df["review"].apply(_clean_text)
    return df.reset_index(drop=True)


def encode_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Map string user/item ids to contiguous integers."""
    u_map = {u: i for i, u in enumerate(sorted(df["user"].unique()))}
    i_map = {it: i for i, it in enumerate(sorted(df["item"].unique()))}
    out = df.copy()
    out["user"] = df["user"].map(u_map).astype(int)
    out["item"] = df["item"].map(i_map).astype(int)
    return out


def apply_kcore(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iteratively drop users and items with fewer than k interactions."""
    while True:
        u_counts = df["user"].value_counts()
        i_counts = df["item"].value_counts()
        valid_u = set(u_counts[u_counts >= k].index)
        valid_i = set(i_counts[i_counts >= k].index)
        mask = df["user"].isin(valid_u) & df["item"].isin(valid_i)
        if mask.all():
            return df.reset_index(drop=True)
        df = df[mask]


def identify_cold_start(df: pd.DataFrame, min_n: int) -> tuple[set, set]:
    """Return user ids and item ids with fewer than min_n interactions."""
    u_counts = df["user"].value_counts()
    i_counts = df["item"].value_counts()
    return set(u_counts[u_counts < min_n].index), set(i_counts[i_counts < min_n].index)


def per_user_split(
    warm_df: pd.DataFrame, test_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """Random per-user split. Returns (train_indices, test_indices)."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for _, grp in warm_df.groupby("user"):
        idx = np.array(grp.index.tolist())
        if len(idx) < 2:
            train_idx.extend(idx.tolist())
            continue
        n_test = max(1, round(len(idx) * test_frac))
        perm = rng.permutation(len(idx))
        test_idx.extend(idx[perm[:n_test]].tolist())
        train_idx.extend(idx[perm[n_test:]].tolist())
    return train_idx, test_idx


def _item_lists(df: pd.DataFrame, indices: list[int]) -> dict[str, list[int]]:
    """Return {str(user): [item, ...]} for the given row indices."""
    sub = df.loc[indices]
    return {str(u): g["item"].tolist() for u, g in sub.groupby("user")}


def create_split(df: pd.DataFrame, cfg: dict) -> tuple[dict, dict]:
    """Produce the one coordinated train/test split and cold-start holdout."""
    min_n = cfg["data"]["min_interactions_warm"]
    test_frac = cfg["data"]["test_fraction"]
    seed = cfg["seed"]

    cold_u, cold_i = identify_cold_start(df, min_n)
    warm_mask = ~df["user"].isin(cold_u) & ~df["item"].isin(cold_i)
    warm_df = df[warm_mask]
    cold_df = df[~warm_mask]

    train_idx, test_idx = per_user_split(warm_df, test_frac, seed)

    split: dict = {
        "train_idx": [int(i) for i in train_idx],
        "test_idx": [int(i) for i in test_idx],
        "user_train_items": _item_lists(df, train_idx),
        "user_test_items": _item_lists(df, test_idx),
    }
    cold: dict = {
        "cold_users": [int(u) for u in sorted(cold_u)],
        "cold_items": [int(i) for i in sorted(cold_i)],
        "cold_rows": [int(i) for i in cold_df.index.tolist()],
    }
    return split, cold


def write_artifacts(
    df: pd.DataFrame, split: dict, cold: dict, paths: dict
) -> None:
    """Write clean.parquet, split.json, and cold_start.json to disk."""
    for key in ("clean", "split", "cold_start"):
        Path(paths[key]).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths["clean"], index=True)
    with open(paths["split"], "w") as fh:
        json.dump(split, fh)
    with open(paths["cold_start"], "w") as fh:
        json.dump(cold, fh)
    print(f"  clean: {paths['clean']}")
    print(f"  split: {paths['split']}")
    print(f"  cold:  {paths['cold_start']}")



def build(cfg: dict) -> None:
    """Run the full data pipeline and write all artifacts."""
    paths = cfg["paths"]
    p_clean, p_split, p_cold = Path(paths["clean"]), Path(paths["split"]), Path(paths["cold_start"])
    if p_clean.exists() and p_split.exists() and p_cold.exists():
        print("[data] artifacts found, skipping.")
        return
    raw_path = p_clean.parent / "raw.parquet"
    if not raw_path.exists():
        print("[data] downloading …")
        download_data(cfg, raw_path)
    print("[data] cleaning …")
    df = clean_data(raw_path)
    k = cfg["data"].get("k_core", 0)
    if k > 1:
        before = len(df)
        df = apply_kcore(df, k)
        print(f"[data] k_core={k}: {before:,} → {len(df):,} rows")
    df = encode_ids(df)
    print("[data] splitting …")
    split, cold = create_split(df, cfg)
    write_artifacts(df, split, cold, paths)
    run_eda(df, Path(paths.get("reports_dir", "reports")) / "figures")
    print(
        f"[data] done.  train={len(split['train_idx']):,}  test={len(split['test_idx']):,}"
        f"  cold_users={len(cold['cold_users']):,}  cold_items={len(cold['cold_items']):,}"
    )
