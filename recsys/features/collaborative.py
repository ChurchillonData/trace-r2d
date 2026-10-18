"""
Collaborative features: out-of-fold SVD scores and latent factor vectors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from surprise import Dataset, Reader, SVD
from tqdm import tqdm


def _fit_svd(df: pd.DataFrame, row_idx: list[int], n_factors: int, seed: int) -> SVD:
    """Fit SVD on specified rows and return the trained algorithm."""
    reader = Reader(rating_scale=(1, 5))
    sub = df.loc[row_idx, ["user", "item", "rating"]].copy()
    sub["user"] = sub["user"].astype(str)
    sub["item"] = sub["item"].astype(str)
    data = Dataset.load_from_df(sub, reader)
    algo = SVD(n_factors=n_factors, random_state=seed, verbose=False)
    algo.fit(data.build_full_trainset())
    return algo


def _predict_pairs(algo: SVD, df: pd.DataFrame, row_idx: list[int]) -> pd.DataFrame:
    """Predict (user, item) pairs and return DataFrame with svd_score."""
    sub = df.loc[row_idx, ["user", "item"]]
    scores = [
        algo.predict(str(r.user), str(r.item)).est
        for r in sub.itertuples(index=False)
    ]
    return sub.assign(svd_score=scores)


def oof_svd_scores(
    df: pd.DataFrame, train_idx: list[int], cfg: dict
) -> pd.DataFrame:
    """Generate out-of-fold SVD scores for all training rows."""
    n_folds = cfg["folds"]["n_folds"]
    n_factors = cfg["features"]["svd"]["n_factors"]
    seed = cfg["seed"]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    arr = np.array(train_idx)
    parts: list[pd.DataFrame] = []
    for tr_pos, va_pos in tqdm(kf.split(arr), total=n_folds, desc="  SVD folds"):
        algo = _fit_svd(df, arr[tr_pos].tolist(), n_factors, seed)
        parts.append(_predict_pairs(algo, df, arr[va_pos].tolist()))
    return pd.concat(parts, ignore_index=True)


def test_svd_scores(
    df: pd.DataFrame, train_idx: list[int], test_idx: list[int], cfg: dict
) -> tuple[pd.DataFrame, SVD]:
    """Train SVD on the full training partition. Return (scores_df, model)."""
    n_factors = cfg["features"]["svd"]["n_factors"]
    seed = cfg["seed"]
    print("  fitting SVD on full training partition …")
    algo = _fit_svd(df, train_idx, n_factors, seed)
    return _predict_pairs(algo, df, test_idx), algo


def save_factors(algo: SVD, n_users: int, n_items: int, out_path: Path) -> None:
    """Save user and item latent factor matrices to an npz file."""
    n_f = algo.pu.shape[1]
    uf = np.zeros((n_users, n_f), dtype=np.float32)
    itf = np.zeros((n_items, n_f), dtype=np.float32)
    ts = algo.trainset
    for inner in ts.all_users():
        uf[int(ts.to_raw_uid(inner))] = algo.pu[inner]
    for inner in ts.all_items():
        itf[int(ts.to_raw_iid(inner))] = algo.qi[inner]
    np.savez(out_path, user_factors=uf, item_factors=itf)
    print(f"  factors saved to {out_path}")


def build(cfg: dict) -> None:
    """Run OOF SVD generation and write artifacts."""
    paths = cfg["paths"]
    feat_dir = Path(paths["features_dir"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    oof_path = feat_dir / "svd_oof.parquet"
    test_path = feat_dir / "svd_test.parquet"
    factors_path = feat_dir / "svd_factors.npz"
    if oof_path.exists() and test_path.exists() and factors_path.exists():
        print("[collab] artifacts found, skipping.")
        return
    df = pd.read_parquet(paths["clean"])
    split = json.load(open(paths["split"]))
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    n_users = int(df["user"].max()) + 1
    n_items = int(df["item"].max()) + 1

    print("[collab] generating OOF SVD scores …")
    oof = oof_svd_scores(df, train_idx, cfg)
    oof.to_parquet(oof_path, index=False)

    print("[collab] scoring test partition …")
    test_df, full_algo = test_svd_scores(df, train_idx, test_idx, cfg)
    test_df.to_parquet(test_path, index=False)
    save_factors(full_algo, n_users, n_items, factors_path)
    print(f"[collab] done.  oof={len(oof):,}  test={len(test_df):,}")
