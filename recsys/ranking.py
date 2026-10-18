"""
Stage 2 ranking: LightGBM ranker trained on OOF features.

Trains in both lambdarank and pointwise modes, scores candidates
from Stage 1, sorts per user, and writes the final ranked lists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLS: list[str] = [
    "svd_score",
    "semantic_score",
    "itemitem_score",
    "item_popularity",
    "user_activity",
]


def _svd_scores(
    candidates: pd.DataFrame, user_factors: np.ndarray, item_factors: np.ndarray
) -> np.ndarray:
    """Dot product of user and item SVD factors, batched to avoid OOM."""
    users = candidates["user"].values
    items = candidates["item"].values
    out = np.empty(len(candidates), dtype=np.float32)
    for start in range(0, len(candidates), 50_000):
        sl = slice(start, start + 50_000)
        out[sl] = (user_factors[users[sl]] * item_factors[items[sl]]).sum(axis=1)
    return out


def _semantic_scores(candidates: pd.DataFrame, user_embs: np.ndarray, item_embs: np.ndarray) -> np.ndarray:
    """Cosine similarity between user and item SBERT embeddings, batched to avoid OOM."""
    users = candidates["user"].values
    items = candidates["item"].values
    out = np.empty(len(candidates), dtype=np.float32)
    for i in range(0, len(candidates), 10000):
        u = user_embs[users[i:i + 10000]]
        it = item_embs[items[i:i + 10000]]
        u_n = np.linalg.norm(u, axis=1, keepdims=True)
        i_n = np.linalg.norm(it, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            u_s = np.where(u_n > 0, u / u_n, 0.0)
            i_s = np.where(i_n > 0, it / i_n, 0.0)
        out[i:i + 10000] = (u_s * i_s).sum(axis=1)
    return out


def _itemitem_scores(
    candidates: pd.DataFrame,
    neigh_dict: dict[int, list[tuple[int, float]]],
    user_train_items: dict[int, list[int]],
) -> list[float]:
    """Average neighbour similarity for items in the user's training history."""
    scores = []
    for row in candidates.itertuples(index=False):
        history = set(user_train_items.get(int(row.user), []))
        neighs = neigh_dict.get(int(row.item), [])
        sims = [s for n, s in neighs if n in history]
        scores.append(float(np.mean(sims)) if sims else 0.0)
    return scores


def _compute_candidate_features(candidates: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Load on-disk artefacts and compute ranker features plus baseline scores."""
    paths = cfg["paths"]
    feat_dir = Path(paths["features_dir"])
    factors = np.load(feat_dir / "svd_factors.npz")
    user_embs = np.load(feat_dir / "user_embeddings.npy")
    item_embs = np.load(feat_dir / "item_embeddings.npy")
    neighbours = pd.read_parquet(feat_dir / "itemitem_neighbours.parquet")
    neigh_dict = {
        int(it): list(zip(g["neighbour"].tolist(), g["similarity"].tolist()))
        for it, g in neighbours.groupby("item")
    }
    tfidf_neighbours = pd.read_parquet(feat_dir / "tfidf_neighbours.parquet")
    tfidf_dict = {
        int(it): list(zip(g["neighbour"].tolist(), g["similarity"].tolist()))
        for it, g in tfidf_neighbours.groupby("item")
    }
    split = json.load(open(paths["split"]))
    user_train_items = {int(u): v for u, v in split["user_train_items"].items()}
    oof = pd.read_parquet(feat_dir / "oof_scores.parquet")
    item_pop_map = oof.groupby("item")["item_popularity"].first()
    user_act_map = oof.groupby("user")["user_activity"].first()
    out = candidates[["user", "item"]].copy()
    out["svd_score"] = _svd_scores(candidates, factors["user_factors"], factors["item_factors"])
    out["semantic_score"] = _semantic_scores(candidates, user_embs, item_embs)
    out["itemitem_score"] = _itemitem_scores(candidates, neigh_dict, user_train_items)
    # tfidf_score is the rejected baseline signal — not in FEATURE_COLS for the ranker.
    out["tfidf_score"] = _itemitem_scores(candidates, tfidf_dict, user_train_items)
    out["item_popularity"] = candidates["item"].map(item_pop_map).fillna(0).astype(int)
    out["user_activity"] = candidates["user"].map(user_act_map).fillna(0).astype(int)
    return out


def _make_groups(df: pd.DataFrame) -> np.ndarray:
    """Return group-size array aligned to df sorted by user."""
    return df.groupby("user", sort=False).size().values.astype(np.int32)


def _load_train_val(cfg: dict) -> tuple:
    """Load oof_scores and split users 80/20 for training and early stopping."""
    feat_dir = Path(cfg["paths"]["features_dir"])
    df = pd.read_parquet(feat_dir / "oof_scores.parquet")
    rng = np.random.default_rng(cfg["seed"])
    users = df["user"].unique()
    n_val = max(1, int(len(users) * 0.2))
    val_users = set(rng.choice(users, size=n_val, replace=False).tolist())
    tr = df[~df["user"].isin(val_users)].sort_values("user").reset_index(drop=True)
    va = df[df["user"].isin(val_users)].sort_values("user").reset_index(drop=True)
    X_tr = tr[FEATURE_COLS].values.astype(np.float32)
    X_va = va[FEATURE_COLS].values.astype(np.float32)
    return X_tr, tr["rating"].values, _make_groups(tr), X_va, va["rating"].values, _make_groups(va)


def train_lambdarank(
    X_tr: np.ndarray, y_tr: np.ndarray, g_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray, g_va: np.ndarray,
    cfg: dict,
):
    """Train LightGBM lambdarank on binary relevance labels."""
    import lightgbm as lgb
    threshold = float(cfg["evaluation"]["relevance_threshold"])
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": cfg["ranking"]["learning_rate"],
        "max_depth": cfg["ranking"]["max_depth"],
        "verbose": -1,
    }
    y_tr_bin = (y_tr >= threshold).astype(int)
    y_va_bin = (y_va >= threshold).astype(int)
    ds_tr = lgb.Dataset(X_tr, label=y_tr_bin, group=g_tr, feature_name=FEATURE_COLS)
    ds_va = lgb.Dataset(X_va, label=y_va_bin, group=g_va, reference=ds_tr)
    cbs = [lgb.early_stopping(cfg["ranking"]["early_stopping_rounds"]), lgb.log_evaluation(50)]
    return lgb.train(params, ds_tr, num_boost_round=1000, valid_sets=[ds_va], callbacks=cbs)


def train_pointwise(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    cfg: dict,
):
    """Train LightGBM regression ranker on raw ratings."""
    import lightgbm as lgb
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": cfg["ranking"]["learning_rate"],
        "max_depth": cfg["ranking"]["max_depth"],
        "verbose": -1,
    }
    ds_tr = lgb.Dataset(X_tr, label=y_tr.astype(float), feature_name=FEATURE_COLS)
    ds_va = lgb.Dataset(X_va, label=y_va.astype(float), reference=ds_tr)
    cbs = [lgb.early_stopping(cfg["ranking"]["early_stopping_rounds"]), lgb.log_evaluation(50)]
    return lgb.train(params, ds_tr, num_boost_round=1000, valid_sets=[ds_va], callbacks=cbs)


def save_feature_importance(model, out_path: Path) -> None:
    """Bar chart of feature gain importance saved to out_path."""
    import matplotlib.pyplot as plt
    importances = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, importances)
    ax.set_xlabel("Gain")
    ax.set_title("Ranker feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  feature importance saved to {out_path}")


def build(cfg: dict) -> None:
    """Train ranker, score candidates, and write ranked lists."""
    paths = cfg["paths"]
    rank_dir = Path(paths["ranking_dir"])
    rank_dir.mkdir(parents=True, exist_ok=True)
    model_path = rank_dir / "model.json"
    lists_path, cand_path = rank_dir / "ranked_lists.parquet", rank_dir / "candidate_scores.parquet"
    if model_path.exists() and lists_path.exists() and cand_path.exists():
        print("[ranking] artifacts found, skipping.")
        return
    print("[ranking] loading train/val data …")
    X_tr, y_tr, g_tr, X_va, y_va, g_va = _load_train_val(cfg)
    print(f"[ranking] train={len(y_tr):,}  val={len(y_va):,}")
    print("[ranking] training lambdarank model …")
    lr_model = train_lambdarank(X_tr, y_tr, g_tr, X_va, y_va, g_va, cfg)
    lr_model.save_model(str(model_path))
    print("[ranking] training pointwise model …")
    pw_model = train_pointwise(X_tr, y_tr, X_va, y_va, cfg)
    pw_model.save_model(str(rank_dir / "model_pointwise.json"))
    save_feature_importance(lr_model, rank_dir / "feature_importance.png")
    print("[ranking] scoring candidates …")
    candidates = pd.read_parquet(Path(paths["retrieval_dir"]) / "candidates.parquet")
    feats = _compute_candidate_features(candidates, cfg)
    feats = feats.assign(score=lr_model.predict(feats[FEATURE_COLS].values.astype(np.float32)))
    feats.to_parquet(cand_path, index=False)
    ranked = feats.sort_values(["user", "score"], ascending=[True, False]).groupby("user").head(cfg["ranking"]["top_n"])
    ranked.to_parquet(lists_path, index=False)
    print(f"[ranking] done.  {ranked['user'].nunique():,} users  {len(ranked):,} ranked rows")
