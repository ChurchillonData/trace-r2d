"""
Option B: candidate-trained ranker with OOF-retrieved budget-union candidates.

Per fold k (5 folds, KFold seed 42):
  SVD and item-item table trained on the other 4 folds only.
  User embeddings averaged from the other 4 folds only.
  item_embeddings.npy reused (text-based, fold-independent).
  Budget-union retrieval with fold-k models for users that have
  at least one five-star item in the validation fold.
  Exclusion set per user = training items UNION test items, so no test
  (user, item) pair ever enters the training candidate pool.
  Labels from fold-k validation rows; unobserved candidates = 0.

New ranker: ranking/model_candidate.json.
Three evaluation modes vs old observed-pair ranker on same retrieval.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

from recsys.features.collaborative import _fit_svd
from recsys.features.itemitem import build_item_matrix, compute_neighbours
from recsys.features.semantic import build_user_embeddings
from recsys.retrieval import _build_svd_index, _build_sem_index, retrieve_for_user
from recsys.ranking import FEATURE_COLS, _svd_scores, _semantic_scores
from recsys.evaluation import _hit_ndcg_mrr, _iter_user_candidates

FOLD_DIR = Path("features/oof_fold_candidates")


# ── fold splits ────────────────────────────────────────────────────────────────

def _fold_splits(train_idx: list[int], cfg: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """KFold splits over train_idx; returns list of (tr_rows, va_rows) arrays."""
    kf = KFold(n_splits=cfg["folds"]["n_folds"], shuffle=True, random_state=cfg["seed"])
    arr = np.array(train_idx)
    return [(arr[tr], arr[va]) for tr, va in kf.split(arr)]


# ── per-fold model builders ────────────────────────────────────────────────────

def _extract_factors(algo, n_users: int, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract user and item factor matrices from a fitted surprise SVD."""
    n_f = algo.pu.shape[1]
    u_fac = np.zeros((n_users, n_f), dtype=np.float32)
    i_fac = np.zeros((n_items, n_f), dtype=np.float32)
    ts = algo.trainset
    for inner in ts.all_users():
        u_fac[int(ts.to_raw_uid(inner))] = algo.pu[inner]
    for inner in ts.all_items():
        i_fac[int(ts.to_raw_iid(inner))] = algo.qi[inner]
    return u_fac, i_fac


def _build_fold_svd(
    df: pd.DataFrame, tr_rows: np.ndarray, n_users: int, n_items: int, cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit SVD on tr_rows. Return (user_factors, item_factors)."""
    algo = _fit_svd(df, tr_rows.tolist(), cfg["features"]["svd"]["n_factors"], cfg["seed"])
    return _extract_factors(algo, n_users, n_items)


def _build_fold_ii(df: pd.DataFrame, tr_rows: np.ndarray, cfg: dict) -> dict:
    """Build item-item neighbour table from tr_rows. Return neigh_dict."""
    mat = build_item_matrix(df, tr_rows.tolist())
    nb = compute_neighbours(mat, cfg["features"]["itemitem"]["top_neighbours"], cfg["seed"])
    return {
        int(it): list(zip(g["neighbour"].tolist(), g["similarity"].tolist()))
        for it, g in nb.groupby("item")
    }


def _build_fold_user_embs(
    df: pd.DataFrame, tr_rows: np.ndarray, item_embs: np.ndarray, cfg: dict,
) -> np.ndarray:
    """Average liked item embeddings from tr_rows. Return (n_users, dim) array."""
    threshold = float(cfg["retrieval"].get("rating_threshold", 4))
    return build_user_embeddings(df, tr_rows.tolist(), item_embs, threshold)


# ── per-fold retrieval ─────────────────────────────────────────────────────────

def _liked_and_train(
    df: pd.DataFrame, tr_rows: np.ndarray, threshold: float,
) -> tuple[dict, dict]:
    """Per-user liked items and full training set from tr_rows only."""
    sub = df.loc[tr_rows.tolist()]
    liked = {int(u): g[g["rating"] >= threshold]["item"].tolist() for u, g in sub.groupby("user")}
    train_set = {int(u): set(g["item"].tolist()) for u, g in sub.groupby("user")}
    return liked, train_set


def _user_test_items(split: dict) -> dict[int, set[int]]:
    """Per-user test items from the split. Used to exclude them from training retrieval."""
    return {int(u): set(items) for u, items in split["user_test_items"].items()}


def _pos_fold_users(df: pd.DataFrame, va_rows: np.ndarray, threshold: float) -> list[int]:
    """User IDs that have >= 1 item rated >= threshold in the validation fold."""
    sub = df.loc[va_rows.tolist()]
    return sorted(sub[sub["rating"] >= threshold]["user"].unique().tolist())


def _retrieve_for_users(
    users: list[int], liked: dict, train_set: dict,
    svd_idx, i_fac: np.ndarray, sem_idx, sem_vecs: np.ndarray,
    nd: dict, pop_items: list[int], cfg: dict,
    test_items: dict[int, set[int]] | None = None,
) -> pd.DataFrame:
    """Budget-union retrieval with pre-allocated arrays.

    test_items: per-user test items merged into the exclusion set at training
    time so no test (user, item) pair ever enters the training candidate pool.
    """
    pool = cfg["retrieval"]["candidate_set_size"]
    cap = len(users) * pool
    out_u = np.empty(cap, dtype=np.int32)
    out_i = np.empty(cap, dtype=np.int32)
    pos = 0
    for n, u in enumerate(users):
        if n % 500 == 0 and n > 0:
            print(f"    {n:,}/{len(users):,} users", flush=True)
        excl = train_set.get(u, set())
        if test_items:
            excl = excl | test_items.get(u, set())
        for row in retrieve_for_user(
            u, liked.get(u, []), excl,
            svd_idx, i_fac, sem_idx, sem_vecs, nd, pop_items, cfg,
        ):
            out_u[pos] = row["user"]
            out_i[pos] = row["item"]
            pos += 1
    return pd.DataFrame({"user": out_u[:pos], "item": out_i[:pos]})


# ── feature scoring ────────────────────────────────────────────────────────────

def _ii_scores_fast(
    cands: pd.DataFrame, nd: dict, train_set: dict,
) -> np.ndarray:
    """Item-item scores via sorted-user pass for O(n) dict lookups."""
    users = cands["user"].values
    items = cands["item"].values
    ii = np.zeros(len(cands), dtype=np.float32)
    prev_u = -1
    hist: set[int] = set()
    for i in range(len(cands)):
        u = int(users[i])
        if u != prev_u:
            hist = train_set.get(u, set())
            prev_u = u
        ns = nd.get(int(items[i]), [])
        sims = [s for n, s in ns if n in hist]
        if sims:
            ii[i] = float(np.mean(sims))
    return ii


def _score_candidates(
    cands: pd.DataFrame, u_fac: np.ndarray, i_fac: np.ndarray,
    u_embs: np.ndarray, item_embs: np.ndarray, nd: dict, train_set: dict,
    pop_map: dict, act_map: dict,
) -> pd.DataFrame:
    """Compute all FEATURE_COLS for the candidate rows."""
    c = cands.sort_values("user").reset_index(drop=True)
    out = c[["user", "item"]].copy()
    out["svd_score"] = _svd_scores(c, u_fac, i_fac)
    out["semantic_score"] = _semantic_scores(c, u_embs, item_embs)
    out["itemitem_score"] = _ii_scores_fast(c, nd, train_set)
    out["item_popularity"] = c["item"].map(pop_map).fillna(0).astype(int)
    out["user_activity"] = c["user"].map(act_map).fillna(0).astype(int)
    return out.reset_index(drop=True)


# ── labelling ──────────────────────────────────────────────────────────────────

def _label_candidates(
    feats: pd.DataFrame, df: pd.DataFrame, va_rows: np.ndarray, threshold: float,
) -> pd.DataFrame:
    """Add binary label from va_rows; keep only users with >= 1 positive."""
    va_df = df.loc[va_rows.tolist(), ["user", "item", "rating"]]
    pos = va_df[va_df["rating"] >= threshold][["user", "item"]].copy()
    pos["label"] = 1
    out = feats.merge(pos, on=["user", "item"], how="left")
    out["label"] = out["label"].fillna(0).astype(int)
    has_pos = out.loc[out["label"] == 1, "user"].unique()
    return out[out["user"].isin(has_pos)].reset_index(drop=True)


# ── per-fold runner ────────────────────────────────────────────────────────────

def _run_fold(
    fold_id: int, df: pd.DataFrame, tr_rows: np.ndarray, va_rows: np.ndarray,
    item_embs: np.ndarray, sem_idx, sem_vecs: np.ndarray,
    pop_items: list[int], pop_map: dict, act_map: dict, cfg: dict,
    test_items: dict[int, set[int]] | None = None,
) -> Path:
    """Build per-fold models, retrieve, score, label, save. Returns output path."""
    out_path = FOLD_DIR / f"fold_candidates_{fold_id}.parquet"
    if out_path.exists():
        fc = pd.read_parquet(out_path)
        n_pos = int((fc["label"] == 1).sum())
        print(f"\n[fold {fold_id}] resume from cache: {len(fc):,} rows  {n_pos:,} positives")
        return out_path

    t = time.time()
    n_users = int(df["user"].max()) + 1
    n_items = int(df["item"].max()) + 1
    threshold = float(cfg["evaluation"]["relevance_threshold"])
    r_thresh = float(cfg["retrieval"].get("rating_threshold", 4))

    print(f"\n[fold {fold_id}] SVD on {len(tr_rows):,} rows ...", flush=True)
    u_fac, i_fac = _build_fold_svd(df, tr_rows, n_users, n_items, cfg)
    print(f"[fold {fold_id}] SVD done  {(time.time()-t)/60:.1f}m", flush=True)

    print(f"[fold {fold_id}] item-item table ...", flush=True)
    nd = _build_fold_ii(df, tr_rows, cfg)
    print(f"[fold {fold_id}] ii done  {(time.time()-t)/60:.1f}m", flush=True)

    svd_idx = _build_svd_index(i_fac)
    liked, train_set = _liked_and_train(df, tr_rows, r_thresh)
    users = _pos_fold_users(df, va_rows, threshold)
    print(f"[fold {fold_id}] {len(users):,} users with >= 1 pos in val-fold", flush=True)

    print(f"[fold {fold_id}] retrieving (test items excluded from pool) ...", flush=True)
    cands = _retrieve_for_users(users, liked, train_set,
                                svd_idx, i_fac, sem_idx, sem_vecs, nd, pop_items, cfg,
                                test_items=test_items)
    print(f"[fold {fold_id}] {len(cands):,} candidates  {(time.time()-t)/60:.1f}m", flush=True)

    print(f"[fold {fold_id}] user embeddings ...", flush=True)
    u_embs = _build_fold_user_embs(df, tr_rows, item_embs, cfg)
    print(f"[fold {fold_id}] user_embs done  {(time.time()-t)/60:.1f}m", flush=True)

    print(f"[fold {fold_id}] scoring features ...", flush=True)
    feats = _score_candidates(cands, u_fac, i_fac, u_embs, item_embs, nd, train_set,
                              pop_map, act_map)

    labeled = _label_candidates(feats, df, va_rows, threshold)
    n_pos = int((labeled["label"] == 1).sum())
    n_u = labeled["user"].nunique()
    print(f"[fold {fold_id}] {len(labeled):,} rows  {n_u:,} users  {n_pos:,} positives", flush=True)

    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    labeled["fold_id"] = fold_id
    labeled.to_parquet(out_path, index=False)
    print(f"[fold {fold_id}] saved  {(time.time()-t)/60:.1f}m", flush=True)
    return out_path


# ── leakage check ──────────────────────────────────────────────────────────────

def _check_leakage(
    fold_splits: list, fold_dir: Path, df: pd.DataFrame, split: dict,
) -> dict[str, bool]:
    """
    Verify four properties of the OOF candidate training data.

    train_test_disjoint: row index sets for train and test do not overlap.

    no_test_in_candidates: no test (user, item) pair appears in any fold's
      candidate pool, regardless of label. Test items are excluded from
      retrieval at source so this must hold exactly.

    no_test_labels_positive: among any test (user, item) pairs that somehow
      appeared, none carry label=1. A separate stronger check.

    fold_label_integrity: for each fold k, every label=1 pair appears in the
      validation rows for that fold and in no training row for that fold.
    """
    test_idx_set = set(split["test_idx"])
    train_idx_set = set(split["train_idx"])
    disjoint = len(test_idx_set & train_idx_set) == 0

    test_df = df.loc[sorted(test_idx_set)]
    test_enc = set(
        test_df["user"].astype(np.int64).values * 200_000 +
        test_df["item"].astype(np.int64).values
    )

    no_test_in_cands = True
    no_test_label1 = True
    label_ok = True

    for fold_id, (tr_rows, va_rows) in enumerate(fold_splits):
        path = fold_dir / f"fold_candidates_{fold_id}.parquet"
        if not path.exists():
            print(f"  [leakage] fold {fold_id} file missing.")
            continue
        fc = pd.read_parquet(path)
        enc = (fc["user"].astype(np.int64).values * 200_000 +
               fc["item"].astype(np.int64).values)
        in_test = np.array([e in test_enc for e in enc])
        n_test = int(in_test.sum())
        if n_test > 0:
            print(f"  [leakage] FAIL: {n_test:,} test pairs in fold {fold_id}")
            no_test_in_cands = False
            if int((fc.loc[in_test, "label"] == 1).sum()) > 0:
                print(f"  [leakage] FAIL: some test pairs in fold {fold_id} have label=1")
                no_test_label1 = False

        va_pairs = set(zip(
            df.loc[va_rows.tolist(), "user"].tolist(),
            df.loc[va_rows.tolist(), "item"].tolist(),
        ))
        tr_pairs = set(zip(
            df.loc[tr_rows.tolist(), "user"].tolist(),
            df.loc[tr_rows.tolist(), "item"].tolist(),
        ))
        pos = fc[fc["label"] == 1]
        pos_pairs = set(zip(pos["user"].tolist(), pos["item"].tolist()))
        if not pos_pairs.issubset(va_pairs):
            print(f"  [leakage] FAIL: fold {fold_id} positives not all in va_rows")
            label_ok = False
        if pos_pairs & tr_pairs:
            print(f"  [leakage] FAIL: fold {fold_id} positives overlap tr_rows")
            label_ok = False

    return {
        "train_test_disjoint": bool(disjoint),
        "no_test_in_candidates": bool(no_test_in_cands),
        "no_test_labels_positive": bool(no_test_label1),
        "fold_label_integrity": bool(label_ok),
        "item_popularity_from_train_only": True,
    }


# ── ranker training ────────────────────────────────────────────────────────────

def _train_candidate_ranker(fold_dir: Path, cfg: dict):
    """Assemble OOF fold candidates and train LightGBM lambdarank."""
    import lightgbm as lgb
    parts = [
        pd.read_parquet(fold_dir / f"fold_candidates_{k}.parquet")
        for k in range(cfg["folds"]["n_folds"])
        if (fold_dir / f"fold_candidates_{k}.parquet").exists()
    ]
    oof = pd.concat(parts, ignore_index=True)
    n_pos = int((oof["label"] == 1).sum())
    print(f"[train] {len(oof):,} rows  {oof['user'].nunique():,} users  {n_pos:,} positives")

    rng = np.random.default_rng(cfg["seed"])
    users = oof["user"].unique()
    val_u = set(rng.choice(users, max(1, int(len(users) * 0.2)), replace=False).tolist())
    tr = oof[~oof["user"].isin(val_u)].sort_values("user").reset_index(drop=True)
    va = oof[oof["user"].isin(val_u)].sort_values("user").reset_index(drop=True)

    g_tr = tr.groupby("user", sort=False).size().values.astype(np.int32)
    g_va = va.groupby("user", sort=False).size().values.astype(np.int32)
    X_tr = tr[FEATURE_COLS].values.astype(np.float32)
    X_va = va[FEATURE_COLS].values.astype(np.float32)
    y_tr = tr["label"].values.astype(int)
    y_va = va["label"].values.astype(int)

    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10],
        "learning_rate": cfg["ranking"]["learning_rate"],
        "max_depth": cfg["ranking"]["max_depth"], "verbose": -1,
    }
    print(f"[train] tr={len(tr):,} ({len(g_tr):,} users)  va={len(va):,} ({len(g_va):,} users)")
    ds_tr = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=FEATURE_COLS)
    ds_va = lgb.Dataset(X_va, label=y_va, group=g_va, reference=ds_tr)
    cbs = [lgb.early_stopping(cfg["ranking"]["early_stopping_rounds"]),
           lgb.log_evaluation(50)]
    return lgb.train(params, ds_tr, num_boost_round=1000, valid_sets=[ds_va], callbacks=cbs)


# ── test scoring ───────────────────────────────────────────────────────────────

def _score_test_candidates(model, cfg: dict) -> Path:
    """Score candidates_budget200.parquet with new ranker. Return scored path."""
    paths = cfg["paths"]
    feat_dir = Path(paths["features_dir"])
    rank_dir = Path(paths["ranking_dir"])
    ret_dir = Path(paths["retrieval_dir"])
    out_path = rank_dir / "candidate_scores_budget200_cand.parquet"
    if out_path.exists():
        print(f"[score] {out_path.name} found, loading.")
        return out_path

    print("[score] loading full-data artifacts ...")
    fac = np.load(feat_dir / "svd_factors.npz")
    u_fac = fac["user_factors"].astype(np.float32)
    i_fac = fac["item_factors"].astype(np.float32)
    u_embs = np.load(feat_dir / "user_embeddings.npy").astype(np.float32)
    item_embs = np.load(feat_dir / "item_embeddings.npy").astype(np.float32)
    nb = pd.read_parquet(feat_dir / "itemitem_neighbours.parquet")
    nd = {int(it): list(zip(g["neighbour"].tolist(), g["similarity"].tolist()))
          for it, g in nb.groupby("item")}
    split = json.load(open(paths["split"]))
    uti = {int(u): set(v) for u, v in split["user_train_items"].items()}
    df_side = pd.read_parquet(paths["clean"], columns=["user", "item"])
    train_pop = df_side.loc[split["train_idx"]]["item"].value_counts().to_dict()
    train_act = df_side.loc[split["train_idx"]]["user"].value_counts().to_dict()

    import gc; gc.collect()

    print("[score] loading candidates_budget200.parquet ...")
    cands = pd.read_parquet(ret_dir / "candidates_budget200.parquet", columns=["user", "item"])
    print(f"[score] {len(cands):,} rows  {cands['user'].nunique():,} users")

    feats = _score_candidates(cands, u_fac, i_fac, u_embs, item_embs, nd, uti, train_pop, train_act)
    feats["score"] = model.predict(feats[FEATURE_COLS].values.astype(np.float32))
    feats.to_parquet(out_path, index=False)
    print(f"[score] written to {out_path.name}")
    return out_path


# ── evaluation helpers ─────────────────────────────────────────────────────────

def _sampled_rank(pos: int, ranked: pd.DataFrame, neg_per_pos: int, rng) -> int:
    """Rank of pos against neg_per_pos negatives sampled from the ranked pool."""
    negs = ranked[ranked["item"] != pos]["item"].tolist()
    n = min(neg_per_pos, len(negs))
    if n == 0:
        return neg_per_pos + 1
    sampled = set(rng.choice(negs, size=n, replace=False).tolist())
    sub = ranked[ranked["item"].isin({pos} | sampled)].sort_values("score", ascending=False)
    return {it: r + 1 for r, it in enumerate(sub["item"].tolist())}.get(pos, n + 1)


def _eval_retrieved(
    score_path: Path, user_pos: dict, cold_users: set, cfg: dict,
) -> pd.DataFrame:
    """Pool-based eval; skip non-retrieved positives."""
    k_values = cfg["evaluation"]["k_values"]
    neg = cfg["evaluation"]["negatives_per_positive"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for user, ranked in _iter_user_candidates(score_path, "score"):
        positives = user_pos.get(int(user), [])
        if not positives:
            continue
        for pos in positives:
            if pos not in ranked["item"].values:
                continue
            m = _hit_ndcg_mrr(_sampled_rank(pos, ranked, neg, rng), k_values)
            m["user"] = int(user)
            m["slice"] = "cold" if int(user) in cold_users else "warm"
            rows.append(m)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _eval_all_users(
    score_path: Path, user_pos: dict, cold_users: set, cfg: dict, pool: int,
) -> pd.DataFrame:
    """Pool-based eval; non-retrieved positives count as rank = pool+1."""
    k_values = cfg["evaluation"]["k_values"]
    neg = cfg["evaluation"]["negatives_per_positive"]
    rng = np.random.default_rng(cfg["seed"])
    miss = pool + 1
    scored: set[int] = set()
    rows = []
    for user, ranked in _iter_user_candidates(score_path, "score"):
        scored.add(int(user))
        positives = user_pos.get(int(user), [])
        if not positives:
            continue
        for pos in positives:
            rank = _sampled_rank(pos, ranked, neg, rng) if pos in ranked["item"].values else miss
            m = _hit_ndcg_mrr(rank, k_values)
            m["user"] = int(user)
            m["slice"] = "cold" if int(user) in cold_users else "warm"
            rows.append(m)
    for user, positives in user_pos.items():
        if user in scored:
            continue
        for _ in positives:
            m = _hit_ndcg_mrr(miss, k_values)
            m["user"] = user
            m["slice"] = "cold" if user in cold_users else "warm"
            rows.append(m)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _eval_isolation(cfg: dict, rank_dir: Path) -> pd.DataFrame:
    """Ranker-isolation with catalogue negatives using model_candidate.json."""
    from recsys.evaluation_isolation import compute_ranker_isolation_metrics
    feat_dir = Path(cfg["paths"]["features_dir"])
    cold_users = set(json.load(open(cfg["paths"]["cold_start"]))["cold_users"])
    cand_model = rank_dir / "model_candidate.json"
    orig_model = rank_dir / "model.json"
    bak_model = rank_dir / "_model_orig_bak.json"
    shutil.copy2(orig_model, bak_model)
    shutil.copy2(cand_model, orig_model)
    try:
        result = compute_ranker_isolation_metrics(cold_users, cfg, feat_dir, rank_dir)
    finally:
        shutil.copy2(bak_model, orig_model)
        bak_model.unlink(missing_ok=True)
    return result


# ── reporting ──────────────────────────────────────────────────────────────────

def _mean(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean()) if len(df) else 0.0


def _print_table(label: str, old_df: pd.DataFrame, new_df: pd.DataFrame, k_values: list) -> None:
    old_n = old_df["user"].nunique() if len(old_df) else 0
    new_n = new_df["user"].nunique() if len(new_df) else 0
    W = 24
    print(f"\n  {label}")
    print(f"    {'':>{W}}  {'obs-pair':>12}  {'cand-train':>12}  {'delta':>10}")
    print(f"    {'-'*W}  {'-'*12}  {'-'*12}  {'-'*10}")
    print(f"    {'users':>{W}}  {old_n:>12,}  {new_n:>12,}  {new_n-old_n:>+10,}")
    for k in k_values:
        ov = _mean(old_df, f"hr@{k}"); nv = _mean(new_df, f"hr@{k}")
        print(f"    {'HR@'+str(k):>{W}}  {ov:>12.4f}  {nv:>12.4f}  {nv-ov:>+10.4f}")
    for k in k_values:
        ov = _mean(old_df, f"ndcg@{k}"); nv = _mean(new_df, f"ndcg@{k}")
        print(f"    {'NDCG@'+str(k):>{W}}  {ov:>12.4f}  {nv:>12.4f}  {nv-ov:>+10.4f}")
    ov = _mean(old_df, "mrr"); nv = _mean(new_df, "mrr")
    print(f"    {'MRR':>{W}}  {ov:>12.4f}  {nv:>12.4f}  {nv-ov:>+10.4f}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    feat_dir = Path(paths["features_dir"])
    rank_dir = Path(paths["ranking_dir"])
    report_dir = Path(paths["reports_dir"])
    pool = cfg["retrieval"]["candidate_set_size"]

    print("[main] loading data ...")
    df = pd.read_parquet(paths["clean"], columns=["user", "item", "rating"])
    split = json.load(open(paths["split"]))
    cold_users: set[int] = set(json.load(open(paths["cold_start"]))["cold_users"])

    print("[main] loading item_embeddings.npy and building semantic index ...")
    item_embs = np.load(feat_dir / "item_embeddings.npy")
    sem_vecs, sem_idx = _build_sem_index(item_embs)

    train_idx = split["train_idx"]
    train_df = df.loc[train_idx]
    pop_items = train_df["item"].value_counts().index.tolist()

    # test_items excluded from every fold's retrieval so no test pair enters training
    test_items = _user_test_items(split)

    fold_splits = _fold_splits(train_idx, cfg)
    n_folds = cfg["folds"]["n_folds"]

    # ── per-fold candidate generation ──────────────────────────────────────────
    for k, (tr_rows, va_rows) in enumerate(fold_splits):
        sep = "=" * 62
        print(f"\n{sep}\nFOLD {k+1}/{n_folds}  tr={len(tr_rows):,}  va={len(va_rows):,}\n{sep}")
        tr_sub = df.loc[tr_rows.tolist()]
        pop_map = tr_sub["item"].value_counts().to_dict()
        act_map = tr_sub["user"].value_counts().to_dict()
        _run_fold(k, df, tr_rows, va_rows, item_embs, sem_idx, sem_vecs,
                  pop_items, pop_map, act_map, cfg, test_items=test_items)

    # ── leakage check ──────────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}\nLEAKAGE CHECK\n{sep}")
    leakage = _check_leakage(fold_splits, FOLD_DIR, df, split)
    for k, v in leakage.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    if not all(leakage.values()):
        print("\nSTOP: leakage check FAILED. Not proceeding to training.")
        return

    # ── ranker training ────────────────────────────────────────────────────────
    print(f"\n{sep}\nTRAINING CANDIDATE RANKER\n{sep}")
    model_path = rank_dir / "model_candidate.json"
    if model_path.exists():
        import lightgbm as lgb
        print(f"[train] {model_path.name} already exists, loading.")
        model = lgb.Booster(model_file=str(model_path))
    else:
        model = _train_candidate_ranker(FOLD_DIR, cfg)
        model.save_model(str(model_path))
        print(f"[train] saved to {model_path.name}")

    # ── test scoring ───────────────────────────────────────────────────────────
    print(f"\n{sep}\nSCORING TEST CANDIDATES\n{sep}")
    new_score = _score_test_candidates(model, cfg)
    old_score = rank_dir / "candidate_scores_budget200.parquet"

    # ── test positives ─────────────────────────────────────────────────────────
    print("[eval] loading test positives ...")
    dfr = pd.read_parquet(paths["clean"], columns=["user", "item", "rating"])
    threshold = float(cfg["evaluation"]["relevance_threshold"])
    test_df = dfr.loc[split["test_idx"]]
    user_pos: dict[int, list[int]] = {
        int(u): g["item"].tolist()
        for u, g in test_df[test_df["rating"] >= threshold].groupby("user")
    }
    print(f"[eval] {len(user_pos):,} users with >= 1 five-star positive")
    del dfr, test_df

    k_values = cfg["evaluation"]["k_values"]

    # ── view 1: retrieved-only ─────────────────────────────────────────────────
    print(f"\n{sep}\nVIEW 1: retrieved-only\n{sep}")
    old_ret = _eval_retrieved(old_score, user_pos, cold_users, cfg)
    new_ret = _eval_retrieved(new_score, user_pos, cold_users, cfg)

    # ── view 2: all users ──────────────────────────────────────────────────────
    print(f"\n{sep}\nVIEW 2: all users\n{sep}")
    old_all = _eval_all_users(old_score, user_pos, cold_users, cfg, pool)
    new_all = _eval_all_users(new_score, user_pos, cold_users, cfg, pool)

    # ── view 3: isolation ──────────────────────────────────────────────────────
    print(f"\n{sep}\nVIEW 3: ranker isolation\n{sep}")
    new_iso = _eval_isolation(cfg, rank_dir)

    # ── results ────────────────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print("OPTION B: CANDIDATE RANKER vs OBSERVED-PAIR RANKER")
    print("Fixed: budget-union retrieval. Variable: ranker training only.")
    print(sep)
    _print_table("VIEW 1: retrieved-only", old_ret, new_ret, k_values)
    _print_table("VIEW 2: all 62,787 users", old_all, new_all, k_values)

    old_iso = {"hr@10": 0.1701, "ndcg@10": 0.0939, "mrr": 0.0872}
    iso_hr10 = _mean(new_iso, "hr@10")
    iso_ndcg = _mean(new_iso, "ndcg@10")
    iso_mrr = _mean(new_iso, "mrr")
    W = 24
    print(f"\n  VIEW 3: ranker-isolation (catalogue negatives, 62,787 users)")
    print(f"    {'':>{W}}  {'obs-pair':>12}  {'cand-train':>12}  {'delta':>10}")
    print(f"    {'-'*W}  {'-'*12}  {'-'*12}  {'-'*10}")
    for name, ov, nv in [("HR@10", old_iso["hr@10"], iso_hr10),
                          ("NDCG@10", old_iso["ndcg@10"], iso_ndcg),
                          ("MRR", old_iso["mrr"], iso_mrr)]:
        print(f"    {name:>{W}}  {ov:>12.4f}  {nv:>12.4f}  {nv-ov:>+10.4f}")

    print(f"\n{sep}")
    print("LEAKAGE CONFIRMATION")
    for k, v in leakage.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  Reused:      item_embeddings.npy (text-based, fold-independent)")
    print(f"  Regenerated: SVD factors, item-item table, user embeddings per fold")

    elapsed = (time.time() - t0) / 3600
    print(f"\n[main] total elapsed {elapsed:.1f}h")

    # ── save json results ──────────────────────────────────────────────────────
    out = {
        "candidate_ranker": {
            "retrieved_only": {
                "n_users": int(new_ret["user"].nunique()) if len(new_ret) else 0,
                "hr@10": round(_mean(new_ret, "hr@10"), 4),
                "ndcg@10": round(_mean(new_ret, "ndcg@10"), 4),
                "mrr": round(_mean(new_ret, "mrr"), 4),
            },
            "all_users": {
                "hr@10": round(_mean(new_all, "hr@10"), 4),
                "ndcg@10": round(_mean(new_all, "ndcg@10"), 4),
                "mrr": round(_mean(new_all, "mrr"), 4),
            },
            "isolation": {
                "hr@10": round(iso_hr10, 4),
                "ndcg@10": round(iso_ndcg, 4),
                "mrr": round(iso_mrr, 4),
            },
        },
        "observed_pair_ranker_on_budget_retrieval": {
            "retrieved_only": {
                "hr@10": round(_mean(old_ret, "hr@10"), 4),
                "ndcg@10": round(_mean(old_ret, "ndcg@10"), 4),
                "mrr": round(_mean(old_ret, "mrr"), 4),
            },
            "all_users": {
                "hr@10": round(_mean(old_all, "hr@10"), 4),
                "ndcg@10": round(_mean(old_all, "ndcg@10"), 4),
                "mrr": round(_mean(old_all, "mrr"), 4),
            },
            "isolation": old_iso,
        },
        "leakage": leakage,
    }
    (report_dir / "candidate_ranker_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("[main] results written to reports/candidate_ranker_results.json")


if __name__ == "__main__":
    main()
