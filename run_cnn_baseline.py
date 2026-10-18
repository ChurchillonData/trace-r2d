"""
CNN text-encoder baseline.

Trains a TextCNN on training reviews to predict ratings, extracts item
representations from the penultimate layer, scores the same retrieved
candidate set as every other baseline, and writes results into metrics.json.

Split contract
--------------
Vocabulary, tokenisation, CNN weights, and user profiles all derive
exclusively from split["train_idx"]. No test row is seen during training.

Isolation
---------
Writes: ranking/baseline_cnn.parquet  (candidate list scored by CNN)
        reports/metrics.json          (CNN row appended to baselines + beyond_accuracy)
Reads (never writes): everything else.

Usage
-----
  python run_cnn_baseline.py
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Constants — mirrors what the methodology describes; not tuned beyond "fair"
# ---------------------------------------------------------------------------
MAX_VOCAB    = 10_000
MAX_SEQ_LEN  = 128
EMBED_DIM    = 100
NUM_FILTERS  = 100
FILTER_SIZES = [2, 3, 4]
REPR_DIM     = 128      # item / user embedding dimension
DROPOUT      = 0.3
EPOCHS       = 5
BATCH_SIZE   = 256
LR           = 1e-3
SEED         = 42


# ---------------------------------------------------------------------------
# Vocabulary and tokenisation — training data only
# ---------------------------------------------------------------------------

def _build_vocab(texts: list[str], max_vocab: int) -> dict[str, int]:
    counter: Counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def _tokenise(texts: list[str], vocab: dict[str, int], max_len: int) -> np.ndarray:
    unk = vocab["<UNK>"]
    out = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, unk) for w in t.lower().split()[:max_len]]
        out[i, : len(ids)] = ids
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _ReviewDataset(Dataset):
    def __init__(self, tokens: np.ndarray, ratings: np.ndarray) -> None:
        self.tokens  = torch.from_numpy(tokens).long()
        self.ratings = torch.from_numpy(ratings).float()

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.ratings[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TextCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(MAX_VOCAB + 2, EMBED_DIM, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(EMBED_DIM, NUM_FILTERS, fs, padding=fs // 2)
            for fs in FILTER_SIZES
        ])
        self.repr_proj = nn.Linear(NUM_FILTERS * len(FILTER_SIZES), REPR_DIM)
        self.dropout   = nn.Dropout(DROPOUT)
        self.head      = nn.Linear(REPR_DIM, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).permute(0, 2, 1)  # (B, embed_dim, seq_len)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))                          # (B, num_filters, L)
            pooled.append(F.max_pool1d(h, h.size(2)).squeeze(2))  # (B, num_filters)
        cat = self.dropout(torch.cat(pooled, dim=1))       # (B, num_filters * n_sizes)
        return F.normalize(self.repr_proj(cat), dim=-1)    # (B, REPR_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)).squeeze(-1)       # (B,)


# ---------------------------------------------------------------------------
# Training — uses training rows only
# ---------------------------------------------------------------------------

def _train(tokens: np.ndarray, ratings: np.ndarray) -> TextCNN:
    torch.manual_seed(SEED)
    model  = TextCNN()
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(_ReviewDataset(tokens, ratings),
                        batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  epoch {epoch}/{EPOCHS}  loss={total_loss / len(loader):.4f}")
    return model


# ---------------------------------------------------------------------------
# Item embeddings — aggregate up to 3 training reviews per item
# ---------------------------------------------------------------------------

def _item_embeddings(
    df: pd.DataFrame,
    train_idx: list[int],
    vocab: dict[str, int],
    model: TextCNN,
) -> tuple[np.ndarray, list[int]]:
    train_df   = df.loc[train_idx, ["item", "review"]]
    item_texts = (train_df.groupby("item")["review"]
                  .apply(lambda x: " ".join(x.tolist()[:3])))
    item_ids   = item_texts.index.tolist()
    tokens     = _tokenise(item_texts.tolist(), vocab, MAX_SEQ_LEN)
    model.eval()
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tokens), BATCH_SIZE):
            t = torch.from_numpy(tokens[start : start + BATCH_SIZE]).long()
            embs.append(model.encode(t).numpy())
    return np.vstack(embs).astype(np.float32), item_ids


# ---------------------------------------------------------------------------
# Full item embedding array indexed by item ID (dense IDs assumed)
# ---------------------------------------------------------------------------

def _build_item_array(item_embs: np.ndarray, item_ids: list[int]) -> np.ndarray:
    n = max(item_ids) + 1
    arr = np.zeros((n, REPR_DIM), dtype=np.float32)
    for idx, iid in enumerate(item_ids):
        arr[iid] = item_embs[idx]
    return arr


# ---------------------------------------------------------------------------
# User profiles — avg CNN embedding of high-rated training items
# ---------------------------------------------------------------------------

def _build_profile_array(
    df: pd.DataFrame,
    train_idx: list[int],
    item_arr: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_df = df.loc[train_idx, ["user", "item", "rating"]]
    n_users  = int(train_df["user"].max()) + 1
    prof_arr = np.zeros((n_users, REPR_DIM), dtype=np.float32)
    has_prof = np.zeros(n_users, dtype=bool)

    for user, grp in train_df.groupby("user"):
        u    = int(user)
        liked = grp.loc[grp["rating"] >= threshold, "item"].values
        if len(liked) == 0:
            liked = grp["item"].values
        valid = liked[liked < len(item_arr)]
        vecs  = item_arr[valid]
        if len(vecs):
            prof_arr[u] = vecs.mean(axis=0)
            has_prof[u] = True

    return prof_arr, has_prof


# ---------------------------------------------------------------------------
# Score candidates — vectorised per-user groupby (avoids 15.97M Python calls)
# ---------------------------------------------------------------------------

def _score_candidates(
    cand_path: Path,
    item_arr: np.ndarray,
    prof_arr: np.ndarray,
    has_prof: np.ndarray,
) -> pd.DataFrame:
    cand   = pd.read_parquet(cand_path, columns=["user", "item"])
    scores = np.zeros(len(cand), dtype=np.float32)

    for user, grp in cand.groupby("user"):
        u = int(user)
        if u >= len(has_prof) or not has_prof[u]:
            continue
        idx   = grp.index.values
        items = grp["item"].values
        mask  = items < len(item_arr)
        if mask.any():
            scores[idx[mask]] = item_arr[items[mask]] @ prof_arr[u]

    cand["score"] = scores
    return cand.sort_values("user").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    with open("configs/default.yaml") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Ranking evaluation — identical protocol to every other baseline
# ---------------------------------------------------------------------------

def _run_ranking_eval(cnn_path: Path, cfg: dict) -> dict:
    from recsys.evaluation import compute_ranking_metrics, _ranking_dict

    df_light = pd.read_parquet(cfg["paths"]["clean"],
                               columns=["user", "item", "rating"])
    split     = json.loads(Path(cfg["paths"]["split"]).read_text())
    cold_path = cfg["paths"].get("cold_start", "data/cold_start.json")
    cold_users: set[int] = set(
        json.loads(Path(cold_path).read_text()).get("cold_users", [])
    )
    threshold = float(cfg["evaluation"]["relevance_threshold"])
    test_df   = df_light.loc[split["test_idx"]]
    pos_df    = test_df[test_df["rating"] >= threshold]
    user_positives = {int(u): g["item"].tolist()
                      for u, g in pos_df.groupby("user")}

    print("[eval] computing ranking metrics ...")
    results = compute_ranking_metrics(
        cnn_path, "score", user_positives, cold_users, cfg
    )
    k_values     = cfg["evaluation"]["k_values"]
    ranking_dict = _ranking_dict("cnn", results, k_values)
    print(f"  evaluated {len(results):,} (user, positive) pairs")
    for sl in ["all", "warm"]:
        s = ranking_dict.get(sl, {})
        if s:
            print(f"  [{sl}]  HR@10={s.get('hr@10', 0):.4f}  "
                  f"NDCG@10={s.get('ndcg@10', 0):.4f}  MRR={s.get('mrr', 0):.4f}")
    return ranking_dict


# ---------------------------------------------------------------------------
# Beyond-accuracy metrics — ILD / novelty / Gini
# ---------------------------------------------------------------------------

def _run_beyond_eval(cnn_path: Path, cfg: dict) -> dict:
    from eval_beyond_accuracy import _load_top_n, _metrics_for_lists, _train_popularity

    feat_dir = Path(cfg["paths"]["features_dir"])
    split    = json.loads(Path(cfg["paths"]["split"]).read_text())

    embs_raw = np.load(feat_dir / "item_embeddings.npy")
    norms    = np.linalg.norm(embs_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs    = (embs_raw / norms).astype(np.float32)
    n_items = embs.shape[0]

    pop_map, total_train = _train_popularity(cfg["paths"]["clean"], split["train_idx"])

    print("[beyond] computing ILD / novelty / Gini ...")
    top_lists = _load_top_n(cnn_path, cfg["ranking"]["top_n"])
    vals      = _metrics_for_lists(top_lists, embs, pop_map, total_train, n_items)
    print(f"  ILD={vals['ild']:.4f}  Novelty={vals['novelty']:.4f}  Gini={vals['gini']:.4f}")
    return vals


# ---------------------------------------------------------------------------
# Patch metrics.json — append CNN to baselines and beyond_accuracy
# ---------------------------------------------------------------------------

def _patch_metrics(ranking_dict: dict, beyond_dict: dict, cfg: dict) -> None:
    report_path = Path(cfg["paths"]["reports_dir"]) / "metrics.json"
    existing    = json.loads(report_path.read_text(encoding="utf-8"))
    existing.setdefault("baselines", {})["cnn"] = ranking_dict
    existing.setdefault("beyond_accuracy", {})["cnn"] = beyond_dict
    report_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"[patch] metrics.json updated — CNN added to baselines + beyond_accuracy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0  = time.time()
    sep = "=" * 62
    print(f"\n{sep}")
    print("CNN TEXT-ENCODER BASELINE")
    print(f"{sep}")
    print(f"  vocab={MAX_VOCAB}  seq_len={MAX_SEQ_LEN}  embed_dim={EMBED_DIM}")
    print(f"  filters={NUM_FILTERS}x{FILTER_SIZES}  repr_dim={REPR_DIM}")
    print(f"  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}  seed={SEED}")
    print(f"\n  Isolation: reads only clean.parquet, split.json, candidate_scores.parquet,")
    print(f"             item_embeddings.npy, metrics.json")
    print(f"  Writes:    ranking/baseline_cnn.parquet  reports/metrics.json")

    cfg      = _load_cfg()
    paths    = cfg["paths"]
    cnn_path = Path(paths["ranking_dir"]) / "baseline_cnn.parquet"

    # 1. Load training data
    print(f"\n[step 1] loading training data ...")
    df    = pd.read_parquet(paths["clean"], columns=["user", "item", "rating", "review"])
    split = json.loads(Path(paths["split"]).read_text())
    train_idx = split["train_idx"]
    threshold = float(cfg["evaluation"]["relevance_threshold"])
    n_train   = len(train_idx)
    n_items   = df.loc[train_idx, "item"].nunique()
    print(f"  training rows={n_train:,}  unique items={n_items:,}  "
          f"unique users={df.loc[train_idx, 'user'].nunique():,}")

    # 2. Vocabulary from training reviews only
    print(f"\n[step 2] building vocabulary ...")
    train_texts = df.loc[train_idx, "review"].tolist()
    vocab = _build_vocab(train_texts, MAX_VOCAB)
    print(f"  vocab size={len(vocab):,} (cap={MAX_VOCAB})")

    # 3. Tokenise
    print(f"\n[step 3] tokenising {n_train:,} training reviews ...")
    tokens  = _tokenise(train_texts, vocab, MAX_SEQ_LEN)
    ratings = df.loc[train_idx, "rating"].values.astype(np.float32)
    r_min, r_max = ratings.min(), ratings.max()
    ratings = (ratings - r_min) / (r_max - r_min + 1e-8)
    print(f"  tokens shape={tokens.shape}  ratings normalised [{r_min:.1f},{r_max:.1f}] -> [0,1]")

    # 4. Train CNN
    print(f"\n[step 4] training TextCNN ({EPOCHS} epochs, batch={BATCH_SIZE}) ...")
    model = _train(tokens, ratings)

    # 5. Item embeddings
    print(f"\n[step 5] encoding items from training reviews ...")
    item_embs, item_ids = _item_embeddings(df, train_idx, vocab, model)
    item_arr = _build_item_array(item_embs, item_ids)
    print(f"  encoded {len(item_ids):,} items  embedding shape={item_embs.shape}")

    # 6. User profiles
    print(f"\n[step 6] building user profiles (threshold={threshold}) ...")
    prof_arr, has_prof = _build_profile_array(df, train_idx, item_arr, threshold)
    n_prof = int(has_prof.sum())
    print(f"  profiles built for {n_prof:,} users")

    # 7. Score candidates
    print(f"\n[step 7] scoring candidate pairs (vectorised per-user) ...")
    cand_path = Path(paths["ranking_dir"]) / "candidate_scores.parquet"
    cnn_df    = _score_candidates(cand_path, item_arr, prof_arr, has_prof)
    cnn_df.to_parquet(cnn_path, index=False)
    covered   = int((cnn_df["score"] != 0).sum())
    print(f"  {len(cnn_df):,} pairs total  non-zero={covered:,} ({100*covered/len(cnn_df):.1f}%)")
    print(f"  saved -> {cnn_path}")

    # 8. Ranking evaluation
    print()
    ranking_dict = _run_ranking_eval(cnn_path, cfg)

    # 9. Beyond-accuracy
    print()
    beyond_dict = _run_beyond_eval(cnn_path, cfg)

    # 10. Patch metrics.json
    print()
    _patch_metrics(ranking_dict, beyond_dict, cfg)

    elapsed = (time.time() - t0) / 60
    print(f"\n{sep}")
    print(f"DONE  total elapsed={elapsed:.1f}m")
    print(f"{sep}")

    # Final comparison table
    existing = json.loads((Path(paths["reports_dir"]) / "metrics.json").read_text())
    sbert    = existing["baselines"].get("semantic", {}).get("all", {})
    cnn      = existing["baselines"].get("cnn", {}).get("all", {})
    print("\nDirect comparison (warm + cold, all test users):")
    print(f"  {'Model':<24}  HR@10   NDCG@10   MRR")
    print(f"  {'-'*24}  ------  -------  ------")
    print(f"  {'SBERT (pretrained)':<24}  "
          f"{sbert.get('hr@10', 0):.4f}  {sbert.get('ndcg@10', 0):.4f}    "
          f"{sbert.get('mrr', 0):.4f}")
    print(f"  {'CNN (trained on data)':<24}  "
          f"{cnn.get('hr@10', 0):.4f}  {cnn.get('ndcg@10', 0):.4f}    "
          f"{cnn.get('mrr', 0):.4f}")
    delta = cnn.get("hr@10", 0) - sbert.get("hr@10", 0)
    winner = "CNN" if delta > 0 else "SBERT"
    print(f"\n  HR@10 delta (CNN - SBERT): {delta:+.4f}  -> {winner} wins")


if __name__ == "__main__":
    main()
