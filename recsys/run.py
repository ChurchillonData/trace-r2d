"""
Pipeline runner: executes all stages in order.

Usage:
    python -m recsys.run
    python -m recsys.run --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Callable

# Claim torch's OpenMP DLL before scikit-surprise/faiss loads its own copy (WinError 1114).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401

import yaml

from recsys import baselines, evaluation, retrieval
from recsys.data import build as build_data
from recsys.features import build_all as build_features
from recsys.ranking import build as build_ranking


def _load_cfg(path: str) -> dict:
    """Load YAML config and return as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def _elapsed(start: float) -> str:
    """Return human-readable elapsed time."""
    s = time.time() - start
    return f"{s / 60:.1f}m" if s >= 60 else f"{s:.1f}s"


def _stage(name: str, fn: Callable, cfg: dict, t0: float) -> None:
    """Print a stage header, run fn(cfg), then report elapsed time."""
    print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
    fn(cfg)
    print(f"  [{_elapsed(t0)}]")


def _cleanup_post_ranking(cfg: dict) -> None:
    """Delete large intermediates that are no longer needed after ranking.

    ranking/candidate_scores.parquet captures every signal, so the raw
    embedding arrays, svd factors, neighbour tables, and per-feature OOF/test
    parquets can all go.  oof_scores.parquet and test_scores.parquet are kept
    because evaluation still reads them.  candidates.parquet is kept because
    evaluation reads it for retrieval recall; cleanup.py removes it afterward.
    """
    feat_dir = Path(cfg["paths"]["features_dir"])
    targets = [
        feat_dir / "user_embeddings.npy",
        feat_dir / "item_embeddings.npy",
        feat_dir / "svd_factors.npz",
        feat_dir / "itemitem_neighbours.parquet",
        feat_dir / "tfidf_neighbours.parquet",
        feat_dir / "svd_oof.parquet",
        feat_dir / "semantic_oof.parquet",
        feat_dir / "itemitem_oof.parquet",
        feat_dir / "tfidf_oof.parquet",
        feat_dir / "svd_test.parquet",
        feat_dir / "semantic_test.parquet",
        feat_dir / "itemitem_test.parquet",
        feat_dir / "tfidf_test.parquet",
    ]
    freed = 0
    for p in targets:
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
    print(f"  [cleanup] freed {freed / 1_000_000:.0f} MB of ranking intermediates")


def run(cfg: dict) -> None:
    """Execute every stage in dependency order."""
    t0 = time.time()
    _stage("Stage 0: data", build_data, cfg, t0)
    _stage("Stage 1a: features", build_features, cfg, t0)
    _stage("Stage 1b: retrieval", retrieval.build, cfg, t0)
    _stage("Stage 2: ranking", build_ranking, cfg, t0)
    _cleanup_post_ranking(cfg)
    _stage("Stage 2b: baselines", baselines.build, cfg, t0)
    _stage("Stage 3: evaluation", evaluation.build, cfg, t0)
    print(f"\nPipeline complete in {_elapsed(t0)}.")


def main() -> None:
    """Parse args and run the pipeline."""
    parser = argparse.ArgumentParser(description="Two-stage recommender pipeline")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    args = parser.parse_args()
    run(_load_cfg(args.config))


if __name__ == "__main__":
    main()
