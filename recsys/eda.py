"""EDA utilities: sparsity stats and distribution plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def run_eda(df: pd.DataFrame, out_dir: Path) -> None:
    """Print sparsity stats and save distribution plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_u, n_i = df["user"].nunique(), df["item"].nunique()
    sparsity = 1.0 - len(df) / (n_u * n_i)
    print(f"  users={n_u:,}  items={n_i:,}  rows={len(df):,}  sparsity={sparsity:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["rating"], bins=5, ax=axes[0], discrete=True)
    axes[0].set_title("Rating distribution")
    uc = df["user"].value_counts()
    sns.histplot(uc[uc <= uc.quantile(0.95)], bins=40, ax=axes[1])
    axes[1].set_title("Reviews per user (95th pct)")
    fig.tight_layout()
    fig.savefig(out_dir / "eda.png", dpi=100)
    plt.close(fig)
    print(f"  EDA plot saved to {out_dir / 'eda.png'}")
