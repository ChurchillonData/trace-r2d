"""
Two figures for the diagnostic experiments.

fig_oracle_injection.png
  (a) HR@10 against injection probability, with a straight reference line, so
      the near-linear dose response is visible rather than inferred from a
      table. (b) conversion rate against candidate budget, the rate at which a
      retrieved positive becomes a top-ten hit, which stays flat as the pool
      grows tenfold.

fig_feature_importance.png
  Normalised LightGBM gain per feature under both training schemes. This is
  mechanistic rather than descriptive: it shows what the aligned ranker
  actually learns to rely on.

Palette is sampled from the figures in the original paper so the whole set
matches. Blue against orange also stays separable under the common colour
vision deficiencies and in greyscale print.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("reports/figures")

# Palette sampled from the figures in the original paper, so the new figures
# match. Blue against orange is also the safest pairing under the common
# colour vision deficiencies, so matching costs nothing in accessibility.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#8c8c8c"
INK = "#222222"


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors=INK, labelsize=8.5, length=3)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.35, color=GREY)
    ax.set_axisbelow(True)


def oracle_figure() -> None:
    orc = json.loads(Path("reports/oracle_retrieval_results.json").read_text())
    abl = json.loads(Path("reports/budget_ablation_results.json").read_text())

    ps = [0.0, 0.25, 0.50, 0.75, 1.0]
    hr = [orc["results"][str(p)]["hr@10"] for p in ps]

    budgets = abl["budgets"]
    conv, recall = [], []
    for b in budgets:
        r = abl["results"][str(b)]["cand"]
        recall.append(r["recall"] * 100)
        conv.append(r["hr@10"] / r["recall"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.1))
    fig.subplots_adjust(wspace=0.30)

    # ---- (a) oracle dose response ----
    ideal = [hr[0] + (hr[-1] - hr[0]) * p for p in ps]
    ax1.plot(ps, ideal, linestyle="--", linewidth=1.4, color=GREY,
             zorder=2, label="Linear reference")
    ax1.plot(ps, hr, marker="o", linewidth=2.0, markersize=6, color=BLUE,
             markerfacecolor="white", markeredgewidth=1.8, zorder=3,
             label="Measured HR@10")
    for p, v in zip(ps, hr):
        ax1.annotate(f"{v:.4f}", (p, v), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=7.5, color=INK)
    ax1.set_xlabel("Injection probability $p$", fontsize=9.5)
    ax1.set_ylabel("HR@10 (all users)", fontsize=9.5)
    ax1.set_title("(a) Retrieval dose response", fontsize=10, fontweight="bold")
    ax1.set_xticks(ps)
    ax1.set_xlim(-0.06, 1.06)
    ax1.set_ylim(0, max(hr) * 1.26)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax1)

    # ---- (b) conversion rate ----
    ax2.plot(budgets, conv, marker="s", linewidth=2.0, markersize=6,
             color=ORANGE, markerfacecolor="white", markeredgewidth=1.8, zorder=3)
    for b, c, r in zip(budgets, conv, recall):
        ax2.annotate(f"{r:.1f}%", (b, c), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=7, color=GREY)
    ax2.set_xlabel("Candidate budget (recall annotated)", fontsize=9.5)
    ax2.set_ylabel("HR@10 / recall", fontsize=9.5)
    ax2.set_title("(b) Conversion rate is flat", fontsize=10, fontweight="bold")
    ax2.set_xticks(budgets)
    ax2.set_ylim(0, 1.0)
    _style(ax2)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_oracle_injection.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


def importance_figure() -> None:
    import lightgbm as lgb

    names = {"svd_score": "SVD", "semantic_score": "SBERT",
             "itemitem_score": "Item-to-item",
             "item_popularity": "Item popularity",
             "user_activity": "User activity"}

    data = {}
    for lab, f in [("Observed-pair", "model.json"),
                   ("Aligned", "model_candidate.json")]:
        b = lgb.Booster(model_file=f"ranking/{f}")
        g = np.asarray(b.feature_importance(importance_type="gain"), dtype=float)
        data[lab] = dict(zip(b.feature_name(), g / g.sum() * 100))
        data[lab + "_trees"] = b.num_trees()

    order = ["item_popularity", "semantic_score", "svd_score",
             "itemitem_score", "user_activity"]
    y = np.arange(len(order))
    h = 0.38

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    obs = [data["Observed-pair"][k] for k in order]
    ali = [data["Aligned"][k] for k in order]

    ax.barh(y + h / 2, obs, h, color=BLUE,
            label=f"Observed-pair ({data['Observed-pair_trees']} trees)")
    ax.barh(y - h / 2, ali, h, color=ORANGE,
            label=f"Aligned ({data['Aligned_trees']} trees)")
    for yy, v in zip(y + h / 2, obs):
        ax.text(v + 1.2, yy, f"{v:.1f}%", va="center", fontsize=7.5, color=INK)
    for yy, v in zip(y - h / 2, ali):
        ax.text(v + 1.2, yy, f"{v:.1f}%", va="center", fontsize=7.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([names[k] for k in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Share of total split gain (%)", fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_title("What each training scheme learns to rely on",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors=INK, labelsize=8.5, length=3)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.35, color=GREY)
    ax.set_axisbelow(True)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_feature_importance.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")

    print("\nnormalised gain (%):")
    for k in order:
        print(f"  {names[k]:18s} obs={data['Observed-pair'][k]:5.1f}  "
              f"aligned={data['Aligned'][k]:5.1f}")


if __name__ == "__main__":
    oracle_figure()
    importance_figure()
