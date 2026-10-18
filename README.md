# Retrieval or Ranking?

Diagnosing the dominant bottleneck in a hybrid two-stage recommender on a sparse product catalogue.

<p align="center">
  <img src="paper/figures/fig_architecture_new.png" alt="System architecture" width="100%">
</p>

A two-stage recommender built on Amazon Electronics Reviews 2023. SVD, SBERT and item-to-item signals retrieve a 200-item candidate pool, and a LightGBM LambdaRank model ranks it. Retrieval and ranking are measured separately to identify which stage limits end-to-end performance.

## Findings

- **Retrieval is the dominant bottleneck.** Recall@200 is 3.30%. Injecting the held-out item into the pool with the ranker held fixed raises HR@10 by 8.3×.
- **Training-serving alignment matters.** Training the ranker on retrieved candidates instead of observed pairs raises isolated HR@10 from 0.17 to 0.52. Feature attribution shows the corrected ranker relies mostly on item popularity.
- **Cold-start is the dominant regime.** 70% of users and 83% of items are cold. The semantic signal reaches cold items but not cold users.

## Setup

```bash
pip install -r requirements.txt
python setup_data.py
python setup_sbert.py
```

## Reproducing the results

Run the pipeline and train the aligned ranker:

```bash
python -m recsys.run
python measure_budget_recall.py
python build_candidate_ranker.py
python fix_candidate_leakage.py
```

Each result in the paper then has its own script:

| Result | Script |
|---|---|
| Main results | `final_results_table.py` |
| Retrieval recall | `measure_recall_conditions.py` |
| Ranker in isolation | `run_isolation_eval.py`, `recover_isolation_hr.py` |
| Beyond-accuracy metrics | `eval_beyond_accuracy.py` |
| Text-encoder comparison | `run_cnn_baseline.py` |
| Cold-start | `coldstart_experiment.py`, `coldstart_extended.py` |
| Oracle retrieval | `oracle_retrieval.py` |
| Initialisation variance | `confidence_intervals.py` |
| Candidate-budget ablation | `budget_ablation.py` |
| MovieLens density check | `ml_density_check.py` |
| Figures | `make_new_figures.py` |

Results are written to `reports/`. To check every number in the paper against them:

```bash
python final_check.py
```

## Structure

```
recsys/     pipeline: data, features, retrieval, ranking, evaluation
configs/    all tunable parameters
reports/    result files behind every reported number
paper/      LaTeX source and figures
```

## Data

- [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), Electronics category, downloaded by `setup_data.py`
- MovieLens 20M, used only for the density check, expected at `MovieLens/rating.csv`

Neither dataset is included in this repository.
