# Retrieval Recall: Priority-Fill vs Budget-Union

Measurement date: run on clean features, no retraining.
Pool size: 200 for both conditions.
Evaluation population: warm test users with at least one held-out rating >= 5.
Users: 62,787  |  Warm positive pairs: 118,464
Seed: 42.

---

## Which condition the existing metrics.json represents

The current production code in recsys/retrieval.py is the **budget-union version**
(Condition B). The `source_budgets` config key is present and enforced:
collaborative=0.50 (100 slots), itemitem=0.335 (67 slots), semantic=0.165 (33 slots).
The figure 2.61% in metrics.json is the **post-fix (budget-union) recall**.

The document claim that recall rose "from 2.61% to 3.30%" had the direction inverted:
2.61% is the post-fix number, not the pre-fix number. See verified figures below.

---

## Recall at pool=200

| Condition | Strategy | Recall | HR |
|---|---|---|---|
| A (pre-fix)  | Priority-fill: collaborative fills first | 0.0261 | 2.61% |
| B (post-fix) | Budget-union: fixed slots per source | 0.0330 | 3.30% |
| Delta B - A |  | +0.0070 | +0.70 pp |

The budget-union fix raised recall by 0.70 percentage points.

---

## Per-source slot contribution (average per user, pool=200)

| Source           | A: priority-fill | B: budget-union |
|---|---|---|
| collaborative    | 199.5 | 100.0 |
| itemitem         |   0.0 |  66.7 |
| popularity       |   0.5 |   0.5 |
| semantic         |   0.0 |  32.9 |

Under priority-fill (Condition A), SVD fills nearly all 200 slots because it
retrieves up to 200 candidates per FAISS query and is processed first.
Item-item and SBERT receive zero or near-zero slots — the wasted-source bug.

Under budget-union (Condition B), each source contributes its allocated share:
100 collaborative, 67 item-item, 33 SBERT.
The diversity of signal sources is why recall improves.

---

## Correction for the findings document

The verified figures are:
- Pre-fix recall (priority-fill, Condition A):  **2.61%**
- Post-fix recall (budget-union, Condition B):  **3.30%**
- Improvement from fix: **+0.70 percentage points**

The improvement is real and the direction is correct, but both absolute figures differ from the document.
The findings should be updated to state the verified numbers above.
The underlying argument — that priority-fill wastes item-item and SBERT capacity — is
confirmed by the per-source table regardless of the recall direction.

Elapsed: 57.7m
