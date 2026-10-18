# Cold-Start Supplementary Experiment — Extended Report

All numbers here are supplementary to the main dissertation results.
This experiment wrote only to data/coldstart/ and reports/coldstart/
and did not modify any main pipeline artefact.

---

## 1. Scale of the Cold-Start Problem

The k_core=5 filter that defines the warm training set removes the majority of
the raw catalogue. Cold-start is not an edge case — it is the dominant regime.

| Entity | Raw count | Warm (k_core=5) | Cold | Cold % |
|--------|-----------|-----------------|------|--------|
| Users | 378,510 | 79,838 | 263,723 | 69.7% |
| Items | 401,056 | 48,154 | 332,471 | 82.9% |

Of the 332,471 cold items, 144,609 (36.1% of the raw catalogue)
have zero interactions with any warm user and are therefore completely untestable
from interaction data. Only 187,862 cold items (56.5% of cold items)
had at least one warm-user rating that could serve as a test signal.

---

## 2. Dataset Construction

Cold items: raw interaction count in [1, 4] AND absent from the k_core=5 clean set.
Cold users: raw interaction count in [1, 4] AND absent from the k_core=5 clean set.

| Entity | Count |
|--------|-------|
| Cold items total | 332,471 |
| Cold items with at least one warm-user pair (testable) | 187,862 |
| Truly cold items (no warm-user pair — zero test signal) | 144,609 |
| Cold users total | 263,723 |
| Warm-user / cold-item test pairs (rating >= 4) | 3,919 |

Leakage check: cold entities were identified from raw data outside the main
training split. No cold item or cold user appears in data/clean.parquet.
Item popularity used for scoring comes from training rows only.

---

## 3. Cold-Item Retrieval — Headline Results

Question: can the semantic signal retrieve cold items that interaction-based signals cannot?

Warm users who rated a cold item (rating >= 4) form the test pairs.
Protocol: sampled negatives — 99 random cold items per positive, top-10.
Pool size: 3,000 cold items (sampled with seed=42 from 187,862 testable).

| Signal | HR@10 | NDCG@10 | MRR | n pairs |
|--------|-------|---------|-----|---------|
| SBERT (semantic) | 0.1967 | 0.0957 | 0.0891 | 3,919 |
| SVD (collaborative) | N/A — cold items have no latent factors | | | |
| Item-Item CF | N/A — cold items have no co-rating history | | | |

Random baseline (sampled negatives, k=10, pool=100): 0.1000.

SBERT achieves HR@10=0.1967, 97% above random.
SVD and item-item CF score exactly zero on cold items by design: without any
interaction history there are no latent factors and no co-rated neighbours.
SBERT is the only signal that can retrieve cold items at all, and it does so
meaningfully.

---

## 4. Cold-Item Retrieval — by Interaction Count

Items with more raw interactions have more review text available for embedding.
The table below tests whether this helps SBERT retrieval quality.

| Item raw count | HR@10 | NDCG@10 | MRR | n pairs |
|---------------|-------|---------|-----|---------|
| 1 | 0.1807 | 0.0890 | 0.0847 | 1,566 |
| 2 | 0.2132 | 0.1055 | 0.0968 | 1,032 |
| 3 | 0.1992 | 0.0942 | 0.0873 | 798 |
| 4 | 0.2084 | 0.0988 | 0.0896 | 523 |

Random baseline: 0.1000.

---

## 5. Cold-User Recommendation — Headline Results

Question: can we recommend for users with almost no interaction history?

Cold users with >= 2 raw interactions, leave-one-out split
(hold out highest-rated item as test). History items must be warm
(in k_core=5) to have an embedding.
Protocol: sampled negatives — 99 random warm items per positive, top-10.
Pool size: 48,154 warm items.

| Signal | HR@10 | NDCG@10 | MRR | n pairs |
|--------|-------|---------|-----|---------|
| SBERT avg-history profile | 0.1563 | 0.0804 | 0.0807 | 2,815 |
| Popularity baseline | 0.5869 | 0.3995 | 0.3571 | 2,815 |

Random baseline: 0.1000.

Popularity (HR@10=0.5869) beats SBERT profiles (HR@10=0.1563) by a ratio
of 3.8x. With only 1-4 interactions the average embedding vector
is dominated by noise — each additional item shifts the mean substantially,
so the profile does not converge to a stable preference direction.
A flat popularity prior, which encodes global frequency without any user signal,
is more reliable at this sparsity level.
Note that SBERT still beats random (0.10) by 56%, so it is not useless;
it simply loses to a very strong simple baseline.

---

## 6. Cold-User Recommendation — by Interaction Count

Does having more history interactions improve the SBERT profile quality?

| Interactions | SBERT HR@10 | Pop HR@10 | n evaluated | n skipped |
|-------------|-------------|-----------|-------------|-----------|
| 2 | 0.1492 | 0.5941 | 3,439 | 4,231 |
| 3 | 0.1621 | 0.5539 | 2,979 | 1,997 |
| 4 | 0.1558 | 0.5547 | 2,214 | 1,205 |

SBERT profile quality is roughly flat across 2-4 interactions (0.1492 to 0.1558), indicating that even the jump from 2 to 4 interactions is insufficient to stabilise the preference profile.

The skipped column counts cold users whose test item or entire history was also
cold (not in the warm k_core=5 set). These 7,433 users are a
second-order cold-start case: they interacted only with other cold items, making
it impossible to build any embedding-based profile. No system can serve these
users from content alone without additional context such as session signals
or demographic priors.

---

## 7. Budget Allocation Argument

In the main two-stage pipeline, SBERT receives 16.5% of the retrieval candidate
budget. For warm items this is a minority contribution. For the 83%
of the catalogue that is cold, SBERT is the only signal available — SVD and
item-item CF contribute zero candidates for these items by design.

A natural extension is an adaptive budget that raises the SBERT share when the
candidate pool is known to contain cold items (for example, when a query item
is absent from the warm catalogue, or when the retrieval stage detects that
collaborative sources returned very few candidates).

---

## 8. Recommendations for Future Work

Cold items:
- The current SBERT signal already retrieves cold items at HR@10=0.1967, well
  above random. Including SBERT in the retrieval union is already the right call.
- Richer item representations (product category, price tier, brand embeddings)
  could strengthen the signal for the 144K truly cold items that have no
  warm-user test pairs at all.

Cold users:
- With 1-4 interactions, popularity is a hard baseline to beat. A hybrid of
  popularity prior and a sparse SBERT profile (weighted by interaction count)
  would likely narrow the gap.
- Session-level signals (click sequence, dwell time) would provide richer
  context than rating counts alone and are the standard industry approach
  for cold-user personalisation.

See fig26_coldstart_warmup.png for the visual summary.
