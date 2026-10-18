# Leakage Check Report

Overall: PASS

- train_test_disjoint: PASS
- oof_only_train_rows: PASS
- item_popularity_from_train_only: PASS

A PASS on all three checks confirms that no test row informed any
feature it depends on and that the split contract held throughout.