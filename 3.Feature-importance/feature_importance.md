# Feature importance

This folder scores **three models** on the **held-out test set** (`data/test_data.csv`) using the same 0.5 probability threshold as in `2.Modeling/`, compares **test accuracy**, and runs **feature importance only for the winning model**.

**How to re-run**

```text
conda activate bank-customer-churn-R
Rscript 3.Feature-importance/feature_importance.R
```

Run from the **project root**.

## Models compared

1. **Ridge logistic regression** (`glmnet`, `alpha = 0`, `lambda.min` from 5-fold CV on the training set only).
2. **Pruned decision tree** (`rpart`, complexity parameter chosen from cross-validated error).
3. **Random forest** (250 trees, same settings as `2.Modeling/tree_random_forest.R`).

**Selection rule:** pick the model with the **highest test accuracy** (ties broken by `which.max` order: ridge, then tree, then forest).

On the current split, results are also saved to `model_selection.txt` after each run.

## Figures

### Test accuracy comparison

![Model comparison](fi_model_accuracy_comparison.png)

### Importance ranking (winner model only)

The script uses:

- **Random forest:** mean decrease Gini from `randomForest::importance`.
- **Ridge logistic:** absolute fitted coefficients at `lambda.min` (model-matrix columns, including factor dummies).
- **Decision tree:** `rpart` variable importance (sum of impurity reductions; if the pruned tree has none, the unpruned fit is used as fallback).

![Feature ranking](fi_ranking.png)

### Cumulative share (top terms)

The cumulative chart uses the **top 20** terms from the ranking so the x-axis stays readable.

![Cumulative importance](fi_cumulative_importance.png)

## Interpretation

Importance is **relative to the chosen model**, not a causal statement. Use it to prioritize monitoring (e.g. balance, complaints) and to align with the correlation story in `1.EDA/eda.md`.
