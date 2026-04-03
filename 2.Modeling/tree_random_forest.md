# Modeling: Decision Tree and Random Forest

This note summarizes two related **tree-based** models trained on `data/train_data.csv` and evaluated on `data/test_data.csv` (same split as EDA).

**How to re-run**

```text
conda activate bank-customer-churn-R
Rscript 2.Modeling/tree_random_forest.R
```

Run from the **project root**.

## Decision tree (rpart)

- A classification tree is grown with `rpart`, using cross-validated error (`plotcp`) to pick a **pruning** complexity parameter.
- The pruned tree is plotted for interpretability (`rpart.plot`).
- Test-set predictions use the **estimated probability of class 1** with a **0.5** threshold for the confusion matrix.

![CP vs error](dt_plotcp.png)

![Pruned tree](dt_rpart_structure.png)

![Decision tree confusion matrix](dt_confusion_matrix.png)

## Random forest

- `randomForest` is fit with **250** trees, `mtry = 4`, and moderate `nodesize` for stability on ~92k training rows.
- **Variable importance** (mean decrease in Gini) shows which features contribute most to splits across trees.

![Variable importance](rf_variable_importance.png)

![Random forest confusion matrix](rf_confusion_matrix.png)

## ROC comparison (test)

Both models are scored on the test set and compared on the same ROC plot.

![ROC comparison](dt_rf_roc_comparison.png)

## Practical notes

- The single tree is easy to explain; the forest usually **improves AUC** and smooths variance at the cost of interpretability.
- Like logistic regression, the default **0.5** threshold may not match business costs; adjust after reviewing sensitivity/specificity.

