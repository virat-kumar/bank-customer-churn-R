# Modeling: Logistic Regression (ridge / glmnet)

This note summarizes a **ridge logistic regression** model for churn (`Churn` = 1). We use `glmnet` with `alpha = 0` (L2 penalty) because ordinary `glm()` on this dataset shows **quasi-complete separation** (MLE does not converge; coefficients and odds ratios are unstable). Ridge keeps the same **linear logit** structure; only the estimation method changes.

Training data: `data/train_data.csv`. Test data: `data/test_data.csv` (same stratified split as `1.EDA/eda.R`).

**How to re-run**

```text
conda activate bank-customer-churn-R
Rscript 2.Modeling/logistic_regression.R
```

Run from the **project root**.

## What the script does

- Builds the design matrix with `model.matrix(Churn ~ ., ...)`.
- Runs `cv.glmnet(..., family = "binomial", alpha = 0)` with **5-fold CV** and `type.measure = "auc"`.
- Selects **`lambda.min`** (best CV AUC), refits conceptually at that lambda, scores the test set.
- Uses threshold **0.5** for the confusion matrix; reports **AUC** on the test ROC curve.

## Figures (saved under `2.Modeling/`)

### Cross-validated AUC vs penalty (training)

![CV glmnet](lr_cv_glmnet.png)

### ROC curve (test)

![ROC curve](lr_roc_curve.png)

### Confusion matrix (test)

![Confusion matrix](lr_confusion_matrix.png)

### Odds ratios at lambda.min

Exponentiated coefficients at the chosen lambda (interpret as approximate multiplicative effects on odds per unit, subject to regularization shrinkage).

![Odds ratios](lr_odds_ratios.png)

## Notes

- For deployment, tune the **probability threshold** if false negatives/positives have unequal cost.
- For a sparser linear model, you could try **elastic net** (`alpha` between 0 and 1) in the same script.
