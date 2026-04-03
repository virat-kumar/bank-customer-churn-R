# Bank customer churn - ridge logistic regression (glmnet) + test evaluation
#
# Plain glm() hits quasi-complete separation on this data (non-convergence).
# Ridge (alpha=0) logistic regression: same linear logit model with L2 penalty;
# lambda chosen by 5-fold CV (AUC) on the training set.
#
# Run from project root:
#   conda activate bank-customer-churn-R
#   Rscript 2.Modeling/logistic_regression.R

set.seed(42)
Sys.setenv(TZ = "UTC")

pkgs <- c("ggplot2", "caret", "pROC", "scales", "glmnet")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p, repos = "https://cran.r-project.org")
}
library(ggplot2)
library(caret)
library(pROC)
library(scales)
library(glmnet)

out_dir <- "2.Modeling"
train <- read.csv("data/train_data.csv", stringsAsFactors = FALSE)
test  <- read.csv("data/test_data.csv", stringsAsFactors = FALSE)

cat_cols <- c("Gender", "MaritalStatus", "EducationLevel", "CustomerSegment", "CommChannel")
for (col in cat_cols) {
  train[[col]] <- factor(train[[col]])
  test[[col]]  <- factor(test[[col]], levels = levels(train[[col]]))
}
train$Churn <- factor(train$Churn, levels = c("0", "1"))
test$Churn  <- factor(test$Churn, levels = c("0", "1"))

cat("Train rows:", nrow(train), "| Test rows:", nrow(test), "\n")

x_train <- model.matrix(Churn ~ ., data = train)[, -1L, drop = FALSE]
y_train <- as.integer(train$Churn) - 1L
x_test  <- model.matrix(Churn ~ ., data = test)[, -1L, drop = FALSE]

cv_fit <- cv.glmnet(
  x_train, y_train,
  family = "binomial",
  alpha = 0,
  nfolds = 5L,
  type.measure = "auc",
  standardize = TRUE
)
cat("\n=== cv.glmnet: lambda.min =", cv_fit$lambda.min, "===\n")

png(file.path(out_dir, "lr_cv_glmnet.png"), width = 720, height = 520, res = 120)
plot(cv_fit)
title(main = "Ridge logistic: CV AUC vs log(lambda)")
dev.off()

prob_test <- as.numeric(predict(cv_fit, newx = x_test, s = "lambda.min", type = "response"))
pred_class <- factor(ifelse(prob_test >= 0.5, "1", "0"), levels = c("0", "1"))
cm <- confusionMatrix(pred_class, test$Churn, positive = "1")
cat("\n=== Confusion matrix (test, threshold 0.5) ===\n")
print(cm)

roc_obj <- roc(
  response = test$Churn,
  predictor = prob_test,
  levels = c("0", "1"),
  direction = "<",
  quiet = TRUE
)
auc_val <- as.numeric(auc(roc_obj))
cat("\nTest AUC (ROC):", round(auc_val, 4), "\n")

roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "#2980b9", linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  labs(
    title = "Ridge logistic regression - ROC curve (test set)",
    subtitle = paste0("AUC = ", round(auc_val, 4)),
    x = "False positive rate (1 - specificity)",
    y = "True positive rate (sensitivity)"
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)
ggsave(file.path(out_dir, "lr_roc_curve.png"), p_roc, width = 7, height = 6, dpi = 150)

tab <- as.table(cm$table)
cm_df <- as.data.frame(tab)
colnames(cm_df) <- c("Reference", "Prediction", "Freq")
p_cm <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 8) +
  scale_fill_gradient(low = "#34495e", high = "#e74c3c") +
  labs(
    title = "Ridge logistic regression - confusion matrix (test)",
    subtitle = paste0(
      "Accuracy ", scales::percent(cm$overall["Accuracy"], accuracy = 0.1),
      " | Kappa ", round(cm$overall["Kappa"], 3)
    ),
    x = "Actual churn",
    y = "Predicted churn"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
ggsave(file.path(out_dir, "lr_confusion_matrix.png"), p_cm, width = 6.5, height = 5.5, dpi = 150)

beta_mat <- as.matrix(coef(cv_fit, s = "lambda.min"))
coef_df <- data.frame(term = rownames(beta_mat), beta = beta_mat[, 1], row.names = NULL)
coef_df <- coef_df[coef_df$term != "(Intercept)", , drop = FALSE]
coef_df$OR <- exp(coef_df$beta)
coef_df <- coef_df[is.finite(coef_df$OR) & coef_df$OR > 0, , drop = FALSE]
coef_df$label <- reorder(coef_df$term, coef_df$OR)

p_or <- ggplot(coef_df, aes(x = label, y = OR)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray40") +
  geom_col(fill = "#16a085", alpha = 0.85, width = 0.65) +
  coord_flip() +
  scale_y_log10() +
  labs(
    title = "Ridge logistic - odds ratios at lambda.min",
    x = NULL,
    y = "Odds ratio (log scale)"
  ) +
  theme_minimal(base_size = 11)
ggsave(file.path(out_dir, "lr_odds_ratios.png"), p_or, width = 8, height = 7, dpi = 150)

cat("\nFigures written to", out_dir, "\n")