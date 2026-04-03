# Bank customer churn - decision tree (rpart) and random forest
#
# Run from project root:
#   conda activate bank-customer-churn-R
#   Rscript 2.Modeling/tree_random_forest.R

set.seed(42)
Sys.setenv(TZ = "UTC")

pkgs <- c("ggplot2", "caret", "pROC", "scales", "rpart", "rpart.plot", "randomForest")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p, repos = "https://cran.r-project.org")
}
library(ggplot2)
library(caret)
library(pROC)
library(scales)
library(rpart)
library(rpart.plot)
library(randomForest)

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

# ----- Decision tree (rpart) -----
fit_tree <- rpart(Churn ~ ., data = train, method = "class", cp = 1e-4,
                  parms = list(split = "information"))
best_cp <- fit_tree$cptable[which.min(fit_tree$cptable[, "xerror"]), "CP"]
fit_pruned <- prune(fit_tree, cp = best_cp)
cat("\n=== Decision tree: chosen cp =", best_cp, "===\n")

png(file.path(out_dir, "dt_plotcp.png"), width = 720, height = 520, res = 120)
plotcp(fit_tree)
title(main = "Decision tree - cross-validated error vs cp")
dev.off()

png(file.path(out_dir, "dt_rpart_structure.png"), width = 1000, height = 700, res = 120)
rpart.plot(fit_pruned, extra = 104, fallen.leaves = TRUE,
           main = "Pruned decision tree (churn = 1)")
dev.off()

prob_dt <- predict(fit_pruned, test, type = "prob")[, "1"]
pred_dt <- factor(ifelse(prob_dt >= 0.5, "1", "0"), levels = c("0", "1"))
cm_dt <- confusionMatrix(pred_dt, test$Churn, positive = "1")
cat("\n=== Decision tree - confusion matrix (test) ===\n")
print(cm_dt)

roc_dt <- roc(test$Churn, prob_dt, levels = c("0", "1"), direction = "<", quiet = TRUE)
auc_dt <- as.numeric(auc(roc_dt))

# ----- Random forest -----
fit_rf <- randomForest(
  Churn ~ .,
  data = train,
  ntree = 250,
  importance = TRUE,
  nodesize = 80,
  mtry = 4,
  keep.forest = TRUE
)
cat("\n=== Random forest OOB error ===\n")
print(fit_rf)

prob_rf <- predict(fit_rf, test, type = "prob")[, "1"]
pred_rf <- factor(ifelse(prob_rf >= 0.5, "1", "0"), levels = c("0", "1"))
cm_rf <- confusionMatrix(pred_rf, test$Churn, positive = "1")
cat("\n=== Random forest - confusion matrix (test) ===\n")
print(cm_rf)

roc_rf <- roc(test$Churn, prob_rf, levels = c("0", "1"), direction = "<", quiet = TRUE)
auc_rf <- as.numeric(auc(roc_rf))

cat("\nTest AUC - Decision tree:", round(auc_dt, 4), "| Random forest:", round(auc_rf, 4), "\n")

imp <- importance(fit_rf)
imp_df <- data.frame(
  Feature = rownames(imp),
  MeanDecreaseGini = imp[, "MeanDecreaseGini"]
)
imp_df <- imp_df[order(imp_df$MeanDecreaseGini, decreasing = TRUE), ]
imp_df$Feature <- reorder(imp_df$Feature, imp_df$MeanDecreaseGini)

p_imp <- ggplot(imp_df, aes(x = Feature, y = MeanDecreaseGini)) +
  geom_col(fill = "#8e44ad", alpha = 0.85, width = 0.65) +
  coord_flip() +
  labs(
    title = "Random forest - variable importance (Mean decrease Gini)",
    x = NULL,
    y = "Mean decrease Gini"
  ) +
  theme_minimal(base_size = 12)
ggsave(file.path(out_dir, "rf_variable_importance.png"), p_imp, width = 8, height = 6, dpi = 150)

roc_dt_df <- data.frame(
  FPR = 1 - roc_dt$specificities,
  TPR = roc_dt$sensitivities,
  Model = "Decision tree"
)
roc_rf_df <- data.frame(
  FPR = 1 - roc_rf$specificities,
  TPR = roc_rf$sensitivities,
  Model = "Random forest"
)
roc_cmp <- rbind(roc_dt_df, roc_rf_df)

p_roc_cmp <- ggplot(roc_cmp, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  scale_color_manual(values = c("Decision tree" = "#d35400", "Random forest" = "#27ae60")) +
  labs(
    title = "ROC curves on test set",
    subtitle = paste0(
      "AUC tree = ", round(auc_dt, 4),
      " | AUC RF = ", round(auc_rf, 4)
    ),
    x = "False positive rate",
    y = "True positive rate"
  ) +
  coord_equal() +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")
ggsave(file.path(out_dir, "dt_rf_roc_comparison.png"), p_roc_cmp, width = 7.5, height = 6.5, dpi = 150)

tab_dt <- as.data.frame(as.table(cm_dt$table))
colnames(tab_dt) <- c("Reference", "Prediction", "Freq")
p_cmdt <- ggplot(tab_dt, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 8) +
  scale_fill_gradient(low = "#34495e", high = "#e67e22") +
  labs(
    title = "Decision tree - confusion matrix (test)",
    subtitle = paste0(
      "Acc ", scales::percent(cm_dt$overall["Accuracy"], accuracy = 0.1),
      " | Sens ", scales::percent(cm_dt$byClass["Sensitivity"], accuracy = 0.1)
    ),
    x = "Actual churn",
    y = "Predicted churn"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
ggsave(file.path(out_dir, "dt_confusion_matrix.png"), p_cmdt, width = 6.5, height = 5.5, dpi = 150)

tab_rf <- as.data.frame(as.table(cm_rf$table))
colnames(tab_rf) <- c("Reference", "Prediction", "Freq")
p_cmrf <- ggplot(tab_rf, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 8) +
  scale_fill_gradient(low = "#34495e", high = "#27ae60") +
  labs(
    title = "Random forest - confusion matrix (test)",
    subtitle = paste0(
      "Acc ", scales::percent(cm_rf$overall["Accuracy"], accuracy = 0.1),
      " | Sens ", scales::percent(cm_rf$byClass["Sensitivity"], accuracy = 0.1)
    ),
    x = "Actual churn",
    y = "Predicted churn"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
ggsave(file.path(out_dir, "rf_confusion_matrix.png"), p_cmrf, width = 6.5, height = 5.5, dpi = 150)

cat("\nFigures written to", out_dir, "\n")
