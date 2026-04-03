# Feature importance: pick best test accuracy among ridge logistic, decision tree, RF
#
# Run from project root:
#   conda activate bank-customer-churn-R
#   Rscript 3.Feature-importance/feature_importance.R

set.seed(42)
Sys.setenv(TZ = "UTC")

pkgs <- c("ggplot2", "caret", "glmnet", "rpart", "randomForest", "scales")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p, repos = "https://cran.r-project.org")
}
library(ggplot2)
library(caret)
library(glmnet)
library(rpart)
library(randomForest)
library(scales)

out_dir <- "3.Feature-importance"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

train <- read.csv("data/train_data.csv", stringsAsFactors = FALSE)
test  <- read.csv("data/test_data.csv", stringsAsFactors = FALSE)

cat_cols <- c("Gender", "MaritalStatus", "EducationLevel", "CustomerSegment", "CommChannel")
for (col in cat_cols) {
  train[[col]] <- factor(train[[col]])
  test[[col]]  <- factor(test[[col]], levels = levels(train[[col]]))
}
train$Churn <- factor(train$Churn, levels = c("0", "1"))
test$Churn  <- factor(test$Churn, levels = c("0", "1"))

x_train <- model.matrix(Churn ~ ., data = train)[, -1L, drop = FALSE]
y_train <- as.integer(train$Churn) - 1L
x_test  <- model.matrix(Churn ~ ., data = test)[, -1L, drop = FALSE]

cv_glm <- cv.glmnet(
  x_train, y_train,
  family = "binomial",
  alpha = 0,
  nfolds = 5L,
  type.measure = "auc",
  standardize = TRUE
)
prob_lr <- as.numeric(predict(cv_glm, newx = x_test, s = "lambda.min", type = "response"))
pred_lr <- factor(ifelse(prob_lr >= 0.5, "1", "0"), levels = c("0", "1"))
cm_lr <- confusionMatrix(pred_lr, test$Churn, positive = "1")
acc_lr <- as.numeric(cm_lr$overall["Accuracy"])

fit_tree <- rpart(Churn ~ ., data = train, method = "class", cp = 1e-4,
                  parms = list(split = "information"))
best_cp <- fit_tree$cptable[which.min(fit_tree$cptable[, "xerror"]), "CP"]
fit_dt <- prune(fit_tree, cp = best_cp)
prob_dt <- predict(fit_dt, test, type = "prob")[, "1"]
pred_dt <- factor(ifelse(prob_dt >= 0.5, "1", "0"), levels = c("0", "1"))
cm_dt <- confusionMatrix(pred_dt, test$Churn, positive = "1")
acc_dt <- as.numeric(cm_dt$overall["Accuracy"])

fit_rf <- randomForest(
  Churn ~ .,
  data = train,
  ntree = 250,
  importance = TRUE,
  nodesize = 80,
  mtry = 4
)
prob_rf <- predict(fit_rf, test, type = "prob")[, "1"]
pred_rf <- factor(ifelse(prob_rf >= 0.5, "1", "0"), levels = c("0", "1"))
cm_rf <- confusionMatrix(pred_rf, test$Churn, positive = "1")
acc_rf <- as.numeric(cm_rf$overall["Accuracy"])

models <- c("Ridge logistic", "Decision tree", "Random forest")
accs <- c(acc_lr, acc_dt, acc_rf)
cmp <- data.frame(Model = factor(models, levels = models), Test_accuracy = accs)

cat("\n=== Test accuracy (threshold 0.5, positive class = churn) ===\n")
print(cmp)

win_idx <- which.max(accs)
winner <- models[win_idx]
cat("\nSelected model for feature importance:", winner, "\n")

writeLines(
  c(
    paste("Ridge logistic test accuracy:", round(acc_lr, 6)),
    paste("Decision tree test accuracy:", round(acc_dt, 6)),
    paste("Random forest test accuracy:", round(acc_rf, 6)),
    paste("Winner:", winner)
  ),
  file.path(out_dir, "model_selection.txt")
)

p_cmp <- ggplot(cmp, aes(x = Model, y = Test_accuracy, fill = Model)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  geom_text(aes(label = percent(Test_accuracy, accuracy = 0.01)), vjust = -0.4, size = 4) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.05)) +
  scale_fill_manual(values = c("Ridge logistic" = "#2980b9", "Decision tree" = "#d35400", "Random forest" = "#27ae60")) +
  labs(
    title = "Test-set accuracy: model comparison",
    subtitle = "Higher bar wins; that model is used for feature importance below",
    x = NULL,
    y = "Accuracy"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
ggsave(file.path(out_dir, "fi_model_accuracy_comparison.png"), p_cmp, width = 8, height = 5.5, dpi = 150)

fi_df <- NULL
if (winner == "Random forest") {
  imp <- importance(fit_rf)
  fi_df <- data.frame(
    Feature = rownames(imp),
    Score = imp[, "MeanDecreaseGini"],
    stringsAsFactors = FALSE
  )
  fi_df <- fi_df[order(fi_df$Score, decreasing = TRUE), ]
  fi_df$Feature <- reorder(fi_df$Feature, fi_df$Score)
  p_fi <- ggplot(fi_df, aes(x = Feature, y = Score)) +
    geom_col(fill = "#27ae60", alpha = 0.9, width = 0.65) +
    coord_flip() +
    labs(
      title = "Feature importance: Random forest (Mean decrease Gini)",
      subtitle = "Selected model (best test accuracy)",
      x = NULL,
      y = "Mean decrease Gini"
    ) +
    theme_minimal(base_size = 12)

} else if (winner == "Ridge logistic") {
  beta <- as.matrix(coef(cv_glm, s = "lambda.min"))
  fi_df <- data.frame(
    Feature = rownames(beta)[-1],
    Score = abs(as.numeric(beta[-1, 1])),
    stringsAsFactors = FALSE
  )
  fi_df <- fi_df[order(fi_df$Score, decreasing = TRUE), ]
  fi_df$Feature <- reorder(fi_df$Feature, fi_df$Score)
  p_fi <- ggplot(fi_df, aes(x = Feature, y = Score)) +
    geom_col(fill = "#2980b9", alpha = 0.9, width = 0.65) +
    coord_flip() +
    labs(
      title = "Feature importance: ridge logistic (|coefficient| at lambda.min)",
      subtitle = "Dummy-coded factor levels; selected model (best test accuracy)",
      x = NULL,
      y = "|Coefficient|"
    ) +
    theme_minimal(base_size = 11)

} else {
  vi <- fit_dt$variable.importance
  if (length(vi) == 0) {
    vi <- fit_tree$variable.importance
  }
  fi_df <- data.frame(
    Feature = names(vi),
    Score = as.numeric(vi),
    stringsAsFactors = FALSE
  )
  fi_df <- fi_df[order(fi_df$Score, decreasing = TRUE), ]
  fi_df$Feature <- reorder(fi_df$Feature, fi_df$Score)
  p_fi <- ggplot(fi_df, aes(x = Feature, y = Score)) +
    geom_col(fill = "#d35400", alpha = 0.9, width = 0.65) +
    coord_flip() +
    labs(
      title = "Feature importance: decision tree (rpart variable importance)",
      subtitle = "Selected model (best test accuracy)",
      x = NULL,
      y = "Importance"
    ) +
    theme_minimal(base_size = 12)
}

ggsave(file.path(out_dir, "fi_ranking.png"), p_fi, width = 8.5, height = 6.5, dpi = 150)

imp_top <- head(fi_df, min(20L, nrow(fi_df)))
imp_top <- imp_top[order(imp_top$Score, decreasing = TRUE), ]
imp_top$Share <- imp_top$Score / sum(imp_top$Score)
imp_top$cum_share <- cumsum(imp_top$Share)
imp_top$Feature <- factor(imp_top$Feature, levels = imp_top$Feature)

p_cum <- ggplot(imp_top, aes(x = Feature, y = cum_share, group = 1)) +
  geom_line(color = "#2c3e50", linewidth = 1) +
  geom_point(size = 2, color = "#c0392b") +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "Cumulative share of importance (top terms, winner model)",
    subtitle = paste0("Based on top ", nrow(imp_top), " entries in fi_ranking"),
    x = NULL,
    y = "Cumulative share"
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(out_dir, "fi_cumulative_importance.png"), p_cum, width = 9, height = 5.5, dpi = 150)

cat("\nFigures written to", out_dir, "\n")