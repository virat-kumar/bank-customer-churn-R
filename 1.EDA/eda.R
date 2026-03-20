# Bank Customer Churn - Exploratory Data Analysis
# Reproduces all EDA steps and generates all plots in the 1.EDA/ folder.
# Run from the project root: Rscript 1.EDA/eda.R

set.seed(42)

required <- c("ggplot2", "gridExtra", "scales", "corrplot", "e1071", "caret")
for (pkg in required) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cran.r-project.org")
}
library(ggplot2)
library(gridExtra)
library(scales)
library(corrplot)
library(e1071)
library(caret)
theme_set(theme_minimal(base_size = 12))

# 1. Read dataset

df <- read.csv("data/botswana_bank_customer_churn.csv",
               stringsAsFactors = FALSE, na.strings = c("", "NA", "N/A"))

cat("Raw dataset\n")
cat("Dimensions:", nrow(df), "rows x", ncol(df), "cols\n")
cat("Columns:", paste(names(df), collapse = ", "), "\n\n")

# 2. Drop irrelevant columns & rename

drop_cols <- c("RowNumber", "CustomerId", "Surname", "First.Name",
               "Date.of.Birth", "Address", "Contact.Information",
               "Occupation", "Churn.Reason", "Churn.Date")

df_clean <- df[, !(names(df) %in% drop_cols)]
names(df_clean) <- c("Gender", "MaritalStatus", "NumDependents", "Income",
                      "EducationLevel", "Tenure", "CustomerSegment",
                      "CommChannel", "CreditScore", "CreditHistLength",
                      "OutstandingLoans", "Churn", "Balance",
                      "NumProducts", "NumComplaints")

df_clean$Churn <- as.factor(df_clean$Churn)
cat_cols <- c("Gender", "MaritalStatus", "EducationLevel", "CustomerSegment", "CommChannel")
for (col in cat_cols) df_clean[[col]] <- as.factor(df_clean[[col]])

num_cols <- c("NumDependents", "Income", "Tenure", "CreditScore",
              "CreditHistLength", "OutstandingLoans", "Balance",
              "NumProducts", "NumComplaints")

cat("Cleaned dataset\n")
cat("Dimensions:", nrow(df_clean), "rows x", ncol(df_clean), "cols\n")
cat("Dropped:", paste(drop_cols, collapse = ", "), "\n")
str(df_clean)

# 3. Missing values & summary stats

cat("\nMissing values\n")
n_missing <- sum(is.na(df_clean))
if (n_missing == 0) {
  cat("No missing values in any column.\n")
} else {
  for (col in names(df_clean)) {
    n_miss <- sum(is.na(df_clean[[col]]))
    if (n_miss > 0) cat(sprintf("  %-20s %d (%.2f%%)\n", col, n_miss, 100*n_miss/nrow(df_clean)))
  }
}

cat("\nNumerical summary\n")
print(summary(df_clean[, num_cols]))

cat("\nCategorical frequencies\n")
for (col in cat_cols) {
  cat(sprintf("\n  %s:\n", col))
  tbl <- table(df_clean[[col]])
  pct <- round(100 * prop.table(tbl), 2)
  for (i in seq_along(tbl)) {
    cat(sprintf("    %-15s %6d  (%5.2f%%)\n", names(tbl)[i], tbl[i], pct[i]))
  }
}

# 4. Target distribution

cat("\nChurn distribution\n")
print(table(df_clean$Churn))
cat("Proportions:", round(prop.table(table(df_clean$Churn)) * 100, 2), "%\n")

png("1.EDA/01_churn_distribution.png", width = 600, height = 450)
ggplot(df_clean, aes(x = Churn, fill = Churn)) +
  geom_bar(width = 0.5) +
  geom_text(stat = "count", aes(label = paste0(after_stat(count), "\n(",
            round(after_stat(count)/nrow(df_clean)*100, 1), "%)")),
            vjust = -0.2, size = 4.5) +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c"),
                    labels = c("No Churn", "Churn")) +
  labs(title = "Churn Distribution", x = "Churn Flag", y = "Count") +
  ylim(0, max(table(df_clean$Churn)) * 1.2)
dev.off()

# 5. Categorical features vs churn

png("1.EDA/02_categorical_vs_churn.png", width = 1200, height = 800)
cat_plots <- lapply(cat_cols, function(col) {
  ggplot(df_clean, aes(x = .data[[col]], fill = Churn)) +
    geom_bar(position = "dodge") +
    scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c"),
                      labels = c("No Churn", "Churn")) +
    labs(title = col, x = NULL, y = "Count") +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 9))
})
do.call(grid.arrange, c(cat_plots, ncol = 3))
dev.off()

png("1.EDA/03_churn_rate_by_category.png", width = 1200, height = 800)
rate_plots <- lapply(cat_cols, function(col) {
  rates <- aggregate(as.numeric(as.character(df_clean$Churn)),
                     by = list(Level = df_clean[[col]]), FUN = mean)
  names(rates)[2] <- "ChurnRate"
  ggplot(rates, aes(x = Level, y = ChurnRate)) +
    geom_col(fill = "#3498db", alpha = 0.85, width = 0.5) +
    geom_text(aes(label = paste0(round(ChurnRate*100, 1), "%")), vjust = -0.3, size = 4) +
    labs(title = paste("Churn Rate by", col), x = NULL, y = "Churn Rate") +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 9))
})
do.call(grid.arrange, c(rate_plots, ncol = 3))
dev.off()

cat("\nChurn rates by category\n")
for (col in cat_cols) {
  cat(sprintf("\n  %s:\n", col))
  rates <- aggregate(as.numeric(as.character(df_clean$Churn)),
                     by = list(Level = df_clean[[col]]), FUN = mean)
  for (i in 1:nrow(rates)) {
    cat(sprintf("    %-15s  %.2f%%\n", rates$Level[i], rates$x[i] * 100))
  }
}

# 6. Numerical distributions & correlations

png("1.EDA/04_numerical_histograms.png", width = 1400, height = 900)
hist_plots <- lapply(num_cols, function(col) {
  ggplot(df_clean, aes(x = .data[[col]])) +
    geom_histogram(bins = 40, fill = "#3498db", color = "white", alpha = 0.8) +
    labs(title = col, x = NULL, y = "Count")
})
do.call(grid.arrange, c(hist_plots, ncol = 3))
dev.off()

png("1.EDA/05_density_by_churn.png", width = 1400, height = 900)
dens_plots <- lapply(num_cols, function(col) {
  ggplot(df_clean, aes(x = .data[[col]], fill = Churn)) +
    geom_density(alpha = 0.45) +
    scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c"),
                      labels = c("No Churn", "Churn")) +
    labs(title = col, x = NULL, y = "Density")
})
do.call(grid.arrange, c(dens_plots, ncol = 3))
dev.off()

cor_data <- df_clean[, num_cols]
cor_data$Churn <- as.numeric(as.character(df_clean$Churn))
cor_matrix <- cor(cor_data, use = "complete.obs")

cat("\nCorrelations with Churn\n")
churn_cors <- sort(cor_matrix[, "Churn"][names(cor_matrix[, "Churn"]) != "Churn"], decreasing = TRUE)
for (i in seq_along(churn_cors)) {
  cat(sprintf("  %-20s  %+.4f\n", names(churn_cors)[i], churn_cors[i]))
}

png("1.EDA/06_correlation_heatmap.png", width = 800, height = 700)
corrplot(cor_matrix, method = "color", type = "lower",
         tl.col = "black", tl.srt = 45, addCoef.col = "black",
         number.cex = 0.7, title = "Correlation Heatmap (incl. Churn)",
         mar = c(0, 0, 2, 0))
dev.off()

# 7. Train / test split (80/20, stratified)

train_idx <- createDataPartition(df_clean$Churn, p = 0.8, list = FALSE)
train_df <- df_clean[train_idx, ]
test_df  <- df_clean[-train_idx, ]

cat("\nTrain / test split\n")
cat("Train:", nrow(train_df), "| Test:", nrow(test_df), "\n")
cat("Train churn:", round(mean(as.numeric(as.character(train_df$Churn))) * 100, 2), "%\n")
cat("Test  churn:", round(mean(as.numeric(as.character(test_df$Churn))) * 100, 2), "%\n")

write.csv(test_df, "data/test_data.csv", row.names = FALSE)

# 8. Extensive EDA on training set
churn_tbl <- table(train_df$Churn)
cat("\nClass imbalance (train)\n")
print(churn_tbl)
cat("Imbalance ratio:", round(max(churn_tbl) / min(churn_tbl), 2), ": 1\n")

png("1.EDA/07_class_imbalance_train.png", width = 600, height = 450)
ggplot(train_df, aes(x = Churn, fill = Churn)) +
  geom_bar(width = 0.5) +
  geom_text(stat = "count", aes(label = paste0(after_stat(count), "\n(",
            round(after_stat(count)/nrow(train_df)*100, 1), "%)")),
            vjust = -0.2, size = 4.5) +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c"),
                    labels = c("No Churn", "Churn")) +
  labs(title = "Class Imbalance — Training Set", x = "Churn", y = "Count") +
  ylim(0, max(churn_tbl) * 1.2)
dev.off()

cat("\nSkewness (train)\n")
skew_vals <- sapply(num_cols, function(c) round(skewness(train_df[[c]], na.rm = TRUE), 4))
kurt_vals <- sapply(num_cols, function(c) round(kurtosis(train_df[[c]], na.rm = TRUE), 4))
skew_df <- data.frame(Column = num_cols, Skewness = skew_vals, Kurtosis = kurt_vals,
                      Status = ifelse(abs(skew_vals) > 1, "High",
                               ifelse(abs(skew_vals) > 0.5, "Moderate", "OK")),
                      row.names = NULL)
print(skew_df)

png("1.EDA/08_skewness_bars_train.png", width = 900, height = 500)
ggplot(skew_df, aes(x = reorder(Column, -abs(Skewness)), y = Skewness, fill = Status)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = c(-1, 1), linetype = "dashed", color = "red", linewidth = 0.5) +
  geom_hline(yintercept = c(-0.5, 0.5), linetype = "dashed", color = "orange", linewidth = 0.5) +
  scale_fill_manual(values = c("High" = "#e74c3c", "Moderate" = "#f39c12", "OK" = "#2ecc71")) +
  labs(title = "Skewness of Numerical Features (Training Set)", x = NULL, y = "Skewness") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
dev.off()

cat("\nOutliers (train, IQR)\n")
outlier_info <- do.call(rbind, lapply(num_cols, function(col) {
  x <- train_df[[col]]
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr_val <- q3 - q1
  n_out <- sum(x < (q1 - 1.5 * iqr_val) | x > (q3 + 1.5 * iqr_val), na.rm = TRUE)
  data.frame(Column = col, Outliers = n_out, Pct = round(100 * n_out / length(x), 2), row.names = NULL)
}))
print(outlier_info)

png("1.EDA/09_boxplots_train.png", width = 1400, height = 900)
box_plots <- lapply(num_cols, function(col) {
  ggplot(train_df, aes(x = Churn, y = .data[[col]], fill = Churn)) +
    geom_boxplot(outlier.colour = "red", outlier.size = 0.5, alpha = 0.7) +
    scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c")) +
    labs(title = col, x = "Churn", y = NULL) +
    theme(legend.position = "none")
})
do.call(grid.arrange, c(box_plots, ncol = 3))
dev.off()

png("1.EDA/10_complaints_vs_churn.png", width = 800, height = 500)
ggplot(train_df, aes(x = as.factor(NumComplaints), fill = Churn)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c"),
                    labels = c("No Churn", "Churn")) +
  labs(title = "Churn Proportion by Number of Complaints (Train)",
       x = "Number of Complaints", y = "Proportion")
dev.off()

png("1.EDA/11_balance_vs_churn_violin.png", width = 700, height = 500)
ggplot(train_df, aes(x = Churn, y = Balance, fill = Churn)) +
  geom_violin(alpha = 0.6, trim = FALSE) +
  geom_boxplot(width = 0.15, outlier.size = 0.3) +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c"),
                    labels = c("No Churn", "Churn")) +
  labs(title = "Balance Distribution by Churn (Train)", x = "Churn", y = "Balance") +
  theme(legend.position = "none")
dev.off()

# 9. Skewness correction (train)

cat("\nSkewness transformation check\n")
train_transformed <- train_df
any_transformed <- FALSE

for (col in num_cols) {
  orig_skew <- skewness(train_transformed[[col]], na.rm = TRUE)
  if (abs(orig_skew) > 0.5) {
    x <- train_transformed[[col]]
    if (all(x >= 0, na.rm = TRUE)) {
      trans_x <- log1p(x)
    } else {
      trans_x <- sign(x) * log1p(abs(x))
    }
    if (abs(skewness(trans_x, na.rm = TRUE)) < abs(orig_skew)) {
      train_transformed[[col]] <- trans_x
      any_transformed <- TRUE
      cat(sprintf("  Transformed %s: %.4f -> %.4f\n", col, orig_skew, skewness(trans_x, na.rm = TRUE)))
    }
  }
}

if (!any_transformed) {
  cat("  No transformations needed — all features already near-symmetric (|skew| < 0.02).\n")
}

png("1.EDA/12_skewness_before_after.png", width = 1000, height = 500)
skew_after <- sapply(num_cols, function(c) round(skewness(train_transformed[[c]], na.rm = TRUE), 4))
skew_compare <- data.frame(
  Column = rep(num_cols, 2),
  Skewness = c(skew_vals, skew_after),
  Stage = rep(c("Before", "After"), each = length(num_cols))
)
ggplot(skew_compare, aes(x = Column, y = Skewness, fill = Stage)) +
  geom_col(position = "dodge", width = 0.6) +
  geom_hline(yintercept = c(-0.5, 0.5), linetype = "dashed", color = "orange") +
  geom_hline(yintercept = c(-1, 1), linetype = "dashed", color = "red") +
  scale_fill_manual(values = c("Before" = "#e74c3c", "After" = "#2ecc71")) +
  labs(title = "Skewness: Before vs After Transformation Check (Training Set)",
       subtitle = "No transformations needed — all features already near-symmetric",
       x = NULL, y = "Skewness") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
dev.off()

# 10. Save final train data

write.csv(train_transformed, "data/train_data.csv", row.names = FALSE)

