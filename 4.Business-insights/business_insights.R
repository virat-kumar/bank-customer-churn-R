# Business-oriented visuals derived from the churn project (training data summary)
#
# Run from project root:
#   conda activate bank-customer-churn-R
#   Rscript 4.Business-insights/business_insights.R

set.seed(42)
Sys.setenv(TZ = "UTC")

if (!requireNamespace("ggplot2", quietly = TRUE))
  install.packages("ggplot2", repos = "https://cran.r-project.org")
if (!requireNamespace("scales", quietly = TRUE))
  install.packages("scales", repos = "https://cran.r-project.org")
library(ggplot2)
library(scales)

out_dir <- "4.Business-insights"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

train <- read.csv("data/train_data.csv", stringsAsFactors = FALSE)
train$Churn_lab <- factor(ifelse(train$Churn == "1", "Churned", "Retained"), levels = c("Retained", "Churned"))

vars <- c("Balance", "NumComplaints", "CreditScore", "NumProducts")
agg <- aggregate(train[vars], by = list(Churn = train$Churn_lab), FUN = mean)
rownames(agg) <- NULL

m <- data.frame(
  Churn = rep(agg$Churn, length(vars)),
  Metric = rep(vars, each = nrow(agg)),
  Value = c(agg$Balance, agg$NumComplaints, agg$CreditScore, agg$NumProducts)
)
m$Metric <- factor(m$Metric, levels = vars)

p1 <- ggplot(m, aes(x = Metric, y = Value, fill = Churn)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  facet_wrap(~ Metric, scales = "free", nrow = 1) +
  scale_fill_manual(values = c("Retained" = "#2ecc71", "Churned" = "#e74c3c")) +
  labs(
    title = "Average profile: churned vs retained customers (train set)",
    subtitle = "Strongest separation on Balance and complaints (see EDA correlations)",
    x = NULL,
    y = NULL,
    fill = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top", axis.text.x = element_blank(), axis.ticks.x = element_blank())
ggsave(file.path(out_dir, "bi_churn_vs_retained_profile.png"), p1, width = 10, height = 4.5, dpi = 150)

train$NumComplaints_f <- factor(train$NumComplaints, levels = 0:10)
r <- aggregate(
  as.integer(as.character(train$Churn)),
  by = list(Complaints = train$NumComplaints_f),
  FUN = mean
)
names(r)[2] <- "Churn_rate"

p2 <- ggplot(r, aes(x = Complaints, y = Churn_rate, group = 1)) +
  geom_line(color = "#8e44ad", linewidth = 1) +
  geom_point(size = 3, color = "#8e44ad") +
  scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
  labs(
    title = "Churn rate by number of complaints (train set)",
    subtitle = "Monotone pattern: complaint handling is a lever for retention",
    x = "Number of complaints",
    y = "Churn rate"
  ) +
  theme_minimal(base_size = 12)
ggsave(file.path(out_dir, "bi_churn_rate_by_complaints.png"), p2, width = 8, height = 5, dpi = 150)

cat("Figures written to", out_dir, "\n")