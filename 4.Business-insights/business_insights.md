# Business insights

This note translates project findings into **actionable banking and retention themes**. It complements **EDA** (`1.EDA/`), **modeling** (`2.Modeling/`), and **feature importance** (`3.Feature-importance/`).

Figures below are produced by `business_insights.R` (train-set summaries for communication charts).

## 1. Who churns versus who stays

Churners in this dataset differ most clearly on **account balance**, **complaint counts**, **credit score**, and **number of products**—not on broad demographics such as gender or segment, which show flat churn rates in EDA. That pattern suggests **relationship health and product depth** matter more than coarse customer labels.

![Churned vs retained profile](bi_churn_vs_retained_profile.png)

## 2. Complaints are a leading indicator

Churn rate rises steadily with the number of complaints. Operationally, **early complaint resolution** and **quality monitoring** are natural levers: each additional complaint band is associated with higher churn risk.

![Churn rate by complaints](bi_churn_rate_by_complaints.png)

## 3. Balance and engagement

Lower balances and fewer products align with higher churn in both correlation analysis and tree-based models. Business initiatives might combine **liquidity or fee support** for at-risk low-balance customers with **cross-sell and onboarding** to deepen product use—always subject to policy and fairness review.

## 4. Imbalance and decisions

Roughly **12%** churn means accuracy alone can hide poor capture of churners. For live decisions, define **costs** of false negatives versus false positives and tune thresholds or use class weights rather than fixing 0.5 as the only cutoff.

## 5. Modeling and monitoring

Use the **test-set** evaluation in `2.Modeling/` for headline metrics, and `3.Feature-importance/` to see which inputs drive the strongest classifier on this split. Refresh models as **data drift** or campaigns change customer behavior.

## 6. Synthetic data caveat

The Kaggle-style synthetic generator makes relationships clean and linear-friendly. **Real bank data** will add noise, seasonality, and compliance constraints; treat these insights as a **structured template** for analysis, not a production scorecard without validation.
