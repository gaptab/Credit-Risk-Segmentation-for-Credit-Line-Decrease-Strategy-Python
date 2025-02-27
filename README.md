![alt text](https://github.com/gaptab/Credit-Risk-Segmentation-for-Credit-Line-Decrease-Strategy-Python/blob/main/489.png)

**Data Preparation & Feature Engineering**

Generated  data for 5000 customers with relevant attributes such as:

Credit Score: Indicator of financial reliability (300-850).

Income & Total Debt: Determines financial capability.

Default History & Missed Payments: Signals risky behavior.

Utilization Ratio: Measures how much of the available credit is used.

Account Age & Existing Credit Lines: Provides customer history.

Created new features:

Risk Score: A calculated metric that accounts for credit score, missed payments, and utilization ratio.

Profitability Score: A measure based on income, total debt, and credit usage.

**Customer Segmentation using Machine Learning**

Applied K-Means Clustering to segment customers into 4 categories:

Low Risk - High Profit

Medium Risk - Medium Profit

High Risk - Low Profit

Very High Risk - Negative Profit

Why Segmentation?

Helps in identifying the right customers for a credit limit decrease.

Ensures that high-value, low-risk customers are not negatively impacted.

**Credit Line Decrease (CLD) Strategy**

Business rules were defined for reducing credit limits based on customer risk categories:

Very High-Risk Customers: 50% reduction in credit limit to minimize losses.

High-Risk Customers: 20-30% reduction depending on account age.

Medium-Risk Customers: 5-10% reduction based on default history.

Low-Risk Customers: No reduction to retain profitable customers.


Key Takeaways:

The model identifies risky customers and optimizes exposure reduction.

Business rules help ensure that profitable customers remain engaged.

The approach balances risk minimization and profit maximization.

This project provides a data-driven framework to reduce credit risk while maintaining customer satisfaction. By applying machine learning and business rules, financial institutions can make informed decisions on credit line adjustments, protecting their portfolio while retaining good customers.
