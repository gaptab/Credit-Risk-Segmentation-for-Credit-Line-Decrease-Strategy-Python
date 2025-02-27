import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Generate 5000 customers with random attributes
num_customers = 5000
data = {
    "Customer_ID": np.arange(10001, 10001 + num_customers),
    "Credit_Score": np.random.randint(300, 850, num_customers),
    "Income": np.random.randint(20000, 200000, num_customers),
    "Total_Debt": np.random.randint(1000, 100000, num_customers),
    "Account_Age": np.random.randint(1, 30, num_customers),
    "Default_History": np.random.choice([0, 1], size=num_customers, p=[0.85, 0.15]),  # 15% default rate
    "Missed_Payments": np.random.randint(0, 10, num_customers),
    "Utilization_Ratio": np.random.uniform(0.1, 1.0, num_customers),
    "Existing_Credit_Lines": np.random.randint(1, 10, num_customers),
}

df = pd.DataFrame(data)

# Compute risk score: Higher score means lower risk
df["Risk_Score"] = (df["Credit_Score"] / 850) * 100 - (df["Missed_Payments"] * 2) - (df["Utilization_Ratio"] * 20)
df["Risk_Score"] = df["Risk_Score"].clip(0, 100)  # Keep score between 0-100

# Compute profitability score based on income and utilization
df["Profitability_Score"] = (df["Income"] / df["Total_Debt"]) * 10 + (1 - df["Utilization_Ratio"]) * 50
df["Profitability_Score"] = df["Profitability_Score"].clip(0, 100)

# Save as CSV
df.to_csv("credit_risk_data.csv", index=False)

print("Dummy data created and saved as 'credit_risk_data.csv'")

# Select features for segmentation
features = ["Risk_Score", "Profitability_Score"]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Apply K-Means clustering (Choosing 4 risk segments)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Risk_Segment"] = kmeans.fit_predict(df_scaled)

# Define Segments
segment_labels = {
    0: "Low Risk - High Profit",
    1: "Medium Risk - Medium Profit",
    2: "High Risk - Low Profit",
    3: "Very High Risk - Negative Profit"
}
df["Segment_Label"] = df["Risk_Segment"].map(segment_labels)

# Save Segmented Data
df.to_csv("credit_risk_segmented.csv", index=False)

print("Credit risk segmentation completed. Results saved in 'credit_risk_segmented.csv'.")

# Plot Segmentation
plt.figure(figsize=(8, 6))
for segment in df["Risk_Segment"].unique():
    segment_data = df[df["Risk_Segment"] == segment]
    plt.scatter(segment_data["Risk_Score"], segment_data["Profitability_Score"], label=segment_labels[segment], alpha=0.6)
plt.xlabel("Risk Score")
plt.ylabel("Profitability Score")
plt.title("Credit Risk Segmentation")
plt.legend()
plt.show()

# Define Credit Line Decrease Strategy
def assign_cld(risk_segment, account_age, default_history, utilization_ratio):
    if risk_segment == "Very High Risk - Negative Profit":
        return 50  # Reduce credit limit by 50%
    elif risk_segment == "High Risk - Low Profit":
        return 30 if account_age < 5 else 20  # Reduce by 30% if account age < 5 years, else 20%
    elif risk_segment == "Medium Risk - Medium Profit":
        return 10 if default_history == 1 else 5  # Small reduction for medium-risk customers
    else:
        return 0  # No reduction for low-risk profitable customers

df["Credit_Line_Decrease_%"] = df.apply(
    lambda row: assign_cld(row["Segment_Label"], row["Account_Age"], row["Default_History"], row["Utilization_Ratio"]),
    axis=1
)

# Save final output
df.to_csv("credit_line_decrease_strategy.csv", index=False)
print("Credit line decrease strategy saved in 'credit_line_decrease_strategy.csv'.")
