import numpy as np
import pandas as pd

np.random.seed(42)

# Number of customers
n = 150

# Generate features
age = np.random.randint(18, 70, n)
total_orders = np.random.randint(0, 20, n)
last_purchase_days = np.random.randint(1, 180, n)
browsing_time = np.random.uniform(1, 20, n)  # minutes per session
cart_abandon = np.random.randint(0, 10, n)
sessions_per_month = np.random.randint(1, 40, n)

# Create churn probability based on behavior (realistic logic)
churn_prob = (
    0.3 * (last_purchase_days / 180) +     # long time since last purchase → more churn
    0.2 * (cart_abandon / 10) +            # more abandoned carts → more churn
    0.1 * (1 - total_orders / 20) +        # fewer orders → more churn
    0.1 * (1 - sessions_per_month / 40)    # fewer sessions → more churn
)

# Convert probability to binary churn (0/1)
churn = (churn_prob > np.random.rand(n)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "Age": age,
    "TotalOrders": total_orders,
    "LastPurchaseDaysAgo": last_purchase_days,
    "BrowsingTime": browsing_time.round(2),
    "CartAbandonCount": cart_abandon,
    "SessionsPerMonth": sessions_per_month,
    "Churn": churn
})
df.to_csv("churn_data.csv", index=False)
print("Dataset saved as ecommerce_churn_data.csv")

print(df.head())
print("\nDataset created with shape:", df.shape)