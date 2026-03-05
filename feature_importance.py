import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("kaggle_ecommerce.csv")
df = df.drop("CustomerID", axis=1)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)

# Load model
model = joblib.load("models/churn_model.pkl")

# Feature importance
importances = model.feature_importances_

features = pd.Series(importances, index=X.columns)
features = features.sort_values(ascending=False)

print(features)

# Plot
plt.figure(figsize=(10,6))
features.plot(kind="bar")
plt.title("Feature Importance")
plt.show()