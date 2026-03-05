from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
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
# Select features and label
X = df[["Age", "TotalOrders", "LastPurchaseDaysAgo", 
        "BrowsingTime", "CartAbandonCount", "SessionsPerMonth"]]
y = df["Churn"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE balancing:")
print(pd.Series(y_train).value_counts())

# Train Logistic Regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create pipeline: scaling + logistic regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=500))
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]


print("Model Training Complete!")
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# AUC Score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print("AUC Score:", roc_auc)

# Optional: ROC Curve plot
plt.plot(fpr, tpr, label="AUC = {:.2f}".format(roc_auc))
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Print coefficients (feature importance)
print("\nFeature Importances (Coefficients):")
for feature, coef in zip(X.columns, model.named_steps["logreg"].coef_[0]):
    print(f"{feature}: {coef}")