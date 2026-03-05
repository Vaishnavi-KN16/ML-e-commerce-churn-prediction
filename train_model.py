import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
# load dataset 

df = pd.read_csv("data/kaggle_ecommerce.csv")
# dropping the customer ID problem 
df = df.drop("CustomerID", axis=1)
#features printing 
print(df.head())
print(df.shape)
print(df.columns)
# encode the categorial cols 
df = pd.get_dummies(df, drop_first=True)
# checking for null values 
print(df.isnull().sum())
# fixing the null value 
df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median())
print(df["Churn"].value_counts())
print(df.isnull().sum())
# splitting the features and the target 
X = df.drop("Churn", axis=1)
y = df["Churn"]
# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

best_model = None
best_score = 0

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n",name)
    print("Accuracy:", acc)
    print("AUC:", auc)

    if auc > best_score:
        best_score = auc
        best_model = model
# evaluation 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Save best model

import joblib

joblib.dump(model, "models/churn_model.pkl")
print("Model saved!")