import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Example new customer data
data = {
    "Age": [35],
    "Tenure": [12],
    "CityTier": [1],
    "WarehouseToHome": [15],
    "HoursSpentOnApp": [3],
    "NumberOfDevicesRegistered": [2],
    "SatisfactionScore": [3],
    "NumberOfAddress": [2],
    "Complain": [1],
    "OrderCount": [10],
    "DaySinceLastOrder": [20],
    "CashbackAmount": [100],
    "Gender_Male": [1],
    "Gender_Other": [0],
    "PreferredLoginDevice_Mobile": [1]
}

# Convert to dataframe
df = pd.DataFrame(data)

# Predict churn
prediction = model.predict(df)

# Predict probability
probability = model.predict_proba(df)

print("Prediction:", prediction[0])
print("Churn Probability:", probability[0][1])
