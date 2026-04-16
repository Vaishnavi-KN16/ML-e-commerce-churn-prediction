import streamlit as st
import pandas as pd
import joblib

# Load the model and features we saved earlier
data = joblib.load('churn_model.pkl')
model = data['model']
features = data['features']

st.title("🛍️ E-Commerce Customer Churn Predictor")
st.markdown("Enter customer details below to predict the risk of them leaving.")

# Create input fields for the top features
st.sidebar.header("Customer Details")
tenure = st.sidebar.slider("Tenure (Months)", 0, 100, 5)
complain = st.sidebar.selectbox("Has Complain?", [0, 1])
day_since_last_order = st.sidebar.number_input("Days Since Last Order", 0, 100, 2)
cashback = st.sidebar.number_input("Cashback Amount", 0, 500, 150)

# Build a dataframe for prediction (filling others with 0 for simplicity)
input_dict = {f: 0.0 for f in features}
input_dict['Tenure'] = float(tenure)
input_dict['Complain'] = float(complain)
input_dict['DaySinceLastOrder'] = float(day_since_last_order)
input_dict['CashbackAmount'] = float(cashback)

input_df = pd.DataFrame([input_dict])

if st.button("Predict Churn Risk"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ High Risk! Probability of Churn: {prob:.2%}")
    else:
        st.success(f"✅ Low Risk. Probability of Churn: {prob:.2%}")