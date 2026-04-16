# 🛒 E-Commerce Customer Churn Predictor

An end-to-end Machine Learning pipeline that predicts customer churn with **98% accuracy** and provides model transparency using **SHAP** values.

## 🚀 Features
* **Predictive Modeling:** Random Forest Classifier trained on e-commerce behavior data.
* **Explainable AI (XAI):** Integrated SHAP summary plots to show why customers leave.
* **Containerized Deployment:** Packaged with **Docker** and **Streamlit** for a seamless UI experience.

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Libraries:** Pandas, Scikit-Learn, SHAP, Joblib
* **Deployment:** Docker, Streamlit

## 📦 How to Run
1. **Using Docker (Recommended):**
   ```bash
   docker build -t churn-app .
   docker run -p 8501:8501 churn-app

    pip install -r requirements.txt
    streamlit run app.py

