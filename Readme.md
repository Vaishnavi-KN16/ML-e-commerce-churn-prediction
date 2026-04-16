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
### 3. The License
For a student/portfolio project, the **MIT License** is the industry standard. It’s "short and sweet"—it says anyone can use your code, but they can't hold you liable if something goes wrong.

Create a file named `LICENSE` and paste the MIT template:

> **MIT License**
> 
> Copyright (c) 2026 Vaishnavi KN
> 
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software... (and so on).

---

### 4. The Final Polish: `requirements.txt`
Docker uses this to know what to install. Run this command in your terminal to generate it automatically:

```powershell
pip freeze > requirements.txt