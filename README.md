# 📊 Customer Churn Analytics

**AI-Powered Customer Retention Dashboard** — an end-to-end machine learning app that analyzes telecom customer data, visualizes churn drivers, and predicts churn risk for individual customers with SHAP-based explainability.

🔗 **Live App:** [customer-churn-analytics-nfaikvg8ryrkwrnivqp8yb.streamlit.app](https://customer-churn-analytics-nfaikvg8ryrkwrnivqp8yb.streamlit.app)

---

## 🚀 Overview

This project analyzes a telecom customer dataset (7,032 customers) to understand **why customers churn** and predicts churn risk using a trained machine learning model. The Streamlit dashboard makes the insights and predictions accessible through an interactive UI — no code required to explore.

## ✨ Features

- **📈 Dashboard** — At-a-glance KPIs (total customers, churn rate, average bill, average tenure) plus churn distribution and contract-type breakdown visuals
- **🤖 Predict** — Enter a customer's details and get a real-time churn prediction, with SHAP explanations showing *why* the model made that call
- **📊 Analytics** — Deeper exploratory visuals: churn by tenure, monthly charges, internet service, payment method, and a correlation heatmap
- **🧠 Model** — Model performance details and feature importance
- **ℹ️ About** — Project background and methodology

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| App Framework | Streamlit, streamlit-option-menu |
| Data Processing | pandas, numpy |
| Modeling | scikit-learn |
| Explainability | SHAP |
| Visualization | Plotly |
| Model Persistence | joblib |

## 📁 Project Structure

```
Customer-Churn-Analytics/
├── data/
│   ├── telecom_churn.csv       # Cleaned dataset
│   ├── X_train.csv / X_test.csv
│   └── y_train.csv / y_test.csv
├── model/
│   ├── churn_pipeline.pkl      # Trained model
│   └── feature_names.pkl
├── notebooks/
│   └── 01_EDA.ipynb            # Exploratory data analysis
├── images/
├── app.py                      # Streamlit app entry point
├── train.py                    # Model training script
└── requirements.txt
```

## 🧪 Dataset

Based on the classic Telco Customer Churn dataset, with features including:
- **Demographics:** gender, senior citizen, partner, dependents
- **Account info:** tenure, contract type, payment method, paperless billing
- **Services:** phone, internet, online security, tech support, streaming
- **Charges:** monthly charges, total charges

After cleaning (removing blank `TotalCharges` entries), the dataset contains **7,032 customers** with a churn rate of **~26.5%**.

## 💻 Run Locally

```bash
git clone https://github.com/SruthiK30/Customer-Churn-Analytics.git
cd Customer-Churn-Analytics
pip install -r requirements.txt
streamlit run app.py
```

## 📸 Screenshots

### Dashboard
![Dashboard](images/Dashboard.jpeg)

---

### Prediction Page (with SHAP Explainability)
![Predict](images/Predict.png)

---

### Analytics
![Analytics](images/Analytics.jpeg)

---

### Model Performance
![Model](images/Model.jpeg)

---

## 🔮 Future Improvements

- Hyperparameter optimization (GridSearchCV / Optuna)
- Customer segmentation clustering
- Cloud deployment (Streamlit Community Cloud)
- SQLite integration for business analytics via SQL queries
- A/B testing framework for retention offer effectiveness

---

## 👩‍💻 Developed By

**Sruthi K**

B.E. Computer Science Engineering

AI | Machine Learning | Data Analytics | Software Development

GitHub: https://github.com/SruthiK30

LinkedIn: https://www.linkedin.com/in/sruthi30082004

---

## ⭐ If you found this project useful, don't forget to star the repository!