# 📊 Customer Churn Analytics & Prediction

An end-to-end Machine Learning project that predicts customer churn using the IBM Telco Customer Churn dataset and provides interactive business insights through a Streamlit dashboard.

---

## 🚀 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. This project predicts whether a customer is likely to churn based on customer demographics, contract details, billing information, and services used.

The project includes:

- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Machine Learning Model Training
- Customer Churn Prediction
- Interactive Streamlit Dashboard
- Business Recommendations based on prediction

---

## ✨ Features

- 📈 Interactive analytics dashboard
- 🤖 Real-time customer churn prediction
- 📊 Business KPI cards
- 📉 Customer churn visualizations
- 💡 Automated retention recommendations
- 🧠 Random Forest Machine Learning model
- 🎨 Clean and responsive Streamlit UI

---

## 🛠 Tech Stack

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Joblib

### Machine Learning
- Random Forest Classifier
- Logistic Regression
- Pipeline
- One-Hot Encoding
- Standard Scaling

---

## 📂 Project Structure

```text
Customer-Churn-Analytics/
│
├── data/
│   ├── telecom_churn.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── model/
│   ├── churn_pipeline.pkl
│   ├── churn_model.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Preprocessing.ipynb
│
├── images/
│
├── app.py
├── train.py
├── train_selected.py
├── style.css
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

**IBM Telco Customer Churn Dataset**

The dataset contains customer information such as:

- Gender
- Contract Type
- Tenure
- Monthly Charges
- Total Charges
- Internet Service
- Payment Method
- Online Security
- Tech Support
- Churn Status

Total Records: **7043**

---

## 📈 Exploratory Data Analysis

Performed comprehensive EDA including:

- Customer Churn Distribution
- Contract Type Analysis
- Internet Service Analysis
- Monthly Charges Distribution
- Payment Method Analysis
- Correlation Heatmap
- Customer Tenure Analysis

---

## ⚙️ Machine Learning Pipeline

### Data Preprocessing

- Missing value handling
- Data cleaning
- One-Hot Encoding
- Standard Scaling
- Train-Test Split

### Models Compared

- Logistic Regression
- Random Forest Classifier

### Best Model

✅ Random Forest

Performance:

- Accuracy: **76.9%**
- ROC-AUC Score: **0.8288**

---

## 📊 Important Features

The model identified these as the most influential factors affecting churn:

1. Tenure
2. Total Charges
3. Monthly Charges
4. Month-to-Month Contract
5. Online Security
6. Fiber Optic Internet
7. Tech Support
8. Payment Method

---

## 🖥 Dashboard

The Streamlit application includes:

### 📊 Dashboard

- KPI Cards
- Churn Distribution
- Contract Analysis
- Internet Service Analysis
- Monthly Charges Analysis

### 🤖 Prediction

Users can enter customer details to:

- Predict churn probability
- View churn risk
- Receive business recommendations

### 📈 Analytics

Business insights and customer trends.

### 🧠 Model Performance

Displays:

- Accuracy
- ROC-AUC
- Best Model
- Important Features

---

## 💡 Business Recommendations

The application categorizes customers into:

### 🔴 High Risk

- Loyalty Discounts
- Annual Contract Offers
- Premium Customer Support
- Immediate Customer Follow-up

### 🟠 Medium Risk

- Promotional Plans
- Personalized Offers
- Customer Engagement

### 🟢 Low Risk

- Upselling Opportunities
- Premium Service Recommendations
- Regular Customer Engagement

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Customer-Churn-Analytics.git
```

Navigate into the project

```bash
cd Customer-Churn-Analytics
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📸 Screenshots

### Dashboard

*(Add screenshot here)*

---

### Prediction Page

*(Add screenshot here)*

---

### Analytics

*(Add screenshot here)*

---

### Model Performance

*(Add screenshot here)*

---

## 🔮 Future Improvements

- XGBoost Model
- LightGBM
- Hyperparameter Optimization
- SHAP Explainability
- Customer Segmentation
- Cloud Deployment
- Live Database Integration
- REST API using FastAPI

---

## 👩‍💻 Developed By

**Sruthi K**

B.E. Computer Science Engineering

AI | Machine Learning | Data Analytics | Software Development

GitHub: https://github.com/SruthiK30

LinkedIn: https://www.linkedin.com/in/sruthi30082004

---

## ⭐ If you found this project useful, don't forget to star the repository!