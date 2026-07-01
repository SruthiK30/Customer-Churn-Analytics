import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from streamlit_option_menu import option_menu

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)
def load_css():

    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()
# -------------------------------------------------
# Load Data
# -------------------------------------------------
df = pd.read_csv("data/telecom_churn.csv")

model = joblib.load("model/churn_pipeline.pkl")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/combo-chart--v1.png",
        width=80
    )

    st.markdown("# Customer Churn")

    st.caption("AI Powered Dashboard")

    st.divider()

    page = option_menu(

        menu_title=None,

        options=[
            "Dashboard",
            "Predict",
            "Analytics",
            "Model",
            "About"
        ],

        icons=[
            "speedometer2",
            "robot",
            "bar-chart-line",
            "graph-up-arrow",
            "info-circle"
        ],

        default_index=0
    )

# =================================================
# DASHBOARD
# =================================================
if page == "Dashboard":

    st.markdown(
        """
        # 📊 Customer Churn Analytics

        ### AI Powered Customer Retention Dashboard
        """
        )

    total = len(df)
    churn = len(df[df["Churn"] == "Yes"])
    churn_rate = round(churn / total * 100, 2)

    avg_bill = round(df["MonthlyCharges"].mean(), 2)
    avg_tenure = round(df["tenure"].mean(), 1)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("👥 Customers", total)
    c2.metric("📉 Churn Rate", f"{churn_rate}%")
    c3.metric("💰 Avg Bill", f"${avg_bill}")
    c4.metric("📅 Avg Tenure", avg_tenure)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            df,
            names="Churn",
            title="Customer Churn Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df,
            x="Contract",
            color="Churn",
            barmode="group",
            title="Contract Type vs Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(
            df,
            x="InternetService",
            color="Churn",
            title="Internet Service"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            title="Monthly Charges vs Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

# =================================================
# PREDICT
# =================================================
elif page == "Predict":

    st.title("🤖 Customer Churn Prediction")
    st.markdown("### Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        tenure = st.slider("Tenure", 0, 72, 24)

        monthly = st.number_input(
            "Monthly Charges",
            18.0,
            120.0,
            70.0
        )

        total = round(tenure * monthly, 2)
        st.number_input(
            "Total Charges",
            value=total,
            disabled=True
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    with col2:

        internet = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    # Button OUTSIDE columns
    if st.button(
        "🚀 Predict Customer Churn",
        type="primary",
        use_container_width=True
    ):

        customer = pd.DataFrame({

            "tenure":[tenure],
            "MonthlyCharges":[monthly],
            "TotalCharges":[total],
            "Contract":[contract],
            "InternetService":[internet],
            "PaymentMethod":[payment],
            "OnlineSecurity":[security],
            "TechSupport":[support]

        })

        try:

            prediction = model.predict(customer)[0]
            probability = model.predict_proba(customer)[0][1]

            st.divider()

            if prediction == 1:
                st.error("⚠️ Customer is likely to churn")
            else:
                st.success("✅ Customer is likely to stay")

            st.metric(
                "Churn Probability",
                f"{probability*100:.2f}%"
            )

            st.subheader("💡 Business Recommendation")

            if probability >= 0.80:

                st.error("""
    ### 🔴 High Risk

    - Offer Annual Contract
    - Provide Loyalty Discount
    - Assign Premium Support
    - Contact Customer Immediately
    """)

            elif probability >= 0.50:

                st.warning("""
    ### 🟠 Medium Risk

    - Send Promotional Offers
    - Recommend Better Plans
    - Schedule Follow-up
    """)

            else:

                st.success("""
    ### 🟢 Low Risk

    - Customer is likely to stay
    - Continue Engagement
    - Recommend Premium Services
    """)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# =================================================
# MODEL PERFORMANCE
# =================================================
elif page == "Model":

    st.title("📈 Model Performance")

    st.success("🏆 Best Model : Random Forest")

    st.metric("ROC-AUC", "0.8288")

    st.write("### Top Features")

    st.write("""
1. Tenure

2. Total Charges

3. Monthly Charges

4. Month-to-month Contract

5. Online Security

6. Fiber Optic Internet

7. Tech Support

8. Payment Method
""")

# =================================================
# ANALYTICS
# =================================================
elif page == "Analytics":

    st.title("📊 Business Analytics")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="PaymentMethod",
            color="Churn",
            barmode="group",
            title="Payment Method vs Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            nbins=30,
            title="Tenure Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    corr = df.select_dtypes(include="number").corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap"
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "About":

    st.title("ℹ About Project")

    st.markdown("""
## Customer Churn Analytics

This application predicts customer churn using Machine Learning.

### Features

- Customer Churn Prediction
- Interactive Dashboard
- Business Insights
- Customer Retention Recommendations
- Random Forest ML Model

### Dataset

IBM Telco Customer Churn Dataset

### Tech Stack

- Python
- Streamlit
- Scikit-learn
- Plotly
- Pandas
- Joblib

Developed by **Sruthi K**
""")


