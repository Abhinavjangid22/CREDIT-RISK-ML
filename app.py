import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

st.title("💳 Credit Risk Prediction App")
st.write("""
This app predicts the likelihood that a borrower will default on their credit obligations. 
Fill in the details below and click **Predict Default Risk** to see the result.
""")

# Load trained model
model = joblib.load("model/credit_model.pkl")

# Features and labels for user input
feature_map = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving Credit Utilization (%)",
    "age": "Age (years)",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30-59 Days Past Due",
    "DebtRatio": "Debt Ratio",
    "MonthlyIncome": "Monthly Income ($)",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines & Loans",
    "NumberOfTimes90DaysLate": "Times 90+ Days Late",
    "NumberRealEstateLoansOrLines": "Real Estate Loans/Lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60-89 Days Past Due",
    "NumberOfDependents": "Number of Dependents"
}

st.subheader("📝 Enter Borrower Details")

# Collect user input
input_data = {}
for key, label in feature_map.items():
    if "Income" in label or "Ratio" in label or "Utilization" in label:
        input_data[key] = st.number_input(label, min_value=0.0, step=0.01)
    else:
        input_data[key] = st.number_input(label, min_value=0, step=1)

# Convert input to DataFrame
data = pd.DataFrame([input_data])

st.markdown("---")
st.subheader("⚠️ Default Risk Prediction")

# Predict button
if st.button("Predict Default Risk"):
    try:
        # Make prediction
        pred = model.predict(data)[0]
        pred_prob = model.predict_proba(data)[0][1] * 100  # Probability of default

        # Determine risk level
        if pred_prob < 25:
            risk_level = "Very Low"
            color = "green"
        elif pred_prob < 50:
            risk_level = "Low"
            color = "lightgreen"
        elif pred_prob < 75:
            risk_level = "Medium"
            color = "orange"
        else:
            risk_level = "High"
            color = "red"

        # Display results
        st.markdown(f"**Prediction:** {'Yes' if pred == 1 else 'No'} (Default Risk)")
        st.markdown(f"**Probability of Default:** {pred_prob:.2f}%")
        st.markdown(f"<span style='color:{color};font-weight:bold'>Risk Level: {risk_level}</span>", unsafe_allow_html=True)

        st.info("""
        **Interpretation Guide:**
        - **Yes** = Borrower likely to default, take caution.
        - **No** = Borrower unlikely to default.
        - **Probability (%)** = Likelihood of default. Higher percentage = higher risk.
        - **Risk Level** = Quick visual indicator of credit risk.
        """)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
