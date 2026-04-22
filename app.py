# Import required libraries for UI, data handling, and model loading
import streamlit as st
import pandas as pd
import joblib


# Load the trained fraud detection model pipeline
model = joblib.load("fraud_detection_pipeline.pkl")


# Set the title and description of the application
st.title("AML & Fraud Detection System")
st.markdown("Enter transaction details to evaluate fraud risk")
st.divider()


# Collect user inputs for transaction details
transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]
)

amount = st.number_input("Amount", min_value=0.0, value=1000.0)

oldbalanceOrg = st.number_input(
    "Old Balance (Sender)", min_value=0.0, value=10000.0
)

oldbalanceDest = st.number_input(
    "Old Balance (Receiver)", min_value=0.0, value=0.0
)


# Automatically calculate new balances after transaction
newbalanceOrig = oldbalanceOrg - amount
newbalanceDest = oldbalanceDest + amount

st.write(f"New Balance (Sender): {newbalanceOrig:.2f}")
st.write(f"New Balance (Receiver): {newbalanceDest:.2f}")

st.divider()


# Allow user to adjust fraud detection sensitivity threshold
threshold = st.slider("Set Fraud Detection Threshold", 0.0, 1.0, 0.5)

st.info("""
Lower threshold → catches more fraud (high recall) but more false alarms  
Higher threshold → fewer false alarms (high precision) but may miss fraud
""")


# Trigger prediction when user clicks the button
if st.button("Predict"):

    # Prepare input data in model-required format
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # Generate fraud probability from the model
    prob = model.predict_proba(input_data)[0][1]

    # Display predicted fraud probability
    st.subheader(f"Fraud Probability: {prob:.2%}")

    # Categorize transaction into AML-style risk levels
    if prob > 0.8:
        risk = "🔴 High Risk"
    elif prob > 0.4:
        risk = "🟠 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    st.subheader(f"Risk Level: {risk}")

    # Display final decision based on probability and threshold
    if prob > 0.8:
        st.error("🚨 High Risk Transaction - Likely Fraud")
    elif prob > threshold:
        st.warning("⚠️ Medium Risk Transaction - Needs Review")
    else:
        st.success("✅ Transaction Looks Safe")

    # Show current threshold used for decision making
    st.write(f"Decision Threshold: {threshold}")

    # Apply rule-based checks for additional anomaly detection
    if amount > oldbalanceOrg:
        st.error("🚨 Insufficient balance! Suspicious transaction")

    if newbalanceOrig < 0:
        st.warning("⚠️ Negative balance detected")

    if amount > 1000000:
        st.warning("⚠️ High-value transaction detected")

    if newbalanceDest == 0:
        st.warning("⚠️ Receiver balance remains zero (possible anomaly)")