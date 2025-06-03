
# Author - By Madhurya Jagadeesh

import streamlit as st
import pandas as pd
import joblib

model = joblib.load('churn_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("\U0001F4F1 Telco Customer Churn Predictor")
st.write("Fill in the customer information below to predict churn likelihood.")

with st.form("churn_form"):
    CallFailure = st.slider("Call Failures", 0, 10, 1)
    SubscriptionLength = st.slider("Subscription Length (months)", 0, 60, 12)
    ChargeAmount = st.slider("Charge Amount (1 = lowest, 9 = highest)", 1, 9, 5)
    SecondsUse = st.number_input("Seconds of Use (last year)", min_value=0, value=3000)
    FrequencyUse = st.number_input("Frequency of Use (calls/year)", min_value=0, value=50)
    FrequencySMS = st.number_input("SMS Sent (last year)", min_value=0, value=10)
    DistinctCalls = st.number_input("Distinct Numbers Called", min_value=0, value=25)
    Age = st.slider("Age", 18, 80, 30)
    CustomerValue = st.number_input("Customer Value", min_value=0, value=50)

    Complains = st.selectbox("Filed Complaints?", ["No", "Yes"])
    TariffPlan = st.selectbox("Tariff Plan", ["Pay as you go", "Contract"])
    AgeGroup = st.selectbox("Age Group", [1, 2, 3, 4, 5])
    Status = st.selectbox("Status", ["Active", "Not Active"])

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_dict = {
        'CallFailure': [CallFailure],
        'SubscriptionLength': [SubscriptionLength],
        'ChargeAmount': [ChargeAmount],
        'SecondsUse': [SecondsUse],
        'FrequencyUse': [FrequencyUse],
        'FrequencySMS': [FrequencySMS],
        'DistinctCalls': [DistinctCalls],
        'Age': [Age],
        'CustomerValue': [CustomerValue],
        'Complains_1': [1 if Complains == 'Yes' else 0],
        'TariffPlan_2': [1 if TariffPlan == 'Contract' else 0],
        'AgeGroup_2': [1 if AgeGroup == 2 else 0],
        'AgeGroup_3': [1 if AgeGroup == 3 else 0],
        'AgeGroup_4': [1 if AgeGroup == 4 else 0],
        'AgeGroup_5': [1 if AgeGroup == 5 else 0],
        'Status_2': [1 if Status == 'Not Active' else 0]
    }

    X_input = pd.DataFrame(input_dict)
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.info(f"Churn Probability: {prob:.2%}")


