import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/cancer_model.pkl")

st.set_page_config(
    page_title="Cancer Survival Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Cancer Patient Survival Prediction")
st.write("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120)

gender = st.selectbox("Gender", ["Male", "Female"])

state = st.selectbox("State", ["UP", "Delhi", "Bihar", "Maharashtra", "Other"])
city = st.selectbox("City", ["Lucknow", "Delhi", "Patna", "Mumbai", "Other"])
hospital_name = st.selectbox("Hospital Name", ["AIIMS", "Apollo", "Local Hospital", "Other"])

cancer_type = st.selectbox(
    "Cancer Type",
    ["Breast Cancer", "Lung Cancer", "Colon Cancer", "Prostate Cancer", "Leukemia"]
)

stage = st.selectbox(
    "Stage",
    ["Stage I", "Stage II", "Stage III", "Stage IV"]
)

treatment_type = st.selectbox(
    "Treatment Type",
    ["Chemotherapy", "Radiation", "Surgery", "Immunotherapy"]
)

survival_months = st.number_input("Survival Months", min_value=0.0)

diagnosis_year = st.number_input("Diagnosis Year", min_value=2000, max_value=2030)

diagnosis_month = st.number_input("Diagnosis Month", min_value=1, max_value=12)


if st.button("Predict"):

    input_data = pd.DataFrame([[
        age,
        gender,
        state,
        city,
        hospital_name,
        cancer_type,
        stage,
        treatment_type,
        survival_months,
        diagnosis_year,
        diagnosis_month
    ]], columns=[
        "Age",
        "Gender",
        "State",
        "City",
        "Hospital_Name",
        "Cancer_Type",
        "Stage",
        "Treatment_Type",
        "Survival_Months",
        "Diagnosis_Year",
        "Diagnosis_Month"
    ])

    prediction = model.predict(input_data)[0]

    if prediction == "Alive":
        st.success(f"Prediction: {prediction}")
    else:
        st.error(f"Prediction: {prediction}")

    st.balloons()