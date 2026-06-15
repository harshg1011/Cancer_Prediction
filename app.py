import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("model/cancer_model.pkl")

st.set_page_config(
    page_title="Cancer Survival Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Cancer Patient Survival Prediction")
st.write("Enter Patient Details")

# Input Fields

age = st.number_input("Age", min_value=1, max_value=120)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

state = st.text_input("State")

city = st.text_input("City")

hospital_name = st.text_input("Hospital Name")

cancer_type = st.selectbox(
    "Cancer Type",
    [
        "Breast Cancer",
        "Lung Cancer",
        "Colon Cancer",
        "Prostate Cancer",
        "Leukemia"
    ]
)

stage = st.selectbox(
    "Stage",
    ["Stage I", "Stage II", "Stage III", "Stage IV"]
)

treatment_type = st.selectbox(
    "Treatment Type",
    [
        "Chemotherapy",
        "Radiation",
        "Surgery",
        "Immunotherapy"
    ]
)

survival_months = st.number_input(
    "Survival Months",
    min_value=0.0
)

diagnosis_year = st.number_input(
    "Diagnosis Year",
    min_value=2000,
    max_value=2030
)

diagnosis_month = st.number_input(
    "Diagnosis Month",
    min_value=1,
    max_value=12
)

# Prediction

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "State": [state],
        "City": [city],
        "Hospital_Name": [hospital_name],
        "Cancer_Type": [cancer_type],
        "Stage": [stage],
        "Treatment_Type": [treatment_type],
        "Survival_Months": [survival_months],
        "Diagnosis_Year": [diagnosis_year],
        "Diagnosis_Month": [diagnosis_month]
    })

    prediction = model.predict(input_data)[0]

    if prediction == "Alive":
        st.success(f"Prediction: {prediction}")
    else:
        st.error(f"Prediction: {prediction}")