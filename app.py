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

age = st.number_input("Age", min_value=0, max_value=120,value = 0)

gender = st.selectbox(
    "Gender",
    ["Select Gender","Male", "Female"])
 
state = st.selectbox(
     "State",
     ["Select State","UP", "Delhi", "Bihar", "Maharashtra", "Other"])
city = st.selectbox(
     "City",
     ["Select City","Lucknow", "Delhi", "Patna", "Mumbai", "Other"])
hospital_name = st.selectbox(
     "Hospital Name",
     ["Select Hospital Name", "AIIMS", "Apollo", "Local Hospital", "Other"])

cancer_type = st.selectbox(
        "Cancer Type",
        ["Select Cancer Type","Breast Cancer", "Lung Cancer", "Colon Cancer", "Prostate Cancer", "Leukemia"]
    )

stage = st.selectbox(
        "Stage",
        ["Select Stage","Stage I", "Stage II", "Stage III", "Stage IV"]
    )

treatment_type = st.selectbox(
        "Treatment Type",
        ["Select Treatment Type","Chemotherapy", "Radiation", "Surgery", "Immunotherapy"]
    )

survival_months = st.number_input("Survival Months", min_value=0.0,value=0.0)

diagnosis_year = st.number_input("Diagnosis Year", min_value=2000, max_value=2030,value = 2000)

diagnosis_month = st.number_input("Diagnosis Month", min_value=0, max_value=12, value=0)


if st.button("Predict"):
        if age == 0:
             st.warning("Please Enter Age")
             st.stop()

        if gender == "Select Gender":
             st.warning("Please Select Gender")
             st.stop()

        if state == "Select State":
             st.warning("Please Select State")
             st.stop()

        if city == "Select City":
             st.warning("Please Select City")
             st.stop()

        if hospital_name== "Select Hospital":
             st.warning("Please Select Hospital")
             st.stop()

        if cancer_type == "Select Cancer Type":
             st.warning("Please Select Cancer Type")
             st.stop()

        if stage == "Select Stage":
             st.warning("Please Select Stage")
             st.stop()

        if treatment_type == "Select Treatment":
             st.warning("Please Select Treatment Type")
             st.stop()

        if survival_months <=0:
             st.warning("Please Enter Survival Months")
             st.stop()

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
        ]], columns = [
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

        st.snow()