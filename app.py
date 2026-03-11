import streamlit as st
from src.prediction import Diabetes_Prediction

st.title("Diabetes Disease Progression Predictor")
st.write("Enter the patient's clinical details to predict the disease progression score 🏥")

Age                = st.number_input("Enter Age (years) : ",           min_value=1,   max_value=120, value=45)
BMI                = st.number_input("Enter BMI : ",                   min_value=10.0, max_value=60.0, value=26.0)
Avg_Blood_Pressure = st.number_input("Enter Average Blood Pressure : ", min_value=50.0, max_value=200.0, value=94.0)
Total_Cholesterol  = st.number_input("Enter Total Cholesterol (mg/dL) : ", min_value=50.0, max_value=400.0, value=189.0)
LDL_Cholesterol    = st.number_input("Enter LDL Cholesterol (mg/dL) : ",   min_value=10.0, max_value=300.0, value=115.0)
HDL_Cholesterol    = st.number_input("Enter HDL Cholesterol (mg/dL) : ",   min_value=10.0, max_value=150.0, value=49.0)
TC_HDL_Ratio       = st.number_input("Enter TC/HDL Ratio : ",          min_value=1.0,  max_value=15.0, value=4.1)
Log_Triglycerides  = st.number_input("Enter Log Triglycerides : ",     min_value=3.0,  max_value=7.0,  value=4.6)
Blood_Sugar        = st.number_input("Enter Blood Sugar Level (mg/dL) : ", min_value=50,  max_value=300,  value=91)

if st.button("Predict"):
    model  = Diabetes_Prediction()
    result = model.prediction(Age, BMI, Avg_Blood_Pressure, Total_Cholesterol,
                               LDL_Cholesterol, HDL_Cholesterol, TC_HDL_Ratio,
                               Log_Triglycerides, Blood_Sugar)

    st.success(f"The predicted disease progression score is : {result} 🏥")

    if result < 100:
        st.info("🟢 Risk Category : LOW — Regular monitoring recommended.")
    elif result < 175:
        st.warning("🟡 Risk Category : MODERATE — Consult your physician for a treatment review.")
    else:
        st.error("🔴 Risk Category : HIGH — Immediate clinical intervention advised.")
