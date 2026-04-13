import streamlit as st
import numpy as np
import joblib

# Title
st.title("Diabetes Prediction App")

# Load model and scaler with error handling
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Input fields
st.subheader("Enter Patient Details")

pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0, step=1)

# Prediction button
if st.button("Predict"):
    try:
        # Prepare input data
        data = np.array([[pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age]])

        # Scale data
        data_scaled = scaler.transform(data)

        # Predict
        prediction = model.predict(data_scaled)

        # Output
        if prediction[0] == 1:
            st.error("Diabetic")
        else:
            st.success("Not Diabetic")

    except Exception as e:
        st.error(f"Prediction error: {e}")