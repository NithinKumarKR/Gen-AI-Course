import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load model
model = load_model('model.h5')

# Load encoders
with open('gender_label.pkl','rb') as f:
    gender_label = pickle.load(f)
with open('family_history_label.pkl','rb') as f:
    family_history_label = pickle.load(f)
with open('Insulin_label.pkl','rb') as f:
    Insulin_label = pickle.load(f)
with open('food_intake_label.pkl','rb') as f:
    food_intake_one = pickle.load(f)
with open('previous_medications.pkl','rb') as f:
    previous_medications = pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

# UI
st.title("Insulin Dosage Prediction")

gender = st.selectbox("Gender", gender_label.classes_)
age = st.number_input("Age", min_value=10, max_value=90, value=30)
family_history = st.selectbox("Family History", family_history_label.classes_)
glucose_level = st.number_input("Glucose Level", min_value=10, max_value=500, value=100)
physical_activity = st.number_input("Physical Activity", min_value=0, max_value=500, value=30)
food_intake = st.selectbox("Food Intake", food_intake_one.classes_)

# FIXED NAME HERE
previous_meds_input = st.selectbox("Previous Medications", previous_medications.classes_)

BMI = st.number_input("BMI", min_value=10, max_value=100, value=25)
weight = st.number_input("Weight", min_value=10, max_value=300, value=70)
HbA1c = st.number_input("HbA1c", min_value=3, max_value=20, value=5)
insulin_sensitivity = st.number_input("Insulin Sensitivity", min_value=1, max_value=1000, value=50)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
creatinine = st.number_input("Creatinine", min_value=0.1, max_value=20.0, value=1.0)

# ---- Encoding ----
input_dict = {
    'gender': gender_label.transform([gender])[0],
    'age': age,
    'family_history': family_history_label.transform([family_history])[0],
    'glucose_level': glucose_level,
    'physical_activity': physical_activity,
    'food_intake': food_intake_one.transform([food_intake])[0],
    'previous_medications': previous_medications.transform([previous_meds_input])[0],
    'BMI': BMI,
    'HbA1c': HbA1c,
    'weight': weight,
    'insulin_sensitivity': insulin_sensitivity,
    'sleep_hours': sleep_hours,
    'creatinine': creatinine
}

# Convert to DataFrame (IMPORTANT!)
X = pd.DataFrame([input_dict])

# Scale
X_scaled = scaler.transform(X)

# Prediction
pred = model.predict(X_scaled)
pred_class = np.argmax(pred, axis=1)

st.success(f"Predicted Insulin Dosage: {Insulin_label.classes_[pred_class[0]]}")
