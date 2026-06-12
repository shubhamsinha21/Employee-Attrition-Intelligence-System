import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------
# LOAD ARTIFACTS
# --------------------------------

model = load_model("artifacts/employee_attrition_ann.keras")
scaler = joblib.load("artifacts/scaler.pkl")

# --------------------------------
# HEADER
# --------------------------------

st.title("📊 Employee Attrition Prediction System")

st.markdown("""
Predict whether an employee is likely to leave the organization
using a Deep Learning (ANN) model trained on IBM HR Analytics data.
""")