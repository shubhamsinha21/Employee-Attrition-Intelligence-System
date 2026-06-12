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

# Sidebar for input features

st.sidebar.header("Model Information")

st.sidebar.info("""
Dataset: IBM HR Analytics

Model: Artificial Neural Network

Architecture:
44 → 64 → 32 → 16 → 1

AUC Score: 0.785

Class Weighting: Yes

Threshold: 0.6
""")

# Create Input Sections

st.subheader("Personal Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Divorced", "Married", "Single"]
    )

with col2:
    education = st.slider(
        "Education",
        1,
        5,
        3
    )

    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Other",
            "Technical Degree",
            "Human Resources"
        ]
    )
