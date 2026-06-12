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

# Job Information

st.subheader("Job Information")

col1, col2 = st.columns(2)

with col1:

    department = st.selectbox(
        "Department",
        [
            "Human Resources",
            "Research & Development",
            "Sales"
        ]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Healthcare Representative",
            "Human Resources",
            "Laboratory Technician",
            "Manager",
            "Manufacturing Director",
            "Research Director",
            "Research Scientist",
            "Sales Executive",
            "Sales Representative"
        ]
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )

    overtime = st.selectbox(
        "OverTime",
        [
            "No",
            "Yes"
        ]
    )

with col2:

    job_level = st.slider(
        "Job Level",
        1,
        5,
        2
    )

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1,
        4,
        3
    )

    job_involvement = st.slider(
        "Job Involvement",
        1,
        4,
        3
    )

    environment_satisfaction = st.slider(
        "Environment Satisfaction",
        1,
        4,
        3
    )
    
    
# Salary Information and Experience

st.subheader("Salary & Experience")

col1, col2 = st.columns(2)

with col1:

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        value=5000
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        value=10
    )

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        value=5
    )

    years_in_current_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        value=3
    )

with col2:

    years_since_last_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        value=1
    )

    years_with_curr_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        value=3
    )

    num_companies_worked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        value=2
    )

