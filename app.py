import os

# --------------------------------
# 🔥 HARD FIX: FORCE CPU + DISABLE METAL/GPU
# --------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_METAL_DEVICE_PLACEMENT"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------
# 🎨 UI THEME ENHANCEMENT (SAFE CSS ONLY)
# --------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: #e5e7eb;
}

h1, h2, h3 {
    color: #f9fafb !important;
    font-weight: 700;
}

div.stButton > button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    font-weight: 700;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    border: none;
    transition: 0.3s ease;
    box-shadow: 0px 4px 12px rgba(34,197,94,0.3);
}

div.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0px 6px 18px rgba(34,197,94,0.5);
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 10px;
}

section[data-testid="stSidebar"] {
    background-color: #0b1220;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# LOAD MODEL + SCALER
# --------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        "artifacts/employee_attrition_ann.keras",
        compile=False,
        custom_objects=None
    )
    scaler = joblib.load("artifacts/scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# --------------------------------
# HEADER
# --------------------------------
st.title("📊 Employee Attrition Intelligence System")
st.markdown("AI-powered ANN model to predict employee attrition risk.")
st.markdown("---")

# --------------------------------
# INPUT UI
# --------------------------------
st.subheader("Employee Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
    education = st.slider("Education", 1, 5, 3)

with col2:
    education_field = st.selectbox(
        "Education Field",
        ["Life Sciences", "Medical", "Marketing", "Other", "Technical Degree", "Human Resources"]
    )

st.subheader("Job Information")

col1, col2 = st.columns(2)

with col1:
    department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
    job_role = st.selectbox(
        "Job Role",
        ["Healthcare Representative", "Human Resources", "Laboratory Technician",
         "Manager", "Manufacturing Director", "Research Director",
         "Research Scientist", "Sales Executive", "Sales Representative"]
    )
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    overtime = st.selectbox("OverTime", ["No", "Yes"])

with col2:
    job_level = st.slider("Job Level", 1, 5, 2)
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    job_involvement = st.slider("Job Involvement", 1, 4, 3)
    environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)

st.subheader("Salary & Experience")

col1, col2 = st.columns(2)

with col1:
    monthly_income = st.number_input("Monthly Income", 1000, value=5000)
    total_working_years = st.number_input("Total Working Years", 0, value=10)
    years_at_company = st.number_input("Years At Company", 0, value=5)
    years_in_current_role = st.number_input("Years In Current Role", 0, value=3)
    daily_rate = st.number_input("Daily Rate", 100, value=800)

with col2:
    years_since_last_promotion = st.number_input("Years Since Last Promotion", 0, value=1)
    years_with_curr_manager = st.number_input("Years With Current Manager", 0, value=3)
    num_companies_worked = st.number_input("Companies Worked", 0, value=2)
    hourly_rate = st.number_input("Hourly Rate", 30, value=65)
    monthly_rate = st.number_input("Monthly Rate", 1000, value=15000)
    percent_salary_hike = st.number_input("Percent Salary Hike", 10, 25, 15)

st.subheader("Work Metrics")

col1, col2 = st.columns(2)

with col1:
    distance_from_home = st.number_input("Distance From Home", 1, value=10)
    work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
    relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)

with col2:
    training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)
    stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
    performance_rating = st.slider("Performance Rating", 3, 4, 3)

# --------------------------------
# FEATURE LIST
# --------------------------------
feature_columns = [
    'Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction',
    'HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome',
    'MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating',
    'RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
    'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
    'YearsSinceLastPromotion','YearsWithCurrManager',
    'BusinessTravel_Travel_Frequently','BusinessTravel_Travel_Rarely',
    'Department_Research & Development','Department_Sales',
    'EducationField_Life Sciences','EducationField_Marketing',
    'EducationField_Medical','EducationField_Other',
    'EducationField_Technical Degree','Gender_Male',
    'JobRole_Human Resources','JobRole_Laboratory Technician','JobRole_Manager',
    'JobRole_Manufacturing Director','JobRole_Research Director',
    'JobRole_Research Scientist','JobRole_Sales Executive',
    'JobRole_Sales Representative','MaritalStatus_Married',
    'MaritalStatus_Single','OverTime_Yes'
]

# --------------------------------
# PDF REPORT FUNCTION
# --------------------------------
def generate_pdf_report(probability, risk_level,
                        salary_risk, overtime_risk,
                        satisfaction_risk, experience_risk,
                        stability_risk):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Employee Attrition Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Attrition Probability: {probability*100:.2f}%", styles["Normal"]))
    story.append(Paragraph(f"Risk Level: {risk_level}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Risk Breakdown", styles["Heading2"]))
    story.append(Paragraph(f"Salary Risk: {salary_risk}%", styles["Normal"]))
    story.append(Paragraph(f"Overtime Risk: {overtime_risk}%", styles["Normal"]))
    story.append(Paragraph(f"Satisfaction Risk: {satisfaction_risk:.1f}%", styles["Normal"]))
    story.append(Paragraph(f"Experience Risk: {experience_risk}%", styles["Normal"]))
    story.append(Paragraph(f"Stability Risk: {stability_risk}%", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

# --------------------------------
# PREDICTION FLOW
# --------------------------------
if st.button("🚀 Predict Attrition Risk"):

    try:
        progress = st.progress(0)
        status = st.empty()

        status.info("Preparing input...")
        progress.progress(20)
        time.sleep(0.2)

        input_data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

        # numeric
        input_data['Age'] = age
        input_data['DailyRate'] = daily_rate
        input_data['DistanceFromHome'] = distance_from_home
        input_data['Education'] = education
        input_data['EnvironmentSatisfaction'] = environment_satisfaction
        input_data['HourlyRate'] = hourly_rate
        input_data['JobInvolvement'] = job_involvement
        input_data['JobLevel'] = job_level
        input_data['JobSatisfaction'] = job_satisfaction
        input_data['MonthlyIncome'] = monthly_income
        input_data['MonthlyRate'] = monthly_rate
        input_data['NumCompaniesWorked'] = num_companies_worked
        input_data['PercentSalaryHike'] = percent_salary_hike
        input_data['PerformanceRating'] = performance_rating
        input_data['RelationshipSatisfaction'] = relationship_satisfaction
        input_data['StockOptionLevel'] = stock_option_level
        input_data['TotalWorkingYears'] = total_working_years
        input_data['TrainingTimesLastYear'] = training_times_last_year
        input_data['WorkLifeBalance'] = work_life_balance
        input_data['YearsAtCompany'] = years_at_company
        input_data['YearsInCurrentRole'] = years_in_current_role
        input_data['YearsSinceLastPromotion'] = years_since_last_promotion
        input_data['YearsWithCurrManager'] = years_with_curr_manager

        # categorical
        if business_travel == "Travel_Frequently":
            input_data["BusinessTravel_Travel_Frequently"] = 1
        elif business_travel == "Travel_Rarely":
            input_data["BusinessTravel_Travel_Rarely"] = 1

        if department == "Research & Development":
            input_data["Department_Research & Development"] = 1
        elif department == "Sales":
            input_data["Department_Sales"] = 1

        if education_field == "Life Sciences":
            input_data["EducationField_Life Sciences"] = 1
        elif education_field == "Marketing":
            input_data["EducationField_Marketing"] = 1
        elif education_field == "Medical":
            input_data["EducationField_Medical"] = 1
        elif education_field == "Other":
            input_data["EducationField_Other"] = 1
        elif education_field == "Technical Degree":
            input_data["EducationField_Technical Degree"] = 1

        if gender == "Male":
            input_data["Gender_Male"] = 1

        role_map = {
            "Human Resources":"JobRole_Human Resources",
            "Laboratory Technician":"JobRole_Laboratory Technician",
            "Manager":"JobRole_Manager",
            "Manufacturing Director":"JobRole_Manufacturing Director",
            "Research Director":"JobRole_Research Director",
            "Research Scientist":"JobRole_Research Scientist",
            "Sales Executive":"JobRole_Sales Executive",
            "Sales Representative":"JobRole_Sales Representative"
        }

        if job_role in role_map:
            input_data[role_map[job_role]] = 1

        if marital_status == "Married":
            input_data["MaritalStatus_Married"] = 1
        elif marital_status == "Single":
            input_data["MaritalStatus_Single"] = 1

        if overtime == "Yes":
            input_data["OverTime_Yes"] = 1

        input_data = input_data[feature_columns]

        status.info("Running model...")
        progress.progress(80)

        scaled_data = scaler.transform(input_data).astype(np.float32)

        with tf.device('/CPU:0'):
            x = tf.convert_to_tensor(scaled_data)
            prediction_raw = model(x, training=False).numpy()

        probability = float(prediction_raw[0][0])

        progress.progress(100)
        status.success("Done")

        st.markdown("---")

        # results
        st.subheader("Prediction Result")

        risk_level = (
            "High" if probability >= 0.7 else
            "Medium" if probability >= 0.4 else
            "Low"
        )

        st.metric("Probability", f"{probability*100:.2f}%")
        st.metric("Risk Level", risk_level)

        # risk
        salary_risk = 80 if monthly_income < 3000 else 50 if monthly_income < 6000 else 20
        overtime_risk = 80 if overtime == "Yes" else 20
        satisfaction_risk = (
            (4 - job_satisfaction) * 20 +
            (4 - environment_satisfaction) * 15 +
            (4 - relationship_satisfaction) * 15
        )
        experience_risk = 80 if total_working_years < 3 else 40 if total_working_years < 7 else 20
        stability_risk = 80 if years_at_company < 2 else 50 if years_at_company < 5 else 25

        st.subheader("Risk Breakdown")

        st.write({
            "Salary Risk": salary_risk,
            "Overtime Risk": overtime_risk,
            "Satisfaction Risk": satisfaction_risk,
            "Experience Risk": experience_risk,
            "Stability Risk": stability_risk
        })

        # download report button (NOW CORRECT PLACE)
        pdf_buffer = generate_pdf_report(
            probability,
            risk_level,
            salary_risk,
            overtime_risk,
            satisfaction_risk,
            experience_risk,
            stability_risk
        )

        st.download_button(
            "📥 Download Report",
            pdf_buffer,
            file_name="attrition_report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error("Prediction Failed")
        st.exception(e)