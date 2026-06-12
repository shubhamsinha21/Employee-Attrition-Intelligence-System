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
    
    daily_rate = st.number_input(
    "Daily Rate",
    min_value=100,
    value=800
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
    
    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=30,
        value=65
)

    monthly_rate = st.number_input(
        "Monthly Rate",
         min_value=1000,
         value=15000
)

    percent_salary_hike = st.number_input(
        "Percent Salary Hike",
         min_value=10,
         max_value=25,
         value=15
)


# Work metrics

st.subheader("Work Metrics")

col1, col2 = st.columns(2)

with col1:

    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=1,
        value=10
    )

    work_life_balance = st.slider(
        "Work Life Balance",
        1,
        4,
        3
    )

    relationship_satisfaction = st.slider(
        "Relationship Satisfaction",
        1,
        4,
        3
    )

with col2:

    training_times_last_year = st.slider(
        "Training Times Last Year",
        0,
        6,
        2
    )

    stock_option_level = st.slider(
        "Stock Option Level",
        0,
        3,
        1
    )

    performance_rating = st.slider(
        "Performance Rating",
        3,
        4,
        3
    )
    
# Create predict button

feature_columns = [
    'Age',
    'DailyRate',
    'DistanceFromHome',
    'Education',
    'EnvironmentSatisfaction',
    'HourlyRate',
    'JobInvolvement',
    'JobLevel',
    'JobSatisfaction',
    'MonthlyIncome',
    'MonthlyRate',
    'NumCompaniesWorked',
    'PercentSalaryHike',
    'PerformanceRating',
    'RelationshipSatisfaction',
    'StockOptionLevel',
    'TotalWorkingYears',
    'TrainingTimesLastYear',
    'WorkLifeBalance',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager',
    'BusinessTravel_Travel_Frequently',
    'BusinessTravel_Travel_Rarely',
    'Department_Research & Development',
    'Department_Sales',
    'EducationField_Life Sciences',
    'EducationField_Marketing',
    'EducationField_Medical',
    'EducationField_Other',
    'EducationField_Technical Degree',
    'Gender_Male',
    'JobRole_Human Resources',
    'JobRole_Laboratory Technician',
    'JobRole_Manager',
    'JobRole_Manufacturing Director',
    'JobRole_Research Director',
    'JobRole_Research Scientist',
    'JobRole_Sales Executive',
    'JobRole_Sales Representative',
    'MaritalStatus_Married',
    'MaritalStatus_Single',
    'OverTime_Yes'
]

# Create predict button

if st.button("Predict Attrition Risk"):
        input_data = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )
        
# Fill numeric features

        input_data['Age'] = age
        input_data['DistanceFromHome'] = distance_from_home
        input_data['Education'] = education
        input_data['EnvironmentSatisfaction'] = environment_satisfaction
        input_data['JobInvolvement'] = job_involvement
        input_data['JobLevel'] = job_level
        input_data['JobSatisfaction'] = job_satisfaction
        input_data['MonthlyIncome'] = monthly_income
        input_data['NumCompaniesWorked'] = num_companies_worked
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
        input_data['DailyRate'] = daily_rate
        input_data['HourlyRate'] = hourly_rate
        input_data['MonthlyRate'] = monthly_rate
        input_data['PercentSalaryHike'] = percent_salary_hike
        
        
        # Business Travel
        if business_travel == "Travel_Frequently":
            input_data["BusinessTravel_Travel_Frequently"] = 1

        elif business_travel == "Travel_Rarely":
            input_data["BusinessTravel_Travel_Rarely"] = 1
            
        # Department
        if department == "Research & Development":
            input_data["Department_Research & Development"] = 1

        elif department == "Sales":
            input_data["Department_Sales"] = 1        

        # Education Field
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
            
        # Gender
        if gender == "Male":
            input_data["Gender_Male"] = 1   
            
        # Job Role     
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
            
        # Marital Status
        if marital_status == "Married":
            input_data["MaritalStatus_Married"] = 1

        elif marital_status == "Single":
            input_data["MaritalStatus_Single"] = 1
            
        # OverTime
        if overtime == "Yes":
            input_data["OverTime_Yes"] = 1
            
        scaled_data = scaler.transform(input_data)

        probability = model.predict(scaled_data)[0][0]
        
        # Apply threshold
        prediction = 1 if probability > 0.6 else 0
        
        # Display results
        st.subheader("Prediction Result")

        st.metric(
            "Attrition Probability",
            f"{probability*100:.2f}%"
        )
        
        # Risk badge
        
        if probability >= 0.7:

            st.error(
                "🔴 High Attrition Risk"
            )

        elif probability >= 0.4:

            st.warning(
            "🟡 Medium Attrition Risk"
            )

        else:

            st.success(
            "🟢 Low Attrition Risk"
        )
            
            
        # Final decision display
        
        if prediction == 1:

            st.error(
                "Employee Likely To Leave"
            )

        else:

            st.success(
                "Employee Likely To Stay"
            )
        