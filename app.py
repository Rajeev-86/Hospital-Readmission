import streamlit as st
import cloudpickle
import pandas as pd
import numpy as np
import os
from model_loader import ensure_models_exist

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# Ensure models are available before loading
if not ensure_models_exist():
    st.stop()

# Load the preprocessor and model
@st.cache_resource
def load_models():
    with open('preprocessor.pkl', 'rb') as file:
        preprocessor = cloudpickle.load(file)
    with open('model.pkl', 'rb') as file:
        model = cloudpickle.load(file)
    return preprocessor, model

loaded_preprocessor, loaded_model = load_models()
OPTIMAL_THRESHOLD = 0.1174

def predict_readmission(input_features: dict) -> tuple:
    """
    Predicts the likelihood of patient readmission based on input features.
    """
    expected_columns = [
        'race', 'gender', 'age', 'time_in_hospital', 'medical_specialty',
        'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum',
        'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
        'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
        'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone',
        'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
        'diabetesMed', 'admission_type_description',
        'discharge_disposition_description', 'admission_source_description'
    ]

    input_df = pd.DataFrame([input_features], columns=expected_columns)
    
    numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                    'num_medications', 'number_outpatient', 'number_emergency', 
                    'number_inpatient', 'number_diagnoses']
    
    for col in numeric_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    processed_features = loaded_preprocessor.transform(input_df)
    predicted_probability = loaded_model.predict_proba(processed_features)[:, 1][0]
    predicted_class = 1 if predicted_probability >= OPTIMAL_THRESHOLD else 0

    return predicted_class, predicted_probability

# App title and description
st.title("🏥 Hospital Readmission Risk Predictor")
st.markdown("""
This application predicts the risk of hospital readmission for diabetic patients.
Fill in the patient information below to get a prediction.
""")

# Create tabs
tab1, tab2 = st.tabs(["📋 Patient Input", "ℹ️ About"])

with tab1:
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        race = st.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        age = st.selectbox("Age Range", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                                         '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
        
        st.subheader("Hospital Stay Details")
        time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=14, value=5)
        medical_specialty = st.text_input("Medical Specialty", value="Cardiology")
        
        st.subheader("Admission Information")
        admission_type = st.selectbox("Admission Type", ['Emergency', 'Urgent', 'Elective', 'Not Available'])
        discharge_disposition = st.selectbox("Discharge Disposition", 
                                             ['Discharged to home', 
                                              'Discharged/transferred to SNF',
                                              'Discharged/transferred to home with home health service',
                                              'Discharged/transferred to another short term hospital',
                                              'Other'])
        admission_source = st.selectbox("Admission Source", 
                                       ['Emergency Room', 'Physician Referral', 'Clinic Referral', 'Other'])
        
        st.subheader("Procedures and Tests")
        num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=132, value=60)
        num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=6, value=2)
        num_medications = st.number_input("Number of Medications", min_value=1, max_value=81, value=15)
        
    with col2:
        st.subheader("Previous Visits")
        number_outpatient = st.number_input("Number of Outpatient Visits", min_value=0, max_value=42, value=0)
        number_emergency = st.number_input("Number of Emergency Visits", min_value=0, max_value=76, value=0)
        number_inpatient = st.number_input("Number of Inpatient Visits", min_value=0, max_value=21, value=1)
        
        st.subheader("Diagnosis")
        diag_1 = st.text_input("Primary Diagnosis Code", value="428")
        diag_2 = st.text_input("Secondary Diagnosis Code", value="250.01")
        diag_3 = st.text_input("Additional Diagnosis Code", value="401")
        number_diagnoses = st.number_input("Total Number of Diagnoses", min_value=1, max_value=16, value=8)
        
        st.subheader("Lab Results")
        max_glu_serum = st.selectbox("Max Glucose Serum", ['None', '>200', '>300', 'Norm'])
        a1c_result = st.selectbox("A1C Result", ['None', '>7', '>8', 'Norm'])
        
        st.subheader("Medication Changes")
        change = st.selectbox("Medication Changed?", ['No', 'Ch'])
        diabetes_med = st.selectbox("Diabetes Medication Prescribed?", ['Yes', 'No'])
    
    # Medications section
    st.subheader("💊 Diabetes Medications")
    st.markdown("Specify if the patient is taking any of the following medications:")
    
    medication_options = ['No', 'Steady', 'Up', 'Down']
    
    med_col1, med_col2, med_col3, med_col4 = st.columns(4)
    
    with med_col1:
        metformin = st.selectbox("Metformin", medication_options, index=1)
        repaglinide = st.selectbox("Repaglinide", medication_options)
        nateglinide = st.selectbox("Nateglinide", medication_options)
        chlorpropamide = st.selectbox("Chlorpropamide", medication_options)
        glimepiride = st.selectbox("Glimepiride", medication_options, index=1)
        acetohexamide = st.selectbox("Acetohexamide", medication_options)
    
    with med_col2:
        glipizide = st.selectbox("Glipizide", medication_options)
        glyburide = st.selectbox("Glyburide", medication_options)
        tolbutamide = st.selectbox("Tolbutamide", medication_options)
        pioglitazone = st.selectbox("Pioglitazone", medication_options)
        rosiglitazone = st.selectbox("Rosiglitazone", medication_options)
        acarbose = st.selectbox("Acarbose", medication_options)
    
    with med_col3:
        miglitol = st.selectbox("Miglitol", medication_options)
        troglitazone = st.selectbox("Troglitazone", medication_options)
        tolazamide = st.selectbox("Tolazamide", medication_options)
        examide = st.selectbox("Examide", medication_options)
        citoglipton = st.selectbox("Citoglipton", medication_options)
        insulin = st.selectbox("Insulin", medication_options, index=2)
    
    with med_col4:
        glyburide_metformin = st.selectbox("Glyburide-Metformin", medication_options)
        glipizide_metformin = st.selectbox("Glipizide-Metformin", medication_options)
        glimepiride_pioglitazone = st.selectbox("Glimepiride-Pioglitazone", medication_options)
        metformin_rosiglitazone = st.selectbox("Metformin-Rosiglitazone", medication_options)
        metformin_pioglitazone = st.selectbox("Metformin-Pioglitazone", medication_options)
    
    # Predict button
    st.markdown("---")
    if st.button("🔍 Predict Readmission Risk", type="primary", use_container_width=True):
        # Collect all input features
        patient_data = {
            'race': race,
            'gender': gender,
            'age': age,
            'time_in_hospital': time_in_hospital,
            'medical_specialty': medical_specialty,
            'num_lab_procedures': num_lab_procedures,
            'num_procedures': num_procedures,
            'num_medications': num_medications,
            'number_outpatient': number_outpatient,
            'number_emergency': number_emergency,
            'number_inpatient': number_inpatient,
            'diag_1': diag_1,
            'diag_2': diag_2,
            'diag_3': diag_3,
            'number_diagnoses': number_diagnoses,
            'max_glu_serum': max_glu_serum,
            'A1Cresult': a1c_result,
            'metformin': metformin,
            'repaglinide': repaglinide,
            'nateglinide': nateglinide,
            'chlorpropamide': chlorpropamide,
            'glimepiride': glimepiride,
            'acetohexamide': acetohexamide,
            'glipizide': glipizide,
            'glyburide': glyburide,
            'tolbutamide': tolbutamide,
            'pioglitazone': pioglitazone,
            'rosiglitazone': rosiglitazone,
            'acarbose': acarbose,
            'miglitol': miglitol,
            'troglitazone': troglitazone,
            'tolazamide': tolazamide,
            'examide': examide,
            'citoglipton': citoglipton,
            'insulin': insulin,
            'glyburide-metformin': glyburide_metformin,
            'glipizide-metformin': glipizide_metformin,
            'glimepiride-pioglitazone': glimepiride_pioglitazone,
            'metformin-rosiglitazone': metformin_rosiglitazone,
            'metformin-pioglitazone': metformin_pioglitazone,
            'change': change,
            'diabetesMed': diabetes_med,
            'admission_type_description': admission_type,
            'discharge_disposition_description': discharge_disposition,
            'admission_source_description': admission_source
        }
        
        # Make prediction
        with st.spinner("Analyzing patient data..."):
            predicted_class, predicted_prob = predict_readmission(patient_data)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if predicted_class == 1:
                st.error("⚠️ **High Risk of Readmission**")
            else:
                st.success("✅ **Low Risk of Readmission**")
        
        with result_col2:
            st.metric("Readmission Probability", f"{predicted_prob:.2%}")
        
        # Risk level visualization
        st.markdown("### Risk Assessment")
        
        # Create risk level indicator
        if predicted_prob < 0.1:
            risk_level = "Very Low"
            risk_color = "🟢"
        elif predicted_prob < 0.2:
            risk_level = "Low"
            risk_color = "🟡"
        elif predicted_prob < 0.3:
            risk_level = "Moderate"
            risk_color = "🟠"
        else:
            risk_level = "High"
            risk_color = "🔴"
        
        st.markdown(f"{risk_color} **Risk Level: {risk_level}**")
        
        # Progress bar for probability
        st.progress(min(predicted_prob, 1.0))
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if predicted_class == 1:
            st.warning("""
            **Based on the high risk prediction, consider:**
            - Enhanced discharge planning and patient education
            - Follow-up appointment within 7 days
            - Home health services if applicable
            - Medication reconciliation
            - Ensure patient understands medication regimen
            """)
        else:
            st.info("""
            **Based on the low risk prediction:**
            - Standard discharge procedures
            - Follow-up as per normal protocol
            - Continue monitoring patient condition
            """)

with tab2:
    st.subheader("About This Application")
    st.markdown("""
    ### Purpose
    This application uses machine learning to predict the risk of hospital readmission for diabetic patients
    within 30 days of discharge. The model was trained on historical hospital data and considers various
    factors including:
    
    - **Patient Demographics**: Age, gender, race
    - **Hospital Stay Information**: Length of stay, medical specialty, admission type
    - **Medical History**: Previous visits, diagnoses, procedures
    - **Medications**: Diabetes medications and changes
    - **Lab Results**: Glucose levels, A1C results
    
    ### Model Information
    - **Optimal Threshold**: 0.1174
    - **Model Type**: Machine Learning Classifier with Preprocessing Pipeline
    
    ### Important Note
    This tool is designed to assist healthcare professionals in identifying patients at higher risk
    of readmission. It should be used as a decision support tool and not as a replacement for
    clinical judgment.
    
    ### How to Use
    1. Navigate to the **Patient Input** tab
    2. Fill in all required patient information
    3. Click the **Predict Readmission Risk** button
    4. Review the prediction results and recommendations
    
    ### Contact
    For questions or support, please contact your system administrator.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Hospital Readmission Predictor v1.0 | "
    "Built with Streamlit</div>",
    unsafe_allow_html=True
)
