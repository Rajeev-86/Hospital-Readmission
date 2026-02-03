# Diabetes 130 US Hospital Readmission Prediction

A comprehensive machine learning project to identify diabetic patients at high risk of hospital readmission within 30 days of discharge. This project includes a complete data science workflow from exploratory data analysis to model deployment.

**Author:** Rajeev Ahirwar | **Date:** January 2026

---

## Quick Links

| Resource | Link |
|----------|------|
|  **Kaggle Notebook** | [Run the Full Analysis](https://www.kaggle.com/code/rajeev86/diabetes-130-us-hospital-readmission) |
|  **Live Streamlit App** | [Hospital Readmission Predictor](https://hospital-readmission-by-rajeev.streamlit.app/) |
|  **Dataset** | [UCI ML Repository - Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |

---

## Project Overview

### Objective
Build machine learning models to identify which diabetic patients are at **high risk of being readmitted to the hospital within 30 days** of discharge. This enables healthcare providers to:
- Prioritize follow-up care for high-risk patients
- Reduce hospital readmission rates and associated costs
- Improve patient outcomes through targeted interventions

### Dataset
- **Source:** [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Records:** ~100,000 patient encounters
- **Features:** 50+ clinical and demographic variables
- **Target:** Binary classification (Readmitted within 30 days: Yes/No)

---

## Analysis Workflow (Notebook)

### 1. Data Loading & Integrity Checks
- Loaded diabetic patient data and IDS mapping tables
- Merged admission type, discharge disposition, and admission source descriptions
- Created binary target variable (1 = readmitted <30 days, 0 = otherwise)

### 2. Structural Cleaning
- **Excluded "impossible" predictions**: Removed patients who expired or entered hospice care
- **Handled missing values**:
  - Dropped columns with >40% missing (`weight`, `payer_code`)
  - Treated `?` as NaN and handled appropriately
  - Created "Missing" category for `race` and `medical_specialty`
  - Custom encoding for `A1Cresult` and `max_glu_serum` (None → -1, Norm → 5, >7/200 → 10, >8/300 → 15)

### 3. Train-Test Split (Patient-Level)
- Used **GroupShuffleSplit** to ensure the same patient doesn't appear in both train and test sets
- Prevents data leakage from repeated patient visits
- 80/20 train-test split maintaining class distribution

### 4. Feature Engineering

#### ICD-9 Code Grouping
Mapped diagnosis codes to 9 clinically meaningful categories based on [Strack et al. (2014)](https://www.researchgate.net/publication/262114048_Impact_of_HbA1c_Measurement_on_Hospital_Readmission_Rates_Analysis_of_70000_Clinical_Database_Patient_Records):
- Circulatory, Respiratory, Digestive, Diabetes, Injury
- Musculoskeletal, Genitourinary, Neoplasms, Other

#### Created Interaction Features
| Feature | Description |
|---------|-------------|
| `meds_per_day` | Medications / Time in hospital |
| `labs_per_day` | Lab procedures / Time in hospital |
| `procedures_per_day` | Procedures / Time in hospital |
| `total_visits` | Sum of inpatient + emergency + outpatient visits |
| `diagnoses_per_day` | Number of diagnoses / Time in hospital |
| `emergency_long_stay` | Emergency admission × Time in hospital |

#### Grouping Rare Categories
- **Medical Specialty:** Top 10 specialties kept; rest grouped as "Other_Specialty"
- **Discharge Disposition:** Grouped into Home, Home Health, Transfer/Facility, Left AMA, Other
- **Admission Source:** Grouped into Emergency, Referral, Transfer, Other

### 5. Exploratory Data Analysis (EDA)

Key findings:
- **Diagnosis patterns:** Diabetes and Circulatory conditions in primary diagnosis correlate with higher readmission
- **Glucose levels:** Higher `max_glu_serum` values correlate with increased readmission risk
- **Previous visits:** `number_inpatient` is the strongest predictor - history repeats itself
- **Discharge destination:** Transfer to facility = high risk; discharged home = lower risk

### 6. Data Preprocessing Pipeline

```python
# Early preprocessing (feature engineering)
early_preprocessor = Pipeline([
    ('replace_question_mark', ...),
    ('handle_med_missing_values', ...),
    ('handle_soft_missing', ...),
    ('handle_micro_missing', ...),
    ('create_interaction_features', ...),
    ('group_icd_codes', ...),
    ('grouping', ...),
    ('handle_age_mapping', ...),
    ('drop_old_columns', ...)
])

# Final preprocessing (encoding & scaling)
final_preprocessor = ColumnTransformer([
    ('meds', OrdinalEncoder, medication_columns),
    ('log_nums', Log + StandardScaler, skewed_columns),
    ('scale_nums', StandardScaler, numerical_columns),
    ('one_hot', OneHotEncoder, categorical_columns)
])
```

---

## Model Building

### Models Trained
| Model | Recall | ROC-AUC | Notes |
|-------|--------|---------|-------|
| Random Forest (Balanced) | 0.60 | 0.63 | Undersampled majority class |
| LightGBM | 0.55 | 0.65 | With `scale_pos_weight` |
| XGBoost | 0.54 | 0.64 | With `scale_pos_weight` |
| CatBoost | 0.56 | 0.65 | With `auto_class_weights='Balanced'` |
| Voting Ensemble | 0.58 | 0.65 | RF + CatBoost + LightGBM |
| **Stacking Ensemble** | **0.60** | **0.66** | **RF + CatBoost → Logistic Regression** |

### Final Model: Stacking Classifier
- **Base Learners:** Random Forest (Balanced) + CatBoost
- **Meta-Model:** Logistic Regression
- **Optimal Threshold:** 0.1174 (tuned for 60% recall)

### Performance at Optimal Threshold
- **Recall:** 60% (catches 6 out of 10 readmissions)
- **Precision:** 18% (1 in 5 flagged patients will definitely be readmitted)
- **Business Value:** Top 20% of patients flagged captures 38% of all readmissions

---

## Model Interpretation

### Feature Importance (Permutation Importance)
Top predictors of readmission:
1. `number_inpatient` - Previous hospital admissions (strongest predictor)
2. `discharge_disposition_Transfer/Facility` - Discharged to nursing facility
3. `total_visits` - Total healthcare utilization
4. `discharge_disposition_Home` - Discharged home (protective factor)
5. `insulin` - Insulin usage patterns

### SHAP Analysis Insights
- **High inpatient history** → Massive increase in readmission risk
- **Discharged to home** → Significant risk reduction
- **Insulin (especially increasing dosage)** → Higher risk
- **Transfer to facility** → Overwhelms other protective factors

---

## Streamlit Web Application

The deployed app provides an interactive interface for healthcare providers to:

### Features
- **Patient Data Input:** Comprehensive form for demographics, hospital stay, diagnoses, and medications
- **Real-time Prediction:** Instant risk assessment with probability score
- **Visual Risk Display:** Color-coded risk levels (High/Low)
- ℹ**About Section:** Model information and usage guidelines

### Input Categories
| Category | Features |
|----------|----------|
| Demographics | Race, Gender, Age Range |
| Hospital Stay | Time in hospital, Medical specialty |
| Admission Info | Admission type, Discharge disposition, Admission source |
| Procedures | Lab procedures, Procedures, Medications count |
| Previous Visits | Outpatient, Emergency, Inpatient visits |
| Diagnosis | Primary, Secondary, Additional diagnosis codes (ICD-9) |
| Lab Results | Max glucose serum, A1C result |
| Medications | 23 diabetes-related medications with dosage changes |

### Try It Live
**[Hospital Readmission Predictor App](https://hospital-readmission-by-rajeev.streamlit.app/)**

---

## Local Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Hospital-Readmission.git
cd Hospital-Readmission

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Model Files
Model files (`model.pkl` and `preprocessor.pkl`) are automatically downloaded from cloud storage on first run. See [model_loader.py](model_loader.py) for details.

---

## Project Structure

```
Hospital-Readmission/
├── app.py                                    # Streamlit web application
├── model_loader.py                           # Model download utilities
├── requirements.txt                          # Python dependencies
├── .streamlit
    └──config.toml                            # Streamlit Webapp congigs
├── .gitignore
├── README.md
└── LICENSE                                   # MIT License
```

---

## Key References

- **Dataset Paper:** Strack, B., DeShazo, J. P., et al. (2014). [Impact of HbA1c Measurement on Hospital Readmission Rates](https://www.researchgate.net/publication/262114048_Impact_of_HbA1c_Measurement_on_Hospital_Readmission_Rates_Analysis_of_70000_Clinical_Database_Patient_Records)
- **UCI ML Repository:** [Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

---

## Business Impact

| Metric | Value |
|--------|-------|
| Readmissions identified | 60% of all cases |
| Resource efficiency | 38% of problems in top 20% of calls |
| False positive rate | 4 out of 5 flagged patients may benefit from follow-up anyway |

**Use Case:** Hospitals with limited resources for follow-up calls can prioritize the top 10-20% highest-risk patients, capturing a disproportionately high percentage of potential readmissions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you found this project useful, please consider giving it a star!**

Made with ❤️ by Rajeev Ahirwar

</div>
