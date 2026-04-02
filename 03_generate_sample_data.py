# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Generate Sample Data (DEMO ONLY)
# MAGIC
# MAGIC **This notebook is for demonstration purposes only.**
# MAGIC In production, you would ingest real clinical chart PDFs into the UC Volume
# MAGIC and register them in `raw.charts`. Skip this notebook entirely for production use.
# MAGIC
# MAGIC **What this does:**
# MAGIC - Generates 100 synthetic clinical chart PDFs across 10 clinical profiles
# MAGIC - Writes PDFs directly to UC Volume FUSE path (serverless-compatible)
# MAGIC - Registers chart metadata in `raw.charts`
# MAGIC
# MAGIC **Clinical profiles included:**
# MAGIC - Uncontrolled T2DM with metabolic syndrome
# MAGIC - Diabetic CKD with anemia
# MAGIC - CHF with CAD and atrial fibrillation
# MAGIC - COPD exacerbation with thyroid/GERD comorbidities
# MAGIC - Prediabetes with sleep apnea and anxiety
# MAGIC - Hyperthyroidism with fatty liver
# MAGIC - Depression with diabetes and lipid disorders
# MAGIC - Elderly polypharmacy with neuropathy and CKD
# MAGIC - Routine wellness with controlled conditions
# MAGIC - Complex multi-morbidity (new patient transfer)
# MAGIC
# MAGIC **Estimated runtime:** ~2 minutes

# COMMAND ----------

# MAGIC %pip install fpdf2
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("CATALOG", "mv_catalog", "Unity Catalog Name")
CATALOG = dbutils.widgets.get("CATALOG")

VOLUME_PATH = f"/Volumes/{CATALOG}/raw/chart_pdfs"

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clinical Data Templates
# MAGIC
# MAGIC Each profile represents a realistic clinical encounter with diagnoses,
# MAGIC lab results (including specimen type, method, and timing for LOINC disambiguation),
# MAGIC medications, and vitals.

# COMMAND ----------

import random
import uuid
from datetime import datetime, timedelta
from fpdf import FPDF
import os

random.seed(42)

# --- Patient name pools ---
FIRST_NAMES_M = ["James", "Robert", "Michael", "David", "William", "Richard", "Thomas", "Charles", "Daniel", "Joseph",
                 "Anthony", "Mark", "Steven", "Paul", "Andrew", "Kenneth", "George", "Edward", "Brian", "Ronald",
                 "Kevin", "Jason", "Matthew", "Gary", "Timothy", "Jose", "Larry", "Jeffrey", "Frank", "Raymond"]
FIRST_NAMES_F = ["Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan", "Jessica", "Sarah", "Karen",
                 "Lisa", "Nancy", "Betty", "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily", "Donna",
                 "Michelle", "Carol", "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
              "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
              "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
              "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores"]

PROVIDERS = [
    ("Dr. Sarah Chen", "Internal Medicine"),
    ("Dr. Michael Patel", "Family Medicine"),
    ("Dr. Lisa Rodriguez", "Endocrinology"),
    ("Dr. James Kim", "Cardiology"),
    ("Dr. Amanda Foster", "Nephrology"),
    ("Dr. Robert Okonkwo", "Internal Medicine"),
    ("Dr. Maria Santos", "Family Medicine"),
    ("Dr. David Goldstein", "Pulmonology"),
]

FACILITIES = [
    "Valley Medical Center", "Sunrise Health Clinic", "Heritage Primary Care",
    "Lakeside Medical Group", "Mountain View Health Partners", "Pacific Medical Associates",
    "Community Health Center", "Metro Internal Medicine",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Diagnosis Profiles
# MAGIC
# MAGIC Each profile contains diagnoses with clinical context, lab results with
# MAGIC specimen/method/timing detail (critical for LOINC disambiguation), medications, and vitals.

# COMMAND ----------

DIAGNOSIS_PROFILES = [
    {
        "name": "diabetes_uncontrolled",
        "diagnoses": [
            ("Type 2 diabetes mellitus with hyperglycemia", "Patient has been non-compliant with metformin. Last HbA1c was 9.1%, up from 7.8% six months ago. Increased thirst and urination reported."),
            ("Essential hypertension", "Blood pressure remains elevated at 148/92. Currently on lisinopril 20mg daily."),
            ("Dyslipidemia", "LDL cholesterol elevated at 156 mg/dL despite atorvastatin 20mg. Triglycerides 245 mg/dL."),
            ("Morbid obesity due to excess calories", "BMI 42.3. Weight 285 lbs. Patient reports difficulty with exercise due to knee pain."),
        ],
        "labs": [
            ("Hemoglobin A1c", "9.1", "%", "whole blood", "immunoassay", "point in time"),
            ("Fasting glucose", "186", "mg/dL", "serum", "", "fasting"),
            ("Total cholesterol", "248", "mg/dL", "serum", "", "fasting"),
            ("LDL cholesterol", "156", "mg/dL", "serum", "calculated (Friedewald)", "fasting"),
            ("HDL cholesterol", "38", "mg/dL", "serum", "", "fasting"),
            ("Triglycerides", "245", "mg/dL", "serum", "", "fasting"),
            ("Creatinine", "1.2", "mg/dL", "serum", "", ""),
            ("eGFR", "62", "mL/min/1.73m2", "serum", "CKD-EPI equation", ""),
            ("Urine albumin-to-creatinine ratio", "45", "mg/g", "urine", "", "random spot"),
            ("Potassium", "4.5", "mEq/L", "serum", "", ""),
        ],
        "meds": ["Metformin 1000mg BID", "Lisinopril 20mg daily", "Atorvastatin 20mg daily", "Aspirin 81mg daily"],
        "vitals": {"BP": "148/92", "HR": "82", "Temp": "98.4F", "RR": "16", "SpO2": "97%", "Weight": "285 lbs", "BMI": "42.3"},
    },
    {
        "name": "ckd_diabetes",
        "diagnoses": [
            ("Type 2 diabetes mellitus with diabetic chronic kidney disease", "Diabetes diagnosed 12 years ago. Progressive decline in renal function. Currently stage 3b CKD."),
            ("Chronic kidney disease, stage 3", "eGFR 38 mL/min, down from 45 six months ago. Proteinuria present on urine studies."),
            ("Essential hypertension", "Blood pressure at goal on current regimen. 132/78 today."),
            ("Iron deficiency anemia", "Hemoglobin 10.2, ferritin 18. Likely related to CKD and chronic disease."),
        ],
        "labs": [
            ("Hemoglobin A1c", "7.4", "%", "whole blood", "HPLC", "point in time"),
            ("Creatinine", "1.8", "mg/dL", "serum", "", ""),
            ("eGFR", "38", "mL/min/1.73m2", "serum", "CKD-EPI equation", ""),
            ("BUN", "32", "mg/dL", "serum", "", ""),
            ("Urine albumin-to-creatinine ratio", "320", "mg/g", "urine", "", "random spot"),
            ("Hemoglobin", "10.2", "g/dL", "whole blood", "automated analyzer", ""),
            ("Hematocrit", "31.5", "%", "whole blood", "automated analyzer", ""),
            ("Ferritin", "18", "ng/mL", "serum", "", ""),
            ("Iron", "42", "mcg/dL", "serum", "", ""),
            ("TIBC", "420", "mcg/dL", "serum", "", ""),
            ("Potassium", "5.1", "mEq/L", "serum", "", ""),
            ("Calcium", "8.8", "mg/dL", "serum", "", ""),
            ("Phosphorus", "4.8", "mg/dL", "serum", "", ""),
        ],
        "meds": ["Insulin glargine 28 units nightly", "Lisinopril 40mg daily", "Amlodipine 10mg daily", "Ferrous sulfate 325mg TID", "Sodium bicarbonate 650mg TID"],
        "vitals": {"BP": "132/78", "HR": "76", "Temp": "98.2F", "RR": "18", "SpO2": "96%", "Weight": "198 lbs", "BMI": "31.2"},
    },
    {
        "name": "cardiac_hf",
        "diagnoses": [
            ("Chronic systolic congestive heart failure", "EF 35% on last echo. NYHA Class II. Stable on current medications. No recent hospitalizations."),
            ("Atherosclerotic heart disease of native coronary artery", "History of MI 2019, s/p PCI to LAD with drug-eluting stent. No recurrent angina."),
            ("Unspecified atrial fibrillation", "Paroxysmal atrial fibrillation, rate-controlled. On anticoagulation."),
            ("Essential hypertension", "Blood pressure well controlled at 118/72 on current regimen."),
            ("Type 2 diabetes mellitus without complications", "HbA1c at goal 6.8%. Diet-controlled with metformin."),
        ],
        "labs": [
            ("Hemoglobin A1c", "6.8", "%", "whole blood", "", "point in time"),
            ("Total cholesterol", "162", "mg/dL", "serum", "", "fasting"),
            ("LDL cholesterol", "68", "mg/dL", "serum", "direct assay", "fasting"),
            ("HDL cholesterol", "45", "mg/dL", "serum", "", "fasting"),
            ("Triglycerides", "155", "mg/dL", "serum", "", "fasting"),
            ("Creatinine", "1.1", "mg/dL", "serum", "", ""),
            ("eGFR", "68", "mL/min/1.73m2", "serum", "CKD-EPI equation", ""),
            ("Sodium", "138", "mEq/L", "serum", "", ""),
            ("Potassium", "4.2", "mEq/L", "serum", "", ""),
            ("BNP", "285", "pg/mL", "plasma", "", ""),
            ("INR", "2.4", "", "whole blood", "coagulation assay", ""),
            ("PT", "27.2", "seconds", "platelet-poor plasma", "coagulation assay", ""),
            ("Hemoglobin", "13.1", "g/dL", "whole blood", "automated analyzer", ""),
        ],
        "meds": ["Metformin 500mg BID", "Carvedilol 25mg BID", "Lisinopril 20mg daily", "Spironolactone 25mg daily",
                 "Furosemide 40mg daily", "Warfarin 5mg daily", "Rosuvastatin 20mg daily", "Aspirin 81mg daily"],
        "vitals": {"BP": "118/72", "HR": "68", "Temp": "98.6F", "RR": "16", "SpO2": "95%", "Weight": "210 lbs", "BMI": "29.5"},
    },
    {
        "name": "copd_metabolic",
        "diagnoses": [
            ("Chronic obstructive pulmonary disease with acute exacerbation", "Patient presents with increased dyspnea and productive cough x 3 days. Yellow-green sputum. FEV1 45% predicted at last PFT."),
            ("Essential hypertension", "Blood pressure elevated today likely related to respiratory distress. 156/94."),
            ("Hypothyroidism, unspecified", "Stable on levothyroxine. TSH within normal limits."),
            ("Gastro-esophageal reflux disease with esophagitis", "Ongoing symptoms despite PPI. EGD 6 months ago showed Grade B esophagitis."),
            ("Personal history of nicotine dependence", "Former smoker, quit 3 years ago. 40 pack-year history."),
        ],
        "labs": [
            ("TSH", "2.8", "mIU/L", "serum", "", ""),
            ("Free T4", "1.2", "ng/dL", "serum", "", ""),
            ("CBC - WBC", "12.4", "x10^3/uL", "whole blood", "automated analyzer", ""),
            ("Hemoglobin", "14.8", "g/dL", "whole blood", "automated analyzer", ""),
            ("Hematocrit", "44.2", "%", "whole blood", "automated analyzer", ""),
            ("Platelets", "268", "x10^3/uL", "whole blood", "automated analyzer", ""),
            ("CRP", "18.5", "mg/L", "serum", "", ""),
            ("Total cholesterol", "198", "mg/dL", "serum", "", ""),
            ("Creatinine", "0.9", "mg/dL", "serum", "", ""),
            ("eGFR", "82", "mL/min/1.73m2", "serum", "CKD-EPI equation", ""),
        ],
        "meds": ["Levothyroxine 75mcg daily", "Tiotropium 18mcg inhaled daily", "Albuterol PRN", "Omeprazole 40mg daily",
                 "Amlodipine 5mg daily", "Prednisone taper starting 40mg", "Azithromycin 500mg day 1 then 250mg x4 days"],
        "vitals": {"BP": "156/94", "HR": "96", "Temp": "99.8F", "RR": "22", "SpO2": "91%", "Weight": "175 lbs", "BMI": "25.8"},
    },
    {
        "name": "wellness_prediabetes",
        "diagnoses": [
            ("Prediabetes", "HbA1c 6.1%. Fasting glucose 112. Patient counseled on lifestyle modifications. Will recheck in 3 months."),
            ("Pure hypercholesterolemia, unspecified", "LDL 148 mg/dL. 10-year ASCVD risk 8.2%. Starting statin per ACC/AHA guidelines."),
            ("Obstructive sleep apnea", "Diagnosed via home sleep test last month. AHI 22. CPAP initiated, patient reports improved daytime alertness."),
            ("Low back pain", "Chronic low back pain, mechanical. No red flags. Managed with PT and NSAIDs PRN."),
            ("Generalized anxiety disorder", "Managed with sertraline 50mg. GAD-7 score 8, improved from 14."),
        ],
        "labs": [
            ("Hemoglobin A1c", "6.1", "%", "whole blood", "immunoassay", "point in time"),
            ("Fasting glucose", "112", "mg/dL", "serum/plasma", "", "fasting"),
            ("Total cholesterol", "232", "mg/dL", "serum", "", "fasting"),
            ("LDL cholesterol", "148", "mg/dL", "serum", "calculated (Friedewald)", "fasting"),
            ("HDL cholesterol", "52", "mg/dL", "serum", "", "fasting"),
            ("Triglycerides", "160", "mg/dL", "serum", "", "fasting"),
            ("TSH", "1.9", "mIU/L", "serum", "", ""),
            ("Vitamin D, 25-hydroxy", "22", "ng/mL", "serum", "", ""),
            ("Vitamin B12", "380", "pg/mL", "serum", "", ""),
            ("CBC - Hemoglobin", "14.2", "g/dL", "whole blood", "automated analyzer", ""),
            ("AST", "28", "U/L", "serum", "", ""),
            ("ALT", "35", "U/L", "serum", "", ""),
        ],
        "meds": ["Sertraline 50mg daily", "Atorvastatin 10mg daily (new)", "Ibuprofen 400mg PRN", "CPAP nightly"],
        "vitals": {"BP": "128/82", "HR": "72", "Temp": "98.4F", "RR": "14", "SpO2": "98%", "Weight": "205 lbs", "BMI": "30.1"},
    },
    {
        "name": "thyroid_liver",
        "diagnoses": [
            ("Thyrotoxicosis, unspecified without thyrotoxic crisis", "New diagnosis. TSH suppressed at 0.05, Free T4 elevated at 3.2. Patient reports palpitations, weight loss, tremor. Thyroid uptake scan ordered."),
            ("Fatty liver, not elsewhere classified", "ALT 68, AST 52. Ultrasound shows moderate hepatic steatosis. No alcohol use."),
            ("Essential hypertension", "Blood pressure 142/88, likely exacerbated by hyperthyroid state."),
            ("Unspecified atrial fibrillation", "New onset, likely thyroid-related. Rate 112. Starting beta-blocker."),
        ],
        "labs": [
            ("TSH", "0.05", "mIU/L", "serum", "", ""),
            ("Free T4", "3.2", "ng/dL", "serum", "", ""),
            ("Free T3", "8.8", "pg/mL", "serum", "", ""),
            ("ALT", "68", "U/L", "serum", "", ""),
            ("AST", "52", "U/L", "serum", "", ""),
            ("Alkaline phosphatase", "98", "U/L", "serum", "", ""),
            ("Total bilirubin", "1.1", "mg/dL", "serum", "", ""),
            ("Albumin", "3.8", "g/dL", "serum", "", ""),
            ("Hemoglobin A1c", "5.4", "%", "whole blood", "", "point in time"),
            ("Total cholesterol", "168", "mg/dL", "serum", "", "fasting"),
            ("Creatinine", "0.8", "mg/dL", "serum", "", ""),
            ("eGFR", "95", "mL/min/1.73m2", "serum", "CKD-EPI equation", ""),
        ],
        "meds": ["Methimazole 10mg TID", "Propranolol 20mg TID", "Amlodipine 5mg daily"],
        "vitals": {"BP": "142/88", "HR": "112", "Temp": "99.2F", "RR": "18", "SpO2": "98%", "Weight": "155 lbs", "BMI": "23.1"},
    },
    {
        "name": "depression_metabolic",
        "diagnoses": [
            ("Major depressive disorder, single episode, moderate", "PHQ-9 score 16. Patient reports persistent sadness, insomnia, poor appetite x 6 weeks. No SI/HI. Starting SSRI."),
            ("Type 2 diabetes mellitus without complications", "Diabetes well-controlled. HbA1c 6.9% on metformin alone."),
            ("Mixed hyperlipidemia", "Both cholesterol and triglycerides elevated. LDL 142, TG 220."),
            ("Anemia, unspecified", "Hemoglobin 11.2. Iron studies pending. Patient reports fatigue which may be multifactorial."),
        ],
        "labs": [
            ("Hemoglobin A1c", "6.9", "%", "whole blood", "HPLC", "point in time"),
            ("Fasting glucose", "128", "mg/dL", "serum", "", "fasting"),
            ("Total cholesterol", "238", "mg/dL", "serum", "", "fasting"),
            ("LDL cholesterol", "142", "mg/dL", "serum", "calculated (Friedewald)", "fasting"),
            ("HDL cholesterol", "42", "mg/dL", "serum", "", "fasting"),
            ("Triglycerides", "220", "mg/dL", "serum", "", "fasting"),
            ("Hemoglobin", "11.2", "g/dL", "whole blood", "automated analyzer", ""),
            ("Hematocrit", "34.1", "%", "whole blood", "automated analyzer", ""),
            ("WBC", "7.2", "x10^3/uL", "whole blood", "automated analyzer", ""),
            ("Platelets", "312", "x10^3/uL", "whole blood", "automated analyzer", ""),
            ("TSH", "3.4", "mIU/L", "serum", "", ""),
            ("Vitamin B12", "245", "pg/mL", "serum", "", ""),
            ("Folate", "8.2", "ng/mL", "serum", "", ""),
            ("Creatinine", "0.7", "mg/dL", "serum", "", ""),
        ],
        "meds": ["Metformin 1000mg BID", "Escitalopram 10mg daily (new)", "Atorvastatin 40mg daily", "Trazodone 50mg QHS PRN"],
        "vitals": {"BP": "136/84", "HR": "78", "Temp": "98.6F", "RR": "16", "SpO2": "98%", "Weight": "192 lbs", "BMI": "28.4"},
    },
    {
        "name": "elderly_polypharm",
        "diagnoses": [
            ("Type 2 diabetes mellitus with diabetic neuropathy, unspecified", "Tingling and numbness in bilateral feet. Monofilament exam shows decreased sensation. HbA1c 7.8%."),
            ("Hypertensive heart disease without heart failure", "Echo shows concentric LVH. EF 55%. No diastolic dysfunction."),
            ("Chronic kidney disease, stage 3", "eGFR 52, stable over past year. Proteinuria present."),
            ("Primary osteoarthritis, right knee", "Crepitus and decreased ROM. X-ray shows moderate joint space narrowing."),
            ("Vitamin D deficiency", "25-OH Vitamin D 14 ng/mL. Starting supplementation."),
        ],
        "labs": [
            ("Hemoglobin A1c", "7.8", "%", "whole blood", "immunoassay", "point in time"),
            ("Fasting glucose", "156", "mg/dL", "serum", "", "fasting"),
            ("Creatinine", "1.4", "mg/dL", "serum", "", ""),
            ("eGFR", "52", "mL/min/1.73m2", "serum", "CKD-EPI equation", ""),
            ("BUN", "28", "mg/dL", "serum", "", ""),
            ("Microalbumin, urine", "85", "mg/L", "urine", "", "random spot"),
            ("Vitamin D, 25-hydroxy total", "14", "ng/mL", "serum", "", ""),
            ("Calcium", "9.2", "mg/dL", "serum", "", ""),
            ("Sodium", "140", "mEq/L", "serum", "", ""),
            ("Potassium", "4.8", "mEq/L", "serum", "", ""),
            ("Hemoglobin", "12.8", "g/dL", "whole blood", "automated analyzer", ""),
            ("PSA, total", "3.2", "ng/mL", "serum", "", ""),
            ("ESR", "28", "mm/hr", "whole blood", "Westergren method", ""),
        ],
        "meds": ["Insulin glargine 22 units nightly", "Metformin 500mg BID", "Losartan 100mg daily", "Amlodipine 10mg daily",
                 "Gabapentin 300mg TID", "Vitamin D3 5000 IU daily", "Acetaminophen 500mg PRN"],
        "vitals": {"BP": "138/76", "HR": "70", "Temp": "98.0F", "RR": "16", "SpO2": "96%", "Weight": "182 lbs", "BMI": "27.8"},
    },
]

# Additional simpler profiles for variety
EXTRA_PROFILES = [
    {
        "name": "annual_wellness",
        "diagnoses": [
            ("Essential hypertension", "Well controlled on current medications. BP 124/78."),
            ("Type 2 diabetes mellitus without complications", "HbA1c 6.5%, at goal. Continue current regimen."),
        ],
        "labs": [
            ("Hemoglobin A1c", "6.5", "%", "whole blood", "", "point in time"),
            ("Total cholesterol", "188", "mg/dL", "serum", "", "fasting"),
            ("LDL cholesterol", "102", "mg/dL", "serum", "calculated (Friedewald)", "fasting"),
            ("HDL cholesterol", "55", "mg/dL", "serum", "", "fasting"),
            ("Triglycerides", "130", "mg/dL", "serum", "", "fasting"),
            ("Creatinine", "0.9", "mg/dL", "serum", "", ""),
            ("eGFR", "85", "mL/min/1.73m2", "serum", "CKD-EPI equation", ""),
        ],
        "meds": ["Metformin 500mg BID", "Lisinopril 10mg daily"],
        "vitals": {"BP": "124/78", "HR": "72", "Temp": "98.6F", "RR": "14", "SpO2": "99%", "Weight": "175 lbs", "BMI": "26.1"},
    },
    {
        "name": "new_patient_complex",
        "diagnoses": [
            ("Type 2 diabetes mellitus with unspecified diabetic retinopathy without macular edema", "Fundoscopic exam shows dot-blot hemorrhages OU. Referred to ophthalmology."),
            ("Heart failure, unspecified", "New diagnosis. Echo pending. BNP elevated. Bilateral lower extremity edema."),
            ("Chronic obstructive pulmonary disease, unspecified", "Former smoker. PFTs show FEV1/FVC 0.62. Starting bronchodilators."),
            ("Moderate persistent asthma, uncomplicated", "Overlap with COPD likely. Eosinophils elevated. ICS/LABA prescribed."),
        ],
        "labs": [
            ("Hemoglobin A1c", "8.4", "%", "whole blood", "HPLC", "point in time"),
            ("Fasting glucose", "168", "mg/dL", "serum/plasma", "", "fasting"),
            ("Creatinine", "1.3", "mg/dL", "serum", "", ""),
            ("eGFR", "55", "mL/min/1.73m2", "serum", "MDRD equation", ""),
            ("BNP", "580", "pg/mL", "plasma", "", ""),
            ("hs-CRP", "8.2", "mg/L", "serum", "high sensitivity method", ""),
            ("Hemoglobin", "12.5", "g/dL", "whole blood", "automated analyzer", ""),
            ("WBC", "8.8", "x10^3/uL", "whole blood", "automated analyzer", ""),
            ("Sodium", "136", "mEq/L", "serum", "", ""),
            ("Potassium", "4.6", "mEq/L", "serum", "", ""),
            ("Total bilirubin", "0.8", "mg/dL", "serum", "", ""),
            ("ALT", "42", "U/L", "serum", "", ""),
            ("INR", "1.1", "", "whole blood", "coagulation assay", ""),
        ],
        "meds": ["Insulin lispro sliding scale", "Metformin 1000mg BID", "Furosemide 20mg daily", "Budesonide/formoterol inhaler BID",
                 "Albuterol PRN", "Lisinopril 20mg daily"],
        "vitals": {"BP": "152/88", "HR": "92", "Temp": "98.8F", "RR": "20", "SpO2": "93%", "Weight": "245 lbs", "BMI": "35.2"},
    },
]

ALL_PROFILES = DIAGNOSIS_PROFILES + EXTRA_PROFILES

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDF Generation Functions

# COMMAND ----------

def generate_chart_text(profile, patient_name, age, sex, provider, facility, chart_date):
    """Generate realistic clinical note text."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"{facility}")
    lines.append(f"CLINICAL NOTE")
    lines.append(f"{'='*60}")
    lines.append(f"")
    lines.append(f"Patient: {patient_name}")
    lines.append(f"DOB: {(chart_date - timedelta(days=age*365)).strftime('%m/%d/%Y')}  Age: {age}  Sex: {sex}")
    lines.append(f"Date of Service: {chart_date.strftime('%m/%d/%Y')}")
    lines.append(f"Provider: {provider[0]} ({provider[1]})")
    lines.append(f"Visit Type: {'Follow-up' if random.random() > 0.3 else 'New Patient'}")
    lines.append(f"")

    # Chief Complaint
    complaints = {
        "diabetes_uncontrolled": "Follow-up for diabetes management. Reports increased thirst.",
        "ckd_diabetes": "Routine CKD and diabetes follow-up. Fatigue worsening.",
        "cardiac_hf": "Heart failure check. No change in exercise tolerance.",
        "copd_metabolic": "Acute worsening of cough and shortness of breath x3 days.",
        "wellness_prediabetes": "Annual wellness visit. Concerns about weight and sleep.",
        "thyroid_liver": "New patient. Palpitations, weight loss, anxiety x2 months.",
        "depression_metabolic": "Follow-up depression. Also checking diabetes labs.",
        "elderly_polypharm": "Routine follow-up. Numbness in feet worsening.",
        "annual_wellness": "Annual wellness examination. No new complaints.",
        "new_patient_complex": "New patient transfer of care. Multiple chronic conditions.",
    }
    lines.append(f"CHIEF COMPLAINT:")
    lines.append(f"{complaints.get(profile['name'], 'Follow-up visit.')}")
    lines.append(f"")

    # Vitals
    lines.append(f"VITAL SIGNS:")
    for k, v in profile["vitals"].items():
        lines.append(f"  {k}: {v}")
    lines.append(f"")

    # Assessment and Plan
    lines.append(f"ASSESSMENT AND PLAN:")
    lines.append(f"")
    for i, (dx, note) in enumerate(profile["diagnoses"], 1):
        lines.append(f"{i}. {dx}")
        lines.append(f"   {note}")
        lines.append(f"")

    # Labs
    lines.append(f"LABORATORY RESULTS:")
    lines.append(f"  Specimen Collection Date: {chart_date.strftime('%m/%d/%Y')}")
    lines.append(f"  Lab: Quest Diagnostics" if random.random() > 0.5 else f"  Lab: LabCorp")
    lines.append(f"")
    for lab in profile["labs"]:
        name, value, unit, specimen, method, timing = lab
        line = f"  {name}: {value} {unit}"
        details = []
        if specimen:
            details.append(f"specimen: {specimen}")
        if method:
            details.append(f"method: {method}")
        if timing:
            details.append(f"timing: {timing}")
        if details:
            line += f"  ({', '.join(details)})"
        lines.append(line)
    lines.append(f"")

    # Medications
    lines.append(f"CURRENT MEDICATIONS:")
    for med in profile["meds"]:
        lines.append(f"  - {med}")
    lines.append(f"")

    # Plan
    plans = [
        "Follow-up in 3 months with repeat labs.",
        "Patient educated on medication compliance.",
        "Referral to nutrition counseling placed.",
        "Will order imaging as indicated above.",
        "Return to clinic in 6 weeks for reassessment.",
        "Discussed treatment options. Patient agrees with plan.",
    ]
    lines.append(f"PLAN:")
    for p in random.sample(plans, min(3, len(plans))):
        lines.append(f"  - {p}")
    lines.append(f"")
    lines.append(f"Electronically signed by {provider[0]}")
    lines.append(f"Date: {chart_date.strftime('%m/%d/%Y %I:%M %p')}")

    return "\n".join(lines)


def create_pdf(text, filepath):
    """Create a PDF from clinical note text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", size=9)
    pdf.set_auto_page_break(auto=True, margin=15)

    for line in text.split("\n"):
        if line.startswith("=") or line.startswith("CHIEF") or line.startswith("VITAL") or \
           line.startswith("ASSESSMENT") or line.startswith("LABORATORY") or line.startswith("CURRENT MED") or \
           line.startswith("PLAN:"):
            pdf.set_font("Courier", style="B", size=9)
            pdf.cell(0, 4, line.encode('latin-1', 'replace').decode('latin-1'), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Courier", size=9)
        else:
            pdf.cell(0, 4, line.encode('latin-1', 'replace').decode('latin-1'), new_x="LMARGIN", new_y="NEXT")

    pdf.output(filepath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate 100 Charts
# MAGIC
# MAGIC Writes PDFs directly to the UC Volume FUSE path. This works on both
# MAGIC serverless and classic compute without needing `dbutils.fs.cp`.

# COMMAND ----------

import os

# Write PDFs directly to UC Volume FUSE path (serverless-compatible)
local_dir = VOLUME_PATH
os.makedirs(local_dir, exist_ok=True)
print(f"  Writing PDFs to: {local_dir}")

charts_metadata = []

for i in range(100):
    sex = random.choice(["M", "F"])
    first = random.choice(FIRST_NAMES_M) if sex == "M" else random.choice(FIRST_NAMES_F)
    last = random.choice(LAST_NAMES)
    patient_name = f"{first} {last}"
    patient_id = f"PAT-{uuid.uuid4().hex[:8].upper()}"
    age = random.randint(35, 85)

    profile = random.choice(ALL_PROFILES)
    provider = random.choice(PROVIDERS)
    facility = random.choice(FACILITIES)
    chart_date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 365))
    chart_id = f"CHT-{uuid.uuid4().hex[:12].upper()}"

    text = generate_chart_text(profile, patient_name, age, sex, provider, facility, chart_date)

    filename = f"chart_{i+1:03d}_{patient_id}.pdf"
    filepath = os.path.join(local_dir, filename)
    create_pdf(text, filepath)

    charts_metadata.append({
        "chart_id": chart_id,
        "patient_id": patient_id,
        "file_name": filename,
        "chart_type": "clinical_note",
        "provider": provider[0],
        "facility": facility,
        "chart_date": chart_date.strftime("%Y-%m-%d"),
        "page_count": 1,
        "profile": profile["name"],
        "raw_text": text,
    })

print(f"  Generated {len(charts_metadata)} PDFs in {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify and Register Charts

# COMMAND ----------

pdf_files = [f for f in os.listdir(local_dir) if f.endswith('.pdf')]
print(f"  {len(pdf_files)} PDFs verified in {VOLUME_PATH}")

# COMMAND ----------

from pyspark.sql.functions import lit, current_timestamp, to_date
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

schema = StructType([
    StructField("chart_id", StringType()),
    StructField("patient_id", StringType()),
    StructField("file_name", StringType()),
    StructField("chart_type", StringType()),
    StructField("provider", StringType()),
    StructField("facility", StringType()),
    StructField("chart_date", StringType()),
    StructField("page_count", IntegerType()),
    StructField("profile", StringType()),
    StructField("raw_text", StringType()),
])

charts_df = spark.createDataFrame(charts_metadata, schema=schema)

charts_final = (charts_df
    .withColumn("file_path", lit(VOLUME_PATH).cast("string"))
    .withColumn("ingested_at", current_timestamp())
    .withColumn("extraction_method", lit("synthetic_generation"))
    .withColumn("chart_date", to_date("chart_date"))
    .select(
        "chart_id", "patient_id", "file_name",
        "file_path", "chart_type", "provider", "facility",
        "chart_date", "ingested_at", "raw_text", "page_count", "extraction_method"
    )
)

charts_final.write.mode("overwrite").saveAsTable(f"{CATALOG}.raw.charts")
print(f"  Registered {charts_final.count()} charts in {CATALOG}.raw.charts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Generated 100 synthetic clinical chart PDFs across 10 clinical profiles.
# MAGIC Each chart includes realistic diagnoses, labs (with specimen/method/timing detail),
# MAGIC medications, and vitals.
# MAGIC
# MAGIC **Next:** Run `04_extract_entities` to extract clinical entities using AI.
