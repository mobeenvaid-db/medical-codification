# Databricks notebook source
# MAGIC %md
# MAGIC # 02b -- Load Medical Ontologies (UMLS, SNOMED, RxNorm)
# MAGIC
# MAGIC Loads UMLS Metathesaurus, SNOMED-CT hierarchy, RxNorm drug concepts,
# MAGIC and a comprehensive clinical abbreviation dictionary, then builds a unified
# MAGIC search-optimized medical dictionary from all reference sources.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## PREREQUISITES (One-Time Setup -- Read This First)
# MAGIC
# MAGIC All four ontology datasets (UMLS, SNOMED, RxNorm, SNOMED-ICD10 map) are available
# MAGIC from the **NLM (National Library of Medicine) UTS Download API** using a single API key.
# MAGIC This notebook can download them **programmatically** -- no manual file uploads needed.
# MAGIC
# MAGIC ### Step 1: Create a Free NLM/UTS Account (5 minutes, one-time)
# MAGIC 1. Go to https://uts.nlm.nih.gov/uts/signup-login
# MAGIC 2. Create a free account (NIH login or standalone)
# MAGIC 3. Sign the **UMLS Metathesaurus License Agreement** (covers UMLS, SNOMED, and RxNorm)
# MAGIC 4. Go to **My Profile** (https://uts.nlm.nih.gov/uts/profile) and copy your **API Key**
# MAGIC
# MAGIC ### Step 2: Store the API Key in Databricks
# MAGIC **Recommended:** Store as a Databricks secret:
# MAGIC ```
# MAGIC databricks secrets create-scope medical-ontology
# MAGIC databricks secrets put-secret medical-ontology umls-api-key --string-value "YOUR_KEY_HERE"
# MAGIC ```
# MAGIC **Alternative:** Enter it in the `umls_api_key` widget when running this notebook.
# MAGIC
# MAGIC ### Step 3: Run This Notebook
# MAGIC The notebook will automatically:
# MAGIC 1. Download UMLS Metathesaurus (~35 GB full, or subset via API)
# MAGIC 2. Download SNOMED-CT US Edition (~1-2 GB) including ICD-10 mapping
# MAGIC 3. Download RxNorm full release (~200 MB)
# MAGIC 4. Parse all files and build the unified medical dictionary
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Data Loading Priority (Per Dataset)
# MAGIC Each dataset uses a 3-tier loading strategy:
# MAGIC 1. **API Download** (preferred) -- automated download via UTS API with your key
# MAGIC 2. **UC Volume** (fallback) -- load from pre-uploaded files in UC Volumes
# MAGIC 3. **REST API** (last resort) -- individual concept lookups (slow, for small subsets only)
# MAGIC
# MAGIC ### What Gets Loaded
# MAGIC | Table | Source | Records | Description |
# MAGIC |-------|--------|---------|-------------|
# MAGIC | `reference.clinical_abbreviations` | Hardcoded | ~340 | Common clinical abbreviations |
# MAGIC | `reference.umls_concepts` | UMLS MRCONSO.RRF + MRSTY.RRF | ~2.3M concepts | Concept dictionary for entity linking |
# MAGIC | `reference.snomed_hierarchy` | SNOMED RF2 files | ~350K concepts | IS-A hierarchy for ontology traversal |
# MAGIC | `reference.snomed_icd10_map` | SNOMED RF2 mapping file | ~100K maps | Official SNOMED-to-ICD-10 crosswalk |
# MAGIC | `reference.rxnorm_concepts` | RxNorm RXNCONSO.RRF | ~115K concepts | Drug normalization concepts |
# MAGIC | `reference.medical_dictionary` | All of the above + ICD-10 + LOINC | ~3M+ terms | Unified term dictionary for entity detection |
# MAGIC
# MAGIC ### UC Volume Locations (If Uploading Manually)
# MAGIC - UMLS: `/Volumes/{CATALOG}/reference/umls_files/` (MRCONSO.RRF, MRSTY.RRF)
# MAGIC - SNOMED: `/Volumes/{CATALOG}/reference/snomed_files/` (RF2 Snapshot)
# MAGIC - RxNorm: `/Volumes/{CATALOG}/reference/rxnorm_files/` (RXNCONSO.RRF)
# MAGIC
# MAGIC ### Graceful Degradation
# MAGIC If no API key is provided and no files are uploaded, the notebook still builds
# MAGIC a useful dictionary from ICD-10 + LOINC + abbreviations (~280K terms). Each
# MAGIC ontology source is additive -- load what you can, and the dictionary improves.
# MAGIC
# MAGIC **Estimated runtime:** 5-30 minutes (depends on download speed for first run; subsequent runs use cached files in UC Volumes)

# COMMAND ----------

# MAGIC %pip install requests
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration & API Key Setup

# COMMAND ----------

dbutils.widgets.text("CATALOG", "mv_catalog", "Unity Catalog Name")
dbutils.widgets.text("umls_api_key", "", "UTS/UMLS API Key (or use Databricks secret)")
CATALOG = dbutils.widgets.get("CATALOG")

spark.sql(f"USE CATALOG {CATALOG}")

MODEL = "databricks-claude-sonnet-4-6"

# --- Resolve API Key ---
# Priority: Databricks secret > widget > empty (skip API downloads)
UTS_API_KEY = ""
try:
    UTS_API_KEY = dbutils.secrets.get("medical-ontology", "umls-api-key")
    print("  UTS API key loaded from Databricks secrets (medical-ontology/umls-api-key)")
except Exception:
    UTS_API_KEY = dbutils.widgets.get("umls_api_key").strip()
    if UTS_API_KEY:
        print("  UTS API key provided via widget")
    else:
        print("  No UTS API key found. Automated downloads will be skipped.")
        print("  To enable: store key in secrets or enter in the umls_api_key widget.")
        print("  Get your free key at: https://uts.nlm.nih.gov/uts/profile")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create UC Volumes for Ontology Files
# MAGIC
# MAGIC These volumes serve as a local cache. Once files are downloaded via API,
# MAGIC they are stored here so subsequent runs skip the download.

# COMMAND ----------

for vol_name in ["umls_files", "snomed_files", "rxnorm_files", "download_temp"]:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.reference.{vol_name}")
    print(f"  Volume {CATALOG}.reference.{vol_name} ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## UTS Download API Utility
# MAGIC
# MAGIC The NLM provides a unified bulk download API at `https://uts-ws.nlm.nih.gov/download`.
# MAGIC All datasets (UMLS, SNOMED CT US Edition, RxNorm) use the same endpoint with
# MAGIC just the API key and the target file URL.

# COMMAND ----------

import requests
import zipfile
import os
import shutil

DOWNLOAD_TEMP = f"/Volumes/{CATALOG}/reference/download_temp"

def uts_download(file_url, dest_volume, extract=True, required_files=None):
    """
    Download a file from NLM via the UTS Download API.

    Args:
        file_url: The NLM download URL (e.g., https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_current.zip)
        dest_volume: UC Volume path to store extracted files
        extract: Whether to extract ZIP contents
        required_files: List of filenames to extract (None = extract all). Case-insensitive matching.

    Returns:
        True if download succeeded, False otherwise
    """
    if not UTS_API_KEY:
        return False

    # Check if required files already exist in dest_volume (skip re-download)
    if required_files:
        try:
            existing = [f.name for f in dbutils.fs.ls(dest_volume)]
            if all(rf in existing for rf in required_files):
                print(f"  Files already cached in {dest_volume} -- skipping download")
                return True
        except Exception:
            pass

    print(f"  Downloading from NLM: {file_url.split('/')[-1]}...")

    try:
        # UTS download API with API key
        download_url = "https://uts-ws.nlm.nih.gov/download"
        resp = requests.get(
            download_url,
            params={"url": file_url, "apiKey": UTS_API_KEY},
            stream=True,
            timeout=600,
            allow_redirects=True
        )

        if resp.status_code == 401:
            print("  Authentication failed. Check your UTS API key.")
            print("  Make sure you have signed the UMLS license agreement at https://uts.nlm.nih.gov")
            return False

        if resp.status_code != 200:
            print(f"  Download failed: HTTP {resp.status_code}")
            return False

        # Write to temp location
        zip_path = f"{DOWNLOAD_TEMP}/download.zip"
        total_bytes = 0
        with open(zip_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192 * 1024):  # 8MB chunks
                f.write(chunk)
                total_bytes += len(chunk)

        print(f"  Downloaded {total_bytes / (1024*1024):.1f} MB")

        if not extract:
            # Move the file directly
            dest_path = f"{dest_volume}/{file_url.split('/')[-1]}"
            shutil.move(zip_path, dest_path)
            return True

        # Extract ZIP
        print(f"  Extracting to {dest_volume}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue  # Skip directories

                # If required_files specified, only extract those
                if required_files:
                    if not any(filename.upper() == rf.upper() for rf in required_files):
                        # Also check partial match for SNOMED files with version numbers
                        if not any(rf.upper() in filename.upper() for rf in (required_files or [])):
                            continue

                dest_path = f"{dest_volume}/{filename}"
                with zf.open(member) as src, open(dest_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                    print(f"    Extracted: {filename}")

        # Clean up temp
        os.remove(zip_path)
        return True

    except requests.exceptions.Timeout:
        print("  Download timed out. The file may be very large. Try manual download.")
        return False
    except Exception as e:
        print(f"  Download error: {e}")
        return False

# --- NLM Download URLs ---
# These are the current production URLs for each dataset.
# The UTS download API handles authentication and redirects.

NLM_URLS = {
    "umls_full": "https://download.nlm.nih.gov/umls/kss/2025AB/umls-2025AB-metathesaurus-full.zip",
    "umls_mrconso_only": "https://download.nlm.nih.gov/umls/kss/2025AB/umls-2025AB-metathesaurus-mrconso.zip",
    "snomed_us": "https://download.nlm.nih.gov/mlb/utsauth/USExt/SnomedCT_USEditionRF2_PRODUCTION_20250301T120000Z.zip",
    "rxnorm_full": "https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_current.zip",
    "rxnorm_prescribable": "https://download.nlm.nih.gov/rxnorm/RxNorm_full_prescribe_current.zip",
}

print("  UTS Download utility ready")
if UTS_API_KEY:
    print("  API key present -- automated downloads enabled")
else:
    print("  No API key -- will look for pre-uploaded files in UC Volumes")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 1: Clinical Abbreviation Dictionary
# MAGIC
# MAGIC A comprehensive hardcoded dictionary of ~220 common clinical abbreviations
# MAGIC organized by category: diagnosis, medication, lab, procedure, general.
# MAGIC These are critical for entity detection since clinical notes are heavily
# MAGIC abbreviated.

# COMMAND ----------

from datetime import datetime

CLINICAL_ABBREVIATIONS = [
    # --- Diagnosis abbreviations ---
    ("HTN", "hypertension", "diagnosis", "ASSESSMENT_PLAN"),
    ("DM", "diabetes mellitus", "diagnosis", "ASSESSMENT_PLAN"),
    ("DM1", "type 1 diabetes mellitus", "diagnosis", "ASSESSMENT_PLAN"),
    ("DM2", "type 2 diabetes mellitus", "diagnosis", "ASSESSMENT_PLAN"),
    ("T2DM", "type 2 diabetes mellitus", "diagnosis", "ASSESSMENT_PLAN"),
    ("T1DM", "type 1 diabetes mellitus", "diagnosis", "ASSESSMENT_PLAN"),
    ("CHF", "congestive heart failure", "diagnosis", "ASSESSMENT_PLAN"),
    ("HF", "heart failure", "diagnosis", "ASSESSMENT_PLAN"),
    ("HFrEF", "heart failure with reduced ejection fraction", "diagnosis", "ASSESSMENT_PLAN"),
    ("HFpEF", "heart failure with preserved ejection fraction", "diagnosis", "ASSESSMENT_PLAN"),
    ("COPD", "chronic obstructive pulmonary disease", "diagnosis", "ASSESSMENT_PLAN"),
    ("CAD", "coronary artery disease", "diagnosis", "ASSESSMENT_PLAN"),
    ("MI", "myocardial infarction", "diagnosis", "ASSESSMENT_PLAN"),
    ("STEMI", "ST-elevation myocardial infarction", "diagnosis", "ASSESSMENT_PLAN"),
    ("NSTEMI", "non-ST-elevation myocardial infarction", "diagnosis", "ASSESSMENT_PLAN"),
    ("CVA", "cerebrovascular accident", "diagnosis", "ASSESSMENT_PLAN"),
    ("TIA", "transient ischemic attack", "diagnosis", "ASSESSMENT_PLAN"),
    ("DVT", "deep vein thrombosis", "diagnosis", "ASSESSMENT_PLAN"),
    ("PE", "pulmonary embolism", "diagnosis", "ASSESSMENT_PLAN"),
    ("VTE", "venous thromboembolism", "diagnosis", "ASSESSMENT_PLAN"),
    ("AFib", "atrial fibrillation", "diagnosis", "ASSESSMENT_PLAN"),
    ("AF", "atrial fibrillation", "diagnosis", "ASSESSMENT_PLAN"),
    ("AFlutter", "atrial flutter", "diagnosis", "ASSESSMENT_PLAN"),
    ("SVT", "supraventricular tachycardia", "diagnosis", "ASSESSMENT_PLAN"),
    ("VT", "ventricular tachycardia", "diagnosis", "ASSESSMENT_PLAN"),
    ("VFib", "ventricular fibrillation", "diagnosis", "ASSESSMENT_PLAN"),
    ("PNA", "pneumonia", "diagnosis", "ASSESSMENT_PLAN"),
    ("UTI", "urinary tract infection", "diagnosis", "ASSESSMENT_PLAN"),
    ("GERD", "gastroesophageal reflux disease", "diagnosis", "ASSESSMENT_PLAN"),
    ("CKD", "chronic kidney disease", "diagnosis", "ASSESSMENT_PLAN"),
    ("ESRD", "end-stage renal disease", "diagnosis", "ASSESSMENT_PLAN"),
    ("AKI", "acute kidney injury", "diagnosis", "ASSESSMENT_PLAN"),
    ("OSA", "obstructive sleep apnea", "diagnosis", "ASSESSMENT_PLAN"),
    ("RA", "rheumatoid arthritis", "diagnosis", "ASSESSMENT_PLAN"),
    ("OA", "osteoarthritis", "diagnosis", "ASSESSMENT_PLAN"),
    ("SLE", "systemic lupus erythematosus", "diagnosis", "ASSESSMENT_PLAN"),
    ("MS", "multiple sclerosis", "diagnosis", "ASSESSMENT_PLAN"),
    ("ALS", "amyotrophic lateral sclerosis", "diagnosis", "ASSESSMENT_PLAN"),
    ("PD", "Parkinson disease", "diagnosis", "ASSESSMENT_PLAN"),
    ("BPH", "benign prostatic hyperplasia", "diagnosis", "ASSESSMENT_PLAN"),
    ("ADHD", "attention deficit hyperactivity disorder", "diagnosis", "ASSESSMENT_PLAN"),
    ("MDD", "major depressive disorder", "diagnosis", "ASSESSMENT_PLAN"),
    ("GAD", "generalized anxiety disorder", "diagnosis", "ASSESSMENT_PLAN"),
    ("PTSD", "post-traumatic stress disorder", "diagnosis", "ASSESSMENT_PLAN"),
    ("OCD", "obsessive-compulsive disorder", "diagnosis", "ASSESSMENT_PLAN"),
    ("IBD", "inflammatory bowel disease", "diagnosis", "ASSESSMENT_PLAN"),
    ("IBS", "irritable bowel syndrome", "diagnosis", "ASSESSMENT_PLAN"),
    ("UC", "ulcerative colitis", "diagnosis", "ASSESSMENT_PLAN"),
    ("CD", "Crohn disease", "diagnosis", "ASSESSMENT_PLAN"),
    ("PAD", "peripheral artery disease", "diagnosis", "ASSESSMENT_PLAN"),
    ("PVD", "peripheral vascular disease", "diagnosis", "ASSESSMENT_PLAN"),
    ("AAA", "abdominal aortic aneurysm", "diagnosis", "ASSESSMENT_PLAN"),
    ("ARDS", "acute respiratory distress syndrome", "diagnosis", "ASSESSMENT_PLAN"),
    ("DKA", "diabetic ketoacidosis", "diagnosis", "ASSESSMENT_PLAN"),
    ("HHS", "hyperosmolar hyperglycemic state", "diagnosis", "ASSESSMENT_PLAN"),
    ("NAFLD", "nonalcoholic fatty liver disease", "diagnosis", "ASSESSMENT_PLAN"),
    ("NASH", "nonalcoholic steatohepatitis", "diagnosis", "ASSESSMENT_PLAN"),
    ("HLD", "hyperlipidemia", "diagnosis", "ASSESSMENT_PLAN"),
    ("SOB", "shortness of breath", "diagnosis", "HPI"),
    ("DOE", "dyspnea on exertion", "diagnosis", "HPI"),
    ("CP", "chest pain", "diagnosis", "HPI"),
    ("HA", "headache", "diagnosis", "HPI"),
    ("N/V", "nausea and vomiting", "diagnosis", "HPI"),
    ("N/V/D", "nausea, vomiting, and diarrhea", "diagnosis", "HPI"),
    ("AMS", "altered mental status", "diagnosis", "HPI"),
    ("LOC", "loss of consciousness", "diagnosis", "HPI"),
    ("LBP", "low back pain", "diagnosis", "HPI"),
    ("RLE", "right lower extremity", "diagnosis", "HPI"),
    ("LLE", "left lower extremity", "diagnosis", "HPI"),
    ("RUE", "right upper extremity", "diagnosis", "HPI"),
    ("LUE", "left upper extremity", "diagnosis", "HPI"),
    ("SIRS", "systemic inflammatory response syndrome", "diagnosis", "ASSESSMENT_PLAN"),
    ("DIC", "disseminated intravascular coagulation", "diagnosis", "ASSESSMENT_PLAN"),
    ("IDA", "iron deficiency anemia", "diagnosis", "ASSESSMENT_PLAN"),
    ("TTP", "thrombotic thrombocytopenic purpura", "diagnosis", "ASSESSMENT_PLAN"),
    ("HIT", "heparin-induced thrombocytopenia", "diagnosis", "ASSESSMENT_PLAN"),
    ("GBS", "Guillain-Barre syndrome", "diagnosis", "ASSESSMENT_PLAN"),
    ("PID", "pelvic inflammatory disease", "diagnosis", "ASSESSMENT_PLAN"),
    ("STI", "sexually transmitted infection", "diagnosis", "ASSESSMENT_PLAN"),
    ("HIV", "human immunodeficiency virus", "diagnosis", "ASSESSMENT_PLAN"),
    ("HCV", "hepatitis C virus", "diagnosis", "ASSESSMENT_PLAN"),
    ("HBV", "hepatitis B virus", "diagnosis", "ASSESSMENT_PLAN"),

    # --- Medication abbreviations ---
    ("ACE", "angiotensin converting enzyme", "medication", "MEDICATIONS"),
    ("ACEi", "angiotensin converting enzyme inhibitor", "medication", "MEDICATIONS"),
    ("ARB", "angiotensin receptor blocker", "medication", "MEDICATIONS"),
    ("NSAID", "nonsteroidal anti-inflammatory drug", "medication", "MEDICATIONS"),
    ("PPI", "proton pump inhibitor", "medication", "MEDICATIONS"),
    ("SSRI", "selective serotonin reuptake inhibitor", "medication", "MEDICATIONS"),
    ("SNRI", "serotonin-norepinephrine reuptake inhibitor", "medication", "MEDICATIONS"),
    ("TCA", "tricyclic antidepressant", "medication", "MEDICATIONS"),
    ("BB", "beta blocker", "medication", "MEDICATIONS"),
    ("CCB", "calcium channel blocker", "medication", "MEDICATIONS"),
    ("ASA", "aspirin", "medication", "MEDICATIONS"),
    ("APAP", "acetaminophen", "medication", "MEDICATIONS"),
    ("MOM", "milk of magnesia", "medication", "MEDICATIONS"),
    ("NTG", "nitroglycerin", "medication", "MEDICATIONS"),
    ("EPO", "erythropoietin", "medication", "MEDICATIONS"),
    ("HCTZ", "hydrochlorothiazide", "medication", "MEDICATIONS"),
    ("MTX", "methotrexate", "medication", "MEDICATIONS"),
    ("PCN", "penicillin", "medication", "MEDICATIONS"),
    ("ABX", "antibiotics", "medication", "MEDICATIONS"),
    ("TCN", "tetracycline", "medication", "MEDICATIONS"),
    ("INH", "isoniazid", "medication", "MEDICATIONS"),
    ("OCP", "oral contraceptive pill", "medication", "MEDICATIONS"),
    ("HRT", "hormone replacement therapy", "medication", "MEDICATIONS"),
    ("DOAC", "direct oral anticoagulant", "medication", "MEDICATIONS"),
    ("NOAC", "novel oral anticoagulant", "medication", "MEDICATIONS"),
    ("UFH", "unfractionated heparin", "medication", "MEDICATIONS"),
    ("LMWH", "low molecular weight heparin", "medication", "MEDICATIONS"),
    ("IVF", "intravenous fluids", "medication", "MEDICATIONS"),
    ("NS", "normal saline", "medication", "MEDICATIONS"),
    ("LR", "lactated Ringers", "medication", "MEDICATIONS"),
    ("D5W", "5% dextrose in water", "medication", "MEDICATIONS"),
    ("TPN", "total parenteral nutrition", "medication", "MEDICATIONS"),

    # --- Dosing / route abbreviations ---
    ("BID", "twice daily", "medication", "MEDICATIONS"),
    ("TID", "three times daily", "medication", "MEDICATIONS"),
    ("QID", "four times daily", "medication", "MEDICATIONS"),
    ("PRN", "as needed", "medication", "MEDICATIONS"),
    ("qd", "daily", "medication", "MEDICATIONS"),
    ("qhs", "at bedtime", "medication", "MEDICATIONS"),
    ("qAM", "every morning", "medication", "MEDICATIONS"),
    ("qPM", "every evening", "medication", "MEDICATIONS"),
    ("q4h", "every 4 hours", "medication", "MEDICATIONS"),
    ("q6h", "every 6 hours", "medication", "MEDICATIONS"),
    ("q8h", "every 8 hours", "medication", "MEDICATIONS"),
    ("q12h", "every 12 hours", "medication", "MEDICATIONS"),
    ("po", "by mouth", "medication", "MEDICATIONS"),
    ("PO", "by mouth", "medication", "MEDICATIONS"),
    ("IV", "intravenous", "medication", "MEDICATIONS"),
    ("IM", "intramuscular", "medication", "MEDICATIONS"),
    ("SQ", "subcutaneous", "medication", "MEDICATIONS"),
    ("SubQ", "subcutaneous", "medication", "MEDICATIONS"),
    ("SC", "subcutaneous", "medication", "MEDICATIONS"),
    ("SL", "sublingual", "medication", "MEDICATIONS"),
    ("PR", "per rectum", "medication", "MEDICATIONS"),
    ("INH", "inhaled", "medication", "MEDICATIONS"),
    ("TOP", "topical", "medication", "MEDICATIONS"),
    ("OD", "right eye", "medication", "MEDICATIONS"),
    ("OS", "left eye", "medication", "MEDICATIONS"),
    ("OU", "both eyes", "medication", "MEDICATIONS"),
    ("gtts", "drops", "medication", "MEDICATIONS"),
    ("tab", "tablet", "medication", "MEDICATIONS"),
    ("cap", "capsule", "medication", "MEDICATIONS"),
    ("susp", "suspension", "medication", "MEDICATIONS"),
    ("sol", "solution", "medication", "MEDICATIONS"),
    ("mcg", "microgram", "medication", "MEDICATIONS"),
    ("mEq", "milliequivalent", "medication", "MEDICATIONS"),
    ("mg", "milligram", "medication", "MEDICATIONS"),
    ("mL", "milliliter", "medication", "MEDICATIONS"),

    # --- Lab abbreviations ---
    ("Hgb", "hemoglobin", "lab", "LABS"),
    ("Hb", "hemoglobin", "lab", "LABS"),
    ("Hct", "hematocrit", "lab", "LABS"),
    ("WBC", "white blood cell count", "lab", "LABS"),
    ("RBC", "red blood cell count", "lab", "LABS"),
    ("Plt", "platelets", "lab", "LABS"),
    ("PLT", "platelet count", "lab", "LABS"),
    ("MCV", "mean corpuscular volume", "lab", "LABS"),
    ("MCH", "mean corpuscular hemoglobin", "lab", "LABS"),
    ("MCHC", "mean corpuscular hemoglobin concentration", "lab", "LABS"),
    ("RDW", "red cell distribution width", "lab", "LABS"),
    ("MPV", "mean platelet volume", "lab", "LABS"),
    ("BMP", "basic metabolic panel", "lab", "LABS"),
    ("CMP", "comprehensive metabolic panel", "lab", "LABS"),
    ("CBC", "complete blood count", "lab", "LABS"),
    ("BUN", "blood urea nitrogen", "lab", "LABS"),
    ("Cr", "creatinine", "lab", "LABS"),
    ("SCr", "serum creatinine", "lab", "LABS"),
    ("GFR", "glomerular filtration rate", "lab", "LABS"),
    ("eGFR", "estimated glomerular filtration rate", "lab", "LABS"),
    ("Na", "sodium", "lab", "LABS"),
    ("K", "potassium", "lab", "LABS"),
    ("Cl", "chloride", "lab", "LABS"),
    ("CO2", "bicarbonate", "lab", "LABS"),
    ("Ca", "calcium", "lab", "LABS"),
    ("Mg", "magnesium", "lab", "LABS"),
    ("Phos", "phosphorus", "lab", "LABS"),
    ("AST", "aspartate aminotransferase", "lab", "LABS"),
    ("ALT", "alanine aminotransferase", "lab", "LABS"),
    ("ALP", "alkaline phosphatase", "lab", "LABS"),
    ("GGT", "gamma-glutamyl transferase", "lab", "LABS"),
    ("T. bili", "total bilirubin", "lab", "LABS"),
    ("Tbili", "total bilirubin", "lab", "LABS"),
    ("D. bili", "direct bilirubin", "lab", "LABS"),
    ("Dbili", "direct bilirubin", "lab", "LABS"),
    ("Alb", "albumin", "lab", "LABS"),
    ("TP", "total protein", "lab", "LABS"),
    ("LDH", "lactate dehydrogenase", "lab", "LABS"),
    ("CK", "creatine kinase", "lab", "LABS"),
    ("CPK", "creatine phosphokinase", "lab", "LABS"),
    ("CK-MB", "creatine kinase MB fraction", "lab", "LABS"),
    ("BNP", "B-type natriuretic peptide", "lab", "LABS"),
    ("NT-proBNP", "N-terminal pro-B-type natriuretic peptide", "lab", "LABS"),
    ("HbA1c", "hemoglobin A1c", "lab", "LABS"),
    ("A1c", "hemoglobin A1c", "lab", "LABS"),
    ("TSH", "thyroid stimulating hormone", "lab", "LABS"),
    ("FT4", "free thyroxine", "lab", "LABS"),
    ("FT3", "free triiodothyronine", "lab", "LABS"),
    ("PSA", "prostate-specific antigen", "lab", "LABS"),
    ("ESR", "erythrocyte sedimentation rate", "lab", "LABS"),
    ("CRP", "C-reactive protein", "lab", "LABS"),
    ("hsCRP", "high-sensitivity C-reactive protein", "lab", "LABS"),
    ("PT", "prothrombin time", "lab", "LABS"),
    ("INR", "international normalized ratio", "lab", "LABS"),
    ("PTT", "partial thromboplastin time", "lab", "LABS"),
    ("aPTT", "activated partial thromboplastin time", "lab", "LABS"),
    ("D-dimer", "D-dimer fibrin degradation product", "lab", "LABS"),
    ("ABG", "arterial blood gas", "lab", "LABS"),
    ("VBG", "venous blood gas", "lab", "LABS"),
    ("pO2", "partial pressure of oxygen", "lab", "LABS"),
    ("pCO2", "partial pressure of carbon dioxide", "lab", "LABS"),
    ("SpO2", "oxygen saturation by pulse oximetry", "lab", "VITALS"),
    ("UA", "urinalysis", "lab", "LABS"),
    ("UDS", "urine drug screen", "lab", "LABS"),
    ("U/A", "urinalysis", "lab", "LABS"),
    ("Lytes", "electrolytes", "lab", "LABS"),
    ("LFT", "liver function tests", "lab", "LABS"),
    ("LFTs", "liver function tests", "lab", "LABS"),
    ("RFT", "renal function tests", "lab", "LABS"),
    ("TFT", "thyroid function tests", "lab", "LABS"),
    ("ANA", "antinuclear antibody", "lab", "LABS"),
    ("RF", "rheumatoid factor", "lab", "LABS"),
    ("Trop", "troponin", "lab", "LABS"),
    ("cTnI", "cardiac troponin I", "lab", "LABS"),
    ("cTnT", "cardiac troponin T", "lab", "LABS"),
    ("Fib", "fibrinogen", "lab", "LABS"),
    ("Ferr", "ferritin", "lab", "LABS"),
    ("TIBC", "total iron binding capacity", "lab", "LABS"),
    ("Retics", "reticulocyte count", "lab", "LABS"),

    # --- Procedure abbreviations ---
    ("CABG", "coronary artery bypass graft", "procedure", "ASSESSMENT_PLAN"),
    ("PCI", "percutaneous coronary intervention", "procedure", "ASSESSMENT_PLAN"),
    ("PTCA", "percutaneous transluminal coronary angioplasty", "procedure", "ASSESSMENT_PLAN"),
    ("ERCP", "endoscopic retrograde cholangiopancreatography", "procedure", "ASSESSMENT_PLAN"),
    ("EGD", "esophagogastroduodenoscopy", "procedure", "ASSESSMENT_PLAN"),
    ("EKG", "electrocardiogram", "procedure", "ASSESSMENT_PLAN"),
    ("ECG", "electrocardiogram", "procedure", "ASSESSMENT_PLAN"),
    ("EEG", "electroencephalogram", "procedure", "ASSESSMENT_PLAN"),
    ("EMG", "electromyography", "procedure", "ASSESSMENT_PLAN"),
    ("CXR", "chest X-ray", "procedure", "ASSESSMENT_PLAN"),
    ("CT", "computed tomography", "procedure", "ASSESSMENT_PLAN"),
    ("MRI", "magnetic resonance imaging", "procedure", "ASSESSMENT_PLAN"),
    ("US", "ultrasound", "procedure", "ASSESSMENT_PLAN"),
    ("TTE", "transthoracic echocardiogram", "procedure", "ASSESSMENT_PLAN"),
    ("TEE", "transesophageal echocardiogram", "procedure", "ASSESSMENT_PLAN"),
    ("LP", "lumbar puncture", "procedure", "ASSESSMENT_PLAN"),
    ("HD", "hemodialysis", "procedure", "ASSESSMENT_PLAN"),
    ("PD", "peritoneal dialysis", "procedure", "ASSESSMENT_PLAN"),
    ("CPAP", "continuous positive airway pressure", "procedure", "ASSESSMENT_PLAN"),
    ("BiPAP", "bilevel positive airway pressure", "procedure", "ASSESSMENT_PLAN"),
    ("I&D", "incision and drainage", "procedure", "ASSESSMENT_PLAN"),
    ("ORIF", "open reduction internal fixation", "procedure", "ASSESSMENT_PLAN"),
    ("TKA", "total knee arthroplasty", "procedure", "ASSESSMENT_PLAN"),
    ("THA", "total hip arthroplasty", "procedure", "ASSESSMENT_PLAN"),
    ("lap chole", "laparoscopic cholecystectomy", "procedure", "ASSESSMENT_PLAN"),
    ("appy", "appendectomy", "procedure", "ASSESSMENT_PLAN"),
    ("cath", "catheterization", "procedure", "ASSESSMENT_PLAN"),
    ("trach", "tracheostomy", "procedure", "ASSESSMENT_PLAN"),
    ("ICD", "implantable cardioverter-defibrillator", "procedure", "ASSESSMENT_PLAN"),
    ("PPM", "permanent pacemaker", "procedure", "ASSESSMENT_PLAN"),

    # --- General / vital sign abbreviations ---
    ("BP", "blood pressure", "general", "VITALS"),
    ("SBP", "systolic blood pressure", "general", "VITALS"),
    ("DBP", "diastolic blood pressure", "general", "VITALS"),
    ("HR", "heart rate", "general", "VITALS"),
    ("RR", "respiratory rate", "general", "VITALS"),
    ("T", "temperature", "general", "VITALS"),
    ("Temp", "temperature", "general", "VITALS"),
    ("Wt", "weight", "general", "VITALS"),
    ("Ht", "height", "general", "VITALS"),
    ("BMI", "body mass index", "general", "VITALS"),
    ("I/O", "intake and output", "general", "VITALS"),
    ("I&O", "intake and output", "general", "VITALS"),
    ("PMH", "past medical history", "general", "HPI"),
    ("PSH", "past surgical history", "general", "HPI"),
    ("FH", "family history", "general", "FAMILY_HISTORY"),
    ("SH", "social history", "general", "SOCIAL_HISTORY"),
    ("HPI", "history of present illness", "general", "HPI"),
    ("ROS", "review of systems", "general", "ROS"),
    ("PE", "physical exam", "general", "OTHER"),
    ("A/P", "assessment and plan", "general", "ASSESSMENT_PLAN"),
    ("A&P", "assessment and plan", "general", "ASSESSMENT_PLAN"),
    ("CC", "chief complaint", "general", "HPI"),
    ("Dx", "diagnosis", "general", "ASSESSMENT_PLAN"),
    ("DDx", "differential diagnosis", "general", "ASSESSMENT_PLAN"),
    ("Hx", "history", "general", "HPI"),
    ("Rx", "prescription", "general", "MEDICATIONS"),
    ("Tx", "treatment", "general", "ASSESSMENT_PLAN"),
    ("Sx", "symptoms", "general", "HPI"),
    ("Fx", "fracture", "general", "ASSESSMENT_PLAN"),
    ("Bx", "biopsy", "general", "ASSESSMENT_PLAN"),
    ("Cx", "culture", "general", "LABS"),
    ("Pt", "patient", "general", "OTHER"),
    ("yo", "year old", "general", "HPI"),
    ("y/o", "year old", "general", "HPI"),
    ("M", "male", "general", "HPI"),
    ("F", "female", "general", "HPI"),
    ("WNL", "within normal limits", "general", "LABS"),
    ("NAD", "no acute distress", "general", "OTHER"),
    ("AAOx3", "alert and oriented times three", "general", "OTHER"),
    ("AAOx4", "alert and oriented times four", "general", "OTHER"),
    ("c/o", "complains of", "general", "HPI"),
    ("s/p", "status post", "general", "HPI"),
    ("w/o", "without", "general", "OTHER"),
    ("w/", "with", "general", "OTHER"),
    ("d/c", "discharge", "general", "ASSESSMENT_PLAN"),
    ("D/C", "discharge or discontinue", "general", "ASSESSMENT_PLAN"),
    ("f/u", "follow up", "general", "ASSESSMENT_PLAN"),
    ("h/o", "history of", "general", "HPI"),
    ("r/o", "rule out", "general", "ASSESSMENT_PLAN"),
    ("NKA", "no known allergies", "general", "ALLERGIES"),
    ("NKDA", "no known drug allergies", "general", "ALLERGIES"),
    ("PTA", "prior to admission", "general", "HPI"),
    ("POD", "postoperative day", "general", "ASSESSMENT_PLAN"),
    ("DNR", "do not resuscitate", "general", "OTHER"),
    ("DNI", "do not intubate", "general", "OTHER"),
    ("POLST", "physician orders for life-sustaining treatment", "general", "OTHER"),
    ("ED", "emergency department", "general", "OTHER"),
    ("ER", "emergency room", "general", "OTHER"),
    ("ICU", "intensive care unit", "general", "OTHER"),
    ("MICU", "medical intensive care unit", "general", "OTHER"),
    ("SICU", "surgical intensive care unit", "general", "OTHER"),
    ("OR", "operating room", "general", "OTHER"),
    ("OT", "occupational therapy", "general", "ASSESSMENT_PLAN"),
    ("PT", "physical therapy", "general", "ASSESSMENT_PLAN"),
    ("ST", "speech therapy", "general", "ASSESSMENT_PLAN"),
    ("ADL", "activities of daily living", "general", "OTHER"),
    ("BLE", "bilateral lower extremities", "general", "OTHER"),
    ("BUE", "bilateral upper extremities", "general", "OTHER"),
    ("ROM", "range of motion", "general", "OTHER"),
    ("CMS", "circulation, motor, sensory", "general", "OTHER"),
    ("PERRLA", "pupils equal round reactive to light and accommodation", "general", "OTHER"),
    ("HEENT", "head, eyes, ears, nose, and throat", "general", "ROS"),
    ("EOMI", "extraocular movements intact", "general", "OTHER"),
    ("CN", "cranial nerves", "general", "OTHER"),
    ("DTR", "deep tendon reflexes", "general", "OTHER"),
    ("RRR", "regular rate and rhythm", "general", "OTHER"),
    ("S1S2", "first and second heart sounds", "general", "OTHER"),
    ("CTAB", "clear to auscultation bilaterally", "general", "OTHER"),
    ("CTA B/L", "clear to auscultation bilaterally", "general", "OTHER"),
    ("NT/ND", "nontender, nondistended", "general", "OTHER"),
    ("NABS", "normoactive bowel sounds", "general", "OTHER"),
    ("BS", "bowel sounds", "general", "OTHER"),
    ("PPP", "pink, pulsatile, and perfused", "general", "OTHER"),
    ("2+", "normal (for reflexes or pulses)", "general", "OTHER"),
]

print(f"  Prepared {len(CLINICAL_ABBREVIATIONS)} clinical abbreviations")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Abbreviations to Delta Table

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import current_timestamp

abbrev_schema = StructType([
    StructField("abbreviation", StringType()),
    StructField("expansion", StringType()),
    StructField("category", StringType()),
    StructField("context_hint", StringType()),
])

abbrev_df = spark.createDataFrame(
    [row[:4] for row in CLINICAL_ABBREVIATIONS],
    schema=abbrev_schema
)
abbrev_df = abbrev_df.withColumn("loaded_at", current_timestamp())

abbrev_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference.clinical_abbreviations")

total = spark.table(f"{CATALOG}.reference.clinical_abbreviations").count()
print(f"  Loaded {total} abbreviations into {CATALOG}.reference.clinical_abbreviations")

# Show category breakdown
spark.sql(f"""
    SELECT category, COUNT(*) as count
    FROM {CATALOG}.reference.clinical_abbreviations
    GROUP BY category
    ORDER BY count DESC
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 2: UMLS Metathesaurus
# MAGIC
# MAGIC The UMLS Metathesaurus contains **4.4M concept names** covering **2.3M unique concepts**
# MAGIC from 200+ source vocabularies. This is the backbone of the medical dictionary.
# MAGIC
# MAGIC **Loading priority:**
# MAGIC 1. **API Download** -- download MRCONSO.RRF + MRSTY.RRF via UTS API (requires API key)
# MAGIC 2. **UC Volume** -- load from pre-uploaded files in `/Volumes/{CATALOG}/reference/umls_files/`
# MAGIC 3. **REST API** -- individual concept lookups (slow, limited to ~50K concepts)
# MAGIC
# MAGIC **Note:** The full UMLS Metathesaurus ZIP is ~35 GB. We download only the
# MAGIC MRCONSO subset (~5 GB) which contains all concept names and synonyms.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step A: Attempt API Download of UMLS Files

# COMMAND ----------

UMLS_VOLUME = f"/Volumes/{CATALOG}/reference/umls_files"
umls_loaded = False

# Try downloading MRCONSO and MRSTY via UTS API
if UTS_API_KEY:
    uts_download(
        NLM_URLS["umls_mrconso_only"],
        UMLS_VOLUME,
        extract=True,
        required_files=["MRCONSO.RRF", "MRSTY.RRF"]
    )
else:
    print("  Skipping UMLS API download (no API key)")
    print(f"  To load manually, upload MRCONSO.RRF and MRSTY.RRF to {UMLS_VOLUME}/")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B: Parse UMLS RRF Files (from API download or manual upload)

# COMMAND ----------

import os
from pyspark.sql.functions import col, collect_list, collect_set, first, current_timestamp, lit

def load_umls_from_rrf():
    """Load UMLS concepts from MRCONSO.RRF and MRSTY.RRF files."""
    global umls_loaded

    mrconso_path = f"{UMLS_VOLUME}/MRCONSO.RRF"
    mrsty_path = f"{UMLS_VOLUME}/MRSTY.RRF"

    # Check if files exist
    try:
        files = [f.name for f in dbutils.fs.ls(UMLS_VOLUME)]
    except Exception:
        print(f"  UMLS volume not found or empty: {UMLS_VOLUME}")
        return False

    has_mrconso = "MRCONSO.RRF" in files
    has_mrsty = "MRSTY.RRF" in files

    if not has_mrconso:
        print(f"  MRCONSO.RRF not found in {UMLS_VOLUME}")
        return False

    print(f"  Loading MRCONSO.RRF from {UMLS_VOLUME}...")

    # MRCONSO.RRF columns (pipe-delimited, no header):
    # CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
    mrconso_df = (spark.read
        .option("delimiter", "|")
        .option("header", "false")
        .option("quote", "")
        .csv(mrconso_path)
    )

    # Select relevant columns and filter to English only
    mrconso_df = mrconso_df.select(
        col("_c0").alias("cui"),
        col("_c1").alias("lat"),
        col("_c2").alias("ts"),
        col("_c4").alias("stt"),
        col("_c6").alias("ispref"),
        col("_c11").alias("sab"),  # Source vocabulary
        col("_c14").alias("str"),  # String (concept name)
    ).filter(col("lat") == "ENG")

    print(f"  English concepts loaded from MRCONSO.RRF")

    # Get preferred names (TS=P, STT=PF, ISPREF=Y)
    preferred = (mrconso_df
        .filter((col("ts") == "P") & (col("stt") == "PF") & (col("ispref") == "Y"))
        .groupBy("cui")
        .agg(first("str").alias("preferred_name"))
    )

    # Get all synonyms grouped by CUI
    synonyms = (mrconso_df
        .groupBy("cui")
        .agg(
            collect_set("str").alias("synonyms"),
            collect_set("sab").alias("source_vocabularies"),
        )
    )

    # Join preferred names with synonyms
    concepts = preferred.join(synonyms, "cui", "outer")

    # Load semantic types from MRSTY.RRF if available
    if has_mrsty:
        print(f"  Loading MRSTY.RRF for semantic types...")
        # MRSTY.RRF columns: CUI|TUI|STN|STY|ATUI|CVF
        mrsty_df = (spark.read
            .option("delimiter", "|")
            .option("header", "false")
            .option("quote", "")
            .csv(mrsty_path)
        )

        # Semantic group mapping (TUI -> group)
        semantic_types = mrsty_df.select(
            col("_c0").alias("cui"),
            col("_c3").alias("semantic_type"),
        )

        # Map semantic types to groups
        from pyspark.sql.functions import when
        semantic_types = semantic_types.withColumn(
            "semantic_group",
            when(col("semantic_type").isin(
                "Disease or Syndrome", "Neoplastic Process", "Mental or Behavioral Dysfunction",
                "Cell or Molecular Dysfunction", "Congenital Abnormality", "Acquired Abnormality",
                "Injury or Poisoning", "Pathologic Function", "Sign or Symptom",
                "Anatomical Abnormality", "Finding"
            ), "DISO")
            .when(col("semantic_type").isin(
                "Pharmacologic Substance", "Clinical Drug", "Antibiotic",
                "Biomedical or Dental Material", "Hormone", "Vitamin",
                "Immunologic Factor", "Enzyme", "Amino Acid, Peptide, or Protein"
            ), "CHEM")
            .when(col("semantic_type").isin(
                "Therapeutic or Preventive Procedure", "Diagnostic Procedure",
                "Laboratory Procedure", "Health Care Activity",
                "Research Activity", "Molecular Biology Research Technique"
            ), "PROC")
            .when(col("semantic_type").isin(
                "Body Part, Organ, or Organ Component", "Tissue", "Cell",
                "Body Location or Region", "Body Space or Junction",
                "Body System", "Fully Formed Anatomical Structure"
            ), "ANAT")
            .when(col("semantic_type").isin(
                "Laboratory or Test Result", "Clinical Attribute",
                "Organism Attribute"
            ), "PHYS")
            .otherwise("OTHER")
        )

        # Take the first semantic type per CUI (most concepts have one)
        semantic_by_cui = (semantic_types
            .groupBy("cui")
            .agg(
                first("semantic_type").alias("semantic_type"),
                first("semantic_group").alias("semantic_group"),
            )
        )

        concepts = concepts.join(semantic_by_cui, "cui", "left")
    else:
        print(f"  MRSTY.RRF not found -- skipping semantic types")
        concepts = (concepts
            .withColumn("semantic_type", lit(None).cast("string"))
            .withColumn("semantic_group", lit(None).cast("string"))
        )

    # Add timestamp and write
    concepts = concepts.withColumn("loaded_at", current_timestamp())

    concepts.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference.umls_concepts")

    total = spark.table(f"{CATALOG}.reference.umls_concepts").count()
    print(f"  Loaded {total:,} UMLS concepts into {CATALOG}.reference.umls_concepts")
    umls_loaded = True
    return True

load_umls_from_rrf()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option B: UMLS REST API (Fallback)
# MAGIC
# MAGIC If RRF files are not available, attempt to load concepts via the UMLS REST API.
# MAGIC Requires a UMLS API key from https://uts.nlm.nih.gov/uts/profile.

# COMMAND ----------

import requests
import time

def load_umls_from_api(api_key, semantic_types=None, max_concepts=50000):
    """Load UMLS concepts via REST API. Slower but requires no file download."""
    global umls_loaded

    if umls_loaded:
        print("  UMLS already loaded from RRF files -- skipping API")
        return

    if not api_key:
        print("  No UMLS API key provided -- skipping API load")
        print("  To use the API:")
        print("  1. Get an API key from https://uts.nlm.nih.gov/uts/profile")
        print("  2. Set the umls_api_key widget")
        return

    if semantic_types is None:
        semantic_types = [
            "T047",  # Disease or Syndrome
            "T184",  # Sign or Symptom
            "T121",  # Pharmacologic Substance
            "T059",  # Laboratory Procedure
            "T061",  # Therapeutic or Preventive Procedure
            "T060",  # Diagnostic Procedure
            "T033",  # Finding
            "T034",  # Laboratory or Test Result
        ]

    base_url = "https://uts-ws.nlm.nih.gov/rest"
    concepts = []

    for sty in semantic_types:
        page = 1
        while True:
            url = f"{base_url}/search/current"
            params = {
                "apiKey": api_key,
                "string": "*",
                "sabs": "SNOMEDCT_US,ICD10CM,RXNORM",
                "searchType": "exact",
                "pageNumber": page,
                "pageSize": 100,
            }

            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code != 200:
                    print(f"  API returned {resp.status_code} for STY {sty}, page {page}")
                    break

                data = resp.json()
                results = data.get("result", {}).get("results", [])
                if not results:
                    break

                for r in results:
                    concepts.append({
                        "cui": r.get("ui", ""),
                        "preferred_name": r.get("name", ""),
                        "semantic_type": sty,
                        "semantic_group": "OTHER",
                        "synonyms": [],
                        "source_vocabularies": [],
                    })

                page += 1
                if len(concepts) >= max_concepts:
                    break
                time.sleep(0.2)

            except Exception as e:
                print(f"  API error: {e}")
                break

        if len(concepts) >= max_concepts:
            print(f"  Reached max_concepts limit ({max_concepts})")
            break

        print(f"  Fetched {len(concepts):,} concepts so far...")

    if concepts:
        from pyspark.sql.types import StructType, StructField, StringType, ArrayType
        schema = StructType([
            StructField("cui", StringType()),
            StructField("preferred_name", StringType()),
            StructField("semantic_type", StringType()),
            StructField("semantic_group", StringType()),
            StructField("synonyms", ArrayType(StringType())),
            StructField("source_vocabularies", ArrayType(StringType())),
        ])
        df = spark.createDataFrame(concepts, schema=schema)
        df = df.withColumn("loaded_at", current_timestamp())
        df.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference.umls_concepts")
        total = spark.table(f"{CATALOG}.reference.umls_concepts").count()
        print(f"  Loaded {total:,} UMLS concepts via API into {CATALOG}.reference.umls_concepts")
        umls_loaded = True

# Try API fallback if RRF files were not available
if not umls_loaded:
    try:
        dbutils.widgets.text("umls_api_key", "", "UMLS API Key")
        UMLS_API_KEY = dbutils.widgets.get("umls_api_key")
    except Exception:
        UMLS_API_KEY = ""

    load_umls_from_api(UMLS_API_KEY)

if not umls_loaded:
    print("  UMLS not loaded. To enable:")
    print(f"  1. Get a free API key at https://uts.nlm.nih.gov/uts/profile")
    print(f"  2. Store it: databricks secrets put-secret medical-ontology umls-api-key --string-value YOUR_KEY")
    print(f"  3. Re-run this notebook -- it will download UMLS automatically")
    print(f"  OR: manually upload MRCONSO.RRF and MRSTY.RRF to {UMLS_VOLUME}/")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 3: SNOMED-CT Hierarchy
# MAGIC
# MAGIC SNOMED-CT contains ~350,000 clinical concepts in a rich IS-A hierarchy.
# MAGIC The US Edition includes the **official SNOMED-to-ICD-10-CM mapping** -- this is
# MAGIC the key crosswalk for ontology-guided coding.
# MAGIC
# MAGIC **Loading priority:**
# MAGIC 1. **API Download** -- download SNOMED CT US Edition RF2 via UTS API
# MAGIC 2. **UC Volume** -- load from pre-uploaded RF2 files
# MAGIC
# MAGIC **Required files:** `sct2_Relationship_Snapshot_*.txt`, `sct2_Description_Snapshot_*.txt`
# MAGIC **Optional (ICD-10 map):** `der2_iisssccRefset_ExtendedMap*.txt` or `tls_Icd10cmHumanReadableMap*.tsv`

# COMMAND ----------

SNOMED_VOLUME = f"/Volumes/{CATALOG}/reference/snomed_files"
snomed_loaded = False

# Try downloading SNOMED CT US Edition via UTS API
if UTS_API_KEY:
    uts_download(
        NLM_URLS["snomed_us"],
        SNOMED_VOLUME,
        extract=True,
        required_files=["sct2_Relationship", "sct2_Description", "ExtendedMap", "Icd10cmHumanReadableMap"]
    )
else:
    print("  Skipping SNOMED API download (no API key)")
    print(f"  To load manually, upload SNOMED RF2 Snapshot files to {SNOMED_VOLUME}/")

# COMMAND ----------

def load_snomed_hierarchy():
    """Load SNOMED-CT hierarchy from RF2 files."""
    global snomed_loaded

    try:
        files = [f.name for f in dbutils.fs.ls(SNOMED_VOLUME)]
    except Exception:
        print(f"  SNOMED volume not found or empty: {SNOMED_VOLUME}")
        return False

    # Find relationship and description files
    rel_file = None
    desc_file = None
    map_file = None

    for f in files:
        f_lower = f.lower()
        if "relationship" in f_lower and f_lower.endswith(".txt"):
            rel_file = f
        if "description" in f_lower and f_lower.endswith(".txt"):
            desc_file = f
        if "extendedmap" in f_lower and f_lower.endswith(".txt"):
            map_file = f

    if not rel_file or not desc_file:
        print(f"  Required SNOMED RF2 files not found in {SNOMED_VOLUME}")
        print(f"  Found files: {files}")
        print(f"  Need: sct2_Relationship_Snapshot_*.txt and sct2_Description_Snapshot_*.txt")
        return False

    print(f"  Loading SNOMED descriptions from {desc_file}...")

    # Load descriptions (concept names)
    # Columns: id, effectiveTime, active, moduleId, conceptId, languageCode, typeId, term, caseSignificanceId
    desc_df = (spark.read
        .option("delimiter", "\t")
        .option("header", "true")
        .csv(f"{SNOMED_VOLUME}/{desc_file}")
    )

    # Filter to active, English, Fully Specified Names (typeId=900000000000003001)
    # and Preferred Terms (typeId=900000000000013009)
    desc_df = (desc_df
        .filter(col("active") == "1")
        .filter(col("languageCode") == "en")
    )

    # Get preferred term for each concept (FSN preferred, then PT)
    from pyspark.sql.functions import row_number
    from pyspark.sql.window import Window

    w = Window.partitionBy("conceptId").orderBy(col("typeId").asc())
    concept_names = (desc_df
        .withColumn("rn", row_number().over(w))
        .filter(col("rn") == 1)
        .select(
            col("conceptId").alias("concept_id"),
            col("term").alias("concept_name"),
        )
    )

    print(f"  Loading SNOMED relationships from {rel_file}...")

    # Load relationships
    # Columns: id, effectiveTime, active, moduleId, sourceId, destinationId, relationshipGroup, typeId, characteristicTypeId, modifierId
    rel_df = (spark.read
        .option("delimiter", "\t")
        .option("header", "true")
        .csv(f"{SNOMED_VOLUME}/{rel_file}")
    )

    # Filter to active IS_A relationships (typeId=116680003)
    is_a_df = (rel_df
        .filter(col("active") == "1")
        .filter(col("typeId") == "116680003")
        .select(
            col("sourceId").alias("concept_id"),
            col("destinationId").alias("parent_id"),
        )
    )

    # Join with concept names
    hierarchy = (is_a_df
        .join(concept_names, "concept_id", "left")
        .join(
            concept_names.withColumnRenamed("concept_id", "parent_id")
                         .withColumnRenamed("concept_name", "parent_name"),
            "parent_id",
            "left"
        )
        .withColumn("relationship_type", lit("IS_A"))
        .withColumn("hierarchy_depth", lit(None).cast("int"))  # Depth requires BFS, computed later if needed
        .withColumn("loaded_at", current_timestamp())
    )

    hierarchy.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference.snomed_hierarchy")

    total = spark.table(f"{CATALOG}.reference.snomed_hierarchy").count()
    print(f"  Loaded {total:,} SNOMED IS-A relationships into {CATALOG}.reference.snomed_hierarchy")
    snomed_loaded = True

    # Load SNOMED-to-ICD-10 mapping if available
    if map_file:
        print(f"  Loading SNOMED-to-ICD-10 mapping from {map_file}...")

        # Extended map columns: id, effectiveTime, active, moduleId, refsetId,
        # referencedComponentId, mapGroup, mapPriority, mapRule, mapAdvice, mapTarget,
        # correlationId, mapCategoryId
        map_df = (spark.read
            .option("delimiter", "\t")
            .option("header", "true")
            .csv(f"{SNOMED_VOLUME}/{map_file}")
        )

        map_df = (map_df
            .filter(col("active") == "1")
            .filter(col("mapTarget").isNotNull())
            .filter(col("mapTarget") != "")
        )

        # Join with concept names for SNOMED names
        snomed_icd10 = (map_df
            .join(
                concept_names.withColumnRenamed("concept_id", "referencedComponentId"),
                "referencedComponentId",
                "left"
            )
            .select(
                col("referencedComponentId").alias("snomed_concept_id"),
                col("concept_name").alias("snomed_name"),
                col("mapTarget").alias("icd10_code"),
                lit(None).cast("string").alias("icd10_name"),  # ICD-10 name resolved separately
                col("mapPriority").cast("int").alias("map_priority"),
                col("mapRule").alias("map_rule"),
                col("mapGroup").cast("int").alias("map_group"),
            )
            .withColumn("loaded_at", current_timestamp())
        )

        snomed_icd10.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference.snomed_icd10_map")

        map_total = spark.table(f"{CATALOG}.reference.snomed_icd10_map").count()
        print(f"  Loaded {map_total:,} SNOMED-to-ICD-10 mappings into {CATALOG}.reference.snomed_icd10_map")
    else:
        print(f"  SNOMED-to-ICD-10 map file not found -- skipping")
        print(f"  Upload der2_iisssccRefset_ExtendedMapSnapshot_*.txt to {SNOMED_VOLUME}/")

    return True

load_snomed_hierarchy()

if not snomed_loaded:
    print(f"  SNOMED-CT not loaded. To load:")
    print(f"  SNOMED not loaded. To enable:")
    print(f"  1. Get a free API key at https://uts.nlm.nih.gov/uts/profile")
    print(f"  2. Store it: databricks secrets put-secret medical-ontology umls-api-key --string-value YOUR_KEY")
    print(f"  3. Re-run this notebook -- it will download SNOMED CT US Edition automatically")
    print(f"  OR: manually upload RF2 Snapshot files to {SNOMED_VOLUME}/")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 4: RxNorm Concepts
# MAGIC
# MAGIC RxNorm provides normalized drug names with ingredient, brand, dosage form, and strength.
# MAGIC Contains ~115K concepts critical for medication entity normalization.
# MAGIC
# MAGIC **Loading priority:**
# MAGIC 1. **API Download** -- download RxNorm full release via UTS API
# MAGIC 2. **UC Volume** -- load from pre-uploaded RXNCONSO.RRF
# MAGIC 3. **REST API** -- RxNav API (no auth required, but slow for bulk)
# MAGIC
# MAGIC **Note:** The RxNorm "prescribable" subset is available without any license at all:
# MAGIC `https://download.nlm.nih.gov/rxnorm/RxNorm_full_prescribe_current.zip`

# COMMAND ----------

RXNORM_VOLUME = f"/Volumes/{CATALOG}/reference/rxnorm_files"
rxnorm_loaded = False

# Try downloading RxNorm via UTS API (or prescribable subset without key)
if UTS_API_KEY:
    uts_download(
        NLM_URLS["rxnorm_full"],
        RXNORM_VOLUME,
        extract=True,
        required_files=["RXNCONSO.RRF"]
    )
else:
    # Prescribable subset is available without any authentication
    print("  No API key -- attempting RxNorm Prescribable download (no license required)...")
    try:
        import requests as req
        resp = req.get(NLM_URLS["rxnorm_prescribable"], stream=True, timeout=300)
        if resp.status_code == 200:
            zip_path = f"{DOWNLOAD_TEMP}/rxnorm_prescribe.zip"
            with open(zip_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192 * 1024):
                    f.write(chunk)
            import zipfile as zf_mod
            with zf_mod.ZipFile(zip_path, 'r') as z:
                for member in z.namelist():
                    if os.path.basename(member).upper() == "RXNCONSO.RRF":
                        dest = f"{RXNORM_VOLUME}/RXNCONSO.RRF"
                        with z.open(member) as src, open(dest, 'wb') as dst:
                            import shutil as sh
                            sh.copyfileobj(src, dst)
                        print(f"  Downloaded RxNorm Prescribable RXNCONSO.RRF")
                        break
            os.remove(zip_path)
        else:
            print(f"  RxNorm prescribable download failed: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  RxNorm prescribable download failed: {e}")
        print(f"  To load manually, upload RXNCONSO.RRF to {RXNORM_VOLUME}/")

# COMMAND ----------

def load_rxnorm_from_rrf():
    """Load RxNorm concepts from RXNCONSO.RRF file."""
    global rxnorm_loaded

    try:
        files = [f.name for f in dbutils.fs.ls(RXNORM_VOLUME)]
    except Exception:
        print(f"  RxNorm volume not found or empty: {RXNORM_VOLUME}")
        return False

    if "RXNCONSO.RRF" not in files:
        print(f"  RXNCONSO.RRF not found in {RXNORM_VOLUME}")
        return False

    print(f"  Loading RXNCONSO.RRF from {RXNORM_VOLUME}...")

    # RXNCONSO.RRF columns (pipe-delimited):
    # RXCUI|LAT|TS|LUI|STT|SUI|ISPREF|RXAUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
    rxn_df = (spark.read
        .option("delimiter", "|")
        .option("header", "false")
        .option("quote", "")
        .csv(f"{RXNORM_VOLUME}/RXNCONSO.RRF")
    )

    # Select relevant columns, filter to English
    rxn_df = rxn_df.select(
        col("_c0").alias("rxcui"),
        col("_c1").alias("lat"),
        col("_c11").alias("sab"),   # Source
        col("_c12").alias("tty"),   # Term type (IN, BN, SCD, etc.)
        col("_c14").alias("str"),   # String name
    ).filter(col("lat") == "ENG")

    # Filter to RxNorm source vocabulary only
    rxn_df = rxn_df.filter(col("sab") == "RXNORM")

    # Get preferred name (first name per RXCUI)
    from pyspark.sql.functions import first, collect_set, array_distinct, array

    concepts = (rxn_df
        .groupBy("rxcui")
        .agg(
            first("str").alias("name"),
            first("tty").alias("term_type"),
            collect_set("str").alias("synonyms"),
        )
    )

    # For ingredients, get IN (ingredient) term type entries
    ingredients = (rxn_df
        .filter(col("tty") == "IN")
        .groupBy("rxcui")
        .agg(collect_set("str").alias("ingredients"))
    )

    concepts = (concepts
        .join(ingredients, "rxcui", "left")
        .withColumn("loaded_at", current_timestamp())
    )

    # Fill null ingredients with empty array
    from pyspark.sql.functions import coalesce, array as spark_array
    concepts = concepts.withColumn(
        "ingredients",
        coalesce(col("ingredients"), spark_array().cast("array<string>"))
    )

    concepts.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference.rxnorm_concepts")

    total = spark.table(f"{CATALOG}.reference.rxnorm_concepts").count()
    print(f"  Loaded {total:,} RxNorm concepts into {CATALOG}.reference.rxnorm_concepts")
    rxnorm_loaded = True
    return True

load_rxnorm_from_rrf()

# COMMAND ----------

# MAGIC %md
# MAGIC ### RxNorm REST API (Fallback)
# MAGIC
# MAGIC If RRF files are not available, fetch common drug concepts from the public RxNorm REST API.
# MAGIC This is slower and limited but requires no file download.

# COMMAND ----------

import requests
import time

def load_rxnorm_from_api(max_concepts=10000):
    """Load RxNorm concepts from the public REST API."""
    global rxnorm_loaded

    if rxnorm_loaded:
        print("  RxNorm already loaded from RRF files -- skipping API")
        return

    print("  Loading RxNorm concepts from REST API (public, no key needed)...")

    base_url = "https://rxnav.nlm.nih.gov/REST"

    # Get all term types we care about
    term_types = ["IN", "BN", "SCD", "SBD", "GPCK", "BPCK"]
    concepts = []

    for tty in term_types:
        try:
            url = f"{base_url}/allconcepts.json?tty={tty}"
            resp = requests.get(url, timeout=120)
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code} for term type {tty}")
                continue

            data = resp.json()
            group = data.get("minConceptGroup", {}).get("minConcept", [])

            for concept in group:
                concepts.append({
                    "rxcui": concept.get("rxcui", ""),
                    "name": concept.get("name", ""),
                    "term_type": tty,
                    "synonyms": [concept.get("name", "")],
                    "ingredients": [],
                })

            print(f"  {tty}: {len(group):,} concepts fetched ({len(concepts):,} total)")

            if len(concepts) >= max_concepts:
                break

            time.sleep(1)

        except Exception as e:
            print(f"  Error fetching {tty}: {e}")
            continue

    if concepts:
        from pyspark.sql.types import StructType, StructField, StringType, ArrayType
        schema = StructType([
            StructField("rxcui", StringType()),
            StructField("name", StringType()),
            StructField("term_type", StringType()),
            StructField("synonyms", ArrayType(StringType())),
            StructField("ingredients", ArrayType(StringType())),
        ])
        df = spark.createDataFrame(concepts, schema=schema)
        df = df.withColumn("loaded_at", current_timestamp())
        df.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference.rxnorm_concepts")
        total = spark.table(f"{CATALOG}.reference.rxnorm_concepts").count()
        print(f"  Loaded {total:,} RxNorm concepts via API into {CATALOG}.reference.rxnorm_concepts")
        rxnorm_loaded = True

if not rxnorm_loaded:
    load_rxnorm_from_api()

if not rxnorm_loaded:
    print(f"  RxNorm not loaded. To load:")
    print(f"  Option A: Upload RXNCONSO.RRF to {RXNORM_VOLUME}/")
    print(f"  Option B: Ensure network access to https://rxnav.nlm.nih.gov/REST/")
    print(f"  Download from: https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 5: Build Unified Medical Dictionary
# MAGIC
# MAGIC Combines ALL reference sources into a single search-optimized
# MAGIC  table. This is the primary lookup
# MAGIC table used by the dictionary-matching entity extraction layer.
# MAGIC
# MAGIC **Sources combined:**
# MAGIC - ICD-10-CM descriptions -> DIAGNOSIS
# MAGIC - LOINC long names / component text -> LAB_RESULT
# MAGIC - RxNorm drug names -> MEDICATION
# MAGIC - UMLS synonyms mapped by semantic group -> all entity types
# MAGIC - Clinical abbreviations -> all categories
# MAGIC
# MAGIC All terms are normalized (lowercased, stripped of punctuation) into
# MAGIC  for efficient fuzzy matching.

# COMMAND ----------

from pyspark.sql.functions import (
    col, lit, lower, regexp_replace, current_timestamp, explode, trim
)
from pyspark.sql.types import StructType, StructField, StringType

dictionary_parts = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5a: ICD-10-CM -> DIAGNOSIS terms

# COMMAND ----------

try:
    icd10_terms = spark.sql(f"""
        SELECT
            description AS term,
            'ICD10' AS source,
            'DIAGNOSIS' AS entity_type,
            code AS source_code,
            NULL AS cui
        FROM {CATALOG}.reference.icd10_codes_full
        WHERE is_billable = true AND description IS NOT NULL
    """)
    icd10_count = icd10_terms.count()
    dictionary_parts.append(icd10_terms)
    print(f"  ICD-10: {icd10_count:,} terms")
except Exception as e:
    print(f"  ICD-10 table not found -- skipping ({e})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5b: LOINC -> LAB_RESULT terms

# COMMAND ----------

try:
    # Long names
    loinc_long = spark.sql(f"""
        SELECT
            long_name AS term,
            'LOINC' AS source,
            'LAB_RESULT' AS entity_type,
            loinc_code AS source_code,
            NULL AS cui
        FROM {CATALOG}.reference.loinc_codes_full
        WHERE long_name IS NOT NULL AND long_name != ''
    """)

    # Component names (additional terms for matching)
    loinc_comp = spark.sql(f"""
        SELECT
            component AS term,
            'LOINC' AS source,
            'LAB_RESULT' AS entity_type,
            loinc_code AS source_code,
            NULL AS cui
        FROM {CATALOG}.reference.loinc_codes_full
        WHERE component IS NOT NULL AND component != ''
    """)

    loinc_terms = loinc_long.union(loinc_comp).distinct()
    loinc_count = loinc_terms.count()
    dictionary_parts.append(loinc_terms)
    print(f"  LOINC: {loinc_count:,} terms (long names + components)")
except Exception as e:
    print(f"  LOINC table not found -- skipping ({e})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5c: RxNorm -> MEDICATION terms

# COMMAND ----------

try:
    rxnorm_names = spark.sql(f"""
        SELECT
            name AS term,
            'RxNorm' AS source,
            'MEDICATION' AS entity_type,
            rxcui AS source_code,
            NULL AS cui
        FROM {CATALOG}.reference.rxnorm_concepts
        WHERE name IS NOT NULL
    """)

    # Also explode synonyms
    rxnorm_syns = (spark.table(f"{CATALOG}.reference.rxnorm_concepts")
        .filter(col("synonyms").isNotNull())
        .select(
            explode(col("synonyms")).alias("term"),
            lit("RxNorm").alias("source"),
            lit("MEDICATION").alias("entity_type"),
            col("rxcui").alias("source_code"),
            lit(None).cast("string").alias("cui"),
        )
    )

    rxnorm_terms = rxnorm_names.union(rxnorm_syns).distinct()
    rxnorm_count = rxnorm_terms.count()
    dictionary_parts.append(rxnorm_terms)
    print(f"  RxNorm: {rxnorm_count:,} terms (names + synonyms)")
except Exception as e:
    print(f"  RxNorm table not found -- skipping ({e})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5d: UMLS -> mapped by semantic group

# COMMAND ----------

try:
    # Explode UMLS synonyms and map semantic groups to entity types
    umls_terms = (spark.table(f"{CATALOG}.reference.umls_concepts")
        .filter(col("synonyms").isNotNull())
        .select(
            explode(col("synonyms")).alias("term"),
            lit("UMLS").alias("source"),
            col("semantic_group"),
            col("cui").alias("source_code"),
            col("cui"),
        )
    )

    from pyspark.sql.functions import when
    umls_terms = umls_terms.withColumn(
        "entity_type",
        when(col("semantic_group") == "DISO", "DIAGNOSIS")
        .when(col("semantic_group") == "CHEM", "MEDICATION")
        .when(col("semantic_group") == "PROC", "PROCEDURE")
        .when(col("semantic_group") == "PHYS", "LAB_RESULT")
        .otherwise("OTHER")
    ).drop("semantic_group")

    umls_count = umls_terms.count()
    dictionary_parts.append(umls_terms)
    print(f"  UMLS: {umls_count:,} terms (exploded synonyms)")
except Exception as e:
    print(f"  UMLS table not found -- skipping ({e})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5e: Clinical Abbreviations -> all categories

# COMMAND ----------

try:
    # Add abbreviation as term
    abbrev_as_term = spark.sql(f"""
        SELECT
            abbreviation AS term,
            'abbreviation' AS source,
            CASE
                WHEN category = 'diagnosis' THEN 'DIAGNOSIS'
                WHEN category = 'medication' THEN 'MEDICATION'
                WHEN category = 'lab' THEN 'LAB_RESULT'
                WHEN category = 'procedure' THEN 'PROCEDURE'
                ELSE 'OTHER'
            END AS entity_type,
            abbreviation AS source_code,
            NULL AS cui
        FROM {CATALOG}.reference.clinical_abbreviations
    """)

    # Also add expansion as term
    abbrev_expansion = spark.sql(f"""
        SELECT
            expansion AS term,
            'abbreviation' AS source,
            CASE
                WHEN category = 'diagnosis' THEN 'DIAGNOSIS'
                WHEN category = 'medication' THEN 'MEDICATION'
                WHEN category = 'lab' THEN 'LAB_RESULT'
                WHEN category = 'procedure' THEN 'PROCEDURE'
                ELSE 'OTHER'
            END AS entity_type,
            abbreviation AS source_code,
            NULL AS cui
        FROM {CATALOG}.reference.clinical_abbreviations
    """)

    abbrev_terms = abbrev_as_term.union(abbrev_expansion)
    abbrev_count = abbrev_terms.count()
    dictionary_parts.append(abbrev_terms)
    print(f"  Abbreviations: {abbrev_count:,} terms (abbreviation + expansion pairs)")
except Exception as e:
    print(f"  Abbreviation table not found -- skipping ({e})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5f: Merge, Normalize, and Write

# COMMAND ----------

if not dictionary_parts:
    print("  No dictionary sources available -- cannot build medical_dictionary")
    print("  Run 02_load_reference_codes first, then re-run this notebook")
else:
    # Union all parts
    from functools import reduce
    from pyspark.sql import DataFrame

    full_dict = reduce(DataFrame.unionByName, dictionary_parts)

    # Normalize terms: lowercase, strip extra whitespace, remove punctuation for matching
    full_dict = (full_dict
        .withColumn("term", trim(col("term")))
        .filter(col("term").isNotNull())
        .filter(col("term") != "")
        .withColumn(
            "term_normalized",
            lower(regexp_replace(trim(col("term")), r"[^a-zA-Z0-9\s]", ""))
        )
        .withColumn("loaded_at", current_timestamp())
        .select("term", "term_normalized", "source", "entity_type", "source_code", "cui", "loaded_at")
        .distinct()
    )

    full_dict.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference.medical_dictionary")

    total = spark.table(f"{CATALOG}.reference.medical_dictionary").count()
    print(f"  Loaded {total:,} terms into {CATALOG}.reference.medical_dictionary")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dictionary Summary

# COMMAND ----------

try:
    print("  === Medical Dictionary Summary ===")
    print("")

    spark.sql(f"""
        SELECT
            source,
            entity_type,
            COUNT(*) AS term_count
        FROM {CATALOG}.reference.medical_dictionary
        GROUP BY source, entity_type
        ORDER BY source, entity_type
    """).show(50, truncate=False)

    spark.sql(f"""
        SELECT
            entity_type,
            COUNT(*) AS total_terms,
            COUNT(DISTINCT term_normalized) AS unique_normalized
        FROM {CATALOG}.reference.medical_dictionary
        GROUP BY entity_type
        ORDER BY total_terms DESC
    """).show(truncate=False)

    grand_total = spark.table(f"{CATALOG}.reference.medical_dictionary").count()
    unique_terms = spark.table(f"{CATALOG}.reference.medical_dictionary").select("term_normalized").distinct().count()
    print(f"  Grand total: {grand_total:,} terms ({unique_terms:,} unique normalized)")
except Exception as e:
    print(f"  Could not generate summary: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ontology Load Complete
# MAGIC
# MAGIC | Table | Status | Source |
# MAGIC |-------|--------|--------|
# MAGIC |  | Always loaded | Hardcoded ~220 abbreviations |
# MAGIC |  | Loaded if RRF files or API key provided | UMLS Metathesaurus |
# MAGIC |  | Loaded if RF2 files provided | SNOMED-CT IS-A hierarchy |
# MAGIC |  | Loaded if extended map file provided | SNOMED-to-ICD-10-CM crosswalk |
# MAGIC |  | Loaded from RRF files or public REST API | RxNorm drug concepts |
# MAGIC |  | Built from all available sources | Unified search dictionary |
# MAGIC
# MAGIC ### Missing ontology files?
# MAGIC
# MAGIC | Ontology | Where to get it | Upload to |
# MAGIC |----------|----------------|-----------|
# MAGIC | **UMLS** | https://uts.nlm.nih.gov/uts/ (free license) |  |
# MAGIC | **SNOMED-CT** | https://mlds.ihtsdotools.org/ (free with NLM license) |  |
# MAGIC | **RxNorm** | https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html (free) |  |
# MAGIC
# MAGIC **Next:** Run  or .
