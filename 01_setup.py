# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Pipeline Setup
# MAGIC
# MAGIC Creates the catalog structure, pipeline tables, feedback tables, and UC Volume
# MAGIC required by the medical codification pipeline.
# MAGIC
# MAGIC **What this notebook creates:**
# MAGIC - 5 schemas: `raw`, `extracted`, `codified`, `reference`, `feedback`
# MAGIC - Pipeline tables: `charts`, `entities`, `icd10_mappings`, `loinc_mappings`
# MAGIC - Feedback table: `human_corrections` (for reviewer decisions from the app)
# MAGIC - UC Volume: `raw.chart_pdfs` (PDF storage)
# MAGIC
# MAGIC **What this notebook does NOT create:**
# MAGIC - Reference data (ICD-10, LOINC) -- that is notebook 02
# MAGIC
# MAGIC **Estimated runtime:** < 30 seconds

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("CATALOG", "mv_catalog", "Unity Catalog Name")
CATALOG = dbutils.widgets.get("CATALOG")

SCHEMAS = ["raw", "extracted", "codified", "reference", "feedback"]

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Schemas

# COMMAND ----------

for schema in SCHEMAS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
    print(f"  Schema {CATALOG}.{schema} ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Pipeline Tables
# MAGIC
# MAGIC These tables hold the data as it flows through each stage:
# MAGIC 1. **raw.charts** -- Ingested chart metadata and extracted text
# MAGIC 2. **extracted.entities** -- Clinical entities (diagnoses, labs, meds, vitals) from NER
# MAGIC 3. **codified.icd10_mappings** -- Final ICD-10 code assignments with full audit trail
# MAGIC 4. **codified.loinc_mappings** -- Final LOINC code assignments with full audit trail
# MAGIC
# MAGIC The codified tables include multi-pass resolution fields: `resolution_path`,
# MAGIC `r1_code`, `r2_verdict`, `r2_code`, `arbiter_code`, and `arbiter_reasoning`.

# COMMAND ----------

# --- raw.charts ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.raw.charts (
    chart_id STRING,
    patient_id STRING,
    file_name STRING,
    file_path STRING,
    chart_type STRING,
    provider STRING,
    facility STRING,
    chart_date DATE,
    ingested_at TIMESTAMP,
    raw_text STRING,
    page_count INT,
    extraction_method STRING
)
USING DELTA
COMMENT 'Raw medical chart metadata and extracted text'
""")
print("  raw.charts ready")

# COMMAND ----------

# --- extracted.entities ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.entities (
    entity_id STRING,
    chart_id STRING,
    entity_type STRING,
    entity_text STRING,
    entity_context STRING,
    start_offset INT,
    end_offset INT,
    confidence DOUBLE,
    specimen_type STRING,
    method STRING,
    timing STRING,
    value STRING,
    unit STRING,
    extracted_at TIMESTAMP
)
USING DELTA
COMMENT 'Clinical entities extracted via NER from chart text'
""")
print("  extracted.entities ready")

# COMMAND ----------

# --- codified.icd10_mappings ---
# Full audit trail schema from the multi-pass codification pipeline (v2)
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.codified.icd10_mappings (
    mapping_id STRING,
    entity_id STRING,
    chart_id STRING,
    entity_text STRING,
    icd10_code STRING,
    icd10_description STRING,
    confidence DOUBLE,
    resolution_path STRING COMMENT 'R1_R2_AGREE | ARBITER_CHOSE_R1 | ARBITER_CHOSE_R2 | DISPUTED_UNRESOLVED',
    r1_code STRING,
    r1_reasoning STRING,
    r2_verdict STRING,
    r2_code STRING,
    r2_reasoning STRING,
    arbiter_code STRING,
    arbiter_reasoning STRING,
    is_specific BOOLEAN,
    codified_at TIMESTAMP
)
USING DELTA
COMMENT 'ICD-10 code mappings with multi-pass audit trail'
""")
print("  codified.icd10_mappings ready")

# COMMAND ----------

# --- codified.loinc_mappings ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.codified.loinc_mappings (
    mapping_id STRING,
    entity_id STRING,
    chart_id STRING,
    entity_text STRING,
    loinc_code STRING,
    loinc_long_name STRING,
    confidence DOUBLE,
    resolution_path STRING COMMENT 'R1_R2_AGREE | ARBITER_CHOSE_R1 | ARBITER_CHOSE_R2 | DISPUTED_UNRESOLVED',
    r1_code STRING,
    r1_reasoning STRING,
    r2_verdict STRING,
    r2_code STRING,
    r2_reasoning STRING,
    arbiter_code STRING,
    arbiter_reasoning STRING,
    specimen_type STRING,
    method STRING,
    timing STRING,
    codified_at TIMESTAMP
)
USING DELTA
COMMENT 'LOINC code mappings with multi-pass audit trail'
""")
print("  codified.loinc_mappings ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feedback Table
# MAGIC
# MAGIC The review app writes human corrections back to this table.
# MAGIC These corrections feed into model-improvement loops and audit reporting.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.feedback.human_corrections (
    correction_id STRING,
    mapping_id STRING,
    code_type STRING COMMENT 'ICD10 or LOINC',
    original_code STRING,
    corrected_code STRING,
    entity_text STRING,
    entity_context STRING,
    corrected_by STRING,
    corrected_at TIMESTAMP
)
USING DELTA
COMMENT 'Human corrections from the review app for model improvement'
""")
print("  feedback.human_corrections ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create UC Volume for PDF Storage
# MAGIC
# MAGIC PDFs are stored in a Unity Catalog Volume at `/Volumes/{CATALOG}/raw/chart_pdfs/`.
# MAGIC This works with both serverless and classic compute via the FUSE mount.

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.raw.chart_pdfs")
print(f"  Volume {CATALOG}.raw.chart_pdfs ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Complete
# MAGIC
# MAGIC | Category | Assets |
# MAGIC |----------|--------|
# MAGIC | **Schemas** | `raw`, `extracted`, `codified`, `reference`, `feedback` |
# MAGIC | **Pipeline tables** | `raw.charts`, `extracted.entities`, `codified.icd10_mappings`, `codified.loinc_mappings` |
# MAGIC | **Feedback** | `feedback.human_corrections` |
# MAGIC | **Storage** | `raw.chart_pdfs` UC Volume |
# MAGIC
# MAGIC **Next:** Run `02_load_reference_codes` to load ICD-10-CM and LOINC reference data.
