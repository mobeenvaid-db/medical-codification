# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Clinical Entity Extraction
# MAGIC
# MAGIC Extracts diagnoses, lab results, medications, and vitals from clinical chart text using
# MAGIC Databricks-native AI functions -- no external dependencies required.
# MAGIC
# MAGIC **Pipeline:**
# MAGIC 1. `ai_parse_document` -- OCR extraction from PDFs stored in UC Volume
# MAGIC 2. `ai_query` -- Structured clinical NER via Foundation Model API
# MAGIC 3. JSON parsing into the `extracted.entities` table
# MAGIC
# MAGIC **Key design decisions:**
# MAGIC - Raw AI output is saved to a temp table first (not cached in memory --
# MAGIC   serverless does not support `.cache()`)
# MAGIC - JSON is cleaned of markdown fences before parsing
# MAGIC - Specimen type, method, and timing are extracted for downstream LOINC disambiguation
# MAGIC
# MAGIC **Estimated runtime:** ~10 minutes for 100 charts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("CATALOG", "mv_catalog", "Unity Catalog Name")
CATALOG = dbutils.widgets.get("CATALOG")

MODEL = "databricks-claude-sonnet-4-6"

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Parse PDFs with ai_parse_document
# MAGIC
# MAGIC This demonstrates the production pattern for extracting text from real scanned PDFs.
# MAGIC `ai_parse_document` takes BINARY content -- use `read_files` to load PDF bytes
# MAGIC from the UC Volume.
# MAGIC
# MAGIC For this demo, we already have `raw_text` in the charts table (generated in notebook 03),
# MAGIC so this step is illustrative. In production, you would use this to extract text from
# MAGIC real scanned/photographed charts.

# COMMAND ----------

# ai_parse_document sample -- shows the production OCR pattern
# This reads PDF binary content directly from the UC Volume
parsed_df = spark.sql(f"""
    SELECT
        path AS file_path,
        SUBSTRING(
            ai_parse_document(content):content::STRING,
            1, 500
        ) AS parsed_preview
    FROM read_files(
        '/Volumes/{CATALOG}/raw/chart_pdfs/',
        format => 'binaryFile'
    )
    LIMIT 3
""")

print("ai_parse_document sample (OCR from PDFs):")
parsed_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Extract Clinical Entities with AI
# MAGIC
# MAGIC Uses `ai_query` with a structured prompt to extract diagnoses, labs, medications,
# MAGIC and vitals from each chart. The prompt requests:
# MAGIC - `entity_type`: DIAGNOSIS, LAB_RESULT, MEDICATION, VITAL_SIGN
# MAGIC - `specimen_type`, `method`, `timing`: Critical metadata for LOINC disambiguation
# MAGIC - `confidence`: Extraction confidence (0.0-1.0)
# MAGIC
# MAGIC Results are saved to a temp table to avoid re-running `ai_query` if downstream
# MAGIC parsing needs adjustment.

# COMMAND ----------

charts_df = spark.table(f"{CATALOG}.raw.charts")
print(f"  Processing {charts_df.count()} charts")

# COMMAND ----------

extraction_sql = f"""
SELECT
    chart_id,
    ai_query(
        '{MODEL}',
        CONCAT(
            'You are a clinical NER system. Extract ALL clinical entities from this medical chart text. ',
            'Return a JSON array where each element has these fields: ',
            '"entity_type" (one of: DIAGNOSIS, LAB_RESULT, MEDICATION, VITAL_SIGN), ',
            '"entity_text" (the exact text of the entity as written in the chart), ',
            '"value" (numeric value if applicable, null otherwise), ',
            '"unit" (unit of measurement if applicable, null otherwise), ',
            '"specimen_type" (for labs: blood, serum, plasma, urine, etc. null if not stated), ',
            '"method" (for labs: HPLC, immunoassay, CKD-EPI, calculated, automated, etc. null if not stated), ',
            '"timing" (for labs: fasting, random, point in time, 2h post-meal, etc. null if not stated), ',
            '"confidence" (your confidence 0.0-1.0 that this entity was correctly extracted). ',
            'Be thorough -- extract every diagnosis, every lab result, every medication, and every vital sign. ',
            'For diagnoses, include the full diagnostic statement as written. ',
            'For labs, capture the specimen type and method if mentioned -- these are critical for LOINC mapping. ',
            'Return ONLY the JSON array, no other text.',
            '\\n\\nCHART TEXT:\\n',
            raw_text
        )
    ) AS extraction_result
FROM {CATALOG}.raw.charts
"""

print("  Running entity extraction on all charts via ai_query...")
print("  This will take several minutes...")

extraction_df = spark.sql(extraction_sql)

# Save to temp table so we don't re-run ai_query if parsing needs adjustment
extraction_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.extracted.raw_extractions")
print(f"  Extraction complete -- results saved to {CATALOG}.extracted.raw_extractions")

# COMMAND ----------

# Preview
spark.sql(f"""
    SELECT chart_id, SUBSTRING(extraction_result, 1, 500) AS preview
    FROM {CATALOG}.extracted.raw_extractions
    LIMIT 3
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Parse Extraction Results into Structured Table
# MAGIC
# MAGIC The AI returns JSON arrays. We parse them into individual entity rows,
# MAGIC handling markdown fences that the model sometimes wraps around JSON output.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType
from pyspark.sql.functions import explode, from_json, col, expr, lit, current_timestamp, concat

# Define schema for extracted entities
entity_schema = ArrayType(StructType([
    StructField("entity_type", StringType()),
    StructField("entity_text", StringType()),
    StructField("value", StringType()),
    StructField("unit", StringType()),
    StructField("specimen_type", StringType()),
    StructField("method", StringType()),
    StructField("timing", StringType()),
    StructField("confidence", DoubleType()),
]))

# Read from saved table
extraction_df = spark.table(f"{CATALOG}.extracted.raw_extractions")

# Clean the JSON response (strip markdown fences if present)
cleaned_df = extraction_df.withColumn(
    "clean_json",
    F.regexp_replace(
        F.regexp_replace("extraction_result", "```json\\n?", ""),
        "\\n?```", ""
    )
)

# Parse JSON and explode into individual entities
entities_df = (cleaned_df
    .withColumn("entities", from_json("clean_json", entity_schema))
    .filter(col("entities").isNotNull())
    .select("chart_id", explode("entities").alias("entity"))
    .select(
        concat(lit("ENT-"), expr("uuid()")).alias("entity_id"),
        "chart_id",
        col("entity.entity_type").alias("entity_type"),
        col("entity.entity_text").alias("entity_text"),
        lit(None).cast("string").alias("entity_context"),
        lit(None).cast("int").alias("start_offset"),
        lit(None).cast("int").alias("end_offset"),
        col("entity.confidence").alias("confidence"),
        col("entity.specimen_type").alias("specimen_type"),
        col("entity.method").alias("method"),
        col("entity.timing").alias("timing"),
        col("entity.value").alias("value"),
        col("entity.unit").alias("unit"),
        current_timestamp().alias("extracted_at"),
    )
)

# Write to extracted.entities
entities_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.extracted.entities")

entity_count = spark.table(f"{CATALOG}.extracted.entities").count()
print(f"  Extracted {entity_count} entities from charts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Extraction Quality Summary

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Entity distribution by type
# MAGIC SELECT
# MAGIC     entity_type,
# MAGIC     COUNT(*) AS entity_count,
# MAGIC     ROUND(AVG(confidence), 3) AS avg_confidence,
# MAGIC     ROUND(MIN(confidence), 3) AS min_confidence
# MAGIC FROM ${CATALOG}.extracted.entities
# MAGIC GROUP BY entity_type
# MAGIC ORDER BY entity_count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sample extracted lab results with disambiguation metadata
# MAGIC SELECT entity_type, entity_text, value, unit, specimen_type, method, timing, confidence
# MAGIC FROM ${CATALOG}.extracted.entities
# MAGIC WHERE entity_type = 'LAB_RESULT'
# MAGIC ORDER BY confidence DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Charts with entity counts
# MAGIC SELECT
# MAGIC     c.chart_id,
# MAGIC     c.provider,
# MAGIC     c.chart_date,
# MAGIC     COUNT(e.entity_id) AS total_entities,
# MAGIC     SUM(CASE WHEN e.entity_type = 'DIAGNOSIS' THEN 1 ELSE 0 END) AS diagnoses,
# MAGIC     SUM(CASE WHEN e.entity_type = 'LAB_RESULT' THEN 1 ELSE 0 END) AS labs,
# MAGIC     SUM(CASE WHEN e.entity_type = 'MEDICATION' THEN 1 ELSE 0 END) AS medications
# MAGIC FROM ${CATALOG}.raw.charts c
# MAGIC LEFT JOIN ${CATALOG}.extracted.entities e ON c.chart_id = e.chart_id
# MAGIC GROUP BY c.chart_id, c.provider, c.chart_date
# MAGIC ORDER BY total_entities DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Extracted clinical entities from all charts using Foundation Model API via `ai_query`.
# MAGIC Each entity includes:
# MAGIC - **entity_type**: DIAGNOSIS, LAB_RESULT, MEDICATION, VITAL_SIGN
# MAGIC - **entity_text**: The clinical text as written
# MAGIC - **specimen_type / method / timing**: Critical metadata for LOINC disambiguation
# MAGIC - **value / unit**: Numeric results for lab tests
# MAGIC - **confidence**: Extraction confidence score
# MAGIC
# MAGIC **Next:** Run `05_codify_multipass` to map entities to ICD-10 and LOINC codes.
