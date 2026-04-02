# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare Charts for Production
# MAGIC
# MAGIC **Use this notebook to register your clinical chart PDFs for processing.**
# MAGIC
# MAGIC This notebook scans a Unity Catalog Volume for PDF files, extracts metadata,
# MAGIC and registers them in the `raw.charts` table so the extraction pipeline can process them.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Run `01_setup.py` first to create schemas and tables
# MAGIC 2. Upload your clinical chart PDFs to: `/Volumes/<CATALOG>/raw/chart_pdfs/`
# MAGIC
# MAGIC ## What This Notebook Does
# MAGIC 1. Scans the chart_pdfs Volume for all PDF files
# MAGIC 2. Deduplicates against already-registered charts (safe to re-run)
# MAGIC 3. Registers new charts in `raw.charts` with metadata
# MAGIC 4. Optionally extracts text from PDFs using `ai_parse_document` for preview

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("CATALOG", "mv_catalog", "Unity Catalog Name")
CATALOG = dbutils.widgets.get("CATALOG")
spark.sql(f"USE CATALOG {CATALOG}")

VOLUME_PATH = f"/Volumes/{CATALOG}/raw/chart_pdfs"
print(f"Scanning: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Scan Volume for PDF Files

# COMMAND ----------

import os
from pyspark.sql.functions import (
    col, lit, current_timestamp, input_file_name,
    regexp_extract, to_date, monotonically_increasing_id, concat
)

# List all PDFs in the volume
pdf_files = []
for f in os.listdir(VOLUME_PATH):
    if f.lower().endswith('.pdf'):
        pdf_files.append({
            "file_name": f,
            "file_path": f"{VOLUME_PATH}/{f}",
            "file_size_bytes": os.path.getsize(f"{VOLUME_PATH}/{f}"),
        })

print(f"Found {len(pdf_files)} PDF files in volume")

if not pdf_files:
    print(f"\n⚠ No PDF files found in {VOLUME_PATH}")
    print(f"  Upload your clinical chart PDFs to this location first.")
    print(f"  You can upload via:")
    print(f"    - Databricks UI: Catalog → Volumes → raw → chart_pdfs → Upload")
    print(f"    - CLI: databricks fs cp /local/path/*.pdf dbfs:{VOLUME_PATH}/")
    print(f"    - Python: dbutils.fs.cp('file:/local/path/chart.pdf', '{VOLUME_PATH}/chart.pdf')")
    dbutils.notebook.exit("No PDFs found — upload files and re-run")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Check for Already-Registered Charts

# COMMAND ----------

# Get existing chart file names to avoid duplicates
existing_charts = spark.sql(f"""
    SELECT file_name FROM {CATALOG}.raw.charts
""").select("file_name").rdd.flatMap(lambda x: x).collect()

existing_set = set(existing_charts)
new_files = [f for f in pdf_files if f["file_name"] not in existing_set]

print(f"Already registered: {len(existing_set)} charts")
print(f"New files to register: {len(new_files)} charts")

if not new_files:
    print("\n✓ All PDFs are already registered. Nothing to do.")
    print("  If you've added new PDFs, make sure they have unique filenames.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Register New Charts
# MAGIC
# MAGIC Each chart gets a unique `chart_id` and is registered with available metadata.
# MAGIC The `raw_text` column is left empty — it will be populated by notebook 04 (extraction).

# COMMAND ----------

if new_files:
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
    import uuid
    from datetime import datetime

    # Build registration records
    records = []
    for f in new_files:
        chart_id = f"CHT-{uuid.uuid4().hex[:12].upper()}"
        records.append({
            "chart_id": chart_id,
            "patient_id": None,  # Will be extracted by NER pipeline
            "file_name": f["file_name"],
            "file_path": VOLUME_PATH,
            "chart_type": "clinical_note",
            "provider": None,  # Will be extracted
            "facility": None,  # Will be extracted
            "chart_date": None,  # Will be extracted
            "raw_text": None,  # Will be populated by extraction notebook
            "page_count": None,  # Will be populated by extraction
            "extraction_method": "pending",
            "file_size_bytes": f["file_size_bytes"],
        })

    schema = StructType([
        StructField("chart_id", StringType()),
        StructField("patient_id", StringType()),
        StructField("file_name", StringType()),
        StructField("file_path", StringType()),
        StructField("chart_type", StringType()),
        StructField("provider", StringType()),
        StructField("facility", StringType()),
        StructField("chart_date", StringType()),
        StructField("raw_text", StringType()),
        StructField("page_count", IntegerType()),
        StructField("extraction_method", StringType()),
        StructField("file_size_bytes", LongType()),
    ])

    new_df = spark.createDataFrame(records, schema=schema)
    new_df = new_df.withColumn("ingested_at", current_timestamp())

    # Append to existing charts table (don't overwrite)
    new_df.select(
        "chart_id", "patient_id", "file_name", "file_path",
        "chart_type", "provider", "facility",
        col("chart_date").cast("date").alias("chart_date"),
        "ingested_at", "raw_text", "page_count", "extraction_method"
    ).write.mode("append").saveAsTable(f"{CATALOG}.raw.charts")

    print(f"✓ Registered {len(records)} new charts in {CATALOG}.raw.charts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Preview — Extract Text from a Sample (Optional)
# MAGIC
# MAGIC Uses `ai_parse_document` to extract text from a few charts as a validation step.
# MAGIC The full extraction runs in notebook 04.

# COMMAND ----------

# Preview: parse 3 random charts to validate PDFs are readable
sample_df = spark.sql(f"""
    SELECT chart_id, file_name,
        SUBSTRING(
            ai_parse_document(
                content, map('mode', 'OCR')
            ):content::STRING, 1, 300
        ) AS text_preview
    FROM (
        SELECT *, ROW_NUMBER() OVER (ORDER BY RAND()) AS rn
        FROM read_files('{VOLUME_PATH}/', format => 'binaryFile')
    )
    WHERE rn <= 3
""")

print("Sample text extraction preview:")
sample_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Summary

# COMMAND ----------

total = spark.table(f"{CATALOG}.raw.charts").count()
pending = spark.sql(f"""
    SELECT COUNT(*) FROM {CATALOG}.raw.charts
    WHERE extraction_method = 'pending' OR raw_text IS NULL
""").collect()[0][0]

print(f"{'='*50}")
print(f"  Total charts registered: {total}")
print(f"  Pending extraction:      {pending}")
print(f"  Already extracted:       {total - pending}")
print(f"{'='*50}")
print()
if pending > 0:
    print(f"Next step: Run notebook 04_extract_entities to process {pending} pending charts.")
    print(f"Estimated time: ~{pending * 10 // 60} minutes ({pending} charts × ~10s each)")
else:
    print("All charts have been extracted. Run notebook 05_codify_multipass next.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix: Bulk Upload Methods
# MAGIC
# MAGIC ### From Local Machine (via CLI)
# MAGIC ```bash
# MAGIC # Upload a directory of PDFs
# MAGIC databricks fs cp -r /local/path/to/charts/ dbfs:/Volumes/<catalog>/raw/chart_pdfs/ --profile <profile>
# MAGIC ```
# MAGIC
# MAGIC ### From Cloud Storage (via Spark)
# MAGIC ```python
# MAGIC # Copy from S3
# MAGIC dbutils.fs.cp("s3://your-bucket/charts/", "/Volumes/<catalog>/raw/chart_pdfs/", recurse=True)
# MAGIC
# MAGIC # Copy from ADLS
# MAGIC dbutils.fs.cp("abfss://container@storage.dfs.core.windows.net/charts/",
# MAGIC               "/Volumes/<catalog>/raw/chart_pdfs/", recurse=True)
# MAGIC ```
# MAGIC
# MAGIC ### From a Delta Table with Binary Content
# MAGIC ```python
# MAGIC # If charts are stored as binary in a Delta table
# MAGIC charts_binary = spark.table("source.charts_binary")
# MAGIC for row in charts_binary.collect():
# MAGIC     path = f"/Volumes/<catalog>/raw/chart_pdfs/{row.file_name}"
# MAGIC     with open(path, "wb") as f:
# MAGIC         f.write(row.content)
# MAGIC ```
# MAGIC
# MAGIC ### Programmatic Registration (Skip Volume)
# MAGIC If charts are already accessible via a path (e.g., external location),
# MAGIC you can register them directly without copying to the Volume:
# MAGIC ```python
# MAGIC spark.sql(f"""
# MAGIC     INSERT INTO {CATALOG}.raw.charts (chart_id, file_name, file_path, chart_type, ingested_at, extraction_method)
# MAGIC     SELECT
# MAGIC         CONCAT('CHT-', uuid()) AS chart_id,
# MAGIC         regexp_extract(path, '[^/]+$', 0) AS file_name,
# MAGIC         '/your/external/path' AS file_path,
# MAGIC         'clinical_note' AS chart_type,
# MAGIC         current_timestamp() AS ingested_at,
# MAGIC         'pending' AS extraction_method
# MAGIC     FROM LIST FILES AT '/your/external/path'
# MAGIC     WHERE path LIKE '%.pdf'
# MAGIC """)
# MAGIC ```
