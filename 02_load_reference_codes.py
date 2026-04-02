# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Load ICD-10-CM & LOINC Reference Codes
# MAGIC
# MAGIC Ingests the **full** ICD-10-CM (~98K codes, ~74K billable) and LOINC (~100K codes)
# MAGIC reference sets from their official sources, then creates search-optimized tables
# MAGIC and a version-tracking mechanism.
# MAGIC
# MAGIC **Sources:**
# MAGIC - **ICD-10-CM:** CDC/CMS public-domain flat files -- no license required
# MAGIC - **LOINC:** CSV download from loinc.org (free registration) or FHIR API fallback
# MAGIC
# MAGIC **Tables created:**
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `reference.icd10_codes_full` | Complete ICD-10-CM with chapter/category hierarchy |
# MAGIC | `reference.loinc_codes_full` | Complete LOINC with 6-axis detail |
# MAGIC | `reference.icd10_search` | Billable-only ICD-10 codes with search text |
# MAGIC | `reference.loinc_search` | Active LOINC codes with search text |
# MAGIC | `reference.icd10_category_index` | Category-grouped codes for the multi-pass codification approach |
# MAGIC | `reference.code_set_versions` | Version tracking for update detection |
# MAGIC
# MAGIC **Update cadence:**
# MAGIC - ICD-10-CM: Annually (October 1), with mid-year addendum (April 1)
# MAGIC - LOINC: ~3x per year (February, June, December typically)
# MAGIC
# MAGIC **Estimated runtime:** 2-5 minutes (depends on download speed)

# COMMAND ----------

# MAGIC %pip install requests
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("CATALOG", "mv_catalog", "Unity Catalog Name")
CATALOG = dbutils.widgets.get("CATALOG")

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 1: Full ICD-10-CM Code Set (CDC/CMS)
# MAGIC
# MAGIC The CDC publishes the complete ICD-10-CM code set as a fixed-width text file
# MAGIC inside a ZIP archive. This is public domain -- no license or authentication required.

# COMMAND ----------

import requests
import zipfile
import io
import re
from datetime import datetime

# CDC FTP mirror -- public domain, no auth needed
ICD10_URLS = [
    # FY 2026 (Oct 2025 - Sep 2026) -- primary
    "https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2026/icd10cm-Code%20Descriptions-2026.zip",
    # Fallback: CMS direct
    "https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip",
]

def download_icd10_zip():
    """Download ICD-10-CM code descriptions ZIP from CDC or CMS."""
    for url in ICD10_URLS:
        try:
            print(f"  Trying: {url}")
            resp = requests.get(url, timeout=60, allow_redirects=True)
            if resp.status_code == 200:
                print(f"  Downloaded {len(resp.content):,} bytes")
                return resp.content
            print(f"  HTTP {resp.status_code}")
        except Exception as e:
            print(f"  Error: {e}")
    raise RuntimeError("Could not download ICD-10-CM from any source")

zip_bytes = download_icd10_zip()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parse the ICD-10-CM Order File
# MAGIC
# MAGIC The order file is fixed-width text:
# MAGIC - Positions 1-5: Order number
# MAGIC - Position 7+: Code (variable length, 3-7 chars)
# MAGIC - Header flag: 0 = header/category, 1 = billable
# MAGIC - Short description, then long description

# COMMAND ----------

def parse_icd10_zip(zip_content):
    """Parse the ICD-10-CM order file from the downloaded ZIP."""
    zf = zipfile.ZipFile(io.BytesIO(zip_content))

    # Find the order file (contains all codes in tabular order)
    order_file = None
    for name in zf.namelist():
        lower = name.lower()
        if 'order' in lower and lower.endswith('.txt'):
            order_file = name
            break
        if 'code' in lower and 'description' in lower and lower.endswith('.txt'):
            order_file = name
            break

    # Fallback: any .txt file
    if not order_file:
        for name in zf.namelist():
            if name.endswith('.txt') and not name.startswith('__'):
                order_file = name
                break

    if not order_file:
        print(f"  Available files in ZIP: {zf.namelist()}")
        raise RuntimeError("Could not find order file in ZIP")

    print(f"  Parsing: {order_file}")

    codes = []
    content = zf.read(order_file).decode('utf-8', errors='replace')

    for line in content.strip().split('\n'):
        line = line.rstrip()
        if len(line) < 20:
            continue

        parts = line.split()
        if len(parts) < 4:
            continue

        try:
            order_num = int(parts[0])
        except ValueError:
            continue

        code = parts[1]

        # Validate ICD-10-CM code pattern (letter + digits)
        if not re.match(r'^[A-Z]\d', code):
            continue

        header_flag = parts[2] if parts[2] in ('0', '1') else '0'
        is_billable = header_flag == '1'

        # Everything after the header flag is the description
        desc_start = line.find(parts[2], line.find(code) + len(code)) + 2
        description = line[desc_start:].strip()

        # Format the code with a dot (e.g., E119 -> E11.9)
        if len(code) > 3 and '.' not in code:
            formatted_code = code[:3] + '.' + code[3:]
        else:
            formatted_code = code

        category = code[:3]

        codes.append({
            "order_number": order_num,
            "code": formatted_code,
            "code_raw": code,
            "is_billable": is_billable,
            "is_header": not is_billable,
            "description": description,
            "category": category,
        })

    print(f"  Parsed {len(codes):,} ICD-10-CM codes")
    return codes

icd10_codes = parse_icd10_zip(zip_bytes)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enrich with Chapter Hierarchy
# MAGIC
# MAGIC ICD-10-CM is organized into 22 chapters by code range.
# MAGIC We map each code to its chapter for downstream category retrieval.

# COMMAND ----------

ICD10_CHAPTERS = [
    ("A00-B99", "Certain infectious and parasitic diseases"),
    ("C00-D49", "Neoplasms"),
    ("D50-D89", "Diseases of the blood and blood-forming organs"),
    ("E00-E89", "Endocrine, nutritional and metabolic diseases"),
    ("F01-F99", "Mental, behavioral and neurodevelopmental disorders"),
    ("G00-G99", "Diseases of the nervous system"),
    ("H00-H59", "Diseases of the eye and adnexa"),
    ("H60-H95", "Diseases of the ear and mastoid process"),
    ("I00-I99", "Diseases of the circulatory system"),
    ("J00-J99", "Diseases of the respiratory system"),
    ("K00-K95", "Diseases of the digestive system"),
    ("L00-L99", "Diseases of the skin and subcutaneous tissue"),
    ("M00-M99", "Diseases of the musculoskeletal system and connective tissue"),
    ("N00-N99", "Diseases of the genitourinary system"),
    ("O00-O9A", "Pregnancy, childbirth and the puerperium"),
    ("P00-P96", "Certain conditions originating in the perinatal period"),
    ("Q00-Q99", "Congenital malformations and chromosomal abnormalities"),
    ("R00-R99", "Symptoms, signs and abnormal clinical and laboratory findings"),
    ("S00-T88", "Injury, poisoning and certain other consequences of external causes"),
    ("U00-U85", "Codes for special purposes"),
    ("V00-Y99", "External causes of morbidity"),
    ("Z00-Z99", "Factors influencing health status and contact with health services"),
]

def get_chapter(code):
    """Determine the ICD-10-CM chapter for a code."""
    first_char = code[0].upper()
    code_num = int(code[1:3]) if len(code) >= 3 and code[1:3].isdigit() else 0

    for range_str, chapter_name in ICD10_CHAPTERS:
        start, end = range_str.split('-')
        start_char, start_num = start[0], int(start[1:3])
        end_char, end_num = end[0], int(end[1:3]) if end[1:3].isdigit() else 99

        if first_char == start_char and start_num <= code_num <= end_num:
            return chapter_name
        elif start_char < first_char < end_char:
            return chapter_name
        elif first_char == end_char and code_num <= end_num:
            return chapter_name

    return "Unknown"

for c in icd10_codes:
    c["chapter"] = get_chapter(c["code_raw"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Full ICD-10-CM to Delta Table

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, BooleanType, IntegerType
from pyspark.sql.functions import current_timestamp, lit

icd10_schema = StructType([
    StructField("order_number", IntegerType()),
    StructField("code", StringType()),
    StructField("code_raw", StringType()),
    StructField("is_billable", BooleanType()),
    StructField("is_header", BooleanType()),
    StructField("description", StringType()),
    StructField("category", StringType()),
    StructField("chapter", StringType()),
])

icd10_df = spark.createDataFrame(icd10_codes, schema=icd10_schema)

icd10_final = (icd10_df
    .withColumn("source", lit("CMS/CDC"))
    .withColumn("version", lit("FY2026"))
    .withColumn("loaded_at", current_timestamp())
)

icd10_final.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference.icd10_codes_full")

total = spark.table(f"{CATALOG}.reference.icd10_codes_full").count()
billable = spark.table(f"{CATALOG}.reference.icd10_codes_full").filter("is_billable = true").count()
print(f"  Loaded {total:,} ICD-10-CM codes ({billable:,} billable) into {CATALOG}.reference.icd10_codes_full")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 2: Full LOINC Code Set
# MAGIC
# MAGIC LOINC codes are loaded in this priority order:
# MAGIC 1. **CSV download** (preferred) -- upload the official LOINC CSV to the UC Volume
# MAGIC 2. **FHIR API** (fallback) -- programmatic access via `fhir.loinc.org`
# MAGIC
# MAGIC For production, use the CSV approach. Download from https://loinc.org/downloads/
# MAGIC and upload to `/Volumes/{CATALOG}/reference/loinc_files/`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option A: Load LOINC from CSV in UC Volume (Recommended)

# COMMAND ----------

# Create the volume for LOINC files if it doesn't exist
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.reference.loinc_files")

def load_loinc_from_csv(volume_path=None):
    """Load LOINC from the official CSV download in a UC Volume."""
    if volume_path is None:
        volume_path = f"/Volumes/{CATALOG}/reference/loinc_files"

    try:
        files = dbutils.fs.ls(volume_path)
        csv_files = [f for f in files if f.name.lower().endswith('.csv') and 'loinc' in f.name.lower()]
        if not csv_files:
            return None

        csv_path = csv_files[0].path
        print(f"  Found LOINC CSV: {csv_path}")

        loinc_df = (spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("quote", '"')
            .option("escape", '"')
            .csv(csv_path)
        )

        # Map official LOINC CSV columns to our schema
        from pyspark.sql.functions import col, coalesce, lit
        result = loinc_df.select(
            col("LOINC_NUM").alias("loinc_code"),
            coalesce(col("LONG_COMMON_NAME"), col("SHORTNAME"), lit("")).alias("long_name"),
            coalesce(col("COMPONENT"), lit("")).alias("component"),
            coalesce(col("PROPERTY"), lit("")).alias("property_type"),
            coalesce(col("TIME_ASPCT"), lit("")).alias("timing"),
            coalesce(col("SYSTEM"), lit("")).alias("system_specimen"),
            coalesce(col("SCALE_TYP"), lit("")).alias("scale"),
            coalesce(col("METHOD_TYP"), lit("")).alias("method"),
            coalesce(col("CLASS"), lit("")).alias("class_type"),
            coalesce(col("STATUS"), lit("")).alias("status"),
            col("LONG_COMMON_NAME").alias("display"),
        )

        count = result.count()
        print(f"  Loaded {count:,} LOINC codes from CSV")
        return result

    except Exception as e:
        print(f"  No LOINC CSV found in {volume_path}: {e}")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option B: LOINC FHIR API (Fallback)
# MAGIC
# MAGIC If no CSV is available, attempt to download from the LOINC FHIR terminology server.
# MAGIC Requires a free account from https://loinc.org/register/.

# COMMAND ----------

import requests
import time

FHIR_BASE = "https://fhir.loinc.org"

def fetch_loinc_fhir(username, password, max_codes=None):
    """Fetch LOINC codes from the FHIR terminology service."""
    auth = (username, password) if username else None
    headers = {"Accept": "application/fhir+json"}

    codes = []
    offset = 0
    page_size = 1000

    while True:
        url = f"{FHIR_BASE}/CodeSystem/loinc?_count={page_size}&_offset={offset}"

        try:
            resp = requests.get(url, auth=auth, headers=headers, timeout=120)
            if resp.status_code == 401:
                print("  LOINC authentication failed. Check credentials.")
                break
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                break

            data = resp.json()
            concepts = data.get("concept", [])
            if not concepts:
                print(f"  No more codes at offset {offset}")
                break

            for concept in concepts:
                code_entry = {
                    "loinc_code": concept.get("code", ""),
                    "display": concept.get("display", ""),
                    "component": "",
                    "property_type": "",
                    "timing": "",
                    "system_specimen": "",
                    "scale": "",
                    "method": "",
                    "class_type": "",
                    "status": "",
                    "long_name": concept.get("display", ""),
                }

                for prop in concept.get("property", []):
                    prop_code = prop.get("code", "")
                    prop_value = prop.get("valueString", prop.get("valueCoding", {}).get("display", ""))

                    if prop_code == "COMPONENT":
                        code_entry["component"] = prop_value
                    elif prop_code == "PROPERTY":
                        code_entry["property_type"] = prop_value
                    elif prop_code == "TIME_ASPCT":
                        code_entry["timing"] = prop_value
                    elif prop_code == "SYSTEM":
                        code_entry["system_specimen"] = prop_value
                    elif prop_code == "SCALE_TYP":
                        code_entry["scale"] = prop_value
                    elif prop_code == "METHOD_TYP":
                        code_entry["method"] = prop_value
                    elif prop_code == "CLASS":
                        code_entry["class_type"] = prop_value
                    elif prop_code == "STATUS":
                        code_entry["status"] = prop_value
                    elif prop_code == "LONG_COMMON_NAME":
                        code_entry["long_name"] = prop_value

                codes.append(code_entry)

            offset += len(concepts)
            print(f"  Fetched {len(codes):,} codes so far...")

            if max_codes and len(codes) >= max_codes:
                print(f"  Reached max_codes limit ({max_codes})")
                break

            time.sleep(0.5)

        except requests.exceptions.Timeout:
            print(f"  Timeout at offset {offset}, retrying...")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"  Error: {e}")
            break

    return codes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load LOINC (Auto-Detect Source)
# MAGIC
# MAGIC Tries CSV first, then FHIR API, then prints instructions for manual upload.

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, lit

# Try CSV first (preferred for production)
loinc_df = load_loinc_from_csv()

# Fallback: FHIR API
if loinc_df is None:
    try:
        dbutils.widgets.text("loinc_username", "", "LOINC Username")
        dbutils.widgets.text("loinc_password", "", "LOINC Password")
        LOINC_USER = dbutils.widgets.get("loinc_username")
        LOINC_PASS = dbutils.widgets.get("loinc_password")
    except Exception:
        LOINC_USER = ""
        LOINC_PASS = ""

    if LOINC_USER:
        print("  Attempting LOINC FHIR API download...")
        loinc_codes = fetch_loinc_fhir(LOINC_USER, LOINC_PASS)

        if loinc_codes:
            from pyspark.sql.types import StructType, StructField, StringType
            loinc_schema = StructType([
                StructField("loinc_code", StringType()),
                StructField("display", StringType()),
                StructField("component", StringType()),
                StructField("property_type", StringType()),
                StructField("timing", StringType()),
                StructField("system_specimen", StringType()),
                StructField("scale", StringType()),
                StructField("method", StringType()),
                StructField("class_type", StringType()),
                StructField("status", StringType()),
                StructField("long_name", StringType()),
            ])
            loinc_df = spark.createDataFrame(loinc_codes, schema=loinc_schema)

if loinc_df is not None:
    loinc_final = (loinc_df
        .withColumn("source", lit("Regenstrief Institute"))
        .withColumn("version", lit("current"))
        .withColumn("loaded_at", current_timestamp())
    )
    loinc_final.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference.loinc_codes_full")
    total = spark.table(f"{CATALOG}.reference.loinc_codes_full").count()
    print(f"  Loaded {total:,} LOINC codes into {CATALOG}.reference.loinc_codes_full")
else:
    print("  No LOINC source available.")
    print("  To load the full set:")
    print(f"  1. Register free at https://loinc.org/register/")
    print(f"  2. Download the LOINC Table CSV from https://loinc.org/downloads/")
    print(f"  3. Upload to /Volumes/{CATALOG}/reference/loinc_files/")
    print(f"  4. Re-run this notebook")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 3: Create Search-Optimized Tables
# MAGIC
# MAGIC These tables are consumed by the codification pipeline (notebook 05).
# MAGIC - **icd10_search**: Billable codes only, with concatenated search text
# MAGIC - **loinc_search**: Active (non-deprecated) codes with concatenated search text

# COMMAND ----------

# ICD-10 search table (billable codes only)
spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.reference.icd10_search AS
SELECT
    code,
    description,
    category,
    chapter,
    is_billable,
    CONCAT(
        code, ' - ', description, '. ',
        'Category: ', category, '. ',
        'Chapter: ', COALESCE(chapter, 'Unknown'), '.'
    ) AS search_text
FROM {CATALOG}.reference.icd10_codes_full
WHERE is_billable = true
""")

billable_count = spark.table(f"{CATALOG}.reference.icd10_search").count()
print(f"  icd10_search: {billable_count:,} billable codes")

# COMMAND ----------

# LOINC search table (active codes only)
loinc_full_table = f"{CATALOG}.reference.loinc_codes_full"
try:
    spark.table(loinc_full_table).limit(1).collect()

    spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.reference.loinc_search AS
    SELECT
        loinc_code,
        long_name,
        component,
        property_type AS property,
        timing,
        system_specimen,
        scale,
        method,
        class_type,
        status,
        CONCAT(
            loinc_code, ' - ', COALESCE(long_name, ''), '. ',
            'Component: ', COALESCE(component, ''), '. ',
            'Property: ', COALESCE(property_type, ''), '. ',
            'Timing: ', COALESCE(timing, ''), '. ',
            'Specimen: ', COALESCE(system_specimen, ''), '. ',
            'Scale: ', COALESCE(scale, ''), '. ',
            CASE WHEN method IS NOT NULL AND method != '' THEN CONCAT('Method: ', method, '. ') ELSE '' END
        ) AS search_text
    FROM {CATALOG}.reference.loinc_codes_full
    WHERE status != 'DEPRECATED' OR status IS NULL
    """)

    active_count = spark.table(f"{CATALOG}.reference.loinc_search").count()
    print(f"  loinc_search: {active_count:,} active codes")

except Exception:
    print("  Skipping loinc_search -- full LOINC table not loaded yet.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Create ICD-10 Category Index
# MAGIC
# MAGIC This is the key table for the category-based retrieval approach used in
# MAGIC notebook 05 (multi-pass codification). With 74K+ billable codes, we cannot
# MAGIC pass them all to the LLM. Instead:
# MAGIC 1. Round 1, Stage 1 classifies each diagnosis to a 3-character category
# MAGIC 2. Round 1, Stage 2 disambiguates among all codes in that category
# MAGIC
# MAGIC This works because ICD-10 is hierarchically organized -- the first 3 characters
# MAGIC identify the category, and the LLM is very good at category-level classification.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.reference.icd10_category_index AS
SELECT
    SUBSTRING(code, 1, CASE WHEN INSTR(code, '.') > 0 THEN INSTR(code, '.') - 1 ELSE LENGTH(code) END) AS category,
    COLLECT_LIST(CONCAT(code, ' -- ', description)) AS codes_in_category,
    COUNT(*) AS code_count
FROM {CATALOG}.reference.icd10_codes_full
WHERE is_billable = true
GROUP BY 1
HAVING COUNT(*) > 0
ORDER BY category
""")

cat_count = spark.table(f"{CATALOG}.reference.icd10_category_index").count()
print(f"  icd10_category_index: {cat_count:,} categories")

# Show top categories by code count
spark.sql(f"""
    SELECT category, code_count
    FROM {CATALOG}.reference.icd10_category_index
    ORDER BY code_count DESC
    LIMIT 10
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 5: Version Tracking & Update Detection
# MAGIC
# MAGIC Tracks which reference versions are loaded and provides a function
# MAGIC to detect when updates are available. Schedule this notebook quarterly
# MAGIC as a Databricks Workflow for automated update checks.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.reference.code_set_versions (
    code_set STRING,
    version STRING,
    source_url STRING,
    record_count LONG,
    loaded_at TIMESTAMP,
    checked_at TIMESTAMP,
    is_current BOOLEAN
)
USING DELTA
COMMENT 'Tracks loaded reference code set versions for update detection'
""")

# Record current versions
from pyspark.sql.functions import lit, current_timestamp

icd10_count = spark.table(f"{CATALOG}.reference.icd10_codes_full").count()

version_data = [{
    "code_set": "ICD-10-CM",
    "version": "FY2026",
    "source_url": "https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2026/",
    "record_count": icd10_count,
    "is_current": True,
}]

try:
    loinc_count = spark.table(f"{CATALOG}.reference.loinc_codes_full").count()
    version_data.append({
        "code_set": "LOINC",
        "version": "current",
        "source_url": "https://loinc.org/downloads/",
        "record_count": loinc_count,
        "is_current": True,
    })
except Exception:
    pass

versions_df = spark.createDataFrame(version_data)
versions_df = (versions_df
    .withColumn("loaded_at", current_timestamp())
    .withColumn("checked_at", current_timestamp())
)
versions_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference.code_set_versions")
print("  Version tracking table updated")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Update Check Function
# MAGIC
# MAGIC Call `check_for_updates()` periodically (e.g., monthly via Databricks Workflow)
# MAGIC to detect new versions.

# COMMAND ----------

def check_for_updates():
    """Check if new versions of ICD-10-CM or LOINC are available."""
    import requests
    from datetime import datetime

    updates = []

    # Check ICD-10-CM: CDC publishes a new FY each October, addendum each April
    current_month = datetime.now().month
    current_year = datetime.now().year

    if current_month >= 10:
        next_fy = current_year + 1
    else:
        next_fy = current_year

    fy_url = f"https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/{next_fy}/"
    try:
        resp = requests.head(fy_url, timeout=10, allow_redirects=True)
        if resp.status_code == 200:
            updates.append(f"ICD-10-CM FY{next_fy} may be available at {fy_url}")
    except Exception:
        pass

    # Check LOINC version via FHIR
    try:
        resp = requests.get(
            "https://fhir.loinc.org/CodeSystem/loinc?_summary=true",
            headers={"Accept": "application/fhir+json"},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            latest_version = data.get("version", "unknown")
            updates.append(f"Latest LOINC version on FHIR server: {latest_version}")
    except Exception:
        pass

    if updates:
        print("=== Update Check Results ===")
        for u in updates:
            print(f"  {u}")
    else:
        print("  No updates detected")

    # Update checked_at timestamp
    spark.sql(f"""
        UPDATE {CATALOG}.reference.code_set_versions
        SET checked_at = current_timestamp()
        WHERE is_current = true
    """)

    return updates

updates = check_for_updates()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What was loaded
# MAGIC - **ICD-10-CM:** Full code set from CDC/CMS (~98K codes, ~74K billable)
# MAGIC - **LOINC:** Full code set from CSV download or FHIR API (~100K codes)
# MAGIC
# MAGIC ### For LOINC production deployment
# MAGIC 1. Register free at https://loinc.org/register/
# MAGIC 2. Download the LOINC Table CSV from https://loinc.org/downloads/
# MAGIC 3. Upload to `/Volumes/{CATALOG}/reference/loinc_files/`
# MAGIC 4. Re-run this notebook -- it auto-detects the CSV
# MAGIC
# MAGIC ### For automated updates
# MAGIC Schedule this notebook as a quarterly Databricks Workflow.
# MAGIC It will download the latest ICD-10-CM and check LOINC versions automatically.
# MAGIC
# MAGIC **Next:** Run `03_generate_sample_data` (demo only) or skip to `04_extract_entities`.
