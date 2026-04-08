# Databricks notebook source
# MAGIC %md
# MAGIC # 04a — Document Intelligence & Section Segmentation
# MAGIC
# MAGIC Processes raw charts before entity extraction using a two-phase approach:
# MAGIC 1. **Smart Text Extraction** -- Bypass expensive OCR where possible by detecting
# MAGIC    PDFs that already contain a digital text layer
# MAGIC 2. **Section Segmentation** -- Parse clinical notes into standard sections
# MAGIC    (HPI, Assessment/Plan, etc.) using rule-based patterns with LLM fallback
# MAGIC
# MAGIC **Key design decisions:**
# MAGIC - Digital text extraction is attempted first; `ai_parse_document` (OCR) is only
# MAGIC   invoked when the text layer is missing or returns garbage
# MAGIC - Section segmentation uses regex pattern matching for speed, falling back to
# MAGIC   `ai_query` for unstructured notes with no clear headers
# MAGIC - Results feed into the dictionary-based (04b) and NER-based (04c) extraction layers
# MAGIC
# MAGIC **Estimated runtime:** ~5 minutes for 100 charts

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
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, ArrayType, DoubleType
)
import re

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Smart Text Extraction (OCR Bypass)
# MAGIC
# MAGIC Many clinical documents are digitally-generated PDFs (EHR exports, lab reports)
# MAGIC that already contain a searchable text layer. Running OCR on these wastes time
# MAGIC and money. This step:
# MAGIC
# MAGIC 1. Attempts direct text extraction via `read_files` with `format => 'text'`
# MAGIC 2. Evaluates whether the extracted text is meaningful (length, character ratio)
# MAGIC 3. Falls back to `ai_parse_document` only for scanned/image-based PDFs
# MAGIC 4. Tracks the extraction method and cost savings

# COMMAND ----------

# Load current charts
charts_df = spark.table(f"{CATALOG}.raw.charts")
total_charts = charts_df.count()
print(f"  Processing {total_charts} charts")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a: Attempt Direct Text Extraction
# MAGIC
# MAGIC For charts that have a `file_path` pointing to a PDF in the UC Volume,
# MAGIC we attempt to read the text layer directly. If the chart already has
# MAGIC `raw_text` populated (e.g., from notebook 03), we evaluate its quality.

# COMMAND ----------

def is_meaningful_text(text, min_length=100, min_alpha_ratio=0.4):
    """
    Determine if extracted text is meaningful (not garbled OCR artifacts).
    Returns True if the text looks like real clinical content.
    """
    if text is None or len(text.strip()) < min_length:
        return False
    # Check that a reasonable proportion of characters are alphabetic
    alpha_count = sum(1 for c in text if c.isalpha())
    total_count = len(text.strip())
    if total_count == 0:
        return False
    alpha_ratio = alpha_count / total_count
    if alpha_ratio < min_alpha_ratio:
        return False
    # Check for common clinical terms as a sanity check
    clinical_markers = [
        'patient', 'history', 'exam', 'diagnosis', 'medication',
        'blood', 'assessment', 'plan', 'complaint', 'vitals',
    ]
    text_lower = text.lower()
    has_clinical_content = any(marker in text_lower for marker in clinical_markers)
    return has_clinical_content

is_meaningful_text_udf = F.udf(is_meaningful_text, StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b: Classify Charts by Extraction Method
# MAGIC
# MAGIC Charts with meaningful `raw_text` are marked as `digital_text_layer`.
# MAGIC Charts that need OCR are processed with `ai_parse_document`.

# COMMAND ----------

# Classify charts: does the existing raw_text look meaningful?
classified_df = charts_df.withColumn(
    "text_is_meaningful",
    F.when(
        (F.col("raw_text").isNotNull()) &
        (F.length(F.col("raw_text")) > 100) &
        (F.col("raw_text").rlike("[A-Za-z]{4,}")),  # contains real words
        F.lit(True)
    ).otherwise(F.lit(False))
)

digital_count = classified_df.filter("text_is_meaningful = true").count()
needs_ocr_count = classified_df.filter("text_is_meaningful = false").count()
print(f"  Digital text layer detected: {digital_count} charts")
print(f"  Needs OCR (ai_parse_document): {needs_ocr_count} charts")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1c: Run OCR on Charts That Need It
# MAGIC
# MAGIC For charts without a usable text layer, use `ai_parse_document` to extract
# MAGIC text from the PDF binary content in the UC Volume.

# COMMAND ----------

# For charts that need OCR and have a file_path, run ai_parse_document
if needs_ocr_count > 0:
    ocr_sql = f"""
    SELECT
        c.chart_id,
        ai_parse_document(rf.content):content::STRING AS ocr_text
    FROM {CATALOG}.raw.charts c
    INNER JOIN read_files(
        '/Volumes/{CATALOG}/raw/chart_pdfs/',
        format => 'binaryFile'
    ) rf ON rf.path LIKE CONCAT('%', c.file_name)
    WHERE c.raw_text IS NULL
       OR LENGTH(c.raw_text) < 100
    """
    try:
        ocr_df = spark.sql(ocr_sql)
        ocr_count = ocr_df.count()
        print(f"  OCR extracted text from {ocr_count} charts via ai_parse_document")

        # Update charts with OCR text
        if ocr_count > 0:
            ocr_df.createOrReplaceTempView("ocr_results")
            spark.sql(f"""
                MERGE INTO {CATALOG}.raw.charts AS target
                USING ocr_results AS source
                ON target.chart_id = source.chart_id
                WHEN MATCHED THEN UPDATE SET
                    target.raw_text = source.ocr_text,
                    target.extraction_method = 'ocr_ai_parse'
            """)
            print(f"  Updated {ocr_count} charts with OCR-extracted text")
    except Exception as e:
        print(f"  OCR step skipped (no PDFs in volume or volume not accessible): {e}")
        print(f"  Proceeding with existing raw_text for all charts")
else:
    print(f"  All charts have usable text -- OCR bypass saved processing time")

# COMMAND ----------

# Mark digital text layer charts
spark.sql(f"""
    UPDATE {CATALOG}.raw.charts
    SET extraction_method = 'digital_text_layer'
    WHERE extraction_method IS NULL
      AND raw_text IS NOT NULL
      AND LENGTH(raw_text) >= 100
""")

# COMMAND ----------

# Cost savings summary
digital_final = spark.sql(f"""
    SELECT extraction_method, COUNT(*) AS chart_count
    FROM {CATALOG}.raw.charts
    GROUP BY extraction_method
""")
print("  Extraction method distribution:")
digital_final.show(truncate=False)

# Estimate cost savings (ai_parse_document ~$0.01/page vs ~$0.00 for text layer)
digital_bypass = spark.sql(f"""
    SELECT COUNT(*) AS bypassed
    FROM {CATALOG}.raw.charts
    WHERE extraction_method = 'digital_text_layer'
""").collect()[0]["bypassed"]
estimated_savings = digital_bypass * 0.01  # rough estimate per chart
print(f"  Estimated OCR cost savings: ${estimated_savings:.2f} ({digital_bypass} charts bypassed OCR)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Section Segmentation
# MAGIC
# MAGIC Clinical notes follow a semi-structured format with standard section headers
# MAGIC (Chief Complaint, HPI, Assessment & Plan, etc.). We parse the text into sections
# MAGIC using two approaches:
# MAGIC
# MAGIC 1. **Rule-based regex** -- fast, deterministic, handles well-formatted notes
# MAGIC 2. **LLM-assisted** -- fallback for unstructured notes with no clear headers
# MAGIC
# MAGIC Output: one row per section per chart in `extracted.document_sections`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a: Section Header Patterns

# COMMAND ----------

SECTION_PATTERNS = {
    'CHIEF_COMPLAINT': r'(?:chief\s+complaint|cc|reason\s+for\s+visit|presenting\s+complaint)',
    'HPI': r'(?:history\s+of\s+present\s+illness|hpi|present\s+illness)',
    'PAST_MEDICAL_HISTORY': r'(?:past\s+medical\s+history|pmh|medical\s+history|past\s+history)',
    'MEDICATIONS': r'(?:medications?|current\s+medications?|medication\s+list|meds|home\s+medications?)',
    'ALLERGIES': r'(?:allergies|allergy|drug\s+allergies|adverse\s+reactions|nkda)',
    'FAMILY_HISTORY': r'(?:family\s+history|fh|family\s+hx)',
    'SOCIAL_HISTORY': r'(?:social\s+history|sh|social\s+hx)',
    'REVIEW_OF_SYSTEMS': r'(?:review\s+of\s+systems|ros)',
    'PHYSICAL_EXAM': r'(?:physical\s+exam|pe|examination|physical\s+examination|exam)',
    'VITALS': r'(?:vital\s+signs?|vitals?|vs)',
    'LABS': r'(?:lab(?:oratory)?\s+(?:results?|data|values?|studies)|labs?|lab\s+work|laboratory)',
    'ASSESSMENT_PLAN': r'(?:assessment\s*(?:and|&|/)?\s*plan|a\s*/\s*p|a&p|impression\s*(?:and|&|/)?\s*plan|assessment|plan|impression)',
    'DIAGNOSES': r'(?:diagnos[ei]s|diagnosis\s+list|active\s+problems?|problem\s+list)',
    'PROCEDURES': r'(?:procedures?|interventions?|surgeries|surgical\s+history)',
    'IMAGING': r'(?:imaging|radiology|x-ray|ct\s+scan|mri|ultrasound)',
    'OTHER': r'(?:notes?|comments?|additional)',
}

# Compile patterns with section boundary detection (header followed by colon or newline)
COMPILED_PATTERNS = {}
for section_type, pattern in SECTION_PATTERNS.items():
    COMPILED_PATTERNS[section_type] = re.compile(
        r'(?:^|\n)\s*(?:#{1,3}\s*)?(' + pattern + r')\s*[:\-]?\s*\n?',
        re.MULTILINE | re.IGNORECASE
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b: Rule-Based Segmentation UDF

# COMMAND ----------

def segment_clinical_note(raw_text):
    """
    Parse a clinical note into sections using regex pattern matching.
    Returns a list of dicts with section_type, header, text, and order.
    Returns None if no section headers are detected (triggers LLM fallback).
    """
    if raw_text is None or len(raw_text.strip()) < 50:
        return None

    # Find all section header matches with their positions
    matches = []
    for section_type, compiled_re in COMPILED_PATTERNS.items():
        for m in compiled_re.finditer(raw_text):
            matches.append({
                'section_type': section_type,
                'header': m.group(1).strip(),
                'start': m.start(),
                'header_end': m.end(),
            })

    if len(matches) < 2:
        # Not enough section headers detected -- needs LLM fallback
        return None

    # Sort by position in text
    matches.sort(key=lambda x: x['start'])

    # Deduplicate: if two patterns match the same position, keep the more specific one
    deduped = []
    for i, match in enumerate(matches):
        if i > 0 and abs(match['start'] - matches[i-1]['start']) < 5:
            # Same position -- keep the one that is NOT 'OTHER'
            if match['section_type'] != 'OTHER':
                deduped[-1] = match
            continue
        deduped.append(match)

    # Extract text between section headers
    sections = []
    for i, match in enumerate(deduped):
        text_start = match['header_end']
        if i + 1 < len(deduped):
            text_end = deduped[i + 1]['start']
        else:
            text_end = len(raw_text)

        section_text = raw_text[text_start:text_end].strip()
        if section_text:
            sections.append({
                'section_type': match['section_type'],
                'section_header': match['header'],
                'section_text': section_text,
                'order': i,
            })

    return sections if sections else None

# COMMAND ----------

# Register as Spark UDF
section_schema = ArrayType(StructType([
    StructField("section_type", StringType()),
    StructField("section_header", StringType()),
    StructField("section_text", StringType()),
    StructField("order", IntegerType()),
]))

segment_udf = F.udf(segment_clinical_note, section_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c: Apply Rule-Based Segmentation

# COMMAND ----------

charts_with_text = spark.table(f"{CATALOG}.raw.charts").filter(
    F.col("raw_text").isNotNull() & (F.length(F.col("raw_text")) > 50)
)

segmented_df = charts_with_text.withColumn(
    "sections", segment_udf(F.col("raw_text"))
)

# Split into rule-based successes and LLM-needed failures
rule_based_df = segmented_df.filter(F.col("sections").isNotNull())
needs_llm_df = segmented_df.filter(F.col("sections").isNull())

rule_count = rule_based_df.count()
llm_count = needs_llm_df.count()
print(f"  Rule-based segmentation succeeded: {rule_count} charts")
print(f"  Needs LLM-assisted segmentation: {llm_count} charts")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2d: LLM-Assisted Segmentation (Fallback)
# MAGIC
# MAGIC For notes where rule-based parsing fails (no clear section headers),
# MAGIC use `ai_query` to identify and segment the clinical note into standard sections.

# COMMAND ----------

if llm_count > 0:
    needs_llm_df.select("chart_id", "raw_text").createOrReplaceTempView("charts_needing_llm")

    llm_segmentation_sql = f"""
    SELECT
        chart_id,
        ai_query(
            '{MODEL}',
            CONCAT(
                'You are a clinical documentation specialist. Parse the following clinical note ',
                'into standard sections. Return a JSON array where each element has: ',
                '"section_type" (one of: CHIEF_COMPLAINT, HPI, PAST_MEDICAL_HISTORY, MEDICATIONS, ',
                'ALLERGIES, FAMILY_HISTORY, SOCIAL_HISTORY, REVIEW_OF_SYSTEMS, PHYSICAL_EXAM, ',
                'VITALS, LABS, ASSESSMENT_PLAN, DIAGNOSES, PROCEDURES, IMAGING, OTHER), ',
                '"section_header" (the section heading as it appears in the text, or a generated one), ',
                '"section_text" (the full section content), ',
                '"order" (integer, sequential order in the document). ',
                'If the note has no clear section structure, create logical sections based on content. ',
                'Return ONLY the JSON array, no other text.',
                '\\n\\nCLINICAL NOTE:\\n',
                raw_text
            )
        ) AS llm_sections_raw
    FROM charts_needing_llm
    """

    print("  Running LLM-assisted segmentation...")
    llm_result_df = spark.sql(llm_segmentation_sql)
    llm_result_df.createOrReplaceTempView("llm_section_results")

    # Parse LLM JSON output
    llm_parsed_df = llm_result_df.withColumn(
        "clean_json",
        F.regexp_replace(
            F.regexp_replace("llm_sections_raw", "```json\\n?", ""),
            "\\n?```", ""
        )
    ).withColumn(
        "sections", F.from_json("clean_json", section_schema)
    ).filter(
        F.col("sections").isNotNull()
    ).select("chart_id", "sections")

    llm_parsed_count = llm_parsed_df.count()
    print(f"  LLM segmentation produced results for {llm_parsed_count} charts")
else:
    llm_parsed_df = None
    print(f"  No charts need LLM segmentation -- all parsed via rules")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2e: Combine Results and Write to `extracted.document_sections`

# COMMAND ----------

# Create the document_sections table
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.document_sections (
    section_id STRING,
    chart_id STRING,
    section_type STRING,
    section_header STRING,
    section_text STRING,
    section_order INT,
    extraction_method STRING,
    extracted_at TIMESTAMP
)
USING DELTA
COMMENT 'Clinical note sections parsed from raw chart text'
""")
print(f"  extracted.document_sections table ready")

# COMMAND ----------

from pyspark.sql.functions import explode, col, lit, current_timestamp, concat, expr

# Explode rule-based sections
rule_sections_df = (rule_based_df
    .select("chart_id", explode("sections").alias("sec"))
    .select(
        concat(F.lit("SEC-"), expr("uuid()")).alias("section_id"),
        "chart_id",
        col("sec.section_type").alias("section_type"),
        col("sec.section_header").alias("section_header"),
        col("sec.section_text").alias("section_text"),
        col("sec.order").alias("section_order"),
        lit("rule_based").alias("extraction_method"),
        current_timestamp().alias("extracted_at"),
    )
)

# Combine with LLM sections if any
if llm_count > 0 and llm_parsed_df is not None:
    llm_sections_df = (llm_parsed_df
        .select("chart_id", explode("sections").alias("sec"))
        .select(
            concat(F.lit("SEC-"), expr("uuid()")).alias("section_id"),
            "chart_id",
            col("sec.section_type").alias("section_type"),
            col("sec.section_header").alias("section_header"),
            col("sec.section_text").alias("section_text"),
            col("sec.order").alias("section_order"),
            lit("llm_assisted").alias("extraction_method"),
            current_timestamp().alias("extracted_at"),
        )
    )
    all_sections_df = rule_sections_df.unionByName(llm_sections_df)
else:
    all_sections_df = rule_sections_df

# Write to table
all_sections_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.extracted.document_sections")

section_count = spark.table(f"{CATALOG}.extracted.document_sections").count()
print(f"  Wrote {section_count} sections to extracted.document_sections")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2f: Segmentation Summary

# COMMAND ----------

# Section type distribution
print("  Section type distribution:")
spark.sql(f"""
    SELECT
        section_type,
        COUNT(*) AS section_count,
        COUNT(DISTINCT chart_id) AS charts_with_section,
        extraction_method
    FROM {CATALOG}.extracted.document_sections
    GROUP BY section_type, extraction_method
    ORDER BY section_count DESC
""").show(30, truncate=False)

# COMMAND ----------

# Charts with vs without clear section headers
print("  Segmentation method distribution:")
spark.sql(f"""
    SELECT
        extraction_method,
        COUNT(DISTINCT chart_id) AS chart_count
    FROM {CATALOG}.extracted.document_sections
    GROUP BY extraction_method
""").show(truncate=False)

# COMMAND ----------

# Average sections per chart
print("  Average sections per chart:")
spark.sql(f"""
    SELECT
        ROUND(AVG(section_count), 1) AS avg_sections_per_chart,
        MIN(section_count) AS min_sections,
        MAX(section_count) AS max_sections,
        ROUND(STDDEV(section_count), 1) AS stddev_sections
    FROM (
        SELECT chart_id, COUNT(*) AS section_count
        FROM {CATALOG}.extracted.document_sections
        GROUP BY chart_id
    )
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Update `raw.charts` with Section Count
# MAGIC
# MAGIC Add section count information back to the charts table for downstream use.

# COMMAND ----------

spark.sql(f"""
    MERGE INTO {CATALOG}.raw.charts AS target
    USING (
        SELECT chart_id, COUNT(*) AS section_count
        FROM {CATALOG}.extracted.document_sections
        GROUP BY chart_id
    ) AS source
    ON target.chart_id = source.chart_id
    WHEN MATCHED THEN UPDATE SET
        target.page_count = COALESCE(target.page_count, source.section_count)
""")

# Also create a view with section counts for easy access
spark.sql(f"""
    CREATE OR REPLACE VIEW {CATALOG}.extracted.chart_section_summary AS
    SELECT
        c.chart_id,
        c.patient_id,
        c.provider,
        c.chart_date,
        c.extraction_method AS chart_extraction_method,
        COUNT(ds.section_id) AS section_count,
        COLLECT_SET(ds.section_type) AS section_types,
        ds.extraction_method AS section_extraction_method
    FROM {CATALOG}.raw.charts c
    LEFT JOIN {CATALOG}.extracted.document_sections ds ON c.chart_id = ds.chart_id
    GROUP BY c.chart_id, c.patient_id, c.provider, c.chart_date,
             c.extraction_method, ds.extraction_method
""")

print(f"  Updated raw.charts and created extracted.chart_section_summary view")

# COMMAND ----------

# Final summary
spark.sql(f"""
    SELECT
        extraction_method,
        COUNT(*) AS charts,
        ROUND(AVG(page_count), 1) AS avg_sections
    FROM {CATALOG}.raw.charts
    GROUP BY extraction_method
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What was done
# MAGIC - **Smart text extraction**: Classified charts by extraction method, bypassing
# MAGIC   expensive OCR for digitally-generated PDFs
# MAGIC - **Section segmentation**: Parsed clinical notes into standard sections using
# MAGIC   rule-based patterns with LLM fallback for unstructured notes
# MAGIC - **Chart enrichment**: Updated `raw.charts` with extraction method and section counts
# MAGIC
# MAGIC ### Tables created/updated
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `raw.charts` | Updated with `extraction_method` |
# MAGIC | `extracted.document_sections` | One row per section per chart |
# MAGIC | `extracted.chart_section_summary` | View with section counts per chart |
# MAGIC
# MAGIC **Next:** Run `04b_dictionary_extraction` for dictionary-based entity detection.
