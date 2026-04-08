# Databricks notebook source
# MAGIC %md
# MAGIC # 04c — Consolidated LLM Entity Extraction & Assertion Classification
# MAGIC
# MAGIC Replaces three separate notebooks (04c_ner_model_extraction, 04d_llm_extraction,
# MAGIC 04f_assertion_classification) with a single notebook that makes **ONE ai_query call
# MAGIC per chart** to perform all three tasks simultaneously:
# MAGIC
# MAGIC 1. **NER-style entity extraction** (explicit entities)
# MAGIC 2. **Implicit/abbreviation entity detection** (LLM-only entities)
# MAGIC 3. **Assertion classification** (negation, temporality, experiencer, certainty)
# MAGIC
# MAGIC **Cost reduction:** ~1,050 ai_query calls per 100 charts down to ~100 (10x reduction).
# MAGIC
# MAGIC **Output tables (compatible with downstream 04e_ensemble_merge):**
# MAGIC - `extracted.ner_entities` -- explicit entities (model_name = 'consolidated_llm_extraction')
# MAGIC - `extracted.llm_entities` -- implicit/abbreviation entities
# MAGIC - `extracted.entity_assertions` -- assertion classifications for ALL entities
# MAGIC
# MAGIC **Input:** `raw.charts`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("CATALOG", "mv_catalog", "Unity Catalog Name")
CATALOG = dbutils.widgets.get("CATALOG")

MODEL = "databricks-claude-sonnet-4-6"

spark.sql(f"USE CATALOG {CATALOG}")

print(f"  Catalog: {CATALOG}")
print(f"  Model: {MODEL}")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType,
    BooleanType, ArrayType
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Load Charts (Incremental Filter)
# MAGIC
# MAGIC Skip charts that have already been processed by checking `extracted.llm_entities`.

# COMMAND ----------

# Ensure all three output tables exist
for ddl in [
    f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.ner_entities (
        entity_id STRING, chart_id STRING, section_id STRING,
        entity_type STRING, entity_text STRING, ner_label STRING,
        model_name STRING, model_confidence DOUBLE,
        start_offset INT, end_offset INT, extracted_at TIMESTAMP
    ) USING DELTA
    COMMENT 'Entities detected via biomedical NER models (Layer 3)'
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.llm_entities (
        entity_id STRING, chart_id STRING, section_id STRING,
        entity_type STRING, entity_text STRING, extraction_role STRING,
        confidence DOUBLE, reasoning STRING,
        specimen_type STRING, method STRING, timing STRING,
        value STRING, unit STRING,
        start_offset INT, end_offset INT, extracted_at TIMESTAMP
    ) USING DELTA
    COMMENT 'Entities detected via scoped LLM extraction (implicit entities, completeness validation)'
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.entity_assertions (
        assertion_id STRING, entity_id STRING, chart_id STRING,
        assertion_status STRING, negation_detected BOOLEAN,
        negation_trigger STRING, temporality STRING,
        experiencer STRING, certainty STRING,
        classification_method STRING, confidence DOUBLE,
        classified_at TIMESTAMP
    ) USING DELTA
    COMMENT 'Assertion classifications: negation, temporality, experiencer attribution'
    """,
]:
    spark.sql(ddl)

print("  Output tables ready: ner_entities, llm_entities, entity_assertions")

# COMMAND ----------

# Load all charts
all_charts_df = spark.table(f"{CATALOG}.raw.charts")
total_chart_count = all_charts_df.count()

if total_chart_count == 0:
    print("  raw.charts is empty -- nothing to process")
    dbutils.notebook.exit("SKIP: no charts to process")

print(f"  Total charts in raw.charts: {total_chart_count:,}")

# COMMAND ----------

# Incremental filter: skip chart_ids already in llm_entities
existing_chart_ids = (
    spark.table(f"{CATALOG}.extracted.llm_entities")
    .select("chart_id")
    .distinct()
)
existing_count = existing_chart_ids.count()

if existing_count > 0:
    charts_df = all_charts_df.join(existing_chart_ids, "chart_id", "left_anti")
    new_chart_count = charts_df.count()
    print(f"  Skipping {existing_count:,} charts already processed")
    print(f"  New charts to process: {new_chart_count:,}")
    if new_chart_count == 0:
        print("  All charts already processed -- nothing new to extract")
        dbutils.notebook.exit("SKIP: all charts already processed")
else:
    charts_df = all_charts_df
    new_chart_count = total_chart_count
    print(f"  No existing LLM entities found -- processing all {new_chart_count:,} charts")

# COMMAND ----------

# Save pending chart IDs as temp table for SQL query
charts_df.select("chart_id").write.mode("overwrite").saveAsTable(
    f"{CATALOG}.extracted._tmp_consolidated_pending_charts"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Run Consolidated ai_query (ONE Call Per Chart)
# MAGIC
# MAGIC A single prompt per chart extracts ALL entities AND classifies each one for
# MAGIC assertion status, negation, temporality, experiencer, and certainty.

# COMMAND ----------

consolidated_sql = f"""
SELECT
    c.chart_id,
    ai_query(
        '{MODEL}',
        CONCAT(
            'You are a clinical NLP system performing comprehensive entity extraction and classification. ',
            'Analyze this clinical note and return a JSON array of ALL clinical entities found. ',
            'For EACH entity, provide:\\n',
            '- "entity_type": DIAGNOSIS, LAB_RESULT, MEDICATION, or VITAL_SIGN\\n',
            '- "entity_text": the exact text from the note\\n',
            '- "section_type": which section it appeared in (HPI, ASSESSMENT_PLAN, MEDICATIONS, LABS, VITALS, etc.)\\n',
            '- "assertion_status": PRESENT, ABSENT, POSSIBLE, CONDITIONAL, HISTORICAL, or FAMILY\\n',
            '- "negation_detected": true/false\\n',
            '- "negation_trigger": the trigger phrase if negated (e.g., "denies", "no evidence of")\\n',
            '- "temporality": CURRENT, HISTORICAL, or FUTURE_PLANNED\\n',
            '- "experiencer": PATIENT, FAMILY_MEMBER, or OTHER\\n',
            '- "certainty": DEFINITE, PROBABLE, POSSIBLE, or UNLIKELY\\n',
            '- "extraction_role": explicit, implicit, or abbreviation\\n',
            '- "confidence": 0.0-1.0\\n',
            '- "value": numeric value if applicable (labs/vitals)\\n',
            '- "unit": measurement unit if applicable\\n',
            '- "specimen_type": for labs (blood, serum, urine, etc.)\\n',
            '- "method": for labs (immunoassay, HPLC, calculated, etc.)\\n',
            '- "timing": for labs (fasting, random, point in time, etc.)\\n\\n',
            'CRITICAL INSTRUCTIONS:\\n',
            '- Extract EVERY diagnosis, lab result, medication, and vital sign. Do NOT stop early.\\n',
            '- Include both explicitly stated AND implied entities.\\n',
            '- Mark negated findings as ABSENT (e.g., "denies chest pain" = ABSENT).\\n',
            '- Mark family history as FAMILY (e.g., "mother had diabetes" = FAMILY).\\n',
            '- Mark historical conditions as HISTORICAL (e.g., "history of MI").\\n',
            '- For abbreviations (HTN, DM, CHF, etc.), expand and extract.\\n',
            '- Return ONLY the JSON array, no other text.\\n\\n',
            'CLINICAL NOTE:\\n',
            SUBSTRING(c.raw_text, 1, 30000)
        )
    ) AS extraction_result
FROM {CATALOG}.raw.charts c
INNER JOIN {CATALOG}.extracted._tmp_consolidated_pending_charts p
    ON c.chart_id = p.chart_id
"""

print(f"  Running consolidated extraction: {new_chart_count:,} charts (1 ai_query per chart)...")
print(f"  This may take several minutes...")

raw_results_df = spark.sql(consolidated_sql)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Save Raw Results to Temp Table

# COMMAND ----------

raw_results_df.write.mode("overwrite").saveAsTable(
    f"{CATALOG}.extracted._tmp_consolidated_raw_results"
)
raw_results_df = spark.table(f"{CATALOG}.extracted._tmp_consolidated_raw_results")

raw_count = raw_results_df.count()
print(f"  Raw results saved: {raw_count:,} charts processed")

# COMMAND ----------

# Preview raw results
spark.sql(f"""
    SELECT chart_id, SUBSTRING(extraction_result, 1, 500) AS preview
    FROM {CATALOG}.extracted._tmp_consolidated_raw_results
    LIMIT 3
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Parse JSON into Entity Rows
# MAGIC
# MAGIC Parse the LLM JSON response into structured entity rows. Handles markdown
# MAGIC code fences (`\`\`\`json ... \`\`\``) and malformed responses gracefully.

# COMMAND ----------

# Schema for individual entities from the consolidated extraction
entity_json_schema = ArrayType(StructType([
    StructField("entity_type", StringType()),
    StructField("entity_text", StringType()),
    StructField("section_type", StringType()),
    StructField("assertion_status", StringType()),
    StructField("negation_detected", BooleanType()),
    StructField("negation_trigger", StringType()),
    StructField("temporality", StringType()),
    StructField("experiencer", StringType()),
    StructField("certainty", StringType()),
    StructField("extraction_role", StringType()),
    StructField("confidence", DoubleType()),
    StructField("value", StringType()),
    StructField("unit", StringType()),
    StructField("specimen_type", StringType()),
    StructField("method", StringType()),
    StructField("timing", StringType()),
]))

# COMMAND ----------

# Clean markdown fences, parse JSON, and explode into rows
parsed_df = (
    raw_results_df
    .withColumn(
        "clean_json",
        F.regexp_replace(
            F.regexp_replace("extraction_result", "```json\\n?", ""),
            "\\n?```", ""
        )
    )
    .withColumn("entities", F.from_json("clean_json", entity_json_schema))
    .filter(F.col("entities").isNotNull())
    .select("chart_id", F.explode("entities").alias("entity"))
    .select(
        "chart_id",
        F.col("entity.entity_type").alias("entity_type"),
        F.col("entity.entity_text").alias("entity_text"),
        F.col("entity.section_type").alias("section_type"),
        F.col("entity.assertion_status").alias("assertion_status"),
        F.col("entity.negation_detected").alias("negation_detected"),
        F.col("entity.negation_trigger").alias("negation_trigger"),
        F.col("entity.temporality").alias("temporality"),
        F.col("entity.experiencer").alias("experiencer"),
        F.col("entity.certainty").alias("certainty"),
        F.col("entity.extraction_role").alias("extraction_role"),
        F.col("entity.confidence").alias("confidence"),
        F.col("entity.value").alias("value"),
        F.col("entity.unit").alias("unit"),
        F.col("entity.specimen_type").alias("specimen_type"),
        F.col("entity.method").alias("method"),
        F.col("entity.timing").alias("timing"),
    )
)

# Filter out empty or near-empty entity texts
parsed_df = parsed_df.filter(
    F.col("entity_text").isNotNull() & (F.length(F.trim(F.col("entity_text"))) >= 2)
)

# Cache for reuse across the three output writes
parsed_df.write.mode("overwrite").saveAsTable(
    f"{CATALOG}.extracted._tmp_consolidated_parsed_entities"
)
parsed_df = spark.table(f"{CATALOG}.extracted._tmp_consolidated_parsed_entities")

total_entities = parsed_df.count()
print(f"  Total parsed entities: {total_entities:,}")

# COMMAND ----------

# Breakdown by extraction_role
print("  Entities by extraction_role:")
parsed_df.groupBy("extraction_role").count().orderBy(F.desc("count")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5: Split by extraction_role and Write to Output Tables
# MAGIC
# MAGIC - **explicit** entities go to `extracted.ner_entities` (compatible with ensemble merge)
# MAGIC - **implicit** and **abbreviation** entities go to `extracted.llm_entities`

# COMMAND ----------

# --- 5a: Explicit entities → extracted.ner_entities ---

# NER label mapping: entity_type doubles as the ner_label since the LLM returns
# standardized types, not model-specific labels
explicit_df = (
    parsed_df
    .filter(F.col("extraction_role") == "explicit")
    .select(
        F.concat(F.lit("ENT-NER-"), F.expr("uuid()")).alias("entity_id"),
        "chart_id",
        F.lit(None).cast("string").alias("section_id"),
        "entity_type",
        "entity_text",
        F.col("entity_type").alias("ner_label"),
        F.lit("consolidated_llm_extraction").alias("model_name"),
        F.col("confidence").cast(DoubleType()).alias("model_confidence"),
        F.lit(None).cast("int").alias("start_offset"),
        F.lit(None).cast("int").alias("end_offset"),
        F.current_timestamp().alias("extracted_at"),
    )
)

explicit_count = explicit_df.count()
print(f"  Explicit entities for ner_entities: {explicit_count:,}")

if explicit_count > 0:
    explicit_df.write.mode("append").saveAsTable(f"{CATALOG}.extracted.ner_entities")
    print(f"  Wrote {explicit_count:,} entities to extracted.ner_entities")
else:
    print("  No explicit entities to write")

# COMMAND ----------

# --- 5b: Implicit + abbreviation entities → extracted.llm_entities ---

implicit_df = (
    parsed_df
    .filter(F.col("extraction_role").isin("implicit", "abbreviation"))
    .select(
        F.concat(F.lit("ENT-LLM-"), F.expr("uuid()")).alias("entity_id"),
        "chart_id",
        F.lit(None).cast("string").alias("section_id"),
        "entity_type",
        "entity_text",
        F.col("extraction_role").alias("extraction_role"),
        F.col("confidence").cast(DoubleType()),
        F.lit(None).cast("string").alias("reasoning"),
        "specimen_type",
        "method",
        "timing",
        "value",
        "unit",
        F.lit(None).cast("int").alias("start_offset"),
        F.lit(None).cast("int").alias("end_offset"),
        F.current_timestamp().alias("extracted_at"),
    )
)

implicit_count = implicit_df.count()
print(f"  Implicit/abbreviation entities for llm_entities: {implicit_count:,}")

if implicit_count > 0:
    implicit_df.write.mode("append").saveAsTable(f"{CATALOG}.extracted.llm_entities")
    print(f"  Wrote {implicit_count:,} entities to extracted.llm_entities")
else:
    print("  No implicit/abbreviation entities to write")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6: Extract Assertion Classifications → extracted.entity_assertions
# MAGIC
# MAGIC Write assertion classifications for ALL extracted entities (both explicit and
# MAGIC implicit/abbreviation). Each entity gets a corresponding assertion row so that
# MAGIC downstream notebooks can filter by assertion status.

# COMMAND ----------

# We need entity_ids that were actually written. Re-read from the tables to get
# the entities we just inserted (matched by chart_id + entity_text from this batch).

# For ner_entities: join back on chart_id + entity_text to get entity_ids
# For llm_entities: same approach

# Simpler approach: write assertions from the parsed_df with fresh UUIDs for both
# entity_id and assertion_id, keyed to chart_id + entity_text.

# Build assertion rows for ALL parsed entities
assertions_df = (
    parsed_df
    .select(
        F.concat(F.lit("ASSERT-"), F.expr("uuid()")).alias("assertion_id"),
        # Generate an entity_id placeholder -- downstream merge will link via chart_id + entity_text
        F.concat(
            F.when(F.col("extraction_role") == "explicit", F.lit("ENT-NER-"))
             .otherwise(F.lit("ENT-LLM-")),
            F.expr("uuid()")
        ).alias("entity_id"),
        "chart_id",
        F.coalesce(F.col("assertion_status"), F.lit("PRESENT")).alias("assertion_status"),
        F.coalesce(F.col("negation_detected"), F.lit(False)).alias("negation_detected"),
        "negation_trigger",
        F.coalesce(F.col("temporality"), F.lit("CURRENT")).alias("temporality"),
        F.coalesce(F.col("experiencer"), F.lit("PATIENT")).alias("experiencer"),
        F.coalesce(F.col("certainty"), F.lit("DEFINITE")).alias("certainty"),
        F.lit("consolidated_llm").alias("classification_method"),
        F.coalesce(F.col("confidence"), F.lit(0.85)).cast(DoubleType()).alias("confidence"),
        F.current_timestamp().alias("classified_at"),
    )
)

# Better approach: use the entity_ids from the tables we just wrote to.
# Since we used uuid() during the write, we cannot recover those exact IDs.
# Instead, query the recently inserted rows by chart_id membership.

# Strategy: query ner_entities and llm_entities for charts in this batch,
# then write assertions keyed to those actual entity_ids.

batch_chart_ids = f"{CATALOG}.extracted._tmp_consolidated_pending_charts"

assertions_from_ner = spark.sql(f"""
    SELECT
        CONCAT('ASSERT-', uuid()) AS assertion_id,
        ne.entity_id,
        ne.chart_id
    FROM {CATALOG}.extracted.ner_entities ne
    INNER JOIN {batch_chart_ids} p ON ne.chart_id = p.chart_id
    WHERE ne.model_name = 'consolidated_llm_extraction'
      AND ne.entity_id NOT IN (
          SELECT entity_id FROM {CATALOG}.extracted.entity_assertions
      )
""")

assertions_from_llm = spark.sql(f"""
    SELECT
        CONCAT('ASSERT-', uuid()) AS assertion_id,
        le.entity_id,
        le.chart_id
    FROM {CATALOG}.extracted.llm_entities le
    INNER JOIN {batch_chart_ids} p ON le.chart_id = p.chart_id
    WHERE le.entity_id NOT IN (
          SELECT entity_id FROM {CATALOG}.extracted.entity_assertions
      )
""")

# Union the entity_ids we need assertions for
entities_needing_assertions = assertions_from_ner.union(assertions_from_llm)

# Join back to parsed_df to get assertion fields
# We need to match on chart_id + entity_text since entity_ids differ
entities_needing_assertions.createOrReplaceTempView("_entities_needing_assertions")

# COMMAND ----------

# Since entity_ids from the written tables don't match parsed_df (both used uuid()),
# we take a pragmatic approach: write assertions for the entities we can match
# from the written tables, using the parsed_df assertion fields matched by
# chart_id + entity_text.

parsed_df.createOrReplaceTempView("_parsed_entities")

# Get ner_entity assertions
ner_assertions_sql = f"""
SELECT
    CONCAT('ASSERT-', uuid()) AS assertion_id,
    ne.entity_id,
    ne.chart_id,
    COALESCE(p.assertion_status, 'PRESENT') AS assertion_status,
    COALESCE(p.negation_detected, FALSE) AS negation_detected,
    p.negation_trigger,
    COALESCE(p.temporality, 'CURRENT') AS temporality,
    COALESCE(p.experiencer, 'PATIENT') AS experiencer,
    COALESCE(p.certainty, 'DEFINITE') AS certainty,
    'consolidated_llm' AS classification_method,
    COALESCE(p.confidence, 0.85) AS confidence,
    CURRENT_TIMESTAMP() AS classified_at
FROM {CATALOG}.extracted.ner_entities ne
INNER JOIN {batch_chart_ids} batch ON ne.chart_id = batch.chart_id
LEFT JOIN _parsed_entities p
    ON ne.chart_id = p.chart_id
    AND LOWER(TRIM(ne.entity_text)) = LOWER(TRIM(p.entity_text))
    AND p.extraction_role = 'explicit'
WHERE ne.model_name = 'consolidated_llm_extraction'
  AND ne.entity_id NOT IN (SELECT entity_id FROM {CATALOG}.extracted.entity_assertions)
"""

# Get llm_entity assertions
llm_assertions_sql = f"""
SELECT
    CONCAT('ASSERT-', uuid()) AS assertion_id,
    le.entity_id,
    le.chart_id,
    COALESCE(p.assertion_status, 'PRESENT') AS assertion_status,
    COALESCE(p.negation_detected, FALSE) AS negation_detected,
    p.negation_trigger,
    COALESCE(p.temporality, 'CURRENT') AS temporality,
    COALESCE(p.experiencer, 'PATIENT') AS experiencer,
    COALESCE(p.certainty, 'DEFINITE') AS certainty,
    'consolidated_llm' AS classification_method,
    COALESCE(p.confidence, 0.85) AS confidence,
    CURRENT_TIMESTAMP() AS classified_at
FROM {CATALOG}.extracted.llm_entities le
INNER JOIN {batch_chart_ids} batch ON le.chart_id = batch.chart_id
LEFT JOIN _parsed_entities p
    ON le.chart_id = p.chart_id
    AND LOWER(TRIM(le.entity_text)) = LOWER(TRIM(p.entity_text))
    AND p.extraction_role IN ('implicit', 'abbreviation')
WHERE le.entity_id NOT IN (SELECT entity_id FROM {CATALOG}.extracted.entity_assertions)
"""

ner_assert_df = spark.sql(ner_assertions_sql)
llm_assert_df = spark.sql(llm_assertions_sql)
all_assertions_df = ner_assert_df.union(llm_assert_df)

assertion_count = all_assertions_df.count()
print(f"  Assertion classifications to write: {assertion_count:,}")

if assertion_count > 0:
    all_assertions_df.write.mode("append").saveAsTable(f"{CATALOG}.extracted.entity_assertions")
    print(f"  Wrote {assertion_count:,} assertions to extracted.entity_assertions")
else:
    print("  No new assertions to write")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7: Cleanup Temp Tables

# COMMAND ----------

for tmp_table in [
    f"{CATALOG}.extracted._tmp_consolidated_pending_charts",
    f"{CATALOG}.extracted._tmp_consolidated_raw_results",
    f"{CATALOG}.extracted._tmp_consolidated_parsed_entities",
]:
    spark.sql(f"DROP TABLE IF EXISTS {tmp_table}")
print("  Temp tables cleaned up")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8: Summary Statistics

# COMMAND ----------

# Overall extraction summary
print("  === Consolidated Extraction Summary ===")
print(f"  Charts processed this run: {new_chart_count:,}")
print(f"  Total entities extracted: {total_entities:,}")
print(f"    Explicit (-> ner_entities): {explicit_count:,}")
print(f"    Implicit/abbreviation (-> llm_entities): {implicit_count:,}")
print(f"  Assertion classifications written: {assertion_count:,}")

# COMMAND ----------

# Entity type distribution
print("  Entity type distribution:")
spark.sql(f"""
    SELECT
        entity_type,
        COUNT(*) AS entity_count,
        ROUND(AVG(confidence), 3) AS avg_confidence,
        COUNT(DISTINCT chart_id) AS charts_found_in
    FROM (
        SELECT entity_type, model_confidence AS confidence, chart_id
        FROM {CATALOG}.extracted.ner_entities
        WHERE model_name = 'consolidated_llm_extraction'
        UNION ALL
        SELECT entity_type, confidence, chart_id
        FROM {CATALOG}.extracted.llm_entities
    )
    GROUP BY entity_type
    ORDER BY entity_count DESC
""").show(20, truncate=False)

# COMMAND ----------

# Assertion status distribution
print("  Assertion status distribution:")
spark.sql(f"""
    SELECT
        assertion_status,
        COUNT(*) AS entity_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM {CATALOG}.extracted.entity_assertions
    WHERE classification_method = 'consolidated_llm'
    GROUP BY assertion_status
    ORDER BY entity_count DESC
""").show(truncate=False)

# COMMAND ----------

# Negation summary
print("  Negation summary:")
spark.sql(f"""
    SELECT
        negation_detected,
        COUNT(*) AS entity_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM {CATALOG}.extracted.entity_assertions
    WHERE classification_method = 'consolidated_llm'
    GROUP BY negation_detected
    ORDER BY entity_count DESC
""").show(truncate=False)

# COMMAND ----------

# Table totals
print("  Table totals:")
for tbl, desc in [
    (f"{CATALOG}.extracted.ner_entities", "NER entities"),
    (f"{CATALOG}.extracted.llm_entities", "LLM entities"),
    (f"{CATALOG}.extracted.entity_assertions", "Entity assertions"),
]:
    cnt = spark.table(tbl).count()
    print(f"    {desc}: {cnt:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary
# MAGIC
# MAGIC ### What was done
# MAGIC - Sent ONE consolidated `ai_query` call per chart (instead of ~10 calls across 3 notebooks)
# MAGIC - Extracted explicit, implicit, and abbreviation entities in a single pass
# MAGIC - Classified assertion status, negation, temporality, experiencer, and certainty inline
# MAGIC - Split results into `ner_entities` and `llm_entities` by extraction_role for ensemble compatibility
# MAGIC - Wrote assertion classifications to `entity_assertions` for all extracted entities
# MAGIC
# MAGIC ### Cost comparison
# MAGIC | Before (3 notebooks) | After (consolidated) |
# MAGIC |----------------------|---------------------|
# MAGIC | ~500 ai_query (NER per section) | 0 |
# MAGIC | ~100 ai_query (implicit per chart) | 0 |
# MAGIC | ~450 ai_query (assertions per entity) | 0 |
# MAGIC | **~1,050 total** | **~100 total (1 per chart)** |
# MAGIC
# MAGIC ### Tables written
# MAGIC | Table | Content | Key field |
# MAGIC |-------|---------|-----------|
# MAGIC | `extracted.ner_entities` | Explicit entities | `model_name = 'consolidated_llm_extraction'` |
# MAGIC | `extracted.llm_entities` | Implicit + abbreviation entities | `extraction_role IN ('implicit', 'abbreviation')` |
# MAGIC | `extracted.entity_assertions` | All assertion classifications | `classification_method = 'consolidated_llm'` |
# MAGIC
# MAGIC **Next:** Run `04e_ensemble_merge` to fuse entities from all extraction layers.
