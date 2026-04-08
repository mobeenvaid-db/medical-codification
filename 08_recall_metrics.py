# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Recall & Quality Analytics
# MAGIC
# MAGIC Comprehensive analytics notebook for understanding pipeline quality across
# MAGIC every dimension: extraction layers, note sections, entity density, assertion
# MAGIC classification, codification accuracy, and trending over time.
# MAGIC
# MAGIC **Input:** `extracted.merged_entities`, `extracted.document_sections`, `extracted.entity_assertions`,
# MAGIC `codified.icd10_mappings`, `codified.loinc_mappings`, `feedback.recall_metrics`, `feedback.gold_annotations`
# MAGIC **Output:** Display-only analytics + executive summary print
# MAGIC
# MAGIC **Estimated runtime:** ~2 minutes

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
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 1: Layer Contribution Analysis
# MAGIC
# MAGIC For each extraction layer (dictionary, NER, LLM), calculate unique contribution
# MAGIC rate, overlap with other layers, and per-entity-type breakdown.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a. Overall Layer Contribution

# COMMAND ----------

layer_contribution_sql = f"""
WITH entity_sources AS (
    SELECT
        entity_id,
        entity_type,
        ensemble_confidence,
        ARRAY_CONTAINS(sources, 'dictionary') AS from_dict,
        ARRAY_CONTAINS(sources, 'ner') AS from_ner,
        ARRAY_CONTAINS(sources, 'llm') AS from_llm,
        SIZE(sources) AS source_count
    FROM {CATALOG}.extracted.merged_entities
)
SELECT
    COUNT(*) AS total_entities,

    -- Layer hit counts
    COUNT(CASE WHEN from_dict THEN 1 END) AS dict_found,
    COUNT(CASE WHEN from_ner THEN 1 END) AS ner_found,
    COUNT(CASE WHEN from_llm THEN 1 END) AS llm_found,

    -- Unique contributions (found ONLY by this layer)
    COUNT(CASE WHEN from_dict AND NOT from_ner AND NOT from_llm THEN 1 END) AS dict_unique,
    COUNT(CASE WHEN from_ner AND NOT from_dict AND NOT from_llm THEN 1 END) AS ner_unique,
    COUNT(CASE WHEN from_llm AND NOT from_dict AND NOT from_ner THEN 1 END) AS llm_unique,

    -- Multi-layer overlap
    COUNT(CASE WHEN from_dict AND from_ner AND from_llm THEN 1 END) AS all_three,
    COUNT(CASE WHEN from_dict AND from_ner AND NOT from_llm THEN 1 END) AS dict_and_ner_only,
    COUNT(CASE WHEN from_dict AND from_llm AND NOT from_ner THEN 1 END) AS dict_and_llm_only,
    COUNT(CASE WHEN from_ner AND from_llm AND NOT from_dict THEN 1 END) AS ner_and_llm_only,

    -- Average confidence by source count
    AVG(CASE WHEN source_count = 1 THEN ensemble_confidence END) AS avg_conf_single_source,
    AVG(CASE WHEN source_count = 2 THEN ensemble_confidence END) AS avg_conf_two_sources,
    AVG(CASE WHEN source_count = 3 THEN ensemble_confidence END) AS avg_conf_three_sources
FROM entity_sources
"""

layer_stats = spark.sql(layer_contribution_sql).collect()[0]

total = layer_stats["total_entities"]
print(f"  Total entities: {total}")
print(f"  Dictionary: {layer_stats['dict_found']} ({100*layer_stats['dict_found']/total:.1f}%) -- {layer_stats['dict_unique']} unique ({100*layer_stats['dict_unique']/total:.1f}%)")
print(f"  NER:        {layer_stats['ner_found']} ({100*layer_stats['ner_found']/total:.1f}%) -- {layer_stats['ner_unique']} unique ({100*layer_stats['ner_unique']/total:.1f}%)")
print(f"  LLM:        {layer_stats['llm_found']} ({100*layer_stats['llm_found']/total:.1f}%) -- {layer_stats['llm_unique']} unique ({100*layer_stats['llm_unique']/total:.1f}%)")
print(f"  All three:  {layer_stats['all_three']} ({100*layer_stats['all_three']/total:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b. Layer Contribution by Entity Type

# COMMAND ----------

layer_by_type_sql = f"""
SELECT
    entity_type,
    COUNT(*) AS total,
    COUNT(CASE WHEN ARRAY_CONTAINS(sources, 'dictionary') THEN 1 END) AS dict_found,
    COUNT(CASE WHEN ARRAY_CONTAINS(sources, 'ner') THEN 1 END) AS ner_found,
    COUNT(CASE WHEN ARRAY_CONTAINS(sources, 'llm') THEN 1 END) AS llm_found,
    COUNT(CASE WHEN ARRAY_CONTAINS(sources, 'dictionary') AND NOT ARRAY_CONTAINS(sources, 'ner') AND NOT ARRAY_CONTAINS(sources, 'llm') THEN 1 END) AS dict_unique,
    COUNT(CASE WHEN ARRAY_CONTAINS(sources, 'ner') AND NOT ARRAY_CONTAINS(sources, 'dictionary') AND NOT ARRAY_CONTAINS(sources, 'llm') THEN 1 END) AS ner_unique,
    COUNT(CASE WHEN ARRAY_CONTAINS(sources, 'llm') AND NOT ARRAY_CONTAINS(sources, 'dictionary') AND NOT ARRAY_CONTAINS(sources, 'ner') THEN 1 END) AS llm_unique,
    ROUND(AVG(ensemble_confidence), 4) AS avg_confidence
FROM {CATALOG}.extracted.merged_entities
GROUP BY entity_type
ORDER BY total DESC
"""

spark.sql(layer_by_type_sql).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 2: Per-Section Recall Analysis
# MAGIC
# MAGIC Break down extraction quality by clinical note section to identify
# MAGIC which sections produce the highest and lowest quality extractions.

# COMMAND ----------

section_analysis_sql = f"""
SELECT
    ds.section_type,
    COUNT(DISTINCT me.entity_id) AS entities_found,
    AVG(me.ensemble_confidence) AS avg_confidence,
    COUNT(DISTINCT CASE WHEN ARRAY_CONTAINS(me.sources, 'dictionary') THEN me.entity_id END) AS dict_found,
    COUNT(DISTINCT CASE WHEN ARRAY_CONTAINS(me.sources, 'ner') THEN me.entity_id END) AS ner_found,
    COUNT(DISTINCT CASE WHEN ARRAY_CONTAINS(me.sources, 'llm') THEN me.entity_id END) AS llm_found,
    COUNT(DISTINCT me.chart_id) AS charts_with_section,
    ROUND(COUNT(DISTINCT me.entity_id) * 1.0 / COUNT(DISTINCT me.chart_id), 2) AS entities_per_chart
FROM {CATALOG}.extracted.merged_entities me
JOIN {CATALOG}.extracted.document_sections ds ON me.section_id = ds.section_id
GROUP BY ds.section_type
ORDER BY entities_found DESC
"""

section_df = spark.sql(section_analysis_sql)
section_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section Confidence Distribution

# COMMAND ----------

section_confidence_sql = f"""
SELECT
    ds.section_type,
    CASE
        WHEN me.ensemble_confidence >= 0.9 THEN 'high (>=0.9)'
        WHEN me.ensemble_confidence >= 0.7 THEN 'medium (0.7-0.9)'
        WHEN me.ensemble_confidence >= 0.5 THEN 'low (0.5-0.7)'
        ELSE 'very_low (<0.5)'
    END AS confidence_bin,
    COUNT(*) AS entity_count
FROM {CATALOG}.extracted.merged_entities me
JOIN {CATALOG}.extracted.document_sections ds ON me.section_id = ds.section_id
GROUP BY ds.section_type, confidence_bin
ORDER BY ds.section_type, confidence_bin
"""

spark.sql(section_confidence_sql).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 3: Note Density Analysis
# MAGIC
# MAGIC Correlate note length and complexity with extraction quality.
# MAGIC Identify the density threshold where quality degrades.

# COMMAND ----------

density_analysis_sql = f"""
WITH chart_stats AS (
    SELECT
        c.chart_id,
        LENGTH(c.raw_text) AS char_count,
        LENGTH(c.raw_text) / 5.0 AS approx_word_count,
        COUNT(me.entity_id) AS entity_count,
        AVG(me.ensemble_confidence) AS avg_confidence,
        COUNT(me.entity_id) / (LENGTH(c.raw_text) / 1000.0) AS entities_per_1k_chars
    FROM {CATALOG}.raw.charts c
    LEFT JOIN {CATALOG}.extracted.merged_entities me ON c.chart_id = me.chart_id
    WHERE c.raw_text IS NOT NULL AND LENGTH(c.raw_text) > 0
    GROUP BY c.chart_id, c.raw_text
)
SELECT
    CASE
        WHEN entities_per_1k_chars < 2 THEN '1. sparse (<2/1k chars)'
        WHEN entities_per_1k_chars < 5 THEN '2. low (2-5/1k chars)'
        WHEN entities_per_1k_chars < 10 THEN '3. medium (5-10/1k chars)'
        WHEN entities_per_1k_chars < 20 THEN '4. high (10-20/1k chars)'
        ELSE '5. very_high (20+/1k chars)'
    END AS density_bin,
    COUNT(*) AS chart_count,
    ROUND(AVG(entity_count), 1) AS avg_entities,
    ROUND(AVG(avg_confidence), 4) AS avg_confidence,
    ROUND(AVG(approx_word_count), 0) AS avg_word_count,
    ROUND(MIN(avg_confidence), 4) AS min_confidence,
    ROUND(MAX(avg_confidence), 4) AS max_confidence
FROM chart_stats
GROUP BY density_bin
ORDER BY density_bin
"""

density_df = spark.sql(density_analysis_sql)
density_df.display()

# COMMAND ----------

# Identify the density threshold where confidence drops below 0.7
density_rows = density_df.collect()
threshold_found = False
for row in density_rows:
    if row["avg_confidence"] and row["avg_confidence"] < 0.7:
        print(f"  Quality degradation threshold: {row['density_bin']} (avg confidence: {row['avg_confidence']:.4f})")
        threshold_found = True
        break

if not threshold_found:
    print(f"  No significant quality degradation detected across density bins")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 4: Assertion Quality
# MAGIC
# MAGIC Analyze assertion classification results: negation detection, temporality,
# MAGIC and experiencer attribution.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Assertion Distribution

# COMMAND ----------

assertion_dist_sql = f"""
SELECT
    assertion_status,
    negation_detected,
    temporality,
    experiencer,
    certainty,
    COUNT(*) AS entity_count,
    ROUND(AVG(confidence), 4) AS avg_confidence
FROM {CATALOG}.extracted.entity_assertions
GROUP BY assertion_status, negation_detected, temporality, experiencer, certainty
ORDER BY entity_count DESC
"""

spark.sql(assertion_dist_sql).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Negation Detection by Section

# COMMAND ----------

negation_by_section_sql = f"""
SELECT
    ds.section_type,
    COUNT(*) AS total_assertions,
    COUNT(CASE WHEN ea.negation_detected = true THEN 1 END) AS negated_count,
    ROUND(
        COUNT(CASE WHEN ea.negation_detected = true THEN 1 END) * 100.0
        / NULLIF(COUNT(*), 0),
        2
    ) AS negation_rate_pct,
    ROUND(
        AVG(CASE WHEN ea.negation_detected = true THEN ea.confidence END),
        4
    ) AS avg_negation_confidence
FROM {CATALOG}.extracted.entity_assertions ea
JOIN {CATALOG}.extracted.merged_entities me ON ea.entity_id = me.entity_id
LEFT JOIN {CATALOG}.extracted.document_sections ds ON me.section_id = ds.section_id
GROUP BY ds.section_type
ORDER BY negation_rate_pct DESC
"""

spark.sql(negation_by_section_sql).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4c. Assertion Accuracy vs Gold Set (if available)

# COMMAND ----------

gold_assertion_count = spark.sql(f"""
    SELECT COUNT(*) AS cnt
    FROM {CATALOG}.feedback.gold_annotations
    WHERE annotation_round IS NOT NULL
""").collect()[0]["cnt"]

if gold_assertion_count > 0:
    print(f"  Gold set available ({gold_assertion_count} annotations) -- assertion accuracy analysis possible")
    print(f"  (Assertion-level gold annotations require dedicated annotation fields)")
else:
    print(f"  No gold set annotations available -- skipping assertion accuracy analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 5: Codification Quality
# MAGIC
# MAGIC Analyze the coding pipeline: ontology mapping success, specificity
# MAGIC improvement, multi-pass agreement, and code validity.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5a. ICD-10 Mapping Success Rate

# COMMAND ----------

icd10_quality_sql = f"""
SELECT
    COUNT(*) AS total_mappings,
    COUNT(CASE WHEN icd10_code IS NOT NULL THEN 1 END) AS mapped_count,
    ROUND(COUNT(CASE WHEN icd10_code IS NOT NULL THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS mapping_success_pct,
    COUNT(DISTINCT icd10_code) AS unique_codes_assigned,
    ROUND(AVG(confidence), 4) AS avg_confidence,
    COUNT(CASE WHEN resolution_path = 'ONTOLOGY_DIRECT' THEN 1 END) AS via_ontology,
    COUNT(CASE WHEN resolution_path = 'LLM_ASSIGNED' THEN 1 END) AS via_llm,
    COUNT(CASE WHEN resolution_path = 'R1_R2_AGREE' THEN 1 END) AS via_multipass
FROM {CATALOG}.codified.icd10_mappings
"""

icd10_stats = spark.sql(icd10_quality_sql)
icd10_stats.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5b. LOINC Mapping Success Rate

# COMMAND ----------

loinc_quality_sql = f"""
SELECT
    COUNT(*) AS total_mappings,
    COUNT(CASE WHEN loinc_code IS NOT NULL THEN 1 END) AS mapped_count,
    ROUND(COUNT(CASE WHEN loinc_code IS NOT NULL THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS mapping_success_pct,
    COUNT(DISTINCT loinc_code) AS unique_codes_assigned,
    ROUND(AVG(confidence), 4) AS avg_confidence
FROM {CATALOG}.codified.loinc_mappings
"""

loinc_stats = spark.sql(loinc_quality_sql)
loinc_stats.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5c. Specificity Enhancement Analysis

# COMMAND ----------

specificity_sql = f"""
SELECT
    CASE
        WHEN LENGTH(icd10_code) <= 4 THEN 'category (3-4 char)'
        WHEN LENGTH(icd10_code) <= 5 THEN 'subcategory (5 char)'
        WHEN LENGTH(icd10_code) <= 6 THEN 'specific (6 char)'
        ELSE 'most_specific (7+ char)'
    END AS specificity_level,
    COUNT(*) AS code_count,
    ROUND(AVG(confidence), 4) AS avg_confidence,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM {CATALOG}.codified.icd10_mappings
WHERE icd10_code IS NOT NULL
GROUP BY specificity_level
ORDER BY specificity_level
"""

spark.sql(specificity_sql).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5d. Multi-Pass Validation Agreement

# COMMAND ----------

multipass_sql = f"""
SELECT
    validation_round,
    validation_role,
    COUNT(*) AS total_reviews,
    COUNT(CASE WHEN validation_result = 'approved' THEN 1 END) AS approved,
    COUNT(CASE WHEN validation_result = 'rejected' THEN 1 END) AS rejected,
    COUNT(CASE WHEN validation_result = 'modified' THEN 1 END) AS modified,
    ROUND(COUNT(CASE WHEN validation_result = 'approved' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS approval_rate_pct
FROM {CATALOG}.codified.validation_results
GROUP BY validation_round, validation_role
ORDER BY validation_round, validation_role
"""

try:
    spark.sql(multipass_sql).display()
except Exception as e:
    print(f"  Multi-pass validation table not available: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5e. Code Validity Check

# COMMAND ----------

code_validity_sql = f"""
SELECT
    'ICD-10' AS code_system,
    COUNT(*) AS total_assigned,
    COUNT(CASE WHEN ref.icd10_code IS NOT NULL THEN 1 END) AS valid_codes,
    COUNT(CASE WHEN ref.icd10_code IS NULL THEN 1 END) AS invalid_codes,
    ROUND(COUNT(CASE WHEN ref.icd10_code IS NOT NULL THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS validity_pct
FROM {CATALOG}.codified.icd10_mappings m
LEFT JOIN {CATALOG}.reference.icd10_codes ref ON m.icd10_code = ref.icd10_code
WHERE m.icd10_code IS NOT NULL

UNION ALL

SELECT
    'LOINC' AS code_system,
    COUNT(*) AS total_assigned,
    COUNT(CASE WHEN ref.loinc_code IS NOT NULL THEN 1 END) AS valid_codes,
    COUNT(CASE WHEN ref.loinc_code IS NULL THEN 1 END) AS invalid_codes,
    ROUND(COUNT(CASE WHEN ref.loinc_code IS NOT NULL THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS validity_pct
FROM {CATALOG}.codified.loinc_mappings m
LEFT JOIN {CATALOG}.reference.loinc_codes ref ON m.loinc_code = ref.loinc_code
WHERE m.loinc_code IS NOT NULL
"""

try:
    spark.sql(code_validity_sql).display()
except Exception as e:
    print(f"  Code validity check skipped (reference tables may not exist): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 6: Trending
# MAGIC
# MAGIC Track quality metrics across pipeline runs to visualize improvement
# MAGIC over time as the feedback loop takes effect.

# COMMAND ----------

trending_sql = f"""
SELECT
    DATE(run_timestamp) AS run_date,
    metric_scope,
    scope_value,
    AVG(recall_score) AS avg_recall,
    AVG(precision_score) AS avg_precision,
    AVG(f1_score) AS avg_f1,
    SUM(true_positives) AS total_tp,
    SUM(false_negatives) AS total_fn
FROM {CATALOG}.feedback.recall_metrics
GROUP BY DATE(run_timestamp), metric_scope, scope_value
ORDER BY run_date
"""

try:
    trending_df = spark.sql(trending_sql)
    if trending_df.count() > 0:
        trending_df.display()
    else:
        print(f"  No recall metrics history available yet -- run notebook 07 first")
except Exception as e:
    print(f"  Trending data not available: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recall Improvement Over Feedback Cycles

# COMMAND ----------

cycle_comparison_sql = f"""
SELECT
    run_date,
    avg_recall,
    avg_precision,
    avg_f1,
    avg_recall - LAG(avg_recall) OVER (ORDER BY run_date) AS recall_delta,
    avg_f1 - LAG(avg_f1) OVER (ORDER BY run_date) AS f1_delta
FROM (
    SELECT
        DATE(run_timestamp) AS run_date,
        AVG(recall_score) AS avg_recall,
        AVG(precision_score) AS avg_precision,
        AVG(f1_score) AS avg_f1
    FROM {CATALOG}.feedback.recall_metrics
    WHERE metric_scope = 'overall'
    GROUP BY DATE(run_timestamp)
)
ORDER BY run_date
"""

try:
    spark.sql(cycle_comparison_sql).display()
except Exception as e:
    print(f"  Cycle comparison not available: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 7: Executive Summary
# MAGIC
# MAGIC Comprehensive text summary of pipeline quality metrics.

# COMMAND ----------

# Gather all summary statistics
charts_processed = spark.table(f"{CATALOG}.raw.charts").count()
total_entities = spark.table(f"{CATALOG}.extracted.merged_entities").count()

# Layer percentages
layer_pcts = spark.sql(f"""
    SELECT
        ROUND(COUNT(CASE WHEN ARRAY_CONTAINS(sources, 'dictionary') THEN 1 END) * 100.0 / COUNT(*), 1) AS dict_pct,
        ROUND(COUNT(CASE WHEN ARRAY_CONTAINS(sources, 'ner') THEN 1 END) * 100.0 / COUNT(*), 1) AS ner_pct,
        ROUND(COUNT(CASE WHEN ARRAY_CONTAINS(sources, 'llm') THEN 1 END) * 100.0 / COUNT(*), 1) AS llm_pct,
        ROUND(AVG(ensemble_confidence) * 100, 1) AS overall_confidence_pct
    FROM {CATALOG}.extracted.merged_entities
""").collect()[0]

# Assertion stats
try:
    assertion_stats = spark.sql(f"""
        SELECT
            ROUND(COUNT(CASE WHEN negation_detected = true THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 1) AS negated_pct,
            ROUND(COUNT(CASE WHEN temporality = 'HISTORICAL' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 1) AS historical_pct,
            ROUND(COUNT(CASE WHEN experiencer = 'FAMILY_MEMBER' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 1) AS family_pct
        FROM {CATALOG}.extracted.entity_assertions
    """).collect()[0]
    negated_pct = assertion_stats["negated_pct"]
    historical_pct = assertion_stats["historical_pct"]
    family_pct = assertion_stats["family_pct"]
except Exception:
    negated_pct = "N/A"
    historical_pct = "N/A"
    family_pct = "N/A"

# Codification counts
try:
    icd10_count = spark.sql(f"SELECT COUNT(DISTINCT icd10_code) AS cnt FROM {CATALOG}.codified.icd10_mappings WHERE icd10_code IS NOT NULL").collect()[0]["cnt"]
except Exception:
    icd10_count = "N/A"

try:
    loinc_count = spark.sql(f"SELECT COUNT(DISTINCT loinc_code) AS cnt FROM {CATALOG}.codified.loinc_mappings WHERE loinc_code IS NOT NULL").collect()[0]["cnt"]
except Exception:
    loinc_count = "N/A"

# Recall from gold set
try:
    latest_recall = spark.sql(f"""
        SELECT recall_score FROM {CATALOG}.feedback.recall_metrics
        WHERE metric_scope = 'overall'
        ORDER BY run_timestamp DESC LIMIT 1
    """).collect()
    recall_str = f"{latest_recall[0]['recall_score']*100:.1f}%" if latest_recall else "not measured"
except Exception:
    recall_str = "not measured"

# Error patterns
try:
    error_pattern_list = spark.sql(f"""
        SELECT pattern_type, severity
        FROM {CATALOG}.feedback.error_patterns
        WHERE status = 'open'
        ORDER BY CASE severity WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END
        LIMIT 5
    """).collect()
    if error_pattern_list:
        top_errors = "; ".join([f"{r['pattern_type']} ({r['severity']})" for r in error_pattern_list])
    else:
        top_errors = "none detected"
except Exception:
    top_errors = "not available"

# COMMAND ----------

print("=" * 50)
print("  PIPELINE QUALITY REPORT")
print("=" * 50)
print(f"  Charts processed:       {charts_processed}")
print(f"  Total entities extracted: {total_entities}")
print(f"  Extraction layers:       dictionary ({layer_pcts['dict_pct']}%), NER ({layer_pcts['ner_pct']}%), LLM ({layer_pcts['llm_pct']}%)")
print(f"  Assertion classification: {negated_pct}% negated, {historical_pct}% historical, {family_pct}% family")
print(f"  Codification:            {icd10_count} ICD-10, {loinc_count} LOINC")
print(f"  Overall confidence:      {layer_pcts['overall_confidence_pct']}%")
print(f"  Recall (vs gold set):    {recall_str}")
print(f"  Top error patterns:      {top_errors}")
print("=" * 50)
