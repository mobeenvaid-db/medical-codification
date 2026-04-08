# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Active Learning & Feedback Pipeline
# MAGIC
# MAGIC This notebook implements the continuous learning loop that turns human corrections
# MAGIC into training data and model improvements. It covers:
# MAGIC
# MAGIC 1. **Gold set management** — import human-annotated gold standard files
# MAGIC 2. **Recall & precision measurement** — compare pipeline output against gold annotations
# MAGIC 3. **Error pattern detection** — identify systematic extraction failures
# MAGIC 4. **Active learning queue** — prioritize the next documents for human annotation
# MAGIC 5. **Dictionary & model update recommendations** — actionable fixes from error analysis
# MAGIC 6. **Feedback loop metrics** — dashboard-ready summary tables
# MAGIC
# MAGIC **Input:** `extracted.merged_entities`, `feedback.gold_annotations`
# MAGIC **Output:** `feedback.recall_metrics`, `feedback.error_patterns`, `feedback.annotation_queue`
# MAGIC
# MAGIC **Estimated runtime:** ~5 minutes

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
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 1: Gold Set Management
# MAGIC
# MAGIC Import human-annotated gold standard files (CSV/JSON) uploaded to the
# MAGIC feedback volume. Supports multiple annotation rounds with inter-annotator
# MAGIC agreement scoring.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a. Import Gold Annotations from Volume

# COMMAND ----------

GOLD_VOLUME = f"/Volumes/{CATALOG}/feedback/gold_sets"

# Ensure volume exists
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.feedback.gold_sets")

print(f"  Scanning gold annotation files in {GOLD_VOLUME}")

try:
    gold_files = dbutils.fs.ls(GOLD_VOLUME)
except Exception:
    gold_files = []
    print(f"  No gold annotation files found (volume is empty)")
csv_files = [f.path for f in gold_files if f.path.endswith(".csv")]
json_files = [f.path for f in gold_files if f.path.endswith(".json")]

print(f"  Found {len(csv_files)} CSV files and {len(json_files)} JSON files")

# COMMAND ----------

# Load CSV gold annotations
if csv_files:
    gold_csv_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(csv_files)
    )
    print(f"  Loaded {gold_csv_df.count()} annotations from CSV files")
else:
    gold_csv_df = spark.createDataFrame([], schema=StructType([
        StructField("chart_id", StringType()),
        StructField("entity_type", StringType()),
        StructField("entity_text", StringType()),
        StructField("start_offset", IntegerType()),
        StructField("end_offset", IntegerType()),
        StructField("annotator_id", StringType()),
        StructField("annotation_round", IntegerType()),
    ]))

# Load JSON gold annotations
if json_files:
    gold_json_df = spark.read.option("multiline", True).json(json_files)
    print(f"  Loaded {gold_json_df.count()} annotations from JSON files")
else:
    gold_json_df = spark.createDataFrame([], schema=gold_csv_df.schema)

# COMMAND ----------

# Combine gold annotations (if any files were found)
gold_combined_df = gold_csv_df.unionByName(gold_json_df, allowMissingColumns=True)
import_count = gold_combined_df.count()

if import_count > 0:
    # Map to gold_annotations table schema
    gold_to_write = gold_combined_df.select(
        F.sha2(F.concat_ws("|",
            F.coalesce(F.col("chart_id"), F.lit("")),
            F.coalesce(F.col("entity_type"), F.lit("")),
            F.coalesce(F.col("entity_text"), F.lit("")),
            F.coalesce(F.col("annotator_id"), F.lit("")),
        ), 256).alias("annotation_id"),
        F.col("chart_id"),
        F.col("entity_type"),
        F.col("entity_text"),
        F.lit(None).cast("string").alias("gold_icd10_code"),
        F.lit(None).cast("string").alias("gold_loinc_code"),
        F.lit(None).cast("string").alias("gold_snomed_id"),
        F.lit(None).cast("string").alias("assertion_status"),
        F.coalesce(F.col("annotator_id"), F.lit("unknown")).alias("annotator"),
        F.coalesce(F.col("annotation_round"), F.lit(1)).alias("annotation_round"),
        F.current_timestamp().alias("annotated_at"),
    )
    gold_to_write.write.mode("append").saveAsTable(f"{CATALOG}.feedback.gold_annotations")
    print(f"  Imported {import_count} gold annotations")
else:
    print(f"  No gold annotation files to import -- skipping write")

gold_count = spark.table(f"{CATALOG}.feedback.gold_annotations").count()
print(f"  Gold annotations table has {gold_count} total annotations")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b. Inter-Annotator Agreement (Cohen's Kappa)

# COMMAND ----------

def calculate_kappa(annotations_df):
    """
    Calculate Cohen's kappa for inter-annotator agreement.
    Compares annotations from round 1 vs round 2 on the same charts.
    Returns kappa score and agreement details.
    """
    # Pivot: for each chart + entity, check if each round found it
    round1 = (
        annotations_df
        .filter(F.col("annotation_round") == 1)
        .select("chart_id", "entity_type", F.lower("entity_text").alias("norm_text"))
        .distinct()
        .withColumn("round1_present", F.lit(1))
    )
    round2 = (
        annotations_df
        .filter(F.col("annotation_round") == 2)
        .select("chart_id", "entity_type", F.lower("entity_text").alias("norm_text"))
        .distinct()
        .withColumn("round2_present", F.lit(1))
    )

    # Full outer join to find all entities across both rounds
    combined = round1.join(
        round2,
        on=["chart_id", "entity_type", "norm_text"],
        how="full_outer"
    ).fillna(0, subset=["round1_present", "round2_present"])

    total = combined.count()
    if total == 0:
        return {"kappa": None, "message": "No overlapping annotations found"}

    # Count agreement categories
    both_yes = combined.filter(
        (F.col("round1_present") == 1) & (F.col("round2_present") == 1)
    ).count()
    both_no = 0  # In this context, we only see entities that at least one annotator found
    r1_only = combined.filter(
        (F.col("round1_present") == 1) & (F.col("round2_present") == 0)
    ).count()
    r2_only = combined.filter(
        (F.col("round1_present") == 0) & (F.col("round2_present") == 1)
    ).count()

    # Observed agreement
    p_observed = (both_yes + both_no) / total if total > 0 else 0

    # Expected agreement by chance
    p_r1_yes = (both_yes + r1_only) / total if total > 0 else 0
    p_r2_yes = (both_yes + r2_only) / total if total > 0 else 0
    p_expected = (p_r1_yes * p_r2_yes) + ((1 - p_r1_yes) * (1 - p_r2_yes))

    # Cohen's kappa
    if p_expected == 1.0:
        kappa = 1.0
    else:
        kappa = (p_observed - p_expected) / (1.0 - p_expected)

    return {
        "kappa": round(kappa, 4),
        "observed_agreement": round(p_observed, 4),
        "expected_agreement": round(p_expected, 4),
        "total_entities": total,
        "agreed_present": both_yes,
        "round1_only": r1_only,
        "round2_only": r2_only,
    }

# COMMAND ----------

# Calculate kappa if multiple annotation rounds exist
annotations_df = spark.table(f"{CATALOG}.feedback.gold_annotations")
num_rounds = annotations_df.select("annotation_round").distinct().count()

if num_rounds >= 2:
    kappa_result = calculate_kappa(annotations_df)
    print(f"  Inter-Annotator Agreement (Cohen's Kappa): {kappa_result['kappa']}")
    print(f"  Observed agreement: {kappa_result['observed_agreement']}")
    print(f"  Total entities evaluated: {kappa_result['total_entities']}")
    print(f"  Agreed present: {kappa_result['agreed_present']}")
    print(f"  Round 1 only: {kappa_result['round1_only']}")
    print(f"  Round 2 only: {kappa_result['round2_only']}")
else:
    print(f"  Only {num_rounds} annotation round(s) found -- skipping kappa calculation")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 2: Recall & Precision Measurement Against Gold Set
# MAGIC
# MAGIC Compare pipeline output against gold annotations to calculate
# MAGIC true positives, false positives, and false negatives at multiple scopes.

# COMMAND ----------

recall_sql = f"""
WITH pipeline_entities AS (
    SELECT chart_id, entity_type, LOWER(entity_text) AS norm_text, section_id
    FROM {CATALOG}.extracted.merged_entities
),
gold_entities AS (
    SELECT chart_id, entity_type, LOWER(entity_text) AS norm_text
    FROM {CATALOG}.feedback.gold_annotations
    WHERE annotation_round = (
        SELECT MAX(annotation_round) FROM {CATALOG}.feedback.gold_annotations
    )
),
-- True positives: entities in both pipeline and gold set
tp AS (
    SELECT p.chart_id, p.entity_type, p.norm_text
    FROM pipeline_entities p
    INNER JOIN gold_entities g
        ON p.chart_id = g.chart_id
        AND p.entity_type = g.entity_type
        AND p.norm_text = g.norm_text
),
-- False positives: entities in pipeline but not in gold set
fp AS (
    SELECT p.chart_id, p.entity_type, p.norm_text
    FROM pipeline_entities p
    LEFT JOIN gold_entities g
        ON p.chart_id = g.chart_id
        AND p.entity_type = g.entity_type
        AND p.norm_text = g.norm_text
    WHERE g.chart_id IS NULL
),
-- False negatives: entities in gold set but not in pipeline
fn AS (
    SELECT g.chart_id, g.entity_type, g.norm_text
    FROM gold_entities g
    LEFT JOIN pipeline_entities p
        ON g.chart_id = p.chart_id
        AND g.entity_type = p.entity_type
        AND g.norm_text = p.norm_text
    WHERE p.chart_id IS NULL
)
SELECT
    'overall' AS metric_scope,
    'all' AS scope_value,
    COUNT(DISTINCT CONCAT(tp.chart_id, tp.entity_type, tp.norm_text)) AS true_positives,
    (SELECT COUNT(DISTINCT CONCAT(chart_id, entity_type, norm_text)) FROM fp) AS false_positives,
    (SELECT COUNT(DISTINCT CONCAT(chart_id, entity_type, norm_text)) FROM fn) AS false_negatives
FROM tp
"""

overall_metrics = spark.sql(recall_sql).collect()[0]
tp = overall_metrics["true_positives"]
fp_count = overall_metrics["false_positives"]
fn_count = overall_metrics["false_negatives"]

recall = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0
precision = tp / (tp + fp_count) if (tp + fp_count) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"  Overall Recall:    {recall:.4f}")
print(f"  Overall Precision: {precision:.4f}")
print(f"  Overall F1:        {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recall by Entity Type

# COMMAND ----------

recall_by_type_sql = f"""
WITH pipeline_entities AS (
    SELECT chart_id, entity_type, LOWER(entity_text) AS norm_text
    FROM {CATALOG}.extracted.merged_entities
),
gold_entities AS (
    SELECT chart_id, entity_type, LOWER(entity_text) AS norm_text
    FROM {CATALOG}.feedback.gold_annotations
    WHERE annotation_round = (
        SELECT MAX(annotation_round) FROM {CATALOG}.feedback.gold_annotations
    )
),
tp_by_type AS (
    SELECT p.entity_type, COUNT(DISTINCT CONCAT(p.chart_id, p.norm_text)) AS tp
    FROM pipeline_entities p
    INNER JOIN gold_entities g
        ON p.chart_id = g.chart_id AND p.entity_type = g.entity_type AND p.norm_text = g.norm_text
    GROUP BY p.entity_type
),
fn_by_type AS (
    SELECT g.entity_type, COUNT(DISTINCT CONCAT(g.chart_id, g.norm_text)) AS fn
    FROM gold_entities g
    LEFT JOIN pipeline_entities p
        ON g.chart_id = p.chart_id AND g.entity_type = p.entity_type AND g.norm_text = p.norm_text
    WHERE p.chart_id IS NULL
    GROUP BY g.entity_type
),
fp_by_type AS (
    SELECT p.entity_type, COUNT(DISTINCT CONCAT(p.chart_id, p.norm_text)) AS fp
    FROM pipeline_entities p
    LEFT JOIN gold_entities g
        ON p.chart_id = g.chart_id AND p.entity_type = g.entity_type AND p.norm_text = g.norm_text
    WHERE g.chart_id IS NULL
    GROUP BY p.entity_type
)
SELECT
    COALESCE(t.entity_type, f.entity_type, fp.entity_type) AS entity_type,
    COALESCE(t.tp, 0) AS true_positives,
    COALESCE(fp.fp, 0) AS false_positives,
    COALESCE(f.fn, 0) AS false_negatives,
    ROUND(COALESCE(t.tp, 0) / NULLIF(COALESCE(t.tp, 0) + COALESCE(f.fn, 0), 0), 4) AS recall_score,
    ROUND(COALESCE(t.tp, 0) / NULLIF(COALESCE(t.tp, 0) + COALESCE(fp.fp, 0), 0), 4) AS precision_score
FROM tp_by_type t
FULL OUTER JOIN fn_by_type f ON t.entity_type = f.entity_type
FULL OUTER JOIN fp_by_type fp ON COALESCE(t.entity_type, f.entity_type) = fp.entity_type
ORDER BY recall_score ASC
"""

recall_by_type_df = spark.sql(recall_by_type_sql)
recall_by_type_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Recall Metrics to Feedback Schema

# COMMAND ----------

# Build metrics rows for all scopes: overall, by_entity_type
from pyspark.sql import Row
from datetime import datetime

run_timestamp = datetime.now()

metrics_rows = []

# Overall
metrics_rows.append(Row(
    metric_scope="overall",
    scope_value="all",
    true_positives=tp,
    false_positives=fp_count,
    false_negatives=fn_count,
    recall_score=float(recall),
    precision_score=float(precision),
    f1_score=float(f1),
    run_timestamp=run_timestamp,
))

# By entity type
for row in recall_by_type_df.collect():
    r = row["recall_score"] or 0.0
    p = row["precision_score"] or 0.0
    f = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
    metrics_rows.append(Row(
        metric_scope="by_entity_type",
        scope_value=row["entity_type"],
        true_positives=row["true_positives"],
        false_positives=row["false_positives"],
        false_negatives=row["false_negatives"],
        recall_score=float(r),
        precision_score=float(p),
        f1_score=float(f),
        run_timestamp=run_timestamp,
    ))

# Write with explicit schema matching feedback.recall_metrics table
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
import uuid as uuid_mod

recall_schema = StructType([
    StructField("run_id", StringType()),
    StructField("run_timestamp", TimestampType()),
    StructField("metric_scope", StringType()),
    StructField("scope_value", StringType()),
    StructField("extraction_layer", StringType()),
    StructField("true_positives", IntegerType()),
    StructField("false_positives", IntegerType()),
    StructField("false_negatives", IntegerType()),
    StructField("precision_score", DoubleType()),
    StructField("recall_score", DoubleType()),
    StructField("f1_score", DoubleType()),
    StructField("charts_evaluated", IntegerType()),
])

run_id = f"RUN-{uuid_mod.uuid4().hex[:12]}"
metrics_dicts = []
for r in metrics_rows:
    d = r.asDict()
    metrics_dicts.append({
        "run_id": run_id,
        "run_timestamp": d.get("run_timestamp"),
        "metric_scope": d.get("metric_scope", ""),
        "scope_value": d.get("scope_value", ""),
        "extraction_layer": "ensemble",
        "true_positives": int(d.get("true_positives", 0)),
        "false_positives": int(d.get("false_positives", 0)),
        "false_negatives": int(d.get("false_negatives", 0)),
        "precision_score": float(d.get("precision_score", 0.0)),
        "recall_score": float(d.get("recall_score", 0.0)),
        "f1_score": float(d.get("f1_score", 0.0)),
        "charts_evaluated": 0,
    })

metrics_df = spark.createDataFrame(metrics_dicts, schema=recall_schema)
metrics_df.write.mode("append").saveAsTable(f"{CATALOG}.feedback.recall_metrics")

print(f"  Wrote {len(metrics_rows)} metric rows to {CATALOG}.feedback.recall_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 3: Error Pattern Detection
# MAGIC
# MAGIC Analyze false negatives (missed entities) and false positives to identify
# MAGIC systematic extraction failures. Uses LLM analysis to cluster error patterns
# MAGIC and suggest targeted fixes.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Collect Missed Entities (False Negatives)

# COMMAND ----------

missed_entities_sql = f"""
SELECT
    g.chart_id,
    g.entity_type,
    g.entity_text,
    ds.section_type,
    LENGTH(c.raw_text) AS note_length,
    (SELECT COUNT(*) FROM {CATALOG}.extracted.merged_entities me2
     WHERE me2.chart_id = g.chart_id) AS total_entities_in_chart
FROM {CATALOG}.feedback.gold_annotations g
LEFT JOIN {CATALOG}.extracted.merged_entities me
    ON g.chart_id = me.chart_id
    AND g.entity_type = me.entity_type
    AND LOWER(g.entity_text) = LOWER(me.entity_text)
LEFT JOIN {CATALOG}.raw.charts c ON g.chart_id = c.chart_id
LEFT JOIN {CATALOG}.extracted.document_sections ds ON g.chart_id = ds.chart_id
WHERE me.chart_id IS NULL
  AND g.annotation_round = (
      SELECT MAX(annotation_round) FROM {CATALOG}.feedback.gold_annotations
  )
"""

missed_entities_df = spark.sql(missed_entities_sql)
missed_count = missed_entities_df.count()
print(f"  Found {missed_count} missed entities (false negatives)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Error Frequency by Entity Type and Section

# COMMAND ----------

# Group missed entities by entity type
print("  Missed entities by entity type:")
missed_entities_df.groupBy("entity_type").count().orderBy(F.desc("count")).display()

# COMMAND ----------

# Group missed entities by section type
print("  Missed entities by section type:")
missed_entities_df.groupBy("section_type").count().orderBy(F.desc("count")).display()

# COMMAND ----------

# Density analysis: do denser notes have worse recall?
print("  Missed entities by note density bin:")
missed_entities_df.withColumn(
    "density_bin",
    F.when(F.col("total_entities_in_chart") < 10, "low (<10)")
    .when(F.col("total_entities_in_chart") < 30, "medium (10-30)")
    .otherwise("high (30+)")
).groupBy("density_bin").count().orderBy("density_bin").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3c. LLM-Powered Error Pattern Analysis
# MAGIC
# MAGIC Send clusters of missed entities to Claude for pattern identification
# MAGIC and fix recommendations.

# COMMAND ----------

# Collect missed entity samples for LLM analysis (limit to avoid token overflow)
missed_samples = missed_entities_df.select(
    "entity_type", "entity_text", "section_type"
).limit(100).toPandas()

missed_entities_list = "\n".join([
    f"- [{r['entity_type']}] \"{r['entity_text']}\" (section: {r['section_type']})"
    for _, r in missed_samples.iterrows()
])

# COMMAND ----------

error_pattern_sql = f"""
SELECT ai_query(
    '{MODEL}',
    CONCAT(
        'Analyze these missed clinical entities and identify the systematic pattern. ',
        'These are entities that human annotators found but the automated pipeline missed.',
        '\n\nMISSED ENTITIES:\n',
        :missed_list,
        '\n\nFor each pattern you identify: describe it, suggest a fix ',
        '(add to dictionary, retrain NER, improve prompt, add abbreviation mapping), ',
        'and rate severity HIGH/MEDIUM/LOW based on clinical impact.',
        '\n\nReturn a JSON array: ',
        '[{{"pattern_type": "...", "description": "...", "affected_entity_types": ["..."], ',
        '"example_entities": ["..."], "suggested_fix": "...", "severity": "..."}}]'
    )
) AS error_pattern_analysis
"""

error_patterns_raw = spark.sql(
    error_pattern_sql, args={"missed_list": missed_entities_list}
).collect()[0]["error_pattern_analysis"]
print(f"  LLM error pattern analysis complete")

# COMMAND ----------

# Parse and write error patterns to table
import re

try:
    patterns = json.loads(error_patterns_raw)
except json.JSONDecodeError:
    json_match = re.search(r'\[.*\]', error_patterns_raw, re.DOTALL)
    patterns = json.loads(json_match.group()) if json_match else []

pattern_rows = []
for p in patterns:
    pattern_rows.append(Row(
        pattern_id=f"PAT-{uuid_mod.uuid4().hex[:12]}",
        pattern_type=p.get("pattern_type", "unknown"),
        description=p.get("description", ""),
        example_entity_text=str(p.get("example_entities", []))[:500],
        example_chart_id=None,
        frequency=1,
        severity=p.get("severity", "MEDIUM"),
        suggested_fix=p.get("suggested_fix", ""),
        detected_at=run_timestamp,
        resolved_at=None,
    ))

if pattern_rows:
    patterns_df = spark.createDataFrame(pattern_rows)
    patterns_df.write.mode("append").saveAsTable(f"{CATALOG}.feedback.error_patterns")
    print(f"  Wrote {len(pattern_rows)} error patterns to {CATALOG}.feedback.error_patterns")
else:
    print(f"  No error patterns detected")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 4: Active Learning Queue Prioritization
# MAGIC
# MAGIC Select the highest-value documents for the next round of human annotation.
# MAGIC Priority scoring combines:
# MAGIC - **Uncertainty sampling** -- low average confidence
# MAGIC - **Disagreement sampling** -- high variance across extraction layers
# MAGIC - **Diversity sampling** -- underrepresented note types
# MAGIC - **Error-pattern sampling** -- documents matching known error patterns

# COMMAND ----------

annotation_queue_sql = f"""
WITH entity_stats AS (
    SELECT
        chart_id,
        AVG(ensemble_confidence) AS avg_confidence,
        STDDEV(ensemble_confidence) AS confidence_variance,
        COUNT(*) AS entity_count,
        COUNT(DISTINCT CASE WHEN ARRAY_CONTAINS(sources, 'dictionary') THEN entity_id END) AS dict_count,
        COUNT(DISTINCT CASE WHEN ARRAY_CONTAINS(sources, 'ner') THEN entity_id END) AS ner_count,
        COUNT(DISTINCT CASE WHEN ARRAY_CONTAINS(sources, 'llm') THEN entity_id END) AS llm_count,
        -- Source diversity: how many layers contributed
        (CASE WHEN COUNT(DISTINCT CASE WHEN ARRAY_CONTAINS(sources, 'dictionary') THEN 1 END) > 0 THEN 1 ELSE 0 END +
         CASE WHEN COUNT(DISTINCT CASE WHEN ARRAY_CONTAINS(sources, 'ner') THEN 1 END) > 0 THEN 1 ELSE 0 END +
         CASE WHEN COUNT(DISTINCT CASE WHEN ARRAY_CONTAINS(sources, 'llm') THEN 1 END) > 0 THEN 1 ELSE 0 END) AS active_layers
    FROM {CATALOG}.extracted.merged_entities
    GROUP BY chart_id
),
-- Exclude already-annotated charts
already_annotated AS (
    SELECT DISTINCT chart_id
    FROM {CATALOG}.feedback.gold_annotations
),
priority_scored AS (
    SELECT
        es.chart_id,
        es.avg_confidence,
        es.confidence_variance,
        es.entity_count,
        es.active_layers,
        -- Priority formula: lower confidence + higher variance + fewer layers = higher priority
        (1.0 - es.avg_confidence) * 0.4
        + COALESCE(es.confidence_variance, 0) * 0.3
        + (1.0 - es.active_layers / 3.0) * 0.3 AS annotation_priority
    FROM entity_stats es
    LEFT JOIN already_annotated aa ON es.chart_id = aa.chart_id
    WHERE aa.chart_id IS NULL
)
SELECT
    chart_id,
    ROUND(avg_confidence, 4) AS avg_confidence,
    ROUND(confidence_variance, 4) AS confidence_variance,
    entity_count,
    active_layers,
    ROUND(annotation_priority, 4) AS annotation_priority
FROM priority_scored
ORDER BY annotation_priority DESC
LIMIT 50
"""

annotation_queue_df = spark.sql(annotation_queue_sql)
queue_count = annotation_queue_df.count()
print(f"  Selected {queue_count} charts for next annotation round")

# COMMAND ----------

annotation_queue_df.display()

# COMMAND ----------

# Write annotation queue
annotation_queue_df.withColumn(
    "queued_at", F.current_timestamp()
).withColumn(
    "queue_status", F.lit("pending")
).write.mode("overwrite").saveAsTable(f"{CATALOG}.feedback.annotation_queue")

print(f"  Annotation queue written to {CATALOG}.feedback.annotation_queue")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 5: Dictionary & Model Update Recommendations
# MAGIC
# MAGIC Based on error patterns, generate specific actionable recommendations for
# MAGIC improving extraction quality in the next pipeline run.

# COMMAND ----------

# Collect open error patterns
open_patterns = spark.table(f"{CATALOG}.feedback.error_patterns").filter(
    F.col("resolved_at").isNull()
).toPandas()

if len(open_patterns) == 0:
    print("  No open error patterns -- skipping recommendation generation")

# COMMAND ----------

if len(open_patterns) > 0:
    patterns_summary = "\n".join([
        f"- [{r['severity']}] {r['pattern_type']}: {r['description']} (fix: {r['suggested_fix']})"
        for _, r in open_patterns.iterrows()
    ])

    recommendation_sql = f"""
    SELECT ai_query(
        '{MODEL}',
        CONCAT(
            'Based on these error patterns from a clinical NLP pipeline, generate specific actionable recommendations. ',
            'The pipeline extracts clinical entities from medical charts using dictionary matching, NER models, and LLM extraction.',
            '\n\nERROR PATTERNS:\n',
            :pattern_summary,
            '\n\nGenerate recommendations in these categories:',
            '\n1. NEW_DICTIONARY_TERMS: terms to add to clinical_abbreviations or medical_dictionary tables',
            '\n2. NER_TRAINING_DATA: entity examples to add to fine-tuning dataset',
            '\n3. PROMPT_ADJUSTMENTS: changes to LLM extraction prompts',
            '\n4. PIPELINE_CONFIG: changes to confidence thresholds or ensemble weights',
            '\n\nReturn JSON array: [{{"category": "...", "action": "...", "details": "...", "priority": 1-5}}]'
        )
    ) AS recommendations
    """

    recommendations_raw = spark.sql(
        recommendation_sql, args={"pattern_summary": patterns_summary}
    ).collect()[0]["recommendations"]

    try:
        recommendations = json.loads(recommendations_raw)
    except json.JSONDecodeError:
        json_match = re.search(r'\[.*\]', recommendations_raw, re.DOTALL)
        recommendations = json.loads(json_match.group()) if json_match else []

    print(f"  Generated {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"    [{rec.get('category')}] (priority {rec.get('priority')}): {rec.get('action')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 6: Feedback Loop Metrics Dashboard Data
# MAGIC
# MAGIC Create summary views that the review application can query for
# MAGIC feedback loop analytics and continuous improvement tracking.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a. Cumulative Recall Over Time

# COMMAND ----------

recall_trend_sql = f"""
CREATE OR REPLACE VIEW {CATALOG}.feedback.vw_recall_trend AS
SELECT
    DATE(run_timestamp) AS run_date,
    metric_scope,
    scope_value,
    AVG(recall_score) AS avg_recall,
    AVG(precision_score) AS avg_precision,
    AVG(f1_score) AS avg_f1,
    SUM(true_positives) AS total_tp,
    SUM(false_positives) AS total_fp,
    SUM(false_negatives) AS total_fn
FROM {CATALOG}.feedback.recall_metrics
GROUP BY DATE(run_timestamp), metric_scope, scope_value
ORDER BY run_date
"""

spark.sql(recall_trend_sql)
print(f"  Created view {CATALOG}.feedback.vw_recall_trend")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6b. Error Patterns -- Resolved vs Outstanding

# COMMAND ----------

error_status_sql = f"""
CREATE OR REPLACE VIEW {CATALOG}.feedback.vw_error_pattern_status AS
SELECT
    severity,
    CASE WHEN resolved_at IS NULL THEN 'open' ELSE 'resolved' END AS status,
    COUNT(*) AS pattern_count,
    COLLECT_LIST(pattern_type) AS pattern_types
FROM {CATALOG}.feedback.error_patterns
GROUP BY severity, CASE WHEN resolved_at IS NULL THEN 'open' ELSE 'resolved' END
ORDER BY
    CASE severity WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 WHEN 'LOW' THEN 3 END,
    status
"""

spark.sql(error_status_sql)
print(f"  Created view {CATALOG}.feedback.vw_error_pattern_status")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6c. Annotation Throughput

# COMMAND ----------

annotation_throughput_sql = f"""
CREATE OR REPLACE VIEW {CATALOG}.feedback.vw_annotation_throughput AS
SELECT
    DATE(annotated_at) AS import_date,
    annotator,
    annotation_round,
    COUNT(*) AS annotations_submitted,
    COUNT(DISTINCT chart_id) AS charts_annotated
FROM {CATALOG}.feedback.gold_annotations
GROUP BY DATE(annotated_at), annotator, annotation_round
ORDER BY import_date DESC
"""

spark.sql(annotation_throughput_sql)
print(f"  Created view {CATALOG}.feedback.vw_annotation_throughput")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6d. Feedback Loop Summary

# COMMAND ----------

# Print summary
total_gold = spark.table(f"{CATALOG}.feedback.gold_annotations").count()
total_patterns = spark.table(f"{CATALOG}.feedback.error_patterns").count()
open_patterns_count = spark.table(f"{CATALOG}.feedback.error_patterns").filter(
    F.col("resolved_at").isNull()
).count()
queue_size = spark.table(f"{CATALOG}.feedback.annotation_queue").count()

latest_metrics = spark.sql(f"""
    SELECT recall_score, precision_score, f1_score
    FROM {CATALOG}.feedback.recall_metrics
    WHERE metric_scope = 'overall'
    ORDER BY run_timestamp DESC
    LIMIT 1
""").collect()

print("=" * 50)
print("  ACTIVE LEARNING PIPELINE SUMMARY")
print("=" * 50)
print(f"  Gold annotations:        {total_gold}")
print(f"  Error patterns detected:  {total_patterns} ({open_patterns_count} open)")
print(f"  Annotation queue size:    {queue_size} charts")
if latest_metrics:
    m = latest_metrics[0]
    print(f"  Latest recall:            {m['recall_score']:.4f}")
    print(f"  Latest precision:         {m['precision_score']:.4f}")
    print(f"  Latest F1:                {m['f1_score']:.4f}")
print("=" * 50)
