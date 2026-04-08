# Databricks notebook source
# MAGIC %md
# MAGIC # 05a — Unified Entity Codification (ICD-10, LOINC, UMLS)
# MAGIC
# MAGIC Consolidates concept normalization, ontology coding, specificity enhancement,
# MAGIC and validation into a single notebook with a two-tier architecture that reduces
# MAGIC LLM calls from ~455 to ~100 (one per chart for unresolved entities).
# MAGIC
# MAGIC ### Two-Tier Architecture
# MAGIC
# MAGIC | Tier | Method | Coverage | LLM Calls |
# MAGIC |------|--------|----------|-----------|
# MAGIC | **1** | Ontology SQL joins (medical_dictionary, snomed_icd10_map, loinc_search, icd10_codes_full) | 60-70% | **0** |
# MAGIC | **2** | Single `ai_query` per chart for all unresolved entities | 30-40% | **~100** (one per chart) |
# MAGIC
# MAGIC ### Replaces
# MAGIC
# MAGIC - `05a_concept_normalization` -- UMLS CUI mapping (3 tiers)
# MAGIC - `05b_ontology_guided_coding` -- SNOMED->ICD-10 + LOINC ontology matching
# MAGIC - `05c_specificity_enhancement` -- LLM-assisted code refinement
# MAGIC - `05d_multipass_validation` -- Coder/Auditor/Arbiter validation
# MAGIC
# MAGIC ### Pipeline Position
# MAGIC
# MAGIC ```
# MAGIC 04_extract -> 05a_codify_entities -> 06_sync_to_app
# MAGIC ```
# MAGIC
# MAGIC **Estimated runtime:** ~5-10 minutes for 100 charts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("CATALOG", "mv_catalog", "Unity Catalog Name")
CATALOG = dbutils.widgets.get("CATALOG")

MODEL = "databricks-claude-sonnet-4-6"

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0: Ensure Output Tables Exist

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.codified.icd10_mappings (
    mapping_id STRING,
    entity_id STRING,
    chart_id STRING,
    entity_text STRING,
    icd10_code STRING,
    icd10_description STRING,
    confidence DOUBLE,
    resolution_path STRING,
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
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.codified.loinc_mappings (
    mapping_id STRING,
    entity_id STRING,
    chart_id STRING,
    entity_text STRING,
    loinc_code STRING,
    loinc_long_name STRING,
    confidence DOUBLE,
    resolution_path STRING,
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
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.codified.umls_mappings (
    mapping_id STRING,
    entity_id STRING,
    chart_id STRING,
    entity_text STRING,
    cui STRING,
    preferred_name STRING,
    semantic_type STRING,
    snomed_concept_id STRING,
    snomed_name STRING,
    mapping_method STRING,
    confidence DOUBLE,
    mapped_at TIMESTAMP
)
USING DELTA
""")

print("  Output tables verified: icd10_mappings, loinc_mappings, umls_mappings")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Load Active Entities (Incremental)
# MAGIC
# MAGIC Load entities from `extracted.merged_entities` that have assertion status = PRESENT,
# MAGIC experiencer = PATIENT, temporality = CURRENT, and are NOT already in the output tables.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW active_entities AS
SELECT
    me.entity_id,
    me.chart_id,
    me.section_id,
    me.entity_type,
    me.entity_text,
    SUBSTRING(me.entity_context, 1, 500) AS entity_context,
    LOWER(TRIM(me.entity_text)) AS entity_text_normalized,
    me.specimen_type,
    me.method,
    me.timing,
    me.value,
    me.unit,
    me.ensemble_confidence
FROM {CATALOG}.extracted.merged_entities me
LEFT JOIN {CATALOG}.extracted.entity_assertions ea
    ON me.entity_id = ea.entity_id AND me.chart_id = ea.chart_id
-- Incremental: skip entities already codified in EITHER output table
LEFT JOIN {CATALOG}.codified.icd10_mappings existing_icd
    ON me.entity_id = existing_icd.entity_id
LEFT JOIN {CATALOG}.codified.loinc_mappings existing_loinc
    ON me.entity_id = existing_loinc.entity_id
WHERE COALESCE(ea.assertion_status, 'PRESENT') = 'PRESENT'
  AND COALESCE(ea.experiencer, 'PATIENT') = 'PATIENT'
  AND COALESCE(ea.temporality, 'CURRENT') = 'CURRENT'
  AND existing_icd.entity_id IS NULL
  AND existing_loinc.entity_id IS NULL
""")

active_count = spark.sql("SELECT COUNT(*) AS cnt FROM active_entities").collect()[0]["cnt"]
print(f"  Active entities to codify (new, PRESENT + PATIENT + CURRENT): {active_count}")

if active_count == 0:
    print("  No new entities to process -- exiting early")
    dbutils.notebook.exit("NO_NEW_ENTITIES")

spark.sql("""
SELECT entity_type, COUNT(*) AS cnt
FROM active_entities
GROUP BY entity_type
ORDER BY cnt DESC
""").show()

# COMMAND ----------

# Split into diagnoses and labs for tier processing
spark.sql("""
CREATE OR REPLACE TEMP VIEW active_diagnoses AS
SELECT * FROM active_entities WHERE entity_type = 'DIAGNOSIS'
""")

spark.sql("""
CREATE OR REPLACE TEMP VIEW active_labs AS
SELECT * FROM active_entities WHERE entity_type = 'LAB_RESULT'
""")

dx_count = spark.sql("SELECT COUNT(*) FROM active_diagnoses").collect()[0][0]
lab_count = spark.sql("SELECT COUNT(*) FROM active_labs").collect()[0][0]
print(f"  Diagnoses: {dx_count}")
print(f"  Lab results: {lab_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Tier 1 — Ontology-Based Coding (NO LLM)
# MAGIC
# MAGIC Pure SQL joins against reference tables. Handles 60-70% of entities at zero LLM cost.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a: Join Diagnoses against Medical Dictionary -> get CUI + source_code

# COMMAND ----------

# Exact match on normalized text
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW tier1_dict_exact AS
SELECT
    e.entity_id,
    e.chart_id,
    e.entity_type,
    e.entity_text,
    e.entity_context,
    e.entity_text_normalized,
    md.cui,
    md.term AS matched_term,
    md.source,
    md.source_code,
    md.entity_type AS dict_entity_type,
    'exact_match' AS match_method,
    0.98 AS match_confidence
FROM active_entities e
INNER JOIN {CATALOG}.reference.medical_dictionary md
    ON e.entity_text_normalized = md.term_normalized
WHERE md.source_code IS NOT NULL
""")

tier1_exact = spark.sql("SELECT COUNT(DISTINCT entity_id) FROM tier1_dict_exact").collect()[0][0]
print(f"  Tier 1 exact dictionary matches: {tier1_exact}")

# COMMAND ----------

# Token-overlap fuzzy match for remaining entities
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW unmapped_after_exact AS
SELECT e.*
FROM active_entities e
LEFT JOIN tier1_dict_exact t1 ON e.entity_id = t1.entity_id
WHERE t1.entity_id IS NULL
""")

unmapped_exact_count = spark.sql("SELECT COUNT(*) FROM unmapped_after_exact").collect()[0][0]
print(f"  Entities remaining after exact match: {unmapped_exact_count}")

# COMMAND ----------

if unmapped_exact_count > 0:
    # Token index from medical_dictionary for fuzzy matching
    spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW dict_token_index AS
    SELECT DISTINCT
        LOWER(token) AS token,
        md.cui,
        md.term AS preferred_name,
        md.source,
        md.source_code,
        md.entity_type AS dict_entity_type
    FROM {CATALOG}.reference.medical_dictionary md
    LATERAL VIEW EXPLODE(SPLIT(LOWER(TRIM(md.term)), '\\\\s+')) AS token
    WHERE md.source_code IS NOT NULL
      AND LENGTH(token) >= 3
    """)

    spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW tier1_dict_fuzzy AS
    WITH entity_tokens AS (
        SELECT entity_id, chart_id, entity_type, entity_text, entity_context,
               entity_text_normalized, LOWER(token) AS token
        FROM unmapped_after_exact
        LATERAL VIEW EXPLODE(SPLIT(entity_text_normalized, '\\\\s+')) AS token
        WHERE LENGTH(token) >= 3
    ),
    token_candidates AS (
        SELECT DISTINCT
            et.entity_id, et.chart_id, et.entity_type, et.entity_text,
            et.entity_context, et.entity_text_normalized,
            di.cui, di.preferred_name, di.source, di.source_code, di.dict_entity_type
        FROM entity_tokens et
        INNER JOIN dict_token_index di ON et.token = di.token
    ),
    scored AS (
        SELECT tc.*,
            LEVENSHTEIN(tc.entity_text_normalized, LOWER(TRIM(tc.preferred_name))) AS edit_distance,
            LENGTH(tc.entity_text_normalized) AS entity_len
        FROM token_candidates tc
        WHERE ABS(LENGTH(tc.entity_text_normalized) - LENGTH(tc.preferred_name)) <= 5
    ),
    filtered AS (
        SELECT * FROM scored
        WHERE edit_distance <= GREATEST(CAST(entity_len * 0.3 AS INT), 3)
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY edit_distance ASC) AS rn
        FROM filtered
    )
    SELECT
        entity_id, chart_id, entity_type, entity_text, entity_context,
        entity_text_normalized, cui, preferred_name AS matched_term,
        source, source_code, dict_entity_type,
        'fuzzy_match' AS match_method,
        ROUND(GREATEST(0.92 - (edit_distance * 0.01), 0.80), 3) AS match_confidence
    FROM ranked
    WHERE rn = 1
    """)

    tier1_fuzzy = spark.sql("SELECT COUNT(DISTINCT entity_id) FROM tier1_dict_fuzzy").collect()[0][0]
    print(f"  Tier 1 fuzzy dictionary matches: {tier1_fuzzy}")
else:
    spark.sql("""
    CREATE OR REPLACE TEMP VIEW tier1_dict_fuzzy AS
    SELECT CAST(NULL AS STRING) AS entity_id, CAST(NULL AS STRING) AS chart_id,
           CAST(NULL AS STRING) AS entity_type, CAST(NULL AS STRING) AS entity_text,
           CAST(NULL AS STRING) AS entity_context, CAST(NULL AS STRING) AS entity_text_normalized,
           CAST(NULL AS STRING) AS cui, CAST(NULL AS STRING) AS matched_term,
           CAST(NULL AS STRING) AS source, CAST(NULL AS STRING) AS source_code,
           CAST(NULL AS STRING) AS dict_entity_type,
           CAST(NULL AS STRING) AS match_method, CAST(NULL AS DOUBLE) AS match_confidence
    WHERE 1=0
    """)
    tier1_fuzzy = 0

# COMMAND ----------

# Combine all dictionary matches
spark.sql("""
CREATE OR REPLACE TEMP VIEW tier1_all_dict AS
SELECT entity_id, chart_id, entity_type, entity_text, entity_context,
       entity_text_normalized, cui, matched_term, source, source_code,
       dict_entity_type, match_method, match_confidence
FROM tier1_dict_exact
UNION ALL
SELECT entity_id, chart_id, entity_type, entity_text, entity_context,
       entity_text_normalized, cui, matched_term, source, source_code,
       dict_entity_type, match_method, match_confidence
FROM tier1_dict_fuzzy
""")

tier1_total = spark.sql("SELECT COUNT(DISTINCT entity_id) FROM tier1_all_dict").collect()[0][0]
print(f"  Tier 1 total dictionary matches: {tier1_total} / {active_count} ({round(tier1_total * 100.0 / max(active_count, 1), 1)}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b: ICD-10 Resolution — Validate via snomed_icd10_map + icd10_codes_full

# COMMAND ----------

# For diagnosis entities with dictionary matches, resolve ICD-10 codes
# Path 1: source_code IS an ICD-10 code (source = 'ICD10') -> validate directly
# Path 2: CUI -> SNOMED -> ICD-10 via snomed_icd10_map
# Path 3: source_code from SNOMED -> snomed_icd10_map

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW tier1_icd10_resolved AS
WITH
-- Path 1: source is already ICD-10
direct_icd10 AS (
    SELECT
        d.entity_id, d.chart_id, d.entity_text, d.entity_context,
        d.cui, d.matched_term, d.match_method, d.match_confidence,
        d.source_code AS icd10_code,
        r.description AS icd10_description,
        r.is_billable,
        'dict_icd10_direct' AS resolution_source,
        d.match_confidence AS tier1_confidence
    FROM tier1_all_dict d
    JOIN {CATALOG}.reference.icd10_codes_full r
        ON UPPER(REPLACE(d.source_code, '.', '')) = r.code_raw
    WHERE d.source = 'ICD10'
      AND d.entity_type = 'DIAGNOSIS'
      AND r.is_billable = true
),

-- Path 2: CUI -> snomed_hierarchy -> snomed_icd10_map
cui_to_snomed_icd10 AS (
    SELECT
        d.entity_id, d.chart_id, d.entity_text, d.entity_context,
        d.cui, d.matched_term, d.match_method, d.match_confidence,
        m.icd10_code,
        m.icd10_name AS icd10_description,
        true AS is_billable,
        'dict_cui_snomed_icd10' AS resolution_source,
        d.match_confidence * 0.95 AS tier1_confidence
    FROM tier1_all_dict d
    JOIN {CATALOG}.reference.medical_dictionary md2
        ON d.cui = md2.cui AND md2.source = 'SNOMED'
    JOIN {CATALOG}.reference.snomed_icd10_map m
        ON md2.source_code = m.snomed_concept_id AND m.map_priority = 1
    WHERE d.entity_type = 'DIAGNOSIS'
      AND d.source != 'ICD10'
      AND d.cui IS NOT NULL
),

-- Path 3: source = SNOMED -> snomed_icd10_map directly
snomed_direct_icd10 AS (
    SELECT
        d.entity_id, d.chart_id, d.entity_text, d.entity_context,
        d.cui, d.matched_term, d.match_method, d.match_confidence,
        m.icd10_code,
        m.icd10_name AS icd10_description,
        true AS is_billable,
        'dict_snomed_direct_icd10' AS resolution_source,
        d.match_confidence * 0.96 AS tier1_confidence
    FROM tier1_all_dict d
    JOIN {CATALOG}.reference.snomed_icd10_map m
        ON d.source_code = m.snomed_concept_id AND m.map_priority = 1
    WHERE d.source = 'SNOMED'
      AND d.entity_type = 'DIAGNOSIS'
),

-- Combine all paths, deduplicate by entity_id (take highest confidence)
combined AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY tier1_confidence DESC) AS rn
    FROM (
        SELECT * FROM direct_icd10
        UNION ALL
        SELECT * FROM cui_to_snomed_icd10
        UNION ALL
        SELECT * FROM snomed_direct_icd10
    )
)
SELECT entity_id, chart_id, entity_text, entity_context,
       cui, matched_term, match_method, match_confidence,
       icd10_code, icd10_description, is_billable,
       resolution_source, tier1_confidence
FROM combined
WHERE rn = 1
""")

tier1_icd10_count = spark.sql("SELECT COUNT(DISTINCT entity_id) FROM tier1_icd10_resolved").collect()[0][0]
print(f"  Tier 1 ICD-10 codes resolved: {tier1_icd10_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c: LOINC Resolution — Join against loinc_search / loinc_codes_full

# COMMAND ----------

# Determine LOINC reference table availability
LOINC_TABLE = None
for table_name in [f"{CATALOG}.reference.loinc_codes_full", f"{CATALOG}.reference.loinc_codes"]:
    try:
        cnt = spark.table(table_name).limit(1).count()
        LOINC_TABLE = table_name
        print(f"  Using LOINC reference: {table_name}")
        break
    except Exception:
        continue

if LOINC_TABLE is None:
    print("  WARNING: No LOINC reference table found -- LOINC Tier 1 will be skipped")

# COMMAND ----------

if LOINC_TABLE:
    # For LAB_RESULT entities with dictionary matches, resolve LOINC codes
    # Path 1: source = 'LOINC' -> validate code in reference table
    # Path 2: component-based equi-join on first word
    spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW tier1_loinc_resolved AS
    WITH
    -- Path 1: source is already LOINC
    direct_loinc AS (
        SELECT
            d.entity_id, d.chart_id, d.entity_text, d.entity_context,
            d.cui, d.matched_term, d.match_method, d.match_confidence,
            d.source_code AS loinc_code,
            l.long_name AS loinc_long_name,
            'dict_loinc_direct' AS resolution_source,
            d.match_confidence AS tier1_confidence
        FROM tier1_all_dict d
        JOIN {LOINC_TABLE} l ON d.source_code = l.loinc_code
        WHERE d.source = 'LOINC'
          AND d.entity_type = 'LAB_RESULT'
    ),

    -- Path 2: first-word component match for lab entities without LOINC source
    lab_entities AS (
        SELECT
            e.entity_id, e.chart_id, e.entity_text, e.entity_context,
            e.entity_text_normalized,
            e.specimen_type, e.method, e.timing,
            LOWER(SPLIT(e.entity_text, ' ')[0]) AS entity_first_word
        FROM active_labs e
        LEFT JOIN direct_loinc dl ON e.entity_id = dl.entity_id
        WHERE dl.entity_id IS NULL
    ),
    loinc_with_first_word AS (
        SELECT l.*, LOWER(SPLIT(l.component, ' ')[0]) AS component_first_word
        FROM {LOINC_TABLE} l
        WHERE l.component IS NOT NULL
    ),
    component_matches AS (
        SELECT
            le.entity_id, le.chart_id, le.entity_text, le.entity_context,
            le.specimen_type, le.method, le.timing,
            l.loinc_code, l.long_name AS loinc_long_name,
            -- Score based on LOINC axis matching
            CASE WHEN LOWER(l.long_name) LIKE CONCAT('%', LOWER(le.entity_text), '%') THEN 3 ELSE 0 END
            + CASE WHEN le.specimen_type IS NOT NULL AND LOWER(l.long_name) LIKE CONCAT('%', LOWER(le.specimen_type), '%') THEN 2 ELSE 0 END
            + CASE WHEN le.method IS NOT NULL AND LOWER(l.long_name) LIKE CONCAT('%', LOWER(le.method), '%') THEN 2 ELSE 0 END
            + CASE WHEN le.timing IS NOT NULL AND LOWER(l.long_name) LIKE CONCAT('%', LOWER(le.timing), '%') THEN 1 ELSE 0 END
            AS axis_match_score
        FROM lab_entities le
        JOIN loinc_with_first_word l ON le.entity_first_word = l.component_first_word
        WHERE (
            LOWER(l.long_name) LIKE CONCAT('%', LOWER(le.entity_text), '%')
            OR (le.specimen_type IS NOT NULL AND LOWER(l.long_name) LIKE CONCAT('%', LOWER(le.specimen_type), '%'))
            OR (le.method IS NOT NULL AND LOWER(l.long_name) LIKE CONCAT('%', LOWER(le.method), '%'))
        )
    ),
    component_ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY axis_match_score DESC) AS rn
        FROM component_matches
        WHERE axis_match_score > 0
    ),
    component_best AS (
        SELECT
            entity_id, chart_id, entity_text, entity_context,
            CAST(NULL AS STRING) AS cui, CAST(NULL AS STRING) AS matched_term,
            'component_match' AS match_method,
            ROUND(0.88 + (axis_match_score * 0.015), 3) AS match_confidence,
            loinc_code, loinc_long_name,
            'component_axis_match' AS resolution_source,
            ROUND(0.88 + (axis_match_score * 0.015), 3) AS tier1_confidence
        FROM component_ranked WHERE rn = 1
    ),

    -- Combine paths
    combined AS (
        SELECT entity_id, chart_id, entity_text, entity_context,
               cui, matched_term, match_method, match_confidence,
               loinc_code, loinc_long_name, resolution_source, tier1_confidence
        FROM direct_loinc
        UNION ALL
        SELECT * FROM component_best
    ),
    deduped AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY tier1_confidence DESC) AS rn
        FROM combined
    )
    SELECT entity_id, chart_id, entity_text, entity_context,
           cui, matched_term, match_method, match_confidence,
           loinc_code, loinc_long_name, resolution_source, tier1_confidence
    FROM deduped WHERE rn = 1
    """)

    tier1_loinc_count = spark.sql("SELECT COUNT(DISTINCT entity_id) FROM tier1_loinc_resolved").collect()[0][0]
    print(f"  Tier 1 LOINC codes resolved: {tier1_loinc_count}")
else:
    spark.sql("""
    CREATE OR REPLACE TEMP VIEW tier1_loinc_resolved AS
    SELECT CAST(NULL AS STRING) AS entity_id, CAST(NULL AS STRING) AS chart_id,
           CAST(NULL AS STRING) AS entity_text, CAST(NULL AS STRING) AS entity_context,
           CAST(NULL AS STRING) AS cui, CAST(NULL AS STRING) AS matched_term,
           CAST(NULL AS STRING) AS match_method, CAST(NULL AS DOUBLE) AS match_confidence,
           CAST(NULL AS STRING) AS loinc_code, CAST(NULL AS STRING) AS loinc_long_name,
           CAST(NULL AS STRING) AS resolution_source, CAST(NULL AS DOUBLE) AS tier1_confidence
    WHERE 1=0
    """)
    tier1_loinc_count = 0

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2d: Identify Entities NOT Resolved by Tier 1

# COMMAND ----------

# Diagnoses not resolved by Tier 1
spark.sql("""
CREATE OR REPLACE TEMP VIEW unresolved_diagnoses AS
SELECT e.*
FROM active_diagnoses e
LEFT JOIN tier1_icd10_resolved t1 ON e.entity_id = t1.entity_id
WHERE t1.entity_id IS NULL
""")

# Labs not resolved by Tier 1
spark.sql("""
CREATE OR REPLACE TEMP VIEW unresolved_labs AS
SELECT e.*
FROM active_labs e
LEFT JOIN tier1_loinc_resolved t1 ON e.entity_id = t1.entity_id
WHERE t1.entity_id IS NULL
""")

unresolved_dx = spark.sql("SELECT COUNT(*) FROM unresolved_diagnoses").collect()[0][0]
unresolved_lab = spark.sql("SELECT COUNT(*) FROM unresolved_labs").collect()[0][0]
print(f"  Unresolved diagnoses for Tier 2 (LLM): {unresolved_dx}")
print(f"  Unresolved labs for Tier 2 (LLM): {unresolved_lab}")

# Combine all unresolved into a single view
spark.sql("""
CREATE OR REPLACE TEMP VIEW unresolved_all AS
SELECT entity_id, chart_id, entity_type, entity_text, entity_context,
       specimen_type, method, timing, value, unit
FROM unresolved_diagnoses
UNION ALL
SELECT entity_id, chart_id, entity_type, entity_text, entity_context,
       specimen_type, method, timing, value, unit
FROM unresolved_labs
""")

unresolved_total = spark.sql("SELECT COUNT(*) FROM unresolved_all").collect()[0][0]
print(f"  Total unresolved for Tier 2: {unresolved_total}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Tier 2 — LLM-Based Coding (One Call Per Chart)
# MAGIC
# MAGIC For all entities NOT resolved by Tier 1, group by chart_id and send a single
# MAGIC `ai_query` call per chart containing ALL unresolved entities (diagnoses + labs).

# COMMAND ----------

if unresolved_total > 0:
    # Build the prompt payload: one row per chart with all unresolved entities concatenated
    spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW tier2_chart_prompts AS
    SELECT
        chart_id,
        CONCAT_WS('\n',
            COLLECT_LIST(
                CONCAT(
                    '- entity_id: ', entity_id,
                    ' | type: ', entity_type,
                    ' | text: ', entity_text,
                    CASE WHEN entity_type = 'LAB_RESULT' AND specimen_type IS NOT NULL
                        THEN CONCAT(' | specimen: ', specimen_type) ELSE '' END,
                    CASE WHEN entity_type = 'LAB_RESULT' AND method IS NOT NULL
                        THEN CONCAT(' | method: ', method) ELSE '' END,
                    CASE WHEN entity_type = 'LAB_RESULT' AND timing IS NOT NULL
                        THEN CONCAT(' | timing: ', timing) ELSE '' END,
                    ' | context: ', COALESCE(SUBSTRING(entity_context, 1, 200), 'none')
                )
            )
        ) AS entities_to_code,
        COLLECT_LIST(entity_id) AS entity_ids,
        COLLECT_LIST(entity_type) AS entity_types,
        COUNT(*) AS entity_count
    FROM unresolved_all
    GROUP BY chart_id
    """)

    chart_count = spark.sql("SELECT COUNT(*) FROM tier2_chart_prompts").collect()[0][0]
    print(f"  Tier 2: {unresolved_total} entities across {chart_count} charts (one LLM call per chart)")

    # Single ai_query per chart — MATERIALIZE results to avoid re-execution
    # CRITICAL: Using a table (not a view) so that ai_query runs exactly once.
    # A TEMP VIEW would re-execute all LLM calls on every downstream query.
    spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.codified._tmp_tier2_llm_results AS
    SELECT
        chart_id,
        entity_ids,
        entity_types,
        entity_count,
        ai_query(
            '{MODEL}',
            CONCAT(
                'You are a certified medical coder (CPC) and LOINC mapper. ',
                'Assign the most specific billable ICD-10-CM code to each diagnosis ',
                'and the most accurate LOINC code to each lab result below. ',
                'For each entity, provide:\n',
                '- "entity_id": the provided entity ID\n',
                '- "entity_type": DIAGNOSIS or LAB_RESULT\n',
                '- "code": the ICD-10-CM or LOINC code\n',
                '- "code_description": description of the assigned code\n',
                '- "confidence": 0.0-1.0\n',
                '- "reasoning": brief justification for code selection\n',
                '- "is_specific": true if code is maximally specific (not .9 unspecified)\n\n',
                'IMPORTANT CODING RULES:\n',
                '- Always prefer the most specific billable code over unspecified (.9) codes\n',
                '- Consider clinical qualifiers: type (1 vs 2), laterality, with/without complications\n',
                '- For labs: match specimen type, method, and timing to LOINC axes\n',
                '- Return ONLY the JSON array, no markdown fences\n\n',
                'ENTITIES TO CODE:\n',
                entities_to_code
            )
        ) AS coding_result
    FROM tier2_chart_prompts
    """)
    spark.sql(f"CREATE OR REPLACE TEMP VIEW tier2_llm_results AS SELECT * FROM {CATALOG}.codified._tmp_tier2_llm_results")

    print(f"  Tier 2 LLM results materialized: {chart_count} charts")
else:
    spark.sql("""
    CREATE OR REPLACE TEMP VIEW tier2_llm_results AS
    SELECT CAST(NULL AS STRING) AS chart_id,
           CAST(NULL AS ARRAY<STRING>) AS entity_ids,
           CAST(NULL AS ARRAY<STRING>) AS entity_types,
           CAST(NULL AS INT) AS entity_count,
           CAST(NULL AS STRING) AS coding_result
    WHERE 1=0
    """)
    print("  Tier 2: No unresolved entities -- LLM calls skipped")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parse Tier 2 LLM Results

# COMMAND ----------

from pyspark.sql.functions import (
    col, coalesce, get_json_object, regexp_replace, lit, current_timestamp,
    concat, posexplode, expr, when, upper, trim
)

if unresolved_total > 0:
    tier2_raw = spark.table("tier2_llm_results")

    # Clean markdown fences, explode entity_ids with position index
    tier2_parsed = (tier2_raw
        .withColumn("_clean", regexp_replace(regexp_replace(col("coding_result"), '```json\\n?', ''), '\\n?```', ''))
        .select("chart_id", "_clean",
                posexplode(col("entity_ids")).alias("pos", "entity_id"),
                col("entity_types"))
        .withColumn("entity_type", expr("entity_types[pos]"))
        .withColumn("item_path", concat(lit("$["), col("pos").cast("string"), lit("]")))
        .withColumn("code", get_json_object("_clean", concat(col("item_path"), lit(".code"))))
        .withColumn("code_description", get_json_object("_clean", concat(col("item_path"), lit(".code_description"))))
        .withColumn("llm_confidence",
            coalesce(get_json_object("_clean", concat(col("item_path"), lit(".confidence"))).cast("double"), lit(0.75)))
        .withColumn("reasoning", get_json_object("_clean", concat(col("item_path"), lit(".reasoning"))))
        .withColumn("is_specific_str", get_json_object("_clean", concat(col("item_path"), lit(".is_specific"))))
        .drop("_clean", "entity_types", "item_path", "pos")
    )
    tier2_parsed.createOrReplaceTempView("tier2_parsed")

    tier2_entity_count = spark.sql("SELECT COUNT(*) FROM tier2_parsed WHERE code IS NOT NULL").collect()[0][0]
    print(f"  Tier 2 parsed results: {tier2_entity_count} entities with codes")
else:
    spark.sql("""
    CREATE OR REPLACE TEMP VIEW tier2_parsed AS
    SELECT CAST(NULL AS STRING) AS chart_id, CAST(NULL AS STRING) AS entity_id,
           CAST(NULL AS STRING) AS entity_type, CAST(NULL AS STRING) AS code,
           CAST(NULL AS STRING) AS code_description, CAST(NULL AS DOUBLE) AS llm_confidence,
           CAST(NULL AS STRING) AS reasoning, CAST(NULL AS STRING) AS is_specific_str
    WHERE 1=0
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3d: Validate LLM-Suggested Codes Against Reference Tables

# COMMAND ----------

if unresolved_total > 0:
    # Validate ICD-10 codes from LLM
    spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW tier2_icd10_validated AS
    SELECT
        t2.entity_id,
        t2.chart_id,
        e.entity_text,
        e.entity_context,
        t2.code AS llm_code,
        t2.code_description AS llm_description,
        t2.llm_confidence,
        t2.reasoning,
        t2.is_specific_str,
        r.description AS ref_description,
        r.is_billable,
        CASE
            WHEN r.code IS NOT NULL AND r.is_billable = true THEN t2.code
            WHEN r.code IS NOT NULL THEN t2.code
            ELSE t2.code
        END AS final_code,
        CASE
            WHEN r.code IS NOT NULL AND r.is_billable = true THEN COALESCE(r.description, t2.code_description)
            WHEN r.code IS NOT NULL THEN COALESCE(r.description, t2.code_description)
            ELSE t2.code_description
        END AS final_description,
        CASE
            WHEN r.code IS NOT NULL AND r.is_billable = true THEN t2.llm_confidence
            WHEN r.code IS NOT NULL THEN GREATEST(t2.llm_confidence - 0.05, 0.60)
            ELSE 0.60  -- Code not in reference, penalize confidence
        END AS final_confidence,
        CASE WHEN r.code IS NOT NULL THEN true ELSE false END AS code_validated
    FROM tier2_parsed t2
    JOIN unresolved_diagnoses e ON t2.entity_id = e.entity_id
    LEFT JOIN {CATALOG}.reference.icd10_codes_full r
        ON UPPER(REPLACE(t2.code, '.', '')) = r.code_raw
    WHERE t2.entity_type = 'DIAGNOSIS'
      AND t2.code IS NOT NULL
    """)

    tier2_icd10_count = spark.sql("SELECT COUNT(*) FROM tier2_icd10_validated").collect()[0][0]
    tier2_icd10_valid = spark.sql("SELECT COUNT(*) FROM tier2_icd10_validated WHERE code_validated = true").collect()[0][0]
    print(f"  Tier 2 ICD-10: {tier2_icd10_count} codes, {tier2_icd10_valid} validated against reference")

    # Validate LOINC codes from LLM
    loinc_join = ""
    if LOINC_TABLE:
        loinc_join = f"""
        LEFT JOIN {LOINC_TABLE} lr ON t2.code = lr.loinc_code
        """
        loinc_valid_expr = "CASE WHEN lr.loinc_code IS NOT NULL THEN true ELSE false END"
        loinc_conf_expr = f"""
        CASE
            WHEN lr.loinc_code IS NOT NULL THEN t2.llm_confidence
            ELSE 0.60
        END
        """
        loinc_desc_expr = "COALESCE(lr.long_name, t2.code_description)"
    else:
        loinc_join = ""
        loinc_valid_expr = "false"
        loinc_conf_expr = "t2.llm_confidence"
        loinc_desc_expr = "t2.code_description"

    spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW tier2_loinc_validated AS
    SELECT
        t2.entity_id,
        t2.chart_id,
        e.entity_text,
        e.entity_context,
        e.specimen_type,
        e.method,
        e.timing,
        t2.code AS loinc_code,
        {loinc_desc_expr} AS loinc_long_name,
        t2.llm_confidence,
        t2.reasoning,
        {loinc_conf_expr} AS final_confidence,
        {loinc_valid_expr} AS code_validated
    FROM tier2_parsed t2
    JOIN unresolved_labs e ON t2.entity_id = e.entity_id
    {loinc_join}
    WHERE t2.entity_type = 'LAB_RESULT'
      AND t2.code IS NOT NULL
    """)

    tier2_loinc_count = spark.sql("SELECT COUNT(*) FROM tier2_loinc_validated").collect()[0][0]
    print(f"  Tier 2 LOINC: {tier2_loinc_count} codes")
else:
    spark.sql("""
    CREATE OR REPLACE TEMP VIEW tier2_icd10_validated AS
    SELECT CAST(NULL AS STRING) AS entity_id, CAST(NULL AS STRING) AS chart_id,
           CAST(NULL AS STRING) AS entity_text, CAST(NULL AS STRING) AS entity_context,
           CAST(NULL AS STRING) AS llm_code, CAST(NULL AS STRING) AS llm_description,
           CAST(NULL AS DOUBLE) AS llm_confidence, CAST(NULL AS STRING) AS reasoning,
           CAST(NULL AS STRING) AS is_specific_str,
           CAST(NULL AS STRING) AS ref_description, CAST(NULL AS BOOLEAN) AS is_billable,
           CAST(NULL AS STRING) AS final_code, CAST(NULL AS STRING) AS final_description,
           CAST(NULL AS DOUBLE) AS final_confidence, CAST(NULL AS BOOLEAN) AS code_validated
    WHERE 1=0
    """)
    spark.sql("""
    CREATE OR REPLACE TEMP VIEW tier2_loinc_validated AS
    SELECT CAST(NULL AS STRING) AS entity_id, CAST(NULL AS STRING) AS chart_id,
           CAST(NULL AS STRING) AS entity_text, CAST(NULL AS STRING) AS entity_context,
           CAST(NULL AS STRING) AS specimen_type, CAST(NULL AS STRING) AS method,
           CAST(NULL AS STRING) AS timing,
           CAST(NULL AS STRING) AS loinc_code, CAST(NULL AS STRING) AS loinc_long_name,
           CAST(NULL AS DOUBLE) AS llm_confidence, CAST(NULL AS STRING) AS reasoning,
           CAST(NULL AS DOUBLE) AS final_confidence, CAST(NULL AS BOOLEAN) AS code_validated
    WHERE 1=0
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Assemble Final ICD-10 Mappings
# MAGIC
# MAGIC Combine Tier 1 (ontology) + Tier 2 (LLM) results into `codified.icd10_mappings`.

# COMMAND ----------

spark.sql(f"""
INSERT INTO {CATALOG}.codified.icd10_mappings
-- Tier 1: Ontology-resolved diagnoses
SELECT
    CONCAT('MAP-ICD-', uuid()) AS mapping_id,
    t1.entity_id,
    t1.chart_id,
    t1.entity_text,
    t1.icd10_code,
    t1.icd10_description,
    t1.tier1_confidence AS confidence,
    'ONTOLOGY_DIRECT' AS resolution_path,
    t1.icd10_code AS r1_code,
    CONCAT('Matched via reference.medical_dictionary (', t1.match_method, ') -> ',
           t1.resolution_source,
           CASE WHEN t1.cui IS NOT NULL THEN CONCAT(', CUI: ', t1.cui) ELSE '' END) AS r1_reasoning,
    CAST(NULL AS STRING) AS r2_verdict,
    CAST(NULL AS STRING) AS r2_code,
    CAST(NULL AS STRING) AS r2_reasoning,
    CAST(NULL AS STRING) AS arbiter_code,
    CAST(NULL AS STRING) AS arbiter_reasoning,
    CASE WHEN t1.icd10_code LIKE '%.9%' OR LENGTH(REPLACE(t1.icd10_code, '.', '')) < 5 THEN false ELSE true END AS is_specific,
    current_timestamp() AS codified_at
FROM tier1_icd10_resolved t1

UNION ALL

-- Tier 2: LLM-coded diagnoses
SELECT
    CONCAT('MAP-ICD-', uuid()) AS mapping_id,
    t2.entity_id,
    t2.chart_id,
    t2.entity_text,
    t2.final_code AS icd10_code,
    t2.final_description AS icd10_description,
    t2.final_confidence AS confidence,
    'LLM_ASSIGNED' AS resolution_path,
    t2.final_code AS r1_code,
    COALESCE(t2.reasoning, '') AS r1_reasoning,
    CAST(NULL AS STRING) AS r2_verdict,
    CAST(NULL AS STRING) AS r2_code,
    CAST(NULL AS STRING) AS r2_reasoning,
    CAST(NULL AS STRING) AS arbiter_code,
    CAST(NULL AS STRING) AS arbiter_reasoning,
    CASE
        WHEN LOWER(t2.is_specific_str) = 'true' THEN true
        WHEN t2.final_code LIKE '%.9%' OR LENGTH(REPLACE(t2.final_code, '.', '')) < 5 THEN false
        ELSE true
    END AS is_specific,
    current_timestamp() AS codified_at
FROM tier2_icd10_validated t2
""")

new_icd10 = spark.sql(f"""
    SELECT COUNT(*) FROM (
        SELECT entity_id FROM tier1_icd10_resolved
        UNION ALL
        SELECT entity_id FROM tier2_icd10_validated
    )
""").collect()[0][0]
total_icd10 = spark.table(f"{CATALOG}.codified.icd10_mappings").count()
print(f"  ICD-10 mappings written this run: {new_icd10}")
print(f"  ICD-10 mappings total: {total_icd10}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5: Assemble Final LOINC Mappings
# MAGIC
# MAGIC Combine Tier 1 (ontology) + Tier 2 (LLM) results into `codified.loinc_mappings`.

# COMMAND ----------

spark.sql(f"""
INSERT INTO {CATALOG}.codified.loinc_mappings
-- Tier 1: Ontology-resolved labs
SELECT
    CONCAT('MAP-LOI-', uuid()) AS mapping_id,
    t1.entity_id,
    t1.chart_id,
    t1.entity_text,
    t1.loinc_code,
    t1.loinc_long_name,
    t1.tier1_confidence AS confidence,
    'ONTOLOGY_DIRECT' AS resolution_path,
    t1.loinc_code AS r1_code,
    CONCAT('Matched via ', t1.resolution_source, ' (',  t1.match_method, ')') AS r1_reasoning,
    CAST(NULL AS STRING) AS r2_verdict,
    CAST(NULL AS STRING) AS r2_code,
    CAST(NULL AS STRING) AS r2_reasoning,
    CAST(NULL AS STRING) AS arbiter_code,
    CAST(NULL AS STRING) AS arbiter_reasoning,
    me.specimen_type,
    me.method,
    me.timing,
    current_timestamp() AS codified_at
FROM tier1_loinc_resolved t1
JOIN {CATALOG}.extracted.merged_entities me ON t1.entity_id = me.entity_id

UNION ALL

-- Tier 2: LLM-coded labs
SELECT
    CONCAT('MAP-LOI-', uuid()) AS mapping_id,
    t2.entity_id,
    t2.chart_id,
    t2.entity_text,
    t2.loinc_code,
    t2.loinc_long_name,
    t2.final_confidence AS confidence,
    'LLM_ASSIGNED' AS resolution_path,
    t2.loinc_code AS r1_code,
    COALESCE(t2.reasoning, '') AS r1_reasoning,
    CAST(NULL AS STRING) AS r2_verdict,
    CAST(NULL AS STRING) AS r2_code,
    CAST(NULL AS STRING) AS r2_reasoning,
    CAST(NULL AS STRING) AS arbiter_code,
    CAST(NULL AS STRING) AS arbiter_reasoning,
    t2.specimen_type,
    t2.method,
    t2.timing,
    current_timestamp() AS codified_at
FROM tier2_loinc_validated t2
""")

new_loinc = spark.sql(f"""
    SELECT COUNT(*) FROM (
        SELECT entity_id FROM tier1_loinc_resolved
        UNION ALL
        SELECT entity_id FROM tier2_loinc_validated
    )
""").collect()[0][0]
total_loinc = spark.table(f"{CATALOG}.codified.loinc_mappings").count()
print(f"  LOINC mappings written this run: {new_loinc}")
print(f"  LOINC mappings total: {total_loinc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5b: Write UMLS Mappings (From Dictionary Matches)
# MAGIC
# MAGIC For entities that matched in the medical_dictionary and have a CUI,
# MAGIC write UMLS mappings. Also resolve SNOMED concept IDs where available.

# COMMAND ----------

spark.sql(f"""
INSERT INTO {CATALOG}.codified.umls_mappings
SELECT
    CONCAT('UMLS-', uuid()) AS mapping_id,
    d.entity_id,
    d.chart_id,
    d.entity_text,
    d.cui,
    d.matched_term AS preferred_name,
    CAST(NULL AS STRING) AS semantic_type,
    -- Resolve SNOMED concept_id if source is SNOMED
    COALESCE(
        CASE WHEN d.source = 'SNOMED' THEN d.source_code ELSE NULL END,
        sh.concept_id
    ) AS snomed_concept_id,
    sh.concept_name AS snomed_name,
    d.match_method AS mapping_method,
    d.match_confidence AS confidence,
    current_timestamp() AS mapped_at
FROM tier1_all_dict d
LEFT JOIN {CATALOG}.reference.snomed_hierarchy sh
    ON d.source = 'SNOMED' AND d.source_code = sh.concept_id
LEFT JOIN {CATALOG}.codified.umls_mappings existing
    ON d.entity_id = existing.entity_id
WHERE d.cui IS NOT NULL
  AND existing.entity_id IS NULL
""")

new_umls = spark.sql(f"""
    SELECT COUNT(DISTINCT entity_id) FROM tier1_all_dict WHERE cui IS NOT NULL
""").collect()[0][0]
total_umls = spark.table(f"{CATALOG}.codified.umls_mappings").count()
print(f"  UMLS mappings written this run: {new_umls}")
print(f"  UMLS mappings total: {total_umls}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6: Summary Statistics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resolution Path Distribution

# COMMAND ----------

print("=== ICD-10 Resolution Paths ===")
spark.sql(f"""
SELECT
    resolution_path,
    COUNT(*) AS count,
    ROUND(AVG(confidence), 3) AS avg_confidence,
    ROUND(MIN(confidence), 3) AS min_confidence,
    ROUND(MAX(confidence), 3) AS max_confidence,
    SUM(CASE WHEN is_specific THEN 1 ELSE 0 END) AS specific_count
FROM {CATALOG}.codified.icd10_mappings
GROUP BY resolution_path
ORDER BY count DESC
""").show(truncate=False)

# COMMAND ----------

print("=== LOINC Resolution Paths ===")
spark.sql(f"""
SELECT
    resolution_path,
    COUNT(*) AS count,
    ROUND(AVG(confidence), 3) AS avg_confidence,
    ROUND(MIN(confidence), 3) AS min_confidence,
    ROUND(MAX(confidence), 3) AS max_confidence
FROM {CATALOG}.codified.loinc_mappings
GROUP BY resolution_path
ORDER BY count DESC
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Coverage Summary

# COMMAND ----------

print("=== Overall Coverage ===")
spark.sql(f"""
WITH active AS (
    SELECT me.entity_id, me.entity_type
    FROM {CATALOG}.extracted.merged_entities me
    LEFT JOIN {CATALOG}.extracted.entity_assertions ea
        ON me.entity_id = ea.entity_id AND me.chart_id = ea.chart_id
    WHERE COALESCE(ea.assertion_status, 'PRESENT') = 'PRESENT'
)
SELECT
    a.entity_type,
    COUNT(DISTINCT a.entity_id) AS total_entities,
    COUNT(DISTINCT icd.entity_id) AS icd10_coded,
    COUNT(DISTINCT loinc.entity_id) AS loinc_coded,
    COUNT(DISTINCT umls.entity_id) AS umls_mapped,
    ROUND(
        GREATEST(COUNT(DISTINCT icd.entity_id), COUNT(DISTINCT loinc.entity_id)) * 100.0
        / GREATEST(COUNT(DISTINCT a.entity_id), 1), 1
    ) AS coding_pct
FROM active a
LEFT JOIN {CATALOG}.codified.icd10_mappings icd ON a.entity_id = icd.entity_id
LEFT JOIN {CATALOG}.codified.loinc_mappings loinc ON a.entity_id = loinc.entity_id
LEFT JOIN {CATALOG}.codified.umls_mappings umls ON a.entity_id = umls.entity_id
GROUP BY a.entity_type
ORDER BY total_entities DESC
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tier Performance

# COMMAND ----------

tier1_icd10_final = spark.sql(f"SELECT COUNT(*) FROM {CATALOG}.codified.icd10_mappings WHERE resolution_path = 'ONTOLOGY_DIRECT'").collect()[0][0]
tier2_icd10_final = spark.sql(f"SELECT COUNT(*) FROM {CATALOG}.codified.icd10_mappings WHERE resolution_path = 'LLM_ASSIGNED'").collect()[0][0]
tier1_loinc_final = spark.sql(f"SELECT COUNT(*) FROM {CATALOG}.codified.loinc_mappings WHERE resolution_path = 'ONTOLOGY_DIRECT'").collect()[0][0]
tier2_loinc_final = spark.sql(f"SELECT COUNT(*) FROM {CATALOG}.codified.loinc_mappings WHERE resolution_path = 'LLM_ASSIGNED'").collect()[0][0]

total_coded = tier1_icd10_final + tier2_icd10_final + tier1_loinc_final + tier2_loinc_final
tier1_total_final = tier1_icd10_final + tier1_loinc_final
tier2_total_final = tier2_icd10_final + tier2_loinc_final

print("=== Tier Performance ===")
print(f"  Tier 1 (Ontology, 0 LLM calls): {tier1_total_final} codes ({round(tier1_total_final * 100.0 / max(total_coded, 1), 1)}%)")
print(f"    ICD-10: {tier1_icd10_final}")
print(f"    LOINC:  {tier1_loinc_final}")
print(f"  Tier 2 (LLM, ~1 call/chart):    {tier2_total_final} codes ({round(tier2_total_final * 100.0 / max(total_coded, 1), 1)}%)")
print(f"    ICD-10: {tier2_icd10_final}")
print(f"    LOINC:  {tier2_loinc_final}")
print(f"  Total coded: {total_coded}")

# Old pipeline estimate: ~455 ai_query calls across 4 notebooks
# New pipeline: chart_count calls (typically ~100 for 100 charts)
if unresolved_total > 0:
    print(f"\n  LLM calls this run: {chart_count}")
    print(f"  Estimated old pipeline calls: ~455")
    print(f"  Reduction: ~{round((1 - chart_count / 455) * 100)}%")
else:
    print(f"\n  LLM calls this run: 0 (all resolved by ontology)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confidence Distribution

# COMMAND ----------

print("=== ICD-10 Confidence Distribution ===")
spark.sql(f"""
SELECT
    CASE
        WHEN confidence >= 0.95 THEN '0.95+'
        WHEN confidence >= 0.90 THEN '0.90-0.94'
        WHEN confidence >= 0.80 THEN '0.80-0.89'
        WHEN confidence >= 0.70 THEN '0.70-0.79'
        ELSE '<0.70'
    END AS confidence_bucket,
    COUNT(*) AS count,
    resolution_path
FROM {CATALOG}.codified.icd10_mappings
GROUP BY 1, resolution_path
ORDER BY confidence_bucket DESC, count DESC
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Mappings

# COMMAND ----------

print("=== Sample ICD-10 Mappings ===")
spark.sql(f"""
SELECT entity_text, icd10_code, icd10_description, resolution_path,
       ROUND(confidence, 3) AS confidence, is_specific
FROM {CATALOG}.codified.icd10_mappings
ORDER BY codified_at DESC
LIMIT 15
""").show(truncate=False)

# COMMAND ----------

print("=== Sample LOINC Mappings ===")
spark.sql(f"""
SELECT entity_text, loinc_code, loinc_long_name, resolution_path,
       ROUND(confidence, 3) AS confidence, specimen_type, method
FROM {CATALOG}.codified.loinc_mappings
ORDER BY codified_at DESC
LIMIT 15
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup Temp Tables

# COMMAND ----------

for tmp in [f"{CATALOG}.codified._tmp_tier2_llm_results"]:
    try:
        spark.sql(f"DROP TABLE IF EXISTS {tmp}")
        print(f"  Cleaned up {tmp}")
    except Exception:
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Next:** Run `06_sync_to_app` to publish codified data to the review application.
