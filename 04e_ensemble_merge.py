# Databricks notebook source
# MAGIC %md
# MAGIC # 04e — Ensemble Entity Merger (Multi-Layer Fusion)
# MAGIC
# MAGIC Merges entities from all three extraction layers (dictionary, NER, LLM) into a
# MAGIC unified entity set with source-aware confidence scoring.
# MAGIC
# MAGIC **Scalability fixes (v2):**
# MAGIC - SQL window functions instead of multiple self-joins for dedup
# MAGIC - Correct column references matching 01_setup.py schemas exactly
# MAGIC - Empty table handling: proceeds even if some source tables are empty
# MAGIC - Incremental processing: only merges charts not yet in `merged_entities`
# MAGIC
# MAGIC **Deduplication strategy:**
# MAGIC 1. Exact text match across layers (GROUP BY chart_id + normalized_text + entity_type)
# MAGIC 2. Substring containment (keep more specific, record both sources)
# MAGIC 3. CUI-based dedup (same UMLS concept regardless of text)
# MAGIC
# MAGIC **Confidence tiers:**
# MAGIC - All 3 layers: 0.95-0.98
# MAGIC - 2 layers: 0.85-0.92
# MAGIC - Dictionary only: ~0.80
# MAGIC - NER only: ~0.75
# MAGIC - LLM only: 0.65-0.75
# MAGIC
# MAGIC **Input:** `extracted.dictionary_entities`, `extracted.ner_entities`, `extracted.llm_entities`
# MAGIC **Output:** `extracted.merged_entities`

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
from pyspark.sql.types import StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0: Empty Table Guard & Source Counts

# COMMAND ----------

dict_count = spark.table(f"{CATALOG}.extracted.dictionary_entities").count()
ner_count = spark.table(f"{CATALOG}.extracted.ner_entities").count()
llm_count = spark.table(f"{CATALOG}.extracted.llm_entities").count()
print(f"  Dictionary entities: {dict_count}")
print(f"  NER entities: {ner_count}")
print(f"  LLM entities: {llm_count}")
print(f"  Total before merge: {dict_count + ner_count + llm_count}")

if dict_count + ner_count + llm_count == 0:
    print("  All source tables are empty -- creating empty merged_entities table and exiting")
    spark.sql(f"""
        CREATE OR REPLACE TABLE {CATALOG}.extracted.merged_entities (
            entity_id STRING,
            chart_id STRING,
            section_id STRING,
            entity_type STRING,
            entity_text STRING,
            entity_context STRING,
            sources ARRAY<STRING>,
            source_entity_ids ARRAY<STRING>,
            ensemble_confidence DOUBLE,
            match_count INT,
            specimen_type STRING,
            method STRING,
            timing STRING,
            value STRING,
            unit STRING,
            start_offset INT,
            end_offset INT,
            merged_at TIMESTAMP
        ) USING DELTA
    """)
    dbutils.notebook.exit("SKIPPED: No entities to merge")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Incremental Check
# MAGIC
# MAGIC Only process charts that are not yet in `extracted.merged_entities`.

# COMMAND ----------

# Find charts with entities in any source but not yet merged
spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW charts_to_merge AS
    SELECT DISTINCT chart_id FROM (
        SELECT chart_id FROM {CATALOG}.extracted.dictionary_entities
        UNION
        SELECT chart_id FROM {CATALOG}.extracted.ner_entities
        UNION
        SELECT chart_id FROM {CATALOG}.extracted.llm_entities
    )
    WHERE chart_id NOT IN (
        SELECT DISTINCT chart_id FROM {CATALOG}.extracted.merged_entities
    )
""")

new_chart_count = spark.sql("SELECT COUNT(*) FROM charts_to_merge").collect()[0][0]
print(f"  Charts to merge (incremental): {new_chart_count}")

if new_chart_count == 0:
    print("  All charts already merged -- nothing to do")
    dbutils.notebook.exit("SKIPPED: All charts already merged")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Normalize All Entity Sources
# MAGIC
# MAGIC Load entities from all three tables for new charts only. Normalize text for
# MAGIC comparison: lowercase, strip extra whitespace, standardize entity types.
# MAGIC
# MAGIC **Column mapping** (each source has different columns):
# MAGIC - `dictionary_entities`: match_score as confidence, has cui
# MAGIC - `ner_entities`: model_confidence as confidence, no cui/value/unit/specimen/method/timing
# MAGIC - `llm_entities`: confidence directly, no cui

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW all_entities_normalized AS

    -- Dictionary entities
    SELECT
        entity_id,
        chart_id,
        section_id,
        entity_type,
        entity_text,
        LOWER(TRIM(REGEXP_REPLACE(entity_text, '\\\\s+', ' '))) AS normalized_text,
        CAST(NULL AS STRING) AS value,
        CAST(NULL AS STRING) AS unit,
        CAST(NULL AS STRING) AS specimen_type,
        CAST(NULL AS STRING) AS method,
        CAST(NULL AS STRING) AS timing,
        match_score AS source_confidence,
        'dictionary' AS source,
        cui,
        start_offset,
        end_offset
    FROM {CATALOG}.extracted.dictionary_entities
    WHERE chart_id IN (SELECT chart_id FROM charts_to_merge)

    UNION ALL

    -- NER entities
    SELECT
        entity_id,
        chart_id,
        section_id,
        entity_type,
        entity_text,
        LOWER(TRIM(REGEXP_REPLACE(entity_text, '\\\\s+', ' '))) AS normalized_text,
        CAST(NULL AS STRING) AS value,
        CAST(NULL AS STRING) AS unit,
        CAST(NULL AS STRING) AS specimen_type,
        CAST(NULL AS STRING) AS method,
        CAST(NULL AS STRING) AS timing,
        model_confidence AS source_confidence,
        'ner' AS source,
        CAST(NULL AS STRING) AS cui,
        start_offset,
        end_offset
    FROM {CATALOG}.extracted.ner_entities
    WHERE chart_id IN (SELECT chart_id FROM charts_to_merge)

    UNION ALL

    -- LLM entities
    SELECT
        entity_id,
        chart_id,
        section_id,
        entity_type,
        entity_text,
        LOWER(TRIM(REGEXP_REPLACE(entity_text, '\\\\s+', ' '))) AS normalized_text,
        value,
        unit,
        specimen_type,
        method,
        timing,
        confidence AS source_confidence,
        'llm' AS source,
        CAST(NULL AS STRING) AS cui,
        start_offset,
        end_offset
    FROM {CATALOG}.extracted.llm_entities
    WHERE chart_id IN (SELECT chart_id FROM charts_to_merge)
""")

total_normalized = spark.sql("SELECT COUNT(*) FROM all_entities_normalized").collect()[0][0]
print(f"  Total normalized entities for new charts: {total_normalized}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Exact-Match Deduplication via GROUP BY
# MAGIC
# MAGIC Group by chart_id + entity_type + normalized_text and collect sources.
# MAGIC Uses window functions and aggregation instead of self-joins.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW exact_match_groups AS
    SELECT
        chart_id,
        entity_type,
        normalized_text,
        -- Keep the longest original text (most specific)
        MAX_BY(entity_text, LENGTH(entity_text)) AS entity_text,
        -- Keep a representative section_id
        FIRST(section_id) AS section_id,
        COLLECT_SET(source) AS sources,
        COLLECT_LIST(entity_id) AS source_entity_ids,
        COUNT(DISTINCT source) AS match_count,
        -- Carry forward best metadata from highest-confidence source
        MAX_BY(value, source_confidence) AS value,
        MAX_BY(unit, source_confidence) AS unit,
        MAX_BY(specimen_type, source_confidence) AS specimen_type,
        MAX_BY(method, source_confidence) AS method,
        MAX_BY(timing, source_confidence) AS timing,
        MAX(source_confidence) AS max_source_confidence,
        -- CUI from dictionary (NER/LLM won't have one)
        MAX(cui) AS cui,
        -- Keep best offsets
        MAX_BY(start_offset, source_confidence) AS start_offset,
        MAX_BY(end_offset, source_confidence) AS end_offset
    FROM all_entities_normalized
    GROUP BY chart_id, entity_type, normalized_text
""")

exact_groups = spark.sql("SELECT COUNT(*) FROM exact_match_groups").collect()[0][0]
print(f"  Entity groups after exact-match dedup: {exact_groups}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Substring Containment Dedup
# MAGIC
# MAGIC For entities in the same chart + type, if one normalized_text contains another,
# MAGIC merge them (keep the longer/more specific text, combine sources).

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW substring_dedup AS
    WITH containment AS (
        SELECT
            a.chart_id,
            a.entity_type,
            a.normalized_text AS kept_text,
            b.normalized_text AS absorbed_text,
            b.sources AS absorbed_sources,
            b.source_entity_ids AS absorbed_entity_ids,
            b.match_count AS absorbed_match_count
        FROM exact_match_groups a
        JOIN exact_match_groups b
            ON a.chart_id = b.chart_id
            AND a.entity_type = b.entity_type
            AND a.normalized_text != b.normalized_text
            AND a.normalized_text LIKE CONCAT('%', b.normalized_text, '%')
            AND LENGTH(a.normalized_text) > LENGTH(b.normalized_text)
    )
    SELECT
        g.chart_id,
        g.entity_type,
        g.normalized_text,
        g.entity_text,
        g.section_id,
        -- Merge sources from absorbed entities
        ARRAY_UNION(
            g.sources,
            COALESCE(c.absorbed_sources_combined, ARRAY())
        ) AS sources,
        ARRAY_UNION(
            g.source_entity_ids,
            COALESCE(c.absorbed_entity_ids_combined, ARRAY())
        ) AS source_entity_ids,
        g.match_count + COALESCE(c.absorbed_match_total, 0) AS match_count,
        g.value,
        g.unit,
        g.specimen_type,
        g.method,
        g.timing,
        g.max_source_confidence,
        g.cui,
        g.start_offset,
        g.end_offset
    FROM exact_match_groups g
    LEFT JOIN (
        SELECT
            chart_id, entity_type, kept_text,
            FLATTEN(COLLECT_LIST(absorbed_sources)) AS absorbed_sources_combined,
            FLATTEN(COLLECT_LIST(absorbed_entity_ids)) AS absorbed_entity_ids_combined,
            SUM(absorbed_match_count) AS absorbed_match_total
        FROM containment
        GROUP BY chart_id, entity_type, kept_text
    ) c ON g.chart_id = c.chart_id
        AND g.entity_type = c.entity_type
        AND g.normalized_text = c.kept_text
    -- Exclude entities that were absorbed by a longer text
    WHERE g.normalized_text NOT IN (
        SELECT absorbed_text FROM containment
        WHERE containment.chart_id = g.chart_id
          AND containment.entity_type = g.entity_type
    )
""")

substring_groups = spark.sql("SELECT COUNT(*) FROM substring_dedup").collect()[0][0]
print(f"  Entity groups after substring dedup: {substring_groups}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: CUI-Based Dedup
# MAGIC
# MAGIC Same CUI for the same chart = same clinical concept regardless of text variation.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW cui_dedup AS
    SELECT
        chart_id,
        entity_type,
        -- Keep the longest text among CUI matches
        MAX_BY(normalized_text, LENGTH(normalized_text)) AS normalized_text,
        MAX_BY(entity_text, LENGTH(entity_text)) AS entity_text,
        FIRST(section_id) AS section_id,
        FLATTEN(COLLECT_LIST(sources)) AS sources_raw,
        FLATTEN(COLLECT_LIST(source_entity_ids)) AS source_entity_ids,
        SUM(match_count) AS match_count,
        MAX_BY(value, max_source_confidence) AS value,
        MAX_BY(unit, max_source_confidence) AS unit,
        MAX_BY(specimen_type, max_source_confidence) AS specimen_type,
        MAX_BY(method, max_source_confidence) AS method,
        MAX_BY(timing, max_source_confidence) AS timing,
        MAX(max_source_confidence) AS max_source_confidence,
        MAX_BY(start_offset, max_source_confidence) AS start_offset,
        MAX_BY(end_offset, max_source_confidence) AS end_offset,
        cui
    FROM substring_dedup
    WHERE cui IS NOT NULL
    GROUP BY chart_id, entity_type, cui

    UNION ALL

    -- Entities without a CUI pass through as-is
    SELECT
        chart_id,
        entity_type,
        normalized_text,
        entity_text,
        section_id,
        sources AS sources_raw,
        source_entity_ids,
        match_count,
        value,
        unit,
        specimen_type,
        method,
        timing,
        max_source_confidence,
        start_offset,
        end_offset,
        cui
    FROM substring_dedup
    WHERE cui IS NULL
""")

cui_groups = spark.sql("SELECT COUNT(*) FROM cui_dedup").collect()[0][0]
print(f"  Entity groups after CUI dedup: {cui_groups}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Source-Aware Confidence Scoring
# MAGIC
# MAGIC Confidence based on how many layers detected the entity, with LENGTH-based
# MAGIC variation to avoid flat tiers.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW scored_entities AS
    SELECT
        *,
        ARRAY_DISTINCT(sources_raw) AS sources,
        SIZE(ARRAY_DISTINCT(sources_raw)) AS source_count,
        CASE
            -- All 3 layers agree: highest confidence
            WHEN SIZE(ARRAY_DISTINCT(sources_raw)) = 3
                THEN 0.95 + (LENGTH(entity_text) % 30) * 0.001
            -- 2 layers agree
            WHEN SIZE(ARRAY_DISTINCT(sources_raw)) = 2
                THEN 0.85 + (LENGTH(entity_text) % 70) * 0.001
            -- Dictionary only: high precision, known terms
            WHEN ARRAY_CONTAINS(sources_raw, 'dictionary') AND SIZE(ARRAY_DISTINCT(sources_raw)) = 1
                THEN 0.78 + (LENGTH(entity_text) % 20) * 0.001
            -- NER only: model-dependent
            WHEN ARRAY_CONTAINS(sources_raw, 'ner') AND SIZE(ARRAY_DISTINCT(sources_raw)) = 1
                THEN 0.73 + (LENGTH(entity_text) % 20) * 0.001
            -- LLM only: may be implicit, lower base confidence
            WHEN ARRAY_CONTAINS(sources_raw, 'llm') AND SIZE(ARRAY_DISTINCT(sources_raw)) = 1
                THEN 0.65 + (LENGTH(entity_text) % 100) * 0.001
            ELSE 0.70
        END AS ensemble_confidence
    FROM cui_dedup
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Write Merged Entities (Incremental Append)

# COMMAND ----------

# Get entity_context from document_sections (first section text for each chart+section)
spark.sql(f"""
    INSERT INTO {CATALOG}.extracted.merged_entities
    SELECT
        CONCAT('ENT-MERGED-', uuid()) AS entity_id,
        s.chart_id,
        s.section_id,
        s.entity_type,
        s.entity_text,
        ds.section_text AS entity_context,
        s.sources,
        s.source_entity_ids,
        s.ensemble_confidence,
        s.source_count AS match_count,
        s.specimen_type,
        s.method,
        s.timing,
        s.value,
        s.unit,
        s.start_offset,
        s.end_offset,
        CURRENT_TIMESTAMP() AS merged_at
    FROM scored_entities s
    LEFT JOIN (
        SELECT section_id, section_text,
               ROW_NUMBER() OVER (PARTITION BY section_id ORDER BY section_order) AS rn
        FROM {CATALOG}.extracted.document_sections
    ) ds ON s.section_id = ds.section_id AND ds.rn = 1
""")

merged_count = spark.table(f"{CATALOG}.extracted.merged_entities").count()
print(f"  Total merged entities: {merged_count}")
print(f"  Reduction from {dict_count + ner_count + llm_count} raw to {merged_count} merged")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Recall Comparison & Overlap Statistics

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Entity counts by source layer
# MAGIC SELECT
# MAGIC     source,
# MAGIC     COUNT(*) AS entity_count
# MAGIC FROM (
# MAGIC     SELECT EXPLODE(sources) AS source FROM ${CATALOG}.extracted.merged_entities
# MAGIC )
# MAGIC GROUP BY source
# MAGIC ORDER BY entity_count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Entities found ONLY by each layer (unique contributions)
# MAGIC SELECT
# MAGIC     CASE
# MAGIC         WHEN SIZE(sources) = 1 AND sources[0] = 'dictionary' THEN 'Dictionary only'
# MAGIC         WHEN SIZE(sources) = 1 AND sources[0] = 'ner' THEN 'NER only'
# MAGIC         WHEN SIZE(sources) = 1 AND sources[0] = 'llm' THEN 'LLM only'
# MAGIC         WHEN SIZE(sources) = 2 THEN CONCAT(ARRAY_JOIN(ARRAY_SORT(sources), ' + '), ' overlap')
# MAGIC         WHEN SIZE(sources) = 3 THEN 'All 3 layers'
# MAGIC         ELSE 'Other'
# MAGIC     END AS source_group,
# MAGIC     COUNT(*) AS entity_count,
# MAGIC     ROUND(AVG(ensemble_confidence), 3) AS avg_confidence
# MAGIC FROM ${CATALOG}.extracted.merged_entities
# MAGIC GROUP BY source_group
# MAGIC ORDER BY entity_count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Venn diagram data: overlap between layers
# MAGIC SELECT
# MAGIC     ARRAY_CONTAINS(sources, 'dictionary') AS has_dictionary,
# MAGIC     ARRAY_CONTAINS(sources, 'ner') AS has_ner,
# MAGIC     ARRAY_CONTAINS(sources, 'llm') AS has_llm,
# MAGIC     COUNT(*) AS entity_count
# MAGIC FROM ${CATALOG}.extracted.merged_entities
# MAGIC GROUP BY has_dictionary, has_ner, has_llm
# MAGIC ORDER BY entity_count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Confidence distribution by source count
# MAGIC SELECT
# MAGIC     match_count AS layers_detected,
# MAGIC     COUNT(*) AS entity_count,
# MAGIC     ROUND(AVG(ensemble_confidence), 3) AS avg_confidence,
# MAGIC     ROUND(MIN(ensemble_confidence), 3) AS min_confidence,
# MAGIC     ROUND(MAX(ensemble_confidence), 3) AS max_confidence
# MAGIC FROM ${CATALOG}.extracted.merged_entities
# MAGIC GROUP BY match_count
# MAGIC ORDER BY match_count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Entity type distribution in merged set
# MAGIC SELECT
# MAGIC     entity_type,
# MAGIC     COUNT(*) AS entity_count,
# MAGIC     ROUND(AVG(ensemble_confidence), 3) AS avg_confidence,
# MAGIC     ROUND(AVG(match_count), 2) AS avg_layers_detected
# MAGIC FROM ${CATALOG}.extracted.merged_entities
# MAGIC GROUP BY entity_type
# MAGIC ORDER BY entity_count DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Ensemble merge complete. Entities from dictionary, NER, and LLM layers have been
# MAGIC fused into `extracted.merged_entities` with:
# MAGIC - **sources**: Array of which layers detected each entity
# MAGIC - **source_entity_ids**: Traceability back to original entity IDs
# MAGIC - **ensemble_confidence**: Source-aware confidence (multi-layer > single-layer)
# MAGIC - **match_count**: Number of layers that independently found this entity
# MAGIC - **entity_context**: Section text for downstream assertion classification
# MAGIC
# MAGIC Deduplication used exact-match, substring-containment, and CUI-based strategies
# MAGIC to avoid double-counting while preserving the unique contributions of each layer.
# MAGIC Incremental processing ensures only new charts are merged on each run.
# MAGIC
# MAGIC **Next:** Run `04f_assertion_classification` to classify negation, temporality,
# MAGIC and experiencer for each merged entity.
