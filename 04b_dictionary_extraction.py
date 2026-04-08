# Databricks notebook source
# MAGIC %md
# MAGIC # 04b — Dictionary-Based Entity Detection (Layer 2)
# MAGIC
# MAGIC The highest-impact extraction layer. Uses the `reference.medical_dictionary` table
# MAGIC to find clinical entities via deterministic string matching -- no LLM calls required.
# MAGIC
# MAGIC **Three matching strategies (in order of precision):**
# MAGIC 1. **Exact Match** -- N-gram tokenization against normalized dictionary terms
# MAGIC 2. **Fuzzy Match** -- Token-level inverted index + Levenshtein for near-misses
# MAGIC 3. **Abbreviation Expansion** -- Expands clinical abbreviations (HTN, DM, CHF) then re-matches
# MAGIC
# MAGIC **Scalability fixes (v2):**
# MAGIC - Inverted token index eliminates cross-join for fuzzy matching
# MAGIC - Broadcast joins on dictionary (fits in memory)
# MAGIC - Incremental processing: skips charts already in `dictionary_entities`
# MAGIC - Empty table guard: handles zero-section edge case
# MAGIC - No `.cache()` -- uses temp tables for serverless compatibility
# MAGIC
# MAGIC **Input:** `extracted.document_sections` (from 04a)
# MAGIC **Output:** `extracted.dictionary_entities`

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
    StructType, StructField, StringType, DoubleType, IntegerType, ArrayType
)
from pyspark.sql.window import Window
import re

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Load Dictionary + Sections (with Incremental Filter)
# MAGIC
# MAGIC Load `reference.medical_dictionary` and `extracted.document_sections`.
# MAGIC Skip any chart_ids that already have entities in `extracted.dictionary_entities`.

# COMMAND ----------

# Ensure output table exists
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.dictionary_entities (
    entity_id STRING,
    chart_id STRING,
    section_id STRING,
    entity_type STRING,
    entity_text STRING,
    matched_term STRING,
    match_type STRING,
    match_score DOUBLE,
    source STRING,
    source_code STRING,
    cui STRING,
    start_offset INT,
    end_offset INT,
    extracted_at TIMESTAMP
)
USING DELTA
COMMENT 'Entities detected via dictionary-based string matching (Layer 2)'
""")
print(f"  extracted.dictionary_entities table ready")

# COMMAND ----------

# Load all sections
all_sections_df = spark.table(f"{CATALOG}.extracted.document_sections")
total_section_count = all_sections_df.count()

if total_section_count == 0:
    print(f"  extracted.document_sections is empty -- nothing to process")
    print(f"  Skipping dictionary extraction")
    dbutils.notebook.exit("SKIP: no sections to process")

print(f"  Total sections in document_sections: {total_section_count:,}")

# COMMAND ----------

# Incremental filter: skip chart_ids that already have dictionary entities
existing_chart_ids = (
    spark.table(f"{CATALOG}.extracted.dictionary_entities")
    .select("chart_id")
    .distinct()
)
existing_count = existing_chart_ids.count()

if existing_count > 0:
    sections_df = all_sections_df.join(existing_chart_ids, "chart_id", "left_anti")
    new_section_count = sections_df.count()
    print(f"  Skipping {existing_count:,} charts already processed")
    print(f"  New sections to process: {new_section_count:,}")
    if new_section_count == 0:
        print(f"  All charts already processed -- nothing new to extract")
        dbutils.notebook.exit("SKIP: all charts already processed")
else:
    sections_df = all_sections_df
    new_section_count = total_section_count
    print(f"  No existing entities found -- processing all {new_section_count:,} sections")

# COMMAND ----------

# Load medical dictionary (already has term_normalized from 02b)
dict_df = spark.table(f"{CATALOG}.reference.medical_dictionary")
dict_count = dict_df.count()
print(f"  Loaded {dict_count:,} terms from reference.medical_dictionary")

# Prepare normalized dictionary with term_length, broadcast for joins
normalized_dict_df = (
    dict_df
    .filter(F.col("term_normalized").isNotNull())
    .filter(F.length(F.col("term_normalized")) >= 2)
    .withColumn("term_length", F.length(F.col("term_normalized")))
)

# Save to temp table (serverless does not support .cache())
normalized_dict_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference._tmp_normalized_dict")
normalized_dict_df = spark.table(f"{CATALOG}.reference._tmp_normalized_dict")
norm_dict_count = normalized_dict_df.count()
print(f"  Normalized dictionary ready ({norm_dict_count:,} terms)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Exact Match (N-gram Tokenization + Equi-Join)
# MAGIC
# MAGIC Tokenize each section's text into overlapping n-grams (1 to 5 tokens),
# MAGIC normalize them, and equi-join against the dictionary on `term_normalized`.

# COMMAND ----------

def normalize_term(text):
    """Normalize a medical term for matching: lowercase, strip punctuation, collapse whitespace."""
    if text is None:
        return None
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

normalize_udf = F.udf(normalize_term, StringType())

def extract_ngrams(text, max_n=5):
    """Extract all n-grams (1 to max_n tokens) from text. Returns deduplicated list."""
    if text is None:
        return []
    tokens = re.findall(r'\b[\w\-]+\b', text.lower())
    ngrams = []
    for n in range(1, min(max_n + 1, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i + n])
            ngrams.append(gram)
    return list(set(ngrams))

extract_ngrams_udf = F.udf(extract_ngrams, ArrayType(StringType()))

# COMMAND ----------

# Generate n-grams for each section
sections_with_ngrams = sections_df.withColumn(
    "ngrams", extract_ngrams_udf(F.col("section_text"))
)

# Explode n-grams for joining
exploded_ngrams = (
    sections_with_ngrams
    .select("section_id", "chart_id", "section_type", "section_text",
            F.explode("ngrams").alias("ngram"))
)

# COMMAND ----------

# Exact match: equi-join n-grams against dictionary on term_normalized
exact_matches = (
    exploded_ngrams
    .join(
        F.broadcast(normalized_dict_df),
        exploded_ngrams["ngram"] == normalized_dict_df["term_normalized"],
        "inner"
    )
    .select(
        exploded_ngrams["section_id"],
        exploded_ngrams["chart_id"],
        normalized_dict_df["entity_type"],
        exploded_ngrams["ngram"].alias("entity_text"),
        normalized_dict_df["term"].alias("matched_term"),
        F.lit("exact").alias("match_type"),
        F.lit(1.0).cast(DoubleType()).alias("match_score"),
        normalized_dict_df["source"],
        normalized_dict_df["source_code"],
        normalized_dict_df["cui"],
    )
    .dropDuplicates(["section_id", "matched_term"])
)

# Save exact matches to temp table to avoid recomputation
exact_matches.write.mode("overwrite").saveAsTable(f"{CATALOG}.extracted._tmp_exact_matches")
exact_matches = spark.table(f"{CATALOG}.extracted._tmp_exact_matches")
exact_count = exact_matches.count()
print(f"  Exact matches found: {exact_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Fuzzy Match (Inverted Token Index)
# MAGIC
# MAGIC Instead of a cross-join, we build an inverted index: for each token in the
# MAGIC dictionary, store which terms contain that token. Then for each section n-gram,
# MAGIC look up its constituent tokens in the index (equality join), producing only
# MAGIC candidate pairs that share at least one token. Levenshtein is computed only
# MAGIC on these candidates.

# COMMAND ----------

# Build inverted token index from dictionary terms
# Each dictionary term is tokenized; each token maps back to the term
dict_with_tokens = (
    normalized_dict_df
    .filter(F.col("term_length") >= 6)
    .withColumn("dict_token", F.explode(F.split(F.col("term_normalized"), r"\s+")))
    .filter(F.length(F.col("dict_token")) >= 3)  # skip very short tokens
)

# Save inverted index to temp table
dict_with_tokens.write.mode("overwrite").saveAsTable(f"{CATALOG}.reference._tmp_dict_inverted_index")
dict_with_tokens = spark.table(f"{CATALOG}.reference._tmp_dict_inverted_index")
index_count = dict_with_tokens.count()
print(f"  Inverted index built: {index_count:,} (token, term) pairs")

# COMMAND ----------

# Get n-grams that did NOT match exactly (candidates for fuzzy matching)
exact_matched_ngrams = exact_matches.select(
    F.col("entity_text").alias("ngram")
).distinct()

fuzzy_candidates = (
    exploded_ngrams
    .join(exact_matched_ngrams, "ngram", "left_anti")
    .filter(F.length(F.col("ngram")) >= 6)
    .select("section_id", "chart_id", "ngram")
    .distinct()
)

# Tokenize fuzzy candidates into constituent tokens
candidate_tokens = (
    fuzzy_candidates
    .withColumn("cand_token", F.explode(F.split(F.col("ngram"), r"\s+")))
    .filter(F.length(F.col("cand_token")) >= 3)
)

# COMMAND ----------

# Join candidate tokens against the inverted index on token equality
# This produces only (ngram, dict_term) pairs that share at least one token
fuzzy_pairs = (
    candidate_tokens
    .join(
        F.broadcast(dict_with_tokens),
        candidate_tokens["cand_token"] == dict_with_tokens["dict_token"],
        "inner"
    )
    .select(
        candidate_tokens["section_id"],
        candidate_tokens["chart_id"],
        candidate_tokens["ngram"],
        dict_with_tokens["term"],
        dict_with_tokens["term_normalized"],
        dict_with_tokens["term_length"],
        dict_with_tokens["entity_type"],
        dict_with_tokens["source"],
        dict_with_tokens["source_code"],
        dict_with_tokens["cui"],
    )
    .distinct()  # deduplicate pairs that share multiple tokens
)

# COMMAND ----------

# Compute Levenshtein distance only on candidate pairs
fuzzy_matches = (
    fuzzy_pairs
    .withColumn("ngram_length", F.length(F.col("ngram")))
    .filter(F.abs(F.col("ngram_length") - F.col("term_length")) <= 2)
    .withColumn(
        "edit_distance",
        F.levenshtein(F.col("ngram"), F.col("term_normalized"))
    )
    .filter(F.col("edit_distance") <= 2)
    .filter(F.col("edit_distance") > 0)  # exclude exact matches (already captured)
    .withColumn(
        "max_len",
        F.greatest(F.col("ngram_length"), F.col("term_length"))
    )
    .withColumn(
        "match_score",
        (1.0 - F.col("edit_distance").cast(DoubleType()) / F.col("max_len").cast(DoubleType()))
    )
    .filter(F.col("match_score") >= 0.85)
    .select(
        "section_id",
        "chart_id",
        "entity_type",
        F.col("ngram").alias("entity_text"),
        F.col("term").alias("matched_term"),
        F.lit("fuzzy").alias("match_type"),
        "match_score",
        "source",
        "source_code",
        "cui",
    )
    .dropDuplicates(["section_id", "matched_term"])
)

fuzzy_matches.write.mode("overwrite").saveAsTable(f"{CATALOG}.extracted._tmp_fuzzy_matches")
fuzzy_matches = spark.table(f"{CATALOG}.extracted._tmp_fuzzy_matches")
fuzzy_count = fuzzy_matches.count()
print(f"  Fuzzy matches found: {fuzzy_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Abbreviation Expansion
# MAGIC
# MAGIC Clinical notes are full of abbreviations (HTN, DM, CHF, COPD, etc.).
# MAGIC Load `reference.clinical_abbreviations`, tokenize sections, expand abbreviation
# MAGIC tokens, and equi-join the expansion against the dictionary.

# COMMAND ----------

try:
    abbrev_df = spark.table(f"{CATALOG}.reference.clinical_abbreviations")
    abbrev_count = abbrev_df.count()
    print(f"  Loaded {abbrev_count:,} clinical abbreviations")
    has_abbreviations = abbrev_count > 0
except Exception as e:
    print(f"  Clinical abbreviations table not found: {e}")
    print(f"  Skipping abbreviation expansion strategy")
    has_abbreviations = False

# COMMAND ----------

if has_abbreviations:
    # Broadcast abbreviation lookup
    abbrev_broadcast = F.broadcast(abbrev_df)

    # Tokenize sections into individual tokens for abbreviation matching
    section_tokens = (
        sections_df
        .select("section_id", "chart_id", "section_text")
        .withColumn("token", F.explode(F.split(F.upper(F.col("section_text")), r"\s+")))
        .withColumn("token", F.regexp_replace(F.col("token"), r"[^\w]", ""))
        .filter(F.length(F.col("token")) >= 2)
        .filter(F.length(F.col("token")) <= 10)  # abbreviations are short
    )

    # Join tokens against abbreviation table
    # Note: abbreviation table column is "expansion" (not "expanded_term")
    expanded = (
        section_tokens
        .join(
            abbrev_broadcast,
            section_tokens["token"] == F.upper(abbrev_broadcast["abbreviation"]),
            "inner"
        )
    )

    # Normalize the expansion and match against dictionary
    expanded_with_norm = expanded.withColumn(
        "expanded_normalized", normalize_udf(F.col("expansion"))
    )

    abbrev_matches = (
        expanded_with_norm
        .join(
            F.broadcast(normalized_dict_df),
            expanded_with_norm["expanded_normalized"] == normalized_dict_df["term_normalized"],
            "inner"
        )
        .select(
            expanded_with_norm["section_id"],
            expanded_with_norm["chart_id"],
            normalized_dict_df["entity_type"],
            expanded_with_norm["token"].alias("entity_text"),
            normalized_dict_df["term"].alias("matched_term"),
            F.lit("abbreviation").alias("match_type"),
            F.lit(0.95).cast(DoubleType()).alias("match_score"),
            normalized_dict_df["source"],
            normalized_dict_df["source_code"],
            normalized_dict_df["cui"],
        )
        .dropDuplicates(["section_id", "matched_term"])
    )

    abbrev_matches.write.mode("overwrite").saveAsTable(f"{CATALOG}.extracted._tmp_abbrev_matches")
    abbrev_matches = spark.table(f"{CATALOG}.extracted._tmp_abbrev_matches")
    abbrev_match_count = abbrev_matches.count()
    print(f"  Abbreviation expansion matches found: {abbrev_match_count:,}")
else:
    abbrev_matches = None
    abbrev_match_count = 0
    print(f"  Abbreviation strategy skipped")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5: Deduplicate and Write Results
# MAGIC
# MAGIC Merge all three strategies into `extracted.dictionary_entities`.
# MAGIC When the same entity is found by multiple strategies, keep the highest-scoring match.

# COMMAND ----------

# Union all match strategies
all_matches = exact_matches
if fuzzy_count > 0:
    all_matches = all_matches.unionByName(fuzzy_matches)
if has_abbreviations and abbrev_matches is not None and abbrev_match_count > 0:
    all_matches = all_matches.unionByName(abbrev_matches)

# Deduplicate: when the same term is found by multiple strategies in the same section,
# keep the highest-scoring match
dedup_window = Window.partitionBy("section_id", "matched_term").orderBy(
    F.desc("match_score"),
    F.asc("match_type")  # prefer exact > abbreviation > fuzzy for ties
)

deduped_matches = (
    all_matches
    .withColumn("rank", F.row_number().over(dedup_window))
    .filter(F.col("rank") == 1)
    .drop("rank")
)

# Add entity IDs, offsets, and timestamp
final_entities = deduped_matches.select(
    F.concat(F.lit("ENT-DICT-"), F.expr("uuid()")).alias("entity_id"),
    "chart_id",
    "section_id",
    "entity_type",
    "entity_text",
    "matched_term",
    "match_type",
    "match_score",
    "source",
    "source_code",
    "cui",
    F.lit(None).cast(IntegerType()).alias("start_offset"),
    F.lit(None).cast(IntegerType()).alias("end_offset"),
    F.current_timestamp().alias("extracted_at"),
)

# COMMAND ----------

# Append new entities (incremental -- do not overwrite existing data)
final_entities.write.mode("append").saveAsTable(f"{CATALOG}.extracted.dictionary_entities")

total_entities = spark.table(f"{CATALOG}.extracted.dictionary_entities").count()
new_entities = final_entities.count()
print(f"  Wrote {new_entities:,} new dictionary-matched entities")
print(f"  Total entities in table: {total_entities:,}")

# COMMAND ----------

# Clean up temp tables
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.reference._tmp_normalized_dict")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.reference._tmp_dict_inverted_index")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.extracted._tmp_exact_matches")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.extracted._tmp_fuzzy_matches")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.extracted._tmp_abbrev_matches")
print(f"  Temp tables cleaned up")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6: Extraction Summary

# COMMAND ----------

# Entities by match type
print("  Entities by match type:")
spark.sql(f"""
    SELECT
        match_type,
        COUNT(*) AS entity_count,
        ROUND(AVG(match_score), 3) AS avg_score,
        ROUND(MIN(match_score), 3) AS min_score
    FROM {CATALOG}.extracted.dictionary_entities
    GROUP BY match_type
    ORDER BY entity_count DESC
""").show(truncate=False)

# COMMAND ----------

# Entities by entity type
print("  Entities by entity type:")
spark.sql(f"""
    SELECT
        entity_type,
        COUNT(*) AS entity_count,
        COUNT(DISTINCT chart_id) AS charts_with_type,
        ROUND(AVG(match_score), 3) AS avg_score
    FROM {CATALOG}.extracted.dictionary_entities
    GROUP BY entity_type
    ORDER BY entity_count DESC
""").show(truncate=False)

# COMMAND ----------

# Top 20 most frequently matched terms
print("  Top 20 most frequently matched terms:")
spark.sql(f"""
    SELECT
        matched_term,
        entity_type,
        match_type,
        source,
        source_code,
        COUNT(*) AS occurrence_count,
        COUNT(DISTINCT chart_id) AS charts_found_in
    FROM {CATALOG}.extracted.dictionary_entities
    GROUP BY matched_term, entity_type, match_type, source, source_code
    ORDER BY occurrence_count DESC
    LIMIT 20
""").show(truncate=False)

# COMMAND ----------

# Coverage: how many charts had at least one dictionary match?
print("  Dictionary extraction coverage:")
spark.sql(f"""
    SELECT
        total_charts,
        charts_with_matches,
        ROUND(100.0 * charts_with_matches / GREATEST(total_charts, 1), 1) AS coverage_pct,
        total_entities,
        ROUND(total_entities * 1.0 / GREATEST(charts_with_matches, 1), 1) AS avg_entities_per_chart
    FROM (
        SELECT
            (SELECT COUNT(DISTINCT chart_id) FROM {CATALOG}.raw.charts) AS total_charts,
            (SELECT COUNT(DISTINCT chart_id) FROM {CATALOG}.extracted.dictionary_entities) AS charts_with_matches,
            (SELECT COUNT(*) FROM {CATALOG}.extracted.dictionary_entities) AS total_entities
    )
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### Dictionary-based extraction results
# MAGIC - **Exact match**: Highest precision, catches standard terminology via equi-join
# MAGIC - **Fuzzy match**: Catches misspellings/variants using inverted token index + Levenshtein (no cross-join)
# MAGIC - **Abbreviation expansion**: Resolves clinical shorthand (HTN, DM, CHF, etc.)
# MAGIC
# MAGIC ### Scalability improvements (v2)
# MAGIC - Inverted token index eliminates O(N*M) cross-join for fuzzy matching
# MAGIC - Broadcast joins on dictionary tables
# MAGIC - Incremental processing: only new chart_ids are processed
# MAGIC - Temp tables instead of `.cache()` for serverless compatibility
# MAGIC
# MAGIC ### Tables written
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `extracted.dictionary_entities` | All dictionary-matched entities with match type and score |
# MAGIC
# MAGIC **Next:** Run `04c_ner_model_extraction` for biomedical NER model extraction (Layer 3).
