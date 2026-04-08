# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Pipeline Setup
# MAGIC
# MAGIC Creates the catalog structure, pipeline tables, feedback tables, and UC Volume
# MAGIC required by the medical codification pipeline.
# MAGIC
# MAGIC **What this notebook creates:**
# MAGIC - 5 schemas: `raw`, `extracted`, `codified`, `reference`, `feedback`
# MAGIC - Pipeline tables: `charts`, `entities`, `icd10_mappings`, `loinc_mappings`
# MAGIC - Reference ontology tables: `umls_concepts`, `snomed_hierarchy`, `snomed_icd10_map`, `rxnorm_concepts`, `clinical_abbreviations`, `medical_dictionary`
# MAGIC - Multi-layer extraction tables: `document_sections`, `dictionary_entities`, `ner_entities`, `llm_entities`, `merged_entities`, `entity_assertions`
# MAGIC - Codification tables: `umls_mappings`
# MAGIC - Feedback tables: `human_corrections`, `gold_annotations`, `recall_metrics`, `error_patterns`
# MAGIC - UC Volume: `raw.chart_pdfs` (PDF storage)
# MAGIC
# MAGIC **What this notebook does NOT create:**
# MAGIC - Reference data (ICD-10, LOINC) -- that is notebook 02
# MAGIC - Medical ontologies (UMLS, SNOMED, RxNorm) -- that is notebook 02b
# MAGIC
# MAGIC **Estimated runtime:** < 30 seconds

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("CATALOG", "mv_catalog", "Unity Catalog Name")
CATALOG = dbutils.widgets.get("CATALOG")

SCHEMAS = ["raw", "extracted", "codified", "reference", "feedback"]

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Schemas

# COMMAND ----------

for schema in SCHEMAS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
    print(f"  Schema {CATALOG}.{schema} ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Pipeline Tables
# MAGIC
# MAGIC These tables hold the data as it flows through each stage:
# MAGIC 1. **raw.charts** -- Ingested chart metadata and extracted text
# MAGIC 2. **extracted.entities** -- Clinical entities (diagnoses, labs, meds, vitals) from NER
# MAGIC 3. **codified.icd10_mappings** -- Final ICD-10 code assignments with full audit trail
# MAGIC 4. **codified.loinc_mappings** -- Final LOINC code assignments with full audit trail
# MAGIC
# MAGIC The codified tables include multi-pass resolution fields: `resolution_path`,
# MAGIC `r1_code`, `r2_verdict`, `r2_code`, `arbiter_code`, and `arbiter_reasoning`.

# COMMAND ----------

# --- raw.charts ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.raw.charts (
    chart_id STRING,
    patient_id STRING,
    file_name STRING,
    file_path STRING,
    chart_type STRING,
    provider STRING,
    facility STRING,
    chart_date DATE,
    ingested_at TIMESTAMP,
    raw_text STRING,
    page_count INT,
    extraction_method STRING
)
USING DELTA
COMMENT 'Raw medical chart metadata and extracted text'
""")
print("  raw.charts ready")

# COMMAND ----------

# --- extracted.entities ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.entities (
    entity_id STRING,
    chart_id STRING,
    entity_type STRING,
    entity_text STRING,
    entity_context STRING,
    start_offset INT,
    end_offset INT,
    confidence DOUBLE,
    specimen_type STRING,
    method STRING,
    timing STRING,
    value STRING,
    unit STRING,
    extracted_at TIMESTAMP
)
USING DELTA
COMMENT 'Clinical entities extracted via NER from chart text'
""")
print("  extracted.entities ready")

# COMMAND ----------

# --- codified.icd10_mappings ---
# Full audit trail schema from the multi-pass codification pipeline (v2)
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.codified.icd10_mappings (
    mapping_id STRING,
    entity_id STRING,
    chart_id STRING,
    entity_text STRING,
    icd10_code STRING,
    icd10_description STRING,
    confidence DOUBLE,
    resolution_path STRING COMMENT 'R1_R2_AGREE | ARBITER_CHOSE_R1 | ARBITER_CHOSE_R2 | DISPUTED_UNRESOLVED',
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
COMMENT 'ICD-10 code mappings with multi-pass audit trail'
""")
print("  codified.icd10_mappings ready")

# COMMAND ----------

# --- codified.loinc_mappings ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.codified.loinc_mappings (
    mapping_id STRING,
    entity_id STRING,
    chart_id STRING,
    entity_text STRING,
    loinc_code STRING,
    loinc_long_name STRING,
    confidence DOUBLE,
    resolution_path STRING COMMENT 'R1_R2_AGREE | ARBITER_CHOSE_R1 | ARBITER_CHOSE_R2 | DISPUTED_UNRESOLVED',
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
COMMENT 'LOINC code mappings with multi-pass audit trail'
""")
print("  codified.loinc_mappings ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feedback Table
# MAGIC
# MAGIC The review app writes human corrections back to this table.
# MAGIC These corrections feed into model-improvement loops and audit reporting.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.feedback.human_corrections (
    correction_id STRING,
    mapping_id STRING,
    code_type STRING COMMENT 'ICD10 or LOINC',
    original_code STRING,
    corrected_code STRING,
    entity_text STRING,
    entity_context STRING,
    corrected_by STRING,
    corrected_at TIMESTAMP
)
USING DELTA
COMMENT 'Human corrections from the review app for model improvement'
""")
print("  feedback.human_corrections ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create UC Volume for PDF Storage
# MAGIC
# MAGIC PDFs are stored in a Unity Catalog Volume at `/Volumes/{CATALOG}/raw/chart_pdfs/`.
# MAGIC This works with both serverless and classic compute via the FUSE mount.

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.raw.chart_pdfs")
print(f"  Volume {CATALOG}.raw.chart_pdfs ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## New: Reference Ontology Tables
# MAGIC
# MAGIC These tables store medical ontology data loaded by notebook `02b_load_medical_ontologies`:
# MAGIC - **UMLS Metathesaurus** -- Concept dictionary for entity linking
# MAGIC - **SNOMED-CT** -- Hierarchical clinical terminology with IS-A relationships
# MAGIC - **SNOMED-to-ICD-10 mapping** -- Official crosswalk for code translation
# MAGIC - **RxNorm** -- Drug normalization concepts
# MAGIC - **Clinical abbreviations** -- Medical abbreviation dictionary
# MAGIC - **Medical dictionary** -- Unified search-optimized term dictionary

# COMMAND ----------

# --- reference.umls_concepts ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.reference.umls_concepts (
    cui STRING COMMENT 'UMLS Concept Unique Identifier',
    preferred_name STRING,
    semantic_type STRING COMMENT 'e.g., Disease or Syndrome, Pharmacologic Substance',
    semantic_group STRING COMMENT 'e.g., DISO, CHEM, PROC',
    synonyms ARRAY<STRING> COMMENT 'All known names/synonyms for this concept',
    source_vocabularies ARRAY<STRING> COMMENT 'Which vocabularies include this concept',
    loaded_at TIMESTAMP
)
USING DELTA
COMMENT 'UMLS Metathesaurus concepts for entity linking and normalization'
""")
print("  reference.umls_concepts ready")

# COMMAND ----------

# --- reference.snomed_hierarchy ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.reference.snomed_hierarchy (
    concept_id STRING,
    concept_name STRING,
    parent_id STRING,
    parent_name STRING,
    relationship_type STRING COMMENT 'IS_A by default',
    hierarchy_depth INT,
    loaded_at TIMESTAMP
)
USING DELTA
COMMENT 'SNOMED-CT IS-A hierarchy for ontology traversal'
""")
print("  reference.snomed_hierarchy ready")

# COMMAND ----------

# --- reference.snomed_icd10_map ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.reference.snomed_icd10_map (
    snomed_concept_id STRING,
    snomed_name STRING,
    icd10_code STRING,
    icd10_name STRING,
    map_priority INT COMMENT 'Priority when multiple ICD-10 codes map',
    map_rule STRING COMMENT 'Mapping rule/condition if any',
    map_group INT,
    loaded_at TIMESTAMP
)
USING DELTA
COMMENT 'Official SNOMED-CT to ICD-10-CM mapping from SNOMED International'
""")
print("  reference.snomed_icd10_map ready")

# COMMAND ----------

# --- reference.rxnorm_concepts ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.reference.rxnorm_concepts (
    rxcui STRING COMMENT 'RxNorm Concept Unique Identifier',
    name STRING,
    term_type STRING COMMENT 'IN=ingredient, BN=brand, SCD=clinical drug, etc.',
    synonyms ARRAY<STRING>,
    ingredients ARRAY<STRING>,
    loaded_at TIMESTAMP
)
USING DELTA
COMMENT 'RxNorm drug concepts for medication normalization'
""")
print("  reference.rxnorm_concepts ready")

# COMMAND ----------

# --- reference.clinical_abbreviations ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.reference.clinical_abbreviations (
    abbreviation STRING,
    expansion STRING,
    category STRING COMMENT 'diagnosis, medication, lab, procedure, general',
    context_hint STRING COMMENT 'Section where this abbreviation is most common',
    loaded_at TIMESTAMP
)
USING DELTA
COMMENT 'Clinical abbreviation dictionary for entity detection'
""")
print("  reference.clinical_abbreviations ready")

# COMMAND ----------

# --- reference.medical_dictionary ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.reference.medical_dictionary (
    term STRING COMMENT 'Canonical or synonym term',
    term_normalized STRING COMMENT 'Lowercased, stripped for matching',
    source STRING COMMENT 'UMLS, ICD10, LOINC, RxNorm, SNOMED, abbreviation',
    entity_type STRING COMMENT 'DIAGNOSIS, LAB_RESULT, MEDICATION, VITAL_SIGN, PROCEDURE',
    source_code STRING COMMENT 'Code from source vocabulary',
    cui STRING COMMENT 'UMLS CUI if available',
    loaded_at TIMESTAMP
)
USING DELTA
COMMENT 'Unified medical term dictionary built from all reference sources'
""")
print("  reference.medical_dictionary ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## New: Multi-Layer Extraction Tables
# MAGIC
# MAGIC These tables support the four-layer entity extraction pipeline:
# MAGIC 1. **Document sections** -- Parsed clinical note sections for section-aware NLP
# MAGIC 2. **Dictionary entities** (Layer 2) -- High-recall deterministic matching
# MAGIC 3. **NER entities** (Layer 3) -- BioClinicalBERT/scispaCy model predictions
# MAGIC 4. **LLM entities** (Layer 4) -- Scoped LLM extraction for implicit entities
# MAGIC 5. **Merged entities** -- Ensemble-merged entity set from all layers
# MAGIC 6. **Entity assertions** -- Negation, temporality, and experiencer classification

# COMMAND ----------

# --- extracted.document_sections ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.document_sections (
    section_id STRING,
    chart_id STRING,
    section_type STRING COMMENT 'HPI, ASSESSMENT_PLAN, MEDICATIONS, LABS, VITALS, ALLERGIES, FAMILY_HISTORY, SOCIAL_HISTORY, ROS, OTHER',
    section_header STRING COMMENT 'Original header text from the note',
    section_text STRING,
    section_order INT,
    extraction_method STRING COMMENT 'rule_based, llm_assisted',
    extracted_at TIMESTAMP
)
USING DELTA
COMMENT 'Clinical note sections parsed for section-aware NLP processing'
""")
print("  extracted.document_sections ready")

# COMMAND ----------

# --- extracted.dictionary_entities ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.dictionary_entities (
    entity_id STRING,
    chart_id STRING,
    section_id STRING,
    entity_type STRING,
    entity_text STRING COMMENT 'Text as it appears in the note',
    matched_term STRING COMMENT 'Dictionary term that matched',
    match_type STRING COMMENT 'exact, fuzzy, abbreviation',
    match_score DOUBLE COMMENT '1.0 for exact, <1.0 for fuzzy',
    source STRING COMMENT 'UMLS, ICD10, LOINC, RxNorm',
    source_code STRING,
    cui STRING,
    start_offset INT,
    end_offset INT,
    extracted_at TIMESTAMP
)
USING DELTA
COMMENT 'Entities detected via dictionary/fuzzy matching (high recall, deterministic)'
""")
print("  extracted.dictionary_entities ready")

# COMMAND ----------

# --- extracted.ner_entities ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.ner_entities (
    entity_id STRING,
    chart_id STRING,
    section_id STRING,
    entity_type STRING,
    entity_text STRING,
    ner_label STRING COMMENT 'Model-specific label (e.g., PROBLEM, TREATMENT, TEST)',
    model_name STRING COMMENT 'Which NER model detected this',
    model_confidence DOUBLE,
    start_offset INT,
    end_offset INT,
    extracted_at TIMESTAMP
)
USING DELTA
COMMENT 'Entities detected via BioClinicalBERT/scispaCy NER models'
""")
print("  extracted.ner_entities ready")

# COMMAND ----------

# --- extracted.llm_entities ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.llm_entities (
    entity_id STRING,
    chart_id STRING,
    section_id STRING,
    entity_type STRING,
    entity_text STRING,
    extraction_role STRING COMMENT 'implicit_detection, completeness_check, disambiguation',
    confidence DOUBLE,
    reasoning STRING COMMENT 'LLM reasoning for why this entity was identified',
    specimen_type STRING,
    method STRING,
    timing STRING,
    value STRING,
    unit STRING,
    start_offset INT,
    end_offset INT,
    extracted_at TIMESTAMP
)
USING DELTA
COMMENT 'Entities detected via scoped LLM extraction (implicit entities, completeness validation)'
""")
print("  extracted.llm_entities ready")

# COMMAND ----------

# --- extracted.merged_entities ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.merged_entities (
    entity_id STRING,
    chart_id STRING,
    section_id STRING,
    entity_type STRING,
    entity_text STRING,
    entity_context STRING,
    sources ARRAY<STRING> COMMENT 'Which layers detected this: dictionary, ner, llm',
    source_entity_ids ARRAY<STRING> COMMENT 'Original entity IDs from each source',
    ensemble_confidence DOUBLE COMMENT 'Agreement-based confidence across sources',
    match_count INT COMMENT 'How many extraction layers found this entity',
    specimen_type STRING,
    method STRING,
    timing STRING,
    value STRING,
    unit STRING,
    start_offset INT,
    end_offset INT,
    merged_at TIMESTAMP
)
USING DELTA
COMMENT 'Ensemble-merged entities from dictionary + NER + LLM layers'
""")
print("  extracted.merged_entities ready")

# COMMAND ----------

# --- extracted.entity_assertions ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.extracted.entity_assertions (
    assertion_id STRING,
    entity_id STRING COMMENT 'References merged_entities.entity_id',
    chart_id STRING,
    assertion_status STRING COMMENT 'PRESENT, ABSENT, POSSIBLE, CONDITIONAL, HISTORICAL, FAMILY',
    negation_detected BOOLEAN,
    negation_trigger STRING COMMENT 'Trigger phrase: denies, no evidence of, etc.',
    temporality STRING COMMENT 'CURRENT, HISTORICAL, FUTURE_PLANNED',
    experiencer STRING COMMENT 'PATIENT, FAMILY_MEMBER, OTHER',
    certainty STRING COMMENT 'DEFINITE, PROBABLE, POSSIBLE, UNLIKELY',
    classification_method STRING COMMENT 'context_rules, llm_assisted, hybrid',
    confidence DOUBLE,
    classified_at TIMESTAMP
)
USING DELTA
COMMENT 'Assertion classifications: negation, temporality, experiencer attribution'
""")
print("  extracted.entity_assertions ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## New: UMLS Codification Table
# MAGIC
# MAGIC Maps extracted entities to UMLS CUIs and SNOMED-CT concepts,
# MAGIC complementing the existing ICD-10 and LOINC mapping tables.

# COMMAND ----------

# --- codified.umls_mappings ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.codified.umls_mappings (
    mapping_id STRING,
    entity_id STRING,
    chart_id STRING,
    entity_text STRING,
    cui STRING COMMENT 'UMLS Concept Unique Identifier',
    preferred_name STRING,
    semantic_type STRING,
    snomed_concept_id STRING,
    snomed_name STRING,
    mapping_method STRING COMMENT 'exact_match, embedding_similarity, llm_assisted',
    confidence DOUBLE,
    mapped_at TIMESTAMP
)
USING DELTA
COMMENT 'UMLS CUI and SNOMED-CT concept mappings for entities'
""")
print("  codified.umls_mappings ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## New: Feedback & Evaluation Tables
# MAGIC
# MAGIC These tables support gold-standard annotation, recall measurement,
# MAGIC and systematic error tracking for continuous pipeline improvement.

# COMMAND ----------

# --- feedback.gold_annotations ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.feedback.gold_annotations (
    annotation_id STRING,
    chart_id STRING,
    entity_type STRING,
    entity_text STRING,
    gold_icd10_code STRING,
    gold_loinc_code STRING,
    gold_snomed_id STRING,
    assertion_status STRING,
    annotator STRING,
    annotation_round INT COMMENT '1=first pass, 2=adjudication',
    annotated_at TIMESTAMP
)
USING DELTA
COMMENT 'Human gold-standard annotations for measuring recall and training models'
""")
print("  feedback.gold_annotations ready")

# COMMAND ----------

# --- feedback.recall_metrics ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.feedback.recall_metrics (
    run_id STRING,
    run_timestamp TIMESTAMP,
    metric_scope STRING COMMENT 'overall, by_entity_type, by_section, by_note_density',
    scope_value STRING COMMENT 'e.g., DIAGNOSIS, HPI, dense_note',
    extraction_layer STRING COMMENT 'dictionary, ner, llm, ensemble',
    true_positives INT,
    false_positives INT,
    false_negatives INT,
    precision_score DOUBLE,
    recall_score DOUBLE,
    f1_score DOUBLE,
    charts_evaluated INT
)
USING DELTA
COMMENT 'Per-run recall and precision metrics against gold annotations'
""")
print("  feedback.recall_metrics ready")

# COMMAND ----------

# --- feedback.error_patterns ---
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.feedback.error_patterns (
    pattern_id STRING,
    pattern_type STRING COMMENT 'missed_entity, false_positive, wrong_code, wrong_assertion',
    description STRING,
    example_entity_text STRING,
    example_chart_id STRING,
    frequency INT COMMENT 'How often this pattern occurs',
    severity STRING COMMENT 'HIGH, MEDIUM, LOW',
    suggested_fix STRING COMMENT 'e.g., add to abbreviation dictionary, retrain NER',
    detected_at TIMESTAMP,
    resolved_at TIMESTAMP
)
USING DELTA
COMMENT 'Systematic error patterns detected across pipeline runs'
""")
print("  feedback.error_patterns ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Complete
# MAGIC
# MAGIC | Category | Assets |
# MAGIC |----------|--------|
# MAGIC | **Schemas** | `raw`, `extracted`, `codified`, `reference`, `feedback` |
# MAGIC | **Raw** | `raw.charts` |
# MAGIC | **Extraction (original)** | `extracted.entities` |
# MAGIC | **Extraction (multi-layer)** | `extracted.document_sections`, `extracted.dictionary_entities`, `extracted.ner_entities`, `extracted.llm_entities`, `extracted.merged_entities`, `extracted.entity_assertions` |
# MAGIC | **Codification** | `codified.icd10_mappings`, `codified.loinc_mappings`, `codified.umls_mappings` |
# MAGIC | **Reference ontologies** | `reference.umls_concepts`, `reference.snomed_hierarchy`, `reference.snomed_icd10_map`, `reference.rxnorm_concepts`, `reference.clinical_abbreviations`, `reference.medical_dictionary` |
# MAGIC | **Feedback & evaluation** | `feedback.human_corrections`, `feedback.gold_annotations`, `feedback.recall_metrics`, `feedback.error_patterns` |
# MAGIC | **Storage** | `raw.chart_pdfs` UC Volume |
# MAGIC
# MAGIC **Next steps:**
# MAGIC - Run `02_load_reference_codes` to load ICD-10-CM and LOINC reference data
# MAGIC - Run `02b_load_medical_ontologies` to load UMLS, SNOMED, RxNorm, and build the unified medical dictionary
