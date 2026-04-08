# Clinical NLP & Medical Codification Pipeline

A production-grade clinical entity extraction and medical coding pipeline built on
Databricks. Processes clinical charts through multi-layer extraction, assertion
classification, and automated ICD-10-CM / LOINC code assignment with full audit trails.

---

## Architecture

```
  CLINICAL CHARTS (PDF / EHR Text)
           |
           v
  +-----------------------------------------------------------+
  |  Document Intelligence                        [04a]       |
  |  Smart OCR bypass + clinical note section segmentation    |
  +-----------------------------------------------------------+
           |
    +------+------+
    |             |
    v             v
  +----------+ +------------------------------------------+
  | Dictionary| | Consolidated LLM Extraction       [04c] |
  | Match     | | Entity detection + assertion             |
  | [04b]     | | classification in a single call/chart    |
  +----------+ +------------------------------------------+
    |             |
    +------+------+
           |
           v
  +-----------------------------------------------------------+
  |  Ensemble Merge                               [04e]       |
  |  Union-based entity fusion with source-aware confidence   |
  +-----------------------------------------------------------+
           |
           v
  +-----------------------------------------------------------+
  |  Medical Coding                               [05a]       |
  |  Tier 1: SQL ontology joins (zero LLM cost)              |
  |  Tier 2: LLM coding (one call per chart for remainder)   |
  |  Code validation against ICD-10 / LOINC reference tables  |
  +-----------------------------------------------------------+
           |
           v
     VALIDATED ICD-10-CM + LOINC CODES
     with confidence scores + audit trail
           |
           v
  +-----------------------------------------------------------+
  |  Active Learning + Quality Analytics      [07, 08]        |
  |  Gold set management, recall/precision tracking,          |
  |  error pattern detection, annotation prioritization       |
  +-----------------------------------------------------------+
```

---

## Notebooks: Required vs Optional

### Required (Core Pipeline)

These 5 notebooks form the minimum viable pipeline. Run them in order.

| # | Notebook | What It Does | Can You Skip It? |
|---|----------|-------------|------------------|
| 1 | `01_setup.py` | Creates schemas and tables | **No** -- run once to set up infrastructure. Safe to re-run (uses `IF NOT EXISTS`). |
| 2 | `04a_document_intelligence.py` | Parses clinical notes into sections | **No** -- downstream extraction depends on `document_sections` table. |
| 3 | `04c_llm_entity_extraction.py` | Extracts entities + classifies assertions | **No** -- this is the core extraction engine. |
| 4 | `04e_ensemble_merge.py` | Merges dictionary + LLM entities | **No** -- produces `merged_entities` which the coding step reads. If you skip dictionary extraction (04b), this still works with LLM entities only. |
| 5 | `05a_codify_entities.py` | Assigns ICD-10 and LOINC codes | **No** -- this is the output you care about. |

### Optional (Enhance Quality and Coverage)

| # | Notebook | What It Does | When to Use |
|---|----------|-------------|-------------|
| 6 | `02b_load_medical_ontologies.py` | Loads UMLS, SNOMED, RxNorm, abbreviation dictionary | **Recommended.** Builds the medical dictionary that powers dictionary extraction (04b) and ontology-based coding (Tier 1 in 05a). Without it, all coding goes through the LLM. With it, 60-70% of coding is SQL joins -- faster and cheaper. |
| 7 | `04b_dictionary_extraction.py` | Dictionary-based entity detection | **Recommended.** Adds a deterministic extraction layer that catches abbreviations and standard terminology the LLM might miss. Requires the medical dictionary from 02b. |
| 8 | `07_active_learning.py` | Gold set management, recall measurement, error detection | **Optional.** Only useful once you have human-annotated gold standard charts to measure against. Skippable for initial deployment. |
| 9 | `08_recall_metrics.py` | Quality analytics and trending | **Optional.** Analytical reporting on pipeline quality. Depends on 07 having run at least once. |

### Minimum Viable Pipeline (4 notebooks)

If you want the fastest possible path to ICD-10/LOINC codes:

```
01_setup.py --> 04a_document_intelligence.py --> 04c_llm_entity_extraction.py --> 05a_codify_entities.py
```

This skips dictionary extraction, ensemble merge (04e handles single-source gracefully), and feedback loops. You get LLM-only extraction and LLM-only coding. Add the optional notebooks incrementally as needed.

---

## Configuration & Portability

### What's Configurable

| Setting | How to Configure | Default |
|---------|-----------------|---------|
| **Catalog name** | `CATALOG` widget on every notebook | `mv_catalog` |
| **Model endpoint** | `MODEL` variable at top of 04c and 05a | `databricks-claude-sonnet-4-6` |
| **Completeness check** | `ENABLE_COMPLETENESS_CHECK` widget on 04c | `false` |
| **NLM API key** | Databricks secret `medical-ontology/umls-api-key` | Empty (skip ontology download) |

### Adapting to Your Environment

**If you already have a catalog and schemas:**
Set the `CATALOG` widget to your catalog name. The pipeline creates 5 schemas (`raw`, `extracted`, `codified`, `reference`, `feedback`) inside that catalog. If your catalog already has schemas with those names, the pipeline's `CREATE SCHEMA IF NOT EXISTS` will simply use them.

**If you already have clinical notes in a table:**
The pipeline reads from `{CATALOG}.raw.charts` which expects these columns:
- `chart_id` (STRING) -- unique identifier
- `raw_text` (STRING) -- the clinical note text
- `provider` (STRING, optional) -- provider name
- `facility` (STRING, optional) -- facility name
- `chart_date` (DATE, optional) -- date of chart

If your notes are in a different table, you have two options:
1. Create a view: `CREATE VIEW {CATALOG}.raw.charts AS SELECT id AS chart_id, note_text AS raw_text, ... FROM your_schema.your_table`
2. Modify the `raw.charts` references in 04a and 04c to point to your table directly

**If you already have ICD-10 / LOINC reference data:**
The pipeline creates its own reference tables (`reference.icd10_codes_full`, `reference.loinc_codes_full`). If you already have these loaded elsewhere, you can either:
1. Skip the reference data loading (notebook 02 from the original pipeline) and create views that map your tables to the expected schema
2. Point the codification notebook (05a) to your tables by changing the table references

The key columns the pipeline expects from ICD-10 reference:
- `code` (STRING) -- formatted code (e.g., "E11.9")
- `code_raw` (STRING) -- unformatted code (e.g., "E119")
- `description` (STRING)
- `is_billable` (BOOLEAN)
- `category` (STRING) -- 3-character prefix

The key columns from LOINC reference:
- `loinc_code` (STRING)
- `long_name` (STRING)
- `component` (STRING)
- `system_specimen` (STRING)
- `method` (STRING)

**If you want to use a different LLM:**
Change the `MODEL` variable in `04c_llm_entity_extraction.py` and `05a_codify_entities.py`. Any model available through Databricks Foundation Model API via `ai_query` will work. The prompts are model-agnostic.

**If you're running on a non-serverless cluster:**
The pipeline is designed for serverless but works on classic compute too. If you have GPU nodes, notebook `04c` can be modified to use scispaCy via `mapPartitions` instead of `ai_query` for the NER layer (see the `04c_ner_model_extraction.py` variant in the extended notebooks).

### Schema Names

The pipeline uses these fixed schema names within your catalog:
- `raw` -- ingested chart data
- `extracted` -- extraction pipeline output
- `codified` -- final code assignments
- `reference` -- lookup tables (ICD-10, LOINC, dictionary)
- `feedback` -- human corrections and quality metrics

These are currently hardcoded. To change them, find-and-replace across the notebooks (e.g., replace `.extracted.` with `.clinical_nlp.`).

---

## How It Works

### Multi-Layer Entity Extraction

Instead of relying on a single model for entity extraction, the pipeline uses two
complementary layers and takes the union:

- **Dictionary Layer (04b):** Deterministic string matching against a medical term
  dictionary built from ICD-10 descriptions, LOINC names, UMLS synonyms, SNOMED
  terms, RxNorm drug names, and 340+ clinical abbreviations. No LLM calls. Catches
  standard clinical terminology including abbreviations (HTN, DM, CHF, SOB, etc.).

- **LLM Layer (04c):** A single comprehensive Claude call per chart that extracts
  all clinical entities (diagnoses, labs, medications, vitals) AND classifies each
  for assertion status (present/absent/possible/historical/family), negation,
  temporality, and experiencer attribution. Catches implicit entities and
  context-dependent mentions that dictionary matching misses.

The ensemble merge (04e) takes the union of both layers, deduplicates by normalized
text, and assigns confidence scores based on how many layers detected each entity.

### Assertion Classification

Every extracted entity is classified for:
- **Assertion Status:** PRESENT, ABSENT, POSSIBLE, CONDITIONAL, HISTORICAL, FAMILY
- **Negation:** "denies chest pain" -> ABSENT, negation_detected=true
- **Temporality:** CURRENT, HISTORICAL, FUTURE_PLANNED
- **Experiencer:** PATIENT, FAMILY_MEMBER, OTHER

Only entities classified as PRESENT + PATIENT + CURRENT proceed to medical coding.
This prevents coding negated findings, family history, and historical conditions.

### Two-Tier Medical Coding

- **Tier 1 (SQL Joins):** Entities that match the medical dictionary get codes
  via direct lookup and validation against ICD-10/LOINC reference tables. Zero
  LLM cost. With UMLS/SNOMED loaded, handles 60-70% of entities.

- **Tier 2 (LLM):** Remaining entities are grouped by chart and sent to Claude
  in a single call per chart for ICD-10-CM and LOINC code assignment. Codes are
  validated against reference tables; invalid codes get confidence penalties.

### Incremental Processing

Every notebook supports incremental runs. On re-execution, it detects which charts
have already been processed and skips them. This means:
- First run: processes all charts (~13 min for 100 charts)
- Subsequent runs with new charts: processes only the new ones
- Re-run with no new data: completes in seconds

### Active Learning

The pipeline includes a feedback loop for continuous improvement:
- Import human gold-standard annotations (CSV/JSON)
- Measure recall, precision, and F1 against the gold set
- Detect systematic error patterns
- Prioritize charts for annotation using uncertainty and disagreement sampling
- Generate dictionary and prompt improvement recommendations

---

## Performance

| Stage | 100 Charts | Notes |
|-------|-----------|-------|
| Setup + Reference Data | ~2 min | One-time; subsequent runs skip |
| Document Intelligence | ~45s | Section segmentation |
| Dictionary Extraction | ~15s | Pure SQL joins, no LLM |
| LLM Extraction + Assertions | ~100s | One ai_query per chart |
| Ensemble Merge | ~12s | SQL dedup + scoring |
| Medical Coding | ~6 min | Tier 1 SQL + Tier 2 LLM |
| Active Learning + Analytics | ~100s | SQL aggregations |
| **Total** | **~13 min** | Serverless compute |

Bottleneck is Foundation Model API throughput for LLM stages. With UMLS/SNOMED
loaded, the coding stage (05a) shifts from mostly-LLM to mostly-SQL, reducing
total runtime further.

---

## Review Application

The pipeline includes a full-stack review application:

- **Backend:** FastAPI with dual-mode database support:
  - **Lakebase mode:** Reads from managed Postgres for low-latency queries
  - **Warehouse mode:** Reads directly from Delta tables via SQL Warehouse
  - V2 analytics always read from Delta via SQL Warehouse regardless of mode

- **Frontend:** React + TypeScript + Tailwind CSS + Recharts with 5 tabs:
  - Pipeline Overview -- KPIs, entity distribution, confidence histograms
  - Review Queue -- Human-in-the-loop review with accept/override actions
  - ICD-10 Analytics -- Top codes, specificity, resolution breakdown
  - LOINC Analytics -- Top codes, disambiguation showcase
  - V2 Analytics -- Extraction layer contribution, assertion breakdown, coding tiers, quality metrics

### App Configuration

| Setting | Environment Variable | Default |
|---------|---------------------|---------|
| Data mode | `DATA_MODE` | `lakebase` |
| Catalog | `CATALOG` | `mv_catalog` |
| SQL Warehouse ID | `WAREHOUSE_ID` | (required for V2 analytics) |

---

## Data Model

### Schemas

| Schema | Purpose |
|--------|---------|
| `raw` | Ingested chart metadata and PDFs |
| `extracted` | Multi-layer entity extraction results |
| `codified` | Final ICD-10 and LOINC code assignments |
| `reference` | ICD-10, LOINC, UMLS, SNOMED, RxNorm, abbreviations, unified dictionary |
| `feedback` | Human corrections, gold annotations, recall metrics, error patterns |

### Tables Created (22 total)

**Pipeline:** `raw.charts`, `extracted.document_sections`, `extracted.dictionary_entities`, `extracted.ner_entities`, `extracted.llm_entities`, `extracted.merged_entities`, `extracted.entity_assertions`, `codified.icd10_mappings`, `codified.loinc_mappings`, `codified.umls_mappings`

**Reference:** `reference.icd10_codes_full`, `reference.loinc_codes_full`, `reference.umls_concepts`, `reference.snomed_hierarchy`, `reference.snomed_icd10_map`, `reference.rxnorm_concepts`, `reference.clinical_abbreviations`, `reference.medical_dictionary`

**Feedback:** `feedback.human_corrections`, `feedback.gold_annotations`, `feedback.recall_metrics`, `feedback.error_patterns`

---

## Reference Data

| Dataset | Records | Source | Required? | Auto-Download |
|---------|---------|-------|-----------|---------------|
| ICD-10-CM | ~74K billable | CDC/CMS | Yes (for code validation) | Yes |
| LOINC | ~109K codes | loinc.org | Yes (for code validation) | Yes |
| Clinical Abbreviations | 340+ | Hardcoded | Included automatically | N/A |
| UMLS Metathesaurus | ~2.3M concepts | NLM UTS API | No (improves coverage) | Yes (with API key) |
| SNOMED-CT US Edition | ~350K concepts | NLM UTS API | No (enables ontology coding) | Yes (with API key) |
| RxNorm | ~115K drug concepts | NLM UTS API | No (improves med matching) | Yes (with API key) |

---

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Serverless SQL warehouse or all-purpose compute (DBR 14.3+)
- Foundation Model API access (Claude Sonnet 4.6 via `ai_query`, or any supported model)
- (Recommended) NLM UTS API key for UMLS/SNOMED/RxNorm -- free at https://uts.nlm.nih.gov
