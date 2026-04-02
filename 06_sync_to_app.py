# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Sync to Lakebase (OPTIONAL)
# MAGIC
# MAGIC **This notebook is optional.** It syncs Delta Lake codification results into
# MAGIC Lakebase (Databricks-managed Postgres) for the review application.
# MAGIC
# MAGIC **If you are NOT using Lakebase**, the review app can query Delta tables directly
# MAGIC via a SQL Warehouse connection. Skip this notebook entirely.
# MAGIC
# MAGIC ### What this does
# MAGIC - Connects to Lakebase via Databricks SDK + REST API
# MAGIC - Syncs `charts`, `entities`, `icd10_mappings`, and `loinc_mappings`
# MAGIC - Handles the integer-to-string ID mapping between Lakebase (serial PK) and Delta (string UUID)
# MAGIC - Updates pipeline stats for the app dashboard
# MAGIC
# MAGIC ### Prerequisites
# MAGIC - A Lakebase project named `medical-codification` with a `production` branch
# MAGIC - Tables must already exist in Lakebase (created via the app deployment)
# MAGIC
# MAGIC ### Note on SDK support
# MAGIC As of 2026, the Databricks SDK does not have a `.postgres` attribute for Lakebase.
# MAGIC We use the REST API directly (`/api/2.0/postgres/...`) with SDK-managed authentication.
# MAGIC
# MAGIC **Estimated runtime:** ~1-2 minutes

# COMMAND ----------

# MAGIC %pip install psycopg2-binary
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
# MAGIC ## Connect to Lakebase
# MAGIC
# MAGIC Uses Databricks SDK `WorkspaceClient` for authentication, then the REST API
# MAGIC for Lakebase endpoint discovery and credential generation.

# COMMAND ----------

import os
import requests
import psycopg2
from psycopg2.extras import execute_values
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Get workspace host and token for REST API calls
ws_host = w.config.host.rstrip('/')
auth_header = f"Bearer {w.config.token}"

# Lakebase project configuration
project = "medical-codification"
branch_path = f"projects/{project}/branches/production"
endpoint_path = f"{branch_path}/endpoints/primary"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discover Lakebase Endpoint

# COMMAND ----------

# Get endpoint host
resp = requests.get(
    f"{ws_host}/api/2.0/postgres/{branch_path}/endpoints",
    headers={"Authorization": auth_header}
)
eps = resp.json()

# Handle different response formats
if isinstance(eps, list):
    host = eps[0]['status']['hosts']['host']
elif 'endpoints' in eps:
    host = eps['endpoints'][0]['status']['hosts']['host']
else:
    host = eps.get('status', {}).get('hosts', {}).get('host', '')

print(f"  Lakebase host: {host}")

# Generate short-lived credential
resp = requests.post(
    f"{ws_host}/api/2.0/postgres/{endpoint_path}:generateCredential",
    headers={"Authorization": auth_header},
    json={}
)
token = resp.json()['token']

# Get current user for Lakebase auth
me = w.current_user.me()
email = me.user_name
print(f"  User: {email}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Establish Connection

# COMMAND ----------

conn = psycopg2.connect(
    host=host,
    port=5432,
    database='medical_codification',
    user=email,
    password=token,
    sslmode='require'
)
cur = conn.cursor()
print("  Connected to Lakebase")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Sync Charts

# COMMAND ----------

charts = spark.sql(f"""
    SELECT chart_id, patient_id, file_name, provider, facility, chart_date,
           SUBSTRING(raw_text, 1, 3000) AS raw_text
    FROM {CATALOG}.raw.charts
""").collect()

cur.execute("TRUNCATE TABLE charts CASCADE")
rows = []
for i, r in enumerate(charts, 1):
    rows.append((i, r.patient_id, r.file_name, r.provider, r.facility, r.chart_date, r.raw_text))

execute_values(cur,
    "INSERT INTO charts (chart_id, patient_id, file_name, provider, facility, chart_date, raw_text) VALUES %s",
    rows)
conn.commit()
print(f"  {len(rows)} charts synced")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sync Entities

# COMMAND ----------

entities = spark.sql(f"""
    SELECT entity_id, chart_id, entity_type, entity_text, confidence,
           specimen_type, method, timing, value, unit
    FROM {CATALOG}.extracted.entities
""").collect()

cur.execute("TRUNCATE TABLE entities CASCADE")

# Build chart_id mapping (Delta string IDs -> Lakebase integer IDs)
delta_chart_ids = spark.sql(f"SELECT DISTINCT chart_id FROM {CATALOG}.raw.charts").collect()
chart_map = {r.chart_id: i+1 for i, r in enumerate(delta_chart_ids)}

rows = []
for i, r in enumerate(entities, 1):
    rows.append((
        i, chart_map.get(r.chart_id, 1), r.entity_type, r.entity_text,
        float(r.confidence) if r.confidence else 0.9,
        r.specimen_type, r.method, r.timing, r.value, r.unit, r.entity_id
    ))

execute_values(cur,
    """INSERT INTO entities
       (entity_id, chart_id, entity_type, entity_text, confidence,
        specimen_type, method, timing, value, unit, delta_entity_id)
       VALUES %s""",
    rows)
conn.commit()

# Build entity_id mapping for downstream tables
ent_map = {r.entity_id: i+1 for i, r in enumerate(entities)}
print(f"  {len(rows)} entities synced")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sync ICD-10 Mappings

# COMMAND ----------

icd10 = spark.sql(f"""
    SELECT entity_id, chart_id, entity_text, icd10_code, icd10_description, confidence,
           is_specific, resolution_path, r1_code,
           SUBSTRING(r1_reasoning, 1, 500) AS r1_reasoning,
           r2_verdict, r2_code,
           SUBSTRING(r2_reasoning, 1, 500) AS r2_reasoning,
           arbiter_code,
           SUBSTRING(arbiter_reasoning, 1, 500) AS arbiter_reasoning
    FROM {CATALOG}.codified.icd10_mappings
""").collect()

cur.execute("TRUNCATE TABLE icd10_mappings")
rows = []
for i, r in enumerate(icd10, 1):
    rows.append((
        i,
        ent_map.get(r.entity_id, 1),
        chart_map.get(r.chart_id, 1),
        r.entity_text or '',
        r.icd10_code or '',
        r.icd10_description or '',
        float(r.confidence) if r.confidence else 0.9,
        bool(r.is_specific) if r.is_specific is not None else True,
        r.resolution_path or 'R1_R2_AGREE',
        r.r1_code or '',
        r.r1_reasoning or '',
        r.r2_verdict or '',
        r.r2_code or '',
        r.r2_reasoning or '',
        r.arbiter_code if r.arbiter_code else None,
        r.arbiter_reasoning if r.arbiter_reasoning else None,
        r.r1_reasoning or '',  # reasoning field
    ))

execute_values(cur,
    """INSERT INTO icd10_mappings
       (mapping_id, entity_id, chart_id, entity_text, icd10_code, icd10_description,
        confidence, is_specific, resolution_path, r1_code, r1_reasoning,
        r2_verdict, r2_code, r2_reasoning, arbiter_code, arbiter_reasoning, reasoning)
       VALUES %s""",
    rows)
conn.commit()
print(f"  {len(rows)} ICD-10 mappings synced")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sync LOINC Mappings

# COMMAND ----------

loinc = spark.sql(f"""
    SELECT entity_id, chart_id, entity_text, loinc_code, loinc_long_name, confidence,
           resolution_path, r1_code,
           SUBSTRING(r1_reasoning, 1, 500) AS r1_reasoning,
           r2_verdict, r2_code,
           SUBSTRING(r2_reasoning, 1, 500) AS r2_reasoning,
           arbiter_code,
           SUBSTRING(arbiter_reasoning, 1, 500) AS arbiter_reasoning,
           specimen_type, method, timing
    FROM {CATALOG}.codified.loinc_mappings
""").collect()

cur.execute("TRUNCATE TABLE loinc_mappings")
rows = []
for i, r in enumerate(loinc, 1):
    rows.append((
        i,
        ent_map.get(r.entity_id, 1),
        chart_map.get(r.chart_id, 1),
        r.entity_text or '',
        r.loinc_code or '',
        r.loinc_long_name or '',
        float(r.confidence) if r.confidence else 0.9,
        r.resolution_path or 'R1_R2_AGREE',
        r.r1_code or '',
        r.r1_reasoning or '',
        r.r2_verdict or '',
        r.r2_code or '',
        r.r2_reasoning or '',
        r.arbiter_code if r.arbiter_code else None,
        r.arbiter_reasoning if r.arbiter_reasoning else None,
        r.specimen_type if r.specimen_type else None,
        r.method if r.method else None,
        r.timing if r.timing else None,
        r.r1_reasoning or '',  # reasoning field
    ))

execute_values(cur,
    """INSERT INTO loinc_mappings
       (mapping_id, entity_id, chart_id, entity_text, loinc_code, loinc_description,
        confidence, resolution_path, r1_code, r1_reasoning,
        r2_verdict, r2_code, r2_reasoning, arbiter_code, arbiter_reasoning,
        specimen_type, method, timing, reasoning)
       VALUES %s""",
    rows)
conn.commit()
print(f"  {len(rows)} LOINC mappings synced")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update Pipeline Stats

# COMMAND ----------

cur.execute("""
    UPDATE pipeline_stats SET
        total_charts = (SELECT COUNT(*) FROM charts),
        total_entities = (SELECT COUNT(*) FROM entities),
        total_icd10_mappings = (SELECT COUNT(*) FROM icd10_mappings),
        total_loinc_mappings = (SELECT COUNT(*) FROM loinc_mappings),
        last_refreshed = NOW()
""")
conn.commit()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Sync

# COMMAND ----------

for tbl in ['charts', 'entities', 'icd10_mappings', 'loinc_mappings']:
    cur.execute(f"SELECT COUNT(*) FROM {tbl}")
    print(f"  {tbl}: {cur.fetchone()[0]}")

print("")
cur.execute("SELECT resolution_path, COUNT(*) FROM icd10_mappings GROUP BY resolution_path ORDER BY COUNT(*) DESC")
print("ICD-10 resolution paths:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

cur.execute("SELECT resolution_path, COUNT(*) FROM loinc_mappings GROUP BY resolution_path ORDER BY COUNT(*) DESC")
print("\nLOINC resolution paths:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

cur.close()
conn.close()
print("\n  Sync complete -- app will reflect real data on next refresh")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Query Delta Directly
# MAGIC
# MAGIC If you are **not using Lakebase**, the review app can query Delta tables directly
# MAGIC via a Databricks SQL Warehouse. Use the Databricks SQL Connector for Python:
# MAGIC
# MAGIC ```python
# MAGIC from databricks import sql
# MAGIC
# MAGIC connection = sql.connect(
# MAGIC     server_hostname="your-workspace.cloud.databricks.com",
# MAGIC     http_path="/sql/1.0/warehouses/your-warehouse-id",
# MAGIC     access_token="your-token"
# MAGIC )
# MAGIC
# MAGIC cursor = connection.cursor()
# MAGIC cursor.execute(f"SELECT * FROM {CATALOG}.codified.icd10_mappings WHERE confidence < 0.85")
# MAGIC results = cursor.fetchall()
# MAGIC ```
# MAGIC
# MAGIC This approach works with any app framework (Streamlit, Gradio, FastAPI)
# MAGIC and avoids the Lakebase dependency entirely.
