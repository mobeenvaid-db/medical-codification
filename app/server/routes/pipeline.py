"""
Pipeline overview route - KPI cards, distributions, and recent activity.

SQL is written in ANSI-compatible syntax that works in both Postgres (Lakebase)
and Spark SQL (Warehouse mode).
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter

from server.config import CATALOG
from server.db import db

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


def _serialize(obj: Any) -> Any:
    """Convert Decimal -> float and date/datetime -> isoformat for JSON."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


@router.get("/stats")
async def pipeline_overview():
    """Return all data needed for the pipeline dashboard in one call."""

    # -- KPI cards ----------------------------------------------------------
    kpi_row = await db.fetchrow(
        """
        SELECT
            total_charts,
            total_entities,
            total_icd10_mappings,
            total_loinc_mappings,
            last_refreshed
        FROM pipeline_stats
        ORDER BY last_refreshed DESC
        LIMIT 1
        """
    )
    kpis = _serialize(kpi_row) if kpi_row else {
        "total_charts": 0,
        "total_entities": 0,
        "total_icd10_mappings": 0,
        "total_loinc_mappings": 0,
        "last_refreshed": None,
    }

    # -- Entity distribution by type (pie chart) ----------------------------
    entity_dist_raw = await db.fetch(
        """
        SELECT entity_type, COUNT(*) AS count
        FROM entities
        GROUP BY entity_type
        ORDER BY count DESC
        """
    )
    entity_dist = [{"name": r["entity_type"], "value": r["count"]} for r in entity_dist_raw]

    # -- Confidence distribution for ICD-10 (bar chart) ---------------------
    icd10_confidence = await db.fetch(
        """
        SELECT
            CASE
                WHEN confidence >= 0.9 THEN 'high'
                WHEN confidence >= 0.7 THEN 'medium'
                ELSE 'low'
            END AS bucket,
            COUNT(*) AS count
        FROM icd10_mappings
        GROUP BY
            CASE
                WHEN confidence >= 0.9 THEN 'high'
                WHEN confidence >= 0.7 THEN 'medium'
                ELSE 'low'
            END
        ORDER BY bucket
        """
    )

    # -- Confidence distribution for LOINC (bar chart) ----------------------
    loinc_confidence = await db.fetch(
        """
        SELECT
            CASE
                WHEN confidence >= 0.9 THEN 'high'
                WHEN confidence >= 0.7 THEN 'medium'
                ELSE 'low'
            END AS bucket,
            COUNT(*) AS count
        FROM loinc_mappings
        GROUP BY
            CASE
                WHEN confidence >= 0.9 THEN 'high'
                WHEN confidence >= 0.7 THEN 'medium'
                ELSE 'low'
            END
        ORDER BY bucket
        """
    )

    # -- Resolution path breakdown (multi-pass AI) ----------------------------
    review_status_raw = await db.fetch(
        """
        SELECT resolution_path, COUNT(*) AS count FROM (
            SELECT resolution_path FROM icd10_mappings
            UNION ALL
            SELECT resolution_path FROM loinc_mappings
        ) t
        GROUP BY resolution_path
        """
    )

    # -- Recent processing activity (charts by date) ------------------------
    recent_activity = await db.fetch(
        """
        SELECT
            DATE(created_at) AS processing_date,
            COUNT(*) AS charts_processed
        FROM charts
        GROUP BY DATE(created_at)
        ORDER BY processing_date DESC
        LIMIT 14
        """
    )

    # Build review_status from resolution paths
    review_map = {"auto_approved": 0, "flagged": 0, "needs_review": 0}
    for row in review_status_raw:
        path = row.get("resolution_path", "")
        cnt = row.get("count", 0)
        if path in ("R1_R2_AGREE", "ONTOLOGY_DIRECT"):
            review_map["auto_approved"] += cnt
        elif path in ("ARBITER_CHOSE_R1", "ARBITER_CHOSE_R2", "LLM_ASSIGNED"):
            review_map["flagged"] += cnt
        elif path == "DISPUTED_UNRESOLVED":
            review_map["needs_review"] += cnt
        else:
            review_map["auto_approved"] += cnt

    # Flatten confidence distributions into a single list
    conf_dist = []
    for row in icd10_confidence:
        conf_dist.append({"band": row["bucket"], "icd10": row["count"], "loinc": 0})
    for row in loinc_confidence:
        bucket = row["bucket"]
        matched = False
        for cd in conf_dist:
            if cd["band"] == bucket:
                cd["loinc"] = row["count"]
                matched = True
                break
        if not matched:
            conf_dist.append({"band": bucket, "icd10": 0, "loinc": row["count"]})

    # -- v2: Entity source breakdown from extracted.merged_entities -----------
    v2_entity_sources = None
    try:
        v2_entity_sources = await db.fetch_delta(
            f"""
            SELECT
                CASE
                    WHEN ARRAY_CONTAINS(sources, 'dictionary') AND ARRAY_CONTAINS(sources, 'llm') THEN 'both'
                    WHEN ARRAY_CONTAINS(sources, 'dictionary') THEN 'dictionary_only'
                    WHEN ARRAY_CONTAINS(sources, 'llm') THEN 'llm_only'
                    WHEN ARRAY_CONTAINS(sources, 'ner') THEN 'ner_only'
                    ELSE 'other'
                END AS source_category,
                COUNT(*) AS entity_count
            FROM {CATALOG}.extracted.merged_entities
            GROUP BY 1
            """
        )
    except Exception:
        pass

    # -- v2: Assertion stats from extracted.entity_assertions ---------------
    v2_assertion_stats = None
    try:
        assertion_rows = await db.fetch_delta(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN assertion_status = 'NEGATED' THEN 1 ELSE 0 END) AS negated,
                SUM(CASE WHEN assertion_status = 'HISTORICAL' THEN 1 ELSE 0 END) AS historical,
                SUM(CASE WHEN assertion_status = 'FAMILY' THEN 1 ELSE 0 END) AS family
            FROM {CATALOG}.extracted.entity_assertions
            """
        )
        if assertion_rows:
            row = assertion_rows[0]
            total = row.get("total", 0) or 0
            v2_assertion_stats = {
                "total": total,
                "negation_rate": round((row.get("negated", 0) or 0) / total, 4) if total > 0 else 0,
                "historical_rate": round((row.get("historical", 0) or 0) / total, 4) if total > 0 else 0,
                "family_rate": round((row.get("family", 0) or 0) / total, 4) if total > 0 else 0,
            }
    except Exception:
        pass

    # -- v2: Section distribution from extracted.document_sections ----------
    v2_section_dist = None
    try:
        v2_section_dist = await db.fetch_delta(
            f"""
            SELECT
                section_type,
                COUNT(*) AS section_count,
                COUNT(DISTINCT chart_id) AS charts_with_section
            FROM {CATALOG}.extracted.document_sections
            GROUP BY section_type
            ORDER BY section_count DESC
            """
        )
    except Exception:
        pass

    return _serialize(
        {
            "charts_processed": kpis.get("total_charts", 0),
            "entities_extracted": kpis.get("total_entities", 0),
            "icd10_codes_assigned": kpis.get("total_icd10_mappings", 0),
            "loinc_codes_assigned": kpis.get("total_loinc_mappings", 0),
            "entity_distribution": entity_dist,
            "confidence_distribution": conf_dist,
            "review_status": review_map,
            "recent_activity": recent_activity,
            # v2 fields (None if v2 tables not available)
            "v2_entity_sources": v2_entity_sources,
            "v2_assertion_stats": v2_assertion_stats,
            "v2_section_distribution": v2_section_dist,
        }
    )
