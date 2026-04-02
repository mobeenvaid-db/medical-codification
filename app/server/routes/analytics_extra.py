"""
Extended analytics routes - rich data for dashboards.

SQL uses ANSI-compatible syntax for dual-mode (Postgres + Spark SQL).
Avoids Postgres-only casts like ::NUMERIC and uses CASE WHEN for bucketing
instead of FLOOR() math which behaves differently in Spark SQL.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter

from server.db import db

router = APIRouter(prefix="/api/analytics", tags=["analytics-extra"])


def _serialize(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


@router.get("/pipeline/extended")
async def pipeline_extended():
    """Rich pipeline analytics - entity breakdown, provider stats, processing timeline."""

    top_entities = await db.fetch("""
        SELECT entity_text, entity_type, COUNT(*) AS frequency
        FROM entities
        GROUP BY entity_text, entity_type
        ORDER BY frequency DESC
        LIMIT 20
    """)

    entity_types = await db.fetch("""
        SELECT entity_type, COUNT(*) AS count
        FROM entities
        GROUP BY entity_type
        ORDER BY count DESC
    """)

    providers = await db.fetch("""
        SELECT c.provider, COUNT(DISTINCT c.chart_id) AS charts,
               COUNT(DISTINCT e.entity_id) AS entities
        FROM charts c
        LEFT JOIN entities e ON e.chart_id = c.chart_id
        GROUP BY c.provider
        ORDER BY charts DESC
    """)

    facilities = await db.fetch("""
        SELECT c.facility, COUNT(DISTINCT c.chart_id) AS charts
        FROM charts c
        GROUP BY c.facility
        ORDER BY charts DESC
    """)

    timeline = await db.fetch("""
        SELECT DATE(created_at) AS date, COUNT(*) AS charts
        FROM charts
        GROUP BY DATE(created_at)
        ORDER BY date
    """)

    resolution_summary = await db.fetch("""
        SELECT resolution_path, code_type, COUNT(*) AS count FROM (
            SELECT resolution_path, 'ICD-10' AS code_type FROM icd10_mappings
            UNION ALL
            SELECT resolution_path, 'LOINC' AS code_type FROM loinc_mappings
        ) t
        GROUP BY resolution_path, code_type
        ORDER BY count DESC
    """)

    # Confidence histogram using CASE WHEN for Spark SQL compatibility
    confidence_hist = await db.fetch("""
        SELECT bucket, SUM(icd10_count) AS icd10, SUM(loinc_count) AS loinc FROM (
            SELECT
                CASE
                    WHEN confidence >= 0.9 AND confidence < 1.0 THEN 0.9
                    WHEN confidence >= 0.8 AND confidence < 0.9 THEN 0.8
                    WHEN confidence >= 0.7 AND confidence < 0.8 THEN 0.7
                    WHEN confidence >= 0.6 AND confidence < 0.7 THEN 0.6
                    WHEN confidence >= 0.5 AND confidence < 0.6 THEN 0.5
                    WHEN confidence >= 0.4 AND confidence < 0.5 THEN 0.4
                    WHEN confidence >= 0.3 AND confidence < 0.4 THEN 0.3
                    WHEN confidence >= 0.2 AND confidence < 0.3 THEN 0.2
                    WHEN confidence >= 0.1 AND confidence < 0.2 THEN 0.1
                    ELSE 0.0
                END AS bucket,
                COUNT(*) AS icd10_count, 0 AS loinc_count
            FROM icd10_mappings WHERE confidence IS NOT NULL
            GROUP BY
                CASE
                    WHEN confidence >= 0.9 AND confidence < 1.0 THEN 0.9
                    WHEN confidence >= 0.8 AND confidence < 0.9 THEN 0.8
                    WHEN confidence >= 0.7 AND confidence < 0.8 THEN 0.7
                    WHEN confidence >= 0.6 AND confidence < 0.7 THEN 0.6
                    WHEN confidence >= 0.5 AND confidence < 0.6 THEN 0.5
                    WHEN confidence >= 0.4 AND confidence < 0.5 THEN 0.4
                    WHEN confidence >= 0.3 AND confidence < 0.4 THEN 0.3
                    WHEN confidence >= 0.2 AND confidence < 0.3 THEN 0.2
                    WHEN confidence >= 0.1 AND confidence < 0.2 THEN 0.1
                    ELSE 0.0
                END
            UNION ALL
            SELECT
                CASE
                    WHEN confidence >= 0.9 AND confidence < 1.0 THEN 0.9
                    WHEN confidence >= 0.8 AND confidence < 0.9 THEN 0.8
                    WHEN confidence >= 0.7 AND confidence < 0.8 THEN 0.7
                    WHEN confidence >= 0.6 AND confidence < 0.7 THEN 0.6
                    WHEN confidence >= 0.5 AND confidence < 0.6 THEN 0.5
                    WHEN confidence >= 0.4 AND confidence < 0.5 THEN 0.4
                    WHEN confidence >= 0.3 AND confidence < 0.4 THEN 0.3
                    WHEN confidence >= 0.2 AND confidence < 0.3 THEN 0.2
                    WHEN confidence >= 0.1 AND confidence < 0.2 THEN 0.1
                    ELSE 0.0
                END AS bucket,
                0, COUNT(*)
            FROM loinc_mappings WHERE confidence IS NOT NULL
            GROUP BY
                CASE
                    WHEN confidence >= 0.9 AND confidence < 1.0 THEN 0.9
                    WHEN confidence >= 0.8 AND confidence < 0.9 THEN 0.8
                    WHEN confidence >= 0.7 AND confidence < 0.8 THEN 0.7
                    WHEN confidence >= 0.6 AND confidence < 0.7 THEN 0.6
                    WHEN confidence >= 0.5 AND confidence < 0.6 THEN 0.5
                    WHEN confidence >= 0.4 AND confidence < 0.5 THEN 0.4
                    WHEN confidence >= 0.3 AND confidence < 0.4 THEN 0.3
                    WHEN confidence >= 0.2 AND confidence < 0.3 THEN 0.2
                    WHEN confidence >= 0.1 AND confidence < 0.2 THEN 0.1
                    ELSE 0.0
                END
        ) t
        GROUP BY bucket
        ORDER BY bucket
    """)

    return _serialize({
        "top_entities": top_entities,
        "entity_types": entity_types,
        "providers": providers,
        "facilities": facilities,
        "timeline": timeline,
        "resolution_summary": resolution_summary,
        "confidence_histogram": confidence_hist,
    })


@router.get("/icd10/extended")
async def icd10_extended():
    """Rich ICD-10 analytics - chapter distribution, disputed codes, confidence spread."""

    chapter_dist = await db.fetch("""
        SELECT
            CASE SUBSTRING(icd10_code, 1, 1)
                WHEN 'A' THEN 'A: Infectious diseases'
                WHEN 'B' THEN 'B: Infectious diseases'
                WHEN 'C' THEN 'C: Neoplasms'
                WHEN 'D' THEN 'D: Blood/immune'
                WHEN 'E' THEN 'E: Endocrine/metabolic'
                WHEN 'F' THEN 'F: Mental/behavioral'
                WHEN 'G' THEN 'G: Nervous system'
                WHEN 'H' THEN 'H: Eye/ear'
                WHEN 'I' THEN 'I: Circulatory'
                WHEN 'J' THEN 'J: Respiratory'
                WHEN 'K' THEN 'K: Digestive'
                WHEN 'L' THEN 'L: Skin'
                WHEN 'M' THEN 'M: Musculoskeletal'
                WHEN 'N' THEN 'N: Genitourinary'
                WHEN 'R' THEN 'R: Symptoms/signs'
                WHEN 'Z' THEN 'Z: Health factors'
                ELSE CONCAT(SUBSTRING(icd10_code, 1, 1), ': Other')
            END AS chapter,
            COUNT(*) AS count
        FROM icd10_mappings
        WHERE icd10_code IS NOT NULL AND icd10_code != ''
        GROUP BY
            CASE SUBSTRING(icd10_code, 1, 1)
                WHEN 'A' THEN 'A: Infectious diseases'
                WHEN 'B' THEN 'B: Infectious diseases'
                WHEN 'C' THEN 'C: Neoplasms'
                WHEN 'D' THEN 'D: Blood/immune'
                WHEN 'E' THEN 'E: Endocrine/metabolic'
                WHEN 'F' THEN 'F: Mental/behavioral'
                WHEN 'G' THEN 'G: Nervous system'
                WHEN 'H' THEN 'H: Eye/ear'
                WHEN 'I' THEN 'I: Circulatory'
                WHEN 'J' THEN 'J: Respiratory'
                WHEN 'K' THEN 'K: Digestive'
                WHEN 'L' THEN 'L: Skin'
                WHEN 'M' THEN 'M: Musculoskeletal'
                WHEN 'N' THEN 'N: Genitourinary'
                WHEN 'R' THEN 'R: Symptoms/signs'
                WHEN 'Z' THEN 'Z: Health factors'
                ELSE CONCAT(SUBSTRING(icd10_code, 1, 1), ': Other')
            END
        ORDER BY count DESC
    """)

    resolution = await db.fetch("""
        SELECT resolution_path, COUNT(*) AS count,
               ROUND(AVG(confidence), 3) AS avg_confidence
        FROM icd10_mappings
        GROUP BY resolution_path
        ORDER BY count DESC
    """)

    disputed = await db.fetch("""
        SELECT icd10_code AS code, entity_text, confidence,
               r1_code, r2_code, resolution_path,
               reasoning
        FROM icd10_mappings
        WHERE resolution_path != 'R1_R2_AGREE'
        ORDER BY confidence ASC
        LIMIT 15
    """)

    # Confidence spread using CASE WHEN buckets for cross-engine compatibility
    confidence_spread = await db.fetch("""
        SELECT
            CASE
                WHEN confidence >= 0.95 THEN 1.0
                WHEN confidence >= 0.85 THEN 0.9
                WHEN confidence >= 0.75 THEN 0.8
                WHEN confidence >= 0.65 THEN 0.7
                WHEN confidence >= 0.55 THEN 0.6
                WHEN confidence >= 0.45 THEN 0.5
                WHEN confidence >= 0.35 THEN 0.4
                WHEN confidence >= 0.25 THEN 0.3
                WHEN confidence >= 0.15 THEN 0.2
                WHEN confidence >= 0.05 THEN 0.1
                ELSE 0.0
            END AS bucket,
            COUNT(*) AS count
        FROM icd10_mappings
        WHERE confidence IS NOT NULL
        GROUP BY
            CASE
                WHEN confidence >= 0.95 THEN 1.0
                WHEN confidence >= 0.85 THEN 0.9
                WHEN confidence >= 0.75 THEN 0.8
                WHEN confidence >= 0.65 THEN 0.7
                WHEN confidence >= 0.55 THEN 0.6
                WHEN confidence >= 0.45 THEN 0.5
                WHEN confidence >= 0.35 THEN 0.4
                WHEN confidence >= 0.25 THEN 0.3
                WHEN confidence >= 0.15 THEN 0.2
                WHEN confidence >= 0.05 THEN 0.1
                ELSE 0.0
            END
        ORDER BY bucket
    """)

    return _serialize({
        "chapter_distribution": chapter_dist,
        "resolution_breakdown": resolution,
        "top_disputed": disputed,
        "confidence_spread": confidence_spread,
    })


@router.get("/loinc/extended")
async def loinc_extended():
    """Rich LOINC analytics - class distribution, method breakdown, specimen types."""

    class_dist = await db.fetch("""
        SELECT
            COALESCE(lc.component, 'Unknown') AS component,
            COUNT(*) AS count
        FROM loinc_mappings lm
        LEFT JOIN loinc_codes lc ON lc.code = lm.loinc_code
        GROUP BY COALESCE(lc.component, 'Unknown')
        ORDER BY count DESC
        LIMIT 15
    """)

    method_dist_raw = await db.fetch("""
        SELECT
            COALESCE(lc.method, lm.method, 'Not specified') AS method_name,
            COUNT(*) AS count
        FROM loinc_mappings lm
        LEFT JOIN loinc_codes lc ON lc.code = lm.loinc_code
        GROUP BY COALESCE(lc.method, lm.method, 'Not specified')
        ORDER BY count DESC
        LIMIT 10
    """)
    method_dist = [{"method": r.get("method_name", ""), "count": r.get("count", 0)} for r in method_dist_raw]

    specimen_dist_raw = await db.fetch("""
        SELECT
            COALESCE(lc.system, lm.specimen_type, 'Not specified') AS specimen_name,
            COUNT(*) AS count
        FROM loinc_mappings lm
        LEFT JOIN loinc_codes lc ON lc.code = lm.loinc_code
        GROUP BY COALESCE(lc.system, lm.specimen_type, 'Not specified')
        ORDER BY count DESC
        LIMIT 10
    """)
    specimen_dist = [{"specimen": r.get("specimen_name", ""), "count": r.get("count", 0)} for r in specimen_dist_raw]

    resolution = await db.fetch("""
        SELECT resolution_path, COUNT(*) AS count,
               ROUND(AVG(confidence), 3) AS avg_confidence
        FROM loinc_mappings
        GROUP BY resolution_path
        ORDER BY count DESC
    """)

    disputed = await db.fetch("""
        SELECT loinc_code AS code, entity_text, confidence,
               r1_code, r2_code, resolution_path,
               reasoning, specimen_type, method
        FROM loinc_mappings
        WHERE resolution_path != 'R1_R2_AGREE'
        ORDER BY confidence ASC
        LIMIT 15
    """)

    return _serialize({
        "class_distribution": class_dist,
        "method_distribution": method_dist,
        "specimen_distribution": specimen_dist,
        "resolution_breakdown": resolution,
        "top_disputed": disputed,
    })
