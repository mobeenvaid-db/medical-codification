"""
Codification analytics routes - ICD-10, LOINC insights and chart-to-code lineage.

SQL uses ANSI-compatible syntax for dual-mode (Postgres + Spark SQL).
"""

from __future__ import annotations

import datetime
from collections import OrderedDict
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, HTTPException

from server.db import db

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


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


# ---------------------------------------------------------------------------
# ICD-10
# ---------------------------------------------------------------------------

@router.get("/icd10")
async def icd10_analytics():
    """Top ICD-10 codes and specificity statistics."""

    top_codes = await db.fetch(
        """
        SELECT
            im.icd10_code AS code,
            ic.display_name AS description,
            COUNT(*) AS count,
            ROUND(AVG(im.confidence), 3) AS avg_confidence
        FROM icd10_mappings im
        LEFT JOIN icd10_codes ic ON ic.code = im.icd10_code
        GROUP BY im.icd10_code, ic.display_name
        ORDER BY count DESC
        LIMIT 15
        """
    )

    specificity_rows = await db.fetch(
        """
        SELECT
            CASE
                WHEN im.is_specific THEN 'specific'
                ELSE 'unspecified'
            END AS specificity,
            COUNT(*) AS count
        FROM icd10_mappings im
        GROUP BY
            CASE
                WHEN im.is_specific THEN 'specific'
                ELSE 'unspecified'
            END
        """
    )
    specificity = {"specific": 0, "unspecified": 0}
    for row in specificity_rows:
        specificity[row["specificity"]] = row["count"]

    recent = await db.fetch(
        """
        SELECT
            im.mapping_id AS id,
            e.entity_text,
            im.icd10_code AS code,
            ic.display_name AS description,
            im.confidence,
            im.reasoning
        FROM icd10_mappings im
        LEFT JOIN entities e ON e.entity_id = im.entity_id
        LEFT JOIN icd10_codes ic ON ic.code = im.icd10_code
        ORDER BY im.created_at DESC
        LIMIT 20
        """
    )

    return _serialize({
        "top_codes": top_codes,
        "specificity": specificity,
        "recent_mappings": recent,
    })


# ---------------------------------------------------------------------------
# LOINC
# ---------------------------------------------------------------------------

@router.get("/loinc")
async def loinc_analytics():
    """Top LOINC codes and disambiguation showcase."""

    top_codes = await db.fetch(
        """
        SELECT
            lm.loinc_code AS code,
            lc.display_name AS description,
            lc.component,
            COUNT(*) AS count,
            ROUND(AVG(lm.confidence), 3) AS avg_confidence
        FROM loinc_mappings lm
        LEFT JOIN loinc_codes lc ON lc.code = lm.loinc_code
        GROUP BY lm.loinc_code, lc.display_name, lc.component
        ORDER BY count DESC
        LIMIT 15
        """
    )

    recent = await db.fetch(
        """
        SELECT
            lm.mapping_id AS id,
            e.entity_text,
            lm.loinc_code AS code,
            lc.display_name AS description,
            lc.method,
            lc.system AS specimen,
            lm.confidence
        FROM loinc_mappings lm
        LEFT JOIN entities e ON e.entity_id = lm.entity_id
        LEFT JOIN loinc_codes lc ON lc.code = lm.loinc_code
        ORDER BY lm.created_at DESC
        LIMIT 20
        """
    )

    disambiguation = await db.fetch(
        """
        SELECT
            lm.loinc_code,
            lc.display_name,
            lc.component,
            COALESCE(lc.method, lm.method, '') AS method,
            COALESCE(lc.system, lm.specimen_type, '') AS specimen,
            COUNT(*) AS occurrences,
            ROUND(AVG(lm.confidence), 3) AS avg_confidence,
            MIN(lm.confidence) AS min_confidence,
            MAX(lm.confidence) AS max_confidence
        FROM loinc_mappings lm
        JOIN loinc_codes lc ON lc.code = lm.loinc_code
        WHERE lm.loinc_code NOT IN ('NO_MAP', 'NO VALID MAPPING', 'NO_VALID_MAPPING', 'UNMAPPABLE', 'not found', '')
          AND lc.component != ''
          AND lc.component IN (
            SELECT lc2.component
            FROM loinc_mappings lm2
            JOIN loinc_codes lc2 ON lc2.code = lm2.loinc_code
            GROUP BY lc2.component
            HAVING COUNT(DISTINCT lm2.loinc_code) > 1
        )
        GROUP BY lm.loinc_code, lc.display_name, lc.component,
                 COALESCE(lc.method, lm.method, ''),
                 COALESCE(lc.system, lm.specimen_type, '')
        ORDER BY lc.component, COALESCE(lc.method, lm.method, '')
        """
    )

    grouped: OrderedDict[str, list] = OrderedDict()
    for row in disambiguation:
        analyte = row.get("component", "Unknown")
        if analyte not in grouped:
            grouped[analyte] = []
        grouped[analyte].append({
            "loinc_code": row.get("loinc_code"),
            "loinc_description": row.get("display_name", ""),
            "method": row.get("method", "") or "Not specified",
            "specimen": row.get("specimen", "") or "Not specified",
            "occurrences": row.get("occurrences", 1),
            "confidence": row.get("avg_confidence", row.get("confidence")),
            "confidence_range": f"{row.get('min_confidence', '')}-{row.get('max_confidence', '')}",
        })

    showcase = [{"analyte": k, "entries": v} for k, v in grouped.items()]

    return _serialize({
        "top_codes": top_codes,
        "disambiguation_showcase": showcase,
        "recent_mappings": recent,
    })


# ---------------------------------------------------------------------------
# Chart-to-Code Lineage
# ---------------------------------------------------------------------------

@router.get("/lineage/{chart_id}")
async def chart_lineage(chart_id: int):
    """Full chart -> entity -> code lineage for a single chart."""

    chart = await db.fetchrow(
        """
        SELECT chart_id, patient_id, encounter_date, raw_text, created_at
        FROM charts
        WHERE chart_id = $1
        """,
        chart_id,
    )
    if not chart:
        raise HTTPException(status_code=404, detail="Chart not found")

    entities = await db.fetch(
        """
        SELECT entity_id, entity_text, entity_type, start_offset, end_offset
        FROM entities
        WHERE chart_id = $1
        ORDER BY start_offset
        """,
        chart_id,
    )

    icd10 = await db.fetch(
        """
        SELECT
            im.mapping_id,
            im.entity_id,
            im.icd10_code,
            ic.display_name,
            im.confidence,
            im.is_specific
        FROM icd10_mappings im
        LEFT JOIN icd10_codes ic ON ic.code = im.icd10_code
        WHERE im.chart_id = $1
        ORDER BY im.confidence DESC
        """,
        chart_id,
    )

    loinc = await db.fetch(
        """
        SELECT
            lm.mapping_id,
            lm.entity_id,
            lm.loinc_code,
            lc.display_name,
            lc.component,
            lm.confidence
        FROM loinc_mappings lm
        LEFT JOIN loinc_codes lc ON lc.code = lm.loinc_code
        WHERE lm.chart_id = $1
        ORDER BY lm.confidence DESC
        """,
        chart_id,
    )

    return _serialize(
        {
            "chart": chart,
            "entities": entities,
            "icd10_mappings": icd10,
            "loinc_mappings": loinc,
        }
    )
