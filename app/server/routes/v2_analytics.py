"""
v2 Analytics routes - extraction layers, assertions, sections, recall metrics,
error patterns, and coding tier breakdown.

All queries target Delta tables via the full catalog path ({CATALOG}.schema.table).
Endpoints return empty results if v2 tables do not exist (backward compatible).
"""

from __future__ import annotations

import datetime
import logging
from decimal import Decimal
from typing import Any

from fastapi import APIRouter

from server.config import CATALOG
from server.db import db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["v2-analytics"])


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
# GET /api/v2/extraction-layers
# ---------------------------------------------------------------------------

@router.get("/extraction-layers")
async def extraction_layers():
    """Extraction layer contribution analysis from merged_entities."""
    try:
        source_categories = await db.fetch_delta(
            f"""
            SELECT
                CASE
                    WHEN ARRAY_CONTAINS(sources, 'dictionary') AND ARRAY_CONTAINS(sources, 'llm') THEN 'both'
                    WHEN ARRAY_CONTAINS(sources, 'dictionary') THEN 'dictionary_only'
                    WHEN ARRAY_CONTAINS(sources, 'llm') THEN 'llm_only'
                    ELSE 'other'
                END AS source_category,
                COUNT(*) AS entity_count,
                ROUND(AVG(ensemble_confidence), 3) AS avg_confidence
            FROM {CATALOG}.extracted.merged_entities
            GROUP BY 1
            """
        )

        # Total entities per source
        totals_per_source = await db.fetch_delta(
            f"""
            SELECT 'dictionary' AS source, COUNT(*) AS total
            FROM {CATALOG}.extracted.merged_entities
            WHERE ARRAY_CONTAINS(sources, 'dictionary')
            UNION ALL
            SELECT 'llm' AS source, COUNT(*) AS total
            FROM {CATALOG}.extracted.merged_entities
            WHERE ARRAY_CONTAINS(sources, 'llm')
            UNION ALL
            SELECT 'ner' AS source, COUNT(*) AS total
            FROM {CATALOG}.extracted.merged_entities
            WHERE ARRAY_CONTAINS(sources, 'ner')
            """
        )

        # Unique contribution rates (found ONLY by one layer)
        unique_contributions = await db.fetch_delta(
            f"""
            SELECT
                CASE
                    WHEN SIZE(sources) = 1 AND ARRAY_CONTAINS(sources, 'dictionary') THEN 'dictionary'
                    WHEN SIZE(sources) = 1 AND ARRAY_CONTAINS(sources, 'llm') THEN 'llm'
                    WHEN SIZE(sources) = 1 AND ARRAY_CONTAINS(sources, 'ner') THEN 'ner'
                    ELSE 'multi_source'
                END AS unique_source,
                COUNT(*) AS count
            FROM {CATALOG}.extracted.merged_entities
            GROUP BY 1
            """
        )

        # Entity type breakdown by source
        type_by_source = await db.fetch_delta(
            f"""
            SELECT
                entity_type,
                SUM(CASE WHEN ARRAY_CONTAINS(sources, 'dictionary') THEN 1 ELSE 0 END) AS dictionary_count,
                SUM(CASE WHEN ARRAY_CONTAINS(sources, 'llm') THEN 1 ELSE 0 END) AS llm_count,
                SUM(CASE WHEN ARRAY_CONTAINS(sources, 'ner') THEN 1 ELSE 0 END) AS ner_count,
                COUNT(*) AS total
            FROM {CATALOG}.extracted.merged_entities
            GROUP BY entity_type
            ORDER BY total DESC
            """
        )

        return _serialize({
            "source_categories": source_categories,
            "totals_per_source": totals_per_source,
            "unique_contributions": unique_contributions,
            "type_by_source": type_by_source,
        })
    except Exception as exc:
        logger.warning("v2 extraction-layers query failed (tables may not exist): %s", exc)
        return {
            "source_categories": [],
            "totals_per_source": [],
            "unique_contributions": [],
            "type_by_source": [],
        }


# ---------------------------------------------------------------------------
# GET /api/v2/assertions
# ---------------------------------------------------------------------------

@router.get("/assertions")
async def assertions():
    """Assertion classification stats from entity_assertions."""
    try:
        status_dist = await db.fetch_delta(
            f"""
            SELECT
                assertion_status,
                COUNT(*) AS count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
            FROM {CATALOG}.extracted.entity_assertions
            GROUP BY assertion_status
            """
        )

        # Negation rate by section (join through merged_entities → document_sections)
        negation_by_section = await db.fetch_delta(
            f"""
            SELECT
                COALESCE(ds.section_type, 'UNKNOWN') AS section_type,
                COUNT(*) AS total,
                SUM(CASE WHEN ea.negation_detected = true THEN 1 ELSE 0 END) AS negated,
                ROUND(SUM(CASE WHEN ea.negation_detected = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS negation_pct
            FROM {CATALOG}.extracted.entity_assertions ea
            JOIN {CATALOG}.extracted.merged_entities me ON ea.entity_id = me.entity_id
            LEFT JOIN {CATALOG}.extracted.document_sections ds ON me.section_id = ds.section_id
            GROUP BY COALESCE(ds.section_type, 'UNKNOWN')
            ORDER BY total DESC
            """
        )

        # Experiencer distribution
        experiencer_dist = await db.fetch_delta(
            f"""
            SELECT
                COALESCE(experiencer, 'UNKNOWN') AS experiencer,
                COUNT(*) AS count
            FROM {CATALOG}.extracted.entity_assertions
            GROUP BY COALESCE(experiencer, 'UNKNOWN')
            ORDER BY count DESC
            """
        )

        # Temporality distribution
        temporality_dist = await db.fetch_delta(
            f"""
            SELECT
                COALESCE(temporality, 'UNKNOWN') AS temporality,
                COUNT(*) AS count
            FROM {CATALOG}.extracted.entity_assertions
            GROUP BY COALESCE(temporality, 'UNKNOWN')
            ORDER BY count DESC
            """
        )

        # Pre-compute KPIs for the frontend
        total_classified = sum(r.get("count", 0) for r in status_dist)
        def _find_pct(status):
            item = next((r for r in status_dist if r.get("assertion_status") == status), None)
            return round(item["pct"], 1) if item and item.get("pct") else 0.0

        return _serialize({
            "status_distribution": status_dist,
            "negation_by_section": negation_by_section,
            "experiencer_distribution": experiencer_dist,
            "temporality_distribution": temporality_dist,
            # Pre-computed KPIs
            "total_classified": total_classified,
            "negated_pct": _find_pct("ABSENT"),
            "historical_pct": _find_pct("HISTORICAL"),
            "family_pct": _find_pct("FAMILY"),
        })
    except Exception as exc:
        logger.warning("v2 assertions query failed (tables may not exist): %s", exc)
        return {
            "status_distribution": [],
            "negation_by_section": [],
            "experiencer_distribution": [],
            "temporality_distribution": [],
        }


# ---------------------------------------------------------------------------
# GET /api/v2/sections
# ---------------------------------------------------------------------------

@router.get("/sections")
async def sections():
    """Section distribution from document_sections."""
    try:
        section_dist = await db.fetch_delta(
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
        return _serialize({"sections": section_dist})
    except Exception as exc:
        logger.warning("v2 sections query failed (tables may not exist): %s", exc)
        return {"sections": []}


# ---------------------------------------------------------------------------
# GET /api/v2/recall-metrics
# ---------------------------------------------------------------------------

@router.get("/recall-metrics")
async def recall_metrics():
    """Recall/precision trending from feedback.recall_metrics."""
    try:
        rows = await db.fetch_delta(
            f"""
            SELECT
                DATE(run_timestamp) AS run_date,
                metric_scope,
                scope_value,
                recall_score,
                precision_score,
                f1_score
            FROM {CATALOG}.feedback.recall_metrics
            ORDER BY run_timestamp DESC
            LIMIT 100
            """
        )
        return _serialize({"metrics": rows})
    except Exception as exc:
        logger.warning("v2 recall-metrics query failed (tables may not exist): %s", exc)
        return {"metrics": []}


# ---------------------------------------------------------------------------
# GET /api/v2/error-patterns
# ---------------------------------------------------------------------------

@router.get("/error-patterns")
async def error_patterns():
    """Active (unresolved) error patterns from feedback.error_patterns."""
    try:
        rows = await db.fetch_delta(
            f"""
            SELECT *
            FROM {CATALOG}.feedback.error_patterns
            WHERE resolved_at IS NULL
            ORDER BY CASE severity WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END
            """
        )
        return _serialize({"patterns": rows})
    except Exception as exc:
        logger.warning("v2 error-patterns query failed (tables may not exist): %s", exc)
        return {"patterns": []}


# ---------------------------------------------------------------------------
# GET /api/v2/coding-tiers
# ---------------------------------------------------------------------------

@router.get("/coding-tiers")
async def coding_tiers():
    """Tier 1 (ontology) vs Tier 2 (LLM) breakdown from icd10_mappings."""
    try:
        rows = await db.fetch_delta(
            f"""
            SELECT
                resolution_path,
                COUNT(*) AS count,
                ROUND(AVG(confidence), 3) AS avg_confidence,
                COUNT(CASE WHEN is_specific = true THEN 1 END) AS specific_codes
            FROM {CATALOG}.codified.icd10_mappings
            GROUP BY resolution_path
            """
        )
        # Pre-compute KPIs
        total_coded = sum(r.get("count", 0) for r in rows)
        ontology_count = sum(r.get("count", 0) for r in rows if r.get("resolution_path") == "ONTOLOGY_DIRECT")
        llm_count = sum(r.get("count", 0) for r in rows if r.get("resolution_path") == "LLM_ASSIGNED")
        weighted_conf = sum(r.get("avg_confidence", 0) * r.get("count", 0) for r in rows)

        return _serialize({
            "tiers": rows,
            "ontology_coverage_pct": round(ontology_count * 100 / max(total_coded, 1), 1),
            "llm_coding_pct": round(llm_count * 100 / max(total_coded, 1), 1),
            "avg_confidence_by_tier": round(weighted_conf / max(total_coded, 1) * 100, 1),
        })
    except Exception as exc:
        logger.warning("v2 coding-tiers query failed (tables may not exist): %s", exc)
        return {"tiers": [], "ontology_coverage_pct": 0, "llm_coding_pct": 0, "avg_confidence_by_tier": 0}
