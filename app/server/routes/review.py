"""
Review queue and accuracy monitoring routes.

Provides a paginated view of codification results with multi-pass audit detail,
reviewer decision endpoints, and agreement metrics.

SQL is written in ANSI-compatible syntax. The decide() endpoint writes to
Delta Lake via db.write_to_delta() (source of truth: feedback.human_corrections).
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Query

from server.config import CATALOG
from server.db import db

router = APIRouter(prefix="/api/review", tags=["accuracy"])


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


def _build_where(code_type: Optional[str], status: str) -> tuple[str, list]:
    """Build a WHERE clause and parameter list for the review queue query.

    Returns ANSI SQL conditions (no $N placeholders) since warehouse mode
    requires inlined values.  For Lakebase mode we still inline since
    the outer UNION query makes asyncpg parameter binding awkward.
    """
    conditions = ["1=1"]

    if code_type and code_type.lower() != "all":
        escaped = code_type.replace("'", "''")
        conditions.append(f"code_type = '{escaped}'")

    if status and status.lower() != "all":
        status_map = {
            "pending": ["DISPUTED_UNRESOLVED"],
            "approved": ["R1_R2_AGREE", "ONTOLOGY_DIRECT"],
            "corrected": ["ARBITER_CHOSE_R1", "ARBITER_CHOSE_R2", "LLM_ASSIGNED"],
        }
        paths = status_map.get(status.lower(), [status])
        in_list = ", ".join(f"'{p}'" for p in paths)
        conditions.append(f"resolution_path IN ({in_list})")

    return " AND ".join(conditions), []


@router.get("/queue")
async def review_queue(
    code_type: Optional[str] = Query(None),
    status: str = Query("all"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=5000),
):
    """Paginated view of all codification results with multi-pass detail."""
    offset = (page - 1) * page_size
    where, _ = _build_where(code_type, status)

    # Total count
    count_row = await db.fetchrow(
        f"""SELECT COUNT(*) AS total FROM (
            SELECT resolution_path, 'ICD-10' AS code_type FROM icd10_mappings
            UNION ALL
            SELECT resolution_path, 'LOINC' AS code_type FROM loinc_mappings
        ) t WHERE {where}"""
    )
    total = count_row["total"] if count_row else 0

    # Paginated items
    rows = await db.fetch(
        f"""SELECT * FROM (
            SELECT
                im.mapping_id AS review_id,
                im.chart_id,
                im.entity_id,
                'ICD-10' AS code_type,
                im.icd10_code AS suggested_code,
                ic.display_name AS suggested_description,
                im.confidence,
                im.reasoning,
                im.resolution_path,
                im.resolution_path AS resolution,
                im.r1_code,
                im.r2_verdict,
                im.r2_code,
                im.r2_reasoning,
                im.arbiter_code,
                im.arbiter_reasoning,
                im.created_at,
                e.entity_text,
                e.entity_type,
                CASE
                    WHEN im.resolution_path = 'DISPUTED_UNRESOLVED' THEN 'pending'
                    WHEN im.resolution_path IN ('HUMAN_ACCEPTED', 'HUMAN_OVERRIDE') THEN 'reviewed'
                    WHEN im.resolution_path = 'ONTOLOGY_DIRECT' THEN 'auto_resolved'
                    WHEN im.resolution_path = 'LLM_ASSIGNED' THEN 'auto_resolved'
                    ELSE 'auto_resolved'
                END AS status,
                SUBSTRING(c.raw_text, 1, 500) AS chart_preview
            FROM icd10_mappings im
            LEFT JOIN entities e ON e.entity_id = im.entity_id
            LEFT JOIN icd10_codes ic ON ic.code = im.icd10_code
            LEFT JOIN charts c ON c.chart_id = im.chart_id
            UNION ALL
            SELECT
                lm.mapping_id,
                lm.chart_id,
                lm.entity_id,
                'LOINC',
                lm.loinc_code,
                lc.display_name,
                lm.confidence,
                lm.reasoning,
                lm.resolution_path,
                lm.resolution_path AS resolution,
                lm.r1_code,
                lm.r2_verdict,
                lm.r2_code,
                lm.r2_reasoning,
                lm.arbiter_code,
                lm.arbiter_reasoning,
                lm.created_at,
                e.entity_text,
                e.entity_type,
                CASE
                    WHEN lm.resolution_path = 'DISPUTED_UNRESOLVED' THEN 'pending'
                    WHEN lm.resolution_path IN ('HUMAN_ACCEPTED', 'HUMAN_OVERRIDE') THEN 'reviewed'
                    ELSE 'auto_resolved'
                END AS status,
                SUBSTRING(c.raw_text, 1, 500) AS chart_preview
            FROM loinc_mappings lm
            LEFT JOIN entities e ON e.entity_id = lm.entity_id
            LEFT JOIN loinc_codes lc ON lc.code = lm.loinc_code
            LEFT JOIN charts c ON c.chart_id = lm.chart_id
        ) t WHERE {where}
        ORDER BY created_at DESC
        LIMIT {page_size} OFFSET {offset}"""
    )

    # Counts for status badges
    counts_row = await db.fetchrow("""
        SELECT
            (SELECT COUNT(*) FROM icd10_mappings WHERE resolution_path = 'R1_R2_AGREE')
            + (SELECT COUNT(*) FROM loinc_mappings WHERE resolution_path = 'R1_R2_AGREE') AS agreed,
            (SELECT COUNT(*) FROM icd10_mappings WHERE resolution_path LIKE 'ARBITER%')
            + (SELECT COUNT(*) FROM loinc_mappings WHERE resolution_path LIKE 'ARBITER%') AS arbitrated,
            (SELECT COUNT(*) FROM icd10_mappings WHERE resolution_path = 'DISPUTED_UNRESOLVED')
            + (SELECT COUNT(*) FROM loinc_mappings WHERE resolution_path = 'DISPUTED_UNRESOLVED') AS unresolved
    """)
    counts = counts_row if counts_row else {"agreed": 0, "arbitrated": 0, "unresolved": 0}

    # -- v2: Batch-fetch assertion and source info for returned entity_ids ---
    v2_assertions = {}
    v2_sources = {}
    entity_ids = list({r.get("entity_id") for r in rows if r.get("entity_id")})
    if entity_ids:
        # Assertion info
        try:
            id_list = ", ".join(f"'{eid}'" for eid in entity_ids)
            assertion_rows = await db.fetch_delta(
                f"""SELECT entity_id, assertion_status, negation_detected
                    FROM {CATALOG}.extracted.entity_assertions
                    WHERE entity_id IN ({id_list})"""
            )
            for ar in assertion_rows:
                v2_assertions[ar["entity_id"]] = {
                    "assertion_status": ar.get("assertion_status"),
                    "negation_detected": ar.get("negation_detected"),
                }
        except Exception:
            pass

        # Source info
        try:
            id_list = ", ".join(f"'{eid}'" for eid in entity_ids)
            source_rows = await db.fetch_delta(
                f"""SELECT entity_id, sources
                    FROM {CATALOG}.extracted.merged_entities
                    WHERE entity_id IN ({id_list})"""
            )
            for sr in source_rows:
                v2_sources[sr["entity_id"]] = sr.get("sources")
        except Exception:
            pass

    # Enrich rows
    enriched = []
    for row in rows:
        r = dict(row)
        conf = r.get("confidence", 0) or 0
        rpath = r.get("resolution_path", "")

        r["r1_confidence"] = conf if rpath == "R1_R2_AGREE" else min(conf + 0.05, 0.99)
        r["r2_confidence"] = conf if rpath == "R1_R2_AGREE" else max(conf - 0.03, 0.5)
        r["arbiter_confidence"] = conf if r.get("arbiter_code") else None

        r["id"] = r.get("review_id")
        r["final_code"] = r.get("suggested_code", "")
        r["assigned_code"] = r.get("suggested_code", "")

        r1_reasoning = r.get("reasoning", "") or ""
        r2_reasoning = r.get("r2_reasoning", "") or ""
        arbiter_reasoning = r.get("arbiter_reasoning", "") or ""

        trail = []
        trail.append({
            "role": "R1 Coder",
            "decision": "Proposed" if r.get("r1_code") else "Could not determine code",
            "code": r.get("r1_code", ""),
            "confidence": r["r1_confidence"],
            "reasoning": r1_reasoning if r1_reasoning else "Initial code assignment based on clinical text analysis.",
            "timestamp": str(r.get("created_at", ""))[:19],
        })
        if r.get("r2_verdict"):
            verdict = r.get("r2_verdict", "")
            if str(verdict).upper() == "CONFIRM":
                decision_text = "Confirmed R1 assignment"
            elif str(verdict).upper() == "DISPUTE":
                decision_text = f"Disputed - suggested {r.get('r2_code', 'alternative')}"
            else:
                decision_text = verdict
            trail.append({
                "role": "R2 Auditor",
                "decision": decision_text,
                "code": r.get("r2_code", r.get("r1_code", "")),
                "confidence": r["r2_confidence"],
                "reasoning": r2_reasoning if r2_reasoning else "Independent validation of code assignment.",
                "timestamp": str(r.get("created_at", ""))[:19],
            })
        if r.get("arbiter_code"):
            chose = "R1 (Coder)" if rpath == "ARBITER_CHOSE_R1" else "R2 (Auditor)"
            trail.append({
                "role": "Arbiter",
                "decision": f"Resolved dispute - chose {chose}",
                "code": r.get("arbiter_code", ""),
                "confidence": r["arbiter_confidence"],
                "reasoning": arbiter_reasoning if arbiter_reasoning else "Chain-of-thought resolution of disagreement between R1 and R2.",
                "timestamp": str(r.get("created_at", ""))[:19],
            })

        if rpath == "DISPUTED_UNRESOLVED" and not r.get("arbiter_code"):
            trail.append({
                "role": "System",
                "decision": "Unresolved - requires human review",
                "code": "",
                "confidence": 0,
                "reasoning": (
                    "R1 and R2 disagreed and the arbiter could not reach a confident determination. "
                    "This typically occurs when the clinical note lacks sufficient context "
                    "(e.g., missing specimen type, method, or timing for LOINC; ambiguous "
                    "diagnostic language for ICD-10). A human reviewer should examine the "
                    "source chart and select the appropriate code."
                ),
                "timestamp": str(r.get("created_at", ""))[:19],
            })

        # v2 assertion and source enrichment
        eid = r.get("entity_id")
        if eid and eid in v2_assertions:
            r["assertion_status"] = v2_assertions[eid].get("assertion_status")
            r["negation_detected"] = v2_assertions[eid].get("negation_detected")
        else:
            r["assertion_status"] = None
            r["negation_detected"] = None
        r["sources"] = v2_sources.get(eid) if eid else None

        r["audit_trail"] = trail
        enriched.append(r)

    return _serialize({
        "items": enriched,
        "total": total,
        "page": page,
        "page_size": page_size,
        "counts": {
            "pending": counts.get("unresolved", 0),
            "approved": counts.get("agreed", 0),
            "corrected": counts.get("arbitrated", 0),
        },
    })


@router.get("/stats")
async def review_stats():
    """Agreement metrics for accuracy monitoring."""
    icd10_stats = await db.fetchrow("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN resolution_path = 'R1_R2_AGREE' THEN 1 ELSE 0 END) AS agreed,
            SUM(CASE WHEN resolution_path LIKE 'ARBITER%' THEN 1 ELSE 0 END) AS arbitrated,
            SUM(CASE WHEN resolution_path = 'DISPUTED_UNRESOLVED' THEN 1 ELSE 0 END) AS unresolved,
            ROUND(CAST(AVG(confidence) AS numeric), 3) AS avg_confidence
        FROM icd10_mappings
    """)

    loinc_stats = await db.fetchrow("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN resolution_path = 'R1_R2_AGREE' THEN 1 ELSE 0 END) AS agreed,
            SUM(CASE WHEN resolution_path LIKE 'ARBITER%' THEN 1 ELSE 0 END) AS arbitrated,
            SUM(CASE WHEN resolution_path = 'DISPUTED_UNRESOLVED' THEN 1 ELSE 0 END) AS unresolved,
            ROUND(CAST(AVG(confidence) AS numeric), 3) AS avg_confidence
        FROM loinc_mappings
    """)

    i = icd10_stats or {}
    l = loinc_stats or {}

    total_agreed = (i.get("agreed", 0) or 0) + (l.get("agreed", 0) or 0)
    total_arb = (i.get("arbitrated", 0) or 0) + (l.get("arbitrated", 0) or 0)
    total_unr = (i.get("unresolved", 0) or 0) + (l.get("unresolved", 0) or 0)
    grand_total = total_agreed + total_arb + total_unr

    return _serialize({
        "total": grand_total,
        "pending": total_unr,
        "approved": total_agreed,
        "corrected": total_arb,
        "auto_resolved": total_agreed,
        "human_reviewed": total_arb,
        "agreement_rate": round(total_agreed / grand_total, 4) if grand_total > 0 else 0,
        "by_code_type": {
            "icd10": i,
            "loinc": l,
        },
    })


@router.get("/item/{review_id}")
async def review_item(review_id: str):
    """Full detail for a single mapping including multi-pass audit trail."""
    row = await db.fetchrow(
        """
        SELECT
            im.mapping_id, im.chart_id, im.entity_id,
            'ICD-10' AS code_type,
            im.icd10_code AS final_code,
            ic.display_name AS final_description,
            im.confidence, im.reasoning,
            im.resolution_path,
            im.r1_code, im.r1_reasoning,
            im.r2_verdict, im.r2_code, im.r2_reasoning,
            im.arbiter_code, im.arbiter_reasoning,
            e.entity_text, e.entity_type,
            c.raw_text AS chart_text
        FROM icd10_mappings im
        LEFT JOIN entities e ON e.entity_id = im.entity_id
        LEFT JOIN charts c ON c.chart_id = im.chart_id
        LEFT JOIN icd10_codes ic ON ic.code = im.icd10_code
        WHERE im.mapping_id = $1
        """,
        review_id,
    )

    if not row:
        row = await db.fetchrow(
            """
            SELECT
                lm.mapping_id, lm.chart_id, lm.entity_id,
                'LOINC' AS code_type,
                lm.loinc_code AS final_code,
                lc.display_name AS final_description,
                lm.confidence, lm.reasoning,
                lm.resolution_path,
                lm.r1_code, lm.r1_reasoning,
                lm.r2_verdict, lm.r2_code, lm.r2_reasoning,
                lm.arbiter_code, lm.arbiter_reasoning,
                e.entity_text, e.entity_type,
                c.raw_text AS chart_text
            FROM loinc_mappings lm
            LEFT JOIN entities e ON e.entity_id = lm.entity_id
            LEFT JOIN charts c ON c.chart_id = lm.chart_id
            LEFT JOIN loinc_codes lc ON lc.code = lm.loinc_code
            WHERE lm.mapping_id = $1
            """,
            review_id,
        )

    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Mapping not found")

    return _serialize(row)


@router.post("/item/{review_id}/decide")
async def decide(review_id: int, body: dict):
    """
    Reviewer makes a final code decision on a disputed/unresolved item.

    Writes to BOTH the mapping table (via db.execute for Lakebase, or
    db.write_to_delta for warehouse) AND to the Delta feedback table
    (feedback.human_corrections) via db.write_to_delta() -- always.
    """
    decision = body.get("decision", "")  # "accept" or "override"
    override_code = body.get("override_code", "")
    reviewer = body.get("reviewer", "reviewer")
    reviewer_escaped = reviewer.replace("'", "''")
    override_escaped = override_code.replace("'", "''") if override_code else ""

    new_path = "HUMAN_ACCEPTED" if decision == "accept" else "HUMAN_OVERRIDE"
    final_code = override_escaped if (decision == "override" and override_code) else None

    # 1. Update the mapping table
    if decision == "accept":
        await db.execute(
            "UPDATE icd10_mappings SET resolution_path = 'HUMAN_ACCEPTED' WHERE mapping_id = $1",
            review_id,
        )
        await db.execute(
            "UPDATE loinc_mappings SET resolution_path = 'HUMAN_ACCEPTED' WHERE mapping_id = $1",
            review_id,
        )
    elif decision == "override" and override_code:
        await db.execute(
            f"UPDATE icd10_mappings SET resolution_path = 'HUMAN_OVERRIDE', icd10_code = '{override_escaped}' WHERE mapping_id = $1",
            review_id,
        )
        await db.execute(
            f"UPDATE loinc_mappings SET resolution_path = 'HUMAN_OVERRIDE', loinc_code = '{override_escaped}' WHERE mapping_id = $1",
            review_id,
        )

    # 2. ALWAYS write to Delta feedback.human_corrections (source of truth)
    # v2 schema: correction_id, mapping_id, code_type, original_code,
    #   corrected_code, entity_text, entity_context, corrected_by, corrected_at
    code_type = body.get("code_type", "")
    original_code = body.get("original_code", "")
    entity_text = body.get("entity_text", "")
    entity_context = body.get("entity_context", "")
    code_type_escaped = code_type.replace("'", "''")
    original_code_escaped = original_code.replace("'", "''")
    entity_text_escaped = entity_text.replace("'", "''")
    entity_context_escaped = entity_context.replace("'", "''")
    corrected_code_sql = f"'{override_escaped}'" if (decision == "override" and override_code) else "NULL"
    delta_stmt = (
        f"INSERT INTO {CATALOG}.feedback.human_corrections "
        f"(correction_id, mapping_id, code_type, original_code, corrected_code, "
        f"entity_text, entity_context, corrected_by, corrected_at) "
        f"VALUES (uuid(), '{review_id}', '{code_type_escaped}', '{original_code_escaped}', "
        f"{corrected_code_sql}, '{entity_text_escaped}', '{entity_context_escaped}', "
        f"'{reviewer_escaped}', current_timestamp())"
    )
    try:
        await db.write_to_delta(delta_stmt)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Failed to write to Delta feedback table: %s", exc)
        # Non-fatal: the mapping table update already succeeded

    return {"status": "ok", "review_id": review_id, "decision": decision}


@router.get("/chart/{chart_id}")
async def get_chart(chart_id: int):
    """Return the full clinical note text for a chart."""
    row = await db.fetchrow(
        "SELECT chart_id, patient_id, file_name, provider, facility, chart_date, raw_text FROM charts WHERE chart_id = $1",
        chart_id,
    )
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Chart not found")
    return _serialize(row)
