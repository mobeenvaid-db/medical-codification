"""
Medical Codification Review App - FastAPI entry point.

Dual-mode backend: reads from Lakebase (Postgres) or Delta Lake (SQL Warehouse),
reviewer decisions always write to Delta feedback.human_corrections.

Serves the API and the React SPA from frontend/dist.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from server.config import DATA_MODE
from server.db import db
from server.routes.codification import router as codification_router
from server.routes.pipeline import router as pipeline_router
from server.routes.review import router as review_router
from server.routes.analytics_extra import router as analytics_extra_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan - startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up - DATA_MODE=%s, initialising database layer ...", DATA_MODE)
    await db.init()
    yield
    logger.info("Shutting down - closing database layer ...")
    await db.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical Codification Review",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# No-cache middleware for /api/ routes
# ---------------------------------------------------------------------------

class NoCacheAPIMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response


app.add_middleware(NoCacheAPIMiddleware)


# ---------------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------------

app.include_router(pipeline_router)
app.include_router(review_router)
app.include_router(codification_router)
app.include_router(analytics_extra_router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "data_mode": db.mode,
    }


# ---------------------------------------------------------------------------
# Compatibility routes - map frontend URLs to existing backend handlers
# ---------------------------------------------------------------------------

from server.routes.pipeline import pipeline_overview
from server.routes.codification import icd10_analytics, loinc_analytics


@app.get("/api/stats")
async def stats_compat():
    """Remap pipeline stats to the field names the frontend expects."""
    data = await pipeline_overview()
    rs = data.get("review_status", {})
    total = (rs.get("auto_approved", 0) or 0) + (rs.get("flagged", 0) or 0) + (rs.get("needs_review", 0) or 0)
    agreed = rs.get("auto_approved", 0) or 0
    return {
        "total_charts": data.get("charts_processed", 0),
        "total_entities": data.get("entities_extracted", 0),
        "total_icd10": data.get("icd10_codes_assigned", 0),
        "total_loinc": data.get("loinc_codes_assigned", 0),
        "auto_resolved": agreed,
        "review_queue_size": rs.get("needs_review", 0) or 0,
        "human_reviewed": rs.get("flagged", 0) or 0,
        "agreement_rate": round(agreed / total, 4) if total > 0 else 0,
        "entity_distribution": [{"entity_type": d.get("name", ""), "count": d.get("value", 0)}
                                for d in data.get("entity_distribution", [])],
        "confidence_distribution": data.get("confidence_distribution", []),
    }


@app.get("/api/analytics/entity-types")
async def entity_types_compat():
    data = await pipeline_overview()
    return [{"entity_type": d.get("name", ""), "count": d.get("value", 0)}
            for d in data.get("entity_distribution", [])]


@app.get("/api/analytics/confidence-distribution")
async def confidence_dist_compat():
    data = await pipeline_overview()
    return [{"bucket": d.get("band", ""), "count": (d.get("icd10", 0) or 0) + (d.get("loinc", 0) or 0)}
            for d in data.get("confidence_distribution", [])]


@app.get("/api/analytics/icd10/top-codes")
async def icd10_top_codes_compat():
    data = await icd10_analytics()
    return data.get("top_codes", [])


@app.get("/api/analytics/icd10/specificity")
async def icd10_specificity_compat():
    data = await icd10_analytics()
    spec = data.get("specificity", {"specific": 0, "unspecified": 0})
    return [
        {"level": "Specific", "count": spec.get("specific", 0)},
        {"level": "Unspecified", "count": spec.get("unspecified", 0)},
    ]


@app.get("/api/analytics/icd10/recent")
async def icd10_recent_compat(page: int = 1, per_page: int = 10):
    data = await icd10_analytics()
    items = data.get("recent_mappings", [])
    start = (page - 1) * per_page
    return {"items": items[start:start + per_page], "total": len(items)}


@app.get("/api/analytics/loinc/top-codes")
async def loinc_top_codes_compat():
    data = await loinc_analytics()
    return data.get("top_codes", [])


@app.get("/api/analytics/loinc/disambiguations")
async def loinc_disambiguations_compat():
    data = await loinc_analytics()
    showcase = data.get("disambiguation_showcase", [])
    result = []
    for group in showcase:
        entries = group.get("entries", [])

        def weighted_score(e):
            max_occ = max((x.get("occurrences", 1) for x in entries), default=1)
            occ_norm = e.get("occurrences", 0) / max_occ if max_occ > 0 else 0
            conf = e.get("confidence", 0) or 0
            return (occ_norm * 0.6) + (conf * 0.4)

        selected = max(entries, key=weighted_score) if entries else {}
        candidates = []
        for e in entries:
            method = e.get("method", "Not specified")
            occ = e.get("occurrences", 0)
            conf = e.get("confidence", 0) or 0
            is_selected = e.get("loinc_code") == selected.get("loinc_code")
            candidates.append({
                "code": e.get("loinc_code", ""),
                "description": e.get("loinc_description", ""),
                "score": conf,
                "reasoning": (
                    f"Method: {method} | Specimen: {e.get('specimen', 'N/A')} | "
                    f"Assigned {occ}x across charts | Avg confidence: {conf:.1%}"
                    + (" - SELECTED (highest combined frequency + confidence)" if is_selected else "")
                ),
                "method": method,
                "specimen": e.get("specimen", "Not specified"),
                "occurrences": occ,
            })
        result.append({
            "analyte": group.get("analyte", ""),
            "selected_code": selected.get("loinc_code", ""),
            "candidates": candidates,
        })
    return result


@app.get("/api/analytics/loinc/recent")
async def loinc_recent_compat(page: int = 1, per_page: int = 10):
    data = await loinc_analytics()
    items = data.get("recent_mappings", [])
    start = (page - 1) * per_page
    return {"items": items[start:start + per_page], "total": len(items)}


# ---------------------------------------------------------------------------
# Serve React SPA from frontend/dist
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"

if FRONTEND_DIR.is_dir():
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """Catch-all: serve the React SPA index.html for any non-API route."""
        if full_path.startswith("api/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        file_path = FRONTEND_DIR / full_path
        if full_path and file_path.is_file():
            return FileResponse(str(file_path))
        index = FRONTEND_DIR / "index.html"
        if index.is_file():
            return FileResponse(str(index))
        return JSONResponse({"detail": "Frontend not built"}, status_code=404)
else:
    @app.get("/")
    async def no_frontend():
        return JSONResponse(
            {"detail": "Frontend not built. Run the React build first."},
            status_code=200,
        )
