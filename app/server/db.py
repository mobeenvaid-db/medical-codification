"""
Dual-mode data layer for the Medical Codification Review App.

Exposes the same interface regardless of DATA_MODE:
  - db.fetch(query, *args)       -> list[dict]
  - db.fetchrow(query, *args)    -> Optional[dict]
  - db.execute(query, *args)     -> str
  - db.write_to_delta(statement) -> writes via SQL Statement API

Lakebase mode  : asyncpg pool for reads, SQL Statement API for Delta writes
Warehouse mode : all reads/writes go through Databricks SQL Statement API
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import ssl
from typing import Any, Dict, List, Optional

import httpx

from server.config import (
    DATA_MODE,
    CATALOG,
    DB_HOST,
    DB_NAME,
    DB_PORT,
    DB_USER,
    WAREHOUSE_ID,
    get_oauth_token,
    get_workspace_host,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helper: parse SQL Statement API response into list[dict]
# ============================================================================

def _parse_statement_response(resp_json: dict) -> List[Dict[str, Any]]:
    """Convert a SQL Statement API response into a list of row-dicts."""
    status = resp_json.get("status", {}).get("state", "")
    if status == "FAILED":
        error = resp_json.get("status", {}).get("error", {})
        raise RuntimeError(f"SQL Statement failed: {error.get('message', resp_json)}")

    result = resp_json.get("result", {}) or {}
    # Columns can be in result.columns OR manifest.schema.columns depending on API version
    columns = result.get("columns", [])
    if not columns:
        columns = resp_json.get("manifest", {}).get("schema", {}).get("columns", [])
    data_array = result.get("data_array", [])

    if not columns or not data_array:
        return []

    col_names = [c["name"] for c in columns]
    col_types = [c.get("type_name", "STRING") for c in columns]

    rows: List[Dict[str, Any]] = []
    for raw_row in data_array:
        row: Dict[str, Any] = {}
        for i, val in enumerate(raw_row):
            coerced = _coerce(val, col_types[i] if i < len(col_types) else "STRING")
            row[col_names[i]] = coerced
        rows.append(row)
    return rows


def _coerce(val: Any, type_name: str) -> Any:
    """Best-effort type coercion from string values returned by the API."""
    if val is None:
        return None
    type_upper = type_name.upper()
    try:
        if type_upper in ("INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "LONG"):
            return int(val)
        if type_upper in ("FLOAT", "DOUBLE", "DECIMAL", "NUMERIC"):
            return float(val)
        if type_upper == "BOOLEAN":
            return str(val).lower() in ("true", "1", "t")
    except (ValueError, TypeError):
        pass
    return val


# ============================================================================
# Postgres (asyncpg) parameter conversion
# ============================================================================

_PG_PARAM_RE = re.compile(r"\$(\d+)")


def _pg_to_positional(query: str, args: tuple) -> tuple[str, list]:
    """Convert ``$1``, ``$2`` style placeholders to ``%s`` for httpx/warehouse
    if needed, or return as-is for asyncpg.  This is only called in warehouse
    mode so we translate to plain ``?`` markers consumed by the SQL Statement API
    (which actually uses parameter binding via the ``parameters`` field).

    However, the Databricks SQL Statement API does NOT support parameter binding
    in the ``parameters`` field for arbitrary queries, so we inline the values
    safely instead.
    """
    if not args:
        return query, []

    def _replace(match: re.Match) -> str:
        idx = int(match.group(1)) - 1
        if idx < len(args):
            v = args[idx]
            if v is None:
                return "NULL"
            if isinstance(v, str):
                escaped = v.replace("'", "''")
                return f"'{escaped}'"
            return str(v)
        return match.group(0)

    inlined = _PG_PARAM_RE.sub(_replace, query)
    return inlined, []


# ============================================================================
# Lakebase pool (only initialised when DATA_MODE == "lakebase")
# ============================================================================

class LakebasePool:
    """Async connection pool for Lakebase with token refresh."""

    def __init__(self) -> None:
        self._pool = None  # asyncpg.Pool | None
        self._refresh_task: Optional[asyncio.Task] = None
        self._demo_mode: bool = False

    async def get_pool(self):
        if self._pool is not None:
            return self._pool

        import asyncpg  # deferred import — not needed in warehouse mode

        password = os.environ.get("LAKEBASE_PASSWORD")
        if not password:
            try:
                password = get_oauth_token()
            except Exception as exc:
                logger.warning("Could not obtain OAuth token: %s", exc)
                password = None

        if not password:
            logger.warning("No Lakebase password available - entering demo mode")
            self._demo_mode = True
            return None

        try:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

            self._pool = await asyncpg.create_pool(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=password,
                ssl=ssl_ctx,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
            logger.info("Lakebase pool created (%s@%s:%s/%s)", DB_USER, DB_HOST, DB_PORT, DB_NAME)
            self._demo_mode = False
        except Exception as exc:
            logger.error("Failed to connect to Lakebase: %s - entering demo mode", exc)
            self._demo_mode = True
            self._pool = None

        return self._pool

    async def close(self) -> None:
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        if self._pool:
            await self._pool.close()
            self._pool = None
        logger.info("Lakebase pool closed")

    def start_refresh_loop(self) -> None:
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def _refresh_loop(self) -> None:
        import asyncpg

        while True:
            await asyncio.sleep(45 * 60)
            try:
                new_password = os.environ.get("LAKEBASE_PASSWORD") or get_oauth_token()
                if self._pool:
                    await self._pool.close()
                    self._pool = None
                    ssl_ctx = ssl.create_default_context()
                    ssl_ctx.check_hostname = False
                    ssl_ctx.verify_mode = ssl.CERT_NONE
                    self._pool = await asyncpg.create_pool(
                        host=DB_HOST,
                        port=DB_PORT,
                        database=DB_NAME,
                        user=DB_USER,
                        password=new_password,
                        ssl=ssl_ctx,
                        min_size=2,
                        max_size=10,
                        command_timeout=30,
                    )
                    logger.info("Lakebase pool refreshed with new token")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Token refresh failed: %s", exc)

    async def fetch(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        pool = await self.get_pool()
        if pool is None:
            return []
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(r) for r in rows]

    async def fetchrow(self, query: str, *args: Any) -> Optional[Dict[str, Any]]:
        pool = await self.get_pool()
        if pool is None:
            return None
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def execute(self, query: str, *args: Any) -> str:
        pool = await self.get_pool()
        if pool is None:
            return ""
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)


# ============================================================================
# Warehouse client (SQL Statement API)
# ============================================================================

class WarehouseClient:
    """Execute SQL via the Databricks SQL Statement API."""

    def __init__(self, warehouse_id: str) -> None:
        self.warehouse_id = warehouse_id
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            token = get_oauth_token()
            host = get_workspace_host()
            self._client = httpx.AsyncClient(
                base_url=host,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._client

    async def _refresh_token(self) -> None:
        """Refresh the auth token on the existing client."""
        token = get_oauth_token()
        if self._client and not self._client.is_closed:
            self._client.headers["Authorization"] = f"Bearer {token}"

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _execute_statement(self, statement: str) -> dict:
        """POST to /api/2.0/sql/statements and return parsed JSON."""
        client = await self._get_client()
        body = {
            "statement": statement,
            "warehouse_id": self.warehouse_id,
            "wait_timeout": "50s",
        }
        resp = await client.post("/api/2.0/sql/statements", json=body)

        # Handle token expiry — refresh and retry once
        if resp.status_code == 401:
            await self._refresh_token()
            client = await self._get_client()
            resp = await client.post("/api/2.0/sql/statements", json=body)

        resp.raise_for_status()
        return resp.json()

    async def fetch(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        statement, _ = _pg_to_positional(query, args)
        resp_json = await self._execute_statement(statement)
        return _parse_statement_response(resp_json)

    async def fetchrow(self, query: str, *args: Any) -> Optional[Dict[str, Any]]:
        rows = await self.fetch(query, *args)
        return rows[0] if rows else None

    async def execute(self, query: str, *args: Any) -> str:
        statement, _ = _pg_to_positional(query, args)
        resp_json = await self._execute_statement(statement)
        state = resp_json.get("status", {}).get("state", "UNKNOWN")
        if state == "FAILED":
            error = resp_json.get("status", {}).get("error", {})
            raise RuntimeError(f"SQL Statement failed: {error.get('message', resp_json)}")
        return state


# ============================================================================
# Unified dual-mode wrapper
# ============================================================================

class DualModeDB:
    """
    Unified database interface.

    - Lakebase mode: reads via asyncpg, Delta writes via SQL Statement API.
    - Warehouse mode: all reads/writes via SQL Statement API.
    """

    def __init__(self) -> None:
        self._mode = DATA_MODE
        self._lakebase: Optional[LakebasePool] = None
        self._warehouse: Optional[WarehouseClient] = None
        self._delta_writer: Optional[WarehouseClient] = None  # for Lakebase mode Delta writes
        self._demo_mode: bool = False

    @property
    def mode(self) -> str:
        return self._mode

    # -- lifecycle -----------------------------------------------------------

    async def init(self) -> None:
        """Initialise the appropriate backend(s)."""
        if self._mode == "lakebase":
            self._lakebase = LakebasePool()
            await self._lakebase.get_pool()
            self._lakebase.start_refresh_loop()
            self._demo_mode = self._lakebase._demo_mode
            # Delta writer for feedback writes (uses warehouse if WAREHOUSE_ID set)
            if WAREHOUSE_ID:
                self._delta_writer = WarehouseClient(WAREHOUSE_ID)
                logger.info("Lakebase mode: Delta writer configured (warehouse %s)", WAREHOUSE_ID)
            else:
                logger.info("Lakebase mode: No WAREHOUSE_ID set, write_to_delta will be a no-op")
        elif self._mode == "warehouse":
            if not WAREHOUSE_ID:
                raise ValueError("DATA_MODE=warehouse requires WAREHOUSE_ID to be set")
            self._warehouse = WarehouseClient(WAREHOUSE_ID)
            logger.info("Warehouse mode: all queries via SQL Statement API (warehouse %s)", WAREHOUSE_ID)
        else:
            raise ValueError(f"Unknown DATA_MODE: {self._mode!r} (expected 'lakebase' or 'warehouse')")

    async def close(self) -> None:
        if self._lakebase:
            await self._lakebase.close()
        if self._warehouse:
            await self._warehouse.close()
        if self._delta_writer:
            await self._delta_writer.close()
        logger.info("DualModeDB closed")

    # -- read interface ------------------------------------------------------

    async def fetch(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        if self._mode == "lakebase" and self._lakebase:
            return await self._lakebase.fetch(query, *args)
        elif self._mode == "warehouse" and self._warehouse:
            return await self._warehouse.fetch(query, *args)
        logger.debug("Demo mode - returning empty result for fetch")
        return []

    async def fetchrow(self, query: str, *args: Any) -> Optional[Dict[str, Any]]:
        if self._mode == "lakebase" and self._lakebase:
            return await self._lakebase.fetchrow(query, *args)
        elif self._mode == "warehouse" and self._warehouse:
            return await self._warehouse.fetchrow(query, *args)
        logger.debug("Demo mode - returning None for fetchrow")
        return None

    async def execute(self, query: str, *args: Any) -> str:
        if self._mode == "lakebase" and self._lakebase:
            return await self._lakebase.execute(query, *args)
        elif self._mode == "warehouse" and self._warehouse:
            return await self._warehouse.execute(query, *args)
        logger.debug("Demo mode - returning empty status for execute")
        return ""

    # -- Delta read (always via SQL Statement API) ---------------------------

    async def fetch_delta(self, query: str) -> List[Dict[str, Any]]:
        """
        Query Delta Lake tables directly via SQL Statement API.

        Use this for v2 pipeline tables that only exist in Delta (not in Lakebase).
        In Warehouse mode this is the same as fetch().
        In Lakebase mode this uses the delta_writer client for reads.
        """
        if self._mode == "warehouse" and self._warehouse:
            rows = await self._warehouse.fetch(query)
            logger.info("fetch_delta (warehouse): %d rows returned", len(rows))
            return rows
        elif self._mode == "lakebase" and self._delta_writer:
            # Log raw response for debugging
            statement, _ = _pg_to_positional(query, ())
            raw_resp = await self._delta_writer._execute_statement(statement)
            status = raw_resp.get("status", {}).get("state", "UNKNOWN")
            result = raw_resp.get("result", {}) or {}
            cols = result.get("columns", [])
            data = result.get("data_array", [])
            manifest = raw_resp.get("manifest", {})
            total_rows = manifest.get("total_row_count", "?")
            logger.info("fetch_delta RAW: status=%s, columns=%d, data_rows=%d, manifest_rows=%s, query=%.80s",
                       status, len(cols), len(data), total_rows, query.strip())
            if status == "FAILED":
                error = raw_resp.get("status", {}).get("error", {})
                logger.error("fetch_delta FAILED: %s", error.get("message", raw_resp))
            rows = _parse_statement_response(raw_resp)
            logger.info("fetch_delta (delta_writer): %d rows parsed", len(rows))
            return rows
        else:
            logger.warning("fetch_delta called but no Delta client available (mode=%s, warehouse_id=%s)",
                          self._mode, WAREHOUSE_ID)
            return []

    # -- Delta write (always via SQL Statement API) --------------------------

    async def write_to_delta(self, statement: str) -> str:
        """
        Write to Delta Lake via SQL Statement API.

        In Lakebase mode this uses the separate delta_writer client.
        In Warehouse mode this is the same as execute().
        Always used for feedback.human_corrections and mapping table updates.
        """
        if self._mode == "warehouse" and self._warehouse:
            return await self._warehouse.execute(statement)
        elif self._mode == "lakebase" and self._delta_writer:
            return await self._delta_writer.execute(statement)
        else:
            logger.warning("write_to_delta called but no writer available (mode=%s)", self._mode)
            return ""


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
db = DualModeDB()
