"""
Configuration for the Medical Codification Review App.
Handles dual-mode (Lakebase / Warehouse) settings, Databricks workspace client,
and Lakebase connection parameters.
"""

import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data-mode configuration
# ---------------------------------------------------------------------------

DATA_MODE: str = os.environ.get("DATA_MODE", "lakebase").lower()  # "lakebase" or "warehouse"
CATALOG: str = os.environ.get("CATALOG", "mv_catalog")
WAREHOUSE_ID: str = os.environ.get("WAREHOUSE_ID", "")

# ---------------------------------------------------------------------------
# Databricks Workspace Client
# ---------------------------------------------------------------------------

IS_DATABRICKS_APP = bool(os.environ.get("DATABRICKS_APP_NAME"))

_workspace_client = None


def get_workspace_client():
    """Return a cached WorkspaceClient (lazy-initialised)."""
    global _workspace_client
    if _workspace_client is None:
        from databricks.sdk import WorkspaceClient

        if IS_DATABRICKS_APP:
            logger.info("Running inside Databricks App - using default WorkspaceClient")
            _workspace_client = WorkspaceClient()
        else:
            profile = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")
            logger.info("Running locally - using WorkspaceClient(profile=%s)", profile)
            _workspace_client = WorkspaceClient(profile=profile)
    return _workspace_client


def get_workspace_host() -> str:
    """Return the Databricks workspace host URL (no trailing slash)."""
    w = get_workspace_client()
    host = w.config.host
    return host.rstrip("/") if host else ""


# ---------------------------------------------------------------------------
# OAuth token helper (used by Lakebase connection)
# ---------------------------------------------------------------------------

def get_oauth_token() -> str:
    """Obtain a short-lived OAuth token from the workspace client."""
    w = get_workspace_client()
    header_factory = w.config.authenticate
    headers = header_factory()
    auth_value = headers.get("Authorization", "")
    if auth_value.startswith("Bearer "):
        return auth_value[len("Bearer "):]
    return auth_value


# ---------------------------------------------------------------------------
# Lakebase (Postgres) connection parameters
# ---------------------------------------------------------------------------

DB_HOST = os.environ.get("DB_HOST") or os.environ.get("PGHOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT") or os.environ.get("PGPORT", "5432"))
DB_NAME = os.environ.get("DB_NAME") or os.environ.get("PGDATABASE", "medical_codification")
DB_USER = os.environ.get("DB_USER") or os.environ.get("PGUSER", "databricks")
