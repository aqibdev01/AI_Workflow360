"""Workflow360 AI Server — FastAPI entry point.

Hosts three AI modules:
  M6  — Task Decomposition   (FLAN-T5)        POST /api/decompose
  M10 — Smart Assignment      (Sentence-BERT)  POST /api/suggest-assignee
  M11 — Bottleneck Detection  (XGBoost)        POST /api/analyze-sprint

Start with:
    uvicorn main:app --reload --port 8000
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure the ai-server directory is on sys.path so relative imports
# work regardless of where uvicorn is invoked from.
_server_dir = os.path.dirname(os.path.abspath(__file__))
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from routers import decomposition, assigner, optimizer
from utils.auth import PayloadInspectorMiddleware
from models.decomposition.model import load_model as load_decomposition
from models.assigner.model import load_model as load_assigner
from models.optimizer.model import load_model as load_optimizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("ai-server")

# Track which models loaded vs fell back to mock
_model_status = {
    "decomposition": False,
    "assigner": False,
    "optimizer": False,
}


# ---------------------------------------------------------------------------
# Lifespan: load models at startup, log status
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== AI Server starting — loading models ===")

    dm = load_decomposition()
    _model_status["decomposition"] = not dm.is_mock
    if dm.is_mock:
        log.warning("Decomposition: MOCK mode (no trained model found)")
    else:
        log.info("Decomposition: loaded successfully (%s)", dm.version)

    am = load_assigner()
    _model_status["assigner"] = not am.is_mock
    if am.is_mock:
        log.warning("Assigner: MOCK mode (no trained model found)")
    else:
        log.info("Assigner: loaded successfully (%s)", am.version)

    om = load_optimizer()
    _model_status["optimizer"] = not om.is_mock
    if om.is_mock:
        log.warning("Optimizer: MOCK mode (no trained model found)")
    else:
        log.info("Optimizer: loaded successfully (%s)", om.version)

    loaded = sum(1 for v in _model_status.values() if v)
    log.info(
        "=== Startup complete — %d/3 models loaded, %d/3 in mock mode ===",
        loaded, 3 - loaded,
    )

    yield
    log.info("=== AI Server shutting down ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Workflow360 AI Server",
    version="0.2.0",
    description=(
        "AI inference server for Workflow360 project management. "
        "Provides task decomposition, smart assignment, and sprint bottleneck detection. "
        "Models that are not yet trained fall back to mock mode with realistic fake data."
    ),
    lifespan=lifespan,
)

# Privacy: inspect every POST payload for forbidden PII fields
app.add_middleware(PayloadInspectorMiddleware)

# CORS — support multiple origins including Vercel preview URLs
# NEXT_APP_URL can be a single URL or comma-separated list
_next_app_urls = os.getenv("NEXT_APP_URL", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _next_app_urls.split(",") if o.strip()]

# Also allow all *.vercel.app preview URLs in production via regex
_vercel_preview_regex = r"^https://.*\.vercel\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=_vercel_preview_regex,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
log.info("CORS configured — origins: %s + vercel.app previews", _allowed_origins)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(decomposition.router, tags=["Decomposition (M6)"])
app.include_router(assigner.router, tags=["Assignment (M10)"])
app.include_router(optimizer.router, tags=["Optimizer (M11)"])


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint — used by Render for uptime pings."""
    return {
        "service": "workflow360-ai-server",
        "version": app.version,
        "status": "ok",
    }



# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "models": {
            "decomposition": _model_status["decomposition"],
            "assigner": _model_status["assigner"],
            "optimizer": _model_status["optimizer"],
        },
    }
