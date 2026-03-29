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

# CORS — allow the Next.js app origin from env
next_app_url = os.getenv("NEXT_APP_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[next_app_url],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(decomposition.router, tags=["Decomposition (M6)"])
app.include_router(assigner.router, tags=["Assignment (M10)"])
app.include_router(optimizer.router, tags=["Optimizer (M11)"])


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
