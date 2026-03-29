"""Sentence-BERT + Scikit-learn scorer for smart task assignment (M10).

Loads:
  - assigner_model.pkl  (CalibratedClassifierCV from training)
  - assigner_scaler.pkl (StandardScaler for feature normalization)
  - Sentence-BERT: all-MiniLM-L6-v2 (auto-downloaded from HuggingFace, ~22MB)

If pkl files are missing, falls back to mock mode.
"""

import os
import json
import logging

log = logging.getLogger(__name__)

_embedder = None
_scorer = None
_scaler = None
_mock_mode: bool = True
_model_version: str = "mock-v0"


class AssignerModel:
    """Wrapper holding the SBERT embedder, sklearn scorer, and scaler."""

    def __init__(self) -> None:
        self.embedder = _embedder
        self.scorer = _scorer
        self.scaler = _scaler
        self.version = _model_version

    @property
    def is_loaded(self) -> bool:
        return self.embedder is not None

    @property
    def is_mock(self) -> bool:
        return _mock_mode


def load_model() -> AssignerModel:
    """Attempt to load Sentence-BERT embedder and sklearn scorer/scaler.

    The SBERT model is downloaded from HuggingFace on first use (~22MB).
    The scorer and scaler are loaded from joblib pkl files.

    Falls back to mock mode if:
      - pkl files don't exist (model not trained yet)
      - sentence-transformers not installed
      - any loading error occurs
    """
    global _embedder, _scorer, _scaler, _mock_mode, _model_version

    if _embedder is not None:
        return AssignerModel()

    model_dir = os.getenv("MODEL_DIR", "./model_weights")
    model_path = os.path.join(model_dir, "assigner")

    # Check for required pkl files
    scorer_path = os.path.join(model_path, "assigner_model.pkl")
    scaler_path = os.path.join(model_path, "assigner_scaler.pkl")

    has_scorer = os.path.isfile(scorer_path)
    has_scaler = os.path.isfile(scaler_path)

    if not has_scorer or not has_scaler:
        missing = []
        if not has_scorer:
            missing.append("assigner_model.pkl")
        if not has_scaler:
            missing.append("assigner_scaler.pkl")
        log.warning(
            "Assigner model files missing at %s (%s) — running in MOCK mode",
            model_path,
            ", ".join(missing),
        )
        _mock_mode = True
        _model_version = "mock-v0"
        return AssignerModel()

    try:
        import joblib
        from sentence_transformers import SentenceTransformer

        # Load Sentence-BERT (downloads from HuggingFace on first use)
        sbert_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Check if metadata specifies a different SBERT model
        metadata_path = os.path.join(model_path, "training_metadata.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            sbert_name = meta.get("sbert_model", sbert_name)
            _model_version = meta.get("model_version", "assigner-v1")
        else:
            _model_version = "assigner-v1"

        device = os.getenv("DEVICE", "cpu")
        log.info("Loading Sentence-BERT: %s (device=%s)...", sbert_name, device)
        _embedder = SentenceTransformer(sbert_name, device=device)
        log.info(
            "SBERT loaded (dim=%d)", _embedder.get_sentence_embedding_dimension()
        )

        # Load sklearn scorer
        log.info("Loading scorer from %s...", scorer_path)
        _scorer = joblib.load(scorer_path)
        log.info("Scorer loaded: %s", type(_scorer).__name__)

        # Load scaler
        log.info("Loading scaler from %s...", scaler_path)
        _scaler = joblib.load(scaler_path)
        log.info("Scaler loaded: %s", type(_scaler).__name__)

        _mock_mode = False
        log.info("Assigner model ready: %s", _model_version)

    except ImportError as exc:
        log.warning(
            "Missing dependency for assigner model (%s) — MOCK mode", exc
        )
        _embedder = None
        _scorer = None
        _scaler = None
        _mock_mode = True
        _model_version = "mock-v0"

    except Exception as exc:
        log.warning("Failed to load assigner model (%s) — MOCK mode", exc)
        _embedder = None
        _scorer = None
        _scaler = None
        _mock_mode = True
        _model_version = "mock-v0"

    return AssignerModel()


def get_model() -> AssignerModel:
    """Return the singleton (may be in mock mode)."""
    return AssignerModel()
