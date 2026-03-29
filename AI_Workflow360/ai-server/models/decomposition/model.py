"""FLAN-T5 fine-tuned model loader for task decomposition (M6).

If no trained model weights exist at MODEL_DIR/decomposition/,
the model falls back to mock mode so the UI can be built and
tested before training is complete.
"""

import os
import logging

log = logging.getLogger(__name__)

_model = None
_tokenizer = None
_mock_mode: bool = True
_model_version: str = "mock-v0"


class DecompositionModel:
    """Wrapper around the FLAN-T5 checkpoint (or mock fallback)."""

    def __init__(self) -> None:
        self.model = _model
        self.tokenizer = _tokenizer
        self.version = _model_version

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @property
    def is_mock(self) -> bool:
        return _mock_mode


def load_model() -> DecompositionModel:
    """Attempt to load the fine-tuned FLAN-T5 model from disk.

    Falls back to mock mode if:
      - The model directory does not exist
      - Required model files are missing
      - transformers / torch are not installed

    Called once during FastAPI lifespan startup.
    """
    global _model, _tokenizer, _mock_mode, _model_version

    if _model is not None:
        return DecompositionModel()

    model_dir = os.getenv("MODEL_DIR", "./model_weights")
    model_path = os.path.join(model_dir, "decomposition")

    # Check if model directory exists
    if not os.path.isdir(model_path):
        log.warning(
            "Decomposition model dir not found at %s — running in MOCK mode",
            model_path,
        )
        _mock_mode = True
        _model_version = "mock-v0"
        return DecompositionModel()

    # Check for required model files
    required_files = ["config.json"]
    has_weights = (
        os.path.isfile(os.path.join(model_path, "model.safetensors"))
        or os.path.isfile(os.path.join(model_path, "pytorch_model.bin"))
    )
    has_config = os.path.isfile(os.path.join(model_path, "config.json"))

    if not has_weights or not has_config:
        log.warning(
            "Decomposition model files incomplete at %s — running in MOCK mode "
            "(need config.json + model.safetensors or pytorch_model.bin)",
            model_path,
        )
        _mock_mode = True
        _model_version = "mock-v0"
        return DecompositionModel()

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        device = os.getenv("DEVICE", "cpu")
        log.info("Loading decomposition model from %s ...", model_path)

        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            _model = _model.to("cuda")
            log.info("Model moved to CUDA")
        else:
            _model = _model.to("cpu")
            if device == "cuda":
                log.warning("CUDA requested but not available — using CPU")

        _model.eval()
        _mock_mode = False

        # Read version from metadata if available
        metadata_path = os.path.join(model_path, "training_metadata.json")
        if os.path.isfile(metadata_path):
            import json
            with open(metadata_path) as f:
                meta = json.load(f)
            _model_version = meta.get("model_version", "flan-t5-pm-v1")
        else:
            _model_version = "flan-t5-pm-v1"

        param_count = sum(p.numel() for p in _model.parameters()) / 1e6
        log.info(
            "Decomposition model loaded: %s (%.1fM params) on %s",
            _model_version, param_count, device,
        )

    except ImportError as exc:
        log.warning(
            "Missing dependency for decomposition model (%s) — MOCK mode", exc
        )
        _model = None
        _tokenizer = None
        _mock_mode = True
        _model_version = "mock-v0"

    except Exception as exc:
        log.warning(
            "Failed to load decomposition model (%s) — MOCK mode", exc
        )
        _model = None
        _tokenizer = None
        _mock_mode = True
        _model_version = "mock-v0"

    return DecompositionModel()


def get_model() -> DecompositionModel:
    """Return the singleton (may be in mock mode)."""
    return DecompositionModel()
