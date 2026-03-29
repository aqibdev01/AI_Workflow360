# Workflow360 AI Server

FastAPI server hosting three AI modules for the Workflow360 project management platform.

## Modules

| Module | Endpoint | Model | Description |
|--------|----------|-------|-------------|
| M6 — Decomposition | `POST /api/decompose` | FLAN-T5 | Breaks tasks into AI-suggested subtasks |
| M10 — Assignment | `POST /api/suggest-assignee` | Sentence-BERT + sklearn | Scores and ranks team members for task assignment |
| M11 — Optimizer | `POST /api/analyze-sprint` | XGBoost | Predicts sprint bottleneck risks and recommends actions |

All modules fall back to **mock mode** with realistic fake data when trained model weights are not available.
This lets you build and test the full UI before model training is complete.

## Quick Start

```bash
# 1. Create and activate a virtual environment
cd AI_Workflow360/ai-server
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set AI_SERVER_API_KEY to a secret key

# 4. Run the server
uvicorn main:app --reload --port 8000
```

The server starts at `http://localhost:8000`.

## API Documentation

Once running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Endpoints

### Health Check
```
GET /health
→ {"status": "ok", "models": {"decomposition": false, "assigner": false, "optimizer": false}}
```

`false` = mock mode (no trained model loaded), `true` = real model loaded.

### Task Decomposition (M6)
```
POST /api/decompose
Headers: X-API-Key: <your_key>
Body: {"task_id": "...", "title": "...", "description": "...", "priority": "medium", "project_context": "", "existing_tags": []}
```

```
GET /api/decompose/{task_id}/history
Headers: X-API-Key: <your_key>
```

### Smart Assignment (M10)
```
POST /api/suggest-assignee
Headers: X-API-Key: <your_key>
Body: {"task_id": "...", "title": "...", "description": "...", "priority": "medium", "tags": [], "story_points": null, "project_members": [...]}
```

### Sprint Analysis (M11)
```
POST /api/analyze-sprint
Headers: X-API-Key: <your_key>
Body: {"sprint_id": "...", "sprint_name": "...", "start_date": "...", "end_date": "...", "capacity": null, "tasks": [...], "member_workloads": {}}
```

```
POST /api/analyze-project
Headers: X-API-Key: <your_key>
Body: {"project_id": "...", "sprints": [<SprintAnalysisRequest>, ...]}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_SERVER_API_KEY` | Shared secret for X-API-Key authentication | (required) |
| `NEXT_APP_URL` | Next.js app origin for CORS | `http://localhost:3000` |
| `MODEL_DIR` | Directory containing trained model weights | `./model_weights` |
| `DEVICE` | PyTorch device (`cpu` or `cuda`) | `cpu` |

## Model Weights

Place trained model weights in `MODEL_DIR`:
```
model_weights/
├── decomposition/    ← FLAN-T5 checkpoint (config.json, model.safetensors, tokenizer.json, ...)
├── assigner/         ← Sentence-BERT checkpoint + optional scorer.pkl
└── optimizer/
    └── xgb_bottleneck.pkl  ← Trained XGBoost model
```

If a model directory doesn't exist, that module runs in mock mode.

## Privacy

All endpoints enforce a strict data privacy boundary:
- A **payload inspector middleware** rejects any request containing forbidden PII fields
- Each inference function runs an `assert_no_pii()` check before the model sees any data
- Violations are logged to `security.log` (field names only, never values)
