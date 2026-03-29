# AI Workflow360 — Manual / Custom Steps

This file documents any steps that must be performed manually or require custom configuration
that cannot be automated through code alone.

---

## Phase 1: Schema Migration

### Step 1 — Run the migration on Supabase

The migration file is at:
```
Worflow360/supabase/migrations/008_ai_module_fields.sql
```

**Option A: Supabase CLI (recommended for local dev)**
```bash
cd Worflow360
supabase db reset        # resets and re-runs all migrations from scratch
# OR apply just this migration:
supabase migration up
```

**Option B: Supabase Dashboard (for hosted project)**
1. Open your Supabase project dashboard → **SQL Editor**
2. Copy-paste the contents of `008_ai_module_fields.sql`
3. Click **Run**
4. Verify in **Table Editor** that new columns and tables appear

### Step 2 — Verify the migration

After running, confirm these exist:

**New columns on `tasks`:**
- `story_points`, `estimated_days`, `actual_days`, `tags`, `complexity_score`
- `ai_suggested_assignee_id`, `ai_assignee_confidence`
- `parent_task_id`, `is_ai_generated`, `decomposition_status`

**New columns on `sprints`:**
- `velocity`, `capacity`, `ai_risk_score`, `ai_risk_factors`, `ai_analyzed_at`

**New tables:**
- `user_skills` (with unique constraint on user_id + skill)
- `ai_task_decompositions`
- `ai_assignment_logs`
- `ai_bottleneck_reports`

**New enum types:**
- `decomposition_status`
- `skill_level`
- `decomposition_review_status`
- `risk_level`

**RLS policies** — Check that all 4 new tables have RLS enabled and policies attached.

### Step 3 — Regenerate Supabase types (if using auto-generated types)

If you use `supabase gen types typescript` to generate types:
```bash
supabase gen types typescript --local > types/supabase.ts
```
This will pick up the new columns and tables automatically.

---

---

## Phase 2: AI Server Setup

### Step 1 — Create a Python virtual environment

```bash
cd AI_Workflow360/ai-server

# Create venv
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** The `torch==2.3.0` in requirements.txt installs CPU-only by default.
> If you have an NVIDIA GPU and want CUDA support, install torch separately first:
> ```bash
> pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
> ```
> Then install the rest: `pip install -r requirements.txt`

### Step 3 — Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set:
- `AI_SERVER_API_KEY` — Generate a random secret (e.g. `openssl rand -hex 32`). This same key must be set in the Next.js app's env to authenticate requests.
- `NEXT_APP_URL` — The origin of your Next.js app (default: `http://localhost:3000`)
- `MODEL_DIR` — Path to model weights directory (default: `./model_weights`)

### Step 4 — Start the server

```bash
cd AI_Workflow360/ai-server
uvicorn main:app --reload --port 8000
```

The server will:
1. Try to load model weights from `MODEL_DIR/{decomposition,assigner,optimizer}/`
2. Fall back to **mock mode** for any model whose weights are missing (returns realistic fake data)
3. Log which models loaded and which are in mock mode
4. Start serving on `http://localhost:8000`

### Step 5 — Verify the server is running

```bash
# Health check — shows which models are loaded vs mock
curl http://localhost:8000/health
# Expected (no trained models yet):
# {"status":"ok","models":{"decomposition":false,"assigner":false,"optimizer":false}}

# Check API docs (interactive Swagger UI)
# Open in browser: http://localhost:8000/docs
```

### Step 6 — Test an endpoint (optional)

```bash
curl -X POST http://localhost:8000/api/decompose \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_key_here" \
  -d '{
    "task_id": "test-1",
    "title": "Implement user authentication",
    "description": "Add login and signup flows with email verification",
    "priority": "high",
    "existing_tags": ["auth", "frontend"]
  }'
```

Even in mock mode, you'll get a realistic response with subtask suggestions.

### Step 7 — Set the AI server URL in Next.js

In your Next.js app's `.env.local`, add:
```
AI_SERVER_URL=http://localhost:8000
AI_API_KEY=your_secret_key_here
```

These will be used by Next.js API routes to call the AI server.

---

---

## Phase 3: Privacy Layer — No manual steps required

The privacy boundary layer is fully automated:
- **Next.js side:** `sanitizeForAI()` runs automatically on every AI server call via `lib/ai/client.ts`.
- **FastAPI side:** `PayloadInspectorMiddleware` runs on every POST request automatically.
- **Inference side:** `assert_no_pii()` runs at the top of every inference function automatically.

### Verifying privacy enforcement

To confirm the payload inspector works, try sending a forbidden field:
```bash
curl -X POST http://localhost:8000/api/analyze-sprint \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key-here" \
  -d '{"sprint_id":"x","project_id":"x","sprint_name":"x","start_date":"2026-01-01","end_date":"2026-01-14","email":"test@example.com","tasks":[],"members":[]}'
```

Expected response (HTTP 400):
```json
{
  "error": "payload_violation",
  "detail": "Request contains forbidden data fields that violate the AI data privacy boundary.",
  "fields": ["Forbidden field detected: email"]
}
```

Check `ai-server/security.log` for the logged violation (field name only, never the value).

### Where to use AIDataNotice component

Import and add `<AIDataNotice />` inside any future AI feature panel:
```tsx
import { AIDataNotice } from "@/components/ai/AIDataNotice";

// Inside your component JSX:
<AIDataNotice />
```

The banner shows once per user, dismissed permanently via "Got it" button (localStorage).

---

## Phase 4: Training the Decomposition Model (FLAN-T5)

### Step 1 — Open the notebook in Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. File > Upload Notebook
3. Upload `AI_Workflow360/ai-server/notebooks/train_decomposition.ipynb`
4. Or upload `train_decomposition.py` and run cell-by-cell

### Step 2 — Enable GPU

1. Runtime > Change runtime type
2. Select **T4 GPU** (free tier)
3. Click Save

### Step 3 — Run all cells

1. Runtime > Run all
2. Training takes approximately **45-90 minutes** on T4 GPU
3. Monitor the training loss and ROUGE-L scores during evaluation

### Step 4 — Download the trained model

After training completes, the notebook will:
- Save the model to `/content/flan-t5-pm-decomposition/`
- Create a zip: `flan-t5-pm-decomposition.zip`
- Auto-download it to your browser (in Colab)

If auto-download doesn't work, go to the Colab file browser (folder icon on left) and download manually.

### Step 5 — Deploy to AI server

```bash
# Unzip the model
cd AI_Workflow360/ai-server
mkdir -p model_weights/decomposition
unzip ~/Downloads/flan-t5-pm-decomposition.zip -d model_weights/decomposition/

# If the zip created a subdirectory, move contents up:
# mv model_weights/decomposition/flan-t5-pm-decomposition/* model_weights/decomposition/
```

### Step 6 — Verify the model loaded

```bash
# Restart the server
uvicorn main:app --reload --port 8000

# Check health endpoint
curl http://localhost:8000/health
# Expected: {"status":"ok","models":{"decomposition":true,"assigner":false,"optimizer":false}}
```

`"decomposition": true` means the real model is loaded (no longer mock).

### Step 7 — Check evaluation metrics

After training, review the console output for:
- **ROUGE-L > 0.35** — target metric for this dataset size
- Manual inspection of 10 test predictions — subtasks should be relevant and well-structured

The model also saves `training_metadata.json` with all metrics for FYP documentation.

---

## Phase 5: Task Decomposition Integration — Configuration

### Step 1 — Ensure AI server environment variables match Next.js

Both sides must share the same API key. In your AI server `.env`:
```
AI_SERVER_API_KEY=your_shared_secret_key
```

In your Next.js `.env.local`:
```
AI_SERVER_URL=http://localhost:8000
AI_API_KEY=your_shared_secret_key
```

These must match exactly.

### Step 2 — Test the full pipeline (mock mode)

With both servers running:

```bash
# 1. Start AI server (mock mode — no trained model needed)
cd AI_Workflow360/ai-server
uvicorn main:app --reload --port 8000

# 2. Start Next.js app
cd Worflow360
npm run dev
```

Test via the Next.js API route (requires a valid Supabase session):
```bash
# This will fail with 401 if not authenticated — that's expected.
# Use the frontend or Supabase auth to get a session cookie first.
curl -X POST http://localhost:3000/api/ai/decompose \
  -H "Content-Type: application/json" \
  -d '{"task_id": "your-task-uuid-here"}'
```

### Step 3 — Using the client library in frontend components

```tsx
import {
  requestDecomposition,
  acceptSubtasks,
  rejectDecomposition,
  getDecompositionHistory,
} from "@/lib/ai/decomposition";

// Request decomposition
const result = await requestDecomposition(taskId);
// result.subtasks — array of suggestions
// result.decomposition_id — DB record ID

// Accept specific subtasks (creates real tasks)
const { created_tasks } = await acceptSubtasks(
  result.decomposition_id,
  [0, 1, 3]  // indices of subtasks to accept
);

// Or reject the entire decomposition
await rejectDecomposition(result.decomposition_id);

// View history
const history = await getDecompositionHistory(taskId);
```

### Step 4 — Check the request log

After making decomposition requests, check the JSON log:
```bash
cat AI_Workflow360/ai-server/logs/decomposition_requests.jsonl
```

Each line is a JSON object: `{timestamp, task_id, num_subtasks, confidence, model_version}`.

---

## Phase 6: Task Decomposition UI — No manual steps required

The decomposition UI is fully integrated into the existing Kanban board. No additional configuration needed beyond Phase 5 (AI server + API key setup).

### How to use

1. Open any task in the Kanban board (click the card)
2. If the task has no subtasks and you have contributor+ role, you'll see an **"AI Decompose"** button
3. Click it — the AI server analyzes the task and returns subtask suggestions
4. Review the suggestions in the panel:
   - Edit titles inline before accepting
   - Select/deselect individual subtasks
   - View confidence scores and metadata
5. Click **"Accept Selected"** to create real task records
6. The accepted subtasks appear on the Kanban board with a purple sparkle icon
7. The parent task card shows a subtask progress bar

### Testing in mock mode

Even without a trained model, the AI server returns realistic mock subtasks based on keyword matching. This lets you test the full UI flow:
- Tasks with "auth" in the title get authentication-related subtasks
- Tasks with "api" get API implementation subtasks
- Tasks with "ui" get frontend subtasks
- Other tasks get generic subtasks based on priority level

---

## Phase 8: Task Assigner Integration — Configuration

### Step 1 — Ensure AI server has the assigner model files

After running the training notebook (Phase 7), your `model_weights/assigner/` directory should contain:
```
model_weights/assigner/
├── assigner_model.pkl        # CalibratedClassifierCV
├── assigner_scaler.pkl       # StandardScaler
└── training_metadata.json    # Model version, SBERT name, metrics
```

### Step 2 — Add SkillsManager to user settings/profile

Import and use `SkillsManager` in any user profile component:
```tsx
import { SkillsManager } from "@/components/ai/SkillsManager";

// Edit mode (own profile):
<SkillsManager userId={currentUserId} />

// Read-only mode (viewing teammate):
<SkillsManager userId={memberId} readOnly />
```

### Step 3 — Test the assignment flow

```bash
# With both servers running:
curl -X POST http://localhost:8000/api/suggest-assignee \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_key_here" \
  -d '{
    "task_id": "test-1",
    "title": "Build login page",
    "description": "Create login and signup forms with validation",
    "priority": "high",
    "tags": ["React", "TypeScript", "auth"],
    "project_members": [
      {"user_id": "u1", "full_name": "u1", "role": "contributor", "skills": ["React", "TypeScript", "CSS"], "current_task_count": 2, "current_story_points": 5, "completed_tasks_last_30d": 8},
      {"user_id": "u2", "full_name": "u2", "role": "contributor", "skills": ["Python", "FastAPI", "PostgreSQL"], "current_task_count": 5, "current_story_points": 20, "completed_tasks_last_30d": 12}
    ]
  }'
```

### Step 4 — Verify in health endpoint

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","models":{"decomposition":...,"assigner":true,"optimizer":...}}
```

---

## Phase 10: Training the Bottleneck Predictor (XGBoost)

### Step 1 — Run the training script

**Option A: Google Colab**
1. Upload `AI_Workflow360/ai-server/notebooks/train_optimizer.ipynb` to Colab
2. Runtime type: CPU is fine (no GPU needed)
3. Run all cells — takes ~5-10 minutes

**Option B: Run locally**
```bash
cd AI_Workflow360/ai-server
python -m venv venv && venv\Scripts\activate
pip install xgboost scikit-learn pandas numpy joblib matplotlib shap
python notebooks/train_optimizer.py
```

### Step 2 — Review metrics

Check the console output for:
- **Accuracy >82%** — target metric
- **Confusion matrix** — should show good separation between all 4 risk levels
- **SHAP feature importance** — days_remaining and completion_rate should rank high
- **Inference time <10ms** per sprint

### Step 3 — Deploy to AI server

```bash
cd AI_Workflow360/ai-server
mkdir -p model_weights/optimizer
unzip ~/Downloads/optimizer-model.zip -d model_weights/optimizer/
```

Required files:
```
model_weights/optimizer/
├── optimizer_model.pkl           # XGBoost model
├── optimizer_scaler.pkl          # StandardScaler
├── optimizer_explainer.pkl       # SHAP TreeExplainer
├── optimizer_label_encoder.pkl   # LabelEncoder (risk levels)
└── training_metadata.json        # Metrics and feature names
```

### Step 4 — Verify

```bash
uvicorn main:app --reload --port 8000
curl http://localhost:8000/health
# Expected: {"status":"ok","models":{"decomposition":...,"assigner":...,"optimizer":true}}
```

---

## Phase 7: Training the Assigner Model (Sentence-BERT + sklearn)

### Step 1 — Run the training script

**Option A: Google Colab (recommended for consistency)**
1. Upload `AI_Workflow360/ai-server/notebooks/train_assigner.ipynb` to Colab
2. Runtime type: CPU is fine (no GPU needed)
3. Run all cells — takes ~5-10 minutes

**Option B: Run locally**
```bash
cd AI_Workflow360/ai-server
python -m venv venv && venv\Scripts\activate   # Windows
pip install sentence-transformers scikit-learn pandas numpy joblib
python notebooks/train_assigner.py
```

### Step 2 — Review metrics

After training, check the console output for:
- **Accuracy >85%** — target metric
- **F1 score** — should be high for the positive class
- **Inference time <50ms** for a team of 10 members
- **Feature importance ranking** — skill_match should be highest

### Step 3 — Deploy to AI server

```bash
cd AI_Workflow360/ai-server
mkdir -p model_weights/assigner

# If from Colab:
unzip ~/Downloads/assigner-model.zip -d model_weights/assigner/

# If local:
cp assigner-model/* model_weights/assigner/
```

### Step 4 — Verify

```bash
uvicorn main:app --reload --port 8000
curl http://localhost:8000/health
# Expected: {"status":"ok","models":{"decomposition":false,"assigner":true,"optimizer":false}}
```

### Note on Sentence-BERT

The training script uses `all-MiniLM-L6-v2` (22 MB). The AI server downloads it automatically from HuggingFace Hub on first use. No manual model file copy needed for SBERT — only the sklearn model and scaler `.pkl` files.

---

## Notes

- The migration uses `ADD COLUMN IF NOT EXISTS` and `CREATE TABLE IF NOT EXISTS` so it is
  **safe to re-run** without errors.
- Enum types use the `DO $$ BEGIN ... EXCEPTION WHEN duplicate_object` pattern for idempotency.
- Foreign keys on `ai_suggested_assignee_id` and `reviewed_by` reference `auth.users` (Supabase
  Auth schema), not `public.users`. This is intentional — it ensures referential integrity
  against the actual auth user records.
