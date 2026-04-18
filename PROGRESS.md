# AI Workflow360 — Development Progress

## Overview
This document tracks all AI-related development progress for the Workflow360 FYP project.
AI server code lives in `AI_Workflow360/`. Database migrations and frontend changes live in `Worflow360/`.

---

## Phase 1: Schema Extensions for AI Modules

**Status:** Completed
**Date:** 2026-03-25
**Migration file:** `Worflow360/supabase/migrations/008_ai_module_fields.sql`

### What was done

#### 1. New Enum Types Created
| Enum | Values |
|------|--------|
| `decomposition_status` | `none`, `suggested`, `partially_accepted`, `fully_accepted` |
| `skill_level` | `beginner`, `intermediate`, `expert` |
| `decomposition_review_status` | `pending`, `accepted`, `partially_accepted`, `rejected` |
| `risk_level` | `low`, `medium`, `high`, `critical` |

#### 2. Tasks Table — New Columns
| Column | Type | Purpose |
|--------|------|---------|
| `story_points` | smallint | Fibonacci scale (1,2,3,5,8,13), NULL = unestimated |
| `estimated_days` | numeric(4,1) | AI or human estimate, e.g. 2.5 |
| `actual_days` | numeric(4,1) | Filled on task completion |
| `tags` | text[] | Labels like `['frontend','auth','bug']` |
| `complexity_score` | numeric(3,2) | AI-computed 0.00–1.00 |
| `ai_suggested_assignee_id` | uuid (FK → auth.users) | M10 module suggestion |
| `ai_assignee_confidence` | numeric(3,2) | 0.00–1.00 confidence |
| `parent_task_id` | uuid (self-FK → tasks) | Subtask hierarchy for M6 |
| `is_ai_generated` | boolean | Marks AI-decomposed subtasks |
| `decomposition_status` | enum | Tracks decomposition acceptance |

**Constraints added:** Fibonacci check on story_points, range checks on confidence/complexity scores.

#### 3. Sprints Table — New Columns
| Column | Type | Purpose |
|--------|------|---------|
| `velocity` | numeric(5,1) | Story points completed (filled at sprint end) |
| `capacity` | numeric(5,1) | Total SP the team can handle |
| `ai_risk_score` | numeric(3,2) | M11 bottleneck risk 0.00–1.00 |
| `ai_risk_factors` | jsonb | `{"overloaded_members": [...], "blockers": [...]}` |
| `ai_analyzed_at` | timestamptz | When AI last analyzed this sprint |

#### 4. New Table: `user_skills`
Tracks team member expertise for AI-based assignment matching (M10).

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid PK | |
| `user_id` | uuid FK → auth.users | |
| `skill` | text | e.g. 'React', 'Python' |
| `level` | skill_level enum | beginner/intermediate/expert |
| `created_at` | timestamptz | |

UNIQUE constraint on `(user_id, skill)`.

#### 5. New Table: `ai_task_decompositions`
Stores M6 decomposition suggestions before user accepts/rejects.

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid PK | |
| `parent_task_id` | uuid FK → tasks | |
| `suggested_subtasks` | jsonb | Array of `{title, description, priority, story_points, tags, estimated_days}` |
| `model_version` | text | e.g. 'flan-t5-pm-v1' |
| `confidence_score` | numeric(3,2) | 0.00–1.00 |
| `status` | decomposition_review_status | pending/accepted/partially_accepted/rejected |
| `reviewed_at` | timestamptz | |
| `reviewed_by` | uuid FK → auth.users | |

#### 6. New Table: `ai_assignment_logs`
Audit trail for M10 smart assignment suggestions.

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid PK | |
| `task_id` | uuid FK → tasks | |
| `suggested_assignee_id` | uuid FK → auth.users | |
| `confidence_score` | numeric(3,2) | |
| `scoring_breakdown` | jsonb | `{skill_match, workload, role_match, availability}` |
| `was_accepted` | boolean (nullable) | NULL until user acts |
| `final_assignee_id` | uuid FK → auth.users | Who was actually assigned |
| `model_version` | text | |

#### 7. New Table: `ai_bottleneck_reports`
Stores M11 sprint risk analysis results.

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid PK | |
| `sprint_id` | uuid FK → sprints | |
| `project_id` | uuid FK → projects | |
| `risk_level` | risk_level enum | low/medium/high/critical |
| `risk_score` | numeric(3,2) | |
| `bottlenecks` | jsonb | Array of `{type, description, affected_tasks, severity}` |
| `recommendations` | jsonb | Array of `{action, reason, priority}` |
| `model_version` | text | |

#### 8. Indexes Added
- `tasks(parent_task_id)`, `tasks(assignee_id, status)`, `tasks(sprint_id, status)`
- `user_skills(user_id)`, `user_skills(skill)`
- `ai_bottleneck_reports(sprint_id)`, `ai_bottleneck_reports(project_id)`
- `ai_assignment_logs(task_id)`
- `ai_task_decompositions(parent_task_id)`

#### 9. Row Level Security (RLS)
All 4 new tables have RLS enabled with policies:
- **user_skills:** Users manage their own; project teammates can view.
- **ai_task_decompositions:** Project members can view; owners/leads can manage.
- **ai_assignment_logs:** Project members can view; owners/leads can manage.
- **ai_bottleneck_reports:** Project members can view; owners/leads can manage.

### Existing Enum Values (confirmed, not changed)
| Enum | Values |
|------|--------|
| `task_status` | `todo`, `in_progress`, `review`, `done`, `blocked` |
| `task_priority` | `low`, `medium`, `high`, `urgent` |
| `sprint_status` | `planned`, `active`, `completed`, `cancelled` |
| `project_role` | `owner`, `lead`, `contributor`, `viewer` |

---

---

## Phase 2: FastAPI AI Server Scaffold

**Status:** Completed
**Date:** 2026-03-25
**Location:** `AI_Workflow360/ai-server/`

### What was done

Built the complete Python FastAPI server that hosts all three AI modules. The Next.js frontend calls this server via HTTP.

#### Server Structure
```
ai-server/
├── main.py                     ← FastAPI app, lifespan model loading, CORS, /health with model status
├── README.md                   ← Setup and usage documentation
├── requirements.txt            ← Pinned dependencies
├── .env.example                ← AI_SERVER_API_KEY, NEXT_APP_URL, MODEL_DIR
├── models/
│   ├── decomposition/
│   │   ├── model.py            ← FLAN-T5 loader with is_mock tracking
│   │   └── inference.py        ← Real inference + mock templates per priority
│   ├── assigner/
│   │   ├── model.py            ← Sentence-BERT loader with is_mock tracking
│   │   └── inference.py        ← Real SBERT scoring + mock plausible candidates
│   └── optimizer/
│       ├── model.py            ← XGBoost loader with is_mock tracking
│       └── inference.py        ← Real ML + rule-based heuristics + mock fallback
├── routers/
│   ├── decomposition.py        ← POST /api/decompose + GET /api/decompose/{id}/history
│   ├── assigner.py             ← POST /api/suggest-assignee
│   └── optimizer.py            ← POST /api/analyze-sprint + POST /api/analyze-project
├── schemas/
│   ├── decomposition.py        ← DecomposeRequest / DecomposeResponse / SubtaskSuggestion
│   ├── assigner.py             ← AssignRequest / AssignResponse / MemberProfile / AssigneeSuggestion
│   └── optimizer.py            ← SprintAnalysisRequest / SprintAnalysisResponse / Bottleneck / Recommendation
└── utils/
    ├── auth.py                 ← X-API-Key header validation middleware
    └── preprocessing.py        ← Text cleaning, task prompt builder, member profile builder
```

#### API Endpoints
| Method | Path | Module | Description |
|--------|------|--------|-------------|
| POST | `/api/decompose` | M6 | Decomposes a task into AI-suggested subtasks |
| GET | `/api/decompose/{task_id}/history` | M6 | Past decomposition results for a task |
| POST | `/api/suggest-assignee` | M10 | Scores team members and suggests best assignee |
| POST | `/api/analyze-sprint` | M11 | Analyzes sprint for bottleneck risks |
| POST | `/api/analyze-project` | M11 | Batch analysis of all active sprints in a project |
| GET | `/health` | — | Health check with model status booleans |

#### Key Design Decisions
- **Mock fallback:** If model weights don't exist at `MODEL_DIR`, each module returns realistic fake data so the UI can be built and tested before training. The `/health` endpoint shows which models are loaded (`true`) vs mock (`false`).
  - Decomposition: mock returns priority-appropriate subtask templates
  - Assigner: mock returns plausible scored candidates
  - Optimizer: mock uses rule-based heuristic detection (still useful in production)
- **API key auth:** All endpoints require `X-API-Key` header matching `AI_SERVER_API_KEY` env var.
- **Singleton model loading:** Models load once during FastAPI lifespan startup, not per-request.
- **CORS configured** from `NEXT_APP_URL` env var (defaults to localhost:3000).
- **Pydantic v2** schemas with full validation.

#### Assigner Scoring Weights (heuristic mode)
| Dimension | Weight | Source |
|-----------|--------|--------|
| Skill match | 0.40 | Cosine similarity of SBERT embeddings |
| Workload | 0.25 | Inverse of current task count / 8 |
| Role match | 0.20 | Static scores: contributor=1.0, lead=0.8, owner=0.6, viewer=0.2 |
| Availability | 0.15 | Inverse of task count / 12 |

#### Optimizer Bottleneck Detection Rules
| Check | Trigger | Severity |
|-------|---------|----------|
| Member overload | >5 active tasks per member | medium/high |
| Blocked chain | Any blocked tasks exist | medium/critical (>20% = critical) |
| Scope creep | Total SP > 120% capacity | high |
| Unassigned tasks | Tasks without assignee | medium/high |
| Timeline pressure | >70% time elapsed + >50% incomplete | +0.15 risk boost |

#### Dependencies (requirements.txt)
```
fastapi==0.111.0, uvicorn==0.29.0, pydantic==2.7.1
transformers==4.41.0, torch==2.3.0
sentence-transformers==3.0.0, scikit-learn==1.5.0
xgboost==2.0.3, numpy==1.26.4
python-dotenv==1.0.1, httpx==0.27.0
```

### Files also updated
- `Worflow360/types/database.ts` — Added 4 new enum types, 4 new table type definitions (Row/Insert/Update), extended Task/Sprint types with AI fields, added AI JSONB structure interfaces.

---

---

## Phase 3: Data Privacy Boundary Layer

**Status:** Completed
**Date:** 2026-03-25

### What was done

Implemented a multi-layered privacy enforcement system ensuring the AI server **only** receives task/sprint/workload data — never communications, files, auth data, or PII.

#### Architecture: 4 Defense Layers

```
Layer 1 (Next.js types)     → AITaskPayload / AIMemberPayload / AISprintPayload
                                Only these shapes can be constructed
Layer 2 (Next.js guard)     → sanitizeForAI() strips forbidden fields recursively
                                Runs on every outbound payload
Layer 3 (FastAPI middleware) → PayloadInspectorMiddleware inspects every POST body
                                Rejects + logs violations before reaching handlers
Layer 4 (Inference guard)   → assert_no_pii() hard-stops before model execution
                                Last resort — model never sees PII
```

#### Files Created (Next.js side)

| File | Purpose |
|------|---------|
| `Worflow360/lib/ai/sanitize.ts` | Typed payload definitions — the ONLY shapes allowed to reach AI |
| `Worflow360/lib/ai/guards.ts` | `sanitizeForAI()` recursive PII stripper + `assertNoForbiddenFields()` validator |
| `Worflow360/lib/ai/client.ts` | HTTP client to AI server — applies `sanitizeForAI()` on every call |
| `Worflow360/app/api/ai/decompose/route.ts` | POST handler for M6 — fetches only permitted task fields |
| `Worflow360/app/api/ai/suggest-assignee/route.ts` | POST handler for M10 — UUIDs only to AI, names added after response |
| `Worflow360/app/api/ai/analyze-sprint/route.ts` | POST handler for M11 — sprint/task data only, no user PII |
| `Worflow360/components/ai/AIDataNotice.tsx` | User transparency banner — dismissible, localStorage-persisted |

#### Files Created/Updated (FastAPI side)

| File | Change |
|------|--------|
| `ai-server/utils/auth.py` | Added `PayloadInspectorMiddleware`, `inspect_payload()`, `FORBIDDEN_FIELDS` set, `security.log` logger |
| `ai-server/utils/privacy.py` | **New** — `assert_no_pii()` last-resort guard called in every inference function |
| `ai-server/main.py` | Registered `PayloadInspectorMiddleware` |
| `ai-server/models/decomposition/inference.py` | Added `assert_no_pii()` call at top of `decompose_task()` |
| `ai-server/models/assigner/inference.py` | Added `assert_no_pii()` call at top of `suggest_assignee()` |
| `ai-server/models/optimizer/inference.py` | Added `assert_no_pii()` call at top of `analyze_sprint()` |

#### Forbidden Fields (blocked at all layers)

**User PII:** email, password, encrypted_password, phone, avatar_url, full_name, security_question, security_answer, raw_user_meta_data, raw_app_meta_data, recovery_token, confirmation_token

**Communication:** content (chat), body (mail), message, subject

**Files:** storage_path, public_url, file_path

**Auth/Session:** token, session, cookie, access_token, refresh_token

#### Blocked Data Sources (Next.js API routes never query these for AI)
- `messages`, `direct_messages`, `direct_message_threads`
- `mail_messages`, `mail_recipients`
- `channels`, `files`, `file_shares`
- `notifications`
- `auth.users` (only `public.users.id` + `full_name` for post-response UI display)

#### Key Privacy Design Decisions
- **Member names are NEVER sent to AI.** The assigner sends `user_id` (UUID) as the `name` field. Real names are fetched **after** the AI responds, purely for frontend display.
- **Violations are logged to `security.log`** with field names only — never the actual values (to avoid logging PII even in error logs).
- **Every API route has a data boundary comment block** documenting exactly what data sources are permitted.
- **`AIDataNotice` component** informs users transparently about what data AI uses, dismissible via "Got it" with localStorage persistence.

---

---

## Phase 4: Task Decomposition Training Notebook

**Status:** Completed
**Date:** 2026-03-25
**Location:** `AI_Workflow360/ai-server/notebooks/`

### What was done

Created a complete Google Colab training notebook for fine-tuning FLAN-T5-small on software PM task decomposition.

#### Files Created
| File | Purpose |
|------|---------|
| `notebooks/train_decomposition.py` | Python script version (runnable locally or cell-by-cell in Colab) |
| `notebooks/train_decomposition.ipynb` | Jupyter notebook version (upload directly to Colab) |

#### Training Pipeline (7 sections)

| Section | What it does |
|---------|-------------|
| 1. Setup | Installs transformers, datasets, torch, accelerate, evaluate, rouge_score |
| 2. Dataset Loading | Loads 500 structural pattern examples from `sander-wood/text-infilling` |
| 3. Synthetic Data | Generates 2100 examples across 6 PM categories (frontend, backend, devops, testing, design, docs) |
| 4. Tokenization | FLAN-T5-small tokenizer, max input 512 tokens, max output 256 tokens |
| 5. Fine-tuning | 5 epochs, batch size 8, lr 3e-4, linear warmup 100 steps, eval every 200 steps, best checkpoint by ROUGE-L |
| 6. Evaluation | ROUGE-1/2/L scores on test set + manual inspection of 10 predictions |
| 7. Save & Export | Saves model + tokenizer + metadata, creates downloadable zip |

#### Synthetic Data Categories (2100+ examples)
| Category | Templates | Augmented to |
|----------|-----------|-------------|
| Frontend | 8 tasks (React, UI, charts, DnD, forms) | 350 |
| Backend | 6 tasks (API, auth, sprints, storage, realtime) | 350 |
| DevOps | 4 tasks (CI/CD, Docker, monitoring, staging) | 350 |
| Testing | 4 tasks (unit, integration, E2E, framework) | 350 |
| Design | 3 tasks (UI kit, wireframes, responsive) | 350 |
| Documentation | 3 tasks (API docs, setup guide, ADRs) | 350 |

#### Model Output Format
```
SUBTASK_1: title | PRIORITY: p | POINTS: n | DAYS: n | TAGS: t1,t2
SUBTASK_2: title | PRIORITY: p | POINTS: n | DAYS: n | TAGS: t1,t2
```

#### Training Configuration
- **Model:** google/flan-t5-small (77M params, ~300 MB)
- **Hardware:** Google Colab T4 GPU (free tier, 15 GB VRAM)
- **Expected time:** 45-90 minutes
- **Target metric:** ROUGE-L > 0.35
- **Data split:** 80% train / 10% val / 10% test
- **FP16:** enabled for T4 memory efficiency
- **Early stopping:** patience 3 evaluations

#### Post-Training Usage
1. Download `flan-t5-pm-decomposition.zip` from Colab
2. Unzip to `ai-server/model_weights/decomposition/`
3. Restart AI server
4. `/health` should show `"decomposition": true`

---

## Phase 5: Task Decomposition — Full Stack Integration

**Status:** Completed
**Date:** 2026-03-25

### What was done

Implemented the complete decomposition pipeline from model loader through to client library, connecting the AI server inference to the Next.js frontend via authenticated API routes with database persistence.

#### AI Server — Model Loader (`models/decomposition/model.py`)
- `DecompositionModel` class wrapping the FLAN-T5 checkpoint (or mock fallback)
- `load_model()` checks for `config.json` + `model.safetensors`/`pytorch_model.bin`
- Uses `AutoModelForSeq2SeqLM` and `AutoTokenizer` from transformers
- Reads `training_metadata.json` for model version if available
- Falls back to mock mode with clear logging if anything is missing
- `get_model()` returns the singleton instance

#### AI Server — Inference (`models/decomposition/inference.py`)
- **Real mode:** Builds prompt matching training format (`Decompose this software task into subtasks: ...`), runs FLAN-T5 with `num_beams=4, max_new_tokens=256, temperature=0.7`
- **Structured output parsing:** `parse_subtask_line()` parses `SUBTASK_N: title | PRIORITY: p | POINTS: n | DAYS: n | TAGS: t1,t2` format with regex
- **Fallback parsing:** If structured format fails, tries line-by-line extraction
- **Confidence scoring:** Based on output quality signals (structured format, subtask count, point variety)
- **Mock mode:** Keyword-matching templates for 6 domains (auth, api, ui, database, test, deploy) + priority-based defaults
- PII hard-stop via `assert_no_pii()` at function entry

#### AI Server — Router (`routers/decomposition.py`)
- `POST /api/decompose` — validates API key, warns on missing description, calls inference, returns `DecomposeResponse`
- `GET /api/decompose/{task_id}/history` — returns past decompositions from in-memory store
- **JSON request logging** to `logs/decomposition_requests.jsonl` — logs `{timestamp, task_id, num_subtasks, confidence, model_version}` for every request

#### Next.js API Route (`app/api/ai/decompose/route.ts`)
- Authenticates user via `supabase.auth.getUser()`
- Validates project membership via `project_members` table
- Fetches only permitted task fields (no user joins)
- Builds project context: project name, active sprint name, existing tags
- Calls AI server via `callAIServer()` (sanitized)
- **Saves result to `ai_task_decompositions` table** with `status: 'pending'`
- Updates parent task `decomposition_status` to `'suggested'`
- Returns result with `decomposition_id` for frontend use

#### Client Library (`lib/ai/decomposition.ts`)

| Function | Purpose |
|----------|---------|
| `requestDecomposition(taskId)` | Calls `/api/ai/decompose`, returns typed `DecomposeResult` |
| `acceptSubtasks(decompositionId, acceptedIndices)` | Creates real task records with `parent_task_id`, `is_ai_generated=true`; updates decomposition status to `accepted`/`partially_accepted` |
| `rejectDecomposition(decompositionId)` | Sets status to `rejected`, resets parent task `decomposition_status` to `none` |
| `getDecompositionHistory(taskId)` | Fetches all `ai_task_decompositions` for a task, ordered newest first |

#### Data Flow
```
Frontend                    Next.js API Route              AI Server
   │                              │                           │
   │ POST /api/ai/decompose       │                           │
   │──────────────────────────────>│                           │
   │                              │ 1. Auth check              │
   │                              │ 2. Project access check    │
   │                              │ 3. Fetch task (safe cols)  │
   │                              │ 4. Build context           │
   │                              │ 5. sanitizeForAI()         │
   │                              │ POST /api/decompose        │
   │                              │──────────────────────────>│
   │                              │                           │ assert_no_pii()
   │                              │                           │ FLAN-T5 generate
   │                              │                           │ parse_subtask_line()
   │                              │          DecomposeResponse│
   │                              │<──────────────────────────│
   │                              │ 6. Save to ai_task_decomp │
   │                              │ 7. Update task status      │
   │           result + decomp_id │                           │
   │<─────────────────────────────│                           │
   │                              │                           │
   │ acceptSubtasks()             │                           │
   │──── creates real tasks ──────│                           │
```

---

## Phase 6: Task Decomposition UI

**Status:** Completed
**Date:** 2026-03-25

### What was done

Built the complete frontend UI for the AI task decomposition feature (M6), integrating into the existing Kanban board and task detail dialogs.

#### New Components Created

| Component | File | Purpose |
|-----------|------|---------|
| `DecomposeButton` | `components/ai/DecomposeButton.tsx` | Wand2 icon button, visible for contributor+ roles on tasks with no subtasks. States: idle/loading/done. |
| `DecompositionPanel` | `components/ai/DecompositionPanel.tsx` | Full subtask review panel with selection, inline title editing, accept/reject/re-analyze actions. |
| `SubtaskHierarchyView` | `components/ai/SubtaskHierarchyView.tsx` | Tree view of parent + child tasks with status dots, progress bar, connecting lines. |

#### DecompositionPanel Features
- Header with overall confidence badge (green/yellow/red based on score)
- Model version display
- Select All / Deselect All toggle
- Each subtask card shows:
  - Checkbox for selection
  - Inline-editable title
  - Priority badge (color-coded: urgent=red, high=orange, medium=yellow, low=gray)
  - Story points badge, estimated days, tag pills
  - Per-item confidence progress bar
  - Expandable description
- Action bar: "Accept Selected (N)" / "Reject All" / "Re-analyze"
- Animations via Tailwind `animate-in slide-in-from-top-2`
- Loading states use shadcn Skeleton pattern
- Sonner toasts for success/error/confirmation

#### Kanban Board Updates (`app/dashboard/projects/[projectId]/page.tsx`)
- **AI-generated indicator:** Tasks with `is_ai_generated=true` show a purple Sparkles icon next to title
- **Subtask progress bar:** Parent tasks show "2/5 subtasks" mini progress bar at card bottom
- Subtask count computed from `visibleTasks` filtering by `parent_task_id`

#### Task Detail Dialog Updates
- Widened to `sm:max-w-[600px]` for AI content
- Shows `SubtaskHierarchyView` when accepted subtasks exist
- Shows `DecomposeButton` when no subtasks and user is contributor+
- Shows `DecompositionPanel` inline after AI analysis completes
- **Decomposition history accordion:** "Past AI Analyses (N)" with timestamps, status, subtask count, confidence
- History loaded via `getDecompositionHistory()` on dialog open
- All state resets cleanly on dialog close

#### Integration Points
- `DecomposeButton` calls `requestDecomposition()` from `lib/ai/decomposition.ts`
- `DecompositionPanel` calls `acceptSubtasks()` / `rejectDecomposition()` from same lib
- On accept: panel closes, `refreshTasks()` called, Kanban updates
- On reject: panel closes, button reappears with "Re-analyze" label

---

## Phase 7: Task Assigner Training Notebook

**Status:** Completed
**Date:** 2026-03-25
**Location:** `AI_Workflow360/ai-server/notebooks/`

### What was done

Created a complete training script for the Task Assigner model (M10) — a Sentence-BERT + Scikit-learn skill-matching and workload scoring system. Runs on CPU in minutes.

#### Files Created
| File | Purpose |
|------|---------|
| `notebooks/train_assigner.py` | Python script version |
| `notebooks/train_assigner.ipynb` | Jupyter notebook version |

#### Architecture
```
Task text + Member skill text
        │                │
        ▼                ▼
   SBERT encode     SBERT encode
   (all-MiniLM)     (all-MiniLM)
        │                │
        └───── cosine ───┘ → semantic_similarity
                                    │
   Jaccard(task_tags, skills) ──────┤
   1 - workload/capacity ──────────┤  → 6-dim feature vector
   role_category_fit ──────────────┤
   completed_30d / max ────────────┤
   availability_threshold ─────────┘
                                    │
                                    ▼
                        GradientBoosting / RandomForest
                                    │
                                    ▼
                        CalibratedClassifierCV
                                    │
                                    ▼
                         confidence: 0.00–1.00
```

#### Training Pipeline (6 sections)

| Section | What it does |
|---------|-------------|
| 1. Setup | Installs sentence-transformers, scikit-learn, pandas, numpy, joblib |
| 2. Synthetic Data | Generates 5000 (task, members, correct_assignee) triples across 6 categories |
| 3. Feature Engineering | Computes 6-dim features: skill Jaccard, SBERT cosine, workload, role fit, performance, availability |
| 4. Model Training | GradientBoosting vs RandomForest, 5-fold CV, picks winner by F1 |
| 5. Calibration | CalibratedClassifierCV for proper probability scores |
| 6. Save & Export | Saves model.pkl + scaler.pkl + metadata.json, benchmarks inference time |

#### Feature Vector (6 dimensions)
| Feature | Computation | Weight |
|---------|-------------|--------|
| `skill_match_score` | Jaccard similarity: task tags vs member skills | 0.40 |
| `semantic_similarity` | SBERT cosine: task description vs skill list | — |
| `workload_score` | `1 - (current_sp / 40)` normalized | 0.25 |
| `role_match_score` | Category-role lookup table, 0.0–1.0 | 0.20 |
| `performance_score` | `completed_30d / max_in_team` normalized | 0.15 |
| `availability_score` | Threshold: <15 SP=1.0, <30=0.5, else 0.0 | — |

#### Synthetic Data Details
- 5000 training examples, ~28,000 (task, member) pair rows
- 6 task categories: frontend (10), backend (10), devops (8), testing (6), design (5), documentation (4)
- 12 member templates with realistic skill profiles and role-category fit tables
- Ground truth: weighted scoring (skill 0.40, workload 0.25, role 0.20, performance 0.15) + 10% noise
- Team size: 3-8 members per example

#### Expected Results
- **Target accuracy:** >85%
- **Inference time:** <50ms for a team of 10 members
- **Model size:** ~2-5 MB (pkl)
- **SBERT model:** all-MiniLM-L6-v2 (22 MB, CPU-friendly)
- **Training time:** ~5-10 minutes on CPU

#### Post-Training Deployment
1. Copy `assigner_model.pkl` and `assigner_scaler.pkl` to `ai-server/model_weights/assigner/`
2. Copy `training_metadata.json` as well
3. Restart the AI server
4. `/health` should show `"assigner": true`

---

## Phase 8: Task Assigner — Full Stack Integration

**Status:** Completed
**Date:** 2026-03-25

### What was done

Implemented the complete smart assignment pipeline from SBERT+sklearn model loader through to client library and SkillsManager component.

#### AI Server — Model Loader (`models/assigner/model.py`)
- Loads `assigner_model.pkl` (CalibratedClassifierCV) and `assigner_scaler.pkl` (StandardScaler) via joblib
- Loads Sentence-BERT `all-MiniLM-L6-v2` (22MB, auto-downloaded from HuggingFace, CPU-friendly)
- Reads `training_metadata.json` for SBERT model name and version
- Falls back to mock mode with clear logging if any pkl file is missing

#### AI Server — Inference (`models/assigner/inference.py`)
- **Real mode:** Computes 6 features per member, runs through scaler + calibrated classifier
  - skill_match (Jaccard), semantic_similarity (SBERT cosine), workload_score, role_match, performance, availability
  - `compute_role_match()` maps task tags to category keywords (frontend/backend/devops/testing/design/docs)
  - Returns top 3 candidates sorted by confidence with full scoring breakdowns
- **Mock mode:** Keyword overlap scoring with slight randomness for variety
- PII guard via `assert_no_pii()` at entry

#### AI Server — Router (`routers/assigner.py`)
- `POST /api/suggest-assignee` with JSON logging to `logs/assignment_requests.jsonl`

#### Next.js API Route (`app/api/ai/suggest-assignee/route.ts`)
- Full auth check + project membership validation
- Fetches from Supabase: task fields, project members, user_skills, active task counts/SP, completed_last_30d
- Sends UUIDs as names to AI (never real names)
- Saves top suggestion to `ai_assignment_logs` table
- Enriches response with display names AFTER AI responds

#### Client Library (`lib/ai/assigner.ts`)

| Function | Purpose |
|----------|---------|
| `suggestAssignee(taskId)` | Calls `/api/ai/suggest-assignee`, returns typed `AssignResult` |
| `confirmAssignment(logId, taskId, userId)` | Sets task.assignee_id, updates log was_accepted=true |
| `rejectSuggestion(logId, taskId)` | Updates log was_accepted=false |
| `getMemberWorkloadSummary(projectId)` | Returns workload data for all members (task counts, SP, skills) |

#### SkillsManager Component (`components/ai/SkillsManager.tsx`)
- **Edit mode:** Add skills via searchable input with 60+ software skill suggestions
- Each skill: name + level selector (Beginner/Intermediate/Expert) + remove button
- Levels stored in `user_skills` table with color-coded badges (gray/blue/green)
- **Read-only mode:** Displays skills as colored badges for viewing teammates
- Saves directly to Supabase `user_skills` table
- Suggestions dropdown filters existing skills, supports custom entries via Enter

---

## Phase 9: Task Assigner UI

**Status:** Completed
**Date:** 2026-03-25

### What was done

Built the complete frontend UI for the AI task assignment feature (M10).

#### New Components Created

| Component | File | Purpose |
|-----------|------|---------|
| `AssigneeSuggestionPanel` | `components/ai/AssigneeSuggestionPanel.tsx` | Inline panel in task detail showing top 3 suggestions with confidence bars, scoring breakdowns, and assign buttons |
| `WorkloadDashboard` | `components/ai/WorkloadDashboard.tsx` | Dialog showing all project members' capacity: SP bars, task counts, skills, color-coded status |
| `BulkAssignDialog` | `components/ai/BulkAssignDialog.tsx` | Sprint-level bulk AI assignment: runs suggestions for all unassigned tasks, table with accept/skip, bulk apply |

#### AssigneeSuggestionPanel Features
- Three states: idle (trigger button) / loading (pulsing animation) / results (suggestion cards)
- Each suggestion card: avatar initials, name, role, "Best Match" badge on #1, confidence bar (green/yellow/red), assign button
- Expandable "Why?" section showing all 6 scoring dimensions (skill match, semantic similarity, workload, role match, performance, availability)
- "Skip — Assign Manually" link to dismiss
- Optimistic UI update on assign + Sonner toast

#### WorkloadDashboard Features
- Opens as dialog from "Team Workload" button
- Grid of member cards showing: avatar, name, role, SP bar (current/capacity), active task count, completed last 30d, top 3 skills
- Color-coded status: green (Available <60%), yellow (Busy 60-80%), red (Overloaded >80%)

#### BulkAssignDialog Features
- Trigger: "AI Assign (N unassigned)" button for sprint leads
- Phase 1: Shows list of unassigned tasks, "Run AI Assignment" button
- Phase 2: Progress bar while running suggestions sequentially
- Phase 3: Results table with task | suggested assignee | confidence | Accept/Skip toggle
- "Accept All Suggestions" bulk action
- "Apply" creates all assignments in one operation

#### Kanban Board Updates
- **Ghost avatar:** Unassigned tasks show dashed circle with "+" and "Unassigned" text, clickable to open task detail
- **AI assignee sparkle:** Tasks where `ai_suggested_assignee_id === assignee_id` show purple Sparkles icon on avatar

#### Task Detail Dialog Updates
- Assignee field shows sparkle icon when AI-suggested
- `AssigneeSuggestionPanel` integrated inline when task is unassigned
- Optimistic assignee update on selection

---

## Phase 10: Bottleneck Predictor Training Notebook

**Status:** Completed
**Date:** 2026-03-25
**Location:** `AI_Workflow360/ai-server/notebooks/`

### What was done

Created a complete training script for the Sprint Bottleneck Predictor (M11) — XGBoost classifier + rule-based bottleneck detector + recommendation engine.

#### Files Created
| File | Purpose |
|------|---------|
| `notebooks/train_optimizer.py` | Python script (958 lines) |
| `notebooks/train_optimizer.ipynb` | Jupyter notebook (15 cells, 826 code lines) |

#### Architecture (3 layers)
```
Sprint features (18-dim)
        │
        ▼
  XGBoost Classifier ────→ risk_level: low/medium/high/critical
        │
        ▼
  Rule-Based Detector ────→ bottlenecks: [{type, description, severity}]
        │
        ▼
  Recommendation Engine ──→ recommendations: [{action, reason, priority}]
        │
        ▼
  SHAP Explainer ─────────→ feature importance (why this risk level?)
```

#### Training Pipeline (7 sections)

| Section | What it does |
|---------|-------------|
| 1. Setup | XGBoost, scikit-learn, SHAP, matplotlib |
| 2. Synthetic Data | 8000 sprint snapshots, 18 features, 4 balanced risk classes |
| 3. Model Training | XGBoost + GridSearchCV (27 param combos), 5-fold stratified CV |
| 4. Explainability | SHAP TreeExplainer, feature importance plots |
| 5. Bottleneck Detector | 8 rule-based checks: overload, blocked, unassigned critical, velocity lag, deadline risk, scope creep, workload imbalance, overdue |
| 6. Recommendations | Maps bottleneck types to actionable advice with priority levels |
| 7. Save & Export | model.pkl + scaler.pkl + explainer.pkl + label_encoder.pkl + metadata |

#### 18 Sprint Features

| Category | Features |
|----------|----------|
| Sprint-level | days_remaining, completion_rate, SP completed/remaining, capacity_utilization, past_velocity_avg |
| Task distribution | blocked_task_count, critical_task_count, unassigned_task_count, overdue_task_count, tasks_without_estimates |
| Member workload | max_member_workload, workload_std_dev, avg_member_utilization, overloaded_member_count |
| Velocity trend | velocity_trend, velocity_vs_plan, scope_change_ratio |

#### 8 Bottleneck Types Detected
| Type | Trigger | Severity |
|------|---------|----------|
| member_overload | Any member >80% capacity | medium–critical |
| blocked_tasks | >2 blocked tasks | medium–critical |
| unassigned_critical | Any critical task unassigned | high |
| velocity_lag | Pace <60% of planned | medium–high |
| deadline_risk | <2 days + <70% complete | critical |
| scope_creep | >120% capacity utilization | medium–high |
| workload_imbalance | Workload std dev >0.5 | medium |
| overdue_tasks | >3 overdue tasks | medium–high |

#### Expected Results
- **Target accuracy:** >82%
- **Inference time:** <10ms per sprint (XGBoost + rules)
- **SHAP explanation:** ~5-10ms additional (on-demand)
- **Training time:** ~5-10 minutes on CPU

---

## Phase 11: Bottleneck Predictor — Full Stack Integration

**Status:** Completed
**Date:** 2026-03-25

### What was done

Implemented the complete sprint bottleneck prediction pipeline from XGBoost model loader through to client library.

#### AI Server — Model Loader (`models/optimizer/model.py`)
- Loads `optimizer_model.pkl` (XGBClassifier), `optimizer_scaler.pkl` (StandardScaler) via joblib
- Optionally loads `optimizer_explainer.pkl` (SHAP TreeExplainer) for feature explanations
- Loads `optimizer_label_encoder.pkl` for risk level class decoding
- `compute_sprint_features()` computes the 18-dim feature vector from raw sprint data (dates, tasks, workloads)
- Falls back to mock mode with clear logging if pkl files missing

#### AI Server — Inference (`models/optimizer/inference.py`) — 3 layers
1. **ML prediction (real mode):** XGBoost predicts risk_level + risk_score, SHAP explains top 3 contributing features
2. **Rule-based detection (always runs):** 9 bottleneck types detected deterministically
3. **Recommendation engine:** Maps each bottleneck to actionable advice with priority

#### Bottleneck Types Detected
| Type | Trigger |
|------|---------|
| member_overload | >5 active tasks per member |
| blocked_tasks | >2 blocked tasks |
| unassigned_critical | Any high-priority task unassigned |
| velocity_lag | Pace <60% of planned |
| deadline_risk | <2 days remaining + <70% complete |
| scope_creep | >120% capacity utilization |
| workload_imbalance | Workload std dev >0.5 |
| overdue_tasks | >3 tasks past due date |
| unassigned_tasks | >3 tasks without assignee |

#### Next.js API Route (`app/api/ai/analyze-sprint/route.ts`)
- Auth check + project membership validation
- Fetches sprint details, tasks (permitted fields only), member workloads, past 3 sprint velocities
- Calls AI server, saves to `ai_bottleneck_reports` table
- Updates `sprints.ai_risk_score`, `ai_risk_factors`, `ai_analyzed_at`

#### Client Library (`lib/ai/optimizer.ts`)

| Function | Purpose |
|----------|---------|
| `analyzeSprint(sprintId)` | Single sprint analysis via API route |
| `analyzeProject(projectId)` | Batch: analyzes all active/planned sprints, returns aggregated risk |
| `getLatestReport(sprintId)` | Most recent `ai_bottleneck_reports` row |
| `getProjectRiskHistory(projectId)` | Historical risk scores for trend charts (up to 50 points) |

---

## Phase 12: Workflow Optimizer UI

**Status:** Completed
**Date:** 2026-03-25

### What was done

Built the complete AI Workflow Optimizer UI — both a compact widget for the Analytics tab and a full dedicated tab with detailed visualizations.

#### New Components Created (8 files)

| Component | File | Type | Purpose |
|-----------|------|------|---------|
| `RiskScoreGauge` | `components/ai/RiskScoreGauge.tsx` | SVG | Animated semicircle gauge: green/yellow/orange/red, center score + risk level label |
| `BottleneckList` | `components/ai/BottleneckList.tsx` | Data | Expandable bottleneck cards: severity badge, type icon, description, affected task links |
| `RecommendationList` | `components/ai/RecommendationList.tsx` | Data | Priority-sorted recommendations with "Mark Done" checkboxes (localStorage persistence) |
| `VelocityTrendChart` | `components/ai/VelocityTrendChart.tsx` | Chart | Recharts line chart: planned (dashed) vs actual velocity across sprints |
| `WorkloadHeatmap` | `components/ai/WorkloadHeatmap.tsx` | SVG | Grid heatmap: members x sprint days, colored by story point load, hover tooltips |
| `ProjectRiskHistory` | `components/ai/ProjectRiskHistory.tsx` | Chart | Recharts bar chart: historical risk scores colored by risk level |
| `SprintRiskWidget` | `components/ai/SprintRiskWidget.tsx` | Widget | Compact card: gauge, top 2 bottleneck pills, sprint selector, auto-refresh if stale |
| `AIOptimizerTab` | `components/ai/AIOptimizerTab.tsx` | Page | Full optimizer tab: gauge + bottlenecks + recommendations + velocity + heatmap + history |

#### SprintRiskWidget (Analytics Tab)
- Embeds alongside existing analytics cards
- Sprint selector if multiple active sprints
- Auto-triggers analysis if last report > 4 hours old
- Shows risk gauge (compact 140px), last analyzed timestamp, top 2 bottleneck pills
- "View Full Report" link navigates to AI Optimizer tab

#### AI Optimizer Tab (Full Page)
- New tab added to project page: "AI Optimizer" with Bot icon (purple-themed)
- Layout: 4 rows
  1. Risk Gauge card + Bottleneck List card (2-column grid)
  2. Recommendation List (full-width, with localStorage done tracking)
  3. Velocity Trend + Risk History charts (2-column grid)
  4. Workload Heatmap (full-width, member x day SVG grid)
- Sprint selector + "Re-analyze" button in header
- Empty state with "Run First Analysis" CTA
- Loading states with shadcn Skeleton pattern
- Sonner toast on analysis complete

#### RiskScoreGauge Details
- Pure SVG, no external charting library
- Animated fill from 0 to score on mount via CSS transition
- Color: green (0-25%) → yellow (25-50%) → orange (50-75%) → red (75-100%)
- Center text: percentage + risk level label

#### WorkloadHeatmap Details
- Pure SVG grid — no external library
- Rows: team members (names truncated), Columns: sprint days
- Cell color intensity: slate (0 SP) → green (1-6 SP) → amber (7-15 SP) → red (>15 SP)
- Fixed tooltip on hover showing member, date, task count, story points
- Legend bar at bottom

---

---

## Phase 13: UI/UX Redesign — "The Digital Curator"

**Status:** Completed
**Date:** 2026-03-31
**Design Source:** Google Stitch exports (22 screens, light + dark modes)

### Design System

The entire UI has been redesigned based on the "Intelligent Canvas" design philosophy from Google Stitch.

#### Design Tokens Updated
| Token Category | Old Value | New Value |
|----------------|-----------|-----------|
| Primary | `#00A6FF` (Bright Blue) | `#4F46E5` (Indigo-600) |
| Secondary | `#7F57FF` (Purple) | `#7C3AED` (Violet-600) |
| Background | `#FFFFFF` / `#0B0F3F` | `#F8F9FF` (surface) / `#0F172A` (slate-900) |
| Sidebar | Dark Navy `#0B0F3F` | Slate-50 (light, tonal hierarchy) |
| Success | `#2ECC71` | `#10B981` (Emerald) |
| Warning | `#F1C40F` | `#F59E0B` (Amber) |
| Destructive | `#E74C3C` | `#F43F5E` (Rose) |
| Border style | 1px solid borders | Ghost borders (15% opacity) + tonal shifts |

#### Files Updated
| File | Changes |
|------|---------|
| `tailwind.config.ts` | New MD3 surface tokens, ambient shadows, new animations, font families |
| `app/globals.css` | New HSL CSS variables for light/dark, glassmorphism utilities, ghost borders |
| `app/layout.tsx` | Added Inter font weights, CommandPalette, suppressHydrationWarning |
| `app/dashboard/layout.tsx` | Complete rewrite: slate-50 sidebar, glassmorphism topbar, theme toggle |

### New Components Created
| Component | Purpose |
|-----------|---------|
| `components/ui/command-palette.tsx` | Cmd+K search overlay with keyboard navigation |
| `components/ui/theme-toggle.tsx` | Light/Dark/System theme toggle |
| `components/ui/empty-state.tsx` | Reusable empty state with illustration area |
| `components/ui/skeleton.tsx` | Skeleton loaders: Card, Table, Kanban, Dashboard |
| `app/not-found.tsx` | 404 page with geometric illustration |
| `app/error.tsx` | 500 error page with retry action |

### Screens Redesigned
| Screen | Key Changes |
|--------|-------------|
| **Login** | Split layout with testimonial panel, gradient CTA, editorial labels |
| **Signup** | Split layout with feature list, password strength, violet gradient |
| **Organization Selection** | Card grid with tonal hierarchy, ghost borders |
| **Organization Dashboard** | Bento welcome banner, AI suggestion card, stat cards |
| **All 33 component files** | Bulk color migration from old brand to new Indigo/Violet palette |

### Color Migration (Bulk)
All 33 component files updated:
- `brand-blue` → `indigo-600` / `indigo-500`
- `brand-purple` → `violet-600` / `violet-500`
- `brand-cyan` → `cyan-400`
- `text-navy-900` → `text-foreground`
- `bg-[#F8F9FC]` → `bg-slate-50 dark:bg-slate-900`
- `border-[#E7E9EF]` → `border-slate-200 dark:border-slate-800`

### Design Principles Applied
1. **No-Line Rule:** Borders replaced with tonal background shifts
2. **Surface Hierarchy:** `surface-container-lowest` → `surface-container-highest` layering
3. **Glassmorphism:** Topbar uses `backdrop-blur-md` with 80% opacity
4. **Editorial Typography:** `text-[0.6875rem] font-bold uppercase tracking-wider` for labels
5. **Gradient Signatures:** Primary-to-Secondary gradient on all main CTAs
6. **Ambient Shadows:** `0px 20px 50px rgba(11, 28, 48, 0.06)` on hover/floating elements
7. **AI Visual Identity:** Violet accents for all AI-related elements

---

## Next Steps
- [ ] Run the decomposition training notebook on Colab
- [ ] Run the assigner training notebook (CPU, ~5 min)
- [ ] Run the optimizer training notebook (CPU, ~5 min)
- [ ] Integrate SkillsManager into user profile/settings page
- [ ] Integrate WorkloadDashboard and BulkAssignDialog into sprint view
- [ ] Add integration tests for AI endpoints
- [ ] Deploy and test full pipeline end-to-end
