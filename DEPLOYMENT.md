# Deploying AI Server on Render + Connecting to Vercel

This guide deploys the **FastAPI AI server** (`AI_Workflow360/ai-server/`) to
[Render](https://render.com) and connects it to the **Next.js app** already
deployed on Vercel.

---

## Architecture

```
┌────────────────────────┐         ┌────────────────────────┐
│  Next.js on Vercel     │  HTTPS  │  FastAPI on Render     │
│  (Workflow360)         │ ──────► │  (AI_Workflow360)      │
│                        │         │                        │
│  /app/api/ai/* routes  │ ◄────── │  /api/decompose        │
│  use AI_SERVER_URL     │         │  /api/suggest-assignee │
│  and AI_API_KEY        │         │  /api/analyze-sprint   │
└────────────────────────┘         └────────────────────────┘
```

---

## What ships to Render

| Model | Size | Ships to Render? | Why |
|-------|------|-----------------|------|
| **Optimizer** (XGBoost + SHAP) | ~250 KB | Yes — included in git | Tiny, well within free tier RAM |
| **Assigner** (SBERT + sklearn) | ~15 MB | Yes — included in git | SBERT downloads from HuggingFace on first startup |
| **Decomposition** (FLAN-T5) | ~300 MB | No — runs in mock mode | Exceeds Render free tier 512 MB RAM |

**Result on free tier**: `/health` returns `{"optimizer": true, "assigner": true, "decomposition": false}`.
The decomposition UI still works via mock responses (realistic fake subtasks).

To enable all 3 models, upgrade to Render Starter ($7/mo, 2 GB RAM) — see
"Upgrade path" at the bottom.

---

## Part 1 — Prepare the repo

### 1a. Verify the trained models exist locally

```bash
cd AI_Workflow360/ai-server
ls model_weights/
# Expected (for free tier deploy): optimizer/
# May also have: assigner/ decomposition/
```

The **optimizer** folder must contain:
- `optimizer_model.pkl`
- `optimizer_scaler.pkl`
- `optimizer_explainer.pkl`
- `optimizer_label_encoder.pkl`
- `training_metadata.json`

If these are missing, run `notebooks/train_optimizer.ipynb` first (CPU, ~5 min).

### 1b. Commit the deployment files and small models

The following files need to be on GitHub for Render to deploy:

```bash
cd AI_Workflow360

# Deploy config
git add ai-server/render.yaml
git add ai-server/runtime.txt
git add ai-server/main.py
git add ai-server/.gitignore

# Trained models (optimizer is small enough to ship)
git add -f ai-server/model_weights/optimizer/optimizer_model.pkl
git add -f ai-server/model_weights/optimizer/optimizer_scaler.pkl
git add -f ai-server/model_weights/optimizer/optimizer_explainer.pkl
git add -f ai-server/model_weights/optimizer/optimizer_label_encoder.pkl
git add -f ai-server/model_weights/optimizer/training_metadata.json

# Deployment guide
git add DEPLOYMENT.md

git commit -m "deploy: Render config + ship trained optimizer model"
git push
```

> The `-f` flag is needed because `.gitignore` blocks `.pkl` by default in some
> configs. After commit, verify with `git ls-files ai-server/model_weights/`.

---

## Part 2 — Deploy to Render

### 2a. Create the web service

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Connect the GitHub repo containing `AI_Workflow360/`
4. Render auto-detects `render.yaml` → click **Apply**

If manual setup is needed:

| Field | Value |
|-------|-------|
| Name | `workflow360-ai-server` |
| Region | `Oregon` (or nearest) |
| Branch | `main` (or your deploy branch) |
| Root Directory | `ai-server` |
| Runtime | `Python 3` |
| Build Command | `pip install --no-cache-dir -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/` |
| Instance Type | `Free` |

### 2b. Set environment variables

In Render dashboard → your service → **Environment** tab:

| Key | Value | Notes |
|-----|-------|-------|
| `AI_API_KEY` | *(auto-generate or set manually)* | Must match Vercel — copy this value |
| `NEXT_APP_URL` | `https://your-app.vercel.app` | Your Vercel production URL |
| `MODEL_DIR` | `./model_weights` | Where models are loaded from |
| `DEVICE` | `cpu` | Render free tier has no GPU |
| `PYTHON_VERSION` | `3.11.9` | Pin Python version |

**Multiple origins**: `NEXT_APP_URL` accepts comma-separated values:
`https://your-app.vercel.app,https://custom-domain.com`.
Vercel preview URLs (`*.vercel.app`) are auto-allowed via regex — no config needed.

### 2c. Deploy & verify

Click **Create Web Service**. The first build takes ~5 minutes (installing
PyTorch, Transformers, SBERT, XGBoost).

Once deployed, visit:
```
https://workflow360-ai-server.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "models": {
    "decomposition": false,
    "assigner": true,
    "optimizer": true
  }
}
```

- `assigner: true` — SBERT downloaded from HuggingFace, sklearn model loaded
- `optimizer: true` — XGBoost model loaded from shipped .pkl
- `decomposition: false` — intentional, skipped on free tier (mock mode)

---

## Part 3 — Connect Vercel to Render

### 3a. Add environment variables on Vercel

Vercel dashboard → your Workflow360 project → **Settings** → **Environment Variables**

Add for **Production**, **Preview**, and **Development**:

| Key | Value |
|-----|-------|
| `AI_SERVER_URL` | `https://workflow360-ai-server.onrender.com` |
| `AI_API_KEY` | *(same value as on Render)* |

### 3b. Redeploy Vercel

Trigger a redeploy to pick up the new env vars:
- Git push any commit, OR
- Vercel dashboard → **Deployments** → kebab menu → **Redeploy**

### 3c. End-to-end test

Open your Vercel app → sign in → open any task.

| Feature | Test | Expected |
|---------|------|----------|
| AI Decompose | Click "AI Decompose" on a task | Returns 3–5 subtasks (mock on free tier) |
| AI Assign | Open an unassigned task | Returns top 3 member suggestions with scores |
| AI Sprint Analysis | Analytics tab → "Analyze Now" | Returns risk gauge + bottlenecks + recommendations |

---

## Part 4 — Free tier considerations

### Cold starts

Render free tier spins down after **15 min of inactivity**. The first request
after idle takes **30–60 seconds** to wake up. Subsequent requests are fast.

**Mitigations:**
- **Use UptimeRobot** (free): set up a monitor hitting
  `https://<service>.onrender.com/` every 5 minutes to keep it warm
- **Show a loader** in the UI — "Warming up AI..." for the first call after
  long idle
- **Upgrade to Starter** ($7/mo) — no spin-down

### Memory limit — 512 MB

Current usage on free tier:
- Python + FastAPI + middleware: ~100 MB
- SBERT model (assigner): ~90 MB
- XGBoost + SHAP (optimizer): ~40 MB
- **Total: ~230 MB** — fits comfortably with headroom

**Do not enable FLAN-T5 on free tier** — it alone needs ~500 MB RAM to load.

### Disk limit — 10 GB ephemeral

Plenty of room for the small shipped models. Models persist across restarts
since they're baked into the container image at build time.

---

## Part 5 — Troubleshooting

### CORS error in browser console

Your Vercel URL isn't in `NEXT_APP_URL` and isn't `*.vercel.app`.
→ Add it to Render's env vars, redeploy.

### 401 "Invalid API key"

`AI_API_KEY` mismatch.
→ Copy the exact value from Render → paste into Vercel → redeploy Vercel.

### Build fails: `ModuleNotFoundError`

Python version mismatch.
→ Ensure `PYTHON_VERSION=3.11.9` on Render and `runtime.txt` contains
`python-3.11.9`.

### `/health` shows all models `false`

Optimizer model files didn't ship.
→ `git ls-files ai-server/model_weights/optimizer/` should list the .pkl files.
If empty, re-run Part 1b with the `-f` flag.

### First request times out

Cold start is longer than browser fetch timeout (30s default).
→ On first API call, show a user-friendly "Warming up..." loader. Subsequent
calls are fast.

### Render build times out during SBERT download

SBERT model (`all-MiniLM-L6-v2`) is downloaded at first startup.
→ The first deploy may be slow; subsequent deploys cache it. If it persistently
fails, pre-download it during build — edit `render.yaml` buildCommand:
```yaml
buildCommand: pip install --no-cache-dir -r requirements.txt && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

---

## Part 6 — Upgrade path (all 3 models)

To run the FLAN-T5 decomposition model in production:

1. **Upgrade Render** → Starter plan ($7/mo, 2 GB RAM)
2. **Train FLAN-T5** — run `notebooks/train_decomposition.ipynb` on Colab (~60 min)
3. **Upload the model** — 300 MB is too big for git. Two options:

   **Option A — HuggingFace Hub (recommended):**
   ```python
   # In model_weights/decomposition/upload.py
   from huggingface_hub import upload_folder
   upload_folder(
       folder_path="./model_weights/decomposition",
       repo_id="your-username/flan-t5-pm-decomposition",
   )
   ```
   Then update `models/decomposition/model.py` to download from Hub on startup.

   **Option B — Git LFS:**
   ```bash
   git lfs install
   git lfs track "model_weights/decomposition/*"
   git add .gitattributes model_weights/decomposition/
   git commit -m "ship FLAN-T5 via Git LFS"
   git push
   ```

4. **Restart Render** → `/health` now shows `decomposition: true`

---

## Summary

| Component | Deployed to | URL |
|-----------|-------------|-----|
| Next.js frontend | Vercel | `https://<project>.vercel.app` |
| FastAPI AI server | Render | `https://<service>.onrender.com` |
| Database | Supabase | `https://<project>.supabase.co` |

**Quick deploy check:**
```bash
curl https://<your-render-url>.onrender.com/health
```

That's it — your AI server is now live and the full Workflow360 AI pipeline
is operational end-to-end.
