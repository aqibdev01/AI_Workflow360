# Deploying AI Server for FREE on HuggingFace Spaces

**Goal:** Deploy all 3 AI modules (including the 300 MB FLAN-T5 decomposition model)
for **$0** — perfect for an FYP evaluation demo.

**Why HuggingFace Spaces over Render?**

| Feature | Render Free | HF Spaces Free |
|---------|-------------|---------------|
| RAM | 512 MB (too small for FLAN-T5) | **16 GB** |
| CPU | Shared, throttled | 2 vCPUs |
| Spin-down | Yes, 15 min idle | **No — always warm** |
| Storage | 10 GB ephemeral | **50 GB persistent with Git LFS** |
| Cold start | 30-60s | **None** |
| FLAN-T5 support | ❌ Crashes (OOM) | **✅ Runs comfortably** |
| Cost | $0 (limited) | **$0 (generous)** |

---

## Architecture

```
┌────────────────────────┐         ┌──────────────────────────┐
│  Next.js on Vercel     │  HTTPS  │  FastAPI on HF Spaces    │
│  (Workflow360)         │ ──────► │  (Docker, 16 GB RAM)     │
│                        │         │                          │
│  /app/api/ai/* routes  │ ◄────── │  All 3 models loaded     │
│  use AI_SERVER_URL     │         │  https://<user>-<space>  │
│  + AI_API_KEY          │         │    .hf.space             │
└────────────────────────┘         └──────────────────────────┘
           │                                    ▲
           │                                    │
           ▼                                    │ (Auto-pull on startup)
  ┌────────────────────┐          ┌────────────────────────┐
  │  Supabase          │          │  HuggingFace Model Hub │
  │  (database)        │          │  flan-t5-pm-v1         │
  └────────────────────┘          │  (stores FLAN-T5 free) │
                                  └────────────────────────┘
```

---

## Step 1 — Create a HuggingFace account

1. Go to [huggingface.co/join](https://huggingface.co/join)
2. Sign up (free, no credit card required)
3. Get an access token: **Settings** → **Access Tokens** → **New token** (select **Write** permission)
4. Save the token somewhere — you'll need it twice

---

## Step 2 — Upload your trained FLAN-T5 model to HuggingFace Hub

The FLAN-T5 model is too big for git (300 MB), so we store it in a **HuggingFace Model Repo** (free, unlimited, purpose-built for ML models).

### 2a. Create the model repo

Go to [huggingface.co/new](https://huggingface.co/new):
- **Owner:** your username
- **Model name:** `flan-t5-pm-decomposition`
- **License:** MIT
- **Visibility:** Public (or Private if you have HF Pro)
- Click **Create model**

### 2b. Upload the trained model

From your local machine (where `model_weights/decomposition/` exists):

```bash
pip install huggingface_hub

# Log in (paste your token when prompted)
huggingface-cli login

# Upload the entire decomposition folder (excludes checkpoints)
huggingface-cli upload <your-username>/flan-t5-pm-decomposition \
  AI_Workflow360/ai-server/model_weights/decomposition \
  . \
  --exclude "checkpoint-*"
```

Or via Python:
```python
from huggingface_hub import upload_folder

api = HfApi()

# Upload large folder
api.upload_large_folder(
    folder_path="AI_Workflow360/ai-server/model_weights/decomposition",
    repo_id="<your-username>/flan-t5-pm-decomposition",
    repo_type="model",
    ignore_patterns=["checkpoint-*"],
)

print("Upload complete!")
```

Upload takes ~2 minutes. When done, browse to
`huggingface.co/<your-username>/flan-t5-pm-decomposition` to verify.

---

## Step 3 — Create the HuggingFace Space

### 3a. Create the space

Go to [huggingface.co/new-space](https://huggingface.co/new-space):
- **Space name:** `workflow360-ai-server`
- **License:** MIT
- **SDK:** **Docker** (important!)
- **Hardware:** **CPU basic (free)** — 16 GB RAM, 2 vCPUs
- **Visibility:** Public
- Click **Create Space**

HuggingFace creates an empty git repo at:
`https://huggingface.co/spaces/<your-username>/workflow360-ai-server`

### 3b. Push your ai-server code to the Space

```bash
# Clone the empty Space
git clone https://huggingface.co/spaces/<your-username>/workflow360-ai-server
cd workflow360-ai-server

# Copy AI server files (from your FYP repo)
cp -r /path/to/AI_Workflow360/ai-server/* .
cp -r /path/to/AI_Workflow360/ai-server/.gitignore .

# Git add / commit / push
git add .
git commit -m "Initial deploy"
git push
```

HF Spaces automatically detects the `Dockerfile`, builds the image, and starts
the container. Build time: ~5 minutes.

### 3c. Watch the build

Go to the Space → **Logs** tab. You'll see:
- Docker image building (installing pytorch, transformers, etc.)
- Container starting
- FastAPI server starting

Once live, the Space shows **Running** status.

### 3d. Test the deployment

Visit: `https://<your-username>-workflow360-ai-server.hf.space/health`

Expected:
```json
{
  "status": "ok",
  "models": {
    "decomposition": true,
    "assigner": true,
    "optimizer": true
  }
}
```

🎉 All 3 models loaded. No more mock mode.

---

## Step 4 — Set environment variables on the Space

HF Space → **Settings** → **Variables and secrets**

Add these as **Secrets** (not variables — they're encrypted):

| Key | Value | Notes |
|-----|-------|-------|
| `AI_API_KEY` | *(any random long string)* | Shared with Vercel — must match |
| `NEXT_APP_URL` | `https://your-app.vercel.app` | Your Vercel production URL |
| `HF_DECOMPOSITION_MODEL` | `<your-username>/flan-t5-pm-decomposition` | Model repo from Step 2 |

Click **Restart Space** after adding secrets.

---

## Step 5 — Connect Vercel to HuggingFace Spaces

Vercel dashboard → Workflow360 project → **Settings** → **Environment Variables**

Add for **Production**, **Preview**, **Development**:

| Key | Value |
|-----|-------|
| `AI_SERVER_URL` | `https://<your-username>-workflow360-ai-server.hf.space` |
| `AI_API_KEY` | *(same value as on HF Spaces)* |

Trigger a Vercel redeploy:
- Push any commit, OR
- Vercel → Deployments → **Redeploy**

---

## Step 6 — Demo end-to-end test

Open your Vercel app → sign in → create a project → create a task.

| Feature | Test | Expected |
|---------|------|----------|
| **AI Decompose** | Open a task → "AI Decompose" button | Real FLAN-T5 generates 3–5 relevant subtasks |
| **AI Assign** | Open unassigned task → "Suggest Assignee" | Top 3 members with skill-match scores |
| **AI Sprint Analysis** | Analytics tab → "Analyze Now" | Risk gauge, bottleneck list, recommendations |

All three use **real trained models**, not mock mode.

---

## Troubleshooting

### Build fails: "out of memory" during pip install

HF free tier has 16 GB RAM during build too, but `torch` can spike.
→ Remove `--no-cache-dir` from `Dockerfile` build (uses more disk but less RAM)

### Space runtime error: "Model not found on HF Hub"

`HF_DECOMPOSITION_MODEL` is wrong or the model repo is private.
→ Verify the model exists at `huggingface.co/<repo>` and is public. If private,
add `HF_TOKEN=<your-write-token>` as another secret on the Space.

### CORS error in browser console

Your Vercel URL isn't whitelisted.
→ Ensure `NEXT_APP_URL` is set on the Space and contains your Vercel URL.
→ Vercel preview URLs (`*.vercel.app`) are auto-allowed via regex.

### "Connection refused" from Vercel

The Space is sleeping.
→ HF Spaces free tier stays warm, but after **48 hours with zero traffic** it pauses.
  Just visit the Space URL once to wake it.

### Space shows "Error" status

→ Check **Logs** tab for the stack trace. Usually a missing dependency or
  env var. Restart the Space after fixing.

---

## Scaling & costs (if you want to go beyond demo)

HF Spaces pricing (only if you outgrow free tier):

| Tier | CPU | RAM | Cost |
|------|-----|-----|------|
| CPU basic | 2 vCPU | 16 GB | **$0** |
| CPU upgrade | 8 vCPU | 32 GB | $0.03/hr |
| Small GPU (T4) | 4 vCPU | 15 GB + 16 GB VRAM | $0.60/hr |

For an FYP demo, **free tier is more than enough** — 16 GB RAM is huge.

---

## Summary

**Your entire stack, deployed for free:**

| Component | Hosted on | Cost |
|-----------|-----------|------|
| Next.js frontend | Vercel (Hobby) | $0 |
| FastAPI AI server | HuggingFace Spaces | $0 |
| Database + Auth | Supabase (Free) | $0 |
| FLAN-T5 model weights | HuggingFace Model Hub | $0 |
| **Total** | | **$0/month** |

**One-line deploy check:**
```bash
curl https://<user>-workflow360-ai-server.hf.space/health
```

Ready for your evaluation committee.
