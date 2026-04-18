#!/usr/bin/env python3
"""
=============================================================================
FLAN-T5-Small Fine-Tuning for Task Decomposition (M6)
=============================================================================
Workflow360 FYP — AI Module Training Notebook

This script fine-tunes google/flan-t5-small on a software project management
task decomposition dataset. Each training example teaches the model to break
a parent task into structured subtasks.

Designed to run end-to-end on Google Colab free tier (T4 GPU, ~12 GB RAM).
Expected training time: 45-90 minutes.

To use as a Colab notebook:
  1. Upload this file to Colab
  2. Or copy-paste sections into cells (delimited by # SECTION comments)
  3. Sections marked # MARKDOWN: are explanatory text cells

Usage:
  python train_decomposition.py          # local run
  # Or run cell-by-cell in Google Colab
=============================================================================
"""

# ============================================================================
# MARKDOWN: # FLAN-T5 Fine-Tuning for Task Decomposition
#
# This notebook fine-tunes **FLAN-T5-small** (77M params) to decompose
# software project management tasks into structured subtasks.
#
# **Model:** google/flan-t5-small
# **Dataset:** 3000-5000 synthetic PM task decomposition examples
# **Target metric:** ROUGE-L > 0.35
# **Hardware:** Google Colab T4 GPU (free tier)
# **Training time:** ~45-90 minutes
# ============================================================================


# ============================================================================
# SECTION 1: Setup & Dependencies
# ============================================================================
# MARKDOWN: ## 1. Setup & Dependencies
# Install required packages. These are pre-installed on Colab except for
# `rouge_score` and `accelerate`. Running `pip install` is idempotent.

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install("transformers>=4.41.0")
install("datasets>=2.19.0")
install("torch>=2.3.0")
install("accelerate>=0.30.0")
install("evaluate>=0.4.0")
install("rouge_score>=0.1.2")
install("sentencepiece>=0.2.0")

import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================================
# SECTION 2: Dataset Loading
# ============================================================================
# MARKDOWN: ## 2. Dataset Loading
#
# We use a two-part approach:
# 1. **Structural patterns** from a public HuggingFace dataset to teach the
#    model structured text generation
# 2. **Synthetic PM data** (Section 3) that directly targets our decomposition
#    format
#
# The synthetic data is the primary training signal. The structural patterns
# dataset helps the model learn to produce well-formatted multi-line output.

print("\n" + "="*60)
print("SECTION 2: Loading structural pattern data")
print("="*60)

from datasets import load_dataset

# Load a small slice of structured text generation data for format learning.
# We only need ~500 examples from this — the synthetic data is the main dataset.
try:
    raw_infill = load_dataset("sander-wood/text-infilling", split="train", streaming=True)
    structural_examples = []
    for i, example in enumerate(raw_infill):
        if i >= 500:
            break
        # Adapt to our format: teach the model to produce structured multi-line output
        text = example.get("text", "") or ""
        if len(text) > 50 and len(text) < 1000:
            # Create a decomposition-style example from structured text
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if len(lines) >= 3:
                input_text = (
                    "Decompose this software task into subtasks:\n"
                    f"Task: {lines[0][:100]}\n"
                    f"Description: Implement the following components\n"
                    f"Priority: medium\n"
                )
                output_lines = []
                for j, line in enumerate(lines[1:5], 1):
                    output_lines.append(
                        f"SUBTASK_{j}: {line[:80]} | PRIORITY: medium | "
                        f"POINTS: {random.choice([1,2,3,5])} | "
                        f"DAYS: {random.choice([0.5, 1.0, 1.5, 2.0])} | "
                        f"TAGS: general"
                    )
                if output_lines:
                    structural_examples.append({
                        "input_text": input_text,
                        "target_text": "\n".join(output_lines),
                    })
    print(f"Loaded {len(structural_examples)} structural pattern examples")
except Exception as e:
    print(f"Could not load structural dataset ({e}) — using synthetic data only")
    structural_examples = []


# ============================================================================
# SECTION 3: Synthetic Data Generation
# ============================================================================
# MARKDOWN: ## 3. Synthetic Data Generation
#
# We generate 2000+ training examples covering 6 categories of software
# project management tasks. Each example is an (input, output) pair where:
#
# - **Input:** Task title + description + priority + context
# - **Output:** Structured subtask list in a parseable format
#
# The format uses `SUBTASK_N: title | PRIORITY: p | POINTS: n | DAYS: n | TAGS: t1,t2`
# so the AI server can reliably parse model output back into structured data.

print("\n" + "="*60)
print("SECTION 3: Generating synthetic PM decomposition data")
print("="*60)

# --- Task templates per category ---

FRONTEND_TASKS = [
    {
        "title": "Build user dashboard page",
        "description": "Create a responsive dashboard showing project statistics, recent tasks, and team activity. Include charts for sprint burndown and velocity.",
        "priority": "high",
        "context": "React/Next.js project with Tailwind CSS and Recharts",
        "subtasks": [
            ("Design dashboard layout and component hierarchy", "high", 3, 1.0, "frontend,design"),
            ("Build statistics cards component", "medium", 2, 0.5, "frontend,ui"),
            ("Implement sprint burndown chart", "high", 5, 2.0, "frontend,charts"),
            ("Add velocity trend chart", "medium", 3, 1.5, "frontend,charts"),
            ("Create recent tasks feed widget", "medium", 3, 1.0, "frontend,ui"),
            ("Add responsive breakpoints and mobile layout", "low", 2, 0.5, "frontend,responsive"),
        ],
    },
    {
        "title": "Implement task card drag-and-drop",
        "description": "Add drag-and-drop functionality to the Kanban board. Tasks should be draggable between status columns with smooth animations.",
        "priority": "high",
        "context": "Using DnD Kit library in React",
        "subtasks": [
            ("Set up DnD Kit context and providers", "high", 2, 0.5, "frontend,dnd"),
            ("Make task cards draggable with drag handles", "high", 3, 1.0, "frontend,dnd"),
            ("Implement droppable status columns", "high", 3, 1.0, "frontend,dnd"),
            ("Add drag overlay with card preview", "medium", 2, 0.5, "frontend,ui"),
            ("Handle state update on drop", "high", 3, 1.0, "frontend,state"),
            ("Add smooth transition animations", "low", 2, 0.5, "frontend,animation"),
        ],
    },
    {
        "title": "Create notification center component",
        "description": "Build a dropdown notification center showing unread notifications, with real-time updates and mark-as-read functionality.",
        "priority": "medium",
        "context": "Next.js app with Supabase realtime subscriptions",
        "subtasks": [
            ("Design notification dropdown UI", "medium", 2, 0.5, "frontend,design"),
            ("Build notification list with virtual scrolling", "medium", 3, 1.5, "frontend,ui"),
            ("Add real-time subscription for new notifications", "high", 3, 1.0, "frontend,realtime"),
            ("Implement mark-as-read and bulk actions", "medium", 2, 1.0, "frontend,ui"),
            ("Add notification badge counter", "low", 1, 0.5, "frontend,ui"),
        ],
    },
    {
        "title": "Build settings page with form validation",
        "description": "Create a multi-tab settings page for user profile, notification preferences, and organization settings with proper form validation.",
        "priority": "medium",
        "context": "React Hook Form + Zod validation",
        "subtasks": [
            ("Create settings page layout with tabs", "medium", 2, 0.5, "frontend,ui"),
            ("Build profile settings form with avatar upload", "medium", 3, 1.5, "frontend,forms"),
            ("Add notification preferences toggle form", "low", 2, 1.0, "frontend,forms"),
            ("Implement organization settings panel", "medium", 3, 1.5, "frontend,forms"),
            ("Add Zod validation schemas for all forms", "medium", 2, 1.0, "frontend,validation"),
            ("Connect forms to API endpoints", "medium", 2, 1.0, "frontend,api"),
        ],
    },
    {
        "title": "Implement search and filter for task list",
        "description": "Add a search bar with auto-complete and advanced filters (status, priority, assignee, date range) for the task list view.",
        "priority": "medium",
        "context": "Task list uses server-side pagination with Supabase",
        "subtasks": [
            ("Build search input with debounced query", "medium", 2, 0.5, "frontend,search"),
            ("Add filter dropdown for status and priority", "medium", 3, 1.0, "frontend,ui"),
            ("Implement assignee filter with member selector", "medium", 3, 1.0, "frontend,ui"),
            ("Add date range picker for due date filter", "medium", 2, 1.0, "frontend,ui"),
            ("Connect filters to Supabase query params", "high", 3, 1.0, "frontend,api"),
            ("Add clear filters and save filter presets", "low", 2, 1.0, "frontend,ui"),
        ],
    },
    {
        "title": "Create onboarding flow for new users",
        "description": "Build a multi-step onboarding wizard that guides new users through profile setup, creating or joining an organization, and setting up their first project.",
        "priority": "high",
        "context": "Next.js app with Supabase Auth",
        "subtasks": [
            ("Design stepper UI for onboarding wizard", "medium", 2, 0.5, "frontend,design"),
            ("Build profile setup step with avatar and bio", "medium", 3, 1.0, "frontend,forms"),
            ("Create organization join/create step", "high", 5, 2.0, "frontend,auth"),
            ("Add first project creation step", "medium", 3, 1.5, "frontend,forms"),
            ("Implement progress persistence and skip logic", "medium", 2, 1.0, "frontend,state"),
            ("Add completion celebration screen", "low", 1, 0.5, "frontend,ui"),
        ],
    },
    {
        "title": "Build analytics chart components",
        "description": "Create reusable chart components for project analytics including burn-down, velocity, cumulative flow, and member contribution charts.",
        "priority": "medium",
        "context": "Using Recharts library",
        "subtasks": [
            ("Create base chart wrapper with common styling", "medium", 2, 0.5, "frontend,charts"),
            ("Build burn-down chart with ideal line overlay", "high", 3, 1.5, "frontend,charts"),
            ("Implement velocity bar chart by sprint", "medium", 3, 1.0, "frontend,charts"),
            ("Add cumulative flow diagram", "medium", 5, 2.0, "frontend,charts"),
            ("Create member contribution pie chart", "low", 2, 1.0, "frontend,charts"),
        ],
    },
    {
        "title": "Implement dark mode toggle",
        "description": "Add system-wide dark mode support with a toggle in the header. Persist user preference and respect system setting on first visit.",
        "priority": "low",
        "context": "Tailwind CSS with class-based dark mode",
        "subtasks": [
            ("Set up Tailwind dark mode config and CSS variables", "medium", 2, 0.5, "frontend,styling"),
            ("Build theme toggle button in header", "low", 1, 0.5, "frontend,ui"),
            ("Add theme persistence to localStorage", "low", 1, 0.5, "frontend,state"),
            ("Update all component styles for dark variants", "medium", 5, 2.0, "frontend,styling"),
            ("Detect system preference on first visit", "low", 1, 0.5, "frontend,ui"),
        ],
    },
]

BACKEND_TASKS = [
    {
        "title": "Build REST API for task CRUD",
        "description": "Create API endpoints for creating, reading, updating, and deleting tasks. Include proper validation, error handling, and authorization checks.",
        "priority": "high",
        "context": "Next.js API routes with Supabase, Row Level Security enabled",
        "subtasks": [
            ("Design API route structure and request schemas", "high", 2, 0.5, "backend,api"),
            ("Implement POST /api/tasks for task creation", "high", 3, 1.0, "backend,api"),
            ("Implement GET /api/tasks with pagination and filters", "high", 3, 1.0, "backend,api"),
            ("Implement PATCH /api/tasks/[id] for updates", "medium", 2, 1.0, "backend,api"),
            ("Implement DELETE /api/tasks/[id] with soft delete", "medium", 2, 0.5, "backend,api"),
            ("Add input validation with Zod schemas", "medium", 2, 1.0, "backend,validation"),
            ("Write error handling middleware", "medium", 2, 0.5, "backend,api"),
        ],
    },
    {
        "title": "Implement user authentication flow",
        "description": "Set up email/password authentication with signup, login, password reset, and email verification using Supabase Auth.",
        "priority": "high",
        "context": "Supabase Auth with Next.js middleware",
        "subtasks": [
            ("Configure Supabase Auth providers and settings", "high", 2, 0.5, "backend,auth"),
            ("Implement signup API with email verification", "high", 3, 1.0, "backend,auth"),
            ("Build login endpoint with session handling", "high", 3, 1.0, "backend,auth"),
            ("Add password reset flow", "medium", 3, 1.0, "backend,auth"),
            ("Create auth middleware for protected routes", "high", 3, 1.0, "backend,auth,middleware"),
            ("Add rate limiting on auth endpoints", "medium", 2, 1.0, "backend,security"),
        ],
    },
    {
        "title": "Create sprint management API",
        "description": "Build endpoints for creating sprints, adding tasks to sprints, starting/completing sprints, and calculating sprint metrics.",
        "priority": "high",
        "context": "Supabase with sprint status workflow enforcement",
        "subtasks": [
            ("Implement POST /api/sprints for creation", "high", 2, 1.0, "backend,api"),
            ("Build sprint task assignment endpoint", "high", 3, 1.0, "backend,api"),
            ("Add sprint start/complete state transitions", "high", 3, 1.5, "backend,api,workflow"),
            ("Calculate velocity and burndown metrics", "medium", 5, 2.0, "backend,analytics"),
            ("Implement sprint history and comparison", "low", 3, 1.5, "backend,api"),
        ],
    },
    {
        "title": "Set up database migration system",
        "description": "Establish a reliable migration workflow for the Supabase PostgreSQL database. Include seeding, rollback support, and CI integration.",
        "priority": "high",
        "context": "Supabase CLI with PostgreSQL",
        "subtasks": [
            ("Set up Supabase CLI and local dev environment", "high", 2, 0.5, "backend,devops"),
            ("Write initial schema migration with all tables", "high", 5, 2.0, "backend,database"),
            ("Create seed data script for development", "medium", 3, 1.0, "backend,database"),
            ("Document migration workflow for team", "low", 1, 0.5, "backend,docs"),
            ("Add migration check to CI pipeline", "medium", 2, 1.0, "backend,devops"),
        ],
    },
    {
        "title": "Implement real-time notifications backend",
        "description": "Build the server-side notification system that creates, stores, and broadcasts notifications via Supabase Realtime when events occur.",
        "priority": "medium",
        "context": "Supabase Realtime with PostgreSQL triggers",
        "subtasks": [
            ("Design notification event types and schema", "medium", 2, 0.5, "backend,design"),
            ("Create notification creation utility functions", "medium", 3, 1.0, "backend,api"),
            ("Add database triggers for task assignment events", "high", 3, 1.5, "backend,database"),
            ("Add triggers for sprint deadline events", "medium", 2, 1.0, "backend,database"),
            ("Set up Supabase Realtime channel for notifications", "medium", 3, 1.0, "backend,realtime"),
            ("Implement mark-as-read and bulk dismiss API", "low", 2, 1.0, "backend,api"),
        ],
    },
    {
        "title": "Build file upload and storage API",
        "description": "Create endpoints for uploading files to projects and tasks, with support for multiple file types, size limits, and version tracking.",
        "priority": "medium",
        "context": "Supabase Storage with signed URLs",
        "subtasks": [
            ("Configure Supabase Storage bucket and policies", "high", 2, 0.5, "backend,storage"),
            ("Build file upload endpoint with multipart support", "high", 3, 1.5, "backend,api"),
            ("Add file type validation and size limit checks", "medium", 2, 0.5, "backend,validation"),
            ("Implement file version tracking", "medium", 3, 1.5, "backend,storage"),
            ("Create file sharing with access control", "medium", 3, 1.0, "backend,api,security"),
            ("Add signed URL generation for downloads", "medium", 2, 0.5, "backend,api"),
        ],
    },
]

DEVOPS_TASKS = [
    {
        "title": "Set up CI/CD pipeline",
        "description": "Configure a complete CI/CD pipeline with linting, testing, building, and deployment stages. Deploy to Vercel for the Next.js frontend.",
        "priority": "high",
        "context": "GitHub Actions with Vercel deployment",
        "subtasks": [
            ("Create GitHub Actions workflow file", "high", 2, 0.5, "devops,ci"),
            ("Add lint and type-check stage", "medium", 2, 0.5, "devops,ci"),
            ("Configure test stage with coverage reporting", "medium", 3, 1.0, "devops,ci,testing"),
            ("Set up build stage with environment variables", "high", 2, 0.5, "devops,ci"),
            ("Configure Vercel deployment integration", "high", 3, 1.0, "devops,deployment"),
            ("Add Slack notifications for build status", "low", 1, 0.5, "devops,ci"),
        ],
    },
    {
        "title": "Containerize the AI server",
        "description": "Create a Docker setup for the FastAPI AI server with multi-stage builds, GPU support, and health checks.",
        "priority": "medium",
        "context": "Python FastAPI server with PyTorch models",
        "subtasks": [
            ("Write multi-stage Dockerfile for AI server", "high", 3, 1.0, "devops,docker"),
            ("Create docker-compose.yml with service config", "medium", 2, 0.5, "devops,docker"),
            ("Add GPU passthrough support for CUDA inference", "medium", 3, 1.5, "devops,docker,gpu"),
            ("Configure health check and restart policy", "medium", 2, 0.5, "devops,docker"),
            ("Set up volume mounts for model weights", "low", 1, 0.5, "devops,docker"),
            ("Add build and push to container registry", "medium", 2, 1.0, "devops,ci"),
        ],
    },
    {
        "title": "Configure monitoring and logging",
        "description": "Set up application monitoring, structured logging, and alerting for both the Next.js app and AI server.",
        "priority": "medium",
        "context": "Vercel Analytics + custom logging",
        "subtasks": [
            ("Set up structured JSON logging format", "medium", 2, 0.5, "devops,logging"),
            ("Configure log levels and rotation", "low", 1, 0.5, "devops,logging"),
            ("Add request/response logging middleware", "medium", 2, 1.0, "devops,logging"),
            ("Set up error tracking and alerting", "high", 3, 1.0, "devops,monitoring"),
            ("Create health check dashboard", "low", 2, 1.0, "devops,monitoring"),
        ],
    },
    {
        "title": "Set up staging environment",
        "description": "Create a staging deployment that mirrors production for testing before releases. Include separate database and environment configuration.",
        "priority": "medium",
        "context": "Vercel preview deployments with Supabase branching",
        "subtasks": [
            ("Create staging Supabase project", "high", 2, 0.5, "devops,infrastructure"),
            ("Configure staging environment variables", "medium", 2, 0.5, "devops,config"),
            ("Set up Vercel preview deployment for staging branch", "medium", 2, 1.0, "devops,deployment"),
            ("Create database seed script for staging data", "medium", 3, 1.0, "devops,database"),
            ("Document staging workflow for team", "low", 1, 0.5, "devops,docs"),
        ],
    },
]

TESTING_TASKS = [
    {
        "title": "Set up testing framework",
        "description": "Configure Jest and React Testing Library for the Next.js app. Set up test utilities, mocks, and coverage thresholds.",
        "priority": "high",
        "context": "Next.js with Jest and React Testing Library",
        "subtasks": [
            ("Install and configure Jest for Next.js", "high", 2, 0.5, "testing,setup"),
            ("Set up React Testing Library with custom render", "high", 2, 0.5, "testing,setup"),
            ("Create mock utilities for Supabase client", "medium", 3, 1.0, "testing,mocks"),
            ("Configure coverage thresholds and reporters", "medium", 2, 0.5, "testing,setup"),
            ("Write example tests as templates", "low", 2, 1.0, "testing,docs"),
        ],
    },
    {
        "title": "Write integration tests for task API",
        "description": "Create integration tests for all task CRUD endpoints. Test auth, validation, error cases, and database state.",
        "priority": "high",
        "context": "API routes with Supabase test database",
        "subtasks": [
            ("Set up test database and fixtures", "high", 3, 1.0, "testing,setup"),
            ("Write tests for task creation endpoint", "high", 3, 1.0, "testing,api"),
            ("Write tests for task listing with filters", "medium", 3, 1.0, "testing,api"),
            ("Write tests for task update and delete", "medium", 3, 1.0, "testing,api"),
            ("Test authorization and error handling", "high", 3, 1.0, "testing,api,security"),
            ("Add cleanup and teardown logic", "medium", 1, 0.5, "testing,setup"),
        ],
    },
    {
        "title": "Write E2E tests for critical user flows",
        "description": "Create end-to-end tests using Playwright for login, task creation, sprint management, and team collaboration flows.",
        "priority": "medium",
        "context": "Playwright with Next.js dev server",
        "subtasks": [
            ("Install and configure Playwright", "medium", 2, 0.5, "testing,e2e"),
            ("Write login and signup flow tests", "high", 3, 1.0, "testing,e2e,auth"),
            ("Write task creation and editing flow tests", "high", 3, 1.5, "testing,e2e"),
            ("Write sprint management flow tests", "medium", 3, 1.5, "testing,e2e"),
            ("Write team invite and collaboration tests", "medium", 3, 1.5, "testing,e2e"),
            ("Set up CI integration for E2E tests", "medium", 2, 1.0, "testing,e2e,ci"),
        ],
    },
    {
        "title": "Add unit tests for utility functions",
        "description": "Write comprehensive unit tests for all shared utility functions, date helpers, validation schemas, and data transformation functions.",
        "priority": "low",
        "context": "Jest with TypeScript",
        "subtasks": [
            ("Test date formatting and timezone utilities", "low", 2, 0.5, "testing,unit"),
            ("Test validation schema helpers", "medium", 2, 0.5, "testing,unit"),
            ("Test data transformation functions", "medium", 2, 0.5, "testing,unit"),
            ("Test permission checking utilities", "medium", 2, 0.5, "testing,unit"),
        ],
    },
]

DESIGN_TASKS = [
    {
        "title": "Design project management UI kit",
        "description": "Create a comprehensive UI kit with reusable components for the project management interface including task cards, sprint boards, and member avatars.",
        "priority": "medium",
        "context": "Figma design system with Tailwind-compatible tokens",
        "subtasks": [
            ("Define color palette and typography scale", "high", 2, 0.5, "design,system"),
            ("Design task card variants (compact, detailed, kanban)", "high", 3, 1.5, "design,ui"),
            ("Create sprint board layout templates", "medium", 3, 1.5, "design,ui"),
            ("Design member avatar and badge components", "low", 2, 0.5, "design,ui"),
            ("Build icon set for status and priority indicators", "medium", 2, 1.0, "design,ui"),
            ("Document component specs for developers", "low", 2, 1.0, "design,docs"),
        ],
    },
    {
        "title": "Create wireframes for AI features",
        "description": "Design wireframes for AI-powered features: task decomposition panel, smart assignment suggestions, and sprint risk dashboard.",
        "priority": "high",
        "context": "AI features integrated into existing PM interface",
        "subtasks": [
            ("Wireframe task decomposition suggestion panel", "high", 3, 1.0, "design,ai,wireframe"),
            ("Design assignment suggestion dropdown UI", "high", 3, 1.0, "design,ai,wireframe"),
            ("Create sprint risk dashboard wireframe", "high", 5, 2.0, "design,ai,wireframe"),
            ("Design confidence score visualizations", "medium", 2, 1.0, "design,ai"),
            ("Create user flow diagrams for AI interactions", "medium", 2, 1.0, "design,ux"),
        ],
    },
    {
        "title": "Design responsive mobile layout",
        "description": "Adapt the project management interface for mobile devices with touch-friendly interactions and simplified navigation.",
        "priority": "low",
        "context": "Mobile-first responsive design",
        "subtasks": [
            ("Design mobile navigation pattern", "medium", 3, 1.0, "design,mobile"),
            ("Adapt task list view for small screens", "medium", 3, 1.0, "design,mobile"),
            ("Design mobile-friendly task creation form", "medium", 2, 1.0, "design,mobile"),
            ("Create touch-friendly drag-and-drop alternative", "high", 3, 1.5, "design,mobile,ux"),
        ],
    },
]

DOCUMENTATION_TASKS = [
    {
        "title": "Write API documentation",
        "description": "Create comprehensive API documentation for all endpoints including request/response examples, error codes, and authentication guide.",
        "priority": "medium",
        "context": "REST API with OpenAPI/Swagger",
        "subtasks": [
            ("Document authentication and API key setup", "high", 2, 0.5, "docs,api"),
            ("Write endpoint reference for task APIs", "medium", 3, 1.5, "docs,api"),
            ("Write endpoint reference for sprint APIs", "medium", 3, 1.5, "docs,api"),
            ("Add request/response examples for each endpoint", "medium", 3, 1.0, "docs,api"),
            ("Document error codes and troubleshooting", "low", 2, 1.0, "docs,api"),
        ],
    },
    {
        "title": "Create developer setup guide",
        "description": "Write a getting-started guide for new developers joining the project. Cover environment setup, dependencies, database configuration, and running locally.",
        "priority": "high",
        "context": "Next.js + Supabase + Python AI server",
        "subtasks": [
            ("Document prerequisite software and versions", "high", 1, 0.5, "docs,setup"),
            ("Write Next.js app setup instructions", "high", 2, 0.5, "docs,setup"),
            ("Write Supabase local setup and migration guide", "high", 2, 1.0, "docs,setup"),
            ("Document AI server Python setup", "medium", 2, 1.0, "docs,setup"),
            ("Add troubleshooting FAQ section", "low", 2, 1.0, "docs,setup"),
        ],
    },
    {
        "title": "Write architecture decision records",
        "description": "Document key architectural decisions including technology choices, data model design, AI model selection, and deployment strategy.",
        "priority": "low",
        "context": "ADR format for FYP documentation",
        "subtasks": [
            ("Document technology stack selection rationale", "medium", 2, 1.0, "docs,architecture"),
            ("Write data model and schema design ADR", "medium", 3, 1.5, "docs,architecture"),
            ("Document AI model architecture decisions", "medium", 3, 1.5, "docs,architecture"),
            ("Write deployment and infrastructure ADR", "low", 2, 1.0, "docs,architecture"),
        ],
    },
]

ALL_CATEGORIES = {
    "frontend": FRONTEND_TASKS,
    "backend": BACKEND_TASKS,
    "devops": DEVOPS_TASKS,
    "testing": TESTING_TASKS,
    "design": DESIGN_TASKS,
    "documentation": DOCUMENTATION_TASKS,
}

# --- Build training examples ---

def format_input(task: dict) -> str:
    """Build the model input prompt from a task dict."""
    return (
        f"Decompose this software task into subtasks:\n"
        f"Task: {task['title']}\n"
        f"Description: {task['description']}\n"
        f"Priority: {task['priority']}\n"
        f"Context: {task['context']}"
    )

def format_output(subtasks: list[tuple]) -> str:
    """Build the structured output from a list of subtask tuples."""
    lines = []
    for i, (title, priority, points, days, tags) in enumerate(subtasks, 1):
        lines.append(
            f"SUBTASK_{i}: {title} | PRIORITY: {priority} | "
            f"POINTS: {points} | DAYS: {days} | TAGS: {tags}"
        )
    return "\n".join(lines)


# Variation generators for data augmentation
PRIORITY_MAP = ["low", "medium", "high", "urgent"]

TITLE_PREFIXES = [
    "Implement", "Build", "Create", "Set up", "Configure",
    "Design", "Write", "Add", "Refactor", "Fix",
    "Integrate", "Optimize", "Migrate", "Update", "Deploy",
]

DESCRIPTION_VARIATIONS = [
    "This involves {action} with proper error handling and validation.",
    "We need to {action}. Should follow best practices and be well-tested.",
    "{action}. Include documentation and code review.",
    "The goal is to {action} while maintaining backward compatibility.",
    "{action}. This is part of the core feature set.",
]

CONTEXT_TEMPLATES = [
    "React/Next.js project with TypeScript and Tailwind CSS",
    "Supabase backend with PostgreSQL and Row Level Security",
    "Python FastAPI server with ML model inference",
    "Full-stack project with Next.js frontend and Supabase backend",
    "Agile team with 2-week sprints and CI/CD pipeline",
]


def generate_augmented_examples(base_tasks: list[dict], category: str, count: int) -> list[dict]:
    """Generate augmented variations from base task templates."""
    examples = []

    # First, add all base examples as-is
    for task in base_tasks:
        examples.append({
            "input_text": format_input(task),
            "target_text": format_output(task["subtasks"]),
        })

    # Then generate variations
    while len(examples) < count:
        task = random.choice(base_tasks)

        # Vary the priority
        new_priority = random.choice(PRIORITY_MAP)

        # Slight variation in title
        prefix = random.choice(TITLE_PREFIXES)
        base_title = task["title"]
        # Sometimes swap the prefix
        for p in TITLE_PREFIXES:
            if base_title.startswith(p):
                base_title = base_title[len(p):].strip()
                break
        new_title = f"{prefix} {base_title}"

        # Vary context
        new_context = random.choice(CONTEXT_TEMPLATES)

        # Vary subtask points slightly
        new_subtasks = []
        for title, prio, points, days, tags in task["subtasks"]:
            # Small random perturbation
            point_options = [max(1, points - 1), points, min(13, points + 1)]
            new_points = random.choice([p for p in point_options if p in {1,2,3,5,8,13}] or [points])
            day_delta = random.choice([-0.5, 0, 0, 0.5])
            new_days = max(0.5, days + day_delta)
            new_prio = prio if random.random() > 0.3 else new_priority
            new_subtasks.append((title, new_prio, new_points, new_days, tags))

        varied = {
            "title": new_title,
            "description": task["description"],
            "priority": new_priority,
            "context": new_context,
            "subtasks": new_subtasks,
        }

        examples.append({
            "input_text": format_input(varied),
            "target_text": format_output(varied["subtasks"]),
        })

    return examples[:count]


# Generate examples per category
TARGET_PER_CATEGORY = 350  # 350 * 6 = 2100, plus structural = ~2500-2600
all_examples = []

for cat_name, cat_tasks in ALL_CATEGORIES.items():
    cat_examples = generate_augmented_examples(cat_tasks, cat_name, TARGET_PER_CATEGORY)
    all_examples.extend(cat_examples)
    print(f"  {cat_name}: {len(cat_examples)} examples")

# Add structural pattern examples
all_examples.extend(structural_examples)

# Shuffle
random.shuffle(all_examples)

print(f"\nTotal training examples: {len(all_examples)}")

# Split into train / validation / test (80/10/10)
n = len(all_examples)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_data = all_examples[:train_end]
val_data = all_examples[train_end:val_end]
test_data = all_examples[val_end:]

print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

# Convert to HuggingFace Dataset
dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data),
})
print(f"\nDataset:\n{dataset}")


# ============================================================================
# SECTION 4: Tokenization
# ============================================================================
# MARKDOWN: ## 4. Tokenization
#
# We load the FLAN-T5-small tokenizer and tokenize all examples.
# - **Max input length:** 512 tokens (task descriptions can be verbose)
# - **Max output length:** 256 tokens (subtask lists are structured but compact)
# - **Padding:** to longest in batch (saves memory vs fixed padding)

print("\n" + "="*60)
print("SECTION 4: Tokenizing dataset")
print("="*60)

MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
print(f"Loaded tokenizer: {MODEL_NAME}")
print(f"Vocab size: {tokenizer.vocab_size}")


def tokenize_function(examples):
    """Tokenize input-output pairs for seq2seq training."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        examples["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )

    # Replace padding token ids in labels with -100 so they're ignored in loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["input_text", "target_text"],
    desc="Tokenizing",
)

print(f"\nTokenized dataset:\n{tokenized_dataset}")

# Sanity check: decode a sample
sample_idx = 0
sample_input = tokenizer.decode(
    tokenized_dataset["train"][sample_idx]["input_ids"],
    skip_special_tokens=True,
)
sample_label_ids = [
    t if t != -100 else tokenizer.pad_token_id
    for t in tokenized_dataset["train"][sample_idx]["labels"]
]
sample_target = tokenizer.decode(sample_label_ids, skip_special_tokens=True)
print(f"\n--- Sample {sample_idx} ---")
print(f"INPUT:\n{sample_input[:300]}...")
print(f"\nTARGET:\n{sample_target[:300]}...")


# ============================================================================
# SECTION 5: Fine-tuning
# ============================================================================
# MARKDOWN: ## 5. Fine-tuning
#
# We fine-tune FLAN-T5-small with these hyperparameters:
# - **Epochs:** 5
# - **Batch size:** 8 (fits in Colab T4 free GPU with 15 GB VRAM)
# - **Learning rate:** 3e-4 with linear warmup over 100 steps
# - **Evaluation:** every 200 steps
# - **Best checkpoint:** saved by ROUGE-L score
# - **FP16:** enabled for T4 GPU memory efficiency
#
# Expected training time on T4: ~45-90 minutes.

print("\n" + "="*60)
print("SECTION 5: Fine-tuning FLAN-T5-small")
print("="*60)

# Load model
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
print(f"Model parameters: {model.num_parameters() / 1e6:.1f}M")

# ROUGE metric for evaluation
rouge_metric = evaluate.load("rouge")


def compute_metrics(eval_preds):
    """Compute ROUGE scores for evaluation."""
    preds, labels = eval_preds

    # Replace -100 in labels with pad token for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Clip predictions to valid token range to prevent OverflowError
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    return {
        "rouge1": round(result["rouge1"], 4),
        "rouge2": round(result["rouge2"], 4),
        "rougeL": round(result["rougeL"], 4),
    }


# Output directory
OUTPUT_DIR = "/content/flan-t5-pm-decomposition" if os.path.exists("/content") else "./flan-t5-pm-decomposition"
LOGGING_DIR = os.path.join(OUTPUT_DIR, "logs")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    # Evaluation
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    # Generation config for evaluation
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    # Performance
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    # Logging
    logging_dir=LOGGING_DIR,
    logging_steps=50,
    report_to="none",  # Set to "tensorboard" if you want TB logging
    # Misc
    seed=SEED,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print(f"\nTraining configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Warmup steps: {training_args.warmup_steps}")
print(f"  Eval every: {training_args.eval_steps} steps")
print(f"  FP16: {training_args.fp16}")
print(f"  Output: {OUTPUT_DIR}")

print(f"\nStarting training...")
train_result = trainer.train()

print(f"\n--- Training Complete ---")
print(f"  Training loss: {train_result.metrics.get('train_loss', 'N/A')}")
print(f"  Training time: {train_result.metrics.get('train_runtime', 0):.0f}s")
print(f"  Metrics: {train_result.metrics}")


# ============================================================================
# SECTION 6: Evaluation
# ============================================================================
# MARKDOWN: ## 6. Evaluation
#
# We evaluate the fine-tuned model on the held-out test set using:
# - **ROUGE-1:** Unigram overlap (individual word matching)
# - **ROUGE-2:** Bigram overlap (two-word phrase matching)
# - **ROUGE-L:** Longest common subsequence (structural similarity)
#
# We also manually inspect 10 predictions to qualitatively assess output quality.
# Target: ROUGE-L > 0.35 (reasonable for this task and dataset size).

print("\n" + "="*60)
print("SECTION 6: Evaluation")
print("="*60)

# Evaluate on test set
test_results = trainer.evaluate(tokenized_dataset["test"])
print(f"\n--- Test Set Results ---")
print(f"  ROUGE-1: {test_results.get('eval_rouge1', 'N/A')}")
print(f"  ROUGE-2: {test_results.get('eval_rouge2', 'N/A')}")
print(f"  ROUGE-L: {test_results.get('eval_rougeL', 'N/A')}")
print(f"  Eval Loss: {test_results.get('eval_loss', 'N/A'):.4f}")

rouge_l = test_results.get("eval_rougeL", 0)
if rouge_l >= 0.35:
    print(f"\n  TARGET MET: ROUGE-L {rouge_l:.4f} >= 0.35")
else:
    print(f"\n  TARGET NOT MET: ROUGE-L {rouge_l:.4f} < 0.35")
    print("  Consider: more training data, more epochs, or larger model")

# Manual inspection: generate predictions for 10 test examples
print(f"\n--- Manual Inspection (10 test examples) ---")

model.eval()
for i in range(min(10, len(test_data))):
    input_text = test_data[i]["input_text"]
    expected = test_data[i]["target_text"]

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True,
            do_sample=False,
        )

    predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n{'='*50}")
    print(f"Example {i+1}")
    print(f"{'='*50}")
    # Print just the task title from input
    title_line = [l for l in input_text.split("\n") if l.startswith("Task:")]
    print(f"TASK: {title_line[0] if title_line else input_text[:80]}")
    print(f"\nEXPECTED:\n{expected[:200]}")
    print(f"\nPREDICTED:\n{predicted[:200]}")


# ============================================================================
# SECTION 7: Save & Export
# ============================================================================
# MARKDOWN: ## 7. Save & Export
#
# We save the fine-tuned model and tokenizer, then create a zip archive
# for download. The saved model can be placed directly into the AI server's
# `model_weights/decomposition/` directory.
#
# Model size is approximately 300 MB (FLAN-T5-small).

print("\n" + "="*60)
print("SECTION 7: Save & Export")
print("="*60)

SAVE_DIR = OUTPUT_DIR  # reuse the output dir from training

# Save model and tokenizer
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Save training metadata
metadata = {
    "model_name": MODEL_NAME,
    "model_version": "flan-t5-pm-v1",
    "fine_tuned_on": "synthetic-pm-decomposition",
    "training_examples": len(train_data),
    "validation_examples": len(val_data),
    "test_examples": len(test_data),
    "epochs": int(training_args.num_train_epochs),
    "batch_size": training_args.per_device_train_batch_size,
    "learning_rate": training_args.learning_rate,
    "test_rouge1": test_results.get("eval_rouge1"),
    "test_rouge2": test_results.get("eval_rouge2"),
    "test_rougeL": test_results.get("eval_rougeL"),
    "max_input_length": MAX_INPUT_LENGTH,
    "max_target_length": MAX_TARGET_LENGTH,
    "output_format": "SUBTASK_N: title | PRIORITY: p | POINTS: n | DAYS: n | TAGS: t1,t2",
}

with open(os.path.join(SAVE_DIR, "training_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved training metadata to {SAVE_DIR}/training_metadata.json")

# Calculate model size
total_size = 0
for root, dirs, files in os.walk(SAVE_DIR):
    for file in files:
        fp = os.path.join(root, file)
        if not os.path.islink(fp):
            total_size += os.path.getsize(fp)

print(f"\nModel saved to: {SAVE_DIR}")
print(f"Model size: {total_size / 1e6:.1f} MB")

# List saved files
print(f"\nSaved files:")
for f in sorted(os.listdir(SAVE_DIR)):
    fp = os.path.join(SAVE_DIR, f)
    if os.path.isfile(fp):
        size = os.path.getsize(fp)
        print(f"  {f}: {size / 1e6:.1f} MB" if size > 1e6 else f"  {f}: {size / 1e3:.1f} KB")

# Zip for download
import shutil
zip_path = shutil.make_archive(
    base_name=SAVE_DIR,
    format="zip",
    root_dir=os.path.dirname(SAVE_DIR),
    base_dir=os.path.basename(SAVE_DIR),
)
zip_size = os.path.getsize(zip_path)
print(f"\nZip archive: {zip_path} ({zip_size / 1e6:.1f} MB)")

# Download in Colab
try:
    from google.colab import files
    print("Downloading zip to your browser...")
    files.download(zip_path)
except ImportError:
    print(f"Not in Colab — zip saved locally at: {zip_path}")

# Optional: push to HuggingFace Hub
# Uncomment and set your token to push:
# from huggingface_hub import login
# login(token="hf_YOUR_TOKEN")
# model.push_to_hub("your-username/flan-t5-pm-decomposition")
# tokenizer.push_to_hub("your-username/flan-t5-pm-decomposition")
# print("Pushed to HuggingFace Hub!")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"\nTo use this model in the AI server:")
print(f"  1. Unzip {os.path.basename(zip_path)}")
print(f"  2. Copy contents to ai-server/model_weights/decomposition/")
print(f"  3. Restart the AI server")
print(f"  4. Check /health — decomposition should show true")
