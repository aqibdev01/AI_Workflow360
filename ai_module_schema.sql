-- =====================================================
-- AI MODULE — COMPLETE STANDALONE SCHEMA
-- =====================================================
-- This file contains ALL database objects needed by the
-- AI modules (M6, M10, M11). It is fully self-contained
-- and does NOT alter or depend on running against any
-- existing migration file.
--
-- Prerequisites (must already exist in the database):
--   - extension: uuid-ossp  (for uuid_generate_v4)
--   - table:    public.tasks    (id UUID PK)
--   - table:    public.sprints  (id UUID PK)
--   - table:    public.projects (id UUID PK)
--   - table:    public.project_members (project_id, user_id, role)
--   - schema:   auth  (Supabase Auth — auth.users with id UUID)
--   - types:    task_status, task_priority (existing enums on tasks)
--
-- Safe to run multiple times — uses IF NOT EXISTS / IF EXISTS
-- and the duplicate_object exception pattern throughout.
-- =====================================================


-- ==========================================================
-- SECTION 1: ENUM TYPES
-- ==========================================================

DO $$ BEGIN
    CREATE TYPE decomposition_status AS ENUM (
        'none',
        'suggested',
        'partially_accepted',
        'fully_accepted'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE skill_level AS ENUM (
        'beginner',
        'intermediate',
        'expert'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE decomposition_review_status AS ENUM (
        'pending',
        'accepted',
        'partially_accepted',
        'rejected'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE risk_level AS ENUM (
        'low',
        'medium',
        'high',
        'critical'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;


-- ==========================================================
-- SECTION 2: AI COLUMNS ON EXISTING TABLES
-- ==========================================================

-- ----- tasks table: new AI columns -----
ALTER TABLE public.tasks
    ADD COLUMN IF NOT EXISTS story_points       SMALLINT,
    ADD COLUMN IF NOT EXISTS estimated_days      NUMERIC(4,1),
    ADD COLUMN IF NOT EXISTS actual_days         NUMERIC(4,1),
    ADD COLUMN IF NOT EXISTS tags                TEXT[]              DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS complexity_score    NUMERIC(3,2),
    ADD COLUMN IF NOT EXISTS ai_suggested_assignee_id UUID,
    ADD COLUMN IF NOT EXISTS ai_assignee_confidence   NUMERIC(3,2),
    ADD COLUMN IF NOT EXISTS parent_task_id      UUID,
    ADD COLUMN IF NOT EXISTS is_ai_generated     BOOLEAN             DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS decomposition_status decomposition_status DEFAULT 'none';

-- Constraints (wrapped in DO blocks so re-runs don't fail)
DO $$ BEGIN
    ALTER TABLE public.tasks
        ADD CONSTRAINT tasks_story_points_fibonacci
            CHECK (story_points IS NULL OR story_points IN (1,2,3,5,8,13));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE public.tasks
        ADD CONSTRAINT tasks_complexity_score_range
            CHECK (complexity_score IS NULL OR (complexity_score >= 0.00 AND complexity_score <= 1.00));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE public.tasks
        ADD CONSTRAINT tasks_ai_assignee_confidence_range
            CHECK (ai_assignee_confidence IS NULL OR (ai_assignee_confidence >= 0.00 AND ai_assignee_confidence <= 1.00));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Foreign keys
DO $$ BEGIN
    ALTER TABLE public.tasks
        ADD CONSTRAINT tasks_ai_suggested_assignee_fk
            FOREIGN KEY (ai_suggested_assignee_id) REFERENCES auth.users(id) ON DELETE SET NULL;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE public.tasks
        ADD CONSTRAINT tasks_parent_task_fk
            FOREIGN KEY (parent_task_id) REFERENCES public.tasks(id) ON DELETE SET NULL;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ----- sprints table: new AI columns -----
ALTER TABLE public.sprints
    ADD COLUMN IF NOT EXISTS velocity        NUMERIC(5,1),
    ADD COLUMN IF NOT EXISTS capacity        NUMERIC(5,1),
    ADD COLUMN IF NOT EXISTS ai_risk_score   NUMERIC(3,2),
    ADD COLUMN IF NOT EXISTS ai_risk_factors JSONB,
    ADD COLUMN IF NOT EXISTS ai_analyzed_at  TIMESTAMPTZ;

DO $$ BEGIN
    ALTER TABLE public.sprints
        ADD CONSTRAINT sprints_ai_risk_score_range
            CHECK (ai_risk_score IS NULL OR (ai_risk_score >= 0.00 AND ai_risk_score <= 1.00));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;


-- ==========================================================
-- SECTION 3: NEW TABLES
-- ==========================================================

-- ----- 3a. user_skills -----
CREATE TABLE IF NOT EXISTS public.user_skills (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    skill       TEXT        NOT NULL,
    level       skill_level NOT NULL DEFAULT 'intermediate',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT user_skills_unique UNIQUE (user_id, skill)
);

-- ----- 3b. ai_task_decompositions -----
CREATE TABLE IF NOT EXISTS public.ai_task_decompositions (
    id                UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_task_id    UUID        NOT NULL REFERENCES public.tasks(id) ON DELETE CASCADE,
    suggested_subtasks JSONB      NOT NULL,
    -- suggested_subtasks format:
    -- [{ "title": "", "description": "", "priority": "medium",
    --    "story_points": 3, "tags": ["frontend"], "estimated_days": 1.5 }, ...]
    model_version     TEXT        NOT NULL,
    confidence_score  NUMERIC(3,2),
    status            decomposition_review_status NOT NULL DEFAULT 'pending',
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at       TIMESTAMPTZ,
    reviewed_by       UUID        REFERENCES auth.users(id) ON DELETE SET NULL
);

DO $$ BEGIN
    ALTER TABLE public.ai_task_decompositions
        ADD CONSTRAINT ai_decompositions_confidence_range
            CHECK (confidence_score IS NULL OR (confidence_score >= 0.00 AND confidence_score <= 1.00));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ----- 3c. ai_assignment_logs -----
CREATE TABLE IF NOT EXISTS public.ai_assignment_logs (
    id                    UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id               UUID        NOT NULL REFERENCES public.tasks(id) ON DELETE CASCADE,
    suggested_assignee_id UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    confidence_score      NUMERIC(3,2),
    scoring_breakdown     JSONB,
    -- scoring_breakdown format:
    -- { "skill_match": 0.8, "workload": 0.6, "role_match": 1.0, "availability": 0.9 }
    was_accepted          BOOLEAN,       -- NULL until user confirms or rejects
    final_assignee_id     UUID        REFERENCES auth.users(id) ON DELETE SET NULL,
    model_version         TEXT        NOT NULL,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

DO $$ BEGIN
    ALTER TABLE public.ai_assignment_logs
        ADD CONSTRAINT ai_assignment_confidence_range
            CHECK (confidence_score IS NULL OR (confidence_score >= 0.00 AND confidence_score <= 1.00));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ----- 3d. ai_bottleneck_reports -----
CREATE TABLE IF NOT EXISTS public.ai_bottleneck_reports (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    sprint_id       UUID        NOT NULL REFERENCES public.sprints(id) ON DELETE CASCADE,
    project_id      UUID        NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
    risk_level      risk_level  NOT NULL,
    risk_score      NUMERIC(3,2),
    bottlenecks     JSONB       NOT NULL,
    -- bottlenecks format:
    -- [{ "type": "overload", "description": "...", "affected_tasks": ["uuid",...], "severity": "high" }, ...]
    recommendations JSONB,
    -- recommendations format:
    -- [{ "action": "reassign task X", "reason": "...", "priority": "high" }, ...]
    model_version   TEXT        NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

DO $$ BEGIN
    ALTER TABLE public.ai_bottleneck_reports
        ADD CONSTRAINT ai_bottleneck_risk_score_range
            CHECK (risk_score IS NULL OR (risk_score >= 0.00 AND risk_score <= 1.00));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;


-- ==========================================================
-- SECTION 4: INDEXES
-- ==========================================================

-- Tasks AI lookups
CREATE INDEX IF NOT EXISTS tasks_parent_task_id_idx   ON public.tasks(parent_task_id);
CREATE INDEX IF NOT EXISTS tasks_assignee_status_idx  ON public.tasks(assignee_id, status);
CREATE INDEX IF NOT EXISTS tasks_sprint_status_idx    ON public.tasks(sprint_id, status);

-- User skills
CREATE INDEX IF NOT EXISTS user_skills_user_id_idx    ON public.user_skills(user_id);
CREATE INDEX IF NOT EXISTS user_skills_skill_idx      ON public.user_skills(skill);

-- AI decompositions
CREATE INDEX IF NOT EXISTS ai_decompositions_parent_task_idx ON public.ai_task_decompositions(parent_task_id);

-- AI assignment logs
CREATE INDEX IF NOT EXISTS ai_assignment_task_id_idx  ON public.ai_assignment_logs(task_id);

-- AI bottleneck reports
CREATE INDEX IF NOT EXISTS ai_bottleneck_sprint_id_idx  ON public.ai_bottleneck_reports(sprint_id);
CREATE INDEX IF NOT EXISTS ai_bottleneck_project_id_idx ON public.ai_bottleneck_reports(project_id);


-- ==========================================================
-- SECTION 5: ROW LEVEL SECURITY
-- ==========================================================

ALTER TABLE public.user_skills              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_task_decompositions   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_assignment_logs       ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_bottleneck_reports    ENABLE ROW LEVEL SECURITY;

-- ---- user_skills policies ----
DO $$ BEGIN
    CREATE POLICY "Users can manage own skills"
        ON public.user_skills FOR ALL
        USING (auth.uid() = user_id)
        WITH CHECK (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE POLICY "Project members can view teammate skills"
        ON public.user_skills FOR SELECT
        USING (
            EXISTS (
                SELECT 1 FROM public.project_members pm1
                JOIN public.project_members pm2 ON pm1.project_id = pm2.project_id
                WHERE pm1.user_id = auth.uid()
                  AND pm2.user_id = user_skills.user_id
            )
        );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ---- ai_task_decompositions policies ----
DO $$ BEGIN
    CREATE POLICY "Project members can view decompositions"
        ON public.ai_task_decompositions FOR SELECT
        USING (
            EXISTS (
                SELECT 1 FROM public.tasks t
                JOIN public.project_members pm ON pm.project_id = t.project_id
                WHERE t.id = ai_task_decompositions.parent_task_id
                  AND pm.user_id = auth.uid()
            )
        );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE POLICY "Project leads can manage decompositions"
        ON public.ai_task_decompositions FOR ALL
        USING (
            EXISTS (
                SELECT 1 FROM public.tasks t
                JOIN public.project_members pm ON pm.project_id = t.project_id
                WHERE t.id = ai_task_decompositions.parent_task_id
                  AND pm.user_id = auth.uid()
                  AND pm.role IN ('owner', 'lead')
            )
        );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ---- ai_assignment_logs policies ----
DO $$ BEGIN
    CREATE POLICY "Project members can view assignment logs"
        ON public.ai_assignment_logs FOR SELECT
        USING (
            EXISTS (
                SELECT 1 FROM public.tasks t
                JOIN public.project_members pm ON pm.project_id = t.project_id
                WHERE t.id = ai_assignment_logs.task_id
                  AND pm.user_id = auth.uid()
            )
        );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE POLICY "Project leads can manage assignment logs"
        ON public.ai_assignment_logs FOR ALL
        USING (
            EXISTS (
                SELECT 1 FROM public.tasks t
                JOIN public.project_members pm ON pm.project_id = t.project_id
                WHERE t.id = ai_assignment_logs.task_id
                  AND pm.user_id = auth.uid()
                  AND pm.role IN ('owner', 'lead')
            )
        );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ---- ai_bottleneck_reports policies ----
DO $$ BEGIN
    CREATE POLICY "Project members can view bottleneck reports"
        ON public.ai_bottleneck_reports FOR SELECT
        USING (
            EXISTS (
                SELECT 1 FROM public.project_members pm
                WHERE pm.project_id = ai_bottleneck_reports.project_id
                  AND pm.user_id = auth.uid()
            )
        );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE POLICY "Project leads can manage bottleneck reports"
        ON public.ai_bottleneck_reports FOR ALL
        USING (
            EXISTS (
                SELECT 1 FROM public.project_members pm
                WHERE pm.project_id = ai_bottleneck_reports.project_id
                  AND pm.user_id = auth.uid()
                  AND pm.role IN ('owner', 'lead')
            )
        );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;


-- ==========================================================
-- END OF AI MODULE SCHEMA
-- ==========================================================
