-- =============================================================================
-- Migration: 20261018_idempotent_complete_fix.sql
-- Project: QRShield Cybersecurity Operations Center
-- Description:
--   1. Safe column additions for scan_history (user_id, destination_domain).
--   2. Safe creation of user_profiles table (id, user_id, email, full_name, account_status).
--   3. Compatible prevention_decision view.
--   4. Foreign key and performance indexes across all tables.
--   5. 100% Idempotent Row Level Security (RLS) policies using DROP POLICY IF EXISTS.
--   6. Enforces strict privilege separation without disabling RLS or assuming role on user_profiles.
-- =============================================================================

-- Enable pgcrypto extension for UUID generation if not already active
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- -----------------------------------------------------------------------------
-- 1. Extend scan_history table with user_id and destination_domain
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
          AND table_name = 'scan_history' 
          AND column_name = 'user_id'
    ) THEN
        ALTER TABLE public.scan_history 
        ADD COLUMN user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
          AND table_name = 'scan_history' 
          AND column_name = 'destination_domain'
    ) THEN
        ALTER TABLE public.scan_history 
        ADD COLUMN destination_domain TEXT;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_scan_history_user_id 
    ON public.scan_history(user_id);
CREATE INDEX IF NOT EXISTS idx_scan_history_destination_domain 
    ON public.scan_history(destination_domain);
CREATE INDEX IF NOT EXISTS idx_scan_history_created_at 
    ON public.scan_history(created_at DESC);

-- -----------------------------------------------------------------------------
-- 2. Ensure foreign key indexes for joining telemetry tables
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_threat_analysis_scan_id 
    ON public.threat_analysis(scan_id);

CREATE INDEX IF NOT EXISTS idx_prevention_decisions_scan_id 
    ON public.prevention_decisions(scan_id);

CREATE INDEX IF NOT EXISTS idx_rl_feedback_scan_id 
    ON public.rl_feedback(scan_id);
CREATE INDEX IF NOT EXISTS idx_rl_feedback_created_at 
    ON public.rl_feedback(created_at DESC);

-- -----------------------------------------------------------------------------
-- 3. Compatibility View: prevention_decision (singular alias)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.prevention_decision AS
SELECT * FROM public.prevention_decisions;

-- -----------------------------------------------------------------------------
-- 4. Create public.user_profiles table (without assuming nonexistent role column)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    full_name TEXT NOT NULL DEFAULT 'SecOps User',
    account_status TEXT NOT NULL DEFAULT 'active' CHECK (account_status IN ('active', 'suspended', 'deactivated')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    last_sign_in_at TIMESTAMPTZ,
    CONSTRAINT uq_user_profiles_user_id UNIQUE (user_id),
    CONSTRAINT uq_user_profiles_email UNIQUE (email)
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON public.user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON public.user_profiles(email);

-- -----------------------------------------------------------------------------
-- 5. Ensure public.user_roles table exists for RBAC
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.user_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user', 'analyst')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    CONSTRAINT uq_user_roles_user_id UNIQUE (user_id),
    CONSTRAINT uq_user_roles_email UNIQUE (email)
);

CREATE INDEX IF NOT EXISTS idx_user_roles_user_id ON public.user_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_email ON public.user_roles(email);

-- =============================================================================
-- 6. Row Level Security (RLS) Configuration (100% Idempotent)
-- =============================================================================

ALTER TABLE public.scan_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.threat_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.prevention_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rl_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;

-- 6a. Service Role Full Access (enables backend workers and operations)
DROP POLICY IF EXISTS "Service role manages user roles" ON public.user_roles;
CREATE POLICY "Service role manages user roles" 
    ON public.user_roles FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

DROP POLICY IF EXISTS "Service role manages all user profiles" ON public.user_profiles;
CREATE POLICY "Service role manages all user profiles" 
    ON public.user_profiles FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

DROP POLICY IF EXISTS "Allow service role full access to scan_history" ON public.scan_history;
CREATE POLICY "Allow service role full access to scan_history" 
    ON public.scan_history FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

DROP POLICY IF EXISTS "Allow service role full access to threat_analysis" ON public.threat_analysis;
CREATE POLICY "Allow service role full access to threat_analysis" 
    ON public.threat_analysis FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

DROP POLICY IF EXISTS "Allow service role full access to prevention_decisions" ON public.prevention_decisions;
CREATE POLICY "Allow service role full access to prevention_decisions" 
    ON public.prevention_decisions FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

DROP POLICY IF EXISTS "Allow service role full access to rl_feedback" ON public.rl_feedback;
CREATE POLICY "Allow service role full access to rl_feedback" 
    ON public.rl_feedback FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

-- 6b. Authenticated User Policies (Self-access only, preventing privilege escalation)
DROP POLICY IF EXISTS "Users can read own role" ON public.user_roles;
CREATE POLICY "Users can read own role" 
    ON public.user_roles FOR SELECT 
    TO authenticated 
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can read own profile" ON public.user_profiles;
CREATE POLICY "Users can read own profile" 
    ON public.user_profiles FOR SELECT 
    TO authenticated 
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own profile name" ON public.user_profiles;
CREATE POLICY "Users can update own profile name" 
    ON public.user_profiles FOR UPDATE 
    TO authenticated 
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can view own scan history" ON public.scan_history;
CREATE POLICY "Users can view own scan history" 
    ON public.scan_history FOR SELECT 
    TO authenticated 
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own scans" ON public.scan_history;
CREATE POLICY "Users can insert own scans" 
    ON public.scan_history FOR INSERT 
    TO authenticated 
    WITH CHECK (auth.uid() = user_id OR user_id IS NULL);
