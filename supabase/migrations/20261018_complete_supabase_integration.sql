-- =============================================================================
-- QRShield: Supabase Complete Database Integration Migration
-- Date: 2026-10-18
-- Tables: scan_history, threat_analysis, prevention_decisions (and prevention_decision view),
--         rl_feedback, user_roles, user_profiles
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Extend scan_history with user_id and destination_domain
-- -----------------------------------------------------------------------------
ALTER TABLE public.scan_history
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS destination_domain TEXT;

CREATE INDEX IF NOT EXISTS idx_scan_history_user_id 
    ON public.scan_history(user_id);
CREATE INDEX IF NOT EXISTS idx_scan_history_destination_domain 
    ON public.scan_history(destination_domain);
CREATE INDEX IF NOT EXISTS idx_scan_history_created_at 
    ON public.scan_history(created_at DESC);

-- -----------------------------------------------------------------------------
-- 2. Foreign Key Indexes for High-Performance Joins across all three child tables
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
-- 4. Create user_profiles table for user profile details
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    full_name TEXT NOT NULL DEFAULT 'SecOps User',
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    account_status TEXT NOT NULL DEFAULT 'Active' CHECK (account_status IN ('Active', 'Suspended', 'Deactivated')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_role ON public.user_profiles(role);

-- -----------------------------------------------------------------------------
-- 5. Row Level Security (RLS) Configuration
-- -----------------------------------------------------------------------------
ALTER TABLE public.scan_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.threat_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.prevention_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rl_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;

-- 5a. Service Role Full Access (enables backend worker and server processes)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'service_role_full_access_scan_history') THEN
        CREATE POLICY service_role_full_access_scan_history ON public.scan_history FOR ALL TO service_role USING (true) WITH CHECK (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'service_role_full_access_threat_analysis') THEN
        CREATE POLICY service_role_full_access_threat_analysis ON public.threat_analysis FOR ALL TO service_role USING (true) WITH CHECK (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'service_role_full_access_prevention_decisions') THEN
        CREATE POLICY service_role_full_access_prevention_decisions ON public.prevention_decisions FOR ALL TO service_role USING (true) WITH CHECK (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'service_role_full_access_rl_feedback') THEN
        CREATE POLICY service_role_full_access_rl_feedback ON public.rl_feedback FOR ALL TO service_role USING (true) WITH CHECK (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'service_role_full_access_user_profiles') THEN
        CREATE POLICY service_role_full_access_user_profiles ON public.user_profiles FOR ALL TO service_role USING (true) WITH CHECK (true);
    END IF;
END $$;

-- 5b. Authenticated Administrator Full Read Access
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'admins_select_all_scans') THEN
        CREATE POLICY admins_select_all_scans ON public.scan_history FOR SELECT TO authenticated
        USING (EXISTS (SELECT 1 FROM public.user_roles WHERE user_roles.user_id = auth.uid() AND user_roles.role = 'admin'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'admins_select_all_threats') THEN
        CREATE POLICY admins_select_all_threats ON public.threat_analysis FOR SELECT TO authenticated
        USING (EXISTS (SELECT 1 FROM public.user_roles WHERE user_roles.user_id = auth.uid() AND user_roles.role = 'admin'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'admins_select_all_decisions') THEN
        CREATE POLICY admins_select_all_decisions ON public.prevention_decisions FOR SELECT TO authenticated
        USING (EXISTS (SELECT 1 FROM public.user_roles WHERE user_roles.user_id = auth.uid() AND user_roles.role = 'admin'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'admins_select_all_rl_feedback') THEN
        CREATE POLICY admins_select_all_rl_feedback ON public.rl_feedback FOR SELECT TO authenticated
        USING (EXISTS (SELECT 1 FROM public.user_roles WHERE user_roles.user_id = auth.uid() AND user_roles.role = 'admin'));
    END IF;
END $$;

-- 5c. Authenticated Users Read Own Records
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'users_select_own_scans') THEN
        CREATE POLICY users_select_own_scans ON public.scan_history FOR SELECT TO authenticated
        USING (auth.uid() = user_id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'users_read_own_profile') THEN
        CREATE POLICY users_read_own_profile ON public.user_profiles FOR SELECT TO authenticated
        USING (auth.uid() = id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'users_update_own_profile_name') THEN
        CREATE POLICY users_update_own_profile_name ON public.user_profiles FOR UPDATE TO authenticated
        USING (auth.uid() = id)
        WITH CHECK (auth.uid() = id AND role = (SELECT role FROM public.user_profiles WHERE id = auth.uid()));
    END IF;
END $$;
