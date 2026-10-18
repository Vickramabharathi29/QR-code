-- =============================================================================
-- Migration: 20261018_admin_auth_and_roles.sql
-- Project: QRShield Cybersecurity Operations Center
-- Description: Creates user_roles table for Role-Based Access Control (RBAC),
--              enforces Row Level Security (RLS), and sets up admin authorization policies.
-- Execute this migration in the Supabase SQL Editor:
-- https://supabase.com/dashboard/project/tjfhuulhzvlkqvjkevki/sql
-- =============================================================================

-- 1. Create public.user_roles table linked to Supabase auth.users
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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_roles_user_id ON public.user_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_email ON public.user_roles(email);

-- 2. Enable Row Level Security (RLS) on user_roles
ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;

-- 3. RLS Policies for user_roles
-- Authenticated users can read their own role assignment
DROP POLICY IF EXISTS "Users can read own role" ON public.user_roles;
CREATE POLICY "Users can read own role"
    ON public.user_roles FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

-- Restrict INSERT, UPDATE, DELETE to service_role exclusively.
-- Standard users cannot escalate their privileges or assign themselves the 'admin' role.
DROP POLICY IF EXISTS "Service role manages user roles" ON public.user_roles;
CREATE POLICY "Service role manages user roles"
    ON public.user_roles FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- 4. Update scan_history RLS policies to protect historical audit records
-- Only authenticated users with admin role or service_role can view full historical telemetry
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
