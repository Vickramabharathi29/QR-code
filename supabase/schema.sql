-- =============================================================================
-- QRShield: Supabase PostgreSQL Database Schema
-- Project: QR Scam & Malicious Redirection Detection with Adaptive Threat Prevention
-- =============================================================================

-- Enable pgcrypto extension for UUID generation if not already active
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- -----------------------------------------------------------------------------
-- Table 1: scan_history
-- Records every scanned QR code payload, extracted URL, and processing status
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.scan_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    extracted_url TEXT,
    qr_payload_type TEXT NOT NULL DEFAULT 'URL' 
        CHECK (qr_payload_type IN ('URL', 'TEXT', 'WIFI', 'VCARD', 'OTHER')),
    scan_status TEXT NOT NULL DEFAULT 'PENDING' 
        CHECK (scan_status IN ('PENDING', 'PROCESSED', 'FAILED')),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    destination_domain TEXT
);

-- Index for temporal ordering and dashboard queries
CREATE INDEX IF NOT EXISTS idx_scan_history_created_at 
    ON public.scan_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scan_history_user_id 
    ON public.scan_history(user_id);
CREATE INDEX IF NOT EXISTS idx_scan_history_destination_domain 
    ON public.scan_history(destination_domain);

-- -----------------------------------------------------------------------------
-- Table 2: threat_analysis
-- Stores machine learning classification scores and detected threat indicators
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.threat_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID NOT NULL REFERENCES public.scan_history(id) ON DELETE CASCADE,
    predicted_class TEXT NOT NULL 
        CHECK (predicted_class IN ('SAFE', 'SUSPICIOUS', 'MALICIOUS')),
    risk_score DOUBLE PRECISION NOT NULL 
        CHECK (risk_score >= 0.0 AND risk_score <= 1.0),
    detected_indicators JSONB DEFAULT '[]'::jsonb,
    model_version TEXT NOT NULL DEFAULT 'v1.0.0',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- Index for joining analysis with parent scan record
CREATE INDEX IF NOT EXISTS idx_threat_analysis_scan_id 
    ON public.threat_analysis(scan_id);

-- -----------------------------------------------------------------------------
-- Table 3: prevention_decisions
-- Records the adaptive threat prevention action chosen by the RL agent
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.prevention_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID NOT NULL REFERENCES public.scan_history(id) ON DELETE CASCADE,
    selected_action TEXT NOT NULL 
        CHECK (selected_action IN ('ALLOW', 'WARN_USER', 'DEEP_INSPECT', 'BLOCK')),
    decision_reason TEXT NOT NULL,
    policy_version TEXT NOT NULL DEFAULT 'v1.0.0',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- Index for fast lookup by scan ID
CREATE INDEX IF NOT EXISTS idx_prevention_decisions_scan_id 
    ON public.prevention_decisions(scan_id);

-- Compatibility View: prevention_decision (singular alias)
CREATE OR REPLACE VIEW public.prevention_decision AS
SELECT * FROM public.prevention_decisions;

-- -----------------------------------------------------------------------------
-- Table 4: rl_feedback
-- Stores user feedback, ground-truth outcomes, and reward signals for RL training
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.rl_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID NOT NULL REFERENCES public.scan_history(id) ON DELETE CASCADE,
    state_label TEXT NOT NULL,
    selected_action TEXT NOT NULL 
        CHECK (selected_action IN ('ALLOW', 'WARN_USER', 'DEEP_INSPECT', 'BLOCK')),
    reward DOUBLE PRECISION NOT NULL,
    feedback_source TEXT NOT NULL DEFAULT 'USER' 
        CHECK (feedback_source IN ('USER', 'AUTOMATED', 'ADMIN')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- Indexes for feedback analytics and RL batch extraction
CREATE INDEX IF NOT EXISTS idx_rl_feedback_scan_id 
    ON public.rl_feedback(scan_id);
CREATE INDEX IF NOT EXISTS idx_rl_feedback_created_at 
    ON public.rl_feedback(created_at DESC);

-- -----------------------------------------------------------------------------
-- Table 5: user_roles
-- Stores RBAC role assignments linked to auth.users for SecOps administration
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

-- -----------------------------------------------------------------------------
-- Table 6: user_profiles
-- Stores user profile identity metadata (full name, account status)
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

-- =============================================================================
-- Row Level Security (RLS) Configuration (100% Idempotent)
-- =============================================================================

-- Enable Row Level Security (RLS) on all tables
ALTER TABLE public.scan_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.threat_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.prevention_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rl_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;

-- 1. user_roles policies: Users can read their own role; service_role manages roles
DROP POLICY IF EXISTS "Users can read own role" ON public.user_roles;
CREATE POLICY "Users can read own role" 
    ON public.user_roles FOR SELECT 
    TO authenticated 
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Service role manages user roles" ON public.user_roles;
CREATE POLICY "Service role manages user roles" 
    ON public.user_roles FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

-- 2. user_profiles policies: Users read and update their own name; service_role manages profiles
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

DROP POLICY IF EXISTS "Service role manages all user profiles" ON public.user_profiles;
CREATE POLICY "Service role manages all user profiles" 
    ON public.user_profiles FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

-- 3. scan_history policies: Users view own scans; service_role has full access
DROP POLICY IF EXISTS "Users can view own scan history" ON public.scan_history;
CREATE POLICY "Users can view own scan history" 
    ON public.scan_history FOR SELECT 
    TO authenticated 
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Allow service role full access to scan_history" ON public.scan_history;
CREATE POLICY "Allow service role full access to scan_history" 
    ON public.scan_history FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

-- 4. threat_analysis policies: Service role has full access
DROP POLICY IF EXISTS "Allow service role full access to threat_analysis" ON public.threat_analysis;
CREATE POLICY "Allow service role full access to threat_analysis" 
    ON public.threat_analysis FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

-- 5. prevention_decisions policies: Service role has full access
DROP POLICY IF EXISTS "Allow service role full access to prevention_decisions" ON public.prevention_decisions;
CREATE POLICY "Allow service role full access to prevention_decisions" 
    ON public.prevention_decisions FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

-- 6. rl_feedback policies: Service role has full access
DROP POLICY IF EXISTS "Allow service role full access to rl_feedback" ON public.rl_feedback;
CREATE POLICY "Allow service role full access to rl_feedback" 
    ON public.rl_feedback FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

