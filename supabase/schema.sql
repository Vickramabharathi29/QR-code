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
        CHECK (scan_status IN ('PENDING', 'PROCESSED', 'FAILED'))
);

-- Index for temporal ordering and dashboard queries
CREATE INDEX IF NOT EXISTS idx_scan_history_created_at 
    ON public.scan_history(created_at DESC);

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

-- =============================================================================
-- Row Level Security (RLS) Configuration
-- =============================================================================

-- Enable Row Level Security (RLS) on all tables
ALTER TABLE public.scan_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.threat_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.prevention_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rl_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;

-- user_roles policies: Users can only read their own role. Only service_role can write.
CREATE POLICY "Users can read own role" 
    ON public.user_roles FOR SELECT 
    TO authenticated 
    USING (auth.uid() = user_id);

CREATE POLICY "Service role manages user roles" 
    ON public.user_roles FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

-- Scan telemetry policies: Service role has full access; anon users cannot view global history.
CREATE POLICY "Allow service role full access to scan_history" 
    ON public.scan_history FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

CREATE POLICY "Allow service role full access to threat_analysis" 
    ON public.threat_analysis FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

CREATE POLICY "Allow service role full access to prevention_decisions" 
    ON public.prevention_decisions FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

CREATE POLICY "Allow service role full access to rl_feedback" 
    ON public.rl_feedback FOR ALL 
    TO service_role 
    USING (true) 
    WITH CHECK (true);

