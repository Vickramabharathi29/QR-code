-- =============================================================================
-- Migration: 20261018_user_profiles_and_scan_association.sql
-- Project: QRShield Cybersecurity Operations Center
-- Description: 
--   1. Adds user_id and destination_domain columns to scan_history.
--   2. Creates public.user_profiles table for user identities.
--   3. Configures Row Level Security (RLS) for profile reads, updates, and scan records.
-- Execute this migration in your Supabase SQL Editor:
-- https://supabase.com/dashboard/project/tjfhuulhzvlkqvjkevki/sql
-- =============================================================================

-- 1. Extend scan_history table with optional user_id and destination_domain
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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_scan_history_user_id ON public.scan_history(user_id);
CREATE INDEX IF NOT EXISTS idx_scan_history_domain ON public.scan_history(destination_domain);

-- 2. Create public.user_profiles table
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

-- Index on user_id and email
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON public.user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON public.user_profiles(email);

-- 3. Enable Row Level Security (RLS) on user_profiles
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;

-- 4. RLS Policies for user_profiles
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

-- 5. Scan History User Association Policy
DROP POLICY IF EXISTS "Users can view own scan history" ON public.scan_history;
CREATE POLICY "Users can view own scan history"
    ON public.scan_history FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);
