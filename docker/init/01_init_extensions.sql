-- ─────────────────────────────────────────────────────────────────────────
-- 01_init_extensions.sql
-- Runs once on first container start (via /docker-entrypoint-initdb.d/)
-- ─────────────────────────────────────────────────────────────────────────

-- 1. Apache AGE — graph engine
CREATE EXTENSION IF NOT EXISTS age;

-- 2. pgvector — vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- 3. pg_trgm — trigram similarity (LIKE optimisation, fuzzy search)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Load AGE shared library and set search path so ag_catalog is always visible
LOAD 'age';
ALTER DATABASE langchain_age SET search_path = ag_catalog, "$user", public;
ALTER DATABASE postgres    SET search_path = ag_catalog, "$user", public;

-- Verify all three extensions are active
DO $$
DECLARE
    missing TEXT := '';
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'age')    THEN missing := missing || ' age';    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN missing := missing || ' vector'; END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') THEN missing := missing || ' pg_trgm'; END IF;
    IF missing <> '' THEN
        RAISE EXCEPTION 'Missing extensions:%', missing;
    END IF;
    RAISE NOTICE 'All extensions loaded: age, vector, pg_trgm';
END;
$$;
