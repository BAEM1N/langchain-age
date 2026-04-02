-- Enable Apache AGE extension
CREATE EXTENSION IF NOT EXISTS age;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Load AGE into search path
LOAD 'age';
ALTER DATABASE postgres SET search_path = ag_catalog, "$user", public;
