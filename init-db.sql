-- Initialize PostgreSQL with extensions for Luna AI

-- Enable full-text search capabilities
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable JSON operations
CREATE EXTENSION IF NOT EXISTS "hstore";

-- Function for updating the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Set up basic performance optimizations
ALTER SYSTEM SET shared_buffers = '256MB';  -- Adjust based on available memory
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET effective_cache_size = '1GB';  -- Adjust based on available memory
ALTER SYSTEM SET random_page_cost = 1.1;  -- For SSD storage

-- Create schema for analytics if you plan to use it
CREATE SCHEMA IF NOT EXISTS analytics;