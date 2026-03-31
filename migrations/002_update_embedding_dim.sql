-- Migration 002: change patients.embedding from vector(768) to vector(512)
-- Required for voyage-3.5-lite which outputs 512-dim vectors natively.
-- Drops and recreates the HNSW index. Nulls out any existing embeddings
-- (they were generated with a different model/dim and must be regenerated).

-- Drop the old HNSW index first (can't alter type with index in place)
DROP INDEX IF EXISTS patients_embedding_idx;

-- Change column type — this nulls all existing embeddings
ALTER TABLE patients
  ALTER COLUMN embedding TYPE vector(512)
  USING NULL;

-- Recreate HNSW index for 512-dim cosine search
CREATE INDEX patients_embedding_idx
  ON patients USING hnsw (embedding vector_cosine_ops);
