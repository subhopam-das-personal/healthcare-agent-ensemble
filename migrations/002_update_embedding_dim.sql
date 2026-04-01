-- Migration 002: change patients.embedding from vector(768) to vector(512)
-- Required for voyage-3-lite which outputs 512-dim vectors natively.
-- IDEMPOTENT: wraps the ALTER in a PL/pgSQL block that checks atttypmod first.
-- On subsequent runs the column is already vector(512) so nothing happens
-- and existing embeddings are preserved.

DO $$
BEGIN
  -- Only run if column is NOT already vector(512)
  -- atttypmod stores the declared dimension for pgvector columns
  IF NOT EXISTS (
    SELECT 1 FROM pg_attribute a
    JOIN pg_class c ON a.attrelid = c.oid
    JOIN pg_type t ON a.atttypid = t.oid
    WHERE c.relname = 'patients'
      AND a.attname = 'embedding'
      AND t.typname = 'vector'
      AND a.atttypmod = 512
  ) THEN
    DROP INDEX IF EXISTS patients_embedding_idx;
    ALTER TABLE patients
      ALTER COLUMN embedding TYPE vector(512)
      USING NULL;
    CREATE INDEX patients_embedding_idx
      ON patients USING hnsw (embedding vector_cosine_ops);
  END IF;
END $$;
