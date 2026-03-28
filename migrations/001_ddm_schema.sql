-- DDM Schema: FHIR NL Query Engine
-- Run once on Railway Postgres (pgvector 0.8.x required)

CREATE EXTENSION IF NOT EXISTS vector;

-- Source registry: config-driven FHIR connectors
CREATE TABLE IF NOT EXISTS fhir_sources (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    auth_type TEXT DEFAULT 'none',       -- 'none' | 'oauth2' | 'bearer'
    auth_config JSONB DEFAULT '{}',      -- {client_id, client_secret, token_url}
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Patient cohort (normalized from FHIR)
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,                  -- FHIR patient id
    source_id INT REFERENCES fhir_sources(id),
    given_name TEXT,
    family_name TEXT,
    birth_date DATE,
    gender TEXT,
    raw_fhir JSONB,
    embedding vector(768),               -- MedCPT embedding of condition+med profile
    indexed_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS patients_embedding_idx ON patients USING hnsw (embedding vector_cosine_ops);

-- Conditions (one row per condition per patient)
CREATE TABLE IF NOT EXISTS patient_conditions (
    id SERIAL PRIMARY KEY,
    patient_id TEXT REFERENCES patients(id) ON DELETE CASCADE,
    icd10_code TEXT,
    snomed_code TEXT,
    display TEXT,
    onset_date DATE,
    clinical_status TEXT DEFAULT 'active',
    snomed_ancestors TEXT[],             -- filled at index time via ontology_cache
    icd10_chapter TEXT,
    indexed_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS patient_conditions_patient_id ON patient_conditions(patient_id);
CREATE INDEX IF NOT EXISTS patient_conditions_icd10 ON patient_conditions(icd10_code);
CREATE INDEX IF NOT EXISTS patient_conditions_snomed ON patient_conditions(snomed_code);

-- Medications (one row per medication per patient)
CREATE TABLE IF NOT EXISTS patient_medications (
    id SERIAL PRIMARY KEY,
    patient_id TEXT REFERENCES patients(id) ON DELETE CASCADE,
    rxnorm_code TEXT,
    display TEXT,
    status TEXT DEFAULT 'active',
    drug_class TEXT,
    drug_class_rxcui TEXT,
    indexed_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS patient_medications_patient_id ON patient_medications(patient_id);
CREATE INDEX IF NOT EXISTS patient_medications_rxnorm ON patient_medications(rxnorm_code);
CREATE INDEX IF NOT EXISTS patient_medications_drug_class ON patient_medications(drug_class);

-- Observations / labs (normalized units at index time)
CREATE TABLE IF NOT EXISTS patient_observations (
    id SERIAL PRIMARY KEY,
    patient_id TEXT REFERENCES patients(id) ON DELETE CASCADE,
    loinc_code TEXT,
    display TEXT,
    value_quantity FLOAT,
    value_unit TEXT,                     -- always canonical unit
    value_string TEXT,
    observation_date DATE,
    indexed_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS patient_observations_patient_id ON patient_observations(patient_id);
CREATE INDEX IF NOT EXISTS patient_observations_loinc ON patient_observations(loinc_code);

-- LOINC → canonical unit mapping
CREATE TABLE IF NOT EXISTS loinc_unit_map (
    loinc_code TEXT PRIMARY KEY,
    canonical_unit TEXT,
    conversion_factor FLOAT DEFAULT 1.0,
    notes TEXT
);
INSERT INTO loinc_unit_map VALUES
    ('2160-0', 'mg/dL', 1.0,    'Creatinine: multiply µmol/L by 0.0113'),
    ('4548-4', '%',     1.0,    'HbA1c: multiply mmol/mol by 0.0915'),
    ('2345-7', 'mg/dL', 1.0,    'Glucose'),
    ('2093-3', 'mg/dL', 1.0,    'Total Cholesterol'),
    ('2085-9', 'mg/dL', 1.0,    'HDL Cholesterol'),
    ('2089-1', 'mg/dL', 1.0,    'LDL Cholesterol'),
    ('8480-6', 'mmHg',  1.0,    'Systolic BP'),
    ('8462-4', 'mmHg',  1.0,    'Diastolic BP'),
    ('8867-4', '/min',  1.0,    'Heart Rate'),
    ('2947-0', 'mEq/L', 1.0,    'Sodium'),
    ('6298-4', 'mEq/L', 1.0,    'Potassium')
ON CONFLICT (loinc_code) DO NOTHING;

-- SNOMED ontology edges (parent-child hierarchy, populated from NLM tx.fhir.org)
CREATE TABLE IF NOT EXISTS ontology_edges (
    child_code TEXT,
    parent_code TEXT,
    child_display TEXT,
    parent_display TEXT,
    depth INT DEFAULT 1,
    PRIMARY KEY (child_code, parent_code)
);
CREATE INDEX IF NOT EXISTS ontology_edges_parent ON ontology_edges(parent_display);

-- Ontology lookup cache (per-concept SNOMED ancestor results)
CREATE TABLE IF NOT EXISTS ontology_cache (
    snomed_code TEXT PRIMARY KEY,
    ancestors JSONB,                     -- [{code, display, depth}]
    fetched_at TIMESTAMPTZ DEFAULT now()
);

-- Drug class map (RxNorm code → drug class, from RxNav, cached)
CREATE TABLE IF NOT EXISTS drug_class_map (
    rxnorm_code TEXT PRIMARY KEY,
    drug_class TEXT,
    drug_class_rxcui TEXT,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index job tracking (pagination resume support)
CREATE TABLE IF NOT EXISTS index_jobs (
    id SERIAL PRIMARY KEY,
    source_id INT REFERENCES fhir_sources(id),
    status TEXT DEFAULT 'running',       -- 'running' | 'done' | 'error' | 'partial'
    last_page_url TEXT,
    patients_fetched INT DEFAULT 0,
    patients_indexed INT DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMPTZ DEFAULT now(),
    finished_at TIMESTAMPTZ
);

-- Seed default FHIR source (SMART Health IT open sandbox)
INSERT INTO fhir_sources (name, base_url, auth_type)
VALUES ('SMART Health IT', 'https://r4.smarthealthit.org', 'none')
ON CONFLICT DO NOTHING;
