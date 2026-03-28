"""SQLAlchemy ORM models for the DDM FHIR NL Query Engine.

Mirrors migrations/001_ddm_schema.sql exactly.
Run the migration first; these models are read/write helpers, not DDL.
"""

from sqlalchemy import (
    Boolean, Column, Date, Float, ForeignKey, Integer, Text, ARRAY,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMPTZ
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class FhirSource(Base):
    __tablename__ = "fhir_sources"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    base_url = Column(Text, nullable=False)
    auth_type = Column(Text, default="none")       # 'none' | 'oauth2' | 'bearer'
    auth_config = Column(JSONB, default=dict)       # {client_id, client_secret, token_url}
    active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMPTZ)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Text, primary_key=True)            # FHIR patient id
    source_id = Column(Integer, ForeignKey("fhir_sources.id"))
    given_name = Column(Text)
    family_name = Column(Text)
    birth_date = Column(Date)
    gender = Column(Text)
    raw_fhir = Column(JSONB)
    # embedding vector(768) populated in Phase 3 (MedCPT) — declared in SQL migration
    indexed_at = Column(TIMESTAMPTZ)


class PatientCondition(Base):
    __tablename__ = "patient_conditions"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Text, ForeignKey("patients.id", ondelete="CASCADE"))
    icd10_code = Column(Text)
    snomed_code = Column(Text)
    display = Column(Text)
    onset_date = Column(Date)
    clinical_status = Column(Text, default="active")
    snomed_ancestors = Column(ARRAY(Text))         # populated by enricher
    icd10_chapter = Column(Text)
    indexed_at = Column(TIMESTAMPTZ)


class PatientMedication(Base):
    __tablename__ = "patient_medications"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Text, ForeignKey("patients.id", ondelete="CASCADE"))
    rxnorm_code = Column(Text)
    display = Column(Text)
    status = Column(Text, default="active")
    drug_class = Column(Text)                      # populated by enricher
    drug_class_rxcui = Column(Text)
    indexed_at = Column(TIMESTAMPTZ)


class PatientObservation(Base):
    __tablename__ = "patient_observations"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Text, ForeignKey("patients.id", ondelete="CASCADE"))
    loinc_code = Column(Text)
    display = Column(Text)
    value_quantity = Column(Float)
    value_unit = Column(Text)                      # always canonical unit
    value_string = Column(Text)
    observation_date = Column(Date)
    indexed_at = Column(TIMESTAMPTZ)


class LoincUnitMap(Base):
    __tablename__ = "loinc_unit_map"

    loinc_code = Column(Text, primary_key=True)
    canonical_unit = Column(Text)
    conversion_factor = Column(Float, default=1.0)
    notes = Column(Text)


class OntologyEdge(Base):
    __tablename__ = "ontology_edges"

    child_code = Column(Text, primary_key=True)
    parent_code = Column(Text, primary_key=True)
    child_display = Column(Text)
    parent_display = Column(Text)
    depth = Column(Integer, default=1)


class OntologyCache(Base):
    __tablename__ = "ontology_cache"

    snomed_code = Column(Text, primary_key=True)
    ancestors = Column(JSONB)                      # [{code, display, depth}]
    fetched_at = Column(TIMESTAMPTZ)


class DrugClassMap(Base):
    __tablename__ = "drug_class_map"

    rxnorm_code = Column(Text, primary_key=True)
    drug_class = Column(Text)
    drug_class_rxcui = Column(Text)
    updated_at = Column(TIMESTAMPTZ)


class IndexJob(Base):
    __tablename__ = "index_jobs"

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("fhir_sources.id"))
    status = Column(Text, default="running")       # 'running' | 'done' | 'error' | 'partial'
    last_page_url = Column(Text)                   # resume point after interruption
    patients_fetched = Column(Integer, default=0)
    patients_indexed = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(TIMESTAMPTZ)
    finished_at = Column(TIMESTAMPTZ)
