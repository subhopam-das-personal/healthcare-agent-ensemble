"""Unit tests for shared/fhir_client.py"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import shared.fhir_client as fhir_module
from shared.fhir_client import (
    _extract_entries,
    _parse_patient,
    _parse_condition,
    _parse_medication,
    _parse_observation,
    _parse_allergy,
    get_patient_data,
    DEFAULT_FHIR_BASE_URL,
)


@pytest.fixture(autouse=True)
def clear_fhir_cache():
    """Clear the in-memory FHIR patient cache before every test.

    Without this, a successful fetch cached under 'smart-1234567' in one test
    would be returned by subsequent tests that mock _fhir_get differently,
    causing those tests to see the cached (wrong) result.
    """
    fhir_module._PATIENT_CACHE.clear()
    yield
    fhir_module._PATIENT_CACHE.clear()


# --------------------------------------------------------------------------- #
# _extract_entries
# --------------------------------------------------------------------------- #

class TestExtractEntries:
    def test_empty_bundle_returns_empty_list(self):
        bundle = {"resourceType": "Bundle", "entry": [], "link": []}
        entries, truncated = _extract_entries(bundle)
        assert entries == []
        assert truncated is False

    def test_error_bundle_returns_empty(self):
        bundle = {"error": "server unavailable"}
        entries, truncated = _extract_entries(bundle)
        assert entries == []
        assert truncated is False

    def test_entries_extracted_correctly(self):
        bundle = {
            "entry": [
                {"resource": {"id": "1", "resourceType": "Condition"}},
                {"resource": {"id": "2", "resourceType": "Condition"}},
            ],
            "link": [],
        }
        entries, truncated = _extract_entries(bundle)
        assert len(entries) == 2
        assert entries[0]["id"] == "1"
        assert truncated is False

    def test_pagination_link_detected(self):
        bundle = {
            "entry": [{"resource": {"id": "1"}}],
            "link": [{"relation": "next", "url": "http://next"}],
        }
        entries, truncated = _extract_entries(bundle)
        assert truncated is True

    def test_self_link_not_truncation(self):
        bundle = {
            "entry": [{"resource": {"id": "1"}}],
            "link": [{"relation": "self", "url": "http://self"}],
        }
        entries, truncated = _extract_entries(bundle)
        assert truncated is False

    def test_entry_without_resource_key(self):
        bundle = {"entry": [{}], "link": []}
        entries, truncated = _extract_entries(bundle)
        assert entries == [{}]

    def test_bundle_without_link_key(self):
        bundle = {"entry": [{"resource": {"id": "x"}}]}
        entries, truncated = _extract_entries(bundle)
        assert truncated is False


# --------------------------------------------------------------------------- #
# _parse_patient
# --------------------------------------------------------------------------- #

class TestParsePatient:
    def test_full_patient(self):
        resource = {
            "id": "pat-1",
            "name": [{"given": ["Jane", "M"], "family": "Doe"}],
            "gender": "female",
            "birthDate": "1990-01-15",
            "maritalStatus": {"text": "Married"},
        }
        p = _parse_patient(resource)
        assert p["id"] == "pat-1"
        assert p["name"] == "Jane M Doe"
        assert p["gender"] == "female"
        assert p["birthDate"] == "1990-01-15"
        assert p["maritalStatus"] == "Married"

    def test_missing_name_fields(self):
        resource = {"id": "pat-2", "name": [{}]}
        p = _parse_patient(resource)
        assert p["name"] == ""

    def test_single_given_name(self):
        resource = {
            "id": "pat-3",
            "name": [{"given": ["Bob"], "family": "Smith"}],
        }
        p = _parse_patient(resource)
        assert p["name"] == "Bob Smith"

    def test_no_marital_status(self):
        resource = {"id": "pat-4", "name": [{"given": ["A"], "family": "B"}]}
        p = _parse_patient(resource)
        assert p["maritalStatus"] is None


# --------------------------------------------------------------------------- #
# _parse_condition
# --------------------------------------------------------------------------- #

class TestParseCondition:
    def test_full_condition(self):
        resource = {
            "code": {
                "coding": [{"code": "44054006", "system": "http://snomed.info/sct", "display": "Type 2 diabetes"}]
            },
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "onsetDateTime": "2015-03-10",
        }
        c = _parse_condition(resource)
        assert c["code"] == "44054006"
        assert c["display"] == "Type 2 diabetes"
        assert c["clinicalStatus"] == "active"
        assert c["onsetDateTime"] == "2015-03-10"

    def test_falls_back_to_text_display(self):
        resource = {
            "code": {"coding": [{}], "text": "Hypertension"},
        }
        c = _parse_condition(resource)
        assert c["display"] == "Hypertension"

    def test_empty_coding_fallback(self):
        resource = {"code": {"coding": [], "text": "Unknown condition"}}
        c = _parse_condition({"code": {"coding": [{}], "text": "Unknown condition"}})
        assert c["display"] == "Unknown condition"


# --------------------------------------------------------------------------- #
# _parse_medication
# --------------------------------------------------------------------------- #

class TestParseMedication:
    def test_full_medication(self):
        resource = {
            "medicationCodeableConcept": {
                "coding": [{"code": "1049502", "system": "http://www.nlm.nih.gov/research/umls/rxnorm", "display": "Metformin 500 MG"}],
            },
            "status": "active",
            "authoredOn": "2023-06-01",
        }
        m = _parse_medication(resource)
        assert m["code"] == "1049502"
        assert m["display"] == "Metformin 500 MG"
        assert m["status"] == "active"

    def test_falls_back_to_text_display(self):
        resource = {
            "medicationCodeableConcept": {"coding": [{}], "text": "Lisinopril"},
        }
        m = _parse_medication(resource)
        assert m["display"] == "Lisinopril"


# --------------------------------------------------------------------------- #
# _parse_observation
# --------------------------------------------------------------------------- #

class TestParseObservation:
    def test_numeric_observation(self):
        resource = {
            "code": {
                "coding": [{"code": "8480-6", "system": "http://loinc.org", "display": "Systolic blood pressure"}]
            },
            "valueQuantity": {"value": 130, "unit": "mmHg"},
            "effectiveDateTime": "2024-01-01T10:00:00Z",
        }
        o = _parse_observation(resource)
        assert o["code"] == "8480-6"
        assert o["value"] == 130
        assert o["unit"] == "mmHg"

    def test_missing_value_quantity(self):
        resource = {
            "code": {"coding": [{"code": "x", "display": "Lab"}]},
        }
        o = _parse_observation(resource)
        assert o["value"] is None
        assert o["unit"] is None


# --------------------------------------------------------------------------- #
# _parse_allergy
# --------------------------------------------------------------------------- #

class TestParseAllergy:
    def test_full_allergy(self):
        resource = {
            "code": {
                "coding": [{"code": "372687004", "system": "http://snomed.info/sct", "display": "Penicillin"}]
            },
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "type": "allergy",
            "criticality": "high",
        }
        a = _parse_allergy(resource)
        assert a["display"] == "Penicillin"
        assert a["clinicalStatus"] == "active"
        assert a["type"] == "allergy"
        assert a["criticality"] == "high"


# --------------------------------------------------------------------------- #
# get_patient_data (integration-level with mocked HTTP)
# --------------------------------------------------------------------------- #

PATIENT_FIXTURE = {
    "id": "smart-1234567",
    "name": [{"given": ["John"], "family": "Doe"}],
    "gender": "male",
    "birthDate": "1970-05-20",
}

EMPTY_BUNDLE = {"resourceType": "Bundle", "entry": [], "link": []}


@pytest.mark.asyncio
async def test_get_patient_data_success():
    """Happy path: valid patient and all sub-bundles return empty results."""
    with patch("shared.fhir_client._fhir_get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [
            PATIENT_FIXTURE,            # Patient demographics
            EMPTY_BUNDLE,               # Conditions
            EMPTY_BUNDLE,               # MedicationRequests
            EMPTY_BUNDLE,               # Observations
            EMPTY_BUNDLE,               # AllergyIntolerances
        ]
        result = await get_patient_data("smart-1234567")

    assert "error" not in result
    assert result["patient"]["id"] == "smart-1234567"
    assert result["conditions"] == []
    assert result["medications"] == []
    assert result["observations"] == []
    assert result["allergies"] == []


@pytest.mark.asyncio
async def test_get_patient_data_patient_not_found():
    """When patient fetch fails, return error dict."""
    with patch("shared.fhir_client._fhir_get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"error": "404 Not Found"}
        result = await get_patient_data("nonexistent-patient")

    assert "error" in result
    assert "nonexistent-patient" in result["error"] or "Could not fetch patient" in result["error"]


@pytest.mark.asyncio
async def test_get_patient_data_partial_failure_generates_warning():
    """If a sub-bundle fails, result includes a warning but still returns patient data."""
    with patch("shared.fhir_client._fhir_get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [
            PATIENT_FIXTURE,
            {"error": "Conditions service unavailable"},
            EMPTY_BUNDLE,
            EMPTY_BUNDLE,
            EMPTY_BUNDLE,
        ]
        result = await get_patient_data("smart-1234567")

    assert "error" not in result
    assert "warnings" in result
    assert any("Conditions" in w for w in result["warnings"])


@pytest.mark.asyncio
async def test_get_patient_data_observations_capped_at_50():
    """Observations are limited to the 50 most recent even if more are returned."""
    obs_entries = [
        {"resource": {
            "code": {"coding": [{"code": str(i), "display": f"Obs {i}"}]},
            "valueQuantity": {"value": i, "unit": "mg"},
        }}
        for i in range(80)
    ]
    obs_bundle = {"entry": obs_entries, "link": []}

    with patch("shared.fhir_client._fhir_get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [
            PATIENT_FIXTURE,
            EMPTY_BUNDLE,
            EMPTY_BUNDLE,
            obs_bundle,
            EMPTY_BUNDLE,
        ]
        result = await get_patient_data("smart-1234567")

    assert len(result["observations"]) == 50


@pytest.mark.asyncio
async def test_get_patient_data_uses_custom_fhir_url():
    """Custom fhir_base_url is forwarded to HTTP calls."""
    custom_url = "https://my-fhir.example.com"
    with patch("shared.fhir_client._fhir_get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [
            PATIENT_FIXTURE,
            EMPTY_BUNDLE,
            EMPTY_BUNDLE,
            EMPTY_BUNDLE,
            EMPTY_BUNDLE,
        ]
        await get_patient_data("p1", fhir_base_url=custom_url)

    # First call should be to custom URL / Patient / p1
    first_call_url = mock_get.call_args_list[0][0][0]
    assert first_call_url.startswith(custom_url)


@pytest.mark.asyncio
async def test_get_patient_data_truncation_warning():
    """When a bundle has a 'next' link, a truncation warning is added."""
    paginated_bundle = {
        "entry": [{"resource": {"code": {"coding": [{"code": "x", "display": "X"}]}, "clinicalStatus": {"coding": [{}]}}}],
        "link": [{"relation": "next", "url": "http://next-page"}],
    }
    with patch("shared.fhir_client._fhir_get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [
            PATIENT_FIXTURE,
            paginated_bundle,       # Conditions truncated
            EMPTY_BUNDLE,
            EMPTY_BUNDLE,
            EMPTY_BUNDLE,
        ]
        result = await get_patient_data("p1")

    assert "warnings" in result
    assert any("Conditions" in w for w in result["warnings"])
