"""FHIR client for fetching patient data from SMART Health IT sandbox."""

import httpx
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_FHIR_BASE_URL = "https://r4.smarthealthit.org"
FHIR_TIMEOUT = 10.0
FHIR_MAX_COUNT = 100

# In-memory patient cache: key=(patient_id, fhir_base_url), value=(data, expires_at)
# TTL of 5 minutes — prevents redundant FHIR fetches when an LLM agent calls
# the same tool multiple times in one session.
_PATIENT_CACHE: dict[tuple, tuple] = {}
_CACHE_TTL = 300  # seconds


async def _fhir_get(
    url: str,
    access_token: Optional[str] = None,
    params: Optional[dict] = None,
) -> dict:
    headers = {"Accept": "application/fhir+json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    async with httpx.AsyncClient(timeout=FHIR_TIMEOUT) as client:
        try:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            logger.warning(f"FHIR request timed out: {url}")
            # One retry
            try:
                resp = await client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"error": f"FHIR server timeout after retry: {str(e)}"}
        except httpx.HTTPStatusError as e:
            return {"error": f"FHIR server error {e.response.status_code}: {str(e)}"}
        except Exception as e:
            return {"error": f"FHIR request failed: {str(e)}"}


def _extract_entries(bundle: dict) -> list[dict]:
    if "error" in bundle:
        return []
    entries = [e.get("resource", {}) for e in bundle.get("entry", [])]
    truncated = "link" in bundle and any(
        l.get("relation") == "next" for l in bundle.get("link", [])
    )
    return entries, truncated if isinstance(entries, list) else (entries, False)


def _extract_entries(bundle: dict) -> tuple[list[dict], bool]:
    if "error" in bundle:
        return [], False
    entries = [e.get("resource", {}) for e in bundle.get("entry", [])]
    truncated = any(
        l.get("relation") == "next" for l in bundle.get("link", [])
    )
    return entries, truncated


def _parse_patient(resource: dict) -> dict:
    name_parts = resource.get("name", [{}])[0]
    given = " ".join(name_parts.get("given", []))
    family = name_parts.get("family", "")
    return {
        "id": resource.get("id"),
        "name": f"{given} {family}".strip(),
        "gender": resource.get("gender"),
        "birthDate": resource.get("birthDate"),
        "maritalStatus": resource.get("maritalStatus", {}).get("text"),
    }


def _parse_condition(resource: dict) -> dict:
    coding = resource.get("code", {}).get("coding", [{}])[0]
    return {
        "code": coding.get("code"),
        "system": coding.get("system", ""),
        "display": coding.get("display", resource.get("code", {}).get("text", "Unknown")),
        "clinicalStatus": resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
        "onsetDateTime": resource.get("onsetDateTime"),
    }


def _parse_medication(resource: dict) -> dict:
    med = resource.get("medicationCodeableConcept", {})
    coding = med.get("coding", [{}])[0]
    return {
        "code": coding.get("code"),
        "system": coding.get("system", ""),
        "display": coding.get("display", med.get("text", "Unknown")),
        "status": resource.get("status"),
        "authoredOn": resource.get("authoredOn"),
    }


def _parse_observation(resource: dict) -> dict:
    coding = resource.get("code", {}).get("coding", [{}])[0]
    value = resource.get("valueQuantity", {})
    return {
        "code": coding.get("code"),
        "system": coding.get("system", ""),
        "display": coding.get("display", resource.get("code", {}).get("text", "Unknown")),
        "value": value.get("value"),
        "unit": value.get("unit"),
        "effectiveDateTime": resource.get("effectiveDateTime"),
    }


def _parse_allergy(resource: dict) -> dict:
    coding = resource.get("code", {}).get("coding", [{}])[0]
    return {
        "code": coding.get("code"),
        "system": coding.get("system", ""),
        "display": coding.get("display", resource.get("code", {}).get("text", "Unknown")),
        "clinicalStatus": resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
        "type": resource.get("type"),
        "criticality": resource.get("criticality"),
    }


async def get_patient_data(
    patient_id: str,
    fhir_base_url: str = DEFAULT_FHIR_BASE_URL,
    access_token: Optional[str] = None,
    patient_json: str = "",
) -> dict:
    """Fetch comprehensive patient data from FHIR server (cached 5 min).

    If patient_json is provided, it is parsed directly and the FHIR server is
    not contacted. Expects a FHIR R4 Bundle with Patient, Condition,
    MedicationRequest, Observation, and AllergyIntolerance entries.
    """
    if patient_json:
        try:
            bundle = json.loads(patient_json)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid patient_json: {e}"}
        entries = [e.get("resource", {}) for e in bundle.get("entry", [])]
        patient_resources = [r for r in entries if r.get("resourceType") == "Patient"]
        if not patient_resources:
            return {"error": "patient_json bundle contains no Patient resource"}
        patient = _parse_patient(patient_resources[0])
        conditions = [_parse_condition(r) for r in entries if r.get("resourceType") == "Condition"]
        medications = [_parse_medication(r) for r in entries if r.get("resourceType") == "MedicationRequest"]
        observations = [_parse_observation(r) for r in entries if r.get("resourceType") == "Observation"]
        allergies = [_parse_allergy(r) for r in entries if r.get("resourceType") == "AllergyIntolerance"]
        return {
            "patient": patient,
            "conditions": conditions,
            "medications": medications,
            "observations": observations[:50],
            "allergies": allergies,
        }

    cache_key = (patient_id, fhir_base_url.rstrip("/"))
    cached = _PATIENT_CACHE.get(cache_key)
    if cached:
        data, expires_at = cached
        if time.monotonic() < expires_at:
            logger.info(f"[get_patient_data] Cache hit for patient {patient_id}")
            return data
        del _PATIENT_CACHE[cache_key]

    base = fhir_base_url.rstrip("/")
    warnings = []

    # Fetch patient demographics
    patient_raw = await _fhir_get(f"{base}/Patient/{patient_id}", access_token)
    if "error" in patient_raw:
        return {"error": f"Could not fetch patient: {patient_raw['error']}"}
    patient = _parse_patient(patient_raw)

    # Fetch related resources in parallel
    import asyncio
    results = await asyncio.gather(
        _fhir_get(f"{base}/Condition", access_token, {"patient": patient_id, "_count": FHIR_MAX_COUNT}),
        _fhir_get(f"{base}/MedicationRequest", access_token, {"patient": patient_id, "_count": FHIR_MAX_COUNT}),
        _fhir_get(f"{base}/Observation", access_token, {"patient": patient_id, "_count": FHIR_MAX_COUNT, "_sort": "-date"}),
        _fhir_get(f"{base}/AllergyIntolerance", access_token, {"patient": patient_id, "_count": FHIR_MAX_COUNT}),
    )

    conditions_raw, meds_raw, obs_raw, allergies_raw = results

    for name, bundle in [("Conditions", conditions_raw), ("Medications", meds_raw),
                          ("Observations", obs_raw), ("Allergies", allergies_raw)]:
        if "error" in bundle:
            warnings.append(f"Failed to fetch {name}: {bundle['error']}")

    conditions, cond_trunc = _extract_entries(conditions_raw)
    medications, med_trunc = _extract_entries(meds_raw)
    observations, obs_trunc = _extract_entries(obs_raw)
    allergies, allergy_trunc = _extract_entries(allergies_raw)

    for name, trunc in [("Conditions", cond_trunc), ("Medications", med_trunc),
                         ("Observations", obs_trunc), ("Allergies", allergy_trunc)]:
        if trunc:
            warnings.append(f"Only first {FHIR_MAX_COUNT} {name} retrieved; more exist.")

    result = {
        "patient": patient,
        "conditions": [_parse_condition(c) for c in conditions],
        "medications": [_parse_medication(m) for m in medications],
        "observations": [_parse_observation(o) for o in observations[:50]],  # limit to most recent 50
        "allergies": [_parse_allergy(a) for a in allergies],
    }
    if warnings:
        result["warnings"] = warnings

    _PATIENT_CACHE[cache_key] = (result, time.monotonic() + _CACHE_TTL)
    return result
