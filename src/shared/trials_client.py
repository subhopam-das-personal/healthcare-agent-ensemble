"""ClinicalTrials.gov REST v2 client — zero-auth, free, public API."""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2"
CTGOV_TIMEOUT = 15.0

_STUDY_FIELDS = ",".join([
    "NCTId", "BriefTitle", "OverallStatus", "Phase",
    "Condition", "EligibilityCriteria", "MinimumAge", "MaximumAge", "Gender",
    "LocationCity", "LocationState", "LocationCountry",
    "BriefSummary", "LeadSponsorName", "StudyType",
    "StartDate", "PrimaryCompletionDate",
])


async def search_trials_by_conditions(
    condition_names: list[str],
    age: Optional[int] = None,
    gender: Optional[str] = None,
    max_results: int = 5,
) -> list[dict]:
    """Search ClinicalTrials.gov for recruiting interventional trials.

    Args:
        condition_names: Condition display names from FHIR (e.g. ["Type 2 diabetes", "Heart failure"])
        age: Patient age in years for eligibility pre-filter (optional)
        gender: "male" or "female" (optional, omit to return all)
        max_results: Max trials to return (capped at 20)

    Returns:
        List of trial summary dicts. Empty list on error or no results.
    """
    if not condition_names:
        return []

    query_str = " OR ".join(condition_names[:3])
    params: dict = {
        "query.cond": query_str,
        "filter.overallStatus": "RECRUITING",
        "filter.studyType": "INTERVENTIONAL",
        "fields": _STUDY_FIELDS,
        "pageSize": min(max_results, 20),
        "format": "json",
    }

    if age is not None:
        params["query.patient"] = (
            f"AREA[MinimumAge]RANGE[MIN,{age}Y] AND AREA[MaximumAge]RANGE[{age}Y,MAX]"
        )

    if gender and gender.lower() in ("male", "female"):
        params["filter.advanced"] = f"AREA[Gender]({gender.upper()} OR ALL)"

    try:
        async with httpx.AsyncClient(timeout=CTGOV_TIMEOUT) as client:
            resp = await client.get(f"{CTGOV_BASE_URL}/studies", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        logger.warning("[search_trials] ClinicalTrials.gov request timed out")
        return []
    except httpx.HTTPStatusError as e:
        logger.error(f"[search_trials] HTTP error {e.response.status_code}")
        return []
    except Exception as e:
        logger.error(f"[search_trials] Request failed: {e}")
        return []

    studies = data.get("studies", [])
    results = []
    for study in studies:
        proto = study.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design_mod = proto.get("designModule", {})
        eligibility_mod = proto.get("eligibilityModule", {})
        conditions_mod = proto.get("conditionsModule", {})
        locations_mod = proto.get("contactsLocationsModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        desc_mod = proto.get("descriptionModule", {})

        locations = locations_mod.get("locations", [])
        cities = list({
            f"{loc.get('city', '')}, {loc.get('state') or loc.get('country', '')}"
            for loc in locations[:10]
            if loc.get("city")
        })[:3]

        results.append({
            "nct_id": id_mod.get("nctId", ""),
            "title": id_mod.get("briefTitle", ""),
            "status": status_mod.get("overallStatus", ""),
            "phase": ", ".join(design_mod.get("phases", [])) or "N/A",
            "conditions": conditions_mod.get("conditions", []),
            "min_age": eligibility_mod.get("minimumAge", ""),
            "max_age": eligibility_mod.get("maximumAge", ""),
            "gender": eligibility_mod.get("sex", "ALL"),
            "eligibility_summary": eligibility_mod.get("eligibilityCriteria", "")[:600],
            "locations": cities,
            "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
            "summary": desc_mod.get("briefSummary", "")[:400],
        })

    logger.info(
        f"[search_trials] {len(results)} trials returned for conditions: {condition_names[:3]}"
    )
    return results


async def get_trial_details(nct_id: str) -> dict:
    """Fetch full protocol for a single ClinicalTrials.gov study.

    Args:
        nct_id: NCT identifier string (e.g. "NCT05123456")

    Returns:
        Full protocol dict, or {"error": "..."} on failure.
    """
    try:
        async with httpx.AsyncClient(timeout=CTGOV_TIMEOUT) as client:
            resp = await client.get(
                f"{CTGOV_BASE_URL}/studies/{nct_id}",
                params={"format": "json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"Trial {nct_id} not found (HTTP {e.response.status_code})"}
    except Exception as e:
        logger.error(f"[get_trial_details] Failed for {nct_id}: {e}")
        return {"error": str(e)}

    proto = data.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design_mod = proto.get("designModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})
    eligibility_mod = proto.get("eligibilityModule", {})
    outcomes_mod = proto.get("outcomesModule", {})
    conditions_mod = proto.get("conditionsModule", {})
    locations_mod = proto.get("contactsLocationsModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

    return {
        "nct_id": id_mod.get("nctId", nct_id),
        "title": id_mod.get("briefTitle", ""),
        "official_title": id_mod.get("officialTitle", ""),
        "status": status_mod.get("overallStatus", ""),
        "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
        "completion_date": status_mod.get("primaryCompletionDateStruct", {}).get("date", ""),
        "phase": ", ".join(design_mod.get("phases", [])) or "N/A",
        "study_type": design_mod.get("studyType", ""),
        "enrollment": design_mod.get("enrollmentInfo", {}).get("count"),
        "conditions": conditions_mod.get("conditions", []),
        "arms": [
            {
                "label": arm.get("label", ""),
                "type": arm.get("type", ""),
                "description": arm.get("description", ""),
            }
            for arm in arms_mod.get("armGroups", [])
        ],
        "primary_outcomes": [
            o.get("measure", "") for o in outcomes_mod.get("primaryOutcomes", [])
        ],
        "eligibility": {
            "criteria": eligibility_mod.get("eligibilityCriteria", ""),
            "min_age": eligibility_mod.get("minimumAge", ""),
            "max_age": eligibility_mod.get("maximumAge", ""),
            "gender": eligibility_mod.get("sex", "ALL"),
            "healthy_volunteers": eligibility_mod.get("healthyVolunteers", ""),
        },
        "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
        "locations": [
            {
                "facility": loc.get("facility", ""),
                "city": loc.get("city", ""),
                "state": loc.get("state", ""),
                "country": loc.get("country", ""),
                "status": loc.get("status", ""),
            }
            for loc in locations_mod.get("locations", [])[:10]
        ],
    }
