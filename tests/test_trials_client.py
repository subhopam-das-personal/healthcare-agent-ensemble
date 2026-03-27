"""Tests for shared/trials_client.py — response parsing and edge cases."""

import json
import sys
import os
import pytest
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shared.trials_client import search_trials_by_conditions, get_trial_details


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_study(
    nct_id="NCT05000001",
    title="Test Trial",
    status="RECRUITING",
    phases=["PHASE3"],
    conditions=["Type 2 Diabetes Mellitus"],
    min_age="18 Years",
    max_age="75 Years",
    gender="ALL",
    sponsor="Test Sponsor",
    summary="A brief summary.",
    eligibility="Inclusion Criteria:\n- Age 18-75\nExclusion Criteria:\n- eGFR < 30",
    locations=None,
) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": title},
            "statusModule": {"overallStatus": status},
            "designModule": {"phases": phases, "studyType": "INTERVENTIONAL"},
            "conditionsModule": {"conditions": conditions},
            "eligibilityModule": {
                "minimumAge": min_age,
                "maximumAge": max_age,
                "sex": gender,
                "eligibilityCriteria": eligibility,
            },
            "contactsLocationsModule": {
                "locations": locations or [
                    {"city": "Boston", "state": "MA", "country": "USA", "facility": "MGH"}
                ]
            },
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": sponsor}},
            "descriptionModule": {"briefSummary": summary},
        }
    }


def _mock_search_response(studies: list) -> httpx.Response:
    return httpx.Response(
        200,
        json={"studies": studies, "totalCount": len(studies)},
    )


def _mock_detail_response(study: dict) -> httpx.Response:
    return httpx.Response(200, json=study)


# ── search_trials_by_conditions ───────────────────────────────────────────────

class TestSearchTrialsByConditions:
    @pytest.mark.asyncio
    async def test_returns_empty_for_no_conditions(self):
        result = await search_trials_by_conditions([])
        assert result == []

    @pytest.mark.asyncio
    async def test_parses_single_study(self, respx_mock):
        study = _make_study()
        respx_mock.get("https://clinicaltrials.gov/api/v2/studies").mock(
            return_value=_mock_search_response([study])
        )
        results = await search_trials_by_conditions(["Type 2 Diabetes"])
        assert len(results) == 1
        r = results[0]
        assert r["nct_id"] == "NCT05000001"
        assert r["title"] == "Test Trial"
        assert r["status"] == "RECRUITING"
        assert r["phase"] == "PHASE3"
        assert r["sponsor"] == "Test Sponsor"

    @pytest.mark.asyncio
    async def test_parses_multiple_studies(self, respx_mock):
        studies = [
            _make_study(nct_id="NCT05000001", title="Trial A"),
            _make_study(nct_id="NCT05000002", title="Trial B"),
        ]
        respx_mock.get("https://clinicaltrials.gov/api/v2/studies").mock(
            return_value=_mock_search_response(studies)
        )
        results = await search_trials_by_conditions(["Diabetes", "Heart Failure"])
        assert len(results) == 2
        assert results[0]["nct_id"] == "NCT05000001"
        assert results[1]["nct_id"] == "NCT05000002"

    @pytest.mark.asyncio
    async def test_empty_studies_returns_empty_list(self, respx_mock):
        respx_mock.get("https://clinicaltrials.gov/api/v2/studies").mock(
            return_value=_mock_search_response([])
        )
        results = await search_trials_by_conditions(["Rare Disease"])
        assert results == []

    @pytest.mark.asyncio
    async def test_timeout_returns_empty_list(self, respx_mock):
        respx_mock.get("https://clinicaltrials.gov/api/v2/studies").mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        results = await search_trials_by_conditions(["Diabetes"])
        assert results == []

    @pytest.mark.asyncio
    async def test_http_error_returns_empty_list(self, respx_mock):
        respx_mock.get("https://clinicaltrials.gov/api/v2/studies").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        results = await search_trials_by_conditions(["Diabetes"])
        assert results == []

    @pytest.mark.asyncio
    async def test_locations_parsed_and_capped_at_three(self, respx_mock):
        locs = [
            {"city": "Boston", "state": "MA", "country": "USA", "facility": "MGH"},
            {"city": "New York", "state": "NY", "country": "USA", "facility": "NYU"},
            {"city": "Chicago", "state": "IL", "country": "USA", "facility": "UIC"},
            {"city": "Denver", "state": "CO", "country": "USA", "facility": "UCH"},
        ]
        study = _make_study(locations=locs)
        respx_mock.get("https://clinicaltrials.gov/api/v2/studies").mock(
            return_value=_mock_search_response([study])
        )
        results = await search_trials_by_conditions(["Diabetes"])
        assert len(results[0]["locations"]) <= 3

    @pytest.mark.asyncio
    async def test_eligibility_summary_capped_at_600_chars(self, respx_mock):
        long_criteria = "A" * 1000
        study = _make_study(eligibility=long_criteria)
        respx_mock.get("https://clinicaltrials.gov/api/v2/studies").mock(
            return_value=_mock_search_response([study])
        )
        results = await search_trials_by_conditions(["Diabetes"])
        assert len(results[0]["eligibility_summary"]) <= 600

    @pytest.mark.asyncio
    async def test_no_phases_returns_na(self, respx_mock):
        study = _make_study(phases=[])
        respx_mock.get("https://clinicaltrials.gov/api/v2/studies").mock(
            return_value=_mock_search_response([study])
        )
        results = await search_trials_by_conditions(["Diabetes"])
        assert results[0]["phase"] == "N/A"

    @pytest.mark.asyncio
    async def test_only_first_three_conditions_sent_in_query(self, respx_mock):
        captured = {}

        def capture(request):
            captured["params"] = dict(request.url.params)
            return _mock_search_response([])

        respx_mock.get("https://clinicaltrials.gov/api/v2/studies").mock(side_effect=capture)
        await search_trials_by_conditions(["CondAlpha", "CondBeta", "CondGamma", "CondDelta", "CondEpsilon"])
        cond_query = captured["params"].get("query.cond", "")
        assert "CondDelta" not in cond_query
        assert "CondEpsilon" not in cond_query
        assert "CondAlpha" in cond_query


# ── get_trial_details ─────────────────────────────────────────────────────────

class TestGetTrialDetails:
    @pytest.mark.asyncio
    async def test_returns_full_detail(self, respx_mock):
        study = _make_study(nct_id="NCT99000001", title="Full Detail Trial")
        study["protocolSection"]["armsInterventionsModule"] = {
            "armGroups": [{"label": "Arm A", "type": "EXPERIMENTAL", "description": "Drug X"}]
        }
        study["protocolSection"]["outcomesModule"] = {
            "primaryOutcomes": [{"measure": "HbA1c reduction at 24 weeks"}]
        }
        respx_mock.get("https://clinicaltrials.gov/api/v2/studies/NCT99000001").mock(
            return_value=_mock_detail_response(study)
        )
        result = await get_trial_details("NCT99000001")
        assert result["nct_id"] == "NCT99000001"
        assert result["title"] == "Full Detail Trial"
        assert len(result["arms"]) == 1
        assert result["arms"][0]["label"] == "Arm A"
        assert result["primary_outcomes"] == ["HbA1c reduction at 24 weeks"]
        assert "criteria" in result["eligibility"]

    @pytest.mark.asyncio
    async def test_404_returns_error_dict(self, respx_mock):
        respx_mock.get("https://clinicaltrials.gov/api/v2/studies/NCT00000000").mock(
            return_value=httpx.Response(404, text="Not Found")
        )
        result = await get_trial_details("NCT00000000")
        assert "error" in result
        assert "404" in result["error"]

    @pytest.mark.asyncio
    async def test_network_error_returns_error_dict(self, respx_mock):
        respx_mock.get("https://clinicaltrials.gov/api/v2/studies/NCT99999999").mock(
            side_effect=httpx.ConnectError("refused")
        )
        result = await get_trial_details("NCT99999999")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_locations_capped_at_ten(self, respx_mock):
        locs = [
            {"city": f"City{i}", "state": "CA", "country": "USA", "facility": f"Hosp{i}"}
            for i in range(15)
        ]
        study = _make_study(nct_id="NCT11000001", locations=locs)
        study["protocolSection"]["armsInterventionsModule"] = {"armGroups": []}
        study["protocolSection"]["outcomesModule"] = {"primaryOutcomes": []}
        respx_mock.get("https://clinicaltrials.gov/api/v2/studies/NCT11000001").mock(
            return_value=_mock_detail_response(study)
        )
        result = await get_trial_details("NCT11000001")
        assert len(result["locations"]) <= 10
