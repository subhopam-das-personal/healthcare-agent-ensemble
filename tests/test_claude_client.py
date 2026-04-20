"""Unit tests for shared/zai_client.py — focused on JSON parsing fallbacks and error handling."""

import pytest
import json
from unittest.mock import MagicMock, patch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shared.zai_client import run_ddx_reasoning, run_drug_interaction_reasoning, run_synthesis


def _make_mock_client(text_response: str):
    """Build a mock ZaiClient that returns the given text."""
    # Zai SDK response structure: response.choices[0].message.content
    message = MagicMock()
    message.content = text_response

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


SAMPLE_PATIENT = {
    "patient": {"id": "p1", "name": "Jane Doe"},
    "conditions": [{"display": "Type 2 Diabetes"}],
    "medications": [{"display": "Metformin"}],
    "observations": [],
    "allergies": [],
}

SAMPLE_DDX = {
    "differentials": [{"rank": 1, "diagnosis": "T2DM", "confidence": "High"}],
    "red_flags": [],
    "reasoning_summary": "Consistent with diabetes.",
}

SAMPLE_INTERACTION = {
    "interactions": [],
    "overall_risk_level": "Low",
    "medication_summary": "Single medication.",
}


# --------------------------------------------------------------------------- #
# run_ddx_reasoning
# --------------------------------------------------------------------------- #

class TestRunDdxReasoning:
    def test_returns_parsed_json_on_clean_response(self):
        payload = {"differentials": [{"rank": 1, "diagnosis": "Hypertension"}]}
        with patch("shared.zai_client.get_client", return_value=_make_mock_client(json.dumps(payload))):
            result = run_ddx_reasoning(SAMPLE_PATIENT)
        assert result["differentials"][0]["diagnosis"] == "Hypertension"

    def test_parses_markdown_json_code_block(self):
        payload = {"differentials": [{"rank": 1, "diagnosis": "CKD"}]}
        text = f"```json\n{json.dumps(payload)}\n```"
        with patch("shared.zai_client.get_client", return_value=_make_mock_client(text)):
            result = run_ddx_reasoning(SAMPLE_PATIENT)
        assert result["differentials"][0]["diagnosis"] == "CKD"

    def test_parses_generic_code_block(self):
        payload = {"differentials": [{"rank": 1, "diagnosis": "Anemia"}]}
        text = f"```\n{json.dumps(payload)}\n```"
        with patch("shared.zai_client.get_client", return_value=_make_mock_client(text)):
            result = run_ddx_reasoning(SAMPLE_PATIENT)
        assert result["differentials"][0]["diagnosis"] == "Anemia"

    def test_falls_back_to_raw_response_when_unparseable(self):
        text = "I cannot provide a JSON response for this patient."
        with patch("shared.zai_client.get_client", return_value=_make_mock_client(text)):
            result = run_ddx_reasoning(SAMPLE_PATIENT)
        assert "raw_response" in result
        assert result["raw_response"] == text

    def test_returns_error_dict_on_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API quota exceeded")
        with patch("shared.zai_client.get_client", return_value=client):
            result = run_ddx_reasoning(SAMPLE_PATIENT)
        assert "error" in result
        assert "API quota exceeded" in result["error"]

    def test_symptoms_included_in_request(self):
        payload = {"differentials": []}
        client = _make_mock_client(json.dumps(payload))
        with patch("shared.zai_client.get_client", return_value=client):
            run_ddx_reasoning(SAMPLE_PATIENT, symptoms="chest pain")

        call_args = client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "chest pain" in user_content

    def test_no_api_key_raises_valueerror(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ZAI_API_KEY", None)
            result = run_ddx_reasoning(SAMPLE_PATIENT)
        assert "error" in result


# --------------------------------------------------------------------------- #
# run_drug_interaction_reasoning
# --------------------------------------------------------------------------- #

class TestRunDrugInteractionReasoning:
    def test_returns_parsed_json(self):
        payload = {"interactions": [], "overall_risk_level": "Low"}
        with patch("shared.zai_client.get_client", return_value=_make_mock_client(json.dumps(payload))):
            result = run_drug_interaction_reasoning(SAMPLE_PATIENT)
        assert result["overall_risk_level"] == "Low"

    def test_parses_markdown_code_block(self):
        payload = {"interactions": [{"drug_pair": ["A", "B"], "severity": "High"}]}
        text = f"```json\n{json.dumps(payload)}\n```"
        with patch("shared.zai_client.get_client", return_value=_make_mock_client(text)):
            result = run_drug_interaction_reasoning(SAMPLE_PATIENT)
        assert result["interactions"][0]["severity"] == "High"

    def test_rxnav_interactions_included_in_prompt(self):
        payload = {"interactions": []}
        client = _make_mock_client(json.dumps(payload))
        rxnav = {"interactions": [{"pair": ["A", "B"]}], "source": "rxnav"}

        with patch("shared.zai_client.get_client", return_value=client):
            run_drug_interaction_reasoning(SAMPLE_PATIENT, rxnav_interactions=rxnav)

        call_args = client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "RxNav" in user_content

    def test_proposed_medications_included_in_prompt(self):
        payload = {"interactions": []}
        client = _make_mock_client(json.dumps(payload))

        with patch("shared.zai_client.get_client", return_value=client):
            run_drug_interaction_reasoning(SAMPLE_PATIENT, proposed_medications=["warfarin"])

        call_args = client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "warfarin" in user_content

    def test_returns_error_on_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("timeout")
        with patch("shared.zai_client.get_client", return_value=client):
            result = run_drug_interaction_reasoning(SAMPLE_PATIENT)
        assert "error" in result

    def test_no_rxnav_adds_ai_only_note(self):
        """Without rxnav data, prompt should note AI-only analysis."""
        payload = {"interactions": []}
        client = _make_mock_client(json.dumps(payload))

        with patch("shared.zai_client.get_client", return_value=client):
            run_drug_interaction_reasoning(SAMPLE_PATIENT, rxnav_interactions=None)

        call_args = client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "AI-generated" in user_content or "AI-only" in user_content


# --------------------------------------------------------------------------- #
# run_synthesis
# --------------------------------------------------------------------------- #

class TestRunSynthesis:
    def test_returns_parsed_json(self):
        payload = {"assessment_summary": "Patient is stable."}
        with patch("shared.zai_client.get_client", return_value=_make_mock_client(json.dumps(payload))):
            result = run_synthesis(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION)
        assert result["assessment_summary"] == "Patient is stable."

    def test_parses_markdown_code_block(self):
        payload = {"key_findings": ["Finding A"]}
        text = f"```json\n{json.dumps(payload)}\n```"
        with patch("shared.zai_client.get_client", return_value=_make_mock_client(text)):
            result = run_synthesis(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION)
        assert result["key_findings"][0] == "Finding A"

    def test_care_gaps_included_when_provided(self):
        payload = {"assessment_summary": "done"}
        client = _make_mock_client(json.dumps(payload))
        care_gaps = {"gaps": ["Missing statin"]}

        with patch("shared.zai_client.get_client", return_value=client):
            run_synthesis(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION, care_gaps=care_gaps)

        call_args = client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "Missing statin" in user_content

    def test_care_gaps_omitted_when_none(self):
        payload = {"assessment_summary": "done"}
        client = _make_mock_client(json.dumps(payload))

        with patch("shared.zai_client.get_client", return_value=client):
            run_synthesis(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION, care_gaps=None)

        call_args = client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "Care Gap" not in user_content

    def test_returns_error_on_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("model overloaded")
        with patch("shared.zai_client.get_client", return_value=client):
            result = run_synthesis(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION)
        assert "error" in result
        assert "model overloaded" in result["error"]
