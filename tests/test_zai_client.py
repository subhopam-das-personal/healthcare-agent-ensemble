"""Tests for shared/zai_client.py — ZAI SDK integration."""

import json
import os
import sys

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shared.zai_client import (
    ZAI_MODEL,
    get_client,
    run_ddx_reasoning,
    run_drug_interaction_reasoning,
    run_synthesis,
    stream_ddx_tokens,
    stream_drug_interaction_tokens,
    stream_synthesis_tokens,
)
from zhipuai import ZhipuAI


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_client(text_response: str) -> MagicMock:
    """Return a mock ZaiClient whose chat.completions.create returns text_response."""
    message = MagicMock()
    message.content = text_response

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def _make_streaming_client(tokens: list[str]) -> MagicMock:
    """Return a mock ZaiClient that streams the given tokens one chunk at a time."""
    chunks = []
    for token in tokens:
        delta = MagicMock()
        delta.content = token
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]
        chunks.append(chunk)

    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks)
    return client


def _user_message(client: MagicMock) -> str:
    """Extract the user-turn content from the last create() call."""
    msgs = client.chat.completions.create.call_args.kwargs["messages"]
    # messages[0] is the system prompt, messages[1] is the user content
    return msgs[1]["content"]


# ── Shared test data ──────────────────────────────────────────────────────────

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


# ── get_client ────────────────────────────────────────────────────────────────


class TestGetClient:
    def test_raises_when_api_key_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "ZAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="ZAI_API_KEY"):
                get_client()

    def test_raises_when_api_key_empty(self):
        with patch.dict(os.environ, {"ZAI_API_KEY": ""}):
            with pytest.raises(ValueError, match="ZAI_API_KEY"):
                get_client()

    def test_uses_default_base_url(self):
        with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}):
            with patch("shared.zai_client.ZhipuAI") as MockZai:
                get_client()
                MockZai.assert_called_once_with(
                    api_key="test-key",
                    base_url="https://api.z.ai/api/paas/v4",
                    timeout=60.0,
                )

    def test_uses_custom_base_url_from_env(self):
        env = {"ZAI_API_KEY": "test-key", "ZAI_BASE_URL": "https://custom.api/v4"}
        with patch.dict(os.environ, env):
            with patch("shared.zai_client.ZhipuAI") as MockZai:
                get_client()
                MockZai.assert_called_once_with(
                    api_key="test-key",
                    base_url="https://custom.api/v4",
                    timeout=60.0,
                )


# ── run_ddx_reasoning ─────────────────────────────────────────────────────────


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
        text = "I cannot provide a JSON response."
        with patch("shared.zai_client.get_client", return_value=_make_mock_client(text)):
            result = run_ddx_reasoning(SAMPLE_PATIENT)
        assert result == {"raw_response": text}

    def test_returns_error_dict_on_api_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API quota exceeded")
        with patch("shared.zai_client.get_client", return_value=client):
            result = run_ddx_reasoning(SAMPLE_PATIENT)
        assert "error" in result
        assert "API quota exceeded" in result["error"]

    def test_symptoms_appended_to_user_message(self):
        payload = {"differentials": []}
        client = _make_mock_client(json.dumps(payload))
        with patch("shared.zai_client.get_client", return_value=client):
            run_ddx_reasoning(SAMPLE_PATIENT, symptoms="chest pain")
        assert "chest pain" in _user_message(client)

    def test_no_symptoms_section_when_empty(self):
        payload = {"differentials": []}
        client = _make_mock_client(json.dumps(payload))
        with patch("shared.zai_client.get_client", return_value=client):
            run_ddx_reasoning(SAMPLE_PATIENT, symptoms="")
        assert "Additional Symptoms" not in _user_message(client)

    def test_uses_correct_model(self):
        payload = {"differentials": []}
        client = _make_mock_client(json.dumps(payload))
        with patch("shared.zai_client.get_client", return_value=client):
            run_ddx_reasoning(SAMPLE_PATIENT)
        assert client.chat.completions.create.call_args.kwargs["model"] == ZAI_MODEL

    def test_missing_api_key_returns_error(self):
        with patch("shared.zai_client.get_client", side_effect=ValueError("ZAI_API_KEY")):
            result = run_ddx_reasoning(SAMPLE_PATIENT)
        assert "error" in result


# ── run_drug_interaction_reasoning ────────────────────────────────────────────


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

    def test_rxnav_interactions_included_in_user_message(self):
        payload = {"interactions": []}
        client = _make_mock_client(json.dumps(payload))
        rxnav = {"interactions": [{"pair": ["A", "B"]}], "source": "rxnav"}
        with patch("shared.zai_client.get_client", return_value=client):
            run_drug_interaction_reasoning(SAMPLE_PATIENT, rxnav_interactions=rxnav)
        assert "RxNav" in _user_message(client)

    def test_proposed_medications_included_in_user_message(self):
        payload = {"interactions": []}
        client = _make_mock_client(json.dumps(payload))
        with patch("shared.zai_client.get_client", return_value=client):
            run_drug_interaction_reasoning(SAMPLE_PATIENT, proposed_medications=["warfarin"])
        assert "warfarin" in _user_message(client)

    def test_no_rxnav_adds_ai_generated_note(self):
        """Without RxNav data, the prompt should flag the analysis as AI-generated."""
        payload = {"interactions": []}
        client = _make_mock_client(json.dumps(payload))
        with patch("shared.zai_client.get_client", return_value=client):
            run_drug_interaction_reasoning(SAMPLE_PATIENT, rxnav_interactions=None)
        assert "AI-generated" in _user_message(client)

    def test_returns_error_on_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("timeout")
        with patch("shared.zai_client.get_client", return_value=client):
            result = run_drug_interaction_reasoning(SAMPLE_PATIENT)
        assert "error" in result
        assert "timeout" in result["error"]

    def test_uses_correct_model(self):
        payload = {"interactions": []}
        client = _make_mock_client(json.dumps(payload))
        with patch("shared.zai_client.get_client", return_value=client):
            run_drug_interaction_reasoning(SAMPLE_PATIENT)
        assert client.chat.completions.create.call_args.kwargs["model"] == ZAI_MODEL


# ── run_synthesis ─────────────────────────────────────────────────────────────


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

    def test_care_gaps_included_in_user_message(self):
        payload = {"assessment_summary": "done"}
        client = _make_mock_client(json.dumps(payload))
        care_gaps = {"gaps": ["Missing statin"]}
        with patch("shared.zai_client.get_client", return_value=client):
            run_synthesis(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION, care_gaps=care_gaps)
        assert "Missing statin" in _user_message(client)

    def test_care_gaps_section_absent_when_none(self):
        payload = {"assessment_summary": "done"}
        client = _make_mock_client(json.dumps(payload))
        with patch("shared.zai_client.get_client", return_value=client):
            run_synthesis(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION, care_gaps=None)
        assert "Care Gap" not in _user_message(client)

    def test_returns_error_on_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("model overloaded")
        with patch("shared.zai_client.get_client", return_value=client):
            result = run_synthesis(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION)
        assert "error" in result
        assert "model overloaded" in result["error"]

    def test_uses_correct_model(self):
        payload = {"assessment_summary": "done"}
        client = _make_mock_client(json.dumps(payload))
        with patch("shared.zai_client.get_client", return_value=client):
            run_synthesis(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION)
        assert client.chat.completions.create.call_args.kwargs["model"] == ZAI_MODEL


# ── stream_ddx_tokens ─────────────────────────────────────────────────────────


class TestStreamDdxTokens:
    def test_yields_all_non_empty_tokens(self):
        tokens = ["{\n", '"differentials":', " []}"]
        client = _make_streaming_client(tokens)
        with patch("shared.zai_client.get_client", return_value=client):
            result = list(stream_ddx_tokens(SAMPLE_PATIENT))
        assert result == tokens

    def test_skips_empty_delta_content(self):
        # Empty string is falsy; the streaming loop should skip it
        tokens = ["hello", "", "world"]
        client = _make_streaming_client(tokens)
        with patch("shared.zai_client.get_client", return_value=client):
            result = list(stream_ddx_tokens(SAMPLE_PATIENT))
        assert result == ["hello", "world"]

    def test_called_with_stream_true(self):
        client = _make_streaming_client(["{}"])
        with patch("shared.zai_client.get_client", return_value=client):
            list(stream_ddx_tokens(SAMPLE_PATIENT))
        assert client.chat.completions.create.call_args.kwargs["stream"] is True

    def test_uses_correct_model(self):
        client = _make_streaming_client(["{}"])
        with patch("shared.zai_client.get_client", return_value=client):
            list(stream_ddx_tokens(SAMPLE_PATIENT))
        assert client.chat.completions.create.call_args.kwargs["model"] == ZAI_MODEL

    def test_symptoms_included_in_stream_prompt(self):
        client = _make_streaming_client(["{}"])
        with patch("shared.zai_client.get_client", return_value=client):
            list(stream_ddx_tokens(SAMPLE_PATIENT, symptoms="fatigue"))
        assert "fatigue" in _user_message(client)


# ── stream_drug_interaction_tokens ────────────────────────────────────────────


class TestStreamDrugInteractionTokens:
    def test_yields_all_tokens(self):
        tokens = ['{"interactions": []}']
        client = _make_streaming_client(tokens)
        with patch("shared.zai_client.get_client", return_value=client):
            result = list(stream_drug_interaction_tokens(SAMPLE_PATIENT))
        assert result == tokens

    def test_called_with_stream_true(self):
        client = _make_streaming_client(["{}"])
        with patch("shared.zai_client.get_client", return_value=client):
            list(stream_drug_interaction_tokens(SAMPLE_PATIENT))
        assert client.chat.completions.create.call_args.kwargs["stream"] is True

    def test_rxnav_data_included_in_stream_prompt(self):
        client = _make_streaming_client(["{}"])
        rxnav = {"source": "rxnav", "interactions": []}
        with patch("shared.zai_client.get_client", return_value=client):
            list(stream_drug_interaction_tokens(SAMPLE_PATIENT, rxnav_interactions=rxnav))
        assert "RxNav" in _user_message(client)

    def test_proposed_medications_in_stream_prompt(self):
        client = _make_streaming_client(["{}"])
        with patch("shared.zai_client.get_client", return_value=client):
            list(stream_drug_interaction_tokens(SAMPLE_PATIENT, proposed_medications=["aspirin"]))
        assert "aspirin" in _user_message(client)

    def test_uses_correct_model(self):
        client = _make_streaming_client(["{}"])
        with patch("shared.zai_client.get_client", return_value=client):
            list(stream_drug_interaction_tokens(SAMPLE_PATIENT))
        assert client.chat.completions.create.call_args.kwargs["model"] == ZAI_MODEL


# ── stream_synthesis_tokens ───────────────────────────────────────────────────


class TestStreamSynthesisTokens:
    def test_yields_all_tokens(self):
        tokens = ['{"assessment_summary": "ok"}']
        client = _make_streaming_client(tokens)
        with patch("shared.zai_client.get_client", return_value=client):
            result = list(stream_synthesis_tokens(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION))
        assert result == tokens

    def test_called_with_stream_true(self):
        client = _make_streaming_client(["{}"])
        with patch("shared.zai_client.get_client", return_value=client):
            list(stream_synthesis_tokens(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION))
        assert client.chat.completions.create.call_args.kwargs["stream"] is True

    def test_care_gaps_in_stream_prompt(self):
        client = _make_streaming_client(["{}"])
        care_gaps = {"gaps": ["No statin"]}
        with patch("shared.zai_client.get_client", return_value=client):
            list(stream_synthesis_tokens(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION, care_gaps=care_gaps))
        assert "No statin" in _user_message(client)

    def test_no_care_gaps_section_when_none(self):
        client = _make_streaming_client(["{}"])
        with patch("shared.zai_client.get_client", return_value=client):
            list(stream_synthesis_tokens(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION, care_gaps=None))
        assert "Care Gap" not in _user_message(client)

    def test_uses_correct_model(self):
        client = _make_streaming_client(["{}"])
        with patch("shared.zai_client.get_client", return_value=client):
            list(stream_synthesis_tokens(SAMPLE_PATIENT, SAMPLE_DDX, SAMPLE_INTERACTION))
        assert client.chat.completions.create.call_args.kwargs["model"] == ZAI_MODEL
