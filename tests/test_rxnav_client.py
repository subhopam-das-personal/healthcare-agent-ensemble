"""Unit tests for shared/rxnav_client.py"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shared.rxnav_client import (
    get_rxcui_from_name,
    get_interactions,
    resolve_medications_to_rxcuis,
)


# --------------------------------------------------------------------------- #
# get_interactions (API discontinued — always returns empty stub)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_get_interactions_returns_empty_stub():
    """RxNav interaction API was discontinued; must return empty interactions with note."""
    result = await get_interactions(["1049502", "854878"])
    assert result["interactions"] == []
    assert "discontinued" in result["source"].lower() or "discontinued" in result["note"].lower()
    assert result["rxcuis_provided"] == ["1049502", "854878"]


@pytest.mark.asyncio
async def test_get_interactions_preserves_rxcuis():
    rxcuis = ["123", "456", "789"]
    result = await get_interactions(rxcuis)
    assert result["rxcuis_provided"] == rxcuis


@pytest.mark.asyncio
async def test_get_interactions_empty_list():
    result = await get_interactions([])
    assert result["interactions"] == []
    assert result["rxcuis_provided"] == []


# --------------------------------------------------------------------------- #
# get_rxcui_from_name (HTTP call to RxNav)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_get_rxcui_from_name_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "idGroup": {"rxnormId": ["1049502"]}
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await get_rxcui_from_name("metformin")

    assert result == "1049502"


@pytest.mark.asyncio
async def test_get_rxcui_from_name_list_response():
    """rxnormId is sometimes a list — should return first element."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "idGroup": {"rxnormId": ["111", "222"]}
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await get_rxcui_from_name("aspirin")

    assert result == "111"


@pytest.mark.asyncio
async def test_get_rxcui_from_name_not_found():
    mock_response = MagicMock()
    mock_response.json.return_value = {"idGroup": {}}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await get_rxcui_from_name("not-a-drug")

    assert result is None


@pytest.mark.asyncio
async def test_get_rxcui_from_name_http_error_returns_none():
    """Network/HTTP errors should return None, not raise."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))

        result = await get_rxcui_from_name("metformin")

    assert result is None


# --------------------------------------------------------------------------- #
# resolve_medications_to_rxcuis
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_resolve_uses_existing_numeric_code():
    """If medication already has a numeric RxNorm code, skip HTTP lookup."""
    medications = [{"code": "1049502", "display": "Metformin 500 MG", "status": "active"}]

    with patch("shared.rxnav_client.get_rxcui_from_name", new_callable=AsyncMock) as mock_lookup:
        result = await resolve_medications_to_rxcuis(medications)

    mock_lookup.assert_not_called()
    assert result[0]["rxcui"] == "1049502"


@pytest.mark.asyncio
async def test_resolve_looks_up_non_numeric_code():
    """Non-numeric codes should trigger RxCUI lookup by display name."""
    medications = [{"code": "SNOMED-123", "display": "Lisinopril 10mg", "status": "active"}]

    with patch("shared.rxnav_client.get_rxcui_from_name", new_callable=AsyncMock) as mock_lookup:
        mock_lookup.return_value = "29046"
        result = await resolve_medications_to_rxcuis(medications)

    mock_lookup.assert_called_once_with("Lisinopril 10mg")
    assert result[0]["rxcui"] == "29046"


@pytest.mark.asyncio
async def test_resolve_no_code_uses_display():
    """Missing code should fall back to display for lookup."""
    medications = [{"display": "Atorvastatin 40mg", "status": "active"}]

    with patch("shared.rxnav_client.get_rxcui_from_name", new_callable=AsyncMock) as mock_lookup:
        mock_lookup.return_value = "617311"
        result = await resolve_medications_to_rxcuis(medications)

    mock_lookup.assert_called_once_with("Atorvastatin 40mg")
    assert result[0]["rxcui"] == "617311"


@pytest.mark.asyncio
async def test_resolve_no_code_no_display_returns_none():
    """Medications with neither code nor display should get rxcui=None."""
    medications = [{"status": "active"}]

    with patch("shared.rxnav_client.get_rxcui_from_name", new_callable=AsyncMock) as mock_lookup:
        result = await resolve_medications_to_rxcuis(medications)

    mock_lookup.assert_not_called()
    assert result[0]["rxcui"] is None


@pytest.mark.asyncio
async def test_resolve_preserves_original_fields():
    """Original medication fields must be preserved in the enriched output."""
    medications = [{"code": "1049502", "display": "Metformin", "status": "active", "authoredOn": "2023-01-01"}]

    result = await resolve_medications_to_rxcuis(medications)

    assert result[0]["status"] == "active"
    assert result[0]["authoredOn"] == "2023-01-01"
    assert result[0]["display"] == "Metformin"


@pytest.mark.asyncio
async def test_resolve_empty_list():
    result = await resolve_medications_to_rxcuis([])
    assert result == []


@pytest.mark.asyncio
async def test_resolve_mixed_medication_list():
    """Mixed list: numeric code, non-numeric code, no code."""
    medications = [
        {"code": "1049502", "display": "Metformin"},   # numeric → direct
        {"code": "SNOMED-X", "display": "Lisinopril"}, # non-numeric → lookup
        {"display": "Aspirin"},                         # no code → lookup by display
    ]

    with patch("shared.rxnav_client.get_rxcui_from_name", new_callable=AsyncMock) as mock_lookup:
        mock_lookup.side_effect = ["29046", "1191"]
        result = await resolve_medications_to_rxcuis(medications)

    assert result[0]["rxcui"] == "1049502"
    assert result[1]["rxcui"] == "29046"
    assert result[2]["rxcui"] == "1191"
    assert mock_lookup.call_count == 2
