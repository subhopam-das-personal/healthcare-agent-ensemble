"""MCP client — calls the deployed MCP server via streamable-http transport."""

import json
import logging
import os

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)

# Full endpoint URL including /mcp suffix, e.g. https://xxx.railway.app/mcp
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/mcp")
_MCP_API_KEY = os.environ.get("MCP_API_KEY", "")
_TIMEOUT = httpx.Timeout(120.0)


async def _call_tool(session: ClientSession, name: str, args: dict) -> dict:
    """Call an MCP tool and return the parsed JSON result."""
    logger.info(f"[mcp_client] → {name}({list(args.keys())})")
    result = await session.call_tool(name, args)
    raw = result.content[0].text if result.content else "{}"
    parsed = json.loads(raw)
    logger.info(f"[mcp_client] ← {name} returned {len(raw)} bytes")
    return parsed


async def mcp_get_patient_summary(
    session: ClientSession,
    patient_id: str,
    fhir_base_url: str = "",
    access_token: str = "",
) -> dict:
    args = {"patient_id": patient_id}
    if fhir_base_url:
        args["fhir_base_url"] = fhir_base_url
    if access_token:
        args["access_token"] = access_token
    return await _call_tool(session, "get_patient_summary", args)


async def mcp_generate_differential_diagnosis(
    session: ClientSession,
    patient_id: str,
    fhir_base_url: str = "",
    symptoms: str = "",
    access_token: str = "",
    patient_json: str = "",
) -> dict:
    args = {"patient_id": patient_id}
    if fhir_base_url:
        args["fhir_base_url"] = fhir_base_url
    if symptoms:
        args["symptoms"] = symptoms
    if access_token:
        args["access_token"] = access_token
    if patient_json:
        args["patient_json"] = patient_json
    return await _call_tool(session, "generate_differential_diagnosis", args)


async def mcp_check_drug_interactions(
    session: ClientSession,
    patient_id: str,
    fhir_base_url: str = "",
    proposed_medications: str = "",
    access_token: str = "",
    patient_json: str = "",
) -> dict:
    args = {"patient_id": patient_id}
    if fhir_base_url:
        args["fhir_base_url"] = fhir_base_url
    if proposed_medications:
        args["proposed_medications"] = proposed_medications
    if access_token:
        args["access_token"] = access_token
    if patient_json:
        args["patient_json"] = patient_json
    return await _call_tool(session, "check_drug_interactions", args)


async def mcp_synthesize_clinical_assessment(
    session: ClientSession,
    patient_summary_json: str,
    ddx_results_json: str,
    interaction_results_json: str,
) -> dict:
    args = {
        "patient_summary_json": patient_summary_json,
        "ddx_results_json": ddx_results_json,
        "interaction_results_json": interaction_results_json,
        "care_gaps_json": "",
    }
    return await _call_tool(session, "synthesize_clinical_assessment", args)


def make_mcp_session():
    """Return an async context manager that yields an initialized ClientSession."""
    return _MCPSessionContext()


class _MCPSessionContext:
    async def __aenter__(self) -> ClientSession:
        headers = {"Authorization": f"Bearer {_MCP_API_KEY}"} if _MCP_API_KEY else {}
        http_client = httpx.AsyncClient(timeout=_TIMEOUT, headers=headers)
        self._http_client = http_client
        self._transport = streamable_http_client(MCP_SERVER_URL, http_client=http_client)
        read, write, _ = await self._transport.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        return self._session

    async def __aexit__(self, *args):
        await self._session.__aexit__(*args)
        await self._transport.__aexit__(*args)
        await self._http_client.aclose()
