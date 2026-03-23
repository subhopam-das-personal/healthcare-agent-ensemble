"""Clinical Intelligence MCP Server — 4 core tools + 1 stretch goal."""

import asyncio
import json
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.transport_security import TransportSecuritySettings

# Add parent to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.fhir_client import get_patient_data, DEFAULT_FHIR_BASE_URL
from shared.claude_client import run_ddx_reasoning, run_drug_interaction_reasoning, run_synthesis
from shared.rxnav_client import get_interactions, resolve_medications_to_rxcuis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_allowed_hosts = os.environ.get("MCP_ALLOWED_HOSTS", "").split(",")
_allowed_hosts = [h.strip() for h in _allowed_hosts if h.strip()]

mcp = FastMCP(
    "ClinicalIntelligence",
    transport_security=TransportSecuritySettings(allowed_hosts=_allowed_hosts) if _allowed_hosts else None,
)

_UI_RESOURCE_URI = "ui://clinical-intelligence/results"

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clinical Intelligence</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #f8fafc; color: #1e293b; }
  pre { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; overflow-x: auto; font-size: 13px; white-space: pre-wrap; word-break: break-word; }
  h2 { font-size: 16px; font-weight: 600; margin: 0 0 12px; color: #0f172a; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; background: #dbeafe; color: #1d4ed8; margin-bottom: 12px; }
</style>
</head>
<body>
<span class="badge">Clinical Intelligence</span>
<h2>Tool Result</h2>
<pre id="output">Loading...</pre>
<script>
window.addEventListener("message", function(e) {
  var d = e.data;
  if (d && d.method === "tools/result" && d.params && d.params.content) {
    var text = d.params.content.map(function(c) { return c.text || ""; }).join("\\n");
    try { text = JSON.stringify(JSON.parse(text), null, 2); } catch(_) {}
    document.getElementById("output").textContent = text;
  }
});
</script>
</body>
</html>"""


@mcp.resource(
    _UI_RESOURCE_URI,
    name="clinical_results",
    description="Renders clinical tool results",
    mime_type="text/html;profile=mcp-app",
)
def clinical_ui_resource() -> str:
    return _UI_HTML


_UI_META = {"ui": {"resourceUri": _UI_RESOURCE_URI}}


@mcp.tool(meta=_UI_META)
async def get_patient_summary(
    patient_id: str,
    fhir_base_url: str = DEFAULT_FHIR_BASE_URL,
    access_token: str = "",
    patient_json: str = "",
    ctx: Context = None,
) -> str:
    """Fetch comprehensive patient summary from FHIR server.

    Retrieves Patient demographics, Conditions (SNOMED), MedicationRequests (RxNorm),
    Observations (LOINC), and AllergyIntolerances for a given patient.

    Args:
        patient_id: FHIR Patient resource ID
        fhir_base_url: FHIR server base URL (default: SMART Health IT sandbox)
        access_token: SMART on FHIR bearer token (optional, from SHARP context)
        patient_json: Optional pre-fetched patient data JSON (skips FHIR fetch if provided)
    """
    logger.info(f"[get_patient_summary] patient_id={patient_id!r} fhir_base_url={fhir_base_url!r} has_patient_json={bool(patient_json)}")
    try:
        if ctx:
            await ctx.info(f"Fetching FHIR data for patient {patient_id}")
    except Exception:
        pass

    if patient_json:
        try:
            data = json.loads(patient_json)
            logger.info(f"[get_patient_summary] Using pre-fetched patient_json ({len(patient_json)} bytes)")
        except json.JSONDecodeError as e:
            logger.error(f"[get_patient_summary] Invalid patient_json: {e}")
            return json.dumps({"error": f"Invalid patient_json: {str(e)}"}, indent=2)
    else:
        token = access_token if access_token else None
        data = await get_patient_data(patient_id, fhir_base_url, token)

    if "error" in data:
        logger.error(f"[get_patient_summary] FHIR error for {patient_id}: {data['error']}")
        return json.dumps({"error": data["error"]}, indent=2)

    summary = (
        f"Found: {len(data['conditions'])} conditions, "
        f"{len(data['medications'])} medications, "
        f"{len(data['observations'])} observations, "
        f"{len(data['allergies'])} allergies"
    )
    logger.info(f"[get_patient_summary] {summary}")

    try:
        if ctx:
            await ctx.info(summary)
    except Exception:
        pass

    result = json.dumps(data, indent=2)
    logger.info(f"[get_patient_summary] Returning {len(result)} bytes")
    return result


@mcp.tool(meta=_UI_META)
async def generate_differential_diagnosis(
    patient_id: str,
    fhir_base_url: str = DEFAULT_FHIR_BASE_URL,
    symptoms: str = "",
    access_token: str = "",
    patient_json: str = "",
    ctx: Context = None,
) -> str:
    """Generate AI-powered differential diagnosis from FHIR patient data.

    Analyzes patient conditions, labs, medications, and optional symptoms to produce
    ranked differential diagnoses with supporting evidence and recommended workup.

    Args:
        patient_id: FHIR Patient resource ID
        fhir_base_url: FHIR server base URL
        symptoms: Optional free-text symptom description to supplement FHIR data
        access_token: SMART on FHIR bearer token (optional)
        patient_json: Optional pre-fetched patient data JSON (skips FHIR fetch if provided)
    """
    logger.info(f"[generate_differential_diagnosis] patient_id={patient_id!r} fhir_base_url={fhir_base_url!r} has_patient_json={bool(patient_json)}")
    try:
        if ctx:
            await ctx.info(f"Building clinical vignette for patient {patient_id}")
    except Exception:
        pass

    if patient_json:
        try:
            patient_data = json.loads(patient_json)
            logger.info(f"[generate_differential_diagnosis] Using pre-fetched patient_json ({len(patient_json)} bytes)")
        except json.JSONDecodeError as e:
            logger.error(f"[generate_differential_diagnosis] Invalid patient_json: {e}")
            return json.dumps({"error": f"Invalid patient_json: {str(e)}"}, indent=2)
    else:
        token = access_token if access_token else None
        patient_data = await get_patient_data(patient_id, fhir_base_url, token)

    if "error" in patient_data:
        logger.error(f"[generate_differential_diagnosis] FHIR error: {patient_data['error']}")
        return json.dumps({"error": patient_data["error"]}, indent=2)

    if not patient_data["conditions"] and not symptoms:
        logger.warning(f"[generate_differential_diagnosis] No conditions and no symptoms for {patient_id}")
        return json.dumps({
            "error": "No conditions found and no symptoms provided. Cannot generate differential.",
            "suggestion": "Provide symptoms parameter with clinical presentation details."
        }, indent=2)

    logger.info(f"[generate_differential_diagnosis] Running DDx reasoning for {patient_id}")
    try:
        if ctx:
            await ctx.info("Running AI differential diagnosis reasoning...")
    except Exception:
        pass

    result = await asyncio.to_thread(run_ddx_reasoning, patient_data, symptoms)
    out = json.dumps(result, indent=2)
    logger.info(f"[generate_differential_diagnosis] Returning {len(out)} bytes")
    return out


@mcp.tool(meta=_UI_META)
async def check_drug_interactions(
    patient_id: str,
    fhir_base_url: str = DEFAULT_FHIR_BASE_URL,
    proposed_medications: str = "",
    access_token: str = "",
    patient_json: str = "",
    ctx: Context = None,
) -> str:
    """Check drug-drug interactions for a patient's medication list with AI clinical interpretation.

    Fetches current medications from FHIR, checks interactions via RxNav database,
    then uses AI to interpret clinical significance considering the patient's specific
    conditions and allergies.

    Args:
        patient_id: FHIR Patient resource ID
        fhir_base_url: FHIR server base URL
        proposed_medications: Optional comma-separated new medications to check against existing regimen
        access_token: SMART on FHIR bearer token (optional)
        patient_json: Optional pre-fetched patient data JSON (skips FHIR fetch if provided)
    """
    logger.info(f"[check_drug_interactions] patient_id={patient_id!r} fhir_base_url={fhir_base_url!r} has_patient_json={bool(patient_json)}")
    try:
        if ctx:
            await ctx.info(f"Checking drug interactions for patient {patient_id}")
    except Exception:
        pass

    if patient_json:
        try:
            patient_data = json.loads(patient_json)
            logger.info(f"[check_drug_interactions] Using pre-fetched patient_json ({len(patient_json)} bytes)")
        except json.JSONDecodeError as e:
            logger.error(f"[check_drug_interactions] Invalid patient_json: {e}")
            return json.dumps({"error": f"Invalid patient_json: {str(e)}"}, indent=2)
    else:
        token = access_token if access_token else None
        patient_data = await get_patient_data(patient_id, fhir_base_url, token)

    if "error" in patient_data:
        logger.error(f"[check_drug_interactions] FHIR error: {patient_data['error']}")
        return json.dumps({"error": patient_data["error"]}, indent=2)

    medications = patient_data.get("medications", [])
    logger.info(f"[check_drug_interactions] {len(medications)} medications found for {patient_id}")
    if not medications:
        return json.dumps({
            "interactions": [],
            "note": "No medications found for this patient.",
            "overall_risk_level": "N/A"
        }, indent=2)

    # Resolve medications to RxCUIs
    try:
        if ctx:
            await ctx.info(f"Resolving {len(medications)} medications to RxCUI codes...")
    except Exception:
        pass

    enriched_meds = await resolve_medications_to_rxcuis(medications)
    rxcuis = [m["rxcui"] for m in enriched_meds if m.get("rxcui")]
    logger.info(f"[check_drug_interactions] Resolved {len(rxcuis)} RxCUIs")

    # Check RxNav interactions
    rxnav_results = None
    if len(rxcuis) >= 2:
        try:
            if ctx:
                await ctx.info(f"Checking {len(rxcuis)} RxCUIs against RxNav interaction database...")
        except Exception:
            pass
        rxnav_results = await get_interactions(rxcuis)

    # Determine if we need AI-only fallback
    has_db_interactions = (
        rxnav_results
        and rxnav_results.get("interactions")
        and len(rxnav_results["interactions"]) > 0
    )

    try:
        if not has_db_interactions and len(medications) >= 3:
            if ctx:
                await ctx.info("No database interactions found for 3+ medications. Running AI-only analysis...")
        if ctx:
            await ctx.info("Running AI clinical significance analysis...")
    except Exception:
        pass

    logger.info(f"[check_drug_interactions] Running AI reasoning (db_interactions={has_db_interactions})")
    proposed = [m.strip() for m in proposed_medications.split(",") if m.strip()] if proposed_medications else None
    result = await asyncio.to_thread(run_drug_interaction_reasoning, patient_data, rxnav_results, proposed)
    out = json.dumps(result, indent=2)
    logger.info(f"[check_drug_interactions] Returning {len(out)} bytes")
    return out


@mcp.tool(meta=_UI_META)
async def synthesize_clinical_assessment(
    patient_summary_json: str,
    ddx_results_json: str,
    interaction_results_json: str,
    care_gaps_json: str = "",
    ctx: Context = None,
) -> str:
    """Synthesize differential diagnosis and drug interaction analyses into a unified clinical briefing.

    This is the crown jewel tool — it identifies cross-cutting insights that individual
    analyses miss, such as a suspected diagnosis that contraindicates a current medication,
    or a drug side effect that could explain a symptom in the differential.

    Args:
        patient_summary_json: JSON string from get_patient_summary
        ddx_results_json: JSON string from generate_differential_diagnosis
        interaction_results_json: JSON string from check_drug_interactions
        care_gaps_json: Optional JSON string from analyze_care_gaps
    """
    try:
        if ctx:
            await ctx.info("Synthesizing cross-cutting clinical assessment...")
    except Exception:
        pass

    try:
        patient_summary = json.loads(patient_summary_json)
        ddx_results = json.loads(ddx_results_json)
        interaction_results = json.loads(interaction_results_json)
        care_gaps = json.loads(care_gaps_json) if care_gaps_json else None
    except json.JSONDecodeError as e:
        logger.error(f"[synthesize_clinical_assessment] JSON parse error: {e}")
        return json.dumps({"error": f"Invalid JSON input: {str(e)}"}, indent=2)

    logger.info("[synthesize_clinical_assessment] Running synthesis")
    result = await asyncio.to_thread(run_synthesis, patient_summary, ddx_results, interaction_results, care_gaps)
    out = json.dumps(result, indent=2)
    logger.info(f"[synthesize_clinical_assessment] Returning {len(out)} bytes")

    try:
        if ctx:
            await ctx.info("Clinical assessment synthesis complete.")
    except Exception:
        pass

    return out


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", os.environ.get("MCP_PORT", 8000)))
    api_key = os.environ.get("MCP_API_KEY", "")
    asgi_app = mcp.streamable_http_app()

    if api_key:
        _expected = api_key.encode()

        class APIKeyMiddleware:
            """Pure ASGI middleware — no Starlette wrapping that would break FastMCP routing."""
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                if scope["type"] != "http":
                    await self.app(scope, receive, send)
                    return
                path = scope.get("path", "")
                method = scope.get("method", "GET")

                import json as _json
                req_headers = dict(scope.get("headers", []))
                host = req_headers.get(b"host", b"localhost").decode()
                proto = req_headers.get(b"x-forwarded-proto", b"https").decode()
                base = f"{proto}://{host}"
                origin = req_headers.get(b"origin", b"*")

                _cors = [
                    (b"access-control-allow-origin", origin),
                    (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
                    (b"access-control-allow-headers",
                     b"content-type, authorization, mcp-protocol-version, x-api-key"),
                    (b"access-control-max-age", b"3600"),
                ]

                # Handle CORS preflights ourselves — FastMCP returns 404 for
                # /.well-known/ paths, which means no CORS headers → browser blocks GET
                if method == "OPTIONS":
                    await send({"type": "http.response.start", "status": 204,
                                "headers": _cors})
                    await send({"type": "http.response.body", "body": b""})
                    return

                # OAuth Protected Resource Metadata (RFC 9728 / MCP 2025-06-18)
                if path == "/.well-known/oauth-protected-resource":
                    body = _json.dumps({
                        "resource": base,
                        "bearer_methods_supported": ["header"],
                        "authorization_servers": [base],
                    }).encode()
                    await send({"type": "http.response.start", "status": 200,
                                "headers": _cors + [
                                    (b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())]})
                    await send({"type": "http.response.body", "body": body})
                    return

                # OAuth Authorization Server Metadata (RFC 8414)
                if path == "/.well-known/oauth-authorization-server":
                    body = _json.dumps({
                        "issuer": base,
                        "token_endpoint": f"{base}/token",
                        "response_types_supported": ["token"],
                        "grant_types_supported": ["client_credentials"],
                    }).encode()
                    await send({"type": "http.response.start", "status": 200,
                                "headers": _cors + [
                                    (b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())]})
                    await send({"type": "http.response.body", "body": body})
                    return

                headers = req_headers
                # Accept X-API-Key header or Authorization: Bearer <key>
                auth_header = headers.get(b"authorization", b"")
                bearer_key = auth_header[len(b"Bearer "):] if auth_header.startswith(b"Bearer ") else None
                if headers.get(b"x-api-key") != _expected and bearer_key != _expected:
                    body = b'{"error":"Unauthorized"}'
                    await send({"type": "http.response.start", "status": 401,
                                "headers": [
                                    (b"content-type", b"application/json"),
                                    (b"access-control-allow-origin", b"*"),
                                    (b"www-authenticate", b'Bearer realm="ClinicalIntelligence"'),
                                    (b"content-length", str(len(body)).encode())]})
                    await send({"type": "http.response.body", "body": body})
                    return
                await self.app(scope, receive, send)

        app = APIKeyMiddleware(asgi_app)
        logger.info("API key authentication enabled (X-API-Key header required)")
    else:
        app = asgi_app
        logger.warning("MCP_API_KEY not set — server is unauthenticated")

    logger.info(f"Starting Clinical Intelligence MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, lifespan="on")
