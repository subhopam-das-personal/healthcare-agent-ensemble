"""FHIR source connector — SMART Health IT (no auth) and OAuth2 (Epic, etc.).

Provides:
  - paginate_patients(source, resume_url) → async generator of (url, patients, next_url)
  - fetch_patient_resources(source, patient_id) → {conditions, medications, observations}

OAuth2 tokens are cached in memory and refreshed 60 s before expiry.
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

import httpx

from .schema import FhirSource

logger = logging.getLogger(__name__)

FHIR_PAGE_SIZE = 50
FHIR_TIMEOUT = 20.0
_MAX_DETAIL_CONCURRENCY = 5  # simultaneous per-patient detail fetches


# ---------------------------------------------------------------------------
# OAuth2 token cache
# ---------------------------------------------------------------------------

class _OAuthTokenCache:
    def __init__(self):
        self._store: dict[int, dict] = {}  # source_id → {token, expires_at}

    async def get_token(
        self, source: FhirSource, client: httpx.AsyncClient
    ) -> Optional[str]:
        if source.auth_type == "bearer":
            return (source.auth_config or {}).get("token")

        if source.auth_type != "oauth2":
            return None

        cached = self._store.get(source.id)
        if cached and time.monotonic() < cached["expires_at"] - 60:
            return cached["token"]

        cfg = source.auth_config or {}
        try:
            resp = await client.post(
                cfg["token_url"],
                data={
                    "grant_type": "client_credentials",
                    "client_id": cfg["client_id"],
                    "client_secret": cfg["client_secret"],
                    "scope": cfg.get("scope", ""),
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            token = data["access_token"]
            expires_in = int(data.get("expires_in", 3600))
            self._store[source.id] = {
                "token": token,
                "expires_at": time.monotonic() + expires_in,
            }
            logger.info(f"OAuth2 token refreshed for source {source.name!r}")
            return token
        except Exception as e:
            logger.error(f"OAuth2 token fetch failed for {source.name!r}: {e}")
            return None


_token_cache = _OAuthTokenCache()


# ---------------------------------------------------------------------------
# Low-level FHIR GET
# ---------------------------------------------------------------------------

async def _fhir_get(
    client: httpx.AsyncClient,
    url: str,
    token: Optional[str] = None,
    params: Optional[dict] = None,
) -> dict:
    headers = {"Accept": "application/fhir+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = await client.get(url, headers=headers, params=params, timeout=FHIR_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning(f"FHIR {url} → HTTP {e.response.status_code}")
        return {"error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        logger.warning(f"FHIR {url} failed: {e}")
        return {"error": str(e)}


def _extract_resources(bundle: dict, resource_type: str) -> list[dict]:
    if "error" in bundle:
        return []
    return [
        e["resource"]
        for e in bundle.get("entry", [])
        if e.get("resource", {}).get("resourceType") == resource_type
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def paginate_patients(
    source: FhirSource,
    resume_url: Optional[str] = None,
) -> AsyncIterator[tuple[str, list[dict], Optional[str]]]:
    """Yield (current_url, patient_resources, next_url) for each bundle page.

    Pass resume_url to continue a previously interrupted indexing run.
    next_url is None on the last page.
    """
    async with httpx.AsyncClient(timeout=FHIR_TIMEOUT) as client:
        token = await _token_cache.get_token(source, client)

        if resume_url:
            url: Optional[str] = resume_url
            params = None
        else:
            url = f"{source.base_url.rstrip('/')}/Patient"
            params = {"_count": FHIR_PAGE_SIZE}

        while url:
            bundle = await _fhir_get(client, url, token, params)
            params = None  # params only on the first (non-resume) request

            if "error" in bundle:
                logger.error(f"Page fetch failed for {url}: {bundle['error']}")
                return

            patients = _extract_resources(bundle, "Patient")

            next_url: Optional[str] = None
            for link in bundle.get("link", []):
                if link.get("relation") == "next":
                    next_url = link["url"]
                    break

            yield url, patients, next_url
            url = next_url


async def fetch_patient_resources(source: FhirSource, patient_id: str) -> dict:
    """Fetch Condition, MedicationRequest, and Observation bundles for one patient."""
    base = source.base_url.rstrip("/")

    async with httpx.AsyncClient(timeout=FHIR_TIMEOUT) as client:
        token = await _token_cache.get_token(source, client)

        cond_f, med_f, obs_f = await asyncio.gather(
            _fhir_get(client, f"{base}/Condition", token,
                      {"patient": patient_id, "_count": 100}),
            _fhir_get(client, f"{base}/MedicationRequest", token,
                      {"patient": patient_id, "_count": 100}),
            _fhir_get(client, f"{base}/Observation", token,
                      {"patient": patient_id, "_count": 50, "_sort": "-date"}),
        )

    return {
        "conditions": _extract_resources(cond_f, "Condition"),
        "medications": _extract_resources(med_f, "MedicationRequest"),
        "observations": _extract_resources(obs_f, "Observation"),
    }
