"""RxNav API client for medication resolution.

Note: The RxNav Drug Interaction API was discontinued in January 2024.
Drug interaction analysis is handled entirely by Claude AI reasoning.
This client is kept for RxCUI resolution (medication identification).
"""

import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
RXNAV_TIMEOUT = 5.0


async def get_rxcui_from_name(drug_name: str) -> Optional[str]:
    """Look up RxCUI for a drug name."""
    async with httpx.AsyncClient(timeout=RXNAV_TIMEOUT) as client:
        try:
            resp = await client.get(
                f"{RXNAV_BASE}/rxcui.json",
                params={"name": drug_name, "search": 2},
            )
            resp.raise_for_status()
            data = resp.json()
            group = data.get("idGroup", {})
            rxnorm_id = group.get("rxnormId")
            if rxnorm_id:
                return rxnorm_id[0] if isinstance(rxnorm_id, list) else rxnorm_id
            return None
        except Exception as e:
            logger.warning(f"RxNav lookup failed for {drug_name}: {e}")
            return None


async def get_interactions(rxcuis: list[str]) -> dict:
    """Drug interaction API was discontinued Jan 2024. Returns empty result with note."""
    return {
        "interactions": [],
        "source": "rxnav-discontinued",
        "note": "RxNav Drug Interaction API discontinued Jan 2024. Using AI-only analysis.",
        "rxcuis_provided": rxcuis,
    }


async def resolve_medications_to_rxcuis(medications: list[dict]) -> list[dict]:
    """Resolve medication display names to RxCUIs. Returns enriched medication list."""
    enriched = []
    for med in medications:
        rxcui = med.get("code")
        display = med.get("display", "")

        # If code looks like an RxNorm code (numeric), use it directly
        if rxcui and rxcui.isdigit():
            enriched.append({**med, "rxcui": rxcui})
        elif display:
            resolved_rxcui = await get_rxcui_from_name(display)
            enriched.append({**med, "rxcui": resolved_rxcui})
        else:
            enriched.append({**med, "rxcui": None})

    return enriched
