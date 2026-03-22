"""RxNav API client for drug interaction lookups."""

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
    """Check drug-drug interactions for a list of RxCUIs using RxNav interaction API."""
    if len(rxcuis) < 2:
        return {"interactions": [], "source": "rxnav", "note": "Need at least 2 medications to check interactions"}

    async with httpx.AsyncClient(timeout=RXNAV_TIMEOUT) as client:
        try:
            rxcui_str = "+".join(rxcuis)
            resp = await client.get(
                f"{RXNAV_BASE}/interaction/list.json",
                params={"rxcuis": rxcui_str},
            )
            resp.raise_for_status()
            data = resp.json()

            interactions = []
            interaction_groups = data.get("fullInteractionTypeGroup", [])
            for group in interaction_groups:
                source = group.get("sourceName", "Unknown")
                for itype in group.get("fullInteractionType", []):
                    for pair in itype.get("interactionPair", []):
                        concepts = pair.get("interactionConcept", [])
                        drug_names = [
                            c.get("minConceptItem", {}).get("name", "Unknown")
                            for c in concepts
                        ]
                        interactions.append({
                            "drug_pair": drug_names,
                            "severity": pair.get("severity", "N/A"),
                            "description": pair.get("description", ""),
                            "source": source,
                        })

            return {
                "interactions": interactions,
                "source": "rxnav",
                "rxcuis_checked": rxcuis,
            }
        except httpx.TimeoutException:
            logger.warning("RxNav interaction check timed out")
            return {"interactions": [], "source": "rxnav", "error": "RxNav timeout - using AI-only analysis"}
        except Exception as e:
            logger.warning(f"RxNav interaction check failed: {e}")
            return {"interactions": [], "source": "rxnav", "error": f"RxNav failed: {str(e)}"}


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
            # Try to look up by name
            resolved_rxcui = await get_rxcui_from_name(display)
            enriched.append({**med, "rxcui": resolved_rxcui})
        else:
            enriched.append({**med, "rxcui": None})

    return enriched
