"""ReplayClient — delegates extract_json to LLMClient; forbids query().

Phase 44 zero-LLM-call contract (LOCK-6, REQ-V264-01):
  - ReplayClient.query() raises RuntimeError unconditionally.
  - ReplayClient.extract_json() wraps response_text in an LLMResponse and
    delegates to LLMClient.extract_json, which is pure json.loads (no network).
  - Construction: LLMClient is instantiated only to bind its extract_json method.
    No API key is needed; no network call is made during construction.

Module-level helper:
  - replay_parse(response_text) -> dict | None
    Instantiates a ReplayClient singleton lazily (via _singleton()) and proxies.
    Imported by Plan 02's snapshot test modules.

sys.path bootstrap: inherited from tests/conftest.py.
Do NOT modify sys.path in this module.
"""
from __future__ import annotations

import functools
from typing import Optional

from llm_sad_sam.llm_client import LLMClient, LLMResponse


class ReplayClient:
    """Replay-safe LLM client: extract_json is live; query() is forbidden.

    Construction does NOT make network calls.  LLMClient is instantiated with
    checkpoint_fallback="claude" so no backend-specific initialisation runs.
    """

    def __init__(self) -> None:
        # Instantiate LLMClient purely to bind its extract_json bound method.
        # checkpoint_fallback="claude" avoids OpenAI client construction.
        self._llm: LLMClient = LLMClient(checkpoint_fallback="claude")

    # ------------------------------------------------------------------
    # Forbidden interface
    # ------------------------------------------------------------------

    def query(self, *args, **kwargs):  # type: ignore[override]
        """Unconditionally raise RuntimeError.

        Phase 44 harness must not contact any LLM backend.  Any accidental
        call to .query() fails loud so the failure is visible in pytest output.
        """
        raise RuntimeError(
            "ReplayClient.query() is forbidden — "
            "Phase 44 harness must not contact any LLM backend"
        )

    # ------------------------------------------------------------------
    # Live interface: delegates to LLMClient.extract_json (pure json.loads)
    # ------------------------------------------------------------------

    def extract_json(self, response_text: str) -> Optional[dict]:
        """Parse JSON from *response_text* using LLMClient.extract_json.

        Wraps *response_text* in LLMResponse(text=..., success=True) and
        delegates to the canonical parser path.  Returns the parsed dict or
        None if no JSON object is found (same semantics as production).

        Args:
            response_text: raw text from a _calls.json record's response_text field.

        Returns:
            Parsed dict or None.
        """
        fake_resp = LLMResponse(text=response_text, success=True)
        return self._llm.extract_json(fake_resp)


# ---------------------------------------------------------------------------
# Module-level singleton helper
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _singleton() -> ReplayClient:
    """Return a lazily-initialised ReplayClient singleton."""
    return ReplayClient()


def replay_parse(response_text: str) -> Optional[dict]:
    """Parse JSON from *response_text* via a singleton ReplayClient.

    Convenience wrapper imported by Plan 02's test modules.

    Args:
        response_text: raw text from a _calls.json record's response_text field.

    Returns:
        Parsed dict or None if no JSON object found.
    """
    return _singleton().extract_json(response_text)
