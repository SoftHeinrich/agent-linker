"""ILinker2 V33f — V33 + forward coref ×2 union for variance stabilization.

Changes vs V33:
- Phase 7: Runs forward coreference pass TWICE and unions results.
  This eliminates run-to-run variance in coref (32% of total variance)
  with minimal FP cost (+1 per dataset in unit tests).

Checkpoint dir: v33f.
"""

import os

from llm_sad_sam.linkers.experimental.ilinker2_v33 import ILinker2V33


class ILinker2V33f(ILinker2V33):
    """V33 + forward coref ×2 union for variance stabilization."""

    # ── Checkpoint dir override ───────────────────────────────────────

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, "v33f", ds)
        os.makedirs(d, exist_ok=True)
        return d

    # ── Phase 7: Coref ×2 union ──────────────────────────────────────

    def _coref_discourse(self, sentences, components, name_to_id, sent_map, discourse_model):
        """Run discourse coref twice and union results."""
        print("    Pass 1/2...")
        links1 = super()._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)
        print("    Pass 2/2...")
        links2 = super()._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)
        return self._union_coref(links1, links2)

    def _coref_debate(self, sentences, components, name_to_id, sent_map):
        """Run debate coref twice and union results."""
        print("    Pass 1/2...")
        links1 = super()._coref_debate(sentences, components, name_to_id, sent_map)
        print("    Pass 2/2...")
        links2 = super()._coref_debate(sentences, components, name_to_id, sent_map)
        return self._union_coref(links1, links2)

    def _union_coref(self, links1, links2):
        """Union two coref link lists by (sentence_number, component_id)."""
        seen = {}
        for lk in links1:
            seen[(lk.sentence_number, lk.component_id)] = lk
        for lk in links2:
            key = (lk.sentence_number, lk.component_id)
            if key not in seen:
                seen[key] = lk
        result = list(seen.values())
        print(f"    Union: {len(links1)} + {len(links2)} → {len(result)} unique")
        return result
