"""ILinker2 V33g — V33f + Java TransArc CSV seed (deterministic Phase 4).

Changes vs V33f:
- Phase 4: Uses Java TransArc CSV seed instead of ILinker2. Deterministic,
  zero LLM cost, zero variance. Falls back to ILinker2 if no CSV provided.
- Phase 7: Inherits coref ×2 union from V33f.

Checkpoint dir: v33g.
"""

import csv
import os

from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.linkers.experimental.ilinker2_v33f import ILinker2V33f


class ILinker2V33g(ILinker2V33f):
    """V33f + Java TransArc CSV seed for zero Phase 4 variance."""

    # ── Checkpoint dir override ───────────────────────────────────────

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, "v33g", ds)
        os.makedirs(d, exist_ok=True)
        return d

    # ── Phase 4: Java TransArc CSV seed ──────────────────────────────

    def _process_transarc(self, transarc_csv, id_to_name, sent_map, name_to_id):
        """Load Java TransArc CSV if provided, else fall back to ILinker2."""
        if transarc_csv and os.path.exists(transarc_csv):
            return self._load_transarc_csv(transarc_csv, id_to_name, sent_map)
        # Fall back to ILinker2
        return super()._process_transarc(transarc_csv, id_to_name, sent_map, name_to_id)

    def _load_transarc_csv(self, csv_path, id_to_name, sent_map):
        """Load TransArc links from Java CLI output CSV."""
        links = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                cid = row.get("modelElementID", "").strip()
                snum_str = row.get("sentence", "").strip()
                if not cid or not snum_str:
                    continue
                snum = int(snum_str)
                cname = id_to_name.get(cid, cid)
                links.append(SadSamLink(
                    sentence_number=snum,
                    component_id=cid,
                    component_name=cname,
                    confidence=1.0,
                    source="transarc",
                ))
        print(f"  Loaded {len(links)} links from Java TransArc CSV")
        return links
