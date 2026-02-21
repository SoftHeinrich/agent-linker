"""ILinker2 + V26a: Replace TransArc baseline with ILinker2 explicit extraction.

Subclasses V26a and overrides Phase 4 (_process_transarc) to use ILinker2
instead of reading a TransArc CSV. V26a's remaining phases (coref, validation,
judge, etc.) run unchanged on top of ILinker2's high-precision explicit links.
"""

from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a
from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2


class ILinker2V26a(AgentLinkerV26a):
    """V26a pipeline with ILinker2 replacing TransArc as Phase 4 seed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ilinker2 = ILinker2(backend=self.llm.backend)

    def _process_transarc(self, transarc_csv, id_to_name, sent_map, name_to_id):
        """Override: run ILinker2 instead of loading TransArc CSV.

        Returns links with source="transarc" so downstream phases
        (TransArc immunity, deliberation judge, boundary filter bypass)
        treat them identically to real TransArc links.
        """
        # ILinker2 needs text_path and model_path — retrieve from cached state
        text_path = self._cached_text_path
        model_path = self._cached_model_path

        raw_links = self._ilinker2.link(text_path, model_path)

        # Re-label source as "transarc" so V26a's downstream logic applies
        # the same immunity/priority as real TransArc links
        result = []
        for lk in raw_links:
            # Filter out any that are in dotted paths (ILinker2 should handle
            # this already, but double-check against V26a's logic)
            sent = sent_map.get(lk.sentence_number)
            if sent and self._in_dotted_path(sent.text, lk.component_name):
                continue
            result.append(SadSamLink(
                sentence_number=lk.sentence_number,
                component_id=lk.component_id,
                component_name=lk.component_name,
                confidence=lk.confidence,
                source="transarc",
            ))
        return result

    def link(self, text_path, model_path, transarc_csv=None):
        """Cache paths for _process_transarc override, then run V26a pipeline."""
        self._cached_text_path = text_path
        self._cached_model_path = model_path
        return super().link(text_path, model_path, transarc_csv=transarc_csv)
