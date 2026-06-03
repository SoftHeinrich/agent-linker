"""S3: s1 + s2 Combined — Fix B restriction AND Phase 6 min alias length.

Double protection: even if Fix B is bypassed somehow, Phase 6 won't auto-approve
2-char aliases.
"""

import re
from llm_sad_sam.core.data_types import DocumentKnowledge
from llm_sad_sam.linkers.experimental.ilinker2_v26a_s1 import ILinker2V26aS1
from llm_sad_sam.linkers.experimental.ilinker2_v26a_s2 import ILinker2V26aS2


class ILinker2V26aS3(ILinker2V26aS1):
    """V26a + Fix B restricted + Phase 6 min alias length >= 3."""

    # Inherit _learn_document_knowledge_enriched from S1 (Fix B restriction)
    # Override _validate_intersect from S2 (min alias length)

    def _validate_intersect(self, candidates, components, sent_map):
        """Delegate to S2's implementation."""
        return ILinker2V26aS2._validate_intersect(self, candidates, components, sent_map)

    def _inject_partial_references(self, sentences, components, name_to_id,
                                    transarc_set, validated_set, coref_set, implicit_set):
        """Delegate to S2's implementation (skip short partials)."""
        return ILinker2V26aS2._inject_partial_references(
            self, sentences, components, name_to_id,
            transarc_set, validated_set, coref_set, implicit_set)
