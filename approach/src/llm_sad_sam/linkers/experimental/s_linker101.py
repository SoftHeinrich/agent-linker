"""S-Linker101 — the merged reading kept as a THIRD blind proposer (recall-led).

The reading round's finding was that two proposal stages beat one merged pass
because they are two looks at the same sentence that cannot see each other. The
round then drew the wrong conclusion from its own mechanism: if blindness is what
buys recall, the merged reading should not *replace* the two proposers. It should
join them.

Scored on the recorded stage-pilot samples, pairing each control sample with the
merged-reading sample from the same invocation and taking the union:

    macro over projects        P      R     F1     F2
    terra, two proposers     76.1   94.9   83.9   89.9
    terra, + third look      74.8   96.1   83.3   90.2
    luna,  two proposers     62.9   90.4   74.1   83.0
    luna,  + third look      61.4   94.4   74.4   85.2

Recall rises every time it is measured -- +1.2 to +1.5 points on terra and +4.0 on
luna -- because the third look is blind to the other two by construction, which is
the property the round showed to be load-bearing. Precision falls, and at the
proposal stage that is the judges' job: three judges already stand downstream, and
the branch's error analysis puts ~95% of residual false negatives at pairs that
never reached a judge at all. Under an F2-led budget, proposal recall is the
binding constraint and proposal precision is not.

The third look is `s_linker97`'s reading -- one prompt, case by case, at the
resolution question's batch size -- which was the strongest single merged arm on
both models (macro F1 +2.3 terra / +10.3 luna at the proposal stage). Its
candidates are unioned into the two streams the head already has, by the same
routing the head uses: a claim whose sentence states the component's name joins
the named stream, otherwise the refer-back stream. Nothing is removed, no judge
changes, and neither of the head's proposal calls is told what the reading found.

Cost: ~21 LLM calls a project against the head's 16.8, about 25% more, spent
entirely on the side of the pipeline that bounds recall.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker90 import SLinker90
from llm_sad_sam.linkers.experimental.s_linker97 import SLinker97


class SLinker101(SLinker97):
    """The head's two proposers, plus the merged reading, all mutually blind."""

    _VARIANT_NAME = "s_linker101"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker101 (head's two proposers + the reading as a third blind look)")

    def _extract_named_mentions(self, sentences, components, name_to_id, sent_map) -> dict:
        """The head's extractor, unioned with the reading's named stream.

        ``SLinker90``'s method is called explicitly rather than through ``super()``:
        the reading must not be handed the extractor's answer, and the extractor
        must not be handed the reading's. Blindness is the mechanism being bought.
        """
        base = SLinker90._extract_named_mentions(
            self, sentences, components, name_to_id, sent_map)
        reading = self._read_document(
            sentences, components, name_to_id, sent_map)["named"]
        added = 0
        for key, cand in reading.items():
            if key not in base:
                base[key] = cand
                added += 1
        print(f"    Third look added {added} named candidates")
        return base

    def _resolve_references(self, sentences, components, name_to_id, sent_map):
        """The head's resolver, unioned with the reading's refer-back stream."""
        links, metadata = SLinker90._resolve_references(
            self, sentences, components, name_to_id, sent_map)
        reading = self._read_document(sentences, components, name_to_id, sent_map)
        seen = {(l.sentence_number, l.component_id) for l in links}
        added = 0
        for link in reading["coref"]:
            key = (link.sentence_number, link.component_id)
            if key in seen:
                continue
            seen.add(key)
            links.append(link)
            if key in reading["metadata"]:
                metadata[key] = reading["metadata"][key]
            added += 1
        print(f"    Third look added {added} refer-back candidates")
        return links, metadata
