"""S-Linker100 — read once, then glean what the first read left (rung I).

Five rungs established what the merge loses and why. On bigbluebutton, whose gold
puts 26 of 62 links on 12 sentences that reference several components, every
single-look arm finds 14.0-14.3 of those 26 while the head's two stages find 19.6:

    rung   added                          bbb gold   multi-component gold
    A      the merged question               40.3          14.0 / 26
    E      the resolver's batch size         41.7          14.0 / 26
    F      the resolver's per-case duty      44.3          14.0 / 26
    G      the resolver's context table      43.0          14.0 / 26
    H      "a sentence may name several"     44.0          14.3 / 26
    head   two proposal stages               49.0          19.6 / 26

The metric never moves. What two stages supply is not two questions and not a
bigger prompt: it is a **second look at the same sentence that is not conditioned
on the first**. Rung C (resample the same look k times) recovered almost nothing
because independent samples of one look are correlated; rung H failed because an
instruction does not change what a single pass attends to.

A second look can be decorrelated deliberately, by conditioning it on the first
one's output and asking only for the remainder. That is the gleaning pattern from
GraphRAG's entity-extraction stage, where repeated conditioned rounds recover
entities a single extraction pass misses, and it addresses this failure directly:
the loss is entirely on sentences that already carry one link, so the second look
only has to be given those.

    pass 1   the whole document, flat, at EXTRACTION_BATCH  -- 2 calls on bbb
    pass 2   the sentences pass 1 linked, as cases, at COREFERENCE_BATCH,
             each carrying what pass 1 already found for it -- ~4 calls on bbb

That is ~6 calls where the head spends 11, and one authored prompt used twice
instead of two prompts -- the round's original goal, reached through the mechanism
the round measured rather than around it. The routing, the carried per-component
note, all three judges, the alias module and the deterministic scan are inherited
untouched.
"""

from __future__ import annotations

import json

from llm_sad_sam.core.data_types_v2 import CandidateLink, SadSamLink
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names, parse_snum
from llm_sad_sam.linkers.experimental.s_linker90 import (
    COREF_RULES,
    ENTITY_EXTRACTION_RULES,
    QUALIFIED_CLAUSE,
    SLinker90,
)
from llm_sad_sam.linkers.experimental.s_linker97 import SLinker97


class SLinker100(SLinker97):
    """One reading prompt, used twice: read the document, then glean the rest."""

    _VARIANT_NAME = "s_linker100"

    #: pass 1 reads the document at the naming question's size; pass 2 revisits
    #: only the sentences pass 1 linked, at the resolution question's size.
    READING_BATCH = SLinker90.EXTRACTION_BATCH
    GLEAN_BATCH = SLinker90.COREFERENCE_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker100 (read at {self.READING_BATCH}, glean at {self.GLEAN_BATCH})")

    def _prompt_reading(self, comp_names, mappings, batch, established,
                        table=None, found=None) -> str:
        """One prompt. With ``found`` it is the gleaning pass; without, the read."""
        carried = (
            "ESTABLISHED EARLIER (component: the last sentence before this block "
            "that named it)\n" + json.dumps(established)
            if established else ""
        )
        if found is None:
            body = "DOCUMENT:\n" + "\n".join(f"S{s.number}: {s.text}" for s in batch)
            ask = ("Report every reference you find. If a sentence refers to no listed "
                   "component, return nothing for it.")
        else:
            body = "\n".join(
                f"--- Case {i} ---\nTARGET S{s.number}: {s.text}\n"
                f"ALREADY FOUND for this sentence: "
                f"{', '.join(found.get(s.number, [])) or 'nothing'}"
                for i, s in enumerate(batch, 1)
            )
            ask = ("Each sentence below has already been read once, and what that read "
                   "found is listed with it. Report only references it did NOT find. A "
                   "sentence can describe several components at once, so take each case "
                   "on its own and report nothing for it if the earlier read was "
                   "complete. Do not repeat anything already found.")
        return f"""Find every reference to a component in the sentences below.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}
{carried}

A reference takes one of two forms.

1. The sentence writes one of the component's names itself.

{ENTITY_EXTRACTION_RULES}

{QUALIFIED_CLAUSE}

2. The sentence refers to a component without writing any of its names, through a
pronoun or a noun phrase that refers back. Give the earlier sentence that states
the name it refers back to. A reference of the first kind needs no antecedent;
leave it null.

{COREF_RULES}

{ask}

{body}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence", "antecedent_sentence": M_INTEGER_OR_NULL, "antecedent_text": "exact quote naming the component, or null"}}]}}
JSON only:"""

    def _read_document(self, sentences, components, name_to_id, sent_map) -> dict:
        if self._reading is not None:
            return self._reading
        comp_names = get_comp_names(components)
        mappings = ([f"{t}={c}" for t, c in self.doc_knowledge.aliases.items()]
                    if self.doc_knowledge else [])
        named: dict = {}
        coref: list[SadSamLink] = []
        metadata: dict = {}
        established: dict[str, int] = {}
        found: dict[int, list[str]] = {}

        def absorb(refs, batch_first):
            """Route one call's references. Identical for both passes."""
            for ref in refs or []:
                cname = ref.get("component")
                snum = parse_snum(ref.get("sentence"))
                if snum is None or not cname or cname not in name_to_id:
                    continue
                sent = sent_map.get(snum)
                if sent is None:
                    continue
                cid = name_to_id[cname]
                key = (snum, cid)
                ant = parse_snum(ref.get("antecedent_sentence"))
                if ant is not None and not (0 < ant < snum and ant in sent_map):
                    ant = None
                if self._states_a_name(sent.text, cname):
                    ant = None
                if ant is None:
                    matched = ref.get("matched_text", "") or ""
                    if matched and matched.lower() not in sent.text.lower():
                        continue
                    if key not in named:
                        named[key] = CandidateLink(snum, sent.text, cname, cid,
                                                   matched, source="full_name")
                elif key not in metadata:
                    coref.append(SadSamLink(snum, cid, cname, source="coreference"))
                    metadata[key] = {
                        "reference": ref.get("matched_text", ""),
                        "antecedent_sentence": ant,
                        "antecedent_text": ref.get("antecedent_text", "") or "",
                        "raw_resolution": ref,
                    }
                if cname not in found.setdefault(snum, []):
                    found[snum].append(cname)
                if ant is None and snum >= batch_first:
                    established[cname] = max(established.get(cname, 0), snum)

        self.llm.set_phase("phase_25_reading")
        for n, batch in self._iter_batches(sentences, self.READING_BATCH):
            data = self._ask(
                self._prompt_reading(comp_names, mappings, batch, established),
                timeout=600, label=f"Reading batch {n}", require_present="references")
            absorb((data or {}).get("references", []), batch[0].number)
        first_pass = len(named) + len(coref)

        # pass 2: only the sentences the first read linked -- the loss the ladder
        # measured is entirely on sentences that already carry one link.
        revisit = [s for s in sentences if found.get(s.number)]
        self.llm.set_phase("phase_25_gleaning")
        for n, batch in self._iter_batches(revisit, self.GLEAN_BATCH):
            data = self._ask(
                self._prompt_reading(comp_names, mappings, batch, established,
                                     found=found),
                timeout=600, label=f"Gleaning batch {n}", require_present="references")
            absorb((data or {}).get("references", []), batch[0].number)

        print(f"    Read: {first_pass} pairs from {len(sentences)} sentences; "
              f"gleaned {len(revisit)} sentences -> {len(named) + len(coref)} total")
        self._reading = {"named": named, "coref": coref, "metadata": metadata}
        return self._reading
