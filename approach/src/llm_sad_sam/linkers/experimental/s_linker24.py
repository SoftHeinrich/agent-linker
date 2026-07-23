"""S24 — narrow, anchored recovery on top of the unchanged S21 floor."""
from __future__ import annotations

import re

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker21 import SLinker21


class SLinker24(SLinker21):
    _VARIANT_NAME = "s_linker24"

    def link(self, text_path, model_path, **kwargs):
        floor = super().link(text_path, model_path, **kwargs)
        try:
            additions = self._recover(text_path, model_path, floor)
        except Exception as exc:
            print(f"  [s24] recovery failed; keeping S21 floor: {exc}")
            additions = []
        existing = {(x.sentence_number, x.component_id) for x in floor}
        final = floor + [x for x in additions if (x.sentence_number, x.component_id) not in existing]
        self._log(
            "phase_24_anchor_recovery",
            {"floor": len(floor)},
            {"additions": len(final) - len(floor), "final": len(final)},
            final,
        )
        # S21 saves its trace before this post-floor recovery. Save again so
        # the tracked S24 trace includes the resolver and inherited coref gate.
        self._save_log(text_path)
        return final

    def _recover(self, text_path, model_path, floor):
        components = parse_pcm_repository(model_path)
        names = {c.name: c.id for c in components}
        sentences = load_sentences(text_path); sent_map = build_sent_map(sentences)
        cases = []
        # Only structurally explicit Client/Server siblings and unique technical prefixes.
        families = {}
        for name in names:
            m = re.match(r"(.+?)\s+(Client|Server)$", name, re.I)
            if m: families.setdefault(m.group(1).lower(), []).append((name, m.group(2).lower()))
        for i, sent in enumerate(sentences):
            low = sent.text.lower()
            context = " ".join(x.text.lower() for x in sentences[max(0, i-5):i+1])
            for base, members in families.items():
                if base not in context: continue
                for name, role in members:
                    if re.search(rf"\b{re.escape(role)}\b", low) and name.lower() not in low:
                        cases.append((sent, name, f"local '{base}' context with role word '{role}'"))
            for name in names:
                prefix = re.split(r"[-\s]", name)[0]
                if (any(ch.isupper() for ch in prefix) or any(ch.isdigit() for ch in prefix)) and \
                   len(prefix) >= 4 and prefix.lower() != name.lower() and \
                   re.search(rf"\b{re.escape(prefix.lower())}\b", low):
                    cases.append((sent, name, f"unique technical prefix '{prefix}'"))
        floor_keys = {(x.sentence_number, x.component_id) for x in floor}
        dedup = {
            (s.number, n): (s, n, why)
            for s, n, why in cases
            if (s.number, names[n]) not in floor_keys
        }
        cases = list(dedup.values())
        if not cases: return []
        approved = []
        for start in range(0, len(cases), 20):
            batch = cases[start:start+20]
            body = "\n\n".join(f"Case {i+1}: candidate {name}\n  Sentence: {s.text}\n  Basis: {why}" for i,(s,name,why) in enumerate(batch))
            prompt = ("Resolve only the listed anchored references. Approve only when the sentence's exact "
                      "client/server or technical shorthand clearly identifies the candidate component; otherwise abstain. "
                      "Return JSON: {\"resolutions\":[{\"case\":1,\"approve\":true}]}\n\n" + body)
            data = self._ask(prompt, phase="phase_24_anchor_resolve", require_present="resolutions", label="S24 resolver")
            yes = {int(x.get("case",0)) for x in data.get("resolutions",[]) if x.get("approve") is True}
            for i,(s,name,_) in enumerate(batch,1):
                if i in yes: approved.append(SadSamLink(s.number, names[name], name, source="s24_anchor"))
        kept, _ = self._validate_coref_links(approved, sent_map, components)
        print(f"  [s24] anchored recovery: {len(cases)} candidates, {len(kept)} approved additions")
        return kept
