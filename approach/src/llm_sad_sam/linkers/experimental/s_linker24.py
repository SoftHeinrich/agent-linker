"""S24 — narrow, anchored recovery on top of the unchanged S21 floor."""
from __future__ import annotations

import re

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker21 import SLinker21


class SLinker24(SLinker21):
    _VARIANT_NAME = "s_linker24"

    def link(self, text_path, model_path, **kwargs):
        floor = super().link(text_path, model_path, **kwargs)
        self._s24_stats = {"eligible": 0, "resolver_approved": 0, "kept": 0}
        try:
            additions = self._recover(text_path, model_path, floor)
        except Exception as exc:
            print(f"  [s24] recovery failed; keeping S21 floor: {exc}")
            additions = []
        existing = {(x.sentence_number, x.component_id) for x in floor}
        final = floor + [x for x in additions if (x.sentence_number, x.component_id) not in existing]
        self._log(
            "phase_24_anchor_recovery",
            {"floor": len(floor), "eligible": self._s24_stats["eligible"]},
            {**self._s24_stats, "additions": len(final) - len(floor), "final": len(final)},
            final,
        )
        # S21 saves its trace before this post-floor recovery. Save again so
        # the tracked S24 trace includes the resolver and anchored validator.
        self._save_log(text_path)
        return final

    @staticmethod
    def _anchor_before(sentences, index, component_name):
        """Find a nearby explicit component mention that can ground a shorthand."""
        needle = re.compile(rf"\b{re.escape(component_name)}\b", re.I)
        for candidate in reversed(sentences[max(0, index - 5):index]):
            if needle.search(candidate.text):
                return candidate
        return None

    def _recover(self, text_path, model_path, floor):
        components = parse_pcm_repository(model_path)
        names = {c.name: c.id for c in components}
        sentences = load_sentences(text_path)
        cases = []

        # Structural sibling candidates. A role word alone is insufficient: the
        # exact candidate must have been explicitly named in the prior local context.
        families = {}
        for name in names:
            match = re.match(r"(.+?)\s+(Client|Server)$", name, re.I)
            if match:
                families.setdefault(match.group(1).lower(), []).append((name, match.group(2).lower()))

        # A prefix is eligible only when it identifies exactly one component.
        prefixes = {}
        for name in names:
            prefix = re.split(r"[-\s]", name)[0]
            if any(ch.isupper() for ch in prefix) or any(ch.isdigit() for ch in prefix):
                prefixes.setdefault(prefix.lower(), []).append(name)

        for index, sent in enumerate(sentences):
            low = sent.text.lower()
            context = " ".join(x.text.lower() for x in sentences[max(0, index - 5):index])
            for base, members in families.items():
                if base not in context:
                    continue
                for name, role in members:
                    anchor = self._anchor_before(sentences, index, name)
                    if anchor and re.search(rf"\b{re.escape(role)}\b", low) and name.lower() not in low:
                        cases.append((sent, name, anchor, f"local '{base}' anchor with role '{role}'"))
            for prefix, component_names in prefixes.items():
                if len(component_names) != 1 or len(prefix) < 4 or not re.search(rf"\b{re.escape(prefix)}\b", low):
                    continue
                name = component_names[0]
                anchor = self._anchor_before(sentences, index, name)
                if anchor and name.lower() != low:
                    cases.append((sent, name, anchor, f"unique technical prefix '{prefix}'"))

        floor_keys = {(x.sentence_number, x.component_id) for x in floor}
        cases = list({
            (sent.number, name): (sent, name, anchor, basis)
            for sent, name, anchor, basis in cases
            if (sent.number, names[name]) not in floor_keys
        }.values())
        self._s24_stats["eligible"] = len(cases)
        if not cases:
            return []

        resolver_approved = []
        for start in range(0, len(cases), 20):
            batch = cases[start:start + 20]
            body = "\n\n".join(
                f"Case {i}: candidate {name}\n"
                f"  Source sentence: {sent.text}\n"
                f"  Explicit anchor S{anchor.number}: {anchor.text}\n"
                f"  Basis: {basis}"
                for i, (sent, name, anchor, basis) in enumerate(batch, 1)
            )
            prompt = (
                "Resolve only the listed locally anchored references. Approve only if an EXACT phrase "
                "in the source sentence denotes the candidate component using the explicit anchor. "
                "Reject hardware/generic uses, code or package paths, and diagram/caption text. "
                "Return JSON: {\"resolutions\":[{\"case\":1,\"approve\":true,\"phrase\":\"exact source phrase\"}]}\n\n"
                + body
            )
            data = self._ask(prompt, phase="phase_24_anchor_resolve", require_present="resolutions", label="S24 resolver")
            for item in data.get("resolutions", []):
                case = int(item.get("case", 0))
                phrase = str(item.get("phrase", "")).strip()
                if not item.get("approve") or not (1 <= case <= len(batch)):
                    continue
                sent, name, anchor, basis = batch[case - 1]
                if phrase and phrase.lower() in sent.text.lower():
                    resolver_approved.append((sent, name, anchor, basis, phrase))

        self._s24_stats["resolver_approved"] = len(resolver_approved)
        kept = self._validate_anchored_links(resolver_approved, names)
        self._s24_stats["kept"] = len(kept)
        print(
            "  [s24] anchored recovery: "
            f"{len(cases)} eligible, {len(resolver_approved)} resolver-approved, {len(kept)} additions"
        )
        return kept

    def _validate_anchored_links(self, candidates, names):
        """Validate an anchored reference, not an S21 pronoun/coreference claim."""
        kept = []
        for start in range(0, len(candidates), 20):
            batch = candidates[start:start + 20]
            body = "\n\n".join(
                f"Case {i}: target {name}\n"
                f"  Source sentence: {sent.text}\n"
                f"  Exact referring phrase: {phrase}\n"
                f"  Anchor S{anchor.number}: {anchor.text}\n"
                f"  Eligibility: {basis}"
                for i, (sent, name, anchor, basis, phrase) in enumerate(batch, 1)
            )
            prompt = (
                "Validate each bounded anchored reference. Approve only if the exact referring phrase "
                "in the source sentence denotes the target component established by the anchor AND the "
                "source sentence makes an architectural claim about that component. Reject generic/hardware "
                "uses, another component with the same prefix, code/package paths, and diagram/caption text. "
                "Return JSON: {\"validations\":[{\"case\":1,\"claim\":\"exact source quote\",\"approve\":true}]}.\n\n"
                + body
            )
            data = self._ask(prompt, phase="phase_24_anchor_validate", require_present="validations", label="S24 anchor validator")
            approved = {int(item.get("case", 0)) for item in data.get("validations", []) if item.get("approve") is True}
            for i, (sent, name, _, _, _) in enumerate(batch, 1):
                if i in approved:
                    kept.append(SadSamLink(sent.number, names[name], name, source="s24_anchor"))
        return kept
