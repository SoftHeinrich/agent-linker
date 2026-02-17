"""AgentLinker Ablation: Configurable variant for ablation study.

Extends AgentLinker with three feature flags to isolate the impact of:
1. Debate-validated coreference (V44-style propose+judge vs discourse-aware)
2. Implicit reference detection (enable/disable Phase 8)
3. Sliding-batch entity extraction (overlapping batches vs cap at 100)

See run_ablation.py for the 6 variant configurations.
"""

import re
from typing import Optional

from ...core import (
    SadSamLink,
    CandidateLink,
    DiscourseContext,
    Sentence,
)
from ...llm_client import LLMBackend
from .agent_linker import AgentLinker


class AgentLinkerAblation(AgentLinker):
    """AgentLinker with configurable ablation flags.

    Coref modes (coref_mode):
      - "discourse"       : AgentLinker default (discourse-aware, single pass)
      - "debate"          : V44-style propose+judge (no discourse context)
      - "discourse_judge" : Discourse-aware proposals + independent judge validation
      - "adaptive"        : Auto-select: debate for large docs (>80 sents), discourse for small

    Other flags:
      - enable_implicit   : True/False or "adaptive" (enable only for ≤80 sentences)
      - use_sliding_batch : Overlapping 100-sentence windows for entity extraction
    """

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        use_debate_coref: bool = False,
        enable_implicit=True,
        use_sliding_batch: bool = False,
        coref_mode: str = None,
        implicit_mode: str = None,
        judge_mode: str = None,
    ):
        super().__init__(backend=backend)
        # New-style flags take precedence over old boolean flags
        if coref_mode:
            self.coref_mode = coref_mode
        else:
            self.coref_mode = "debate" if use_debate_coref else "discourse"

        if implicit_mode:
            self.implicit_mode = implicit_mode
        elif isinstance(enable_implicit, str):
            self.implicit_mode = enable_implicit  # "adaptive"
        else:
            self.implicit_mode = "on" if enable_implicit else "off"

        self.use_sliding_batch = use_sliding_batch
        self.judge_mode = judge_mode  # None = default, "cot" = CoT judge
        # For backward compat
        self.use_debate_coref = self.coref_mode in ("debate", "adaptive")
        self.enable_implicit = self.implicit_mode != "off"

        print(f"  Ablation: coref={self.coref_mode}, implicit={self.implicit_mode}, judge={self.judge_mode}, sliding={use_sliding_batch}")

    def _detect_implicit_references(
        self, sentences, components, name_to_id, sent_map, discourse_model, existing_links
    ) -> list[SadSamLink]:
        """Override: skip Phase 8 based on implicit_mode."""
        skip = False
        if self.implicit_mode == "off":
            skip = True
        elif self.implicit_mode == "adaptive":
            if len(sentences) > 80:
                skip = True
                print(f"    SKIPPED (adaptive: {len(sentences)} sentences > 80)")
            else:
                print(f"    ENABLED (adaptive: {len(sentences)} sentences ≤ 80)")

        if skip:
            if self.implicit_mode == "off":
                print("    SKIPPED (ablation: no_implicit)")
            return []
        return super()._detect_implicit_references(
            sentences, components, name_to_id, sent_map, discourse_model, existing_links
        )

    def _validate_implicit_links(
        self, implicit_links, sentences, components, sent_map
    ) -> list[SadSamLink]:
        """Override: skip validation when implicit is disabled or CoT (already validated)."""
        if self.implicit_mode == "off":
            return []
        if self.implicit_mode == "adaptive" and self.doc_profile and self.doc_profile.sentence_count > 80:
            return []
        if self.implicit_mode in ("cot", "adaptive_cot"):
            # CoT already includes reasoning-based validation; skip second pass
            return implicit_links
        return super()._validate_implicit_links(implicit_links, sentences, components, sent_map)

    def _resolve_coreferences_with_discourse(
        self, sentences: list[Sentence], components: list, name_to_id: dict,
        sent_map: dict, discourse_model: dict
    ) -> list[SadSamLink]:
        """Override: select coref strategy based on coref_mode."""
        mode = self.coref_mode

        if mode == "adaptive":
            if len(sentences) > 80:
                mode = "debate"
                print(f"    Adaptive coref: debate (large doc, {len(sentences)} sents)")
            else:
                mode = "discourse"
                print(f"    Adaptive coref: discourse (small doc, {len(sentences)} sents)")

        if mode == "discourse":
            return super()._resolve_coreferences_with_discourse(
                sentences, components, name_to_id, sent_map, discourse_model
            )
        elif mode == "debate":
            return self._resolve_coreferences_with_debate(
                sentences, components, name_to_id, sent_map
            )
        elif mode == "discourse_judge":
            return self._resolve_coreferences_discourse_judge(
                sentences, components, name_to_id, sent_map, discourse_model
            )
        else:
            return super()._resolve_coreferences_with_discourse(
                sentences, components, name_to_id, sent_map, discourse_model
            )

    def _resolve_coreferences_with_debate(
        self, sentences: list[Sentence], components: list,
        name_to_id: dict, sent_map: dict
    ) -> list[SadSamLink]:
        """V44-style propose+judge coreference resolution.

        Key differences from AgentLinker's discourse-aware coref:
        - Two LLM calls per batch (propose + judge) instead of single call
        - Batch size 20 (vs 12)
        - No DiscourseContext — uses raw context window
        - Judge can adjust confidence and reject proposals
        """
        comp_names = self._get_comp_names(components)
        threshold = self.thresholds.coref_threshold

        ctx = []
        if self.learned_patterns and self.learned_patterns.action_indicators:
            ctx.append(f"ACTION: {', '.join(self.learned_patterns.action_indicators[:3])}")
        if self.learned_patterns and self.learned_patterns.subprocess_terms:
            ctx.append(f"Subprocesses (don't link): {', '.join(list(self.learned_patterns.subprocess_terms)[:5])}")

        all_coref = []
        for batch_start in range(0, len(sentences), 20):
            batch = sentences[batch_start:min(batch_start + 20, len(sentences))]
            ctx_start = max(0, batch_start - self.CONTEXT_WINDOW)
            ctx_sents = sentences[ctx_start:batch_start + 20]

            doc_lines = [
                f"{'*' if s.number >= batch[0].number else ' '}S{s.number}: {s.text}"
                for s in ctx_sents
            ]

            # Pass 1: Propose resolutions
            prompt1 = f"""Resolve pronoun references to components.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DOCUMENT (* = analyze):
{chr(10).join(doc_lines)}

Find pronouns (it, they, this) that refer to components.

Return JSON:
{{"resolutions": [{{"sentence": N, "pronoun": "it", "component": "Name", "confidence": 0.0-1.0, "reasoning": "why"}}]}}
JSON only:"""

            data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=100))
            if not data1:
                continue

            proposed = data1.get("resolutions", [])
            if not proposed:
                continue

            # Pass 2: Judge validates proposals
            proposal_text = [
                f"S{r['sentence']}: \"{r.get('pronoun', '?')}\" -> {r['component']} ({r.get('reasoning', '')[:40]})"
                for r in proposed[:12]
            ]

            prompt2 = f"""JUDGE: Validate these pronoun-to-component resolutions.

COMPONENTS: {', '.join(comp_names)}

PROPOSALS:
{chr(10).join(proposal_text)}

CONTEXT:
{chr(10).join(doc_lines[:15])}

Reject if:
- Pronoun might refer to something else
- Reference is to a subprocess
- Too far from actual component mention

Return JSON:
{{"judgments": [{{"sentence": N, "approve": true/false, "adjusted_confidence": 0.0-1.0}}]}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=100))

            judgments = {}
            if data2:
                for j in data2.get("judgments", []):
                    judgments[j.get("sentence")] = (
                        j.get("approve", False),
                        j.get("adjusted_confidence", 0.8),
                    )

            for res in proposed:
                snum, comp = res.get("sentence"), res.get("component")
                if not (snum and comp and comp in name_to_id):
                    continue

                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue

                if snum in judgments:
                    approved, conf = judgments[snum]
                    if not approved or conf < threshold:
                        continue
                else:
                    conf = res.get("confidence", 0.85)
                    if conf < threshold:
                        continue

                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.subprocess_terms:
                    if any(t.lower() in sent.text.lower() for t in self.learned_patterns.subprocess_terms):
                        conf -= 0.05
                        if conf < threshold:
                            continue

                if comp in self.model_knowledge.architectural_names:
                    conf = min(1.0, conf + 0.02)

                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, conf, "coreference"))

        return all_coref

    def _resolve_coreferences_discourse_judge(
        self, sentences: list[Sentence], components: list, name_to_id: dict,
        sent_map: dict, discourse_model: dict
    ) -> list[SadSamLink]:
        """Hybrid: discourse-aware proposals + independent judge validation.

        Pass 1: Uses DiscourseContext (paragraph topic, recent mentions, likely referent)
                to generate rich proposals — same as AgentLinker's discourse coref.
        Pass 2: Independent judge validates proposals without seeing the discourse
                context, preventing confirmation bias.
        """
        comp_names = self._get_comp_names(components)
        threshold = self.thresholds.coref_threshold
        all_coref = []

        pronoun_sents = [s for s in sentences if self.PRONOUN_PATTERN.search(s.text)]

        batch_size = 12
        for batch_start in range(0, len(pronoun_sents), batch_size):
            batch = pronoun_sents[batch_start:batch_start + batch_size]

            # Pass 1: Discourse-aware proposals (same as AgentLinker)
            cases = []
            for sent in batch:
                ctx = discourse_model.get(sent.number, DiscourseContext())
                prev_context = []
                for i in range(1, self.CONTEXT_WINDOW + 1):
                    prev = sent_map.get(sent.number - i)
                    if prev:
                        prev_context.append(f"S{prev.number}: {prev.text[:70]}")
                cases.append({
                    "sent": sent,
                    "ctx": ctx,
                    "prev": prev_context,
                    "likely": ctx.get_likely_referent()
                })

            prompt1 = f"""Resolve pronoun references using discourse context.

COMPONENTS: {', '.join(comp_names)}

For each sentence, identify what pronouns (it, they, this, etc.) refer to.
Use the discourse context to make accurate resolutions.

"""
            for i, case in enumerate(cases):
                ctx = case["ctx"]
                prompt1 += f"--- Case {i+1}: S{case['sent'].number} ---\n"
                prompt1 += f"DISCOURSE:\n"
                prompt1 += f"  Recent mentions: {ctx.get_context_summary(case['sent'].number)}\n"
                prompt1 += f"  Paragraph topic: {ctx.paragraph_topic or 'None'}\n"
                prompt1 += f"  Likely referent: {case['likely'] or 'Unknown'}\n"
                if case["prev"]:
                    prompt1 += f"PREVIOUS:\n  " + "\n  ".join(reversed(case["prev"])) + "\n"
                prompt1 += f">>> SENTENCE: {case['sent'].text}\n\n"

            prompt1 += """Return JSON:
{"resolutions": [{"case": 1, "sentence": N, "pronoun": "it", "component": "Name", "confidence": 0.0-1.0, "reasoning": "why"}]}

Guidelines:
- Use discourse context (recent mentions, paragraph topic) to resolve ambiguity
- Higher confidence if component was explicit subject in previous sentence
- Lower confidence if pronoun could refer to multiple things
- Skip if uncertain

JSON only:"""

            data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=150))
            if not data1:
                continue

            proposed = data1.get("resolutions", [])
            if not proposed:
                continue

            # Pass 2: Independent judge (NO discourse context — prevents confirmation bias)
            proposal_text = []
            for r in proposed[:15]:
                snum = r.get("sentence")
                sent = sent_map.get(snum) if snum else None
                prev = sent_map.get(snum - 1) if snum else None
                proposal_text.append(
                    f"S{snum}: \"{r.get('pronoun', '?')}\" -> {r['component']}\n"
                    f"    prev: {prev.text[:60] if prev else 'N/A'}...\n"
                    f"    sent: {sent.text[:80] if sent else '?'}"
                )

            prompt2 = f"""JUDGE: Validate these pronoun-to-component resolutions.
Review ONLY based on the sentence text and previous sentence.

COMPONENTS: {', '.join(comp_names)}

PROPOSALS:
{chr(10).join(proposal_text)}

Reject if:
- The pronoun could reasonably refer to something other than the proposed component
- The reference is to a subprocess or internal detail, not a top-level component
- The previous sentence doesn't clearly establish the component as the topic

Return JSON:
{{"judgments": [{{"sentence": N, "approve": true/false, "adjusted_confidence": 0.0-1.0}}]}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=120))

            judgments = {}
            if data2:
                for j in data2.get("judgments", []):
                    judgments[j.get("sentence")] = (
                        j.get("approve", False),
                        j.get("adjusted_confidence", 0.8),
                    )

            for res in proposed:
                comp = res.get("component")
                snum = res.get("sentence")
                if not (comp and snum and comp in name_to_id):
                    continue
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue

                if snum in judgments:
                    approved, conf = judgments[snum]
                    if not approved or conf < threshold:
                        continue
                else:
                    # No judgment — use proposal confidence but require higher bar
                    conf = res.get("confidence", 0.85)
                    if conf < threshold + 0.05:
                        continue

                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.subprocess_terms:
                    if any(t.lower() in sent.text.lower() for t in self.learned_patterns.subprocess_terms):
                        conf -= 0.05
                        if conf < threshold:
                            continue

                if comp in self.model_knowledge.architectural_names:
                    conf = min(1.0, conf + 0.02)

                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, conf, "coreference"))

        return all_coref

    def _detect_implicit_references(
        self, sentences, components, name_to_id, sent_map, discourse_model, existing_links
    ) -> list[SadSamLink]:
        """Override: skip Phase 8 based on implicit_mode, or use CoT."""
        if self.implicit_mode == "cot":
            return self._detect_implicit_cot(
                sentences, components, name_to_id, sent_map, discourse_model, existing_links
            )
        elif self.implicit_mode == "adaptive_cot":
            if len(sentences) > 80:
                print(f"    CoT implicit (adaptive: {len(sentences)} sentences > 80, using CoT)")
                return self._detect_implicit_cot(
                    sentences, components, name_to_id, sent_map, discourse_model, existing_links
                )
            else:
                print(f"    Standard implicit (adaptive: {len(sentences)} sentences ≤ 80)")
                return super()._detect_implicit_references(
                    sentences, components, name_to_id, sent_map, discourse_model, existing_links
                )

        skip = False
        if self.implicit_mode == "off":
            skip = True
        elif self.implicit_mode == "adaptive":
            if len(sentences) > 80:
                skip = True
                print(f"    SKIPPED (adaptive: {len(sentences)} sentences > 80)")
            else:
                print(f"    ENABLED (adaptive: {len(sentences)} sentences ≤ 80)")

        if skip:
            if self.implicit_mode == "off":
                print("    SKIPPED (ablation: no_implicit)")
            return []
        return super()._detect_implicit_references(
            sentences, components, name_to_id, sent_map, discourse_model, existing_links
        )

    def _detect_implicit_cot(
        self, sentences, components, name_to_id, sent_map, discourse_model, existing_links
    ) -> list[SadSamLink]:
        """CoT implicit reference detection — focused recovery approach.

        Instead of scanning ALL sentences (which generates too many FPs),
        focus on sentences adjacent to component mentions that look like
        continuation of that component's description. Uses CoT to verify.
        """
        comp_names = self._get_comp_names(components)
        threshold = self.thresholds.implicit_threshold  # 0.85
        implicit_links = []

        # Build map: for each component, which sentences mention it
        comp_sentences = {}  # comp -> sorted list of sentence numbers
        for sent in sentences:
            text_lower = sent.text.lower()
            for c in comp_names:
                if c.lower() in text_lower:
                    comp_sentences.setdefault(c, []).append(sent.number)

        # Also include synonyms/abbreviations from doc_knowledge
        if self.doc_knowledge:
            for alias, comp in {**self.doc_knowledge.abbreviations, **self.doc_knowledge.synonyms}.items():
                if comp in [c for c in comp_names]:
                    for sent in sentences:
                        if alias.lower() in sent.text.lower() and sent.number not in comp_sentences.get(comp, []):
                            comp_sentences.setdefault(comp, []).append(sent.number)

        # Find candidate sentences: no explicit mention but within 3 sentences of a mention
        candidates = []
        for sent in sentences:
            text_lower = sent.text.lower()
            has_explicit = any(c.lower() in text_lower for c in comp_names)
            if has_explicit:
                continue

            # Check if near a component mention
            for comp, mention_sents in comp_sentences.items():
                for ms in mention_sents:
                    dist = sent.number - ms
                    if 1 <= dist <= 3:  # 1-3 sentences AFTER a mention
                        cid = name_to_id.get(comp, '')
                        if (sent.number, cid) not in existing_links:
                            candidates.append((sent, comp, dist, ms))
                            break
                else:
                    continue
                break

        if not candidates:
            return []

        print(f"    CoT candidates (near mention, no explicit): {len(candidates)}")

        # Process in batches of 10
        batch_size = 10
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start:batch_start + batch_size]

            prompt = f"""For each sentence, determine if it continues describing the nearby component.

COMPONENTS: {', '.join(comp_names)}

"""
            for i, (sent, comp, dist, mention_sent) in enumerate(batch):
                # Show the mention sentence and the candidate
                context_lines = []
                for d in range(min(dist + 1, 4), 0, -1):
                    ctx_sent = sent_map.get(sent.number - d)
                    if ctx_sent:
                        context_lines.append(f"  S{ctx_sent.number}: {ctx_sent.text[:120]}")
                context_lines.append(f"  >>> S{sent.number}: {sent.text[:120]}")

                prompt += f"--- Case {i+1}: S{sent.number}, nearby component: {comp} (mentioned at S{mention_sent}) ---\n"
                prompt += "\n".join(context_lines) + "\n\n"

            prompt += """For each case, think step by step:
1. Is this sentence about the SAME component as the nearby mention? Or has the topic shifted?
2. Is the component the ACTOR (subject performing the action) in this sentence?
3. Could this sentence plausibly describe a DIFFERENT component instead?

Return JSON:
{"analysis": [{"case": 1, "sentence": N, "step1": "same topic / topic shifted", "step2": "actor: yes/no", "step3": "could be other: yes/no", "component": "Name or null", "confidence": 0.0-1.0}]}

Rules:
- ONLY approve if step1="same topic" AND step2="yes" AND step3="no"
- Set component to null for all other cases
- Sentences listing sub-features of a component (e.g. "Managing X", "Providing Y") after
  "The component is responsible for:" ARE implicit refs — approve these
- Sentences about system-wide behavior or different components = null
- Confidence >= 0.88 required

JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
            if not data:
                continue

            for det in data.get("analysis", []):
                comp = det.get("component")
                snum = det.get("sentence")
                conf = det.get("confidence", 0)

                if not comp or comp == "null" or not snum:
                    continue
                if comp not in name_to_id:
                    continue

                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue

                if conf < 0.88:
                    continue

                if (snum, name_to_id[comp]) in existing_links:
                    continue

                # Verify step conditions
                s1 = str(det.get("step1", "")).lower()
                s2 = str(det.get("step2", "")).lower()
                s3 = str(det.get("step3", "")).lower()

                if "shift" in s1 or "no" in s2 or "yes" in s3:
                    continue

                conf = min(conf, 0.87)
                implicit_links.append(SadSamLink(snum, name_to_id[comp], comp, conf, "implicit"))

        return implicit_links

    def _agent_judge_review(self, links, sentences, components, sent_map, transarc_set) -> list[SadSamLink]:
        """Override: use CoT judge or TransArc review based on judge_mode."""
        if getattr(self, 'judge_mode', None) == 'cot':
            return self._agent_judge_review_cot(links, sentences, components, sent_map, transarc_set)
        elif getattr(self, 'judge_mode', None) == 'cot_transarc':
            return self._agent_judge_review_cot_transarc(links, sentences, components, sent_map, transarc_set)
        return super()._agent_judge_review(links, sentences, components, sent_map, transarc_set)

    def _agent_judge_review_cot(self, links, sentences, components, sent_map, transarc_set) -> list[SadSamLink]:
        """CoT Agent-as-Judge: reason step-by-step before approving/rejecting.

        Same structure as base judge (TransArc links pass through, review non-transarc),
        but with chain-of-thought prompting for better reasoning.
        """
        if len(links) < 5:
            return links

        comp_names = self._get_comp_names(components)
        transarc = [l for l in links if (l.sentence_number, l.component_id) in transarc_set]
        non_transarc = [l for l in links if (l.sentence_number, l.component_id) not in transarc_set]

        if not non_transarc:
            return transarc

        cases = []
        for i, l in enumerate(non_transarc[:30]):
            sent = sent_map.get(l.sentence_number)
            context = []
            if l.source in ("implicit", "coreference"):
                prev2 = sent_map.get(l.sentence_number - 2)
                if prev2:
                    context.append(f"    prev2: {prev2.text[:60]}...")
            prev = sent_map.get(l.sentence_number - 1)
            if prev:
                context.append(f"    prev: {prev.text[:60]}...")
            context.append(f"    >>> S{l.sentence_number}: {sent.text if sent else '?'}")
            cases.append(f"Case {i+1}: S{l.sentence_number} -> {l.component_name} (src:{l.source}, conf:{l.confidence:.2f})\n{chr(10).join(context)}")

        prompt = f"""JUDGE: Review each trace link using step-by-step reasoning.

COMPONENTS: {', '.join(comp_names)}

LINKS:
{chr(10).join(cases)}

For each case, reason through these steps:
1. What does this sentence actually describe? (brief paraphrase)
2. Is the component the ACTOR/SUBJECT performing the action, or just mentioned?
3. For coreference links: does the pronoun clearly refer to THIS component?

Return JSON:
{{"judgments": [{{"case": 1, "step1": "describes X", "step2": "actor / not actor", "step3": "clear ref / ambiguous", "approve": true/false, "reason": "brief"}}]}}

REJECT if step2 = "not actor" or step3 = "ambiguous".
APPROVE if the component is clearly the actor or subject being described.

JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=180))
        result = transarc.copy()

        num_judged = min(30, len(non_transarc))
        if data:
            judged = set()
            for j in data.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < num_judged:
                    judged.add(idx)
                    if j.get("approve", False):
                        result.append(non_transarc[idx])
            for i in range(num_judged):
                if i not in judged:
                    result.append(non_transarc[i])
            for i in range(num_judged, len(non_transarc)):
                result.append(non_transarc[i])
        else:
            result.extend(non_transarc)

        return result

    def _agent_judge_review_cot_transarc(self, links, sentences, components, sent_map, transarc_set) -> list[SadSamLink]:
        """Two-stage review: standard review for non-transarc + CoT review for ambiguous TransArc links.

        The base judge passes TransArc links through unreviewed. This variant adds a
        focused CoT review for TransArc links involving ambiguous component names
        (e.g. 'Logic', 'Common', 'Client') where the name could be a common English word.
        """
        # First: run standard judge review for non-transarc links
        result_after_standard = super()._agent_judge_review(links, sentences, components, sent_map, transarc_set)

        # Identify TransArc links with ambiguous component names
        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()
        if not ambiguous:
            return result_after_standard

        transarc_ambiguous = [
            l for l in result_after_standard
            if (l.sentence_number, l.component_id) in transarc_set and l.component_name in ambiguous
        ]
        transarc_safe = [
            l for l in result_after_standard
            if not ((l.sentence_number, l.component_id) in transarc_set and l.component_name in ambiguous)
        ]

        if not transarc_ambiguous:
            return result_after_standard

        print(f"    CoT TransArc review: {len(transarc_ambiguous)} ambiguous TransArc links")

        comp_names = self._get_comp_names(components)
        cases = []
        for i, l in enumerate(transarc_ambiguous[:20]):
            sent = sent_map.get(l.sentence_number)
            prev = sent_map.get(l.sentence_number - 1)
            context = []
            if prev:
                context.append(f"    prev: {prev.text[:80]}")
            context.append(f"    >>> S{l.sentence_number}: {sent.text if sent else '?'}")
            cases.append(f"Case {i+1}: S{l.sentence_number} -> {l.component_name}\n{chr(10).join(context)}")

        prompt = f"""Review whether these sentences reference architecture components or use common words.

COMPONENTS: {', '.join(comp_names)}
NOTE: Some component names are also common English words (e.g. "{', '.join(list(ambiguous)[:3])}").

LINKS TO REVIEW:
{chr(10).join(cases)}

For each case, think step by step:
1. Quote the exact text that matches the component name.
2. In this context, is it used as a PROPER NAME for the component, or as a COMMON ENGLISH WORD?
   - Proper name: "The Logic component handles...", "Logic manages...", "sent to Logic"
   - Common word: "cascade logic", "business logic in the MeetingActor", "client-side"
3. Approve only if it's a proper name reference.

Return JSON:
{{"judgments": [{{"case": 1, "matched_text": "exact quote", "usage": "proper name / common word", "approve": true/false}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
        result = transarc_safe.copy()

        if data:
            judged = set()
            for j in data.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < len(transarc_ambiguous):
                    judged.add(idx)
                    if j.get("approve", False):
                        result.append(transarc_ambiguous[idx])
            # Keep unjudged
            for i in range(len(transarc_ambiguous)):
                if i not in judged:
                    result.append(transarc_ambiguous[i])
        else:
            result.extend(transarc_ambiguous)

        return result

    def _extract_entities(self, sentences, components, name_to_id, sent_map) -> list[CandidateLink]:
        """Override: use sliding batch if flag is set."""
        if not self.use_sliding_batch:
            return super()._extract_entities(sentences, components, name_to_id, sent_map)

        return self._extract_entities_sliding(sentences, components, name_to_id, sent_map)

    def _extract_entities_sliding(self, sentences, components, name_to_id, sent_map) -> list[CandidateLink]:
        """Sliding-batch entity extraction with overlapping windows.

        Processes sentences in batches of 100 with 20-sentence overlap:
        [0:100], [80:180], [160:260], etc.
        Deduplicates by (sentence_number, component_id) across batches.
        """
        comp_names = self._get_comp_names(components)
        comp_lower = {n.lower() for n in comp_names}

        mappings = []
        if self.doc_knowledge:
            mappings.extend([f"{a}={c}" for a, c in self.doc_knowledge.abbreviations.items()])
            mappings.extend([f"{s}={c}" for s, c in self.doc_knowledge.synonyms.items()])
            mappings.extend([f"{p}={c}" for p, c in self.doc_knowledge.partial_references.items()])

        batch_size = 100
        overlap = 20
        step = batch_size - overlap  # 80

        all_candidates = {}  # (snum, cid) -> CandidateLink for dedup

        batch_idx = 0
        start = 0
        while start < len(sentences):
            end = min(start + batch_size, len(sentences))
            batch = sentences[start:end]
            batch_idx += 1

            if len(sentences) > batch_size:
                print(f"    Batch {batch_idx}: sentences S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")

            prompt = f"""Extract component references.

COMPONENTS: {', '.join(comp_names)}
{f'ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N, "component": "Name", "matched_text": "text", "match_type": "exact|synonym|partial"}}]}}
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
            if data:
                for ref in data.get("references", []):
                    snum, cname = ref.get("sentence"), ref.get("component")
                    if not (snum and cname and cname in name_to_id):
                        continue
                    sent = sent_map.get(snum)
                    if not sent or self._in_dotted_path(sent.text, cname):
                        continue

                    matched = ref.get("matched_text", "").lower()
                    is_exact = matched in comp_lower or cname.lower() in matched
                    needs_val = not is_exact or ref.get("match_type") != "exact" or \
                               (self.model_knowledge and cname in self.model_knowledge.ambiguous_names)

                    key = (snum, name_to_id[cname])
                    candidate = CandidateLink(
                        snum, sent.text, cname, name_to_id[cname],
                        ref.get("matched_text", ""), 0.85, "entity",
                        ref.get("match_type", "exact"), needs_val,
                    )
                    # Keep highest-confidence duplicate across overlapping batches
                    if key not in all_candidates or candidate.confidence > all_candidates[key].confidence:
                        all_candidates[key] = candidate

            start += step
            # Avoid tiny trailing batches
            if start < len(sentences) and len(sentences) - start < overlap:
                break

        print(f"    Total unique candidates across {batch_idx} batches: {len(all_candidates)}")
        return list(all_candidates.values())
