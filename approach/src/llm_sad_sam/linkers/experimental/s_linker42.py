"""s_linker42 — one judging call, and a mention label with three values, not five.

This is `s_linker36` (one full-name judging call carrying both criteria, rather than
s_linker25's two calls, one per criterion) with the second simplification that
s_linker38's own traces support: the mention label shrinks from five values to three.

Two audits of s_linker38's six runs decided both changes
(`pilot/s38_audit.py`, `../results/s38_audit/`).

  * THE SECOND SAMPLE IS INERT. s_linker38 asks the one judging prompt twice and ANDs
    the verdicts, and describes that self-agreement as where its precision comes
    from. Over six runs the two samples split on **1.0 of 174.7 cases (0.6%)** --
    0.3 gold and 0.7 not -- so the gate decides almost nothing and the stated
    mechanism is not what holds s_linker38 at parity. One sample is the honest form,
    and it is what this variant asks. The two *criteria* stay: joined to the gold
    standard, uniqueness rejects 3.2 candidates per run that relevance would keep,
    2.7 of them false positives.
  * THE FIVE-VALUE MENTION LABEL COLLAPSES TO THREE. The label tells the judge how
    the name is present in the sentence. Three of its five values behave
    identically in the judge's verdicts -- proper-case standalone 96.9% approved
    (107.0 cases per run), lowercase 100.0% (25.2), indirect 100.0% (1.7) -- while
    the two that matter are clearly separated: via a discovered alias 82.8% (33.0)
    and only-inside-a-qualified-identifier 57.4% (7.8, gold rate 25.5%). So the
    distinction the label needs to draw is *how* the name is present, not what case
    it is written in:

      NAME             the name appears, outside any qualified identifier
      ALIAS            only a name the document introduced for it appears
      QUALIFIED_ONLY   every occurrence sits inside a qualified identifier

    The residual fifth value -- neither the name nor a known alias matched -- had
    1.7 cases per run and is not a way for a name to be present, so the label is
    omitted rather than guessed. That also removes the last case-sensitivity rule
    in the workflow (`matched == comp_name`).

Everything else is s_linker25's. Reference band: s_linker25 macro F1 96.42 +/- 0.43,
macro F2 95.37 +/- 0.57 over six runs; s_linker38 95.95 / 95.58 over six.
Every rubric is generic English structure -- no benchmark vocabulary.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from llm_sad_sam.core.data_types_v2 import (
    SadSamLink, CandidateLink, DocumentKnowledge,
)
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.linkers.experimental.helper_v3 import (
    parse_snum, get_comp_names,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend, LLMResponse

# ─────────────────────────────────────────────────────────────────────────────
# Prompt constants. Byte-identical to the inherited chain.
# ─────────────────────────────────────────────────────────────────────────────

DOC_KNOWLEDGE_JUDGE_RULES = """An alias is valid when the document establishes an equivalence between a phrase and a single named component. An alias is invalid when the phrase is generic vocabulary, names the whole system, or names a different entity. An alias is also invalid when it names a grouping that encompasses multiple elements, because it identifies a grouping rather than a single named unit. When uncertain, prefer APPROVE."""

DOC_KNOWLEDGE_EXTRACTION_RULES = """Find surface forms the document uses to refer to a single named component (introduced short forms, alternate names, or words of multi-word names when they alone clearly mean the full name). Reject terms whose ordinary English use dominates."""

ALIAS_EXCLUSION_RULES = """Qualified-name fragments (package- or member-access paths of the form X.Y or X.Y.Z) are NOT aliases — do not include them."""


ENTITY_EXTRACTION_RULES = """Include a reference when the sentence refers to the component by name, alias, or as a participant in a described interaction. Exclude when the name appears only inside a code-level path — even if the compound identifier is semantically related to the component — or as ordinary English with no architectural intent. Favor inclusion."""


P1_FOCUS = (
    "Check architectural participation: does the sentence name this "
    "component as an architectural participant — performing operations, "
    "providing services, or taking part in the described system behavior, "
    "and not just as a qualified-name identifier (e.g. a package- or "
    "member-access path X.Y.Z)?"
)

P2_FOCUS = (
    "Check referential specificity: is the component name used to identify "
    "this specific architectural element, or does it serve as a generic "
    "technical term in this sentence?"
)


COREF_VALIDATION_FOCUS = (
    "Check coref resolution: does the pronoun, 'it', 'they', 'the service', "
    "or similar noun phrase that refers back in this sentence actually refer to "
    "the named component as an architectural participant — performing "
    "operations, providing services, or being the grammatical topic of the "
    "sentence?"
)

COREF_RULES = """For each case, decide whether a pronoun or noun phrase that refers back in the target sentence refers back to a component named or aliased earlier in the context. Resolve when: (a) the component's name or a known alias appears in the surrounding context sentences, or (b) only one component has been introduced in the immediately preceding sentences — treat it as the topic of the surrounding section and resolve role-referential phrases ("it", "the module", "the service", "the component", "the system") to that topic even without a direct name repetition. Avoid resolving when two or more equally plausible antecedents exist. Known aliases include the terminal word(s) of a multi-word name, documented abbreviations, and alternate forms used in the document."""

# Full-name gate — lenient: a stated name is a link unless a reject signal fires.
LAYERED_ENTITY_RULES = (
    "Approve the link by default: the component is named here and the document treats "
    "it as part of the system. A bare mention, a heading, or a list that includes the "
    "component name all count as valid links — approve them. Reject ONLY when one of "
    "these clearly holds: (1) the component is referred to only through a code-level or "
    "package/member path of the form x.y or x.y.z, even if that path is described as "
    "doing something; (2) the mention is negated (it is NOT a ...); (3) the matching "
    "word actually names a DIFFERENT entity; (4) the matching word is used as a generic "
    "technique or technology term, not as this system's component. When none of these "
    "reject-conditions clearly applies, approve."
)

# Coreference gate — strict: the component is NOT named in the sentence, so demand
# a genuine referring expression plus an architectural claim.
LAYERED_COREF_RULES = (
    "These are coreference links: a pronoun or noun phrase in the sentence is claimed to "
    "refer back to the component, which is NOT named in the sentence itself. Approve only "
    "when the sentence contains a genuine referring expression (a pronoun or definite "
    "noun phrase) that unambiguously points to THIS component AND the sentence makes an "
    "architectural claim about it (it performs an operation, provides/consumes a service, "
    "stores or routes data, connects to another element). Reject when: the sentence is a "
    "bare continuation fragment, gerund phrase, or list item with no referring expression; "
    "the antecedent could equally be a different component; or the reference is only to a "
    "code/package path (x.y.z). When uncertain, reject."
)

# ─────────────────────────────────────────────────────────────────────────────
# Tracing infrastructure — per-LLM-call audit trail
# ─────────────────────────────────────────────────────────────────────────────

_phase_local = threading.local()


def _current_phase() -> str:
    return getattr(_phase_local, "phase", "unknown")


class _TracingLLMClient:
    """Delegating wrapper that records every query() into a phase-tagged trace."""

    def __init__(self, inner: LLMClient, sink: list[dict]):
        self._inner = inner
        self._sink = sink
        self._sink_lock = threading.Lock()

    def set_phase(self, name: str) -> None:
        _phase_local.phase = name

    def query(self, prompt: str, timeout: int = 180, max_retries: int = 3) -> LLMResponse:
        phase = _current_phase()
        t0 = time.time()
        try:
            resp = self._inner.query(prompt, timeout=timeout, max_retries=max_retries)
        except Exception as exc:
            record = {
                "phase": phase, "ts": t0,
                "elapsed_s": round(time.time() - t0, 3),
                "timeout": timeout, "max_retries": max_retries,
                "prompt": prompt,
                "response_text": None,
                "success": False,
                "error": f"FATAL: {exc}",
                "latency_ms": None,
                "model": None,
            }
            with self._sink_lock:
                self._sink.append(record)
            raise
        record = {
            "phase": phase, "ts": t0,
            "elapsed_s": round(time.time() - t0, 3),
            "timeout": timeout, "max_retries": max_retries,
            "prompt": prompt,
            "response_text": getattr(resp, "text", None),
            "success": getattr(resp, "success", None),
            "error": getattr(resp, "error", None),
            "latency_ms": getattr(resp, "latency_ms", None),
            "model": getattr(resp, "model", None),
        }
        usage = getattr(resp, "token_usage", None)
        if usage is not None:
            record["token_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        with self._sink_lock:
            self._sink.append(record)
        # A phase result may only be interpreted after every required request
        # succeeds. Returning a failed response lets extract_json() turn it into
        # None and silently omit an entire batch.
        if not resp.success:
            raise RuntimeError(f"LLM request failed in {phase}: {resp.error}")
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class MentionType(Enum):
    """How the component's name is present in a sentence.

    Three values, one per way a name can be present. The five-value form this
    replaces also graded the *case* of the match (proper-case standalone versus a
    lowercase mention) and carried a value for "neither the name nor an alias
    matched"; over six runs of s_linker38 those three values were approved at 96.9%,
    100.0% and 100.0%, so the grading changed no verdict, while the two values kept
    here separate 82.8% from 57.4%.
    """
    NAME = "the name itself"
    ALIAS = "a name the document introduces for it"
    QUALIFIED_ONLY = "only inside a qualified identifier"


@dataclass
class EvidenceBundle:
    """What a judge is told about a candidate beyond the candidate itself.

    The matched span and the preceding sentence also appear in the case header
    the judge reads (``Case n: "span" -> Component``, then the sentence with its
    ``[prev: ...]`` prefix), so the bundle repeats both. The repetition is
    deliberate and was verified the hard way: dropping either is neutral on the
    judging stage in isolation (span TP +0.8, F2 +0.3, all p >= 0.44; preceding
    sentence TP -0.4, F2 -0.2, p >= 0.30) and costs precision once composed --
    three five-project runs without them hold recall (TP 182.0) and lose it on
    false positives (8.3 against the 4-6 of the six-run reference band), F1 95.2
    against 96.42 +/- 0.42. Repeating the evidence next to the rubric is not
    redundant for the model.
    """

    source: str
    matched_span: str
    mention_type: str          # MentionType.value (str for prompt embedding)
    preceding_text: str
    anchor_sentences: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Main linker
# ─────────────────────────────────────────────────────────────────────────────

class SLinker42:
    """Three linkers, fixed name-evidence order, no controller. Standalone."""

    _VARIANT_NAME = "s_linker42"

    #: Execution order. Full name first (it needs the least), partial name
    #: second, coreference last. The partial-name linker is the only one that
    #: subtracts already-linked pairs, so it must not run first.
    LINKERS = ("full_name", "partial_name", "coreference")

    # ── Resource bounds ──────────────────────────────────────────────────────
    # These cap prompt size and call count. No decision rule reads them:
    # changing one changes how much text a judge sees, never what counts as a
    # link. Named here so they are auditable in one place.
    #
    # Every evidence window is the same width and every anchor list the same
    # length, on purpose — the earlier per-step values (2, 3, 4, 5) implied a
    # calibration that was never measured. One width was verified not to
    # weaken the target-blind denotation step: that step's blindness comes from
    # withholding the target label from the case, not from hiding sentences
    # that name components, and a naming sentence is already visible in the
    # shared batch table for the large majority of candidates at any width.
    CONTEXT_SENTENCES = 5          # sentences either side shown to any judge
    ANCHOR_LIMIT = 5               # naming sentences offered as evidence
    EXTRACTION_BATCH = 50          # sentences per full-name extraction call
    JUDGE_BATCH = 25               # candidates per judging call (all judges)
    COREFERENCE_BATCH = 10         # sentences per coreference-resolution call
    ASK_ATTEMPTS = 2               # initial call + one retry on an empty parse

    def __init__(
        self,
        backend: LLMBackend | None = None,
        model: str | None = None,
        checkpoint_fallback: LLMBackend | str | None = None,
        checkpoint_fallback_model: str | None = None,
        no_knowledge: bool = False,
    ):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")
        real_llm = LLMClient(
            backend=backend or LLMBackend.CLAUDE,
            model=model,
            checkpoint_fallback=checkpoint_fallback,
            checkpoint_fallback_model=checkpoint_fallback_model,
        )
        self._llm_calls: list[dict] = []
        self.llm = _TracingLLMClient(real_llm, self._llm_calls)
        self.no_knowledge = no_knowledge
        self.doc_knowledge: DocumentKnowledge | None = None
        self._phase_log: list[dict] = []
        self._phase_metrics: dict[str, dict] = {}
        self.workflow: list[dict] = []
        print("SLinker42 (one full-name judging call, both criteria)")
        print(f"  Backend: {self.llm.describe_backend()}")

    # ── Main entry ───────────────────────────────────────────────────────────

    def link(self, text_path, model_path, **_kwargs):
        self._phase_log = []
        self._llm_calls.clear()
        self._phase_metrics = {}
        started = time.time()

        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        name_to_id = {component.name: component.id for component in components}
        sent_map = build_sent_map(sentences)
        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        print("\n[Knowledge] Document aliases")
        self.doc_knowledge = (
            DocumentKnowledge() if self.no_knowledge
            else self._learn_document_knowledge(sentences, components)
        )
        self._save_phase(text_path, "knowledge",
                         {"doc_knowledge": self.doc_knowledge})

        current: list[SadSamLink] = []
        history: list[dict] = []
        for linker in self.LINKERS:
            print(f"\n[Linker] {linker}")
            produced, feedback = self._run_linker(
                linker, sentences, components, name_to_id, current, sent_map
            )
            current = self._union(current, produced)
            history.append({
                "linker": linker,
                "feedback": self._linker_feedback(feedback),
            })
            self._save_phase(text_path, f"linker_{linker}", {
                "links": produced, "feedback": feedback, "workflow": history,
            })

        self.workflow = history
        self._phase_metrics = self._compute_phase_metrics()
        self._log(
            "s25_summary",
            {"components": len(components), "sentences": len(sentences)},
            {
                "workflow": history,
                "final": len(current),
                "elapsed_s": round(time.time() - started, 2),
                "llm_calls": len(self._llm_calls),
                "phase_metrics": self._phase_metrics,
            },
            current,
        )
        self._save_phase(text_path, "final", {
            "final": current,
            "workflow": history,
            "elapsed_s": round(time.time() - started, 2),
        })
        self._save_log(text_path)
        print(f"\nFinal: {len(current)} links "
              f"({time.time() - started:.1f}s, {len(self._llm_calls)} LLM calls)")
        return current

    def _run_linker(self, linker, sentences, components, name_to_id, linked, sent_map):
        """Dispatch, with the linked set passed to every linker without exception.

        This is what makes "each linker sees only what the earlier ones left
        unlinked" a property of the pipeline rather than of one linker. The
        subtraction itself is `_unlinked`, called once inside each linker at its
        candidate boundary.
        """
        if linker == "full_name":
            return self._run_full_name_linker(
                sentences, components, name_to_id, linked, sent_map)
        if linker == "partial_name":
            return self._run_partial_name_linker(
                sentences, components, linked, sent_map)
        if linker == "coreference":
            return self._run_coreference_linker(
                sentences, components, name_to_id, linked, sent_map)
        raise RuntimeError(f"unknown linker: {linker!r}")

    @staticmethod
    def _unlinked(candidates, linked):
        """Drop every proposal for a pair an earlier linker already produced.

        Removing them cannot change the final link set -- the union already
        holds each one -- but it does keep them out of the judging batches they
        would otherwise share with the pairs still in question. Measured over
        five runs on all five projects, it removes 57% of the coreference
        judge's cases and, with them, 6.8 false positives (p=0.01) at +0.8 true
        positives (p=0.05).
        """
        return [c for c in candidates
                if (c.sentence_number, c.component_id) not in linked]

    # ── Concurrency and small helpers ────────────────────────────────────────

    @staticmethod
    def _run_parallel(tasks):
        if len(tasks) == 1:
            name, fn = next(iter(tasks.items()))
            return {name: fn()}
        results = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            try:
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()
            except Exception:
                for other in futures:
                    other.cancel()
                raise
        return results

    @staticmethod
    def _iter_batches(items, n):
        """Yield (batch_num, batch_slice) — batch_num is 1-indexed."""
        for i, start in enumerate(range(0, len(items), n), start=1):
            yield i, items[start:start + n]

    @staticmethod
    def _prev_prefix(snum, sent_map) -> str:
        prev = sent_map.get(snum - 1)
        return f"[prev: {prev.text}] " if prev else ""

    @staticmethod
    def _find_exact_form(text, expression):
        """Return the word-boundary match of expression in text, or ""."""
        match = re.search(
            rf"(?<!\w){re.escape(expression)}(?!\w)", text, re.IGNORECASE
        )
        return match.group(0) if match else ""

    def _names_by_component(self):
        """Discovered aliases grouped by component. The model name is added by
        callers; together they are the component's set of names N(c)."""
        aliases = getattr(getattr(self, "doc_knowledge", None), "aliases", {})
        names = {}
        for term, component in aliases.items():
            names.setdefault(component, []).append(term)
        return names

    @staticmethod
    def _union(existing, additions):
        """Merge by (sentence, component). Earlier linkers win ties."""
        result = list(existing)
        keys = {(link.sentence_number, link.component_id) for link in existing}
        for link in additions:
            key = (link.sentence_number, link.component_id)
            if key not in keys:
                result.append(link)
                keys.add(key)
        return result

    @staticmethod
    def _link_view(links, sent_map):
        return [
            {
                "sentence": link.sentence_number,
                "text": sent_map[link.sentence_number].text,
                "component": link.component_name,
                "source": link.source,
            }
            for link in links
            if link.sentence_number in sent_map
        ]

    @staticmethod
    def _decision_view(decisions):
        return [
            {"sentence": sentence, "component_id": component, **decision}
            for (sentence, component), decision in decisions.items()
        ]

    @staticmethod
    def _linker_feedback(feedback):
        """Reduce detailed linker evidence to accepted/rejected references."""
        proposed = feedback.get("candidates", feedback.get("proposed", []))
        accepted = feedback.get("accepted", [])
        accepted_keys = {(i["sentence"], i["component"]) for i in accepted}

        def reference(item):
            return {"sentence": item["sentence"], "component": item["component"]}

        return {
            "accepted": [reference(i) for i in accepted],
            "rejected": [
                reference(i) for i in proposed
                if (i["sentence"], i["component"]) not in accepted_keys
            ],
        }

    # ── Prompt builders ──────────────────────────────────────────────────────

    @staticmethod
    def _prompt_doc_knowledge_extract(comp_names, doc_lines) -> str:
        return f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{DOC_KNOWLEDGE_EXTRACTION_RULES}

{ALIAS_EXCLUSION_RULES}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent"}}]
}}
JSON only:"""

    @staticmethod
    def _prompt_doc_knowledge_judge(comp_names, mapping_list) -> str:
        return f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}



{DOC_KNOWLEDGE_JUDGE_RULES}

Return JSON:
{{"approved": ["term1", "term2"]}}
JSON only:"""

    @staticmethod
    def _prompt_extraction(comp_names, mappings, batch) -> str:
        return f"""Extract ALL references to components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}

{ENTITY_EXTRACTION_RULES}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence"}}]}}
JSON only:"""

    @staticmethod
    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
        """Build a judging prompt. ``strict`` selects the coreference rubric.

        The rubric is asymmetric by design: the full-name gate is lenient (a
        stated name is a link unless a reject signal fires), the coreference
        gate is strict (the name is absent, so demand a referring expression
        and an architectural claim). The caller states which it wants — the
        inherited version inferred it from ``focus.startswith(...)``, so
        rewording the focus text silently swapped the rubric.
        """
        rules = LAYERED_COREF_RULES if strict else LAYERED_ENTITY_RULES
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{rules}

For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:"""

    @staticmethod
    def _prompt_coref(comp_names, cases) -> str:
        prompt = f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

For each TARGET sentence below, identify any pronoun or noun phrase that
refers back to a component listed above. If a target sentence has no such
reference to a listed component, return no resolution for it. Be conservative — only include resolutions you are CERTAIN about.

"""
        for i, case in enumerate(cases):
            prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
            prompt += "CONTEXT:\n" + "\n".join(case["context"]) + "\n"
            prompt += f"TARGET: S{case['sent'].number} (marked with >>>)\n\n"

        prompt += f"""{COREF_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

JSON only:"""
        return prompt

    # ── LLM call helper ──────────────────────────────────────────────────────

    def _ask(
        self,
        prompt: str,
        *,
        timeout: int = 120,
        label: str = "LLM call",
        phase: str | None = None,
        require: str | None = None,
        require_present: str | None = None,
    ) -> dict:
        """Query the LLM, parse JSON, retry once on empty/incomplete response.

        Success rule, in priority order:
          - require_present=KEY  → KEY must appear in the parsed dict (empty OK)
          - require=KEY          → data[KEY] must be truthy
          - neither              → any non-empty parsed dict succeeds
        """
        if phase is not None:
            self.llm.set_phase(phase)

        def _ok(d: dict | None) -> bool:
            if not d:
                return False
            if require_present is not None:
                return require_present in d
            if require is not None:
                return bool(d.get(require))
            return True

        data: dict = {}
        for attempt in range(self.ASK_ATTEMPTS):
            parsed = self.llm.extract_json(self.llm.query(prompt, timeout=timeout))
            # Each attempt replaces the last. Keeping a previous attempt's dict
            # when a later one fails to parse would return a payload this method
            # already rejected, and callers read it as if it had passed.
            data = parsed if parsed is not None else {}
            if _ok(data):
                return data
            if attempt < self.ASK_ATTEMPTS - 1:
                print(f"    {label}: empty response, retrying...")
        return data

    # ── Knowledge module ─────────────────────────────────────────────────────

    def _learn_document_knowledge(self, sentences, components):
        self.llm.set_phase("phase_25_doc_extract")
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        data1 = self._ask(
            self._prompt_doc_knowledge_extract(comp_names, doc_lines),
            timeout=300, label="Doc knowledge",
        )

        all_mappings: dict[str, str] = {}
        if data1:
            abbr_recs = data1.get("abbreviations", [])
            syn_recs = data1.get("synonyms", [])
            if isinstance(abbr_recs, dict):
                abbr_recs = [{"term": k, "component": v}
                             for k, v in abbr_recs.items()]
            if isinstance(syn_recs, dict):
                syn_recs = [{"term": k, "component": v}
                            for k, v in syn_recs.items()]
            for rec in abbr_recs + syn_recs:
                if not isinstance(rec, dict):
                    continue
                term = rec.get("term")
                full = rec.get("component")
                if term and full in comp_names:
                    all_mappings[term] = full

        if all_mappings:
            mapping_list = [f"'{k}' -> {v}" for k, v in all_mappings.items()]
            data2 = self._ask(
                self._prompt_doc_knowledge_judge(comp_names, mapping_list),
                timeout=120, label="Doc knowledge judge",
                phase="phase_25_doc_judge", require="approved",
            )
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings)
        else:
            approved = set()

        knowledge = DocumentKnowledge()
        for term, comp in all_mappings.items():
            if term in approved:
                knowledge.aliases[term] = comp
                print(f"    Alias: {term} -> {comp}")
        return knowledge

    # ═════════════════════════════════════════════════════════════════════════
    # Linker 1 — FULL NAME: the sentence states a name of the component.
    # ═════════════════════════════════════════════════════════════════════════

    def _run_full_name_linker(self, sentences, components, name_to_id, linked,
                              sent_map):
        candidates_by_key = self._extract_named_mentions(
            sentences, components, name_to_id, sent_map
        )
        candidates = self._keep_stated_names(list(candidates_by_key.values()))
        candidates = self._add_spelling_variants(candidates, sentences, components)
        candidates = self._unlinked(candidates, linked)
        bundles = {
            (c.sentence_number, c.component_id): self._build_evidence_bundle(c, sent_map)
            for c in candidates
        }
        approved, decisions = self._validate_with_evidence(
            candidates, bundles, components, sent_map,
            p1_tag="phase_25_full_name_p1",
            p2_tag="phase_25_full_name_p2",
            stage_label="full_name",
        )
        links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name,
                       source=self._full_name_source(c))
            for c in approved
        ]
        return links, {
            "candidates": self._link_view(
                [SadSamLink(c.sentence_number, c.component_id, c.component_name,
                            source="full_name_candidate") for c in candidates],
                sent_map,
            ),
            "accepted": self._link_view(links, sent_map),
            "judge_decisions": self._decision_view(decisions),
        }

    def _extract_named_mentions(self, sentences, components, name_to_id, sent_map) -> dict:
        """One extraction pass over the document, batched.

        An earlier revision sent this prompt twice and unioned the two samples
        as a self-consistency guard. Measured over five runs on all five
        projects, the second sample moved neither score beyond noise (TP -1.2,
        p=0.30; FP -1.2, p=0.42), so the pipeline states one sample and pays for
        one.
        """
        comp_names = get_comp_names(components)
        mappings = (
            [f"{term}={component}"
             for term, component in self.doc_knowledge.aliases.items()]
            if self.doc_knowledge else []
        )
        candidates = self._run_extraction_pass(
            sentences, comp_names, mappings, name_to_id, sent_map,
            phase_tag="phase_25_full_name_extract")
        print(f"    Extracted: {len(candidates)}")
        return candidates

    def _run_extraction_pass(self, sentences, comp_names, mappings,
                             name_to_id, sent_map, phase_tag=None):
        if phase_tag:
            self.llm.set_phase(phase_tag)
        batch_size = self.EXTRACTION_BATCH
        candidates: dict = {}
        for batch_num, batch in self._iter_batches(sentences, batch_size):
            if len(sentences) > batch_size:
                print(f"    batch {batch_num}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")
            data = self._ask(
                self._prompt_extraction(comp_names, mappings, batch),
                timeout=240, label="batch", require="references",
            )
            if not data:
                continue
            for ref in data.get("references", []):
                cname = ref.get("component")
                snum = parse_snum(ref.get("sentence"))
                if snum is None or not cname or cname not in name_to_id:
                    continue
                sent = sent_map.get(snum)
                if not sent:
                    continue
                matched = ref.get("matched_text", "")
                if matched and matched.lower() not in sent.text.lower():
                    continue
                key = (snum, name_to_id[cname])
                if key not in candidates:
                    candidates[key] = CandidateLink(
                        snum, sent.text, cname, name_to_id[cname],
                        matched, source="full_name",
                    )
        return candidates

    def _keep_stated_names(self, candidates):
        """Keep only candidates whose sentence states a name of the component.

        This is what makes the linker a *full-name* linker: the extractor is
        looser than the contract, so the contract is enforced here.

        It was removed once and put back. Judged on its own stage the filter is
        F2-negative -- handing the decision to the judge gave TP +2.8 (p=0.01),
        F1 -0.0, F2 +0.9 (p=0.01) -- but composed into the pipeline it is
        strongly positive: three five-project runs without it give FP 17.3
        against 4.3 with it, at the same recall (TP 182.0 vs 182.3), and macro F2
        94.9 against 95.9. The stage measurement was not predictive because this
        stage's output feeds ``_unlinked``: an admitted false positive is locked
        into the union and also removes the pair from the two later linkers,
        which would have judged it under stricter rubrics.

        Deliberately NOT symmetric with the other two candidate generators: they
        skip spans inside a dotted path, this does not. Measured (pilot
        ``fix4_qualified_path``, N=3 x 5 projects): adding the path filter here
        removes 18 links per run, of which 15 are gold. The judge's reject-rule
        (1) already discriminates among path-only mentions far better than the
        boundary test can, so the lenient filter plus a strict judge beats a
        strict filter. Do not "fix" this asymmetry.
        """
        names_by_component = self._names_by_component()
        return [
            c for c in candidates
            if any(
                self._find_exact_form(c.sentence_text, name)
                for name in (c.component_name,
                             *names_by_component.get(c.component_name, []))
            )
        ]

    def _add_spelling_variants(self, candidates, sentences, components):
        """Add exact, catalog-unique orthographic variants the extractor missed."""
        merged = {(c.sentence_number, c.component_id): c for c in candidates}
        for candidate in self._spelling_variant_candidates(sentences, components):
            merged.setdefault(
                (candidate.sentence_number, candidate.component_id), candidate
            )
        return list(merged.values())

    @staticmethod
    def _full_name_source(candidate):
        if candidate.source == "full_name_variant":
            return "full_name_variant"
        return "full_name"

    @staticmethod
    def _name_signature(expression):
        """Normalize an expression to its sequence of words, splitting CamelCase.

        A spaced form, a hyphenated form, and a run-together form of the same
        words share a signature ("X Y", "x-y", and "XY" all give ("x", "y")),
        which is what makes a spelling variant recognizable.
        """
        normalized = unicodedata.normalize("NFKC", expression)
        normalized = normalized.replace("-", " ").replace("_", " ")
        return tuple(
            token.casefold()
            for token in re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+", normalized
            )
        )

    @staticmethod
    def _in_dotted_path(text, start, end) -> bool:
        """True when text[start:end] is glued to a dot on either side, as in x.y.

        The single definition of "inside a qualified name". Two tests used to
        carry their own copy of it and the copies disagreed -- one asked whether
        the character after the dot ``isalnum()``, the other ``isalpha()``, and
        one required an alphanumeric before the dot while the other did not.
        Neither divergence ever changed a result (0 differences over 3697
        (name, sentence) pairs and 5388 word spans on all five projects), so the
        stricter reading is the one kept.
        """
        before = (start > 1 and text[start - 1] == "."
                  and text[start - 2].isalnum())
        after = (end + 1 < len(text) and text[end] == "."
                 and text[end + 1].isalnum())
        return before or after

    @classmethod
    def _inside_qualified_identifier(cls, text, start, end):
        """True when the span sits inside a dotted path or a larger word."""
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        joined = (before in "-_" or (before and before.isalnum())
                  or after in "-_" or (after and after.isalnum()))
        return cls._in_dotted_path(text, start, end) or joined

    @classmethod
    def _spelling_variant_candidates(cls, sentences, components):
        """Spans whose word signature equals exactly one component's name."""
        word_pattern = re.compile(r"[A-Za-z0-9]+")
        owners = {}
        for component in components:
            signature = cls._name_signature(component.name)
            if signature:
                owners.setdefault(signature, []).append(component)
        max_words = max((len(item) for item in owners), default=0)
        candidates = {}
        for sentence in sentences:
            words = list(word_pattern.finditer(sentence.text))
            for start_index, first in enumerate(words):
                for end_index in range(start_index,
                                       min(len(words), start_index + max_words)):
                    last = words[end_index]
                    if end_index > start_index:
                        separator = sentence.text[
                            words[end_index - 1].end():last.start()
                        ]
                        if not re.fullmatch(r"[\s_-]+", separator):
                            break
                    start, end = first.start(), last.end()
                    if cls._inside_qualified_identifier(sentence.text, start, end):
                        continue
                    surface = sentence.text[start:end]
                    targets = owners.get(cls._name_signature(surface), ())
                    if len(targets) != 1:
                        continue
                    component = targets[0]
                    if surface.casefold() == component.name.casefold():
                        continue  # already the plain name; not a variant
                    candidates[(sentence.number, component.id)] = CandidateLink(
                        sentence.number, sentence.text, component.name,
                        component.id, surface, source="full_name_variant",
                    )
        return list(candidates.values())

    # ── Evidence bundles and the two-pass judge ──────────────────────────────

    def _classify_mention_typed(self, comp_name: str,
                                text: str) -> MentionType | None:
        """Which of the three ways the name is present here, or None for none of them.

        One matching test decides all three, and nothing here reads the case of
        the match: the five-value form graded proper case against lowercase, and
        over six runs of s_linker38 the judge approved 96.9% of the first and
        100.0% of the second, so the grade carried no information the judge used.
        ``None`` replaces that form's INDIRECT value (1.7 cases per run, all
        approved): if neither the name nor a name the document introduced is in the
        sentence, there is nothing to report about how the name is present, and the
        bundle omits the field rather than guessing at it.
        """
        if self._find_exact_form(text, comp_name):
            if self._all_occurrences_in_qualified_path(comp_name.lower(), text):
                return MentionType.QUALIFIED_ONLY
            return MentionType.NAME
        for alias in self._names_by_component().get(comp_name, ()):
            if self._find_exact_form(text, alias):
                return MentionType.ALIAS
        return None

    @classmethod
    def _all_occurrences_in_qualified_path(cls, comp_lower: str, text: str) -> bool:
        any_match = False
        for m in re.finditer(rf'\b{re.escape(comp_lower)}\b', text):
            any_match = True
            if not cls._in_dotted_path(text, m.start(), m.end()):
                return False
        return any_match

    def _build_evidence_bundle(self, candidate, sent_map):
        comp_name = candidate.component_name
        snum = candidate.sentence_number
        mention = self._classify_mention_typed(comp_name, candidate.sentence_text)
        mention_type = mention.value if mention else ""
        prev_sent = sent_map.get(snum - 1)
        anchors = []
        for s in sorted(sent_map.values(), key=lambda x: x.number):
            if s.number == snum:
                continue
            if self._find_exact_form(s.text, comp_name):
                anchors.append(f"S{s.number}: {s.text}")
                if len(anchors) >= self.ANCHOR_LIMIT:
                    break
        return EvidenceBundle(
            source=candidate.source,
            matched_span=candidate.matched_text or comp_name,
            mention_type=mention_type,
            preceding_text=prev_sent.text if prev_sent else "",
            anchor_sentences=anchors,
        )

    def _format_evidence(self, bundle: EvidenceBundle) -> str:
        head = (f"  Evidence: source={bundle.source}, "
                f"span=\"{bundle.matched_span}\"")
        # The mention field says how the name is present. When neither the name nor
        # a discovered alias is in the sentence there is no such fact to state, so
        # the field is left out instead of carrying a value meaning "none of them".
        lines = [f"{head}, mention={bundle.mention_type}" if bundle.mention_type
                 else head]
        if bundle.preceding_text:
            lines.append(f"  [prev: \"{bundle.preceding_text}\"]")
        if bundle.anchor_sentences:
            lines.append("  Anchors (confirmed refs):")
            for a in bundle.anchor_sentences:
                lines.append(f"    {a}")
        return "\n".join(lines)

    def _validate_with_evidence(self, candidates, bundles, components, sent_map,
                                p1_tag, p2_tag, stage_label):
        """One judging call carrying both criteria; a link needs both verdicts.

        s_linker25 sends the same cases twice, once per criterion. The criteria are
        load-bearing -- dropping the uniqueness one costs 10 false positives -- but
        two *calls* are a different claim from two *criteria*. This asks for both
        verdicts in one response, halving the judging calls of the largest stage in
        the workflow (M calls instead of 2M, where M is the number of judging
        batches). The independence that matters in this workflow is between a
        proposal and its judgment, not between two criteria applied to the same
        candidate by the same judge.
        """
        if not candidates:
            return [], {}
        comp_names = get_comp_names(components)
        decisions: dict = {}
        approved = []
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            cases = []
            for i, c in enumerate(batch):
                p = self._prev_prefix(c.sentence_number, sent_map)
                bundle = bundles.get((c.sentence_number, c.component_id))
                evidence_block = self._format_evidence(bundle) if bundle else ""
                cases.append((
                    f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n'
                    f'  {p}"{c.sentence_text}"\n'
                    f'{evidence_block}',
                    c,
                ))
            case_strings = [ct for ct, _ in cases]
            r1, r2 = self._run_two_criteria_pass(comp_names, case_strings, p1_tag)
            for i, (_case_text, c) in enumerate(cases):
                p1 = r1.get(i, False)
                p2 = r2.get(i, False)
                ok = p1 and p2
                decisions[(c.sentence_number, c.component_id)] = {
                    "approved": ok, "p1": p1, "p2": p2,
                    "path": f"{stage_label}_twopass" if ok
                            else f"{stage_label}_twopass_reject",
                    "stage": f"{stage_label}_twopass",
                }
                if ok:
                    approved.append(c)
        return approved, decisions

    def _run_two_criteria_pass(self, comp_names, cases, phase_tag=None):
        """Both full-name criteria in one call: relevance and uniqueness."""
        if phase_tag:
            self.llm.set_phase(phase_tag)
        data = self._ask(
            self._prompt_two_criteria(comp_names, cases),
            timeout=180, label="Validation (both criteria)",
            require="validations",
        )
        first: dict[int, bool] = {}
        second: dict[int, bool] = {}
        for item in data.get("validations", []) or []:
            index = item.get("case", 0) - 1
            if not 0 <= index < len(cases):
                continue
            for key, sink in (("relevant", first), ("unique", second)):
                value = item.get(key, False)
                sink[index] = (value is True
                               or (isinstance(value, str)
                                   and value.lower() == "true"))
        return first, second

    @staticmethod
    def _prompt_two_criteria(comp_names, cases) -> str:
        """The two focuses of s_linker25's passes, asked once.

        Both rubric texts are the ones the separate passes carry, verbatim, so the
        only difference is that one response answers both.
        """
        return f"""Validate components in a document. Answer two questions about each case.

COMPONENTS: {', '.join(comp_names)}

{LAYERED_ENTITY_RULES}

QUESTION 1 (relevant). {P1_FOCUS}

QUESTION 2 (unique). {P2_FOCUS}

For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then answer both questions.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "relevant": true, "unique": true}}]}}
JSON only:"""

    def _run_validation_pass(self, comp_names, cases, focus, phase_tag=None,
                             strict=False):
        if phase_tag:
            self.llm.set_phase(phase_tag)
        data = self._ask(
            self._prompt_validation(comp_names, cases, focus, strict=strict),
            timeout=120, label="Validation pass", require="validations",
        )
        results: dict[int, bool] = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    val = v.get("approve", False)
                    results[idx] = (
                        val is True
                        or (isinstance(val, str) and val.lower() == "true")
                    )
        return results

    # ═════════════════════════════════════════════════════════════════════════
    # Linker 2 — PARTIAL NAME: the sentence carries one word of a name.
    # ═════════════════════════════════════════════════════════════════════════

    def _run_partial_name_linker(self, sentences, components, linked, sent_map):
        candidates = self._unlinked(
            self._name_word_candidates(sentences, components), linked)
        approved, decisions = self._judge_partial_names(candidates, sentences)
        links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name,
                       source="partial_name")
            for c in approved
        ]
        return links, {
            "proposed": self._link_view(
                [SadSamLink(c.sentence_number, c.component_id, c.component_name,
                            source="partial_name_candidate") for c in candidates],
                sent_map,
            ),
            "accepted": self._link_view(links, sent_map),
            "judge_decisions": self._decision_view(decisions),
        }

    def _name_word_candidates(self, sentences, components):
        """Propose a sentence word that matches or extends a word of exactly one
        component's name, in a sentence that states no whole name.

        The prefix test is a general morphology approximation: a sentence word
        that begins with a name word is accepted, so inflected forms pass
        without a suffix list or a pluralizer. Component names are split on
        word boundaries only, not
        CamelCase — splitting compounds was measured to triple the candidate
        set while reaching no additional gold link.
        """
        words_by_component = {
            component.id: [
                word.casefold()
                for word in re.findall(r"[A-Za-z]+[A-Za-z0-9]*|\d+", component.name)
            ]
            for component in components
        }
        names_by_component = self._names_by_component()
        candidates = {}
        for sentence in sentences:
            for match in re.finditer(r"[A-Za-z]+[A-Za-z0-9]*|\d+", sentence.text):
                if self._inside_qualified_identifier(
                    sentence.text, match.start(), match.end()
                ):
                    continue
                surface = match.group(0).casefold()
                owners = [
                    component for component in components
                    if any(surface.startswith(word)
                           for word in words_by_component[component.id])
                ]
                if len(owners) != 1:
                    continue
                component = owners[0]
                key = (sentence.number, component.id)
                names = [component.name,
                         *names_by_component.get(component.name, [])]
                if any(self._find_exact_form(sentence.text, name)
                       for name in names):
                    continue  # the whole name is stated here: not a partial name
                candidates[key] = CandidateLink(
                    sentence.number, sentence.text, component.name,
                    component.id, match.group(0), source="partial_name_candidate",
                )
        return list(candidates.values())

    def _judge_partial_names(self, candidates, sentences):
        """Two steps: denotation without the target, then grounded identity."""
        participants, decisions = self._classify_denotations(candidates, sentences)
        approved, reviewed = self._review_identity(participants, sentences)
        for key, decision in reviewed.items():
            decisions[key] = {**decisions.get(key, {}), **decision}
        return approved, decisions

    def _classify_denotations(self, candidates, sentences):
        """Step 1: does the expression itself denote a software participant?

        The target component is deliberately withheld. Shown the target, the
        model confirms identity rather than testing it.
        """
        sent_map = {s.number: s for s in sentences}
        decisions = {}
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            evidence_ids = {
                sentence.number
                for candidate in batch
                for sentence in sentences
                if abs(sentence.number - candidate.sentence_number)
                <= self.CONTEXT_SENTENCES
            }
            sentence_table = [
                {"sentence": n, "text": sent_map[n].text}
                for n in sorted(evidence_ids)
            ]
            cases = [
                {"case": n, "source": c.sentence_number, "expression": c.matched_text}
                for n, c in enumerate(batch, 1)
            ]
            prompt = f"""Classify what each expression itself denotes in its
local context: participant for a software participant, or associated for
something merely associated with software.

SENTENCES
{json.dumps(sentence_table)}

CASES
{json.dumps(cases)}

Claim must be a contiguous exact substring of the source sentence.

JSON only:
{{"judgments":[{{"case":1,"denotation":"participant",
"claim":"exact source quote"}}]}}
"""
            data = self._ask(
                prompt, phase="phase_25_partial_denotation",
                require_present="judgments", label="Denotation", timeout=240,
            )
            for item in data.get("judgments", []):
                case_value = str(item.get("case", ""))
                if not case_value.isdigit():
                    continue
                number = int(case_value)
                if not 1 <= number <= len(batch):
                    continue
                candidate = batch[number - 1]
                claim = str(item.get("claim", "")).strip().strip("\"'“”‘’")
                denotation = str(item.get("denotation", "")).strip()
                valid = (
                    denotation in {"participant", "associated"}
                    and bool(claim)
                    and claim.casefold() in candidate.sentence_text.casefold()
                )
                decisions[(candidate.sentence_number, candidate.component_id)] = {
                    "approved": False,
                    "requested_keep": False,
                    "evidence_valid": valid,
                    "claim": claim,
                    "denotation": denotation,
                    "alternative": "not reviewed",
                    "path": "denotation",
                    "stage": "partial_name",
                }
        participants = [
            c for c in candidates
            if decisions.get((c.sentence_number, c.component_id), {}).get(
                "denotation") == "participant"
            and decisions[(c.sentence_number, c.component_id)]["evidence_valid"]
        ]
        return participants, decisions

    def _review_identity(self, candidates, sentences):
        """Batch step 2 the way step 1 is batched.

        Below JUDGE_BATCH candidates this is exactly one call, so it is a no-op
        on every current benchmark (the largest project yields 19 participants).
        It bounds the blast radius on a longer document, where the unbatched
        form put every partial-name link of a project behind one parse.
        """
        approved, decisions = [], {}
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            got, made = self._review_identity_batch(batch, sentences)
            approved.extend(got)
            decisions.update(made)
        return approved, decisions

    def _review_identity_batch(self, candidates, sentences):
        """Step 2: do the expression and the target denote the same participant?

        Now the target is shown, together with the nearest sentences that state
        one of its names. Approval requires a listed anchor, an exact source
        quote, and a named strongest alternative.
        """
        if not candidates:
            return [], {}
        names_by_component = self._names_by_component()
        anchors_by_target = {}
        for target in {c.component_name for c in candidates}:
            names = [target, *names_by_component.get(target, [])]
            anchors_by_target[target] = [
                {"sentence": s.number, "text": s.text}
                for s in sentences
                if any(self._find_exact_form(s.text, name) for name in names)
            ]
        sent_map = {s.number: s for s in sentences}
        cases = []
        allowed_anchors = {}
        evidence_sentences = set()
        for number, candidate in enumerate(candidates, 1):
            anchors = sorted(
                anchors_by_target.get(candidate.component_name, []),
                key=lambda item: (abs(item["sentence"] - candidate.sentence_number),
                                  item["sentence"]),
            )[:self.ANCHOR_LIMIT]
            anchor_ids = [a["sentence"] for a in anchors]
            context = [
                s.number for s in sentences
                if abs(s.number - candidate.sentence_number) <= self.CONTEXT_SENTENCES
            ]
            allowed_anchors[number] = set(anchor_ids)
            evidence_sentences.update(context)
            evidence_sentences.update(anchor_ids)
            cases.append({
                "case": number,
                "source": candidate.sentence_number,
                "participant": candidate.matched_text,
                "target": candidate.component_name,
                "context": context,
                "anchors": anchor_ids,
            })
        sentence_table = [
            {"sentence": n, "text": sent_map[n].text}
            for n in sorted(evidence_sentences)
        ]
        prompt = f"""For each case, do the expression and target denote the
same participant? A longer or shorter label may denote the same participant.
Reject when a distinct referent is better supported. Keep only architectural
claims.

SENTENCES
{json.dumps(sentence_table)}

CASES
{json.dumps(cases)}

Use only a listed case anchor. Claim must be one contiguous exact substring of
the source sentence; do not abbreviate it or use ellipses.

JSON only:
{{"judgments":[{{"case":1,"keep":true,"anchor_sentence":1,
"claim":"exact source quote","alternative":"strongest alternative or none"}}]}}
"""
        data = self._ask(
            prompt, phase="phase_25_partial_identity",
            require_present="judgments", label="Identity", timeout=240,
        )
        by_case = {}
        for item in data.get("judgments", []):
            case_value = str(item.get("case", ""))
            anchor_value = str(item.get("anchor_sentence", ""))
            if not case_value.isdigit():
                continue
            number = int(case_value)
            if not 1 <= number <= len(candidates):
                continue
            candidate = candidates[number - 1]
            anchor = int(anchor_value) if anchor_value.isdigit() else 0
            claim = str(item.get("claim", "")).strip().strip("\"'“”‘’")
            alternative = str(item.get("alternative", "")).strip()
            evidence_valid = (
                anchor in allowed_anchors[number]
                and bool(claim)
                and claim.casefold() in candidate.sentence_text.casefold()
                and bool(alternative)
            )
            by_case[number] = {
                "approved": item.get("keep") is True and evidence_valid,
                "requested_keep": item.get("keep") is True,
                "evidence_valid": evidence_valid,
                "anchor_sentence": anchor,
                "claim": claim,
                "alternative": alternative,
            }
        approved = [
            c for number, c in enumerate(candidates, 1)
            if by_case.get(number, {}).get("approved") is True
        ]
        decisions = {
            (c.sentence_number, c.component_id): {
                **by_case.get(number, {
                    "approved": False, "requested_keep": False,
                    "evidence_valid": False, "alternative": "missing judgment",
                }),
                "path": "identity",
                "stage": "partial_name",
            }
            for number, c in enumerate(candidates, 1)
        }
        return approved, decisions

    # ═════════════════════════════════════════════════════════════════════════
    # Linker 3 — COREFERENCE: the sentence states no name and refers back.
    # ═════════════════════════════════════════════════════════════════════════

    def _run_coreference_linker(self, sentences, components, name_to_id, linked,
                                sent_map):
        resolved, metadata = self._resolve_references(
            sentences, components, name_to_id, sent_map
        )
        raw = self._unlinked(resolved, linked)
        approved, decisions = self._validate_coref_links(raw, sent_map, components)
        return approved, {
            "candidates": self._link_view(raw, sent_map),
            "accepted": self._link_view(approved, sent_map),
            "metadata": [
                {"sentence": sentence, "component_id": component, **value}
                for (sentence, component), value in metadata.items()
            ],
            "judge_decisions": self._decision_view(decisions),
        }

    def _resolve_references(self, sentences, components, name_to_id, sent_map):
        """Every sentence goes to the LLM in context; no pronoun regex.

        A resolution survives only when its antecedent sentence itself states a
        name of the component — the structural antecedent constraint.

        Both sentence numbers a resolution reports are checked against the
        document: the target sentence as well as the antecedent. A number the
        model invents cannot name a real sentence, so admitting one could only
        ever add a link the gold standard has no counterpart for.
        """
        comp_names = get_comp_names(components)
        all_coref = []
        coref_metadata: dict = {}
        self.llm.set_phase("phase_25_coreference")

        for batch_num, batch in self._iter_batches(sentences, self.COREFERENCE_BATCH):
            cases = []
            for sent in batch:
                context = []
                lo = max(1, sent.number - self.CONTEXT_SENTENCES)
                for i in range(lo, sent.number + self.CONTEXT_SENTENCES + 1):
                    s = sent_map.get(i)
                    if s:
                        marker = ">>>" if s.number == sent.number else "   "
                        context.append(f"{marker} S{s.number}: {s.text}")
                cases.append({"sent": sent, "context": context})

            data = self._ask(
                self._prompt_coref(comp_names, cases), timeout=600,
                label=f"Coref batch {batch_num}", require_present="resolutions",
            )
            if not data:
                continue

            for res in data.get("resolutions", []):
                comp = res.get("component")
                snum = parse_snum(res.get("sentence"))
                if snum is None or snum not in sent_map:
                    continue
                if not comp or comp not in name_to_id:
                    continue
                ant_snum = parse_snum(res.get("antecedent_sentence"))
                if ant_snum is None:
                    print(f"    Coref skip (no antecedent): S{snum} -> {comp}")
                    continue
                ant_sent = sent_map.get(ant_snum)
                if not ant_sent:
                    continue
                if not self._antecedent_states_name(comp, ant_sent.text):
                    continue
                cid = name_to_id[comp]
                all_coref.append(SadSamLink(snum, cid, comp, source="coreference"))
                coref_metadata[(snum, cid)] = {
                    "reference": res.get("reference", ""),
                    "antecedent_sentence": ant_snum,
                    "antecedent_text": res.get("antecedent_text", ""),
                    "raw_resolution": res,
                }
        return all_coref, coref_metadata

    def _antecedent_states_name(self, comp_name: str, ant_text: str) -> bool:
        """True iff the antecedent sentence states the component's name or an alias.

        One test, applied to the name and to each discovered alias. It used to
        apply a strict case-sensitive predicate to the name and a lenient one to
        the aliases; the two agree on every resolution of the promoted run
        (0 gate flips measured over all five projects), so the asymmetry is gone.
        """
        names = (comp_name, *self._names_by_component().get(comp_name, ()))
        return any(self._find_exact_form(ant_text, name) for name in names)

    def _validate_coref_links(self, coref_links, sent_map, components):
        """Single judging pass — asymmetric to the full-name linker's two passes,
        because resolution asks a narrower question."""
        if not coref_links:
            return [], {}
        comp_names = get_comp_names(components)
        validated = []
        decisions: dict = {}
        self.llm.set_phase("phase_25_coreference_judge")
        for _, batch in self._iter_batches(coref_links, self.JUDGE_BATCH):
            cases = []
            for i, lk in enumerate(batch):
                # _resolve_references admits a resolution only for a sentence
                # the document has, so every link reaching the judge has one.
                sent = sent_map[lk.sentence_number]
                p = self._prev_prefix(lk.sentence_number, sent_map)
                cases.append((
                    lk,
                    f'Case {i+1}: pronoun/role-ref -> {lk.component_name}\n'
                    f'  {p}"{sent.text}"',
                ))
            results = self._run_validation_pass(
                comp_names, [c for _, c in cases], COREF_VALIDATION_FOCUS,
                phase_tag="phase_25_coreference_judge", strict=True,
            )
            for idx, (lk, _case) in enumerate(cases):
                approved = bool(results.get(idx, False))
                decisions[(lk.sentence_number, lk.component_id)] = {
                    "approved": approved,
                    "path": "coref_validated" if approved else "coref_rejected",
                }
                if approved:
                    validated.append(lk)
                else:
                    print(f"    Coref reject: S{lk.sentence_number} -> {lk.component_name}")
        return validated, decisions

    # ── Logging and checkpointing ────────────────────────────────────────────

    def _backend_tag(self) -> str:
        inner = getattr(self.llm, "_inner", self.llm)
        backend = getattr(inner, "backend", None)
        if backend is None:
            return "unknown"
        return getattr(backend, "value", str(backend))

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, self._VARIANT_NAME, self._backend_tag(), ds)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_phase(self, text_path, phase_name, state):
        path = os.path.join(self._checkpoint_dir(text_path), f"{phase_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Checkpoint: {phase_name} saved")

    def _log(self, phase, input_summary, output_summary, links=None):
        entry = {"phase": phase, "ts": time.time(),
                 "in": input_summary, "out": output_summary}
        if links is not None:
            entry["links"] = [
                {"s": l.sentence_number, "c": l.component_name, "src": l.source}
                for l in links
            ]
        self._phase_log.append(entry)

    def _save_log(self, text_path):
        log_dir = os.environ.get("LLM_LOG_DIR", "./results/llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        ds = os.path.splitext(os.path.basename(text_path))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        backend = self._backend_tag()
        summary_path = os.path.join(
            log_dir, f"{self._VARIANT_NAME}_{backend}_{ds}_{ts}.json")
        with open(summary_path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {summary_path}")
        calls_path = os.path.join(
            log_dir, f"{self._VARIANT_NAME}_{backend}_{ds}_{ts}_calls.json")
        trunc_env = os.environ.get("CALLS_TRUNCATE_CHARS", "").strip()
        trunc = int(trunc_env) if trunc_env.isdigit() else 0
        if trunc > 0:
            calls = []
            for c in self._llm_calls:
                cc = dict(c)
                if cc.get("prompt") and len(cc["prompt"]) > trunc:
                    cc["prompt"] = cc["prompt"][:trunc] + "... [truncated]"
                if cc.get("response_text") and len(cc["response_text"]) > trunc:
                    cc["response_text"] = cc["response_text"][:trunc] + "... [truncated]"
                calls.append(cc)
        else:
            calls = self._llm_calls
        with open(calls_path, "w") as f:
            json.dump(calls, f, indent=2, default=str)
        print(f"  LLM call trace saved: {calls_path} ({len(self._llm_calls)} calls)")

    def _compute_phase_metrics(self) -> dict:
        metrics: dict[str, dict] = {}
        for call in self._llm_calls:
            ph = call.get("phase", "unknown")
            m = metrics.setdefault(
                ph, {"calls": 0, "elapsed_s": 0.0, "tokens": 0, "errors": 0})
            m["calls"] += 1
            m["elapsed_s"] = round(m["elapsed_s"] + call.get("elapsed_s", 0.0), 3)
            if call.get("success") is False:
                m["errors"] += 1
            usage = call.get("token_usage")
            if usage:
                m["tokens"] += usage.get("total_tokens", 0) or 0
        return metrics
