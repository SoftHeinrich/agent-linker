"""Spike 001: LLM-fully-driven trailing-word alias discovery.

Current (s_linker12c._enrich_trailing_words):
  Step A — STRUCTURAL GATE (code):
    * split component name (CamelCase/space/hyphen regex)
    * require len(parts) >= 2
    * require trailing word unique across components
    * require trailing word appears in sentence without full name
  Step B — LLM verify batch (approve/reject)

Proposed (LLM-only):
  Single LLM call:
    * input: components + full document
    * output: {"aliases": [{"alias": X, "component": Y, "evidence": "..."}]}
  LLM does discovery + verification in one pass. No regex, no structural filter.

The structural gate exists only to narrow the search space. For docs with N
components and S sentences the prompt size stays bounded (components list +
document). So gate is not load-bearing — it's a heuristic cache hint.

Risk: LLM discovery may hallucinate aliases (false positives) without the
structural uniqueness check. Mitigation: prompt forces cite-sentence evidence
+ an approve-only JSON schema; post-check that evidence sentence actually
contains the alias and NOT the full name (kept as guardrail, not a gate).

This file is a standalone demonstrator:
  * LLM_ONLY_PROMPT — the prompt design
  * fully_llm_driven() — drop-in replacement for _enrich_trailing_words
  * test: mock-LLM fixture covers happy path + hallucination rejection
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


# -------- prompt --------

LLM_ONLY_PROMPT = """ALIAS DISCOVERY: Find all single-word references in the document that are used
as standalone shorthand for a specific multi-word component.

Example (safe, abstract):
  Component "OrderProcessor". Document says "...the Processor validates each item...".
  -> alias "Processor" refers to OrderProcessor.

APPROVE only if ALL hold:
  1. The short word appears in a sentence WHERE the full component name does NOT.
  2. The short word refers to a specific named component (not a generic role).
  3. No other listed component ends with the same short word (no ambiguity).

Cite the sentence number as evidence for each alias.

COMPONENTS: {components}

DOCUMENT:
{document}

Return JSON:
{{"aliases": [{{"alias": "Word", "component": "FullComponent", "evidence_sentence": 42}}]}}
JSON only:"""


# -------- data shims (mirror real s_linker12c interfaces) --------

@dataclass
class _Sentence:
    number: int
    text: str


@dataclass
class _Component:
    name: str


@dataclass
class _Knowledge:
    aliases: dict = field(default_factory=dict)


# -------- current approach (copy of s_linker12c logic, for comparison) --------

def _split_component_name(name: str) -> list[str]:
    if ' ' in name or '-' in name:
        return re.split(r'[\s-]+', name)
    parts = re.findall(r'[A-Z][a-z]*|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)', name)
    return parts if parts else [name]


def structural_plus_llm_verify(knowledge, sentences, components, llm_verify):
    """Reproduces current s_linker12c._enrich_trailing_words logic."""
    existing_lower = {a.lower() for a in knowledge.aliases}
    candidates = []
    for comp in components:
        parts = _split_component_name(comp.name)
        if len(parts) < 2:
            continue
        last_word = parts[-1]
        last_lower = last_word.lower()
        if any(c.name != comp.name and c.name.lower().endswith(last_lower)
               for c in components):
            continue
        if last_lower in existing_lower:
            continue
        full_lower = comp.name.lower()
        if any(last_lower in s.text.lower() and full_lower not in s.text.lower()
               for s in sentences):
            candidates.append((last_word, comp.name))
    if not candidates:
        return knowledge
    approved = llm_verify(candidates, sentences)  # LLM step
    for word, comp_name in candidates:
        if word in approved:
            knowledge.aliases[word] = comp_name
    return knowledge


# -------- proposed approach (LLM-only) --------

def fully_llm_driven(knowledge, sentences, components, llm_call):
    """LLM-only replacement. No regex, no structural gate.

    llm_call(prompt) -> dict  (same contract as s_linker12c llm.extract_json).
    """
    prompt = LLM_ONLY_PROMPT.format(
        components=", ".join(c.name for c in components),
        document="\n".join(f"S{s.number}: {s.text}" for s in sentences),
    )
    data = llm_call(prompt) or {}
    comp_set = {c.name for c in components}
    sent_map = {s.number: s.text for s in sentences}
    existing_lower = {a.lower() for a in knowledge.aliases}
    for entry in data.get("aliases", []):
        alias = entry.get("alias", "").strip()
        comp = entry.get("component", "").strip()
        ev = entry.get("evidence_sentence")
        if not alias or comp not in comp_set:
            continue
        if alias.lower() in existing_lower:
            continue
        # Light guardrail (not a structural gate): evidence sentence must exist
        # and must contain alias but NOT full component name. This defends
        # against LLM hallucination without narrowing the search space.
        ev_text = sent_map.get(ev, "")
        if not ev_text:
            continue
        if alias.lower() not in ev_text.lower():
            continue
        if comp.lower() in ev_text.lower():
            continue
        knowledge.aliases[alias] = comp
    return knowledge


# -------- self-verifying test --------

def _test_happy_path():
    components = [_Component("TaskDispatcher"), _Component("AuthService"),
                  _Component("QueueBroker"), _Component("MediaPlayer")]
    sentences = [
        _Sentence(1, "The TaskDispatcher routes incoming jobs."),
        _Sentence(2, "Each request is queued and the Dispatcher hands it to a worker."),
        _Sentence(3, "User login is validated against the AuthService."),
        _Sentence(4, "The Broker persists queued messages to disk."),
        _Sentence(5, "Playback is driven by the MediaPlayer."),
    ]

    def fake_llm(_prompt):
        return {
            "aliases": [
                {"alias": "Dispatcher", "component": "TaskDispatcher",
                 "evidence_sentence": 2},
                {"alias": "Broker", "component": "QueueBroker",
                 "evidence_sentence": 4},
            ]
        }

    k = _Knowledge()
    fully_llm_driven(k, sentences, components, fake_llm)
    assert k.aliases == {"Dispatcher": "TaskDispatcher", "Broker": "QueueBroker"}, k.aliases
    print("  [pass] happy path: 2 aliases, no regex used")


def _test_hallucination_rejected():
    """LLM claims an alias whose evidence sentence does not support it."""
    components = [_Component("TaskDispatcher")]
    sentences = [_Sentence(1, "The TaskDispatcher routes jobs.")]

    def fake_llm(_prompt):
        return {"aliases": [
            {"alias": "Conductor", "component": "TaskDispatcher",
             "evidence_sentence": 1},  # "Conductor" not in sentence
        ]}

    k = _Knowledge()
    fully_llm_driven(k, sentences, components, fake_llm)
    assert k.aliases == {}, f"hallucination not rejected: {k.aliases}"
    print("  [pass] hallucination rejected by evidence-sentence guardrail")


def _test_full_name_in_evidence_rejected():
    """LLM cites sentence that contains both alias and full name — not a true alias use."""
    components = [_Component("TaskDispatcher")]
    sentences = [_Sentence(1, "The TaskDispatcher is a Dispatcher for jobs.")]

    def fake_llm(_prompt):
        return {"aliases": [
            {"alias": "Dispatcher", "component": "TaskDispatcher",
             "evidence_sentence": 1},
        ]}

    k = _Knowledge()
    fully_llm_driven(k, sentences, components, fake_llm)
    assert k.aliases == {}, k.aliases
    print("  [pass] alias rejected when evidence contains full name (not standalone)")


def _test_parity_with_current():
    """LLM-only and structural+verify should produce same aliases on clean input."""
    components = [_Component("TaskDispatcher"), _Component("QueueBroker")]
    sentences = [
        _Sentence(1, "The TaskDispatcher routes jobs."),
        _Sentence(2, "The Dispatcher hands jobs to workers."),
        _Sentence(3, "The QueueBroker buffers them."),
        _Sentence(4, "The Broker fsyncs to disk."),
    ]

    def fake_llm_only(_prompt):
        return {"aliases": [
            {"alias": "Dispatcher", "component": "TaskDispatcher", "evidence_sentence": 2},
            {"alias": "Broker", "component": "QueueBroker", "evidence_sentence": 4},
        ]}

    def fake_verify(candidates, _sents):
        return {w for w, _ in candidates}

    k1 = _Knowledge()
    fully_llm_driven(k1, sentences, components, fake_llm_only)

    k2 = _Knowledge()
    structural_plus_llm_verify(k2, sentences, components, fake_verify)

    assert k1.aliases == k2.aliases, f"parity broken\n  llm-only: {k1.aliases}\n  current:  {k2.aliases}"
    print(f"  [pass] parity: both produce {k1.aliases}")


def run_tests():
    print("Spike 001: llm-trailing-words tests")
    _test_happy_path()
    _test_hallucination_rejected()
    _test_full_name_in_evidence_rejected()
    _test_parity_with_current()
    print("All tests PASSED")


if __name__ == "__main__":
    run_tests()
