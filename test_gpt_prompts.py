#!/usr/bin/env python3
"""Unit tests for prompt compatibility between Claude Sonnet and GPT-5.2.

Tests each pipeline phase's prompt against the OpenAI backend to find
behavioral divergences. Run with:
    PYTHONPATH=src python test_gpt_prompts.py [--phase N] [--dataset DS]
"""
import os, sys, json, argparse, time

os.environ.setdefault("LLM_BACKEND", "openai")
os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.2")

# Load .env file
from pathlib import Path
_env_file = Path(".env")
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, "src")
from llm_sad_sam.llm_client import LLMClient, LLMBackend, LLMResponse

# ── Test data: real component lists from each dataset ──────────────────

DATASETS = {
    "teammates": {
        "components": ["Common", "UI", "Logic", "Storage", "Test Driver", "E2E", "Client", "GAE Datastore"],
        # Expected (Claude): ambiguous = {Common, Logic}
        "expected_ambiguous": {"Common", "Logic"},
    },
    "bigbluebutton": {
        "components": ["HTML5 Server", "Redis DB", "kurento", "Presentation Conversion",
                       "FreeSWITCH", "HTML5 Client", "BBB web", "Recording Service",
                       "FSESL", "WebRTC-SFU", "Redis PubSub", "Apps"],
        "expected_ambiguous": {"Apps"},
    },
    "jabref": {
        "components": ["gui", "preferences", "model", "logic", "cli", "globals"],
        "expected_ambiguous": {"logic", "globals"},
    },
    "mediastore": {
        "components": ["Facade", "Cache", "MediaManagement", "UserManagement",
                       "UserDBAdapter", "MediaAccess", "AudioWatermarking",
                       "TagWatermarking", "DownloadLoadBalancer", "ParallelWatermarking",
                       "ReEncoder", "Packaging", "FileStorage", "DataStorage"],
        "expected_ambiguous": {"Cache", "Facade"},
    },
    "teastore": {
        "components": ["WebUI", "Auth", "Recommender", "Persistence", "ImageProvider",
                       "Registry", "OrderBasedRecommender", "SlopeOneRecommender",
                       "DummyRecommender", "PopularityBasedRecommender",
                       "PreprocessedSlopeOneRecommender"],
        "expected_ambiguous": set(),
    },
}

# ── FEW-SHOT (from model_analyzer.py) ──────────────────────────────────

FEW_SHOT = """EXAMPLE 1:
NAMES: Lexer, Parser, CodeGenerator, Optimizer, Core, Util, AST, SymbolTable, Base
→ architectural: ["Lexer", "Parser", "CodeGenerator", "Optimizer", "AST", "SymbolTable"]
→ ambiguous: ["Core", "Util", "Base"]
Reasoning: Lexer/Parser/Optimizer name specific compilation roles. Core/Util/Base are
organizational labels that tell you nothing about what the component does.

EXAMPLE 2:
NAMES: Scheduler, Dispatcher, MemoryManager, Monitor, Pool, Helper, ProcessTable
→ architectural: ["Scheduler", "Dispatcher", "MemoryManager", "ProcessTable"]
→ ambiguous: ["Monitor", "Pool", "Helper"]
Reasoning: Scheduler/Dispatcher name specific OS roles. Monitor and Pool are common
English words regularly used generically ("monitor performance", "thread pool").
Helper is an organizational label.

EXAMPLE 3:
NAMES: RenderEngine, SceneGraph, Pipeline, Layer, Proxy, Socket, Router
→ architectural: ["RenderEngine", "SceneGraph", "Socket", "Router"]
→ ambiguous: ["Pipeline", "Layer", "Proxy"]
Reasoning: RenderEngine/SceneGraph are CamelCase compounds — always architectural.
Socket/Router name specific networking roles. Pipeline/Layer/Proxy are common words
used generically in documentation ("processing pipeline", "network layer", "behind a proxy")."""

# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Phase 1 — Component Classification
# ═══════════════════════════════════════════════════════════════════════

def build_phase1_prompt_original(names):
    """Current Phase 1 prompt (from model_analyzer.py)."""
    return f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{FEW_SHOT}

NOW CLASSIFY THE NAMES ABOVE.

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that could easily be used as ordinary words in documentation"]
}}

RULES:
1. ARCHITECTURAL: Names that refer to a specific role or responsibility. If the name tells you
   WHAT the component does (scheduling, parsing, rendering, storing data, managing users), it is
   architectural — even if the word also exists in a dictionary.
   Multi-word names, CamelCase compounds, and abbreviations (API, TCP, RPC) → always architectural.

2. AMBIGUOUS: Short single words that writers commonly use generically in software documentation.
   The test: "Could a technical writer naturally write this word in a sentence about ANY system
   without referring to a specific component?" If yes → ambiguous.

JSON only:"""


def build_phase1_prompt_v2(names):
    """V2 Phase 1 prompt — clarifies edge cases for GPT compatibility."""
    return f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{FEW_SHOT}

NOW CLASSIFY THE NAMES ABOVE.

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that could easily be used as ordinary words in documentation"]
}}

RULES:
1. ARCHITECTURAL: Names that are STRUCTURALLY identifiers — not ordinary English words.
   Always architectural:
   - Multi-word names or hyphenated names (e.g., "Test Driver", "BBB web")
   - CamelCase compounds (e.g., "MediaAccess", "WebRTC-SFU")
   - ALL-UPPERCASE abbreviations (e.g., "DB", "API", "UI", "FSESL")
   - Names with digits (e.g., "HTML5 Server")
   Also architectural: names that describe a SPECIFIC, uncommon technical role
   (e.g., "Scheduler", "Parser", "Router") — even if the word also exists in a dictionary.

2. AMBIGUOUS: Single ordinary-English words that writers commonly use as generic nouns
   or adjectives in software documentation, where the word does NOT uniquely identify
   a technical role.
   The test: "Could a technical writer naturally write this word with its plain English
   meaning in a sentence about ANY system, without intending to name a specific component?"
   Examples: "the system logic", "common utilities", "a caching layer", "the proxy server"
   → "logic", "common", "cache", "proxy" would be ambiguous.
   Counter-examples: "the parser transforms input" → "parser" names a specific role.

IMPORTANT: When in doubt, classify as AMBIGUOUS. It is safer to flag a name as ambiguous
(it will still be linked, just with extra validation) than to miss an ambiguous name.

JSON only:"""


def build_phase1_prompt_v3(names):
    """V3 Phase 1 prompt — balanced, zero benchmark leakage."""
    return f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{FEW_SHOT}

NOW CLASSIFY THE NAMES ABOVE.

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that could easily be used as ordinary words in documentation"]
}}

RULES:
1. ARCHITECTURAL: Names that identify a specific component role or function.
   Always architectural:
   - Multi-word names, hyphenated names
   - CamelCase compounds
   - ALL-UPPERCASE abbreviations (API, TCP, RPC)
   - Well-known computing abbreviations in ANY letter case (vm, io, os, tcp, rpc)
   - Names that describe a specific technical function: Scheduler (=task scheduling),
     Dispatcher (=event routing), Router (=request routing), Renderer (=rendering output)

2. AMBIGUOUS: Single words that a technical writer would commonly use as a GENERIC noun
   in everyday documentation, where the word has a strong plain-English meaning separate
   from any component.
   The test: "Would a writer use this word generically in 'the system X' or 'the X layer'
   without referring to a specific component?"
   - "the system core" → yes, "core" is generic → AMBIGUOUS
   - "a utility module" → yes, "util" is generic → AMBIGUOUS
   - "the base class" → yes, "base" is generic → AMBIGUOUS
   - "a helper function" → yes, "helper" is generic → AMBIGUOUS
   - "a processing pipeline" → yes, "pipeline" is generic → AMBIGUOUS
   - "the network layer" → yes, "layer" is generic → AMBIGUOUS
   Counter-examples where the word IS a specific role (from the few-shot examples):
   - "the Scheduler assigns threads" → "Scheduler" names a specific function → ARCHITECTURAL
   - "the Router forwards packets" → "Router" names a specific function → ARCHITECTURAL

JSON only:"""


def test_phase1(llm, dataset_name):
    """Test Phase 1 classification."""
    ds = DATASETS[dataset_name]
    names = ds["components"]
    expected_ambig = ds["expected_ambiguous"]

    print(f"\n{'='*70}")
    print(f"PHASE 1: Component Classification — {dataset_name}")
    print(f"{'='*70}")
    print(f"Components: {names}")
    print(f"Expected ambiguous (Claude baseline): {sorted(expected_ambig)}")

    for label, builder in [("ORIGINAL", build_phase1_prompt_original),
                           ("V3 (balanced)", build_phase1_prompt_v3)]:
        prompt = builder(names)
        response = llm.query(prompt, timeout=60)
        data = llm.extract_json(response)

        if not data:
            print(f"\n  {label}: FAILED — no valid JSON")
            continue

        arch = set(data.get("architectural", []))
        ambig = set(data.get("ambiguous", []))

        # Apply structural guard (from model_analyzer.py)
        import re
        def is_structural(n):
            if ' ' in n or '-' in n: return True
            if re.search(r'[a-z][A-Z]', n): return True
            if n.isupper(): return True
            return False
        ambig = {n for n in ambig if len(n.split()) == 1 and not is_structural(n)}

        match = ambig == expected_ambig
        extra = ambig - expected_ambig
        missing = expected_ambig - ambig

        status = "✓ PASS" if match else "✗ FAIL"
        print(f"\n  {label}: {status}")
        print(f"    architectural: {sorted(arch)}")
        print(f"    ambiguous (after guard): {sorted(ambig)}")
        if extra: print(f"    EXTRA (over-classified): {sorted(extra)}")
        if missing: print(f"    MISSING (under-classified): {sorted(missing)}")


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Phase 3 — Document Knowledge Judge
# ═══════════════════════════════════════════════════════════════════════

PHASE3_JUDGE_CASES = {
    "teammates": {
        "comp_names": ["Common", "UI", "Logic", "Storage", "Test Driver", "E2E", "Client", "GAE Datastore"],
        "mappings": [
            "'GAE' -> GAE Datastore (abbrev)",
            "'UI Browser' -> UI (synonym)",
            "'UI Server' -> UI (synonym)",
            "'back end' -> Logic (synonym)",
            "'storage layer' -> Storage (synonym)",
            "'end-to-end component' -> E2E (synonym)",
            "'NoSQL database' -> GAE Datastore (synonym)",
            "'datastore' -> GAE Datastore (partial)",
        ],
        # At minimum, GAE should be approved (it's an abbreviation from the component name)
        "must_approve": {"GAE", "datastore"},
        "should_reject": {"storage layer"},  # too generic
    },
}


def build_phase3_judge_original(comp_names, mapping_list):
    """Current Phase 3 judge prompt."""
    return f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

EXAMPLES — study these to calibrate your judgment:

Example 1 — APPROVE (proper name in context):
  'Scheduler' -> TaskScheduler (partial)
  Document says: "The Scheduler assigns threads to available cores."
  Verdict: APPROVE. "Scheduler" is capitalized mid-sentence and used as a proper
  name for the TaskScheduler component, not as a generic concept.

Example 2 — APPROVE (CamelCase identifier):
  'RenderEngine' -> GameRenderEngine (synonym)
  Document says: "The RenderEngine processes draw calls each frame."
  Verdict: APPROVE. CamelCase is a constructed identifier — it is a proper name,
  not a generic English word.

Example 3 — APPROVE (abbreviation with document evidence):
  'AST' -> AbstractSyntaxTree (abbrev)
  Document says: "The Abstract Syntax Tree (AST) represents the parsed program."
  Verdict: APPROVE. Explicitly defined in the document with parenthetical pattern.

Example 4 — REJECT (generic concept, not a component name):
  'process' -> OrderProcessor (partial)
  Document says: "The system will process incoming requests."
  Verdict: REJECT. "process" is used as a verb in its ordinary English sense,
  not as a name for OrderProcessor.

Example 5 — APPROVE (distinctive partial used as proper name):
  'Dispatcher' -> EventDispatcher (partial)
  Document says: "When an event arrives, the Dispatcher routes it to handlers."
  Verdict: APPROVE. "Dispatcher" is capitalized mid-sentence and refers
  specifically to EventDispatcher — it is a distinctive term in this document.

Example 6 — REJECT (ambiguous, refers to the whole system):
  'system' -> PaymentSystem (partial)
  Document says: "The system handles all transactions."
  Verdict: REJECT. "system" refers to the overall system, not specifically
  to PaymentSystem.

NOW JUDGE THE PROPOSED MAPPINGS. Apply these rules:

REJECT if ANY of these are true:
- The term is used in its ordinary English sense, NOT as a name for the component
- The term refers to a different component or to the system as a whole
- The mapping cannot be verified from the actual document text

APPROVE if ANY of these are true:
- The term is a CamelCase identifier (mixed lower-then-upper like "PaymentGateway")
  — CamelCase terms are constructed proper names, not generic English
- The term appears capitalized mid-sentence in the document — this signals
  proper name usage (e.g., "The Optimizer runs" uses Optimizer as a name)
- The term is used AS A NAME for the component in context, refers to exactly
  one component, and this can be verified from the document

Return JSON:
{{
  "approved": ["term1", "term2"],
  "generic_rejected": ["generic_term1"]
}}
JSON only:"""


def build_phase3_judge_v2(comp_names, mapping_list):
    """Improved Phase 3 judge — biases toward approval, clarifies without-context judgment."""
    return f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

EXAMPLES — study these to calibrate your judgment:

Example 1 — APPROVE (abbreviation from component name):
  'AST' -> AbstractSyntaxTree (abbrev)
  Verdict: APPROVE. "AST" is the initials of "AbstractSyntaxTree". Abbreviations
  formed from the component name's words are always valid.

Example 2 — APPROVE (trailing word of multi-word name):
  'Dispatcher' -> EventDispatcher (partial)
  Verdict: APPROVE. "Dispatcher" is the last word of "EventDispatcher".
  If no other component ends in "Dispatcher", this partial is unambiguous.

Example 3 — APPROVE (CamelCase identifier):
  'RenderEngine' -> GameRenderEngine (synonym)
  Verdict: APPROVE. CamelCase is a constructed identifier — always a proper name.

Example 4 — APPROVE (trailing word of multi-word name):
  'Table' -> SymbolTable (partial)
  Verdict: APPROVE. "Table" is the trailing word of "SymbolTable" and
  likely refers to this specific component when no other component uses "Table".

Example 5 — REJECT (generic concept, not a component name):
  'process' -> OrderProcessor (partial)
  Verdict: REJECT. "process" is an ordinary English verb/noun used generically
  in many contexts ("process requests", "the process").

Example 6 — REJECT (refers to whole system):
  'system' -> PaymentSystem (partial)
  Verdict: REJECT. "system" is too generic — it could refer to the overall system.

DECISION RULES (apply in order):

1. AUTO-APPROVE these — they are always valid mappings:
   - Abbreviations formed from the component name's initials or words
   - Trailing words of multi-word component names (if no other component shares that word)
   - CamelCase identifiers
   - Multi-word phrases that contain the component name

2. APPROVE if the term plausibly refers to exactly one component and is NOT
   a generic word like "system", "process", "service", "component", "module".

3. REJECT only if the term is clearly generic and could refer to anything,
   or clearly refers to a different component or the system as a whole.

IMPORTANT: When in doubt, APPROVE. False approvals are filtered by later
pipeline stages; false rejections cause permanent recall loss.

Return JSON:
{{
  "approved": ["term1", "term2"],
  "generic_rejected": ["generic_term1"]
}}
JSON only:"""


def test_phase3_judge(llm, dataset_name="teammates"):
    """Test Phase 3 judge."""
    if dataset_name not in PHASE3_JUDGE_CASES:
        print(f"\nNo Phase 3 test data for {dataset_name}")
        return

    tc = PHASE3_JUDGE_CASES[dataset_name]
    print(f"\n{'='*70}")
    print(f"PHASE 3: Document Knowledge Judge — {dataset_name}")
    print(f"{'='*70}")
    print(f"Mappings to judge: {len(tc['mappings'])}")
    print(f"Must approve: {tc['must_approve']}")

    for label, builder in [("ORIGINAL", build_phase3_judge_original),
                           ("V2 (approve-biased)", build_phase3_judge_v2)]:
        prompt = builder(tc["comp_names"], tc["mappings"])
        response = llm.query(prompt, timeout=60)
        data = llm.extract_json(response)

        if not data:
            print(f"\n  {label}: FAILED — no valid JSON")
            continue

        approved = set(data.get("approved", []))
        rejected = set(data.get("generic_rejected", []))

        # Check must_approve
        approved_must = tc["must_approve"] & approved
        missed_must = tc["must_approve"] - approved

        print(f"\n  {label}:")
        print(f"    Approved: {sorted(approved)}")
        print(f"    Rejected: {sorted(rejected)}")
        if missed_must:
            print(f"    ✗ MISSED must-approve: {sorted(missed_must)}")
        else:
            print(f"    ✓ All must-approve terms approved")


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Phase 5 — Sentence number format
# ═══════════════════════════════════════════════════════════════════════

def test_phase5_format(llm):
    """Test if GPT returns 'S101' or 101 for sentence numbers."""
    print(f"\n{'='*70}")
    print(f"PHASE 5: Sentence Number Format Test")
    print(f"{'='*70}")

    prompt = """Extract ALL references to software architecture components from this document.

COMPONENTS: Scheduler, Dispatcher, Renderer, Pool

RULES — include a reference when:
1. The component name appears directly in the sentence
2. The sentence describes what a specific component does by name

DOCUMENT:
S101: The Scheduler component assigns tasks to available threads.
S102: Dispatcher is responsible for routing events to handlers.
S103: The output is generated by the Renderer.
S104: Pool manages the allocation of reusable resources.

Return JSON:
{"references": [{"sentence": N, "component": "Name", "matched_text": "text", "match_type": "exact"}]}
JSON only:"""

    response = llm.query(prompt, timeout=60)
    print(f"  Response OK={response.success}, text={response.text[:200] if response.text else 'EMPTY'}")
    data = llm.extract_json(response)

    if not data:
        print("  FAILED — no valid JSON")
        return

    refs = data.get("references", [])
    print(f"  Got {len(refs)} references")
    for ref in refs:
        snum = ref.get("sentence")
        stype = type(snum).__name__
        is_string = isinstance(snum, str)
        has_prefix = is_string and str(snum).startswith("S")
        status = "✗ STRING WITH S-PREFIX" if has_prefix else ("⚠ STRING" if is_string else "✓ INTEGER")
        print(f"    S{snum} ({stype}): {status} → {ref.get('component')}")

    # Test with explicit instruction
    prompt2 = prompt.replace(
        '{"references": [{"sentence": N,',
        '{"references": [{"sentence": N (INTEGER, e.g. 101 not "S101"),'
    )

    print(f"\n  With explicit INTEGER instruction:")
    response2 = llm.query(prompt2, timeout=60)
    data2 = llm.extract_json(response2)
    if data2:
        for ref in data2.get("references", []):
            snum = ref.get("sentence")
            stype = type(snum).__name__
            has_prefix = isinstance(snum, str) and str(snum).startswith("S")
            status = "✗ STRING WITH S-PREFIX" if has_prefix else ("⚠ STRING" if isinstance(snum, str) else "✓ INTEGER")
            print(f"    S{snum} ({stype}): {status} → {ref.get('component')}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, help="Test only this phase (1, 3, or 5)")
    parser.add_argument("--dataset", default="all", help="Dataset to test (default: all)")
    args = parser.parse_args()

    llm = LLMClient(backend=LLMBackend.OPENAI)
    print(f"Backend: {llm.backend}, Model: {os.environ.get('OPENAI_MODEL_NAME', 'default')}")

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    if args.phase is None or args.phase == 5:
        test_phase5_format(llm)

    if args.phase is None or args.phase == 1:
        for ds in datasets:
            test_phase1(llm, ds)

    if args.phase is None or args.phase == 3:
        test_phase3_judge(llm, "teammates")


if __name__ == "__main__":
    main()
