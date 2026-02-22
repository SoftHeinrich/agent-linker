#!/usr/bin/env python3
"""Test Phase 1 classification prompt variants — unit test style.

Runs classification on all 4 datasets, multiple prompt variants, 2 runs each.
Checks stability + coverage against expected ambiguous/architectural sets.

Usage:
    python test_analyzer.py                      # all variants, 2 runs
    python test_analyzer.py --variants A2,A5     # specific variants
    python test_analyzer.py --runs 3             # 3 runs for stability
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.pcm_parser import parse_pcm_repository

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)

DATASETS = {
    "ms": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
    "ts": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
    "tm": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
    "bbb": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
}

# ── Unit test expectations ───────────────────────────────────────────
# Names that MUST be architectural (catching these as ambiguous = regression)
MUST_ARCH = {
    "ts": {"Persistence", "Recommender", "Registry", "WebUI", "ImageProvider"},
    "ms": {"FileStorage", "Packaging", "Reencoding", "MediaAccess"},
    "bbb": {"FreeSWITCH", "kurento", "FSESL"},
    "tm": {"GAE Datastore", "Test Driver", "E2E"},
}
# Names that SHOULD be ambiguous (missing these = weak prompt)
WANT_AMBIG = {
    "tm": {"Common", "Logic"},       # top FP sources
    "bbb": {"Apps"},                  # generic
}

# ── Prompt variants ──────────────────────────────────────────────────

# Few-shot example block (safe textbook domains only)
FEW_SHOT = """
EXAMPLE 1:
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
used generically in documentation ("processing pipeline", "network layer", "behind a proxy").""".strip()


def prompt_A2(names):
    """Generic-in-docs — the previous best."""
    return f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that could easily be used as ordinary words in documentation"]
}}

CLASSIFICATION RULES:
1. ARCHITECTURAL: The name refers to a specific, well-defined architectural responsibility that
   is unlikely to appear as a generic word in documentation prose.
   Examples: "Scheduler", "Renderer", "Compiler", "Tokenizer" — these name specific roles.
   Also architectural: multi-word names, CamelCase, abbreviations (DB, API, UI).

2. AMBIGUOUS: The name is a short, common English word that writers frequently use generically
   in software documentation WITHOUT meaning the component. Think: could a technical writer
   naturally use this word in a sentence about ANY system?
   - "common" → "a common pattern" (generic usage, not the component)
   - "logic" → "business logic" (generic usage, not the component)
   - "client" → "the client sends a request" (could be generic or the component)
   - "apps" → "mobile apps" (generic usage)
   Test: If you see the word in a sentence, would you need extra context to know whether
   it refers to the component or is just an ordinary English word?

3. Names that describe a SPECIFIC technical function are architectural even if short:
   "Scheduler" → always means the scheduling component
   "Compiler" → always means the compilation component
   These are unambiguous because their meaning is specific.

JSON only:"""


def prompt_A5(names):
    """A2 + few-shot examples from safe domains."""
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
   Multi-word names, CamelCase compounds, and abbreviations (DB, API, UI) → always architectural.

2. AMBIGUOUS: Short single words that writers commonly use generically in software documentation.
   The test: "Could a technical writer naturally write this word in a sentence about ANY system
   without referring to a specific component?" If yes → ambiguous.

JSON only:"""


def prompt_A6(names):
    """Few-shot + explicit CamelCase/suffix guard in prompt."""
    return f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{FEW_SHOT}

NOW CLASSIFY THE NAMES ABOVE.

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that blend in as ordinary English words in documentation"]
}}

RULES:
1. ALWAYS architectural (never ambiguous):
   - Multi-word names: they are invented compound identifiers
   - CamelCase names (e.g., CodeGenerator, MemoryManager): invented identifiers
   - ALL-CAPS abbreviations (DB, UI, API, AST): these are identifiers, not words
   - Names ending in -er, -or, -tion, -ence, -ment, -ing, -ory, -ary, -ade:
     these suffixes indicate a specific role (Scheduler, Optimizer, Dispatcher)

2. AMBIGUOUS: Single common English words WITHOUT a role-indicating suffix, that
   are regularly used generically in technical writing.
   Test: would a writer use this word in a sentence about ANY system without
   referring to a specific component? If yes → ambiguous.

3. When in doubt → architectural.

JSON only:"""


def prompt_A7(names):
    """Few-shot + chain-of-thought: ask LLM to reason before classifying."""
    return f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{FEW_SHOT}

NOW CLASSIFY THE NAMES ABOVE.

For each name, briefly ask yourself:
- Is this a single common English word, or a compound/technical term?
- Could a writer use this word generically in documentation about any system?
- Does it name a SPECIFIC role (like "Scheduler" or "Parser")?

Then return JSON:
{{
  "architectural": ["names that clearly identify specific components"],
  "ambiguous": ["single common words that blend into ordinary documentation prose"]
}}

IMPORTANT:
- Multi-word, CamelCase, and ALL-CAPS names → always architectural
- Names with role suffixes (-er, -or, -tion, -ence, -ment, -ing) → architectural
- Only short single words commonly used as generic English → ambiguous

JSON only:"""


VARIANTS = {
    "A2": ("generic-in-docs (no fewshot)", prompt_A2),
    "A5": ("fewshot basic", prompt_A5),
    "A6": ("fewshot + suffix guard", prompt_A6),
    "A7": ("fewshot + chain-of-thought", prompt_A7),
}


def _is_structurally_unambiguous(name):
    """CamelCase, multi-word, or all-caps → always architectural."""
    if ' ' in name or '-' in name:
        return True
    if re.search(r'[a-z][A-Z]', name):
        return True
    if name.isupper():
        return True
    return False


def classify(names, prompt_fn, llm):
    """Run a single classification with CamelCase code guard."""
    prompt = prompt_fn(names)
    response = llm.query(prompt, timeout=120)
    data = llm.extract_json(response)
    if not data:
        return set(), set()

    valid = set(names)
    arch = set(data.get("architectural", data.get("clear", []))) & valid
    ambig = set(data.get("ambiguous", data.get("confusable", []))) & valid
    # Code guard: single-word, not CamelCase, not all-caps
    ambig = {n for n in ambig
             if len(n.split()) == 1 and not _is_structurally_unambiguous(n)}
    return arch, ambig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    var_keys = args.variants.split(",") if args.variants else list(VARIANTS.keys())
    ds_keys = args.datasets.split(",") if args.datasets else list(DATASETS.keys())
    n_runs = args.runs

    llm = LLMClient(backend=LLMBackend.CLAUDE)

    # results[variant][dataset] = list of ambiguous sets per run
    results = {}

    for var_key in var_keys:
        label, prompt_fn = VARIANTS[var_key]
        results[var_key] = {}
        print(f"\n{'#'*60}")
        print(f"  {var_key}: {label}")
        print(f"{'#'*60}")

        for ds_key in ds_keys:
            model_path = DATASETS.get(ds_key)
            if not model_path or not model_path.exists():
                continue

            components = parse_pcm_repository(str(model_path))
            names = [c.name for c in components]
            results[var_key][ds_key] = []

            for run in range(n_runs):
                _, ambig = classify(names, prompt_fn, llm)
                results[var_key][ds_key].append(ambig)
                print(f"  {ds_key} run{run+1}: {sorted(ambig)}")

    # ── Stability matrix ──
    print(f"\n{'='*80}")
    print("STABILITY")
    print(f"{'='*80}")
    for var_key in var_keys:
        stable_count = 0
        total = 0
        for ds_key in ds_keys:
            runs = results.get(var_key, {}).get(ds_key, [])
            if len(runs) >= 2:
                total += 1
                if all(r == runs[0] for r in runs):
                    stable_count += 1
        print(f"  {var_key}: {stable_count}/{total} datasets stable")

    # ── Unit test checks ──
    print(f"\n{'='*80}")
    print("UNIT TESTS")
    print(f"{'='*80}")
    for var_key in var_keys:
        label, _ = VARIANTS[var_key]
        passes = 0
        fails = 0
        details = []

        for ds_key in ds_keys:
            runs = results.get(var_key, {}).get(ds_key, [])
            if not runs:
                continue

            # Use union across runs for coverage check
            all_ambig = set().union(*runs)

            # Check MUST_ARCH: these must NOT appear in ambiguous
            must_arch = MUST_ARCH.get(ds_key, set())
            wrongly_ambig = must_arch & all_ambig
            if wrongly_ambig:
                fails += 1
                details.append(f"  FAIL {ds_key}: {sorted(wrongly_ambig)} wrongly ambiguous (should be architectural)")
            else:
                passes += 1

            # Check WANT_AMBIG: these SHOULD appear in ambiguous
            want = WANT_AMBIG.get(ds_key, set())
            missed = want - all_ambig
            if missed:
                details.append(f"  WEAK {ds_key}: missed {sorted(missed)} (wanted ambiguous)")

        # Stability check
        for ds_key in ds_keys:
            runs = results.get(var_key, {}).get(ds_key, [])
            if len(runs) >= 2 and not all(r == runs[0] for r in runs):
                fails += 1
                details.append(f"  FAIL {ds_key}: unstable across runs")

        status = "PASS" if fails == 0 else "FAIL"
        print(f"\n  {var_key} ({label}): {status} ({passes} pass, {fails} fail)")
        for d in details:
            print(d)

    # ── Comparison table ──
    print(f"\n{'='*80}")
    print("COMPARISON (union across runs)")
    print(f"{'='*80}")
    print(f"{'DS':<5}", end="")
    for var_key in var_keys:
        print(f" {var_key:<30}", end="")
    print()
    print("-" * (5 + 31 * len(var_keys)))

    for ds_key in ds_keys:
        print(f"{ds_key:<5}", end="")
        for var_key in var_keys:
            runs = results.get(var_key, {}).get(ds_key, [])
            if runs:
                union = sorted(set().union(*runs))
                print(f" {str(union):<30}", end="")
            else:
                print(f" {'N/A':<30}", end="")
        print()


if __name__ == "__main__":
    main()
