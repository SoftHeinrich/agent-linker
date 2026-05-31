"""Registration + structural tests for s_linker13_trim2_entval_clean (Plan 12-04).

Covers:
  - Imports + class identity (subclass of SLinker13Clean, _VARIANT_NAME pinned).
  - Merged rubric is the shared core of both ENTITY_EXTRACTION_RULES_V3 and
    VALIDATION_RULES_V3 (decision-divergent / rubric-shared design).
  - The two V3 constants differ from prompts_v2 counterparts (variant IS active).
  - Rule-count contraction: merged core ≤ 10 numbered rules (14 → 10 target).
  - Coverage-preservation guard: 9 semantic markers present.
  - GATE-06 spot probe: no benchmark-component leakage.
  - "Favor inclusion" tie-breaker preserved in extraction-side framing
    (V31 phase-contribution analysis: load-bearing).
  - Both extraction-side and validation-side framings detectable.
  - run_ablation registration with canonical=False.
  - Frozen-file safety: variant lives in its own file; parent module + prompts_v2
    + other frozen modules unchanged.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

# 9-name benchmark-component leakage probe (same regex as Plan 12-03 / 12-01).
BENCHMARK_LEAK_PROBE = re.compile(
    r"(?i)\b(Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|"
    r"HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper)\b"
)

# Semantic categories that must survive the merge (case-insensitive substring).
COVERAGE_KEYWORDS = [
    "alias", "synonym", "compound", "interaction", "passive",
    "prepositional", "dotted", "heading", "ordinary",
]


def test_imports_succeed():
    from llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean import (
        SLinker13Trim2EntvalClean,
        ENTVAL_MERGED_RUBRIC_V3,
        ENTITY_EXTRACTION_RULES_V3,
        VALIDATION_RULES_V3,
    )
    # symbols all bound, non-empty
    assert SLinker13Trim2EntvalClean is not None
    assert isinstance(ENTVAL_MERGED_RUBRIC_V3, str) and ENTVAL_MERGED_RUBRIC_V3
    assert isinstance(ENTITY_EXTRACTION_RULES_V3, str) and ENTITY_EXTRACTION_RULES_V3
    assert isinstance(VALIDATION_RULES_V3, str) and VALIDATION_RULES_V3


def test_variant_class_identity():
    from llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean import (
        SLinker13Trim2EntvalClean,
    )
    from llm_sad_sam.linkers.experimental.s_linker13_clean import SLinker13Clean

    assert issubclass(SLinker13Trim2EntvalClean, SLinker13Clean)
    assert SLinker13Trim2EntvalClean._VARIANT_NAME == "s_linker13_trim2_entval_clean"


def test_shared_core_is_substring_of_both_consumer_constants():
    """Rubric-shared / decision-divergent design contract."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean import (
        ENTVAL_MERGED_RUBRIC_V3,
        ENTITY_EXTRACTION_RULES_V3,
        VALIDATION_RULES_V3,
    )
    assert ENTVAL_MERGED_RUBRIC_V3 in ENTITY_EXTRACTION_RULES_V3
    assert ENTVAL_MERGED_RUBRIC_V3 in VALIDATION_RULES_V3
    # Headers differ — each consumer gets a role-specific framing.
    assert ENTITY_EXTRACTION_RULES_V3 != VALIDATION_RULES_V3


def test_v3_constants_differ_from_prompts_v2():
    """Variant IS modifying the prompts (negative control)."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean import (
        ENTITY_EXTRACTION_RULES_V3,
        VALIDATION_RULES_V3,
    )
    from llm_sad_sam.linkers.experimental import prompts_v2

    assert ENTITY_EXTRACTION_RULES_V3 != prompts_v2.ENTITY_EXTRACTION_RULES
    assert VALIDATION_RULES_V3 != prompts_v2.VALIDATION_RULES


def test_rule_count_contraction():
    """Merged rubric body has ≤ 10 numbered rules (down from 14 across originals)."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean import (
        ENTVAL_MERGED_RUBRIC_V3,
    )
    # Count occurrences of "\nN. " markers (numbered list items).
    n = len(re.findall(r"\n\d+\.\s", ENTVAL_MERGED_RUBRIC_V3))
    assert n <= 10, f"merged rubric has {n} rules, expected <= 10"
    # And must be > 0 (sanity).
    assert n > 0


def test_coverage_preservation():
    """Every semantic category from the originals appears in the merged rubric."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean import (
        ENTVAL_MERGED_RUBRIC_V3,
    )
    body_lower = ENTVAL_MERGED_RUBRIC_V3.lower()
    missing = [kw for kw in COVERAGE_KEYWORDS if kw not in body_lower]
    assert not missing, f"missing semantic markers: {missing}"


def test_no_benchmark_component_leakage():
    """GATE-06 spot probe: merged rubric body contains zero benchmark leakage."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean import (
        ENTVAL_MERGED_RUBRIC_V3,
        ENTITY_EXTRACTION_RULES_V3,
        VALIDATION_RULES_V3,
    )
    for body in (
        ENTVAL_MERGED_RUBRIC_V3,
        ENTITY_EXTRACTION_RULES_V3,
        VALIDATION_RULES_V3,
    ):
        assert not BENCHMARK_LEAK_PROBE.search(body), \
            f"benchmark-component leak in: {body!r}"


def test_favor_inclusion_preserved_extraction_side():
    """V31 phase-contribution analysis: this tie-breaker is load-bearing."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean import (
        ENTITY_EXTRACTION_RULES_V3,
    )
    assert "Favor inclusion" in ENTITY_EXTRACTION_RULES_V3 or \
           "favor inclusion" in ENTITY_EXTRACTION_RULES_V3


def test_extraction_and_validation_framings_detectable():
    """Each consumer constant has a role-specific framing the LLM can use."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean import (
        ENTITY_EXTRACTION_RULES_V3,
        VALIDATION_RULES_V3,
    )
    # Extraction framing: "include" appears.
    assert re.search(r"\binclude\b", ENTITY_EXTRACTION_RULES_V3, re.IGNORECASE)
    # Validation framing: "approve" or "reject" appears.
    assert re.search(r"\b(approve|reject)\b", VALIDATION_RULES_V3, re.IGNORECASE)


def test_registered_in_run_ablation():
    from run_ablation import CANONICAL_VARIANTS, VARIANT_SPECS

    assert "s_linker13_trim2_entval_clean" in CANONICAL_VARIANTS
    spec = VARIANT_SPECS["s_linker13_trim2_entval_clean"]
    assert spec["canonical"] is False
    assert spec["class_name"] == "SLinker13Trim2EntvalClean"
    assert spec["module"] == \
        "llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean"
    assert "Technique 3" in spec["description"]


def test_frozen_files_unchanged():
    """T-12-04-01: zero edits to v2.0 frozen files or to s_linker13_clean."""
    frozen = [
        "src/llm_sad_sam/linkers/experimental/prompts_v2.py",
        "src/llm_sad_sam/linkers/experimental/s_linker13.py",
        "src/llm_sad_sam/linkers/experimental/s_linker13_clean.py",
        "src/llm_sad_sam/core/data_types_v2.py",
        "src/llm_sad_sam/core/document_loader_v2.py",
        "src/llm_sad_sam/pcm_parser_v2.py",
    ]
    result = subprocess.run(
        ["git", "diff", "--quiet", "--"] + frozen,
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert result.returncode == 0, (
        f"frozen files modified: rc={result.returncode}\n"
        f"stdout={result.stdout.decode()}\nstderr={result.stderr.decode()}"
    )


def test_instantiation_succeeds():
    """Variant must instantiate (smoke test before live ablation runs)."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean import (
        SLinker13Trim2EntvalClean,
    )
    from llm_sad_sam.llm_client import LLMBackend

    # Construct with CHECKPOINT backend so no live LLM is required.
    linker = SLinker13Trim2EntvalClean(backend=LLMBackend.CHECKPOINT)
    assert linker._VARIANT_NAME == "s_linker13_trim2_entval_clean"
    # The override methods should be on the subclass, not inherited.
    assert "_run_single_extraction_pass" in SLinker13Trim2EntvalClean.__dict__
    assert "_validate_with_evidence" in SLinker13Trim2EntvalClean.__dict__
