"""Registration + structural-guard tests for s_linker13_trim1_judge_clean (Plan 12-03 Task 1).

Pins:
  - Importability of the variant module + exported symbols.
  - Subclass relationship with SLinker13Clean.
  - _VARIANT_NAME isolation (distinct checkpoint namespace).
  - DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 byte-equal to v2 (V35a guard).
  - DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 actually differs from v2 (the trim IS applied).
  - Technique 8 ordering: "When in doubt" appears BEFORE any verdict directive.
  - Technique 3 distillation: zero numbered-rule markers in the rubric body.
  - Coverage preservation: all 4 AUTO-APPROVE sub-categories named.
  - GATE-06 spot check: benchmark-component-name probe returns empty.
  - run_ablation registration: CANONICAL_VARIANTS + VARIANT_SPECS entries
    exist with canonical=False.
  - Rubric length within 80-130% of the v2 original (Technique 3 lossless-
    density window).
"""

from __future__ import annotations

import re

import pytest


def test_module_imports_required_symbols():
    """All three required top-level names import without error."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import (  # noqa: F401
        SLinker13Trim1JudgeClean,
        DOC_KNOWLEDGE_JUDGE_RUBRIC_V3,
        DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3,
    )


def test_variant_name_distinct_checkpoint_namespace():
    """_VARIANT_NAME must override the parent so checkpoint cache directories
    do not collide with s_linker13_clean's per-variant cache subtree."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import (
        SLinker13Trim1JudgeClean,
    )
    assert SLinker13Trim1JudgeClean._VARIANT_NAME == "s_linker13_trim1_judge_clean"


def test_is_subclass_of_slinker13_clean():
    """The trim variant must inherit from SLinker13Clean — surgical override
    pattern, not file-copy."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import (
        SLinker13Trim1JudgeClean,
    )
    from llm_sad_sam.linkers.experimental.s_linker13_clean import SLinker13Clean
    assert issubclass(SLinker13Trim1JudgeClean, SLinker13Clean)


def test_judge_examples_byte_equal_v2():
    """V35a guard: the 7 worked examples are calibration substrate Claude
    leverages. Removing or rewriting them regresses Claude (V35a evidence,
    -2.5pp on TS). Pin byte-equality."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import (
        DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3,
    )
    from llm_sad_sam.linkers.experimental import prompts_v2
    assert DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 == prompts_v2.DOC_KNOWLEDGE_JUDGE_EXAMPLES


def test_judge_rubric_differs_from_v2():
    """The rubric IS modified — that's the whole point of the trim. Sanity
    check that we didn't accidentally alias instead of distill."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import (
        DOC_KNOWLEDGE_JUDGE_RUBRIC_V3,
    )
    from llm_sad_sam.linkers.experimental import prompts_v2
    assert DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 != prompts_v2.DOC_KNOWLEDGE_JUDGE_RULES


def test_technique8_reasoning_before_conclusion():
    """Technique 8 (arXiv 2603.13351): the tie-breaker ("When in doubt") must
    PRECEDE any verdict-format directive. The verdict format (Return JSON: ...)
    lives in the consumer method's prompt template, not in the rubric body —
    so we check the rubric body itself contains no premature verdict-format
    directive AFTER the tie-breaker."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import (
        DOC_KNOWLEDGE_JUDGE_RUBRIC_V3,
    )
    body = DOC_KNOWLEDGE_JUDGE_RUBRIC_V3
    assert "When in doubt" in body
    when_in_doubt_idx = body.index("When in doubt")
    # No JSON / Return / Verdict / format directive should appear before
    # the tie-breaker (we don't want the rubric to lead with answer-first).
    for premature in ("Return", "JSON", "verdict format", "Output:"):
        idx = body.find(premature)
        if idx != -1:
            assert idx > when_in_doubt_idx, (
                f"{premature!r} appears at offset {idx} before 'When in doubt' "
                f"at offset {when_in_doubt_idx} — violates Technique 8 "
                f"reasoning-before-conclusion order"
            )


def test_technique3_prose_not_numbered():
    """Technique 3 (lossless rubric distillation): the rubric is prose, not
    numbered rules. Count `\\n\\d. ` markers — should be zero. The original
    DOC_KNOWLEDGE_JUDGE_RULES used three numbered rules; the distilled form
    merges them into a single continuous block."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import (
        DOC_KNOWLEDGE_JUDGE_RUBRIC_V3,
    )
    numbered_markers = re.findall(r"\n\d\. ", DOC_KNOWLEDGE_JUDGE_RUBRIC_V3)
    assert len(numbered_markers) == 0, (
        f"Expected zero numbered-rule markers (Technique 3 prose form); "
        f"found {len(numbered_markers)}: {numbered_markers}"
    )


def test_auto_approve_coverage_preserved():
    """Coverage preservation guard: all 4 AUTO-APPROVE sub-categories from
    the original Rule 1 must appear in the distilled rubric (abbreviations,
    trailing-word, CamelCase, multi-word phrases). V35a's failure mode was
    losing edge-case coverage when examples replaced rules; we explicitly
    check that the distillation did NOT drop coverage."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import (
        DOC_KNOWLEDGE_JUDGE_RUBRIC_V3,
    )
    rubric_lower = DOC_KNOWLEDGE_JUDGE_RUBRIC_V3.lower()
    for kw in ("abbreviation", "trailing", "camelcase", "multi-word"):
        assert kw in rubric_lower, (
            f"Coverage gap: {kw!r} (AUTO-APPROVE sub-category from original "
            f"Rule 1) is missing from the distilled rubric"
        )


def test_gate06_benchmark_component_probe_clean():
    """GATE-06 spot check: the rubric body contains zero benchmark-component-
    name substrings. Probe covers the highest-risk surface from BENCHMARK_TABOO
    (MediaStore + TeaStore + BBB + JabRef components and aliases). The full
    TABOO sweep runs in Plan 12-06."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import (
        DOC_KNOWLEDGE_JUDGE_RUBRIC_V3,
    )
    probe = re.compile(
        r"\b(Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|"
        r"HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper|UserDBAdapter|"
        r"AudioWatermarking|MediaManagement|WebUI|Recommender|Persistence|"
        r"SlopeOneRecommender|ImageProvider|Datastore|JabRef|bibdatabase|"
        r"bibentry)\b",
        re.IGNORECASE,
    )
    hits = probe.findall(DOC_KNOWLEDGE_JUDGE_RUBRIC_V3)
    assert hits == [], (
        f"GATE-06 violation: benchmark-component names found in rubric: {hits}"
    )


def test_rubric_length_within_lossless_window():
    """Technique 3 is lossless density compression. Length must stay within
    80-130% of the original. Significantly shorter ==> probable coverage
    loss (V35-style). Significantly longer ==> unjustified expansion."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import (
        DOC_KNOWLEDGE_JUDGE_RUBRIC_V3,
    )
    from llm_sad_sam.linkers.experimental import prompts_v2
    orig = len(prompts_v2.DOC_KNOWLEDGE_JUDGE_RULES)
    new = len(DOC_KNOWLEDGE_JUDGE_RUBRIC_V3)
    lo, hi = 0.8 * orig, 1.3 * orig
    assert lo <= new <= hi, (
        f"Rubric length {new} bytes outside lossless-density window "
        f"[{lo:.0f}, {hi:.0f}] (original: {orig} bytes, ratio: {new/orig:.3f})"
    )


def test_registered_in_run_ablation_canonical_variants():
    """Plan 12-03 acceptance: the variant must appear in CANONICAL_VARIANTS
    so the existing sweep + harness tooling can discover it by name."""
    import run_ablation
    assert "s_linker13_trim1_judge_clean" in run_ablation.CANONICAL_VARIANTS


def test_registered_in_run_ablation_variant_specs():
    """Plan 12-03 acceptance: VARIANT_SPECS entry exists with canonical=False
    (trim variants are not canonical, only the eventual Phase 13 union is)."""
    import run_ablation
    spec = run_ablation.VARIANT_SPECS["s_linker13_trim1_judge_clean"]
    assert spec["canonical"] is False
    assert spec["class_name"] == "SLinker13Trim1JudgeClean"
    assert spec["module"] == (
        "llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean"
    )


def test_frozen_files_unchanged():
    """Defense in depth: ensure no frozen files were edited during Plan 12-03
    Task 1. Forbidden surfaces per CLEAN-01 / Phase 10 invariant."""
    import subprocess
    result = subprocess.run(
        [
            "git", "diff", "--quiet",
            "src/llm_sad_sam/linkers/experimental/prompts_v2.py",
            "src/llm_sad_sam/linkers/experimental/s_linker13.py",
            "src/llm_sad_sam/linkers/experimental/s_linker13_clean.py",
            "src/llm_sad_sam/core/data_types_v2.py",
            "src/llm_sad_sam/core/document_loader_v2.py",
            "src/llm_sad_sam/pcm_parser_v2.py",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, (
        "One or more frozen files have uncommitted edits. Plan 12-03 Task 1 "
        "forbids modifying any v2.0 frozen file or s_linker13_clean.py. "
        "Inspect with: git status -s "
        "src/llm_sad_sam/linkers/experimental/prompts_v2.py "
        "src/llm_sad_sam/linkers/experimental/s_linker13_clean.py ..."
    )


def test_instantiation_checkpoint_backend_smoke():
    """Smoke test: the variant instantiates with backend=CHECKPOINT (no live
    LLM). Catches import-time errors, __init__ wiring bugs."""
    from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import (
        SLinker13Trim1JudgeClean,
    )
    from llm_sad_sam.llm_client import LLMBackend
    linker = SLinker13Trim1JudgeClean(backend=LLMBackend.CHECKPOINT)
    assert linker._VARIANT_NAME == "s_linker13_trim1_judge_clean"
