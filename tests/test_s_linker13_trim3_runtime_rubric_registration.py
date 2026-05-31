"""Tests for s_linker13_trim3_runtime_rubric_clean — Phase 12 Step 3 (Plan 12-05).

Verifies the trim variant:
1. Imports cleanly and exposes the expected symbols.
2. Has _VARIANT_NAME == "s_linker13_trim3_runtime_rubric_clean".
3. Subclasses SLinker13Clean (per plan key_links pattern).
4. DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 byte-equal to prompts_v2.DOC_KNOWLEDGE_JUDGE_EXAMPLES (V35a guard).
5. RUBRIC_BUILDER_SEED_EXAMPLE: GATE-06 clean (no benchmark terms).
6. RUBRIC_BUILDER_PROMPT: clean + has "4-6" + has 3 placeholders + no biased JSON example content.
7. Registered in CANONICAL_VARIANTS + VARIANT_SPECS with canonical=False.
8. v2.0 frozen files untouched.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_TABOO_PROBE = (
    r"(?i)\b("
    r"Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|"
    r"HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper|"
    r"UserDBAdapter|AudioWatermarking|MediaManagement|WebUI|"
    r"Recommender|Persistence|SlopeOneRecommender|ImageProvider|"
    r"Datastore|JabRef|bibdatabase|bibentry"
    r")\b"
)


def test_import_symbols():
    from llm_sad_sam.linkers.experimental.s_linker13_trim3_runtime_rubric_clean import (
        SLinker13Trim3RuntimeRubricClean,
        RUBRIC_BUILDER_PROMPT,
        RUBRIC_BUILDER_SEED_EXAMPLE,
        DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3,
    )
    assert SLinker13Trim3RuntimeRubricClean is not None
    assert isinstance(RUBRIC_BUILDER_PROMPT, str)
    assert isinstance(RUBRIC_BUILDER_SEED_EXAMPLE, str)
    assert isinstance(DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3, str)


def test_variant_name():
    from llm_sad_sam.linkers.experimental.s_linker13_trim3_runtime_rubric_clean import (
        SLinker13Trim3RuntimeRubricClean,
    )
    assert SLinker13Trim3RuntimeRubricClean._VARIANT_NAME == (
        "s_linker13_trim3_runtime_rubric_clean"
    )


def test_subclass_of_s_linker13_clean():
    from llm_sad_sam.linkers.experimental.s_linker13_clean import SLinker13Clean
    from llm_sad_sam.linkers.experimental.s_linker13_trim3_runtime_rubric_clean import (
        SLinker13Trim3RuntimeRubricClean,
    )
    assert issubclass(SLinker13Trim3RuntimeRubricClean, SLinker13Clean)


def test_examples_byte_equal_to_v2():
    """V35a guard: example removal regresses Claude — keep them verbatim."""
    from llm_sad_sam.linkers.experimental import prompts_v2
    from llm_sad_sam.linkers.experimental.s_linker13_trim3_runtime_rubric_clean import (
        DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3,
    )
    assert DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 == prompts_v2.DOC_KNOWLEDGE_JUDGE_EXAMPLES


def test_seed_example_taboo_clean():
    from llm_sad_sam.linkers.experimental.s_linker13_trim3_runtime_rubric_clean import (
        RUBRIC_BUILDER_SEED_EXAMPLE,
    )
    assert re.search(_TABOO_PROBE, RUBRIC_BUILDER_SEED_EXAMPLE) is None


def test_rubric_builder_prompt_taboo_clean():
    from llm_sad_sam.linkers.experimental.s_linker13_trim3_runtime_rubric_clean import (
        RUBRIC_BUILDER_PROMPT,
    )
    assert re.search(_TABOO_PROBE, RUBRIC_BUILDER_PROMPT) is None


def test_rubric_builder_prompt_has_target_size_and_placeholders():
    from llm_sad_sam.linkers.experimental.s_linker13_trim3_runtime_rubric_clean import (
        RUBRIC_BUILDER_PROMPT,
    )
    assert "4-6" in RUBRIC_BUILDER_PROMPT
    assert "{document_text}" in RUBRIC_BUILDER_PROMPT
    assert "{candidate_mappings}" in RUBRIC_BUILDER_PROMPT
    assert "{seed_example}" in RUBRIC_BUILDER_PROMPT


def test_rubric_builder_prompt_no_biased_json_example_content():
    """Output shape may be defined; example RULE CONTENT must be empty/abstract.

    V35c lesson: concrete output examples bias model output. The rubric builder
    template is allowed to define structure (key name 'rubric', a list) but the
    EXAMPLE LIST ENTRIES must be placeholders, not real rule content.
    """
    from llm_sad_sam.linkers.experimental.s_linker13_trim3_runtime_rubric_clean import (
        RUBRIC_BUILDER_PROMPT,
    )
    # Disallow obvious biased example values — strings that look like actual
    # rubric content (containing "approve"/"reject" verbs) inside the JSON template.
    # We do allow "item 1", "item 2", ... abstract placeholders.
    # Heuristic: between the JSON-template marker {{"rubric": and the closing }},
    # there must be no occurrence of the words "approve" or "reject".
    m = re.search(
        r"\{\{\s*\"rubric\"\s*:\s*\[(.*?)\]\s*\}\}",
        RUBRIC_BUILDER_PROMPT,
        re.DOTALL,
    )
    assert m is not None, "expected a JSON template with a 'rubric' list"
    inside = m.group(1)
    assert "approve" not in inside.lower(), (
        f"biased example content in JSON template: {inside!r}"
    )
    assert "reject" not in inside.lower(), (
        f"biased example content in JSON template: {inside!r}"
    )


def test_registered_in_canonical_variants():
    from run_ablation import CANONICAL_VARIANTS
    assert "s_linker13_trim3_runtime_rubric_clean" in CANONICAL_VARIANTS


def test_variant_specs_canonical_false_and_shape():
    from run_ablation import VARIANT_SPECS
    spec = VARIANT_SPECS["s_linker13_trim3_runtime_rubric_clean"]
    assert spec["canonical"] is False
    assert spec["class_name"] == "SLinker13Trim3RuntimeRubricClean"
    assert spec["module"] == (
        "llm_sad_sam.linkers.experimental.s_linker13_trim3_runtime_rubric_clean"
    )


def test_v20_frozen_files_unchanged():
    """git diff --quiet on the frozen surface must exit 0."""
    files = [
        "src/llm_sad_sam/linkers/experimental/prompts_v2.py",
        "src/llm_sad_sam/linkers/experimental/s_linker13.py",
        "src/llm_sad_sam/linkers/experimental/s_linker13_clean.py",
        "src/llm_sad_sam/core/data_types_v2.py",
        "src/llm_sad_sam/core/document_loader_v2.py",
        "src/llm_sad_sam/pcm_parser_v2.py",
    ]
    proc = subprocess.run(
        ["git", "diff", "--quiet", "--"] + files,
        cwd=_PROJECT_ROOT,
    )
    assert proc.returncode == 0, (
        f"git diff dirty on frozen files: {files}"
    )
