"""Tests for prompts_v3 (Phase 12 Step 0).

Verifies that prompts_v3.py:
1. Imports cleanly in isolation.
2. Exposes exactly the 9 active constants used by `s_linker13_clean`.
3. Does NOT expose any of the 7 constants dropped from prompts_v2 (WORD_USAGE_PROMPT
   plus 6 STANDALONE_MENTION_RULES_* EXT-01 variants).
4. Each kept constant is byte-equal to its counterpart in prompts_v2 — Step 0 is a
   lossless registration delete, no rephrasing.
5. The module source text contains zero benchmark project-name tokens. The 9-name
   probe below is the narrow, inarguable benchmark-component leakage gate; the
   broader lexical TABOO sweep is Plan 12-06's responsibility.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys


KEPT_NAMES = [
    "AMBIGUITY_FEW_SHOT",
    "AMBIGUITY_RULES",
    "DOC_KNOWLEDGE_EXTRACTION_RULES",
    "DOC_KNOWLEDGE_JUDGE_EXAMPLES",
    "DOC_KNOWLEDGE_JUDGE_RULES",
    "ENTITY_EXTRACTION_RULES",
    "VALIDATION_RULES",
    "COREF_RULES",
    "SEED_DISAMBIGUATION_RULES",
]

DROPPED_NAMES = [
    "WORD_USAGE_PROMPT",
    "STANDALONE_MENTION_RULES_PRE_FILTERED",
    "STANDALONE_MENTION_RULES_LLM_ONLY",
    "STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE",
    "STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE",
    "STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE",
    "STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE",
]

# Narrow benchmark-component probe — the 9 inarguable benchmark-project tokens that
# must never appear in a prompts module. The broader TABOO lexical sweep is Plan 12-06.
TABOO_COMPONENT_REGEX = re.compile(
    r"(?i)\b(Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper)\b"
)


def test_prompts_v3_import_clean():
    """Bare import succeeds in a subprocess (no side effects on import)."""
    result = subprocess.run(
        [sys.executable, "-c", "from llm_sad_sam.linkers.experimental import prompts_v3"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"import failed: {result.stderr}"


def test_kept_constants_present():
    """All 9 kept constants are exposed as module-level non-empty str attributes."""
    from llm_sad_sam.linkers.experimental import prompts_v3

    for name in KEPT_NAMES:
        assert hasattr(prompts_v3, name), f"missing kept constant: {name}"
        val = getattr(prompts_v3, name)
        assert isinstance(val, str), f"{name} is not a str: {type(val)}"
        assert val, f"{name} is empty"


def test_dropped_constants_absent():
    """None of the 7 dropped constants exist on prompts_v3."""
    from llm_sad_sam.linkers.experimental import prompts_v3

    leaked = [n for n in DROPPED_NAMES if hasattr(prompts_v3, n)]
    assert not leaked, f"dropped constants leaked into prompts_v3: {leaked}"


def test_byte_equal_to_v2():
    """Every kept constant in prompts_v3 is byte-equal to prompts_v2 counterpart."""
    from llm_sad_sam.linkers.experimental import prompts_v2, prompts_v3

    mismatched = [
        n for n in KEPT_NAMES if getattr(prompts_v3, n) != getattr(prompts_v2, n)
    ]
    assert not mismatched, (
        f"Step 0 byte-equality violated for: {mismatched} — Step 0 is a lossless "
        "deletion; per-prompt trims must land in Wave 2 variants, not in prompts_v3."
    )


def test_no_benchmark_taboo_terms():
    """The prompts_v3 module source text contains zero benchmark-component tokens.

    This is the narrow 9-token probe — the inarguable benchmark-project leakage
    surface. The full lexical TABOO sweep (with reviewer adjudication of
    compound-context hits like 'config', 'auth', 'logic') happens in Plan 12-06.
    """
    from llm_sad_sam.linkers.experimental import prompts_v3

    src = pathlib.Path(prompts_v3.__file__).read_text()
    matches = TABOO_COMPONENT_REGEX.findall(src)
    assert not matches, (
        f"benchmark-component tokens found in prompts_v3.py: {matches} — GATE-06 narrow probe failed."
    )
