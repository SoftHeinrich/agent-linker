"""helper_v3 — extracted helpers for v2.1 _clean variant chain (Plan 10-02).

CLEAN-02 prerequisite: this module hoists the pure-function helpers currently
inlined inside s_linker13.py / s_linker13d.py into a versioned sibling so that
s_linker13_clean (Plan 10-03) can import them instead of re-inlining them. The
extracted bodies are byte-identical copies of the originals — no semantic
changes; the only mechanical edits permitted are dropping the leading
underscore, dropping the unused `self` receiver on static-method-style
helpers, and (for build_component_profile) lifting `self.model_knowledge` /
`self.doc_knowledge` to explicit parameters.

Frozen-file contract: v2.0 helpers (data_types_v2, document_loader_v2,
pcm_parser_v2) and s_linker13.py / s_linker13d.py / prompts_v2.py are NOT
modified by this extraction. The originals continue to inline these helpers;
the extraction only adds a new module so the v2.1 _clean variant chain has a
shared import surface.

Public exports (extracted verbatim — see column "Origin"):

    ┌─────────────────────────────┬──────────────────────────────────┬────────────────────────────────────────────────────────┐
    │ Name                        │ Origin                           │ Purpose (one line)                                     │
    ├─────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────────────────────┤
    │ MENTION_TYPES               │ s_linker13d.py:95                │ Frozen enum set of accepted LLM mention_type values.   │
    │ coerce_mention_type         │ s_linker13d.py:97-110            │ STRICT D-21a coercion; raises on unknown emission.     │
    │ format_mention_string       │ s_linker13d.py:112-133           │ D-21c byte-identical strings for each enum branch.     │
    │ build_component_profile     │ s_linker13.py:594-615            │ Textual profile for disambiguation prompt; `self`      │
    │                             │                                  │ lifted to explicit (model_knowledge, doc_knowledge).   │
    │ parse_snum                  │ s_linker13.py:1107-1117          │ Parse 'S42'/'s42'/'42'/42 → int (None on failure).     │
    │ has_standalone_mention      │ s_linker13.py:1119-1147          │ Spike 002 RISKY anchor primitive; EXT-01/02 deferred.  │
    │ get_comp_names              │ s_linker13.py:1149-1152          │ [c.name for c in components] convenience.              │
    └─────────────────────────────┴──────────────────────────────────┴────────────────────────────────────────────────────────┘

Notes on signature changes (none are semantic):

  * build_component_profile: the original method reads `self.model_knowledge`
    and `self.doc_knowledge`. The extraction lifts both to explicit parameters
    so the helper can stand alone:

        build_component_profile(comp_name, model_knowledge, doc_knowledge)

    Both parameters accept the None case the original guarded for with truthy
    checks, so the body is otherwise unchanged.

  * All other helpers were already @staticmethod or trivially convertible:
    dropped `self`, kept the body verbatim.

Compliance:

  * GATE-06 (no benchmark leakage): zero benchmark component names appear in
    bodies, docstrings, or examples.
  * D-21a (strict coercion): coerce_mention_type retains the raises-on-unknown
    contract — NEVER silently falls back to "indirect".
  * D-21c (byte-identical strings): format_mention_string outputs match the
    EXPECTED table in tests/test_s_linker13d_parity.py exactly. Plan 10-03
    will clone that parity test pointed at this module.
  * Spike 002 (RISKY KEEP): has_standalone_mention stays regex-backed here;
    EXT-01 (LLM replacement) and EXT-02 (drop dotted-path guard) remain
    deferred per PROJECT.md Key Decisions.
"""

from __future__ import annotations

import re


# ─────────────────────────────────────────────────────────────────────────────
# Mention-type enum (D-20 / D-21 / Spike 003)
# ─────────────────────────────────────────────────────────────────────────────

# Mirrors SLinker13d.MENTION_TYPES verbatim.
MENTION_TYPES = frozenset({"proper_case", "lowercase", "dotted_path", "via_alias", "indirect"})


def coerce_mention_type(value):
    """Strictly coerce a raw LLM emission to a MENTION_TYPES enum value.

    Per D-21a: an unknown value raises ValueError immediately so a prompt-
    conformance regression surfaces as a load-bearing test failure rather
    than as variance noise. NEVER silently fall back to 'indirect'.
    """
    if value not in MENTION_TYPES:
        raise ValueError(
            f"Unknown mention_type emitted by LLM: {value!r} "
            f"(expected one of {sorted(MENTION_TYPES)})"
        )
    return value


def format_mention_string(mention_type, alias_used=None):
    """Format an enum value into the byte-identical string the regex method emitted.

    D-21c: 5 enum branches mapped to the exact strings _classify_mention used.
    The via_alias-without-alias branch ('via known alias' no quotes) is preserved
    verbatim from Spike 003 spike.py:48-51 - the LLM is asked to populate
    alias_used whenever it emits 'via_alias', so this branch is a backstop.
    Assumes mention_type was already coerced via coerce_mention_type.
    """
    if mention_type == "proper_case":
        return "proper case, standalone"
    if mention_type == "lowercase":
        return "lowercase mention"
    if mention_type == "dotted_path":
        return "lowercase, inside dotted path"
    if mention_type == "via_alias":
        if alias_used:
            return f'via known alias "{alias_used}"'
        return "via known alias"
    # mention_type == "indirect"
    return "indirect/unclear match"


# ─────────────────────────────────────────────────────────────────────────────
# Component-profile builder
# ─────────────────────────────────────────────────────────────────────────────


def build_component_profile(comp_name: str, model_knowledge, doc_knowledge) -> str:
    """Build textual component profile for disambiguation prompt.

    Signature change vs. s_linker13.SLinker13._build_component_profile: the
    original reads `self.model_knowledge` and `self.doc_knowledge`; both are
    lifted to explicit parameters here. Body is otherwise byte-identical.
    Either argument may be None — the original guarded both with truthy checks.
    """
    lines = [f"- Name: {comp_name}"]

    is_ambig = (model_knowledge
                and comp_name in model_knowledge.ambiguous_names)
    if is_ambig:
        lines.append(f'- Classification: AMBIGUOUS — "{comp_name}" is a common English word')
    else:
        lines.append("- Classification: DISTINCTIVE — architecturally specific name")

    aliases = []
    if doc_knowledge:
        for a, entry in doc_knowledge.aliases.items():
            if entry.component == comp_name:
                aliases.append(f'"{a}"')

    if aliases:
        lines.append(f"- Known aliases: {', '.join(aliases)}")
    else:
        lines.append("- Known aliases: none")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Shared misc helpers
# ─────────────────────────────────────────────────────────────────────────────


def parse_snum(val):
    """Parse sentence number from LLM output (handles 'S42', 's42', '42', 42)."""
    if val is None:
        return None
    if isinstance(val, str):
        val = val.lstrip("Ss")
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def has_standalone_mention(comp_name, text):
    """Check for non-generic, clean standalone mention of component name."""
    if not comp_name:
        return False
    is_single = ' ' not in comp_name
    if is_single:
        if comp_name[0].islower():
            pattern = rf'\b{re.escape(comp_name)}\b'
        else:
            cap_name = comp_name[0].upper() + comp_name[1:]
            pattern = rf'\b{re.escape(cap_name)}\b'
        flags = 0
    else:
        pattern = rf'\b{re.escape(comp_name)}\b'
        flags = re.IGNORECASE

    for m in re.finditer(pattern, text, flags):
        s, e = m.start(), m.end()
        if s > 0 and text[s-1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
            continue
        if s > 0 and text[s-1] == '-':
            continue
        if e < len(text) and text[e] == '-' and '-' not in comp_name:
            continue
        return True
    return False


def get_comp_names(components) -> list[str]:
    """Get all component names."""
    return [c.name for c in components]
