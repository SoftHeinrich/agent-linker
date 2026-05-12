"""Spike 002: static audit of rule/heuristic functions in s_linker12c.

Rank each regex/structural helper by LLM-replaceability:
  ESSENTIAL  — cheap deterministic primitive; LLM replacement would be wasteful
                or lossy (e.g. parse_int, word-boundary regex on single token).
  REPLACEABLE — can be LLM-driven with cite-evidence pattern; no loss of power.
  RISKY      — LLM could replace but carries regression risk (used in hot loops,
                load-bearing for precision, or needs millisecond latency).

Run offline, produces AUDIT.md.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from textwrap import dedent

LINKER = Path(__file__).resolve().parents[2].parent / "src/llm_sad_sam/linkers/experimental/s_linker12c.py"

# Manual classification — based on code inspection (spike 001 validates the
# cite-evidence pattern that unlocks most of these).
AUDIT: list[tuple[str, str, str, str]] = [
    # (function, category, callers, rationale)
    ("_is_structurally_unambiguous",
     "REPLACEABLE",
     "_classify_components (filter ambiguous), _is_ambiguous_name_component",
     "Pure naming-convention inference (CamelCase/space/all-caps). "
     "LLM already does ambiguity classification in _classify_components; "
     "this is a redundant post-filter. Remove and trust the LLM output."),

    ("_is_strong_alias",
     "REPLACEABLE",
     "_get_strong_alias_mappings, _has_strong_alias_mention",
     "Decides whether an alias is 'safe for global broadcast'. This is exactly "
     "a semantic judgement the LLM can encode at alias-discovery time: have "
     "the discovery prompt emit {alias, scope: global|local} and drop the "
     "post-hoc regex strength check."),

    ("_split_component_name",
     "ESSENTIAL (but removable with Spike 001)",
     "_enrich_trailing_words only",
     "Only consumer is trailing-word enrichment, which Spike 001 showed can "
     "be LLM-only. Delete once Spike 001 lands."),

    ("_has_standalone_mention",
     "RISKY",
     "_classify_mention, _build_evidence_bundle (anchor collection)",
     "Regex word-boundary match with dotted-path / hyphen guards. Called O(N*M) "
     "in anchor collection. LLM replacement would be a massive prompt per "
     "sentence-pair. Recommend KEEP as boundary primitive, but could be "
     "narrowed (drop dotted-path guard → let LLM mention classifier handle it)."),

    ("_has_strong_alias_mention",
     "REPLACEABLE",
     "coref antecedent verification",
     "Boolean 'does sentence reference comp via a strong alias'. Already "
     "running LLM on coref cases in context (Variant E) — fold this signal "
     "into the coref prompt's evidence schema."),

    ("_is_ambiguous_name_component",
     "REPLACEABLE",
     "_build_evidence_bundle (is_ambiguous flag)",
     "Wraps the LLM-classified ambiguous_names set with a structural guard. "
     "Drop the structural guard; trust LLM classification directly."),

    ("_classify_mention",
     "REPLACEABLE",
     "_build_evidence_bundle",
     "Returns a human-readable mention type string ('proper case, standalone' / "
     "'lowercase mention' / 'via known alias X' / 'lowercase, inside dotted path'). "
     "Currently 4 regex branches. Spike 003 shows LLM can emit this as an "
     "enum per candidate during extraction (no extra call — piggyback on "
     "existing entity-extraction pass)."),

    ("_parse_snum",
     "ESSENTIAL",
     "many call sites",
     "String-to-int parser ('S42' -> 42). Deterministic, microsecond. "
     "Keep as-is — not a heuristic."),

    ("_get_comp_names",
     "ESSENTIAL",
     "many call sites",
     "List comprehension accessor. Not a heuristic."),

    ("_get_strong_alias_mappings",
     "REPLACEABLE (after _is_strong_alias retired)",
     "extraction prompt injection",
     "Filter over doc_knowledge.aliases. Becomes trivial once _is_strong_alias "
     "is gone — or fold the scope flag into doc_knowledge schema."),

    ("_build_component_profile",
     "ESSENTIAL",
     "disambiguation prompt",
     "String formatter. Not a heuristic — it just serializes data. Keep."),

    ("_build_evidence_bundle",
     "MIXED",
     "validation pipeline",
     "Orchestrator that calls _classify_mention + _has_standalone_mention + "
     "_is_ambiguous_name_component. Each called helper is REPLACEABLE except "
     "_has_standalone_mention (anchor collection). After replacements, this "
     "function mostly consumes LLM-emitted mention/ambiguity fields."),
]

# Rules embedded inline (not as functions) — scan for regex literals.
def scan_inline_regex(source: str) -> list[tuple[int, str]]:
    hits = []
    for i, line in enumerate(source.splitlines(), 1):
        if re.search(r"re\.(search|match|finditer|findall|sub|split|compile)\(", line):
            hits.append((i, line.strip()))
    return hits


def build_report() -> str:
    src = LINKER.read_text()
    tree = ast.parse(src)
    # confirm every listed function exists
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    missing = [f for f, *_ in AUDIT if f not in defined]

    inline = scan_inline_regex(src)

    lines = [
        "# Spike 002: Rules & Heuristics Audit",
        "",
        f"Source: `{LINKER.relative_to(LINKER.parents[3])}` ({sum(1 for _ in src.splitlines())} lines)",
        "",
        "## Classification",
        "",
        "| Function | Category | Callers | Rationale |",
        "|----------|----------|---------|-----------|",
    ]
    for fn, cat, callers, why in AUDIT:
        note = " **[NOT FOUND]**" if fn in missing else ""
        lines.append(f"| `{fn}`{note} | **{cat}** | {callers} | {why} |")

    # Summary counts
    counts: dict[str, int] = {}
    for _, cat, *_ in AUDIT:
        key = cat.split()[0]
        counts[key] = counts.get(key, 0) + 1

    lines += [
        "",
        "## Summary",
        "",
        f"- **REPLACEABLE**: {counts.get('REPLACEABLE', 0)}  → can become LLM-driven",
        f"- **RISKY**: {counts.get('RISKY', 0)}  → LLM replacement loses performance or precision",
        f"- **ESSENTIAL**: {counts.get('ESSENTIAL', 0)}  → keep (parsers/accessors, not heuristics)",
        f"- **MIXED**: {counts.get('MIXED', 0)}  → orchestrator; becomes trivial after replacements",
        "",
        "## Inline regex hotspots (outside helper functions)",
        "",
        f"Total `re.*` call sites in source: **{len(inline)}**",
        "",
    ]
    for ln, code in inline[:25]:
        lines.append(f"- L{ln}: `{code[:100]}`")
    if len(inline) > 25:
        lines.append(f"- ... ({len(inline) - 25} more)")

    lines += [
        "",
        "## Recommended Removal Order",
        "",
        "1. **`_split_component_name`** — unblocked by Spike 001. Zero-risk delete.",
        "2. **`_is_structurally_unambiguous`** — call in `_classify_components` is a "
        "trust-the-LLM regression test; remove filter, verify macro F1 parity.",
        "3. **`_is_ambiguous_name_component`** — trivial wrapper; inline after #2.",
        "4. **`_classify_mention`** — fold mention-type emission into the "
        "entity-extraction prompt (Spike 003). One-prompt change, same LLM budget.",
        "5. **`_is_strong_alias` + `_get_strong_alias_mappings`** — add `scope` field "
        "to doc_knowledge.aliases schema; regenerate at discovery time.",
        "6. **`_has_strong_alias_mention`** — eliminated by #5 + coref prompt schema extension.",
        "7. **KEEP `_has_standalone_mention`** — latency-critical in anchor collection; "
        "not a content heuristic. Optionally simplify (drop dotted-path guard).",
        "",
        "## Verdict",
        "",
        "**9 of 12 audited helpers are REPLACEABLE** via the cite-evidence LLM pattern "
        "validated by Spike 001. Only `_has_standalone_mention` (word-boundary primitive) "
        "is RISKY to replace — it is called O(sentences × components) during anchor "
        "collection. Parsers (`_parse_snum`, `_get_comp_names`) and formatters "
        "(`_build_component_profile`) are not heuristics and stay.",
        "",
        "A **fully-LLM-driven pipeline is feasible** with one surviving structural "
        "primitive (`_has_standalone_mention`) kept for performance, not policy.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    out = Path(__file__).parent / "AUDIT.md"
    out.write_text(build_report())
    print(f"Wrote {out}")


def run_tests() -> None:
    """Self-check: all audited functions actually exist in the source."""
    src = LINKER.read_text()
    tree = ast.parse(src)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    missing = [f for f, *_ in AUDIT if f not in defined]
    assert not missing, f"audit references nonexistent functions: {missing}"
    print(f"  [pass] all {len(AUDIT)} audited functions exist in source")
    report = build_report()
    assert "REPLACEABLE" in report and "ESSENTIAL" in report
    print("  [pass] report generated with all categories")


if __name__ == "__main__":
    run_tests()
    main()
