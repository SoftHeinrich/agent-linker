"""What the module's authored prompt text actually says, and how general it reads.

No LLM calls, no checkpoints. The compaction round proved authored rules are 5.3%
of a judging call and 4.3% of a resolver call, so this round is not about bytes: it
asks whether each static clause states a *concept* -- something a paper can defend
as general -- or a recipe fitted to the surfaces these five documents happen to
show. A recipe is a leakage-shaped liability even when GATE-06 passes, because it
names no benchmark word yet still encodes what the benchmark looked like.

Two deterministic screens:

  specificity  clauses that ENUMERATE (a dash- or paren-list of instances) or that
               legislate ORTHOGRAPHY/SYNTAX (spacing, hyphenation, capitalization,
               dotted paths) rather than state what a thing IS.
  duplication  clause pairs across different constants with high content-word
               overlap -- the same principle authored twice for two stages, which
               a merge would state once.

    ../.venv/bin/python pilot/static_audit.py
"""
from __future__ import annotations

import itertools
import re
import sys

sys.path.insert(0, "src")

from llm_sad_sam.linkers.experimental import s_linker89 as L        # noqa: E402

CONSTANTS = [
    "DOC_KNOWLEDGE_JUDGE_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
    "ALIAS_EXCLUSION_RULES", "ENTITY_EXTRACTION_RULES",
    "COREF_VALIDATION_FOCUS", "COREF_RULES",
    "LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES",
    "QUALIFIED_CLAUSE", "STRICTER_CLAUSE",
]

#: Which prompt family carries each constant, and how many calls that family makes
#: per five-project run (read off the compaction round's rebuilt inventory).
FAMILY = {
    "DOC_KNOWLEDGE_JUDGE_RULES": ("alias judge", 5.0),
    "DOC_KNOWLEDGE_EXTRACTION_RULES": ("alias proposal", 5.0),
    "ALIAS_EXCLUSION_RULES": ("alias proposal", 5.0),
    "ENTITY_EXTRACTION_RULES": ("full-name extraction", 9.0),
    "COREF_VALIDATION_FOCUS": ("strict judging", 3.6),
    "COREF_RULES": ("resolver", 40.0),
    "LAYERED_ENTITY_RULES": ("lenient judging", 8.7),
    "LAYERED_COREF_RULES": ("strict judging", 3.6),
    "QUALIFIED_CLAUSE": ("lenient judging + extraction + denotation", 22.7),
    "STRICTER_CLAUSE": ("lenient judging", 8.7),
}

#: A clause legislates surface rather than concept when it speaks about how
#: characters are written. These are properties of orthography, not of the domain.
SURFACE = re.compile(
    r"\b(spac(?:ing|ed)|hyphen\w*|compound\w*|capitaliz\w*|spell\w*|dotted|"
    r"joined|upper\s?case|lower\s?case|character|abbreviat\w*)\b", re.I)

#: An enumeration offers instances where a concept would name the class. Either an
#: em-dash list of three or more noun phrases, or a parenthetical list with "or".
ENUM_DASH = re.compile(r"--\s*([^-]{10,200}?)\s*--")
ENUM_PAREN = re.compile(r"\(([^)]{20,200}\bor\b[^)]*)\)")

STOP = set("a an the of to in on for that this it is are be as and or not with by "
           "when only its it's than what which where who whom whose from at into "
           "about under over do does did no nor but if then so such any all each "
           "one two more most some other another same still yet also however "
           "prefer avoid write count report reject approve resolve check state "
           "sentence sentences component components document term terms name "
           "names expression phrase word words".split())


def clauses(text):
    """Split on sentence ends, keeping the parenthetical/dash structure intact."""
    parts = re.split(r"(?<=[.:;])\s+(?=[A-Z(])", " ".join(text.split()))
    return [p.strip() for p in parts if p.strip()]


def content(text):
    return {w for w in re.findall(r"[a-z]+", text.lower())
            if w not in STOP and len(w) > 2}


def main():
    print("=" * 78)
    print("AUTHORED STATIC TEXT IN s_linker89")
    print("=" * 78)
    total = 0
    all_clauses = []
    for name in CONSTANTS:
        text = getattr(L, name)
        total += len(text)
        fam, calls = FAMILY[name]
        print(f"\n{name}  [{len(text)} B]  {fam}, {calls:.1f} calls/run "
              f"= {len(text) * calls:,.0f} B/run")
        for i, c in enumerate(clauses(text), 1):
            flags = []
            if SURFACE.search(c):
                flags.append("SURFACE:" + ",".join(
                    sorted({m.group(0).lower() for m in SURFACE.finditer(c)})))
            for m in itertools.chain(ENUM_DASH.finditer(c), ENUM_PAREN.finditer(c)):
                items = [x.strip() for x in re.split(r",| or ", m.group(1))
                         if x.strip()]
                if len(items) >= 3:
                    flags.append(f"ENUM({len(items)}):{m.group(1)[:56]}")
            mark = "  <-- " + " | ".join(flags) if flags else ""
            print(f"   {i}. {c[:96]}{'...' if len(c) > 96 else ''}{mark}")
            all_clauses.append((name, i, c))
    print(f"\nTOTAL AUTHORED: {total} B")

    print("\n" + "=" * 78)
    print("CLAUSE PAIRS ACROSS CONSTANTS, BY CONTENT-WORD OVERLAP (Jaccard)")
    print("=" * 78)
    scored = []
    for (n1, i1, c1), (n2, i2, c2) in itertools.combinations(all_clauses, 2):
        if n1 == n2:
            continue
        w1, w2 = content(c1), content(c2)
        if not w1 or not w2:
            continue
        j = len(w1 & w2) / len(w1 | w2)
        if j >= 0.12:
            scored.append((j, n1, i1, c1, n2, i2, c2, sorted(w1 & w2)))
    for j, n1, i1, c1, n2, i2, c2, shared in sorted(scored, reverse=True)[:10]:
        print(f"\n  J={j:.2f}  {n1}#{i1}  <->  {n2}#{i2}")
        print(f"    A: {c1[:110]}")
        print(f"    B: {c2[:110]}")
        print(f"    shared: {', '.join(shared)}")


if __name__ == "__main__":
    main()
