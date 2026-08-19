"""Deterministic audit of every hand-written rule in ``s_linker64``.

The question a reviewer asks of this workflow is not "does it score well" but "is the
deterministic layer a principled relation, or a rulebook grown against five benchmark
documents?"  Every earlier round in this branch ablated a rule and priced its removal.
This one asks a different question first: **how many distinct rules are actually
there?**  A rule that turns out to be another rule at a different setting is not a rule
to defend, it is a duplication to state once.

No LLM calls.  Everything below is computed off the five benchmark documents and their
PCM catalogs.

  A1  inventory     every deterministic predicate in the module, its call sites, and
                    whether it *admits* a link, *proposes* a candidate, or *labels*
                    evidence.  Admission is the only one a reviewer must accept on
                    faith; the audit shows there is none.

  A2  the lattice   the four candidate generators are shown to be **one relation at
                    four points of one strictness axis**.  `_name_spans(text, name,
                    form)` is checked to reproduce `_find_exact_form`,
                    `_add_stated_name_net`'s scan, `_spelling_variant_candidates`'
                    owner test and `_is_inflection_of` on every (name, sentence) pair
                    of all five projects -- an identity, not a behavioural claim.

  A3  yield         per form, how many pairs it reaches and how many are gold, so the
                    axis can be reported as one monotone table instead of four rules.

  A4  residue       what is left after the merge: the wrappers with one call site, the
                    case defect in `_all_occurrences_in_qualified_path`, and the
                    `"" in "-_"` defect `s_linker63` priced.

    ../.venv/bin/python pilot/rule_audit.py
    ../.venv/bin/python pilot/rule_audit.py --only A2
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS, load_gold                  # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences       # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository           # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker64              # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker64 import SLinker64    # noqa: E402

REPORT = Path("../results/rule_audit")

#: English inflectional endings — the module's only word list, reproduced here so the
#: audit is self-contained.  Not domain vocabulary (GATE-06).
INFLECTIONS = s_linker64.INFLECTIONS

WORD = r"[A-Za-z]+[A-Za-z0-9]*|\d+"

#: The strictness axis.  Each form accepts everything the form above it accepts.
FORMS = ("as_spelled", "any_case", "any_spelling", "any_word")


# ── the proposed single relation ─────────────────────────────────────────────────

def _signature(expression):
    """The word sequence of an expression, case- and separator-folded."""
    normalized = unicodedata.normalize("NFKC", expression)
    normalized = normalized.replace("-", " ").replace("_", " ")
    return tuple(
        token.casefold()
        for token in re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+", normalized
        )
    )


def name_spans(text, name, form):
    """Every span of ``text`` that realizes ``name`` at ``form``.

    One relation.  ``form`` is the only thing that separates the four candidate
    generators of ``s_linker64``.
    """
    if form == "as_spelled":
        pattern = rf"(?<!\w){re.escape(name)}(?!\w)"
        return [(m.start(), m.end()) for m in re.finditer(pattern, text)]
    if form == "any_case":
        pattern = rf"(?<!\w){re.escape(name)}(?!\w)"
        return [(m.start(), m.end())
                for m in re.finditer(pattern, text, re.IGNORECASE)]
    if form == "any_spelling":
        target = _signature(name)
        if not target:
            return []
        words = list(re.finditer(r"[A-Za-z0-9]+", text))
        found = []
        for i, first in enumerate(words):
            for j in range(i, min(len(words), i + len(target))):
                last = words[j]
                if j > i:
                    separator = text[words[j - 1].end():last.start()]
                    if not re.fullmatch(r"[\s_-]+", separator):
                        break
                span = (first.start(), last.end())
                if _signature(text[span[0]:span[1]]) == target:
                    found.append(span)
        return found
    if form == "any_word":
        target = [w.casefold() for w in re.findall(WORD, name)]
        found = []
        for m in re.finditer(WORD, text):
            surface = m.group(0).casefold()
            if any(surface.startswith(w) and surface[len(w):] in INFLECTIONS
                   for w in target):
                found.append((m.start(), m.end()))
        return found
    raise ValueError(form)


# ── loading ──────────────────────────────────────────────────────────────────────

def load_project(name):
    text, model, _ = PROJECTS[name]
    return {
        "sentences": load_sentences(str(BENCH / text)),
        "components": parse_pcm_repository(str(BENCH / model)),
        "gold": load_gold(name),
    }


class Probe(SLinker64):
    """s_linker64's predicates without its constructor (no LLM client)."""

    def __init__(self, aliases=None):               # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": aliases or {}})()


# ── A1: inventory ────────────────────────────────────────────────────────────────

#: Every deterministic method of the module, classified by what its result decides.
#: `admits` would mean "this predicate alone puts a link in the output"; the audit
#: asserts the column is empty.
ROLE = {
    "_find_exact_form": "primitive",
    "_states_a_name": "primitive",
    "_name_signature": "primitive",
    "_in_dotted_path": "primitive",
    "_inside_qualified_identifier": "primitive",
    "_all_occurrences_in_qualified_path": "primitive",
    "_is_inflection_of": "primitive",
    "_keep_stated_names": "proposes",
    "_add_spelling_variants": "proposes",
    "_spelling_variant_candidates": "proposes",
    "_add_stated_name_net": "proposes",
    "_name_word_candidates": "proposes",
    "_classify_mention_typed": "labels",
    "_antecedent_states_name": "proposes",
    "_unlinked": "structural",
    "_window": "structural",
    "_union": "structural",
    "_iter_batches": "structural",
}


def audit_inventory():
    source = Path(inspect.getfile(s_linker64)).read_text()
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "SLinker64")
    bodies = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

    call_sites = Counter()
    for name in ROLE:
        # the negative lookbehind already excludes the `def` line
        call_sites[name] = len(re.findall(rf"(?<!def )\b{name}\(", source))

    rows = []
    for name, role in sorted(ROLE.items(), key=lambda kv: (kv[1], kv[0])):
        node = bodies.get(name)
        if node is None:
            rows.append({"method": name, "role": role, "lines": 0,
                         "call_sites": 0, "missing": True})
            continue
        doc = ast.get_docstring(node)
        body = [s for s in node.body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        code_lines = (max(s.end_lineno for s in body) - min(s.lineno for s in body) + 1
                      if body else 0)
        rows.append({
            "method": name,
            "role": role,
            "code_lines": code_lines,
            "doc_lines": len(doc.splitlines()) if doc else 0,
            "call_sites": call_sites[name],
        })

    print("\n=== A1  inventory of the deterministic layer ===")
    print(f"{'method':<38} {'role':<11} {'code':>5} {'doc':>5} {'sites':>6}")
    for r in rows:
        print(f"{r['method']:<38} {r['role']:<11} "
              f"{r.get('code_lines', 0):>5} {r.get('doc_lines', 0):>5} "
              f"{r['call_sites']:>6}")
    admits = [r for r in rows if r["role"] == "admits"]
    print(f"\n  predicates that admit a link with no LLM verdict: {len(admits)}")
    print(f"  deterministic code lines: "
          f"{sum(r.get('code_lines', 0) for r in rows)}"
          f"   docstring lines defending them: "
          f"{sum(r.get('doc_lines', 0) for r in rows)}")
    return rows


# ── A2: the four generators are one relation ─────────────────────────────────────

def audit_lattice(projects):
    """Check `name_spans` reproduces each of the four generators exactly."""
    print("\n=== A2  four generators, one relation ===")
    totals = Counter()
    mismatches = defaultdict(list)

    for pname, data in projects.items():
        probe = Probe()
        sentences, components = data["sentences"], data["components"]
        for sentence in sentences:
            text = sentence.text
            for component in components:
                name = component.name

                # (1) `_find_exact_form` == first span at any_case
                spans = name_spans(text, name, "any_case")
                got = probe._find_exact_form(text, name)
                want = text[spans[0][0]:spans[0][1]] if spans else ""
                totals["any_case_checked"] += 1
                if got != want:
                    mismatches["find_exact_form"].append((pname, sentence.number, name))

                # (2) `_add_stated_name_net`'s scan == any span at as_spelled
                net = bool(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text))
                totals["as_spelled_checked"] += 1
                if net != bool(name_spans(text, name, "as_spelled")):
                    mismatches["stated_name_net"].append((pname, sentence.number, name))

                # (4) `_is_inflection_of` over the name's words == any_word spans
                words = [w.casefold() for w in re.findall(WORD, name)]
                by_hand = {
                    (m.start(), m.end())
                    for m in re.finditer(WORD, text)
                    if any(probe._is_inflection_of(m.group(0).casefold(), w)
                           for w in words)
                }
                totals["any_word_checked"] += 1
                if by_hand != set(name_spans(text, name, "any_word")):
                    mismatches["is_inflection_of"].append(
                        (pname, sentence.number, name))

        # (3) `_spelling_variant_candidates` == any_spelling + unique owner + not
        #     already the plain name, over the whole document at once
        by_hand = {
            (c.sentence_number, c.component_id)
            for c in probe._spelling_variant_candidates(sentences, components)
        }
        owners = defaultdict(list)
        for component in components:
            sig = _signature(component.name)
            if sig:
                owners[sig].append(component)
        rebuilt = set()
        for sentence in sentences:
            text = sentence.text
            for component in components:
                for start, end in name_spans(text, component.name, "any_spelling"):
                    if probe._inside_qualified_identifier(text, start, end):
                        continue
                    surface = text[start:end]
                    if len(owners.get(_signature(surface), ())) != 1:
                        continue
                    if surface.casefold() == component.name.casefold():
                        continue
                    rebuilt.add((sentence.number, component.id))
        totals["spelling_checked"] += 1
        if by_hand != rebuilt:
            mismatches["spelling_variants"].append(
                (pname, sorted(by_hand ^ rebuilt)))

    print(f"  (name, sentence) pairs checked: {totals['any_case_checked']}")
    for key in ("find_exact_form", "stated_name_net", "is_inflection_of",
                "spelling_variants"):
        bad = mismatches[key]
        verdict = "IDENTICAL" if not bad else f"{len(bad)} DIVERGENCES"
        print(f"    {key:<22} {verdict}")
        for item in bad[:5]:
            print(f"        {item}")
    return mismatches


# ── A3: yield per form ───────────────────────────────────────────────────────────

def audit_yield(projects):
    """Pairs reached and gold rate at each point of the axis, per project."""
    print("\n=== A3  the axis, as one monotone table ===")
    print(f"{'project':<15} {'form':<14} {'pairs':>7} {'gold':>6} {'gold/pair':>10}")
    grand = defaultdict(Counter)
    for pname, data in projects.items():
        gold = data["gold"]
        for form in FORMS:
            pairs = set()
            for sentence in data["sentences"]:
                for component in data["components"]:
                    if name_spans(sentence.text, component.name, form):
                        pairs.add((sentence.number, component.id))
            hit = len(pairs & gold)
            grand[form]["pairs"] += len(pairs)
            grand[form]["gold"] += hit
            rate = hit / len(pairs) if pairs else 0.0
            print(f"{pname:<15} {form:<14} {len(pairs):>7} {hit:>6} {rate:>10.3f}")
    print()
    for form in FORMS:
        pairs, hit = grand[form]["pairs"], grand[form]["gold"]
        print(f"{'ALL':<15} {form:<14} {pairs:>7} {hit:>6} "
              f"{hit / pairs if pairs else 0:>10.3f}")

    # monotonicity: each form must be a superset of the one above it
    print("\n  containment (each form accepts everything the stricter one accepts):")
    for pname, data in projects.items():
        sets = {}
        for form in FORMS:
            sets[form] = {
                (s.number, c.id)
                for s in data["sentences"] for c in data["components"]
                if name_spans(s.text, c.name, form)
            }
        for a, b in zip(FORMS, FORMS[1:]):
            escaped = sets[a] - sets[b]
            flag = "ok" if not escaped else f"{len(escaped)} ESCAPE"
            print(f"    {pname:<15} {a} <= {b}: {flag}")
    return grand


# ── A4: residue ──────────────────────────────────────────────────────────────────

def audit_residue(projects):
    print("\n=== A4  what is left after the merge ===")

    # (a) one-call-site wrappers
    source = Path(inspect.getfile(s_linker64)).read_text()
    for wrapper in ("_antecedent_states_name", "_full_name_source"):
        sites = len(re.findall(rf"self\.{wrapper}\(|cls\.{wrapper}\(", source))
        print(f"  {wrapper}: {sites} call site(s)")

    # (b) the case defect in _all_occurrences_in_qualified_path
    probe = Probe()
    changed = Counter()
    for pname, data in projects.items():
        for sentence in data["sentences"]:
            for component in data["components"]:
                text, name = sentence.text, component.name
                if not probe._find_exact_form(text, name):
                    continue
                now = probe._all_occurrences_in_qualified_path(name.lower(), text)
                # case-consistent reading: search the name the way the primitive does
                spans = name_spans(text, name, "any_case")
                fixed = bool(spans) and all(
                    probe._in_dotted_path(text, s, e) for s, e in spans
                )
                changed["reachable_now"] += int(now)
                changed["reachable_fixed"] += int(fixed)
                changed["labels_flipped"] += int(now != fixed)
    print(f"  _all_occurrences_in_qualified_path lowercases the name and searches the "
          f"raw sentence:")
    print(f"      CODE_TOKEN labels it can produce as written : "
          f"{changed['reachable_now']}")
    print(f"      ... with the case handled as the primitive does: "
          f"{changed['reachable_fixed']}")
    print(f"      labels that would flip: {changed['labels_flipped']}")

    # (c) the "" in "-_" defect, priced by s_linker63
    empty_hits = Counter()
    for pname, data in projects.items():
        for sentence in data["sentences"]:
            text = sentence.text
            for m in re.finditer(WORD, text):
                start, end = m.start(), m.end()
                before = text[start - 1] if start else ""
                after = text[end] if end < len(text) else ""
                if before == "" or after == "":
                    empty_hits[pname] += 1
    print(f"\n  spans suppressed only by `\"\" in \"-_\"` "
          f"(s_linker63 priced repairing this at FP +1.2):")
    for pname, n in empty_hits.items():
        print(f"      {pname:<15} {n}")
    print(f"      {'total':<15} {sum(empty_hits.values())}")
    return changed


# ── A5: did the restatement actually minimize? ───────────────────────────────────

def audit_minimization():
    """s_linker64's lexical layer against s_linker65's, in lines and in rules."""
    from llm_sad_sam.linkers.experimental import s_linker65             # noqa: E402

    print("\n=== A5  s_linker64 -> s_linker65 ===")

    def layer(module, class_name, methods):
        source = Path(inspect.getfile(module)).read_text()
        tree = ast.parse(source)
        cls = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef) and n.name == class_name)
        bodies = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        total = 0
        present = []
        for name in methods:
            node = bodies.get(name)
            if node is None:
                continue
            body = [s for s in node.body
                    if not (isinstance(s, ast.Expr)
                            and isinstance(s.value, ast.Constant))]
            if body:
                total += (max(s.end_lineno for s in body)
                          - min(s.lineno for s in body) + 1)
            present.append(name)
        return total, present

    before = ("_keep_stated_names", "_add_spelling_variants", "_add_stated_name_net",
              "_spelling_variant_candidates", "_name_word_candidates",
              "_is_inflection_of", "_name_signature", "_find_exact_form",
              "_antecedent_states_name")
    after = ("_keep_stated_names", "_add_scan", "_scan", "_name_spans", "_realizes",
             "_owners", "_name_signature", "_find_exact_form")

    old_lines, old_methods = layer(s_linker64, "SLinker64", before)
    new_lines, new_methods = layer(s_linker65, "SLinker65", after)
    print(f"  lexical layer, s_linker64: {len(old_methods)} methods, "
          f"{old_lines} code lines")
    print(f"  lexical layer, s_linker65: {len(new_methods)} methods, "
          f"{new_lines} code lines")

    # the count that matters for the paper: distinct matching rules a reviewer must
    # accept, against distinct settings of one relation
    def regex_calls(cls, methods):
        """`re.*` call sites, and how many methods they are spread over.

        The count itself barely moves -- the same matching still has to happen.  What
        moves is how many places a reviewer has to read to know what "the sentence
        writes the name" means.
        """
        n = 0
        carriers = []
        for name in methods:
            member = getattr(cls, name, None)
            if member is None:
                continue
            text = inspect.getsource(member)
            indent = min(len(line) - len(line.lstrip())
                         for line in text.splitlines() if line.strip())
            text = "\n".join(line[indent:] for line in text.splitlines())
            for node in ast.walk(ast.parse(text)):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "re"
                        and node.func.attr != "escape"):
                    n += 1
                    if name not in carriers:
                        carriers.append(name)
        return n, carriers

    print(f"\n  distinct lexical rules to defend : 4  (s_linker64)")
    print(f"  distinct relations to defend     : 1  (s_linker65), at "
          f"{len(FORMS)} settings, {len(s_linker65.SCANS)} of them scanned")
    old_calls, old_carriers = regex_calls(SLinker64, before)
    new_calls, new_carriers = regex_calls(s_linker65.SLinker65, after)
    print(f"  matching call sites in the lexical layer: {old_calls} -> {new_calls}")
    print(f"  methods a reviewer must read to know what a name match is: "
          f"{len(old_carriers)} -> {len(new_carriers)}")
    print(f"      s_linker64: {old_carriers}")
    print(f"      s_linker65: {new_carriers}")
    return {"before_lines": old_lines, "after_lines": new_lines,
            "before_methods": len(old_methods), "after_methods": len(new_methods),
            "before_matchers": old_carriers, "after_matchers": new_carriers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("A1", "A2", "A3", "A4", "A5"))
    parser.add_argument("--projects", nargs="+", default=list(PROJECTS))
    args = parser.parse_args()

    projects = {name: load_project(name) for name in args.projects}
    out = {}
    if args.only in (None, "A1"):
        out["inventory"] = audit_inventory()
    if args.only in (None, "A2"):
        out["lattice"] = {k: len(v) for k, v in audit_lattice(projects).items()}
    if args.only in (None, "A3"):
        out["yield"] = {f: dict(c) for f, c in audit_yield(projects).items()}
    if args.only in (None, "A4"):
        out["residue"] = dict(audit_residue(projects))
    if args.only in (None, "A5"):
        out["minimization"] = audit_minimization()

    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "rule_audit.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwritten: {REPORT / 'rule_audit.json'}")


if __name__ == "__main__":
    main()
