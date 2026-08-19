"""Deterministic screen of three proposer repairs for the partial-name linker.

`pilot/partial_audit.py` shows the denotation judge is not the bottleneck (95% recall,
83% precision, headroom +1.0 TP / -3.5 FP). `pilot/partial_gap.py` attributes the 22.8
open-gold pairs it never sees; `pilot/partial_hole.py` refutes the obvious repair to the
whole-name exclusion (+0.7 gold, +10.0 spurious). What is left are two defects in the
proposer's own two boundary predicates, both deterministic and both measurable here
without a single LLM call:

  guard   `_inside_qualified_identifier` computes `before = text[start - 1] if start
          else ""` and then tests `before in "-_"`. **`"" in "-_"` is `True` in
          Python**, so every span that starts at character 0 of a sentence -- and every
          span that ends at the last character of one -- is reported as sitting inside a
          qualified identifier and is dropped. Sentence-initial component names are
          therefore invisible to this proposer. The repair is to test the two
          characters only when they exist.

  exact   owner uniqueness is decided by a bare prefix test in both directions, so a
          sentence word owns a component whenever it *starts with* one of the
          component's name words. `WebRTC` matches `WebRTC-SFU` exactly and `BBB web`
          by prefix, so the pair is dropped as ambiguously owned. The repair is to let
          an exact word match outrank a prefix-only one.

Reported per project: candidate count, gold among candidates, and every pair each
repair adds or removes, so the recall gain and the precision cost are separable before
any judge is paid for.

    ../.venv/bin/python pilot/partial_screen.py
    ../.venv/bin/python pilot/partial_screen.py --arm s_linker59
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS, load_gold                  # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences       # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository           # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker59 import SLinker59    # noqa: E402

WORD = r"[A-Za-z]+[A-Za-z0-9]*|\d+"
VARIANTS = ("base", "guard", "exact", "infl", "inflguard")
#: English inflectional endings. The proposer's docstring already claims this much --
#: "a sentence word that begins with a name word is accepted, so inflected forms pass
#: without a suffix list" -- so naming the endings states the intent instead of
#: approximating it with an unbounded prefix. Not domain vocabulary: no benchmark term
#: appears here, only the suffixes of English number and tense.
INFLECTIONS = ("", "s", "es", "ed", "d", "ing", "ings", "er", "ers")


class Probe(SLinker59):
    def __init__(self, aliases):                    # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": aliases})()

    # ── the two repaired predicates ──────────────────────────────────────────
    @classmethod
    def _inside_guarded(cls, text, start, end):
        """`_inside_qualified_identifier` with the empty-string disjuncts removed."""
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        joined = (before in ("-", "_") or (before and before.isalnum())
                  or after in ("-", "_") or (after and after.isalnum()))
        return cls._in_dotted_path(text, start, end) or joined

    @staticmethod
    def _inflects(surface, word):
        """Is `surface` `word` under an English inflectional ending?"""
        return (surface.startswith(word)
                and surface[len(word):] in INFLECTIONS)

    def candidates(self, sentences, components, variant):
        guard = variant in ("guard", "inflguard")
        exact_wins = variant == "exact"
        inflect = variant in ("infl", "inflguard")
        inside = self._inside_guarded if guard else self._inside_qualified_identifier
        words = {c.id: {w.casefold() for w in re.findall(WORD, c.name)}
                 for c in components}
        out = {}
        for sentence in sentences:
            for match in re.finditer(WORD, sentence.text):
                if inside(sentence.text, match.start(), match.end()):
                    continue
                surface = match.group(0).casefold()
                test = self._inflects if inflect else (
                    lambda s, w: s.startswith(w))
                prefix = [c for c in components
                          if any(test(surface, w) for w in words[c.id])]
                if exact_wins:
                    owners = [c for c in components
                              if surface in words[c.id]] or prefix
                else:
                    owners = prefix
                if len(owners) != 1:
                    continue
                component = owners[0]
                if self._states_a_name(sentence.text, component.name):
                    continue
                out[(sentence.number, component.id)] = match.group(0)
        return out


_CACHE = {}


def project_cache(project):
    if project not in _CACHE:
        text, model, _ = PROJECTS[project]
        _CACHE[project] = (load_sentences(str(BENCH / text)),
                           parse_pcm_repository(str(BENCH / model)))
    return _CACHE[project]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../results/s5960_e2e_r*_20260813")
    ap.add_argument("--arm", default="s_linker49")
    args = ap.parse_args()
    runs = sorted(Path().glob(args.runs))

    agg = Counter()
    changes = Counter()
    suppressed = Counter()
    for run in runs:
        for project in PROJECTS:
            base_dir = run / "phase_states" / args.arm / "openai" / project
            if not (base_dir / "linker_partial_name.pkl").exists():
                continue
            sentences, components = project_cache(project)
            knowledge = pickle.load((base_dir / "knowledge.pkl").open("rb"))
            probe = Probe(getattr(knowledge.get("doc_knowledge"), "aliases", {}) or {})
            linked = {(l.sentence_number, l.component_id) for l in pickle.load(
                (base_dir / "linker_full_name.pkl").open("rb"))["links"]}
            gold = set(load_gold(project))
            sets = {}
            for variant in VARIANTS:
                keys = {k: v for k, v in
                        probe.candidates(sentences, components, variant).items()
                        if k not in linked}
                sets[variant] = keys
                agg[(project, variant, "n")] += len(keys)
                agg[(project, variant, "gold")] += len(set(keys) & gold)
            agg[(project, "runs", "runs")] += 1
            for variant in VARIANTS[1:]:
                for key in set(sets[variant]) - set(sets["base"]):
                    name = next(c.name for c in components if c.id == key[1])
                    changes[(variant, project, key in gold, key[0], name,
                             sets[variant][key], "+")] += 1
                for key in set(sets["base"]) - set(sets[variant]):
                    name = next(c.name for c in components if c.id == key[1])
                    changes[(variant, project, key in gold, key[0], name,
                             sets["base"][key], "-")] += 1
            # How much of the boundary bug's reach is the empty-string disjunct?
            for sentence in sentences:
                for match in re.finditer(WORD, sentence.text):
                    a = probe._inside_qualified_identifier(
                        sentence.text, match.start(), match.end())
                    b = probe._inside_guarded(sentence.text, match.start(),
                                              match.end())
                    if a and not b:
                        suppressed[project] += 1

    print(f"\n{args.arm}, partial-name candidates per run over {len(runs)} runs\n")
    header = f"{'project':<16}"
    for variant in VARIANTS:
        header += f"{variant:>18}"
    print(header + "\n" + " " * 16 + "        n    gold" * len(VARIANTS))
    totals = Counter()
    for project in PROJECTS:
        n = agg[(project, "runs", "runs")] or 1
        if not agg[(project, "base", "n")] and not agg[(project, "both", "n")]:
            continue
        row = f"{project:<16}"
        for variant in VARIANTS:
            row += (f"{agg[(project, variant, 'n')] / n:>9.1f}"
                    f"{agg[(project, variant, 'gold')] / n:>9.1f}")
            totals[(variant, "n")] += agg[(project, variant, "n")] / n
            totals[(variant, "gold")] += agg[(project, variant, "gold")] / n
        print(row)
    row = f"{'TOTAL':<16}"
    for variant in VARIANTS:
        row += f"{totals[(variant, 'n')]:>9.1f}{totals[(variant, 'gold')]:>9.1f}"
    print(row)
    print(f"\n{'':<16}" + "".join(
        f"{('+%.1f gold, +%.1f spurious' % (totals[(v, 'gold')] - totals[('base', 'gold')], (totals[(v, 'n')] - totals[('base', 'n')]) - (totals[(v, 'gold')] - totals[('base', 'gold')]))):>38}"
        for v in VARIANTS[1:]))

    print("\nspans the empty-string disjunct alone suppresses (all spans, "
          "per run per project)")
    n = len(runs) or 1
    for project in PROJECTS:
        if suppressed[project]:
            print(f"    {project:<16}{suppressed[project] / n:>8.0f}")

    for variant in VARIANTS[1:]:
        rows = [(k, v) for k, v in changes.items() if k[0] == variant]
        if not rows:
            continue
        print(f"\n{variant}: candidates added (+) and removed (-)")
        for key, count in sorted(rows, key=lambda kv: -kv[1]):
            _, project, is_gold, snum, name, text, sign = key
            print(f"    {count}x {sign} {'GOLD' if is_gold else '    '}  "
                  f"{project:<14} s{snum:<5} {name:<18} '{text}'")


if __name__ == "__main__":
    main()
