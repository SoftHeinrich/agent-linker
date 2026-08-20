"""Pricing the deletion of the deterministic layer's only word list.

``s_linker83`` and every variant back to ``s_linker62`` decide whether a sentence word
counts as a word of a component's name with ``INFLECTIONS``, nine English endings
stripped off the *sentence token*.  ``s_linker85`` replaces that with WordNet
lemmatization over noun and verb readings, applied to *both* sides.  This pilot prices
that swap before any LLM sees it.  No LLM calls: every number below is computed off the
five benchmark documents and their PCM catalogs.

s85 also carries s83's coreference judge, which nothing here touches: the morphology is
read at one place, ``_name_spans`` at ``ANY_WORD``, and only the partial-name scan reads
it, so comparing the two modules' scans isolates the swap.

  E1  identity     the shipped ``_name_spans`` of both modules, run over every (name,
                   sentence) pair of all five projects, and their ``_scan`` outputs
                   compared as candidate sets against the gold standard.  The claim
                   s85's docstring makes is this table.

  E2  alternatives the same comparison for two rules s85 did *not* take: WordNet
                   without both-sided lemmatization, and spaCy `en_core_web_sm`
                   lemmatizing the sentence in POS context.  E2 is why the shipped rule
                   is symmetric and context-free.  Needs spacy + en_core_web_sm; skipped
                   with a printed note when they are absent.

  E3  endings      which of the nine ``INFLECTIONS`` entries actually fire on these
                   documents.  Reported so the round can say what it declined to do:
                   pruning the dead ones would fit the list to the benchmark (GATE-07).

    ../.venv/bin/python pilot/lemma_swap_pilot.py
    ../.venv/bin/python pilot/lemma_swap_pilot.py --only E1
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS, load_gold                   # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences        # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository            # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker83, s_linker85   # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker83 import SLinker83     # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker85 import SLinker85, lemmas  # noqa: E402


def projects():
    """(name, sentences, components, gold) for each benchmark project."""
    for name in PROJECTS:
        text, model, _ = PROJECTS[name]
        yield (name,
               load_sentences(str(BENCH / text)),
               parse_pcm_repository(str(BENCH / model)),
               load_gold(name))


def candidates(linker_cls, sentences, components):
    """The partial-name candidate set of one module, through its own `_scan`."""
    scan = linker_cls.__new__(linker_cls)._scan(sentences, components)
    return {(c.sentence_number, c.component_id): c.matched_text for c in scan}


def e1_identity():
    """s85's scan against s83's, span by span and candidate by candidate."""
    print("E1  s_linker85 (WordNet, both sides) against s_linker83 (INFLECTIONS)\n")
    pairs = spandiff = 0
    tot = Counter()
    examples = []
    for name, sentences, components, gold in projects():
        for sentence in sentences:
            for component in components:
                pairs += 1
                old = SLinker83._name_spans(sentence.text, component.name,
                                            s_linker83.NameForm.ANY_WORD)
                new = SLinker85._name_spans(sentence.text, component.name,
                                            s_linker85.NameForm.ANY_WORD)
                if old == new:
                    continue
                spandiff += 1
                if len(examples) < 12:
                    examples.append(
                        f"{name} S{sentence.number} {component.name!r}: "
                        f"s83={[sentence.text[a:b] for a, b in old]} "
                        f"s85={[sentence.text[a:b] for a, b in new]}")

        old_set = set(candidates(SLinker83, sentences, components))
        new_set = set(candidates(SLinker85, sentences, components))
        tot["s83"] += len(old_set)
        tot["s85"] += len(new_set)
        tot["gold83"] += len(old_set & gold)
        tot["gold85"] += len(new_set & gold)
        tot["lost"] += len(old_set - new_set)
        tot["lost_gold"] += len((old_set - new_set) & gold)
        tot["new"] += len(new_set - old_set)
        tot["new_gold"] += len((new_set - old_set) & gold)
        print(f"  {name:14s} candidates {len(old_set):4d} -> {len(new_set):4d}"
              f"   gold {len(old_set & gold):3d} -> {len(new_set & gold):3d}")

    print(f"\n  (name, sentence) pairs compared : {pairs}")
    print(f"  pairs whose spans differ        : {spandiff}")
    print(f"  candidates  {tot['s83']} -> {tot['s85']}"
          f"   of which gold  {tot['gold83']} -> {tot['gold85']}")
    print(f"  s85 loses {tot['lost']} ({tot['lost_gold']} gold), "
          f"adds {tot['new']} ({tot['new_gold']} gold)")
    print("\n  every disagreement:")
    for row in examples:
        print(f"    {row}")


def _states_a_name(text, name):
    """`_scan`'s skip, reproduced so the alternatives of E2 can reuse it."""
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE) is not None


def _scan_with(sentences, components, hit):
    """`_scan`'s shape under an arbitrary (sentence, token index, name word) test."""
    out = set()
    for sentence in sentences:
        toks = [m.group(0) for m in re.finditer(s_linker85.WORD_PATTERN, sentence.text)]
        for component in components:
            if _states_a_name(sentence.text, component.name):
                continue
            words = [w.casefold()
                     for w in re.findall(s_linker85.WORD_PATTERN, component.name)]
            if any(hit(sentence, i, tok.casefold(), words)
                   for i, tok in enumerate(toks)):
                out.add((sentence.number, component.id))
    return out


def e2_alternatives():
    """Two rules s85 declined, priced the same way."""
    print("E2  rules not taken\n")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        nlp = None
        print(f"  spaCy arm skipped: {exc}\n")

    def one_sided(sentence, i, tok, words):
        """WordNet on the sentence token only -- the asymmetric reading."""
        return any(w in lemmas(tok) for w in words)

    tot = Counter()
    for name, sentences, components, gold in projects():
        base = set(candidates(SLinker85, sentences, components))
        arms = {"one-sided": _scan_with(sentences, components, one_sided)}

        if nlp is not None:
            docs = {s.number: d for s, d in zip(sentences, nlp.pipe(
                [s.text for s in sentences]))}

            def in_context(sentence, i, tok, words, docs=docs):
                """spaCy's POS-disambiguated lemma for the token in its sentence."""
                spans = [(m.start(), m.end()) for m in
                         re.finditer(s_linker85.WORD_PATTERN, sentence.text)]
                a, b = spans[i]
                for t in docs[sentence.number]:
                    if t.idx < b and t.idx + len(t.text) > a:
                        return any(t.lemma_.casefold() in lemmas(w) for w in words)
                return False

            arms["spacy-ctx"] = _scan_with(sentences, components, in_context)

        for tag, arm in arms.items():
            tot[f"{tag}/n"] += len(arm)
            tot[f"{tag}/lost"] += len(base - arm)
            tot[f"{tag}/lost_gold"] += len((base - arm) & gold)
            tot[f"{tag}/new"] += len(arm - base)
            tot[f"{tag}/new_gold"] += len((arm - base) & gold)

    print("  against s85, pooled over the five projects:")
    for tag in ("one-sided", "spacy-ctx"):
        if f"{tag}/n" not in tot:
            continue
        print(f"    {tag:10s} candidates {tot[f'{tag}/n']:4d}   "
              f"loses {tot[f'{tag}/lost']} ({tot[f'{tag}/lost_gold']} gold), "
              f"adds {tot[f'{tag}/new']} ({tot[f'{tag}/new_gold']} gold)")


def e3_endings():
    """Which INFLECTIONS entries fire at all, over the population s83 actually scans."""
    print("E3  which of the nine endings fire\n")
    fired = Counter()
    for _, sentences, components, _ in projects():
        for sentence in sentences:
            if not sentence.text:
                continue
            for component in components:
                if _states_a_name(sentence.text, component.name):
                    continue
                words = [w.casefold() for w in
                         re.findall(s_linker83.WORD_PATTERN, component.name)]
                for m in re.finditer(s_linker83.WORD_PATTERN, sentence.text):
                    tok = m.group(0).casefold()
                    for word in words:
                        if (tok.startswith(word)
                                and tok[len(word):] in s_linker83.INFLECTIONS):
                            fired[tok[len(word):] or '""'] += 1
                            break

    for ending in s_linker83.INFLECTIONS:
        key = ending or '""'
        print(f"    {key:8s} {fired.get(key, 0)}")
    dead = [e or '""' for e in s_linker83.INFLECTIONS if not fired.get(e or '""')]
    print(f"\n  never fires on these documents: {', '.join(dead)}")
    print("  not pruned: dropping them would fit the list to the benchmark (GATE-07).")


AUDITS = {"E1": e1_identity, "E2": e2_alternatives, "E3": e3_endings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(AUDITS), action="append")
    args = parser.parse_args()
    for key in (args.only or sorted(AUDITS)):
        AUDITS[key]()
        print()


if __name__ == "__main__":
    main()
