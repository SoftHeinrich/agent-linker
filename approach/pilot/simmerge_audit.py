"""Can one string-similarity proposer replace the full-name and partial-name scans?

No LLM calls. Every number is computed off the five benchmark documents, their PCM
catalogs, the recorded alias tables of the head's own runs, and SWATTR's published
link files (`../sota-links/model-doc/swattr-*.csv`).

**The question.** `s_linker110` proposes twice in code: `_extract_named_mentions`
(the sentence writes a whole name of the component, ANY_CASE) and `_scan`
(the sentence writes one word of the name at any WordNet reading, ANY_WORD, minus
the pairs another whole name covers). Each scan has its own judge, and the design law
behind the split is "the looser the form a linker scans, the stricter the judge behind
it". SWATTR (Keim et al., the ArDoCo lexical line) proposes *once*, with a fuzzy string
relation: same token count, then case-insensitive equality OR levenshtein distance
within a word-dependent limit OR Jaro-Winkler >= 0.90, compared against the component's
name and its camel-case parts.

So: is there a single similarity relation that (a) reaches what both head scans reach,
(b) reaches gold neither reaches, and (c) is small enough that one judge can be asked
to refuse on it? A merged proposer is only interesting if the union it produces is not
much larger than what it replaces, because every extra pair is a case a judge must
reject and the branch's whole precision story is that nothing in the deterministic
layer admits a link.

  M1  the relation      SWATTR's word-similarity predicate, reimplemented from
                        `SimilarityUtils`/`WordSimUtils`/`{Levenshtein,JaroWinkler}Measure`
                        at ArDoCo's shipped config, with self-checks.
  M2  the yield table   every arm's pairs / gold / gold-per-pair, per project and
                        total -- the name-relation table of `CLAUDE.md` extended with
                        the fuzzy rows.
  M3  containment       does the similarity scan cover the head's union? what does it
                        add, what does it lose, and how much gold is in each.
  M4  the form label    the merged candidate set cross-tabulated by the *code-computed*
                        form of the match (exact whole / fuzzy whole / lemma word /
                        fuzzy part only), with gold rate per cell -- the fact a single
                        judge would be graded by.
  M5  cost              judge cases and calls at `JUDGE_BATCH`, merged against today's
                        two stages; plus SWATTR's own published links as the
                        lower bound on what its real pipeline proposes.
  M6  the two judges    each stage's judge scored on its own stream over six recorded
                        runs, and both scans checked against the ones those runs wrote.
  M7  marginal gold     every gold pair the similarity scan adds, checked against the
                        recorded final link sets — what the workflow already produces
                        cannot be bought twice.

Usage, from the approach/ directory:
    ../.venv/bin/python pilot/simmerge_audit.py
    ../.venv/bin/python pilot/simmerge_audit.py --only M2 M3
    ../.venv/bin/python pilot/simmerge_audit.py --no-alias
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS, load_gold                  # noqa: E402
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge         # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences       # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository           # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker110 import (           # noqa: E402
    SLinker110, WORD_PATTERN,
)

REPORT = Path("../results/simmerge_audit")
SWATTR = Path("../sota-links/model-doc")

#: The alias tables the head's own promoted runs recorded, in order of preference.
ALIAS_RUNS = [
    "../results/consolidation_e2e_terra_r1_20260825",
    "../results/consolidation_e2e_terra_r2_20260825",
    "../results/consolidation_e2e_terra_r3_20260825",
]

# ─────────────────────────────────────────────────────────────────────────────
# M1 — SWATTR's relation, reimplemented
#
# ArDoCo, core/framework/common, at the shipped CommonTextToolsConfig.properties:
#   levenshtein_Enabled=true  MinLength=2  MaxDistance=1  Threshold=0.90
#   jaroWinkler_Enabled=true  SimilarityThreshold=0.90
#   getMostRecommendedIByRef_MinProportion=0.5
# WordSimUtils.areWordsSimilar = splitLengthTest AND (AT_LEAST_ONE of
#   EqualityMeasure | LevenshteinMeasure | JaroWinklerMeasure).
# ─────────────────────────────────────────────────────────────────────────────

LEV_MIN_LENGTH = 2
LEV_MAX_DISTANCE = 1
LEV_THRESHOLD = 0.90
JW_THRESHOLD = 0.90


def levenshtein(first: str, second: str) -> int:
    if first == second:
        return 0
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def jaro(first: str, second: str) -> float:
    """Jaro similarity, commons-text semantics (case-sensitive, as ArDoCo calls it)."""
    if first == second:
        return 1.0
    if not first or not second:
        return 0.0
    window = max(len(first), len(second)) // 2 - 1
    if window < 0:
        window = 0
    first_flags = [False] * len(first)
    second_flags = [False] * len(second)
    matches = 0
    for i, ch in enumerate(first):
        low = max(0, i - window)
        high = min(i + window + 1, len(second))
        for j in range(low, high):
            if not second_flags[j] and second[j] == ch:
                first_flags[i] = second_flags[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i, flagged in enumerate(first_flags):
        if not flagged:
            continue
        while not second_flags[k]:
            k += 1
        if first[i] != second[k]:
            transpositions += 1
        k += 1
    transpositions //= 2
    m = float(matches)
    return (m / len(first) + m / len(second) + (m - transpositions) / m) / 3.0


def jaro_winkler(first: str, second: str) -> float:
    """Jaro-Winkler, prefix scale 0.1, prefix capped at 4, boost only above 0.7."""
    j = jaro(first, second)
    if j < 0.7:
        return j
    prefix = 0
    for a, b in zip(first[:4], second[:4]):
        if a != b:
            break
        prefix += 1
    return j + 0.1 * prefix * (1.0 - j)


def levenshtein_similar(first: str, second: str) -> bool:
    """`LevenshteinMeasure.areWordsSimilar`, lowercased as that class does."""
    first, second = first.lower(), second.lower()
    dynamic = min(LEV_MAX_DISTANCE, int(LEV_THRESHOLD * min(len(first), len(second))))
    distance = levenshtein(first, second)
    if len(first) <= LEV_MIN_LENGTH:
        return distance <= LEV_MAX_DISTANCE and (first in second or second in first)
    return distance <= dynamic


def words_similar(first: str, second: str) -> bool:
    """`WordSimUtils.areWordsSimilar` at AT_LEAST_ONE over the three shipped measures."""
    if len(first.lower().split(" ")) != len(second.lower().split(" ")):
        return False                                    # splitLengthTest
    if first.lower() == second.lower():                 # EqualityMeasure
        return True
    if levenshtein_similar(first, second):              # LevenshteinMeasure
        return True
    return jaro_winkler(first, second) >= JW_THRESHOLD  # JaroWinklerMeasure


def words_of_lists_similar(originals, tested, min_proportion=JW_THRESHOLD) -> bool:
    """`SimilarityUtils.areWordsOfListsSimilar`. Default proportion is ArDoCo's."""
    if words_similar(" ".join(originals), " ".join(tested)):
        return True
    largest = max(len(originals), len(tested))
    if largest == 0:
        return False
    similar = sum(1 for o in originals for t in tested if words_similar(o, t))
    return similar / largest >= min_proportion


def split_cases(name: str) -> str:
    """`CommonUtilities.splitCases` — snake/kebab, then camel.

    Java's camel split is `(?<!(^|[A-Z]))(?=[A-Z])|(?<!^)(?=[A-Z][a-z])`, a
    variable-width lookbehind Python's `re` refuses; the two conditions are spelled
    out per position instead, which is the same predicate.
    """
    spaced = re.sub(r"\s+", " ", " ".join(name.split("-")).replace("_", " ")).strip()
    out = []
    for i, char in enumerate(spaced):
        if i > 0 and char.isupper():
            previous_not_upper = not spaced[i - 1].isupper()
            next_is_lower = i + 1 < len(spaced) and spaced[i + 1].islower()
            if previous_not_upper or next_is_lower:
                out.append(" ")
        out.append(char)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def name_parts(name: str) -> list[str]:
    """`ModelEntity.getNameParts` — the split words, plus the identifier if it split."""
    parts = [p for p in split_cases(name).split(" ") if p]
    if len(parts) > 1:
        parts = parts + [name]
    return parts


def self_check():
    """The relation's own invariants, so M2 is read off a checked predicate."""
    checks = [
        ("equality ignores case", words_similar("Facade", "facade"), True),
        ("split length test", words_similar("Media Store", "MediaStore"), False),
        ("levenshtein one edit", words_similar("Facade", "Facadee"), True),
        # Two edits is past levenshtein's limit and Jaro-Winkler admits it anyway:
        # a shared 4-character prefix buys the boost. This is the relation's
        # leniency, recorded here rather than argued about later.
        ("two edits, admitted by jw", words_similar("Cache", "Caches2"), True),
        ("levenshtein alone refuses it", levenshtein_similar("Cache", "Caches2"), False),
        ("different stems refused", words_similar("Cache", "Caching"), False),
        ("prefix leniency: Store ~ Storage", words_similar("Store", "Storage"), True),
        ("short word needs containment", words_similar("DB", "DX"), False),
        ("short word contained", words_similar("DB", "DBs"), True),
        ("jw plural of long name", words_similar("Reencoding", "Reencodings"), True),
        ("unrelated words differ", words_similar("Cache", "server"), False),
        ("jaro winkler value", round(jaro_winkler("MARTHA", "MARHTA"), 4), 0.9611),
        ("jaro value", round(jaro("DWAYNE", "DUANE"), 4), 0.8222),
        ("name parts camel", name_parts("UserDBAdapter"),
         ["User", "DB", "Adapter", "UserDBAdapter"]),
        ("name parts single", name_parts("Facade"), ["Facade"]),
        ("name parts spaced", name_parts("Image Provider"),
         ["Image", "Provider", "Image Provider"]),
    ]
    failures = [(what, got, want) for what, got, want in checks if got != want]
    for what, got, want in failures:
        print(f"  FAIL {what}: {got!r} != {want!r}")
    print(f"  M1: {len(checks) - len(failures)}/{len(checks)} relation checks pass")
    return not failures


# ─────────────────────────────────────────────────────────────────────────────
# Loading — the head's scans, run on a bare instance (no LLM client is built)
# ─────────────────────────────────────────────────────────────────────────────

def head_instance(aliases):
    linker = SLinker110.__new__(SLinker110)
    knowledge = DocumentKnowledge()
    knowledge.aliases = dict(aliases)
    linker.doc_knowledge = knowledge
    return linker


def recorded_aliases(project):
    """The alias table of the first promoted head run that recorded this project."""
    for run in ALIAS_RUNS:
        path = (Path(run) / "phase_states" / "s_linker110" / "openai" / project
                / "knowledge.pkl")
        if path.exists():
            with path.open("rb") as handle:
                state = pickle.load(handle)
            return dict(getattr(state["doc_knowledge"], "aliases", {})), run
    return {}, None


def swattr_links(project):
    path = SWATTR / f"swattr-{project}.csv"
    if not path.exists():
        return None
    out = set()
    with path.open() as handle:
        for row in csv.DictReader(handle):
            out.add((int(row["sentence_id"]), row["target_id"].strip()))
    return out


def tokens(text):
    return re.findall(WORD_PATTERN, text)


def load_project(project, use_alias):
    text, model, _ = PROJECTS[project]
    sentences = load_sentences(str(BENCH / text))
    components = parse_pcm_repository(str(BENCH / model))
    aliases, run = recorded_aliases(project) if use_alias else ({}, None)
    return {
        "name": project,
        "sentences": sentences,
        "components": components,
        "gold": load_gold(project),
        "aliases": aliases,
        "alias_run": run,
        "linker": head_instance(aliases),
        "id_of": {c.name: c.id for c in components},
    }


# ─────────────────────────────────────────────────────────────────────────────
# The arms — every arm is a set of (sentence number, component id) pairs
# ─────────────────────────────────────────────────────────────────────────────

def arm_full(data):
    """`_extract_named_mentions`: the sentence writes a whole name, ANY_CASE."""
    linker = data["linker"]
    found = linker._extract_named_mentions(
        data["sentences"], data["components"], data["id_of"], {})
    return set(found)


def arm_partial_all(data):
    """`_scan_all`: one word of the name, any WordNet reading, whole-name pairs out."""
    linker = data["linker"]
    return {(c.sentence_number, c.component_id)
            for c in linker._scan_all(data["sentences"], data["components"])}


def arm_partial(data):
    """`_scan`: `_scan_all` minus the pairs another component's whole name covers."""
    linker = data["linker"]
    return {(c.sentence_number, c.component_id)
            for c in linker._scan(data["sentences"], data["components"])}


def _names_of(data, component):
    """N(c): the catalog name and the run's discovered aliases for it."""
    extra = [term for term, comp in data["aliases"].items()
             if comp == component.name]
    return [component.name, *extra]


def arm_sim_part(data):
    """ArDoCo `isWordSimilarToEntity`: a sentence word is similar to a name part."""
    out = set()
    for sentence in data["sentences"]:
        words = tokens(sentence.text)
        for component in data["components"]:
            parts = [p for name in _names_of(data, component) for p in name_parts(name)]
            if any(words_similar(part, word) for word in words for part in parts):
                out.add((sentence.number, component.id))
    return out


def arm_sim_whole(data):
    """ArDoCo's phrase-level test: a window of the sentence is similar to the name.

    `areWordsOfListsSimilar(nameParts, window)` over every contiguous window whose
    length is the name's part count, plus the 1-word window against the identifier —
    the two shapes `isNounMappingSimilarToModelInstance` reduces to when the noun
    mapping's reference is a phrase of the sentence.
    """
    out = set()
    for sentence in data["sentences"]:
        words = tokens(sentence.text)
        for component in data["components"]:
            hit = False
            for name in _names_of(data, component):
                parts = [p for p in split_cases(name).split(" ") if p]
                widths = {1, len(parts)}
                for width in widths:
                    for start in range(0, max(0, len(words) - width + 1)):
                        window = words[start:start + width]
                        if words_of_lists_similar(name_parts(name), window):
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                out.add((sentence.number, component.id))
    return out


def all_pairs(data):
    return {(s.number, c.id) for s in data["sentences"] for c in data["components"]}


def build_arms(data):
    full = arm_full(data)
    partial = arm_partial(data)
    partial_all = arm_partial_all(data)
    sim_part = arm_sim_part(data)
    sim_whole = arm_sim_whole(data)
    arms = {
        "full (ANY_CASE whole name)": full,
        "partial (ANY_WORD, refusal on)": partial,
        "partial_all (ANY_WORD, no refusal)": partial_all,
        "HEAD UNION (full + partial)": full | partial,
        "sim_whole (fuzzy whole name)": sim_whole,
        "sim_part (fuzzy name part)": sim_part,
        "SIM UNION (sim_whole + sim_part)": sim_whole | sim_part,
        "ALL PAIRS (sentence x component)": all_pairs(data),
    }
    swattr = swattr_links(data["name"])
    if swattr is not None:
        arms["swattr published links"] = swattr
    return arms


# ─────────────────────────────────────────────────────────────────────────────
# M2 — the yield table
# ─────────────────────────────────────────────────────────────────────────────

def m2(projects, sink):
    rows = []
    order = None
    totals = defaultdict(lambda: [0, 0])
    gold_total = 0
    for data in projects:
        gold = data["gold"]
        gold_total += len(gold)
        arms = data["arms"]
        order = list(arms)
        sink(f"\n  {data['name']} — {len(data['sentences'])} sentences, "
             f"{len(data['components'])} components, {len(gold)} gold links, "
             f"{len(data['aliases'])} recorded aliases")
        sink(f"    {'arm':38s} {'pairs':>7s} {'gold':>6s} {'gold/pair':>10s} "
             f"{'recall':>8s}")
        for arm, pairs in arms.items():
            hit = len(pairs & gold)
            totals[arm][0] += len(pairs)
            totals[arm][1] += hit
            sink(f"    {arm:38s} {len(pairs):7d} {hit:6d} "
                 f"{(hit / len(pairs) if pairs else 0):10.3f} "
                 f"{(hit / len(gold) if gold else 0):8.3f}")
            rows.append({"project": data["name"], "arm": arm, "pairs": len(pairs),
                         "gold": hit,
                         "gold_per_pair": round(hit / len(pairs), 4) if pairs else 0,
                         "recall": round(hit / len(gold), 4) if gold else 0})
    sink(f"\n  ALL FIVE PROJECTS — {gold_total} gold links")
    sink(f"    {'arm':38s} {'pairs':>7s} {'gold':>6s} {'gold/pair':>10s} "
         f"{'recall':>8s}")
    for arm in order:
        pairs, hit = totals[arm]
        sink(f"    {arm:38s} {pairs:7d} {hit:6d} "
             f"{(hit / pairs if pairs else 0):10.3f} {hit / gold_total:8.3f}")
        rows.append({"project": "TOTAL", "arm": arm, "pairs": pairs, "gold": hit,
                     "gold_per_pair": round(hit / pairs, 4) if pairs else 0,
                     "recall": round(hit / gold_total, 4)})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# M3 — containment
# ─────────────────────────────────────────────────────────────────────────────

def m3(projects, sink):
    rows = []
    agg = Counter()
    examples = defaultdict(list)
    for data in projects:
        gold = data["gold"]
        head = data["arms"]["HEAD UNION (full + partial)"]
        sim = data["arms"]["SIM UNION (sim_whole + sim_part)"]
        added, lost, shared = sim - head, head - sim, head & sim
        sink(f"\n  {data['name']}")
        sink(f"    head union {len(head):4d} ({len(head & gold):3d} gold)   "
             f"sim union {len(sim):4d} ({len(sim & gold):3d} gold)")
        sink(f"    sim adds   {len(added):4d} ({len(added & gold):3d} gold)   "
             f"sim loses {len(lost):4d} ({len(lost & gold):3d} gold)   "
             f"shared {len(shared):4d} ({len(shared & gold):3d} gold)")
        unreached = gold - head - sim
        sink(f"    gold no scan reaches (coreference territory): {len(unreached)}")
        agg.update({"head": len(head), "head_gold": len(head & gold),
                    "sim": len(sim), "sim_gold": len(sim & gold),
                    "added": len(added), "added_gold": len(added & gold),
                    "lost": len(lost), "lost_gold": len(lost & gold),
                    "unreached": len(unreached), "gold": len(gold)})
        rows.append({"project": data["name"], "head": len(head),
                     "head_gold": len(head & gold), "sim": len(sim),
                     "sim_gold": len(sim & gold), "sim_adds": len(added),
                     "sim_adds_gold": len(added & gold), "sim_loses": len(lost),
                     "sim_loses_gold": len(lost & gold),
                     "gold_unreached_by_either": len(unreached)})
        name_of = {c.id: c.name for c in data["components"]}
        text_of = {s.number: s.text for s in data["sentences"]}
        for pair in sorted(lost):
            examples["lost"].append(
                f"{data['name']} S{pair[0]} {name_of[pair[1]]}"
                f"{' [GOLD]' if pair in gold else ''}: {text_of[pair[0]][:90]}")
        for pair in sorted(added & gold):
            examples["added_gold"].append(
                f"{data['name']} S{pair[0]} {name_of[pair[1]]}: "
                f"{text_of[pair[0]][:90]}")
    sink("\n  ALL FIVE PROJECTS")
    sink(f"    head union {agg['head']} ({agg['head_gold']} gold)  "
         f"sim union {agg['sim']} ({agg['sim_gold']} gold)  of {agg['gold']} gold")
    sink(f"    sim adds {agg['added']} pairs carrying {agg['added_gold']} gold; "
         f"sim loses {agg['lost']} pairs carrying {agg['lost_gold']} gold")
    sink(f"    gold neither scan reaches: {agg['unreached']}")
    if examples["lost"]:
        sink("\n    every pair the head scans reach and the similarity scan does not:")
        for line in examples["lost"][:40]:
            sink(f"      {line}")
    if examples["added_gold"]:
        sink("\n    gold the similarity scan adds:")
        for line in examples["added_gold"][:40]:
            sink(f"      {line}")
    rows.append({"project": "TOTAL", "head": agg["head"], "head_gold": agg["head_gold"],
                 "sim": agg["sim"], "sim_gold": agg["sim_gold"],
                 "sim_adds": agg["added"], "sim_adds_gold": agg["added_gold"],
                 "sim_loses": agg["lost"], "sim_loses_gold": agg["lost_gold"],
                 "gold_unreached_by_either": agg["unreached"]})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# M4 — the form label a single judge would be graded by
# ─────────────────────────────────────────────────────────────────────────────

FORMS = ("exact whole name", "fuzzy whole name", "lemma word of name",
         "fuzzy name part only")


def form_of(pair, data):
    arms = data["arms"]
    if pair in arms["full (ANY_CASE whole name)"]:
        return FORMS[0]
    if pair in arms["sim_whole (fuzzy whole name)"]:
        return FORMS[1]
    if pair in arms["partial (ANY_WORD, refusal on)"]:
        return FORMS[2]
    return FORMS[3]


def m4(projects, sink):
    rows = []
    table = defaultdict(lambda: [0, 0])
    per_project = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for data in projects:
        gold = data["gold"]
        merged = (data["arms"]["HEAD UNION (full + partial)"]
                  | data["arms"]["SIM UNION (sim_whole + sim_part)"])
        for pair in merged:
            form = form_of(pair, data)
            table[form][0] += 1
            per_project[data["name"]][form][0] += 1
            if pair in gold:
                table[form][1] += 1
                per_project[data["name"]][form][1] += 1
    sink("\n  the merged candidate set, split by the form the code can compute")
    sink(f"    {'form':24s} {'pairs':>7s} {'gold':>6s} {'gold/pair':>10s}")
    for form in FORMS:
        pairs, hit = table[form]
        sink(f"    {form:24s} {pairs:7d} {hit:6d} "
             f"{(hit / pairs if pairs else 0):10.3f}")
        rows.append({"project": "TOTAL", "form": form, "pairs": pairs, "gold": hit,
                     "gold_per_pair": round(hit / pairs, 4) if pairs else 0})
    sink("")
    for project, forms in per_project.items():
        cells = "  ".join(
            f"{form.split()[0]} {forms[form][0]}/{forms[form][1]}g" for form in FORMS)
        sink(f"    {project:15s} {cells}")
        for form in FORMS:
            pairs, hit = forms[form]
            rows.append({"project": project, "form": form, "pairs": pairs,
                         "gold": hit,
                         "gold_per_pair": round(hit / pairs, 4) if pairs else 0})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# M5 — cost, and SWATTR's own output as a floor under its candidate set
# ─────────────────────────────────────────────────────────────────────────────

def m5(projects, sink):
    batch = SLinker110.JUDGE_BATCH
    rows = []
    totals = Counter()
    sink(f"\n  judge cases and calls at JUDGE_BATCH = {batch}")
    sink(f"    {'project':15s} {'full':>6s} {'part':>6s} {'2-stage':>8s} "
         f"{'calls':>6s} | {'merged head':>12s} {'calls':>6s} | "
         f"{'merged sim':>11s} {'calls':>6s}")
    for data in projects:
        full = data["arms"]["full (ANY_CASE whole name)"]
        part = data["arms"]["partial (ANY_WORD, refusal on)"]
        head = data["arms"]["HEAD UNION (full + partial)"]
        sim = head | data["arms"]["SIM UNION (sim_whole + sim_part)"]
        calls2 = -(-len(full) // batch) + -(-len(part) // batch)
        calls_head = -(-len(head) // batch)
        calls_sim = -(-len(sim) // batch)
        sink(f"    {data['name']:15s} {len(full):6d} {len(part):6d} "
             f"{len(full) + len(part):8d} {calls2:6d} | {len(head):12d} "
             f"{calls_head:6d} | {len(sim):11d} {calls_sim:6d}")
        totals.update({"full": len(full), "part": len(part),
                       "two": len(full) + len(part), "calls2": calls2,
                       "head": len(head), "calls_head": calls_head,
                       "sim": len(sim), "calls_sim": calls_sim})
        rows.append({"project": data["name"], "full_cases": len(full),
                     "partial_cases": len(part),
                     "two_stage_cases": len(full) + len(part),
                     "two_stage_calls": calls2, "merged_head_cases": len(head),
                     "merged_head_calls": calls_head,
                     "merged_sim_cases": len(sim), "merged_sim_calls": calls_sim})
    sink(f"    {'TOTAL':15s} {totals['full']:6d} {totals['part']:6d} "
         f"{totals['two']:8d} {totals['calls2']:6d} | {totals['head']:12d} "
         f"{totals['calls_head']:6d} | {totals['sim']:11d} {totals['calls_sim']:6d}")
    rows.append({"project": "TOTAL", "full_cases": totals["full"],
                 "partial_cases": totals["part"], "two_stage_cases": totals["two"],
                 "two_stage_calls": totals["calls2"],
                 "merged_head_cases": totals["head"],
                 "merged_head_calls": totals["calls_head"],
                 "merged_sim_cases": totals["sim"],
                 "merged_sim_calls": totals["calls_sim"]})

    sink("\n  SWATTR's published links against the two candidate sets "
         "(its output is a floor under whatever it proposed)")
    sink(f"    {'project':15s} {'links':>6s} {'gold':>5s} {'in head':>8s} "
         f"{'in sim':>7s} {'in neither':>11s}")
    floor = Counter()
    misses = []
    for data in projects:
        published = data["arms"].get("swattr published links")
        if published is None:
            continue
        head = data["arms"]["HEAD UNION (full + partial)"]
        sim = data["arms"]["SIM UNION (sim_whole + sim_part)"]
        gold = data["gold"]
        outside = published - head - sim
        sink(f"    {data['name']:15s} {len(published):6d} {len(published & gold):5d} "
             f"{len(published & head):8d} {len(published & sim):7d} "
             f"{len(outside):11d}")
        floor.update({"links": len(published), "gold": len(published & gold),
                      "head": len(published & head), "sim": len(published & sim),
                      "outside": len(outside)})
        name_of = {c.id: c.name for c in data["components"]}
        for pair in sorted(outside & gold):
            misses.append(f"{data['name']} S{pair[0]} "
                          f"{name_of.get(pair[1], pair[1])} [GOLD]")
    sink(f"    {'TOTAL':15s} {floor['links']:6d} {floor['gold']:5d} "
         f"{floor['head']:8d} {floor['sim']:7d} {floor['outside']:11d}")
    if misses:
        sink("\n    gold SWATTR links that neither scan proposes:")
        for line in misses:
            sink(f"      {line}")
    rows.append({"project": "SWATTR", "full_cases": floor["links"],
                 "partial_cases": floor["gold"], "two_stage_cases": floor["head"],
                 "two_stage_calls": floor["sim"], "merged_head_cases": floor["outside"],
                 "merged_head_calls": 0, "merged_sim_cases": 0, "merged_sim_calls": 0})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# M6 — what the two judges actually do with their two streams
#
# Off the recorded runs, not recomputed: the scan each run wrote is compared to the
# scan this audit computes (so M2-M5 are checked against the head's own bytes), and
# each stage's judge is scored on its own stream.
# ─────────────────────────────────────────────────────────────────────────────

E2E_RUNS = {
    "terra": ["../results/consolidation_e2e_terra_r1_20260825",
              "../results/consolidation_e2e_terra_r2_20260825",
              "../results/consolidation_e2e_terra_r3_20260825"],
    "luna": ["../results/consolidation_e2e_luna_r1_20260825",
             "../results/consolidation_e2e_luna_r2_20260825",
             "../results/consolidation_e2e_luna_r3_20260825"],
}
STAGES = {"full_name": "linker_full_name", "partial_name": "linker_partial_name"}


def _recorded_pairs(view, id_of):
    out = set()
    for row in view:
        cid = id_of.get(row.get("component"))
        if cid is not None:
            out.add((int(row["sentence"]), cid))
    return out


def m6(projects, sink):
    by_name = {d["name"]: d for d in projects}
    rows = []
    agreement = Counter()
    totals = defaultdict(Counter)
    for model, runs in E2E_RUNS.items():
        for run in runs:
            for project in PROJECTS:
                data = by_name[project]
                gold = data["gold"]
                for stage, phase in STAGES.items():
                    path = (Path(run) / "phase_states" / "s_linker110" / "openai"
                            / project / f"{phase}.pkl")
                    if not path.exists():
                        continue
                    with path.open("rb") as handle:
                        state = pickle.load(handle)
                    feedback = state["feedback"]
                    proposed = _recorded_pairs(
                        feedback.get("candidates") or feedback.get("proposed", []),
                        data["id_of"])
                    accepted = _recorded_pairs(feedback.get("accepted", []),
                                               data["id_of"])
                    # The scan is recomputed with *this run's* alias table, not the
                    # audit's fixed one: N(c) is the catalog plus the run's aliases,
                    # and those vary ~2.8 terms a run.
                    run_state = (Path(run) / "phase_states" / "s_linker110" / "openai"
                                 / project / "knowledge.pkl")
                    with run_state.open("rb") as handle:
                        aliases = dict(getattr(
                            pickle.load(handle)["doc_knowledge"], "aliases", {}))
                    scoped = dict(data, linker=head_instance(aliases),
                                  aliases=aliases)
                    computed = (arm_full(scoped) if stage == "full_name"
                                else arm_partial(scoped))
                    agreement[f"{stage}:match" if proposed == computed
                              else f"{stage}:differ"] += 1
                    bucket = totals[(model, stage)]
                    bucket["runs"] += 1
                    bucket["cases"] += len(proposed)
                    bucket["case_gold"] += len(proposed & gold)
                    bucket["kept"] += len(accepted)
                    bucket["tp"] += len(accepted & gold)
                    bucket["fp"] += len(accepted - gold)
    sink("\n  the audit's scans against the scans the recorded runs wrote")
    for key in sorted(agreement):
        sink(f"    {key:22s} {agreement[key]}")
    sink("\n  each judge on its own stream, per project-run "
         "(3 runs x 5 projects a model)")
    sink(f"    {'model':6s} {'stage':13s} {'cases':>7s} {'gold in':>8s} "
         f"{'kept':>6s} {'TP':>6s} {'FP':>6s} {'keep rate':>10s} "
         f"{'gold kept':>10s} {'precision':>10s}")
    for (model, stage), bucket in sorted(totals.items()):
        runs = max(1, bucket["runs"] / len(PROJECTS))
        sink(f"    {model:6s} {stage:13s} {bucket['cases'] / runs:7.1f} "
             f"{bucket['case_gold'] / runs:8.1f} {bucket['kept'] / runs:6.1f} "
             f"{bucket['tp'] / runs:6.1f} {bucket['fp'] / runs:6.1f} "
             f"{bucket['kept'] / max(1, bucket['cases']):10.3f} "
             f"{bucket['tp'] / max(1, bucket['case_gold']):10.3f} "
             f"{bucket['tp'] / max(1, bucket['kept']):10.3f}")
        rows.append({"model": model, "stage": stage,
                     "cases_per_run": round(bucket["cases"] / runs, 1),
                     "gold_in_cases_per_run": round(bucket["case_gold"] / runs, 1),
                     "kept_per_run": round(bucket["kept"] / runs, 1),
                     "tp_per_run": round(bucket["tp"] / runs, 1),
                     "fp_per_run": round(bucket["fp"] / runs, 1),
                     "keep_rate": round(bucket["kept"] / max(1, bucket["cases"]), 4),
                     "gold_kept": round(bucket["tp"] / max(1, bucket["case_gold"]), 4),
                     "precision": round(bucket["tp"] / max(1, bucket["kept"]), 4)})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# M7 — the marginal gold, priced against the pipeline that already exists
#
# A pair the similarity scan adds is only worth its judge case if the workflow does
# not already produce it. The coreference linker reaches what no scan reaches, so
# every gold pair in `sim - head` is checked against the recorded final link sets.
# ─────────────────────────────────────────────────────────────────────────────

def _recorded_finals(project):
    """{run label: {(sentence, component id): source}} over both models' runs."""
    out = {}
    for model, runs in E2E_RUNS.items():
        for run in runs:
            path = (Path(run) / "phase_states" / "s_linker110" / "openai" / project
                    / "final.pkl")
            if not path.exists():
                continue
            with path.open("rb") as handle:
                links = pickle.load(handle)["final"]
            out[f"{model}:{Path(run).name[-13:]}"] = {
                (link.sentence_number, link.component_id): link.source
                for link in links
            }
    return out


def m7(projects, sink):
    rows = []
    sink("\n  every gold pair the similarity scan adds, against six recorded runs")
    sink(f"    {'pair':46s} {'runs found':>11s}  sources")
    found_all = Counter()
    for data in projects:
        head = data["arms"]["HEAD UNION (full + partial)"]
        sim = data["arms"]["SIM UNION (sim_whole + sim_part)"]
        extra_gold = sorted((sim - head) & data["gold"])
        missed_gold = sorted(data["gold"] - head - sim)
        if not extra_gold and not missed_gold:
            continue
        finals = _recorded_finals(data["name"])
        name_of = {c.id: c.name for c in data["components"]}
        for label, pairs in (("sim adds", extra_gold),
                             ("no scan reaches", missed_gold)):
            for pair in pairs:
                hits = [source for run in finals
                        for source in [finals[run].get(pair)] if source]
                found_all[f"{label}:{len(hits)}/{len(finals)}"] += 1
                tag = (f"{data['name']} S{pair[0]} {name_of[pair[1]]} [{label}]")
                sink(f"    {tag:46s} {len(hits):5d}/{len(finals):<5d}  "
                     f"{', '.join(sorted(set(hits))) or '-'}")
                rows.append({"project": data["name"], "sentence": pair[0],
                             "component": name_of[pair[1]], "class": label,
                             "runs_found": len(hits), "runs": len(finals),
                             "sources": "|".join(sorted(set(hits)))})
    sink("")
    for key in sorted(found_all):
        sink(f"    {key:24s} {found_all[key]} pairs")
    return rows


# ─────────────────────────────────────────────────────────────────────────────

def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*",
                        default=["M1", "M2", "M3", "M4", "M5", "M6", "M7"])
    parser.add_argument("--no-alias", action="store_true",
                        help="catalog names only; do not read recorded alias tables")
    args = parser.parse_args()

    lines = []

    def sink(line=""):
        print(line)
        lines.append(line)

    sink("SIMMERGE AUDIT — one similarity proposer against the head's two scans")
    sink(f"  alias source: {'none (catalog only)' if args.no_alias else ALIAS_RUNS[0]}")

    if "M1" in args.only:
        sink("\nM1 — SWATTR's relation, reimplemented")
        self_check()

    projects = []
    for project in PROJECTS:
        data = load_project(project, not args.no_alias)
        data["arms"] = build_arms(data)
        projects.append(data)

    REPORT.mkdir(parents=True, exist_ok=True)
    if "M2" in args.only:
        sink("\nM2 — the yield table")
        write_csv(REPORT / "m2_yield.csv", m2(projects, sink))
    if "M3" in args.only:
        sink("\nM3 — containment")
        write_csv(REPORT / "m3_containment.csv", m3(projects, sink))
    if "M4" in args.only:
        sink("\nM4 — the form label")
        write_csv(REPORT / "m4_forms.csv", m4(projects, sink))
    if "M5" in args.only:
        sink("\nM5 — cost")
        write_csv(REPORT / "m5_cost.csv", m5(projects, sink))
    if "M6" in args.only:
        sink("\nM6 — the two judges on their two streams")
        write_csv(REPORT / "m6_judges.csv", m6(projects, sink))
    if "M7" in args.only:
        sink("\nM7 — the marginal gold, against the runs that already exist")
        write_csv(REPORT / "m7_marginal.csv", m7(projects, sink))

    (REPORT / "audit.txt").write_text("\n".join(lines) + "\n")
    summary = {p["name"]: {arm: len(pairs) for arm, pairs in p["arms"].items()}
               for p in projects}
    (REPORT / "arm_sizes.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwritten: {REPORT}/audit.txt and the CSVs beside it")


if __name__ == "__main__":
    main()
