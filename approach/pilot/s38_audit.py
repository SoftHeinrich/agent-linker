"""What is left to simplify in s_linker38, measured before any pilot is paid for.

s38 replaced s25's two judging passes with one prompt sampled twice, verdicts
ANDed, and reached statistical parity on every score. That closes the question of
how the *full-name* judge is arranged and opens five smaller ones, each about a
structure the paper still has to describe. Everything below is read off s38's own
six-run checkpoints and call traces. No LLM call is made.

  A1  SELF-AGREEMENT       how often do the two samples of the one judging prompt
                           disagree, and when they do, is the AND gate winning
                           precision or losing recall? If they never disagree the
                           second sample is decorative and one sample is simpler;
                           if they disagree and the gate is right, the gate is the
                           design.
  A2  CRITERION OVERLAP    inside a single sample, how often does the uniqueness
                           answer differ from the relevance answer? Two questions
                           that always agree are one question, and the prompt can
                           state one.
  A3  PROTOCOL ASYMMETRY   the workflow judges with three different protocols (two
                           samples ANDed; two different questions in sequence; one
                           single pass). How many calls would one protocol cost?
  A4  JUDGING SURFACE      how much prompt text and how many code-driven
                           rejections remain, each with the number of decisions it
                           actually changes in these runs.
  A5  MENTION-TYPE VALUES  the classifier emits five values into one line of one
                           prompt. Do the judge's verdicts vary by value, or do
                           several values behave identically and collapse?

Usage (from the approach/ directory):
    ../.venv/bin/python pilot/s38_audit.py
    ../.venv/bin/python pilot/s38_audit.py --only A1 A4
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_project, load_gold          # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker38 as L38      # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker38 import (            # noqa: E402
    MentionType, SLinker38)

RUNS = [Path(f"../results/s38_twosample_e2e_r{i}_20260812") for i in range(1, 7)]
VARIANT = "s_linker38"
LINKERS = ("full_name", "partial_name", "coreference")


def phase(run, project, name):
    path = run / "phase_states" / VARIANT / "openai" / project / f"{name}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def calls(run, project):
    out = []
    for path in sorted((run / "llm_logs").glob(
            f"{VARIANT}_openai_{project}_*_calls.json")):
        with path.open() as handle:
            out.extend(json.load(handle))
    return out


def response_json(text):
    """The linker's own parse is lenient; mirror just enough of it to read verdicts."""
    if not text:
        return None
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return None


def runs_with(project, name="linker_full_name"):
    return [r for r in RUNS if phase(r, project, name)]


# ── A1 ───────────────────────────────────────────────────────────────────────

def audit_self_agreement():
    """Is the second sample load-bearing, and is the AND gate right when it fires?

    A candidate is admitted only when both samples approve. Three populations
    matter: unanimous approvals (the gate is inert), unanimous rejections (also
    inert), and split verdicts (the gate decides). Only the third is the design;
    scoring it against the gold standard says whether ANDing is the right way to
    break the tie, and prices ORing them instead.
    """
    print("\n### A1 SELF-AGREEMENT — does the second sample decide anything?")
    grand = Counter()
    splits = []
    for project in PROJECTS:
        gold = load_gold(project)
        info = load_project(project)
        id_to_name = {c.id: c.name for c in info["components"]}
        per = Counter()
        present = runs_with(project)
        for run in present:
            for d in phase(run, project, "linker_full_name")["feedback"][
                    "judge_decisions"]:
                key = (d["sentence"], d["component_id"])
                p1, p2 = bool(d["p1"]), bool(d["p2"])
                is_gold = key in gold
                per["cases"] += 1
                if p1 and p2:
                    per["both_yes"] += 1
                    per["both_yes_gold"] += is_gold
                elif not p1 and not p2:
                    per["both_no"] += 1
                    per["both_no_gold"] += is_gold
                else:
                    per["split"] += 1
                    per["split_gold"] += is_gold
                    splits.append((project, d["sentence"],
                                   id_to_name.get(d["component_id"], "?"),
                                   "GOLD" if is_gold else "not gold"))
        n = len(present) or 1
        print(f"  {project:14s} cases {per['cases']/n:5.1f} | both approve "
              f"{per['both_yes']/n:5.1f} ({per['both_yes_gold']/n:5.1f} gold) | "
              f"both reject {per['both_no']/n:4.1f} ({per['both_no_gold']/n:4.1f} "
              f"gold) | SPLIT {per['split']/n:4.1f} ({per['split_gold']/n:4.1f} gold)")
        grand.update(per)
    n = len(RUNS)
    cases, split = grand["cases"] / n, grand["split"] / n
    split_gold = grand["split_gold"] / n
    print(f"  TOTAL          cases {cases:.1f} | both approve "
          f"{grand['both_yes']/n:.1f} | both reject {grand['both_no']/n:.1f} | "
          f"SPLIT {split:.1f}")
    print(f"\n  The gate fires on {split:.1f} of {cases:.1f} cases "
          f"({split / max(cases, 1e-9) * 100:.1f}%).")
    print(f"  Of those, {split_gold:.1f} are gold and {split - split_gold:.1f} are "
          f"not, so ANDing costs {split_gold:.1f} true positives per run to avoid "
          f"{split - split_gold:.1f} false positives;")
    print(f"  ORing the same two samples would read TP +{split_gold:.1f} / "
          f"FP +{split - split_gold:.1f}, and one sample alone lands between them.")
    if splits:
        print("\n  every split case (project, sentence, component, gold?):")
        counted = Counter(splits)
        for (project, snum, comp, tag), times in counted.most_common(20):
            print(f"    {project:14s} S{snum:<4d} {comp:30s} {tag:9s} "
                  f"in {times}/{len(RUNS)} runs")
    return dict(grand)


# ── A2 ───────────────────────────────────────────────────────────────────────

def audit_criterion_overlap():
    """Do relevance and uniqueness ever disagree inside one response?

    s25 measured that dropping the uniqueness *pass* costs 10 false positives, but
    that was a separate call with its own prompt. In s38 both questions live in one
    prompt and one response, so the question is fresh: if the model answers them
    identically on every case, the second question is prompt text with no effect
    and the prompt can state one criterion.
    """
    print("\n### A2 CRITERION OVERLAP — are relevance and uniqueness one question?")
    grand = Counter()
    for project in PROJECTS:
        per = Counter()
        for run in RUNS:
            for call in calls(run, project):
                if call.get("phase") not in ("phase_25_full_name_p1",
                                             "phase_25_full_name_p2"):
                    continue
                data = response_json(call.get("response_text"))
                for item in (data or {}).get("validations", []) or []:
                    def flag(key):
                        value = item.get(key, False)
                        return value is True or (isinstance(value, str)
                                                 and value.lower() == "true")
                    relevant, unique = flag("relevant"), flag("unique")
                    per["answers"] += 1
                    per["agree"] += relevant == unique
                    per["relevant_not_unique"] += relevant and not unique
                    per["unique_not_relevant"] += unique and not relevant
                    per["claim_none"] += str(item.get("claim", "")).strip().lower() \
                        in ("", "none")
        n = len(RUNS)
        print(f"  {project:14s} answers {per['answers']/n:5.1f} | agree "
              f"{per['agree']/n:5.1f} | relevant-not-unique "
              f"{per['relevant_not_unique']/n:4.1f} | unique-not-relevant "
              f"{per['unique_not_relevant']/n:4.1f} | claim=none "
              f"{per['claim_none']/n:4.1f}")
        grand.update(per)
    n = len(RUNS)
    answers = max(grand["answers"] / n, 1e-9)
    disagree = (grand["answers"] - grand["agree"]) / n
    print(f"  TOTAL          answers {answers:.1f} | disagree {disagree:.1f} "
          f"({disagree / answers * 100:.1f}%) | uniqueness rejects what relevance "
          f"keeps: {grand['relevant_not_unique']/n:.1f} per run | the reverse: "
          f"{grand['unique_not_relevant']/n:.1f}")
    return dict(grand)


# ── A3 ───────────────────────────────────────────────────────────────────────

PROTOCOLS = {
    "phase_25_doc_extract": ("knowledge: propose aliases", "single pass"),
    "phase_25_doc_judge": ("knowledge: judge aliases", "single pass"),
    "phase_25_full_name_extract": ("full-name: propose references", "single pass"),
    "phase_25_full_name_p1": ("full-name: judge, sample 1", "two samples, ANDed"),
    "phase_25_full_name_p2": ("full-name: judge, sample 2", "two samples, ANDed"),
    "phase_25_partial_denotation": ("partial-name: denote, target-blind",
                                    "two questions in sequence"),
    "phase_25_partial_identity": ("partial-name: grounded identity",
                                  "two questions in sequence"),
    "phase_25_coreference": ("coreference: resolve", "single pass"),
    "phase_25_coreference_judge": ("coreference: judge", "single pass"),
}


def audit_protocol_asymmetry():
    """Three judging protocols in one workflow. What would one protocol cost?"""
    print("\n### A3 PROTOCOL ASYMMETRY — the price of judging everything alike")
    per_phase = Counter()
    for project in PROJECTS:
        for run in RUNS:
            for call in calls(run, project):
                per_phase[call.get("phase") or "?"] += 1
    n = len(RUNS)
    print(f"  {'phase':34s} {'calls/run':>9s}  protocol")
    total = 0.0
    for tag, count in sorted(per_phase.items(), key=lambda kv: -kv[1]):
        label, protocol = PROTOCOLS.get(tag, (tag, "UNRECOGNISED"))
        print(f"  {label:34s} {count/n:9.1f}  {protocol}")
        total += count / n
    print(f"  {'TOTAL':34s} {total:9.1f}")
    coref = per_phase["phase_25_coreference_judge"] / n
    partial = (per_phase["phase_25_partial_denotation"]
               + per_phase["phase_25_partial_identity"]) / n
    print(f"\n  Coreference judge -> full-name protocol (same prompt, two samples): "
          f"+{coref:.1f} calls/run ({coref / max(total, 1e-9) * 100:.1f}%).")
    print(f"  Both partial-name steps as well: +{partial:.1f} calls/run.")
    print(f"  One protocol everywhere: +{coref + partial:.1f}, total "
          f"{total + coref + partial:.1f} calls/run against s25's 89.")
    return {k: round(v / n, 2) for k, v in per_phase.items()}


# ── A4 ───────────────────────────────────────────────────────────────────────

def audit_judging_surface():
    """Prompt text and code-driven rejections that remain, with their fire counts.

    Two kinds of thing a reviewer has to be told about: a rubric constant (prose
    the paper must justify) and a hand-written rejection (a condition the paper
    must defend as more than a fitted rule). Both are counted here against what
    they actually changed in these six runs.
    """
    print("\n### A4 JUDGING SURFACE — remaining prompt text and coded rejections")
    rubrics = [n for n in dir(L38)
               if n.isupper() and isinstance(getattr(L38, n), str)]
    print("  rubric constants the paper carries:")
    for name in sorted(rubrics, key=lambda n: -len(getattr(L38, n))):
        text = getattr(L38, name)
        print(f"    {name:34s} {len(text):5d} bytes, "
              f"{len(text.split()):4d} words")
    print(f"    {'TOTAL':34s} "
          f"{sum(len(getattr(L38, n)) for n in rubrics):5d} bytes over "
          f"{len(rubrics)} constants")

    print("\n  one name-matching primitive, verified:")
    source = Path("src/llm_sad_sam/linkers/experimental/s_linker38.py").read_text()
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith(("#", '"""', "``")))
    print(f"    has_standalone_mention referenced in code: "
          f"{'YES — still two primitives' if 'has_standalone_mention(' in code else 'no'}")
    print(f"    _find_exact_form call sites: {code.count('_find_exact_form(')}")

    print("\n  coded rejections, and what each removed in these runs:")
    grand = Counter()
    for project in PROJECTS:
        info = load_project(project)
        gold = info["gold"]
        per = Counter()
        present = runs_with(project)
        for run in present:
            full = phase(run, project, "linker_full_name")
            partial = phase(run, project, "linker_partial_name")
            coref = phase(run, project, "linker_coreference")
            prior_full = {(l.sentence_number, l.component_id) for l in full["links"]}
            prior = set(prior_full)
            if partial:
                prior |= {(l.sentence_number, l.component_id)
                          for l in partial["links"]}
            per["full_candidates"] += len(full["feedback"]["candidates"])
            per["full_rejected"] += sum(
                1 for d in full["feedback"]["judge_decisions"]
                if not d["approved"])
            per["full_rejected_gold"] += sum(
                1 for d in full["feedback"]["judge_decisions"]
                if not d["approved"] and (d["sentence"], d["component_id"]) in gold)
            if partial:
                proposed = partial["feedback"].get(
                    "candidates", partial["feedback"].get("proposed", []))
                per["partial_proposed"] += len(proposed)
                per["partial_accepted"] += len(partial["links"])
            if coref:
                reported = coref["feedback"].get("candidates", [])
                per["coref_reported"] += len(reported)
                per["coref_accepted"] += len(coref["links"])
                per["coref_dup"] += sum(
                    1 for item in reported
                    if (item["sentence"],
                        info["name_to_id"].get(item["component"])) in prior)
        n = len(present) or 1
        print(f"  {project:14s} full-name: {per['full_candidates']/n:5.1f} "
              f"candidates, judge rejects {per['full_rejected']/n:4.1f} "
              f"({per['full_rejected_gold']/n:4.1f} gold) | partial-name: "
              f"{per['partial_proposed']/n:5.1f} proposed -> "
              f"{per['partial_accepted']/n:4.1f} | coreference: "
              f"{per['coref_reported']/n:4.1f} reported -> "
              f"{per['coref_accepted']/n:4.1f} (already linked: "
              f"{per['coref_dup']/n:4.1f})")
        grand.update(per)

    # the two deterministic proposer conditions, priced exactly
    print("\n  the partial-name proposer's two conditions, recomputed:")
    for project in PROJECTS:
        info = load_project(project)
        linker = SLinker38.__new__(SLinker38)
        linker.doc_knowledge = None
        counts = Counter()
        words_by_component = {
            component.id: [w.casefold() for w in re.findall(
                r"[A-Za-z]+[A-Za-z0-9]*|\d+", component.name)]
            for component in info["components"]
        }
        for sentence in info["sentences"]:
            for match in re.finditer(r"[A-Za-z]+[A-Za-z0-9]*|\d+", sentence.text):
                if SLinker38._inside_qualified_identifier(
                        sentence.text, match.start(), match.end()):
                    counts["qualified_suppressed"] += 1
                    continue
                surface = match.group(0).casefold()
                owners = [c for c in info["components"]
                          if any(surface.startswith(w)
                                 for w in words_by_component[c.id])]
                if not owners:
                    continue
                if len(owners) > 1:
                    counts["multi_owner_dropped"] += 1
                    continue
                component = owners[0]
                if SLinker38._find_exact_form(sentence.text, component.name):
                    counts["whole_name_dropped"] += 1
                    continue
                counts["proposed"] += 1
                if surface != component.name.casefold() and any(
                        surface.startswith(w) and surface != w
                        for w in words_by_component[component.id]):
                    counts["by_prefix_only"] += 1
        print(f"    {project:14s} suppressed inside a qualified identifier "
              f"{counts['qualified_suppressed']:5d} | dropped for >1 owner "
              f"{counts['multi_owner_dropped']:4d} | dropped because the whole "
              f"name is stated {counts['whole_name_dropped']:4d} | proposed "
              f"{counts['proposed']:4d} (of which only a prefix match: "
              f"{counts['by_prefix_only']:3d})")
        grand.update({f"{project}_{k}": v for k, v in counts.items()})
    return dict(grand)


# ── A5 ───────────────────────────────────────────────────────────────────────

def audit_mention_types():
    """Do the five mention-type values behave differently in the judge's verdicts?

    The value reaches the model as one line of the case block. If two values carry
    the same approval rate and the same gold rate, the distinction is not doing
    work and the taxonomy can shrink.
    """
    print("\n### A5 MENTION-TYPE VALUES — does each value change a verdict?")
    stats = defaultdict(Counter)
    for project in PROJECTS:
        gold = load_gold(project)
        info = load_project(project)
        sent_map = info["sent_map"]
        id_to_name = {c.id: c.name for c in info["components"]}
        for run in runs_with(project):
            table = phase(run, project, "knowledge")
            linker = SLinker38.__new__(SLinker38)
            linker.doc_knowledge = table["doc_knowledge"] if table else None
            for d in phase(run, project, "linker_full_name")["feedback"][
                    "judge_decisions"]:
                comp = id_to_name.get(d["component_id"])
                sentence = sent_map.get(d["sentence"])
                if not comp or not sentence:
                    continue
                kind = linker._classify_mention_typed(comp, sentence.text)
                bucket = stats[kind.value if isinstance(kind, MentionType) else kind]
                key = (d["sentence"], d["component_id"])
                bucket["cases"] += 1
                bucket["approved"] += bool(d["approved"])
                bucket["gold"] += key in gold
                bucket["approved_gold"] += bool(d["approved"]) and key in gold
                bucket["split"] += bool(d["p1"]) != bool(d["p2"])
    n = len(RUNS)
    print(f"  {'value':22s} {'cases':>7s} {'approved':>9s} {'gold':>7s} "
          f"{'approval':>9s} {'gold rate':>10s} {'split':>6s}")
    for value, bucket in sorted(stats.items(), key=lambda kv: -kv[1]["cases"]):
        cases = max(bucket["cases"], 1)
        print(f"  {value:22s} {bucket['cases']/n:7.1f} {bucket['approved']/n:9.1f} "
              f"{bucket['gold']/n:7.1f} {bucket['approved']/cases*100:8.1f}% "
              f"{bucket['gold']/cases*100:9.1f}% {bucket['split']/n:6.1f}")
    rates = {v: b["approved"] / max(b["cases"], 1) for v, b in stats.items()}
    print("\n  values whose approval rates are within 5 points of each other are "
          "candidates to merge:")
    seen = sorted(rates.items(), key=lambda kv: -kv[1])
    for i, (value, rate) in enumerate(seen):
        near = [w for w, r in seen[i + 1:] if abs(r - rate) <= 0.05]
        if near:
            print(f"    {value} ({rate*100:.0f}%) ~ "
                  + ", ".join(f"{w} ({rates[w]*100:.0f}%)" for w in near))
    return {k: dict(v) for k, v in stats.items()}


# ── A6 ───────────────────────────────────────────────────────────────────────

CASE_HEAD = re.compile(r'^Case (\d+): "(.*)" -> (.+)$', re.M)


def audit_uniqueness_value():
    """What does the uniqueness question buy, joined to the gold standard?

    A2 counts how often uniqueness disagrees with relevance; this says whether
    those disagreements are right. Each case is recovered from the judging prompt
    itself -- the header carries the matched surface and the component, and the
    next line carries the sentence, which identifies the sentence number -- so the
    verdicts can be scored without re-running anything.
    """
    print("\n### A6 UNIQUENESS — is the second criterion right when it disagrees?")
    grand = Counter()
    examples = Counter()
    for project in PROJECTS:
        gold = load_gold(project)
        info = load_project(project)
        name_to_id = info["name_to_id"]
        by_text = {}
        for sentence in info["sentences"]:
            by_text.setdefault(sentence.text.strip(), sentence.number)
        per = Counter()
        for run in RUNS:
            for call in calls(run, project):
                if call.get("phase") not in ("phase_25_full_name_p1",
                                             "phase_25_full_name_p2"):
                    continue
                prompt = call.get("prompt", "")
                data = response_json(call.get("response_text"))
                if not data:
                    continue
                lines = prompt.splitlines()
                cases = {}
                for i, line in enumerate(lines):
                    head = CASE_HEAD.match(line)
                    if not head or i + 1 >= len(lines):
                        continue
                    quoted = re.findall(r'"([^"]*)"', lines[i + 1])
                    snum = None
                    for candidate in quoted:
                        snum = by_text.get(candidate.strip())
                        if snum is not None:
                            break
                    cid = name_to_id.get(head.group(3).strip())
                    if snum is not None and cid:
                        cases[int(head.group(1)) - 1] = (snum, cid)
                for item in data.get("validations", []) or []:
                    key = cases.get(item.get("case", 0) - 1)
                    if key is None:
                        per["unmatched"] += 1
                        continue

                    def flag(field):
                        value = item.get(field, False)
                        return value is True or (isinstance(value, str)
                                                 and value.lower() == "true")
                    relevant, unique = flag("relevant"), flag("unique")
                    per["matched"] += 1
                    if relevant and not unique:
                        per["uniqueness_rejects"] += 1
                        per["uniqueness_rejects_gold"] += key in gold
                        examples[(project, key[0], key[1] in (), key in gold)] += 1
        n = len(RUNS)
        print(f"  {project:14s} answers matched to a pair {per['matched']/n:5.1f} "
              f"(unmatched {per['unmatched']/n:4.1f}) | uniqueness rejects what "
              f"relevance keeps {per['uniqueness_rejects']/n:4.1f}, of which gold "
              f"{per['uniqueness_rejects_gold']/n:4.1f}")
        grand.update(per)
    n = len(RUNS)
    rejects = grand["uniqueness_rejects"] / n
    rejects_gold = grand["uniqueness_rejects_gold"] / n
    print(f"  TOTAL          uniqueness rejects {rejects:.1f} per run, "
          f"{rejects_gold:.1f} gold and {rejects - rejects_gold:.1f} not")
    print(f"  Dropping the uniqueness question would therefore read about "
          f"TP +{rejects_gold:.1f} / FP +{rejects - rejects_gold:.1f} per run, "
          f"before the ANDing of the two samples is taken into account.")
    return dict(grand)


# ── A7 ───────────────────────────────────────────────────────────────────────

def audit_coreference_scope():
    """The largest call consumer, and whether restricting its input is safe.

    Coreference resolution is 46.3 of 101.5 calls per run because it reads every
    sentence of every document in batches of ten, and 64% of what it reports is a
    pair an earlier linker already produced. The obvious saving is to ask only about
    sentences that carry no link yet -- which is also what would make the paper's
    "each linker sees only what the earlier ones left unlinked" literally true of
    the input rather than of the output. This prices it: an accepted coreference
    link whose sentence already carries a link to a *different* component would be
    lost, because the sentence would never be asked about.
    """
    print("\n### A7 COREFERENCE SCOPE — is restricting the input safe?")
    grand = Counter()
    for project in PROJECTS:
        gold = load_gold(project)
        per = Counter()
        present = runs_with(project, "linker_coreference")
        for run in present:
            full = phase(run, project, "linker_full_name")
            partial = phase(run, project, "linker_partial_name")
            coref = phase(run, project, "linker_coreference")
            earlier = [(l.sentence_number, l.component_id) for l in full["links"]]
            if partial:
                earlier += [(l.sentence_number, l.component_id)
                            for l in partial["links"]]
            linked_sentences = {snum for snum, _ in earlier}
            for link in coref["links"]:
                key = (link.sentence_number, link.component_id)
                per["accepted"] += 1
                per["accepted_gold"] += key in gold
                if link.sentence_number in linked_sentences:
                    per["in_a_linked_sentence"] += 1
                    per["in_a_linked_sentence_gold"] += key in gold
            per["sentences_total"] += len(
                {s for s, _ in earlier} | {l.sentence_number
                                           for l in coref["links"]})
            per["sentences_linked"] += len(linked_sentences)
        n = len(present) or 1
        print(f"  {project:14s} coreference links {per['accepted']/n:5.1f} "
              f"({per['accepted_gold']/n:4.1f} gold) | of those, in a sentence an "
              f"earlier linker already used: {per['in_a_linked_sentence']/n:4.1f} "
              f"({per['in_a_linked_sentence_gold']/n:4.1f} gold)")
        grand.update(per)
    n = len(RUNS)
    lost = grand["in_a_linked_sentence"] / n
    lost_gold = grand["in_a_linked_sentence_gold"] / n
    print(f"  TOTAL          {grand['accepted']/n:.1f} coreference links, "
          f"{grand['accepted_gold']/n:.1f} gold | asking only about unlinked "
          f"sentences would drop {lost:.1f} of them, {lost_gold:.1f} gold")
    print("  A sentence can carry a named link to one component and a "
          "coreference link to another, so restricting the input by sentence is "
          "not a subtraction of duplicates -- it is a loss.")
    return dict(grand)


AUDITS = {
    "A1": audit_self_agreement,
    "A6": audit_uniqueness_value,
    "A7": audit_coreference_scope,
    "A2": audit_criterion_overlap,
    "A3": audit_protocol_asymmetry,
    "A4": audit_judging_surface,
    "A5": audit_mention_types,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=list(AUDITS))
    parser.add_argument("--out", type=Path,
                        default=Path("../results/s38_audit/audit.json"))
    args = parser.parse_args()
    report = {key: AUDITS[key]() for key in args.only}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
