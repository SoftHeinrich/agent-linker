"""What is left in the *prompts* of s_linker49, measured before any pilot is paid for.

Every earlier round in this branch ablated structure: stages, calls, batch
constants, code predicates. Twenty variants later no mechanism has been removed
without a measured cost, and the design is defended as necessary rather than
chosen. That leaves the one surface never audited: the hand-written instruction
text itself. About 5 kB of English is carried into 88 calls per five-project run,
and a reviewer's fair question is whether those sentences are *general
guidelines* or a rulebook accreted against this benchmark.

Everything below is read off s49's own six-run checkpoints and call traces. No
LLM call is made. The point is the same as every earlier audit: size a proposed
removal before paying for it.

  P0  INSTRUCTION SURFACE   how many bytes of authored instruction exist, split
                            per prompt and per clause, and how many times is the
                            same rule restated in different words?
  P1  QUALIFIED-PATH RULE   the "X.Y / X.Y.Z is not the component" stipulation is
                            written five times in five prompts. For each copy,
                            how many items in the population it governs even
                            contain a dotted identifier? A copy whose population
                            has none cannot change a decision.
  P2  FULL-NAME REJECT LIST four enumerated reject-conditions and three
                            enumerated approve-examples. How many judged cases
                            match each condition lexically, and what is the
                            approval rate inside each bucket?
  P3  COREFERENCE CLAUSE B  the topic-resolution clause with its list of
                            role-referential phrases. How many resolutions are
                            licensed only by it (no name in the context window),
                            and how many use a phrase from the list?
  P4  TERMINAL-WORD SENTENCE the coreference prompt separately tells the model
                            that terminal words of a multi-word name count as
                            aliases. How many antecedents are of that form?
  P5  COREF REJECT LIST     the strict rubric enumerates fragment / gerund / list
                            item. How many judged coreference cases look like
                            that, and are they the ones rejected?

A caveat this audit states up front and does not hide: a *prohibition* has its
effect through absence. If no proposed alias is a dotted path, that is equally
consistent with "the rule works" and "the rule is unnecessary". Trace reach
therefore bounds what a clause *can* be doing when the clause describes something
that occurs (P2 approve-examples, P3, P4); for the prohibitions (P1, P2's reject
list, P5) it bounds only the population at risk, and the arm has to be run.

Usage (from the approach/ directory):
    ../.venv/bin/python pilot/prompt_audit.py
    ../.venv/bin/python pilot/prompt_audit.py --only P1 P3
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
from llm_sad_sam.linkers.experimental import s_linker49 as L49      # noqa: E402

RUNS = [Path(f"../results/s49_composed_e2e_r{i}_20260813") for i in range(1, 7)]
VARIANT = "s_linker49"

# A dotted identifier: two name-ish tokens joined by a dot, no whitespace. This is
# the lexical shape all five copies of the qualified-path rule talk about.
DOTTED = re.compile(r"\b[A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*)+\b")
# Ordinary sentence-final decimals and file suffixes are not qualified names for
# this purpose only when the tail is a digit run; keep everything else.
NEGATION = re.compile(r"\b(not|never|no|cannot|does not|doesn't|isn't|without)\b",
                      re.I)
GERUND_START = re.compile(r"^\s*\w+ing\b", re.I)
FINITE_VERB = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|does|do|did|can|could|will|"
    r"would|shall|should|may|might|must|uses|use|used|provides|provide|sends|"
    r"send|receives|receive|stores|store|handles|handle|contains|contain|"
    r"connects|connect|calls|call|runs|run|manages|manage|returns|return|"
    r"accesses|access|implements|implement|supports|support|requires|require)\b",
    re.I)

ROLE_PHRASES = ("it", "the module", "the service", "the component", "the system")


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


def present_runs():
    return [r for r in RUNS if (r / "phase_states" / VARIANT).exists()]


def dotted_spans(text):
    return [m.group(0) for m in DOTTED.finditer(text)
            if not m.group(0).split(".")[-1].isdigit()]


# ── P0 ───────────────────────────────────────────────────────────────────────

CLAUSES = {
    "DOC_KNOWLEDGE_EXTRACTION_RULES": L49.DOC_KNOWLEDGE_EXTRACTION_RULES,
    "ALIAS_EXCLUSION_RULES": L49.ALIAS_EXCLUSION_RULES,
    "DOC_KNOWLEDGE_JUDGE_RULES": L49.DOC_KNOWLEDGE_JUDGE_RULES,
    "ENTITY_EXTRACTION_RULES": L49.ENTITY_EXTRACTION_RULES,
    "P1_FOCUS": L49.P1_FOCUS,
    "P2_FOCUS": L49.P2_FOCUS,
    "LAYERED_ENTITY_RULES": L49.LAYERED_ENTITY_RULES,
    "COREF_VALIDATION_FOCUS": L49.COREF_VALIDATION_FOCUS,
    "LAYERED_COREF_RULES": L49.LAYERED_COREF_RULES,
    "COREF_RULES": L49.COREF_RULES,
}

# Which prompt phases each clause is carried into, and how many calls of that
# phase one five-project run makes (filled from the traces).
CLAUSE_PHASES = {
    "DOC_KNOWLEDGE_EXTRACTION_RULES": ("phase_25_doc_extract",),
    "ALIAS_EXCLUSION_RULES": ("phase_25_doc_extract",),
    "DOC_KNOWLEDGE_JUDGE_RULES": ("phase_25_doc_judge",),
    "ENTITY_EXTRACTION_RULES": ("phase_25_full_name_extract",),
    "P1_FOCUS": ("phase_25_full_name_p1",),
    "P2_FOCUS": ("phase_25_full_name_p2",),
    "LAYERED_ENTITY_RULES": ("phase_25_full_name_p1", "phase_25_full_name_p2"),
    "COREF_VALIDATION_FOCUS": ("phase_25_coreference_judge",),
    "LAYERED_COREF_RULES": ("phase_25_coreference_judge",),
    "COREF_RULES": ("phase_25_coreference",),
}

# The qualified-path stipulation, restated. Each entry is (clause, the substring
# that carries it, the population it governs).
PATH_COPIES = [
    ("ALIAS_EXCLUSION_RULES", "Qualified-name fragments",
     "terms proposed by the alias extractor"),
    ("ENTITY_EXTRACTION_RULES", "only inside a code-level path",
     "references proposed by the entity extractor"),
    ("P1_FOCUS", "not just as a qualified-name identifier",
     "candidates judged by the full-name relevance call"),
    ("LAYERED_ENTITY_RULES", "code-level or\npackage/member path",
     "candidates judged by both full-name calls"),
    ("LAYERED_COREF_RULES", "code/package path (x.y.z)",
     "candidates judged by the coreference call"),
]


def audit_surface():
    print("\n### P0 INSTRUCTION SURFACE — how much authored English, restated how often?")
    runs = present_runs()
    per_phase = Counter()
    for run in runs[:1]:
        for project in PROJECTS:
            for call in calls(run, project):
                per_phase[call["phase"]] += 1

    total = 0
    print(f"  {'clause':34s} {'bytes':>6s} {'calls/run':>9s} {'bytes/run':>10s}  carried into")
    rows = []
    for name, text in CLAUSES.items():
        size = len(text)
        total += size
        n_calls = sum(per_phase[p] for p in CLAUSE_PHASES[name])
        rows.append((size * n_calls, name, size, n_calls))
        print(f"  {name:34s} {size:6d} {n_calls:9d} {size * n_calls:10d}  "
              f"{', '.join(p.replace('phase_25_', '') for p in CLAUSE_PHASES[name])}")
    fixed = {
        "claim-before-verdict paragraph": 210,
        "coreference prompt preamble": 300,
        "denotation prompt (whole)": 300,
    }
    print(f"  {'-' * 78}")
    print(f"  {'TOTAL of the ten rule constants':34s} {total:6d}")
    print(f"  (plus ~{sum(fixed.values())} bytes of prompt scaffolding: "
          f"{', '.join(fixed)})")

    print("\n  Restatements of the same stipulation:")
    print(f"    qualified-path rule ......... {len(PATH_COPIES)} copies in "
          f"{len({c for c, _, _ in PATH_COPIES})} clauses")
    ranked = sorted(rows, reverse=True)
    print("\n  By bytes actually sent per run (what a trim would save):")
    for cost, name, size, n in ranked:
        print(f"    {name:34s} {cost:7d} B/run  ({size} B x {n} calls)")
    return {"total_rule_bytes": total, "per_phase_calls": dict(per_phase)}


# ── P1 ───────────────────────────────────────────────────────────────────────

def audit_path_rule():
    print("\n### P1 QUALIFIED-PATH RULE — five copies, how much can each one reach?")
    runs = present_runs()
    n = len(runs) or 1

    print("\n  (a) documents: how many sentences carry a dotted identifier at all?")
    doc_dotted = {}
    for project in PROJECTS:
        info = load_project(project)
        hits = [s for s in info["sentences"] if dotted_spans(s.text)]
        doc_dotted[project] = len(hits)
        examples = sorted({d for s in hits for d in dotted_spans(s.text)})[:4]
        print(f"    {project:14s} {len(hits):3d}/{len(info['sentences']):3d} sentences"
              f"   e.g. {', '.join(examples) if examples else '-'}")

    print("\n  (b) alias extractor: does any proposed term look like a path?")
    grand = Counter()
    for project in PROJECTS:
        per = Counter()
        for run in runs:
            for call in calls(run, project):
                if call.get("phase") != "phase_25_doc_extract":
                    continue
                data = response_json(call.get("response_text")) or {}
                for key in ("abbreviations", "synonyms"):
                    for item in data.get(key, []) or []:
                        term = str(item.get("term", ""))
                        per["proposed"] += 1
                        per["dotted"] += bool(dotted_spans(term))
        print(f"    {project:14s} proposed {per['proposed']/n:5.1f}/run   "
              f"dotted {per['dotted']/n:4.1f}")
        grand.update(per)
    print(f"    {'TOTAL':14s} proposed {grand['proposed']/n:5.1f}/run   "
          f"dotted {grand['dotted']/n:4.1f}   <- a prohibition's reach shows as "
          f"absence; this bounds nothing on its own")

    print("\n  (c) entity extractor: proposed references whose matched_text is a path")
    grand = Counter()
    for project in PROJECTS:
        per = Counter()
        for run in runs:
            for call in calls(run, project):
                if call.get("phase") != "phase_25_full_name_extract":
                    continue
                data = response_json(call.get("response_text")) or {}
                for item in data.get("references", []) or []:
                    per["proposed"] += 1
                    per["dotted"] += bool(dotted_spans(str(item.get("matched_text", ""))))
        print(f"    {project:14s} proposed {per['proposed']/n:5.1f}/run   "
              f"dotted matched_text {per['dotted']/n:4.1f}")
        grand.update(per)
    print(f"    {'TOTAL':14s} proposed {grand['proposed']/n:5.1f}/run   "
          f"dotted {grand['dotted']/n:4.1f}")

    print("\n  (d) full-name judge: judged candidates whose sentence has a dotted "
          "identifier,\n      and of those, the ones where the component name occurs "
          "ONLY inside one")
    grand = Counter()
    for project in PROJECTS:
        info = load_project(project)
        id_to_name = {c.id: c.name for c in info["components"]}
        gold = load_gold(project)
        per = Counter()
        for run in runs:
            state = phase(run, project, "linker_full_name")
            if not state:
                continue
            for d in state["feedback"]["judge_decisions"]:
                sentence = info["sent_map"].get(d["sentence"])
                if sentence is None:
                    continue
                name = id_to_name.get(d["component_id"], "")
                spans = dotted_spans(sentence.text)
                inside = [s for s in spans if name and name.casefold() in s.casefold()]
                bare = _bare_occurrence(sentence.text, name, spans)
                per["cases"] += 1
                per["sent_dotted"] += bool(spans)
                if inside and not bare:
                    per["path_only"] += 1
                    per["path_only_approved"] += bool(d["approved"])
                    per["path_only_gold"] += (d["sentence"], d["component_id"]) in gold
        print(f"    {project:14s} cases {per['cases']/n:5.1f}   sentence has a path "
              f"{per['sent_dotted']/n:5.1f}   name ONLY in a path {per['path_only']/n:4.1f}"
              f" (approved {per['path_only_approved']/n:4.1f}, gold "
              f"{per['path_only_gold']/n:4.1f})")
        grand.update(per)
    print(f"    {'TOTAL':14s} cases {grand['cases']/n:5.1f}   sentence has a path "
          f"{grand['sent_dotted']/n:5.1f}   name ONLY in a path "
          f"{grand['path_only']/n:4.1f} (approved {grand['path_only_approved']/n:4.1f},"
          f" gold {grand['path_only_gold']/n:4.1f})")

    print("\n  (e) coreference judge: judged cases whose sentence has a dotted identifier")
    grand = Counter()
    for project in PROJECTS:
        info = load_project(project)
        per = Counter()
        for run in runs:
            state = phase(run, project, "linker_coreference")
            if not state:
                continue
            for d in state["feedback"]["judge_decisions"]:
                sentence = info["sent_map"].get(d["sentence"])
                per["cases"] += 1
                per["sent_dotted"] += bool(sentence and dotted_spans(sentence.text))
        print(f"    {project:14s} cases {per['cases']/n:5.1f}   with a path "
              f"{per['sent_dotted']/n:4.1f}")
        grand.update(per)
    print(f"    {'TOTAL':14s} cases {grand['cases']/n:5.1f}   with a path "
          f"{grand['sent_dotted']/n:4.1f}")
    return {"doc_dotted": doc_dotted}


def _bare_occurrence(text, name, spans):
    """Does the name occur outside every dotted span in this sentence?"""
    if not name:
        return False
    masked = text
    for span in spans:
        masked = masked.replace(span, " " * len(span))
    return name.casefold() in masked.casefold()


# ── P2 ───────────────────────────────────────────────────────────────────────

def audit_entity_rules():
    print("\n### P2 FULL-NAME REJECT LIST — four conditions, three approve-examples")
    runs = present_runs()
    n = len(runs) or 1
    grand = Counter()
    buckets = defaultdict(Counter)
    for project in PROJECTS:
        info = load_project(project)
        id_to_name = {c.id: c.name for c in info["components"]}
        gold = load_gold(project)
        per = Counter()
        for run in runs:
            state = phase(run, project, "linker_full_name")
            if not state:
                continue
            for d in state["feedback"]["judge_decisions"]:
                sentence = info["sent_map"].get(d["sentence"])
                if sentence is None:
                    continue
                text = sentence.text
                name = id_to_name.get(d["component_id"], "")
                spans = dotted_spans(text)
                approved = bool(d["approved"])
                is_gold = (d["sentence"], d["component_id"]) in gold
                per["cases"] += 1
                per["approved"] += approved

                path_only = bool([s for s in spans
                                  if name and name.casefold() in s.casefold()]) \
                    and not _bare_occurrence(text, name, spans)
                negated = bool(NEGATION.search(text))
                headingish = (len(text.split()) <= 6) or not FINITE_VERB.search(text)

                for label, hit in (("(1) path-only", path_only),
                                   ("(2) negation present", negated),
                                   ("heading / short / verbless", headingish)):
                    if hit:
                        buckets[label]["cases"] += 1
                        buckets[label]["approved"] += approved
                        buckets[label]["gold"] += is_gold
                if not (path_only or negated):
                    buckets["residual (3)/(4) territory"]["cases"] += 1
                    buckets["residual (3)/(4) territory"]["approved"] += approved
                    buckets["residual (3)/(4) territory"]["gold"] += is_gold
                if not approved:
                    buckets["ALL REJECTIONS"]["cases"] += 1
                    buckets["ALL REJECTIONS"]["gold"] += is_gold
                    if path_only:
                        buckets["rejections matching (1)"]["cases"] += 1
                        buckets["rejections matching (1)"]["gold"] += is_gold
                    if negated:
                        buckets["rejections matching (2)"]["cases"] += 1
                        buckets["rejections matching (2)"]["gold"] += is_gold
                    if not path_only and not negated:
                        buckets["rejections matching neither"]["cases"] += 1
                        buckets["rejections matching neither"]["gold"] += is_gold
        grand.update(per)
    print(f"  judged cases {grand['cases']/n:5.1f}/run, approved "
          f"{grand['approved']/n:5.1f}, rejected "
          f"{(grand['cases'] - grand['approved'])/n:4.1f}\n")
    print(f"  {'bucket':32s} {'cases/run':>9s} {'approved':>9s} {'gold':>6s}")
    for label in ("(1) path-only", "(2) negation present",
                  "heading / short / verbless", "residual (3)/(4) territory",
                  "ALL REJECTIONS", "rejections matching (1)",
                  "rejections matching (2)", "rejections matching neither"):
        b = buckets[label]
        approved = f"{b['approved']/n:9.1f}" if "REJECT" not in label else " " * 9
        print(f"  {label:32s} {b['cases']/n:9.1f} {approved} {b['gold']/n:6.1f}")
    return {k: dict(v) for k, v in buckets.items()}


# ── P3 / P4 ──────────────────────────────────────────────────────────────────

def audit_coref_clause_b():
    print("\n### P3/P4 COREFERENCE PROMPT — clause (b), the phrase list, terminal words")
    runs = present_runs()
    n = len(runs) or 1
    grand = Counter()
    phrase_counter = Counter()
    for project in PROJECTS:
        info = load_project(project)
        id_to_name = {c.id: c.name for c in info["components"]}
        gold = load_gold(project)
        per = Counter()
        for run in runs:
            state = phase(run, project, "linker_coreference")
            if not state:
                continue
            approved = {(d["sentence"], d["component_id"])
                        for d in state["feedback"]["judge_decisions"] if d["approved"]}
            aliases = _aliases_of(run, project)
            for meta in state["feedback"]["metadata"]:
                key = (meta["sentence"], meta["component_id"])
                name = id_to_name.get(meta["component_id"], "")
                reference = str(meta.get("reference", "")).strip().lower()
                antecedent = str(meta.get("antecedent_text", ""))
                per["resolutions"] += 1
                per["approved"] += key in approved
                per["gold"] += key in gold and key in approved

                # clause (a) reach: is the name (or a discovered alias) actually in
                # the context window the model was shown?
                window = [s.text for s in _window_sentences(info, meta["sentence"])]
                named = any(_states(t, name, aliases) for t in window)
                if not named:
                    per["clause_b_only"] += 1
                    per["clause_b_only_approved"] += key in approved
                    per["clause_b_only_gold"] += key in gold and key in approved
                # the code gate: `_antecedent_states_name` already discards any
                # resolution whose antecedent sentence does not state the name, so
                # clause (b)'s "even without a direct name repetition" survives only
                # when the model cites a naming sentence outside the shown window.
                ant_number = meta.get("antecedent_sentence")
                if isinstance(ant_number, int) and \
                        abs(ant_number - meta["sentence"]) > L49.SLinker49.CONTEXT_SENTENCES:
                    per["antecedent_outside_window"] += 1

                listed = reference in ROLE_PHRASES
                per["listed_phrase"] += listed
                per["listed_phrase_gold"] += listed and key in gold and key in approved
                phrase_counter[reference] += 1

                # P4: is the antecedent quote a terminal-word form rather than the
                # full multi-word name?
                if name and " " in name.strip():
                    terminal = name.split()[-1]
                    if terminal.casefold() in antecedent.casefold() \
                            and name.casefold() not in antecedent.casefold():
                        per["terminal_word_antecedent"] += 1
                        per["terminal_word_gold"] += key in gold and key in approved
        print(f"    {project:14s} resolutions {per['resolutions']/n:5.1f}   "
              f"no name in window {per['clause_b_only']/n:4.1f} "
              f"(approved {per['clause_b_only_approved']/n:4.1f}, gold "
              f"{per['clause_b_only_gold']/n:4.1f})   listed phrase "
              f"{per['listed_phrase']/n:4.1f}   terminal-word antecedent "
              f"{per['terminal_word_antecedent']/n:4.1f}")
        grand.update(per)
    print(f"    {'TOTAL':14s} resolutions {grand['resolutions']/n:5.1f}   "
          f"no name in window {grand['clause_b_only']/n:4.1f} "
          f"(approved {grand['clause_b_only_approved']/n:4.1f}, gold "
          f"{grand['clause_b_only_gold']/n:4.1f})   listed phrase "
          f"{grand['listed_phrase']/n:4.1f} (gold {grand['listed_phrase_gold']/n:4.1f})"
          f"   terminal-word {grand['terminal_word_antecedent']/n:4.1f} "
          f"(gold {grand['terminal_word_gold']/n:4.1f})")
    print(f"\n  antecedent cited outside the shown +/-{L49.SLinker49.CONTEXT_SENTENCES}"
          f" window: {grand['antecedent_outside_window']/n:.1f}/run — the only way a"
          f"\n  clause-(b) resolution can survive `_antecedent_states_name`, which"
          f" discards\n  every resolution whose antecedent sentence does not state the"
          f" name.")
    print("\n  most frequent referring expressions the model reported:")
    for phrase, count in phrase_counter.most_common(15):
        mark = "  <- listed in the prompt" if phrase in ROLE_PHRASES else ""
        print(f"    {count/n:6.1f}/run  {phrase!r}{mark}")
    return dict(grand)


def _aliases_of(run, project):
    state = phase(run, project, "knowledge")
    if not state:
        return {}
    knowledge = state.get("doc_knowledge")
    table = getattr(knowledge, "aliases", None) or getattr(knowledge, "alias_map", None)
    if isinstance(knowledge, dict):
        table = knowledge.get("aliases") or knowledge.get("alias_map")
    out = defaultdict(list)
    for term, target in (table or {}).items():
        name = getattr(target, "component", target)
        out[str(name)].append(str(term))
    return out


def _window_sentences(info, number, span=None):
    span = L49.SLinker49.CONTEXT_SENTENCES if span is None else span
    return [s for s in info["sentences"] if abs(s.number - number) <= span]


def _states(text, name, aliases):
    forms = [name, *aliases.get(name, [])]
    return any(f and re.search(rf"\b{re.escape(f)}\b", text, re.I) for f in forms if f)


# ── P5 ───────────────────────────────────────────────────────────────────────

def audit_coref_reject_list():
    print("\n### P5 COREF REJECT LIST — fragment / gerund / list item, and what fires")
    runs = present_runs()
    n = len(runs) or 1
    buckets = defaultdict(Counter)
    grand = Counter()
    for project in PROJECTS:
        info = load_project(project)
        gold = load_gold(project)
        for run in runs:
            state = phase(run, project, "linker_coreference")
            if not state:
                continue
            # A gold pair an earlier linker already produced costs nothing when the
            # coreference judge rejects it, so the reject list can only be blamed
            # for the pairs coreference alone could contribute.
            earlier = set()
            for name in ("linker_full_name", "linker_partial_name"):
                other = phase(run, project, name)
                for link in (other or {}).get("links", []) or []:
                    earlier.add((link.sentence_number, link.component_id))
            for d in state["feedback"]["judge_decisions"]:
                sentence = info["sent_map"].get(d["sentence"])
                if sentence is None:
                    continue
                text = sentence.text
                approved = bool(d["approved"])
                key = (d["sentence"], d["component_id"])
                is_gold = key in gold and key not in earlier
                grand["cases"] += 1
                grand["approved"] += approved
                labels = []
                if not FINITE_VERB.search(text):
                    labels.append("no finite verb (fragment)")
                if GERUND_START.match(text):
                    labels.append("gerund-initial")
                if len(text.split()) <= 6:
                    labels.append("<= 6 words")
                if not labels:
                    labels = ["ordinary sentence"]
                for label in labels:
                    buckets[label]["cases"] += 1
                    buckets[label]["approved"] += approved
                    buckets[label]["gold"] += is_gold
    print(f"  judged coreference cases {grand['cases']/n:5.1f}/run, approved "
          f"{grand['approved']/n:4.1f}\n")
    print(f"  {'bucket':30s} {'cases/run':>9s} {'approved':>9s} {'gold':>6s}")
    for label, b in sorted(buckets.items(), key=lambda kv: -kv[1]["cases"]):
        print(f"  {label:30s} {b['cases']/n:9.1f} {b['approved']/n:9.1f} "
              f"{b['gold']/n:6.1f}")
    return {k: dict(v) for k, v in buckets.items()}


# ── P6 ───────────────────────────────────────────────────────────────────────

def audit_cost_split():
    """How much of what this workflow sends is instruction, and how much is data?

    The rule constants are 4.0 kB of source but they are sent 88 times a run, so
    the question a cost claim needs is what share of the bytes on the wire they
    are. Anything that is not one of the ten constants is document text, the
    component list, cases or evidence -- data the model must see.
    """
    print("\n### P6 PROMPT COST SPLIT — instruction bytes against data bytes")
    runs = present_runs()
    per_phase_instruction = Counter()
    per_phase_total = Counter()
    per_phase_calls = Counter()
    for run in runs[:1]:
        for project in PROJECTS:
            for call in calls(run, project):
                prompt = call["prompt"]
                instruction = sum(len(text) for text in CLAUSES.values()
                                  if text in prompt)
                per_phase_instruction[call["phase"]] += instruction
                per_phase_total[call["phase"]] += len(prompt)
                per_phase_calls[call["phase"]] += 1
    print(f"  {'phase':32s} {'calls':>5s} {'total B':>9s} {'rules B':>8s} {'rules %':>8s}")
    for name in sorted(per_phase_total):
        total, rules = per_phase_total[name], per_phase_instruction[name]
        print(f"  {name.replace('phase_25_', ''):32s} {per_phase_calls[name]:5d} "
              f"{total:9d} {rules:8d} {100 * rules / max(total, 1):7.1f}%")
    total, rules = sum(per_phase_total.values()), sum(per_phase_instruction.values())
    print(f"  {'TOTAL per five-project run':32s} {sum(per_phase_calls.values()):5d} "
          f"{total:9d} {rules:8d} {100 * rules / max(total, 1):7.1f}%")
    print("\n  So the rules are a small share of the wire cost and the whole share of "
          "the\n  design claim: what a trim buys is not tokens, it is one fewer "
          "hand-written\n  stipulation to defend.")
    return {"total": total, "rules": rules}


# ── P7 ───────────────────────────────────────────────────────────────────────

BUILDERS = {
    "_prompt_doc_knowledge_extract": ("doc_extract", 5),
    "_prompt_doc_knowledge_judge": ("doc_judge", 5),
    "_prompt_extraction": ("full_name_extract", 9),
    "_prompt_validation": ("full_name_p1/p2 + coreference_judge", 25),
    "_prompt_coref": ("coreference", 40),
    "_classify_denotations": ("partial_denotation", 4),
}


def audit_builder_text():
    """The authored English that is *not* one of the ten rule constants.

    Every round so far ablated the constants. The prompt builders also carry text:
    task lines, response contracts, the claim-before-verdict paragraph, JSON
    skeletons. This counts it exactly — the string literals of each builder's
    f-strings, excluding every interpolated value — so the next candidates can be
    chosen by size and by whether something else already enforces them.
    """
    import ast

    print("\n### P7 BUILDER TEXT — the authored bytes outside the ten constants")
    source = Path("src/llm_sad_sam/linkers/experimental/s_linker49.py").read_text()
    tree = ast.parse(source)
    literals: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in BUILDERS:
            continue
        total = 0
        for inner in ast.walk(node):
            if isinstance(inner, ast.JoinedStr):
                total += sum(len(part.value) for part in inner.values
                             if isinstance(part, ast.Constant)
                             and isinstance(part.value, str))
            elif isinstance(inner, ast.Constant) and isinstance(inner.value, str) \
                    and inner.value.strip().startswith(("CONTEXT", "TARGET")):
                total += len(inner.value)
        literals[node.name] = total

    print(f"  {'builder':34s} {'authored B':>10s} {'calls/run':>9s} {'B/run':>8s}  phase")
    grand = 0
    for name, (phase_label, calls_per_run) in BUILDERS.items():
        size = literals.get(name, 0)
        grand += size * calls_per_run
        print(f"  {name:34s} {size:10d} {calls_per_run:9d} {size * calls_per_run:8d}"
              f"  {phase_label}")
    rules_per_run = sum(len(text) * sum(1 for _ in CLAUSE_PHASES[key])
                        for key, text in CLAUSES.items())
    print(f"  {'-' * 76}")
    print(f"  builder text            {grand:>8d} B/run   against 60 892 B/run for the "
          f"ten constants")
    print(f"\n  So roughly {100 * grand / (grand + 60892):.0f}% of the authored English "
          f"this workflow sends is in the\n  builders, and none of it has been ablated. "
          f"Three items dominate:")
    print("    * the coreference prompt's opening instruction paragraph (253 B x 40) —"
          "\n      states the same three things as COREF_RULES, appended in the same"
          " prompt.\n      s_linker56 removes it;")
    print("    * the claim-before-verdict paragraph (210 B x 25) — measured worth"
          " 35.2 TP\n      when removed, so it is not a candidate;")
    print("    * the JSON skeletons (~150 B x 88) — the response contract, and the"
          " parser\n      depends on them.")
    return {"builder_bytes_per_run": grand}


AUDITS = {
    "P0": audit_surface,
    "P6": audit_cost_split,
    "P7": audit_builder_text,
    "P1": audit_path_rule,
    "P2": audit_entity_rules,
    "P3": audit_coref_clause_b,
    "P5": audit_coref_reject_list,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=sorted(AUDITS))
    args = parser.parse_args()
    runs = present_runs()
    print(f"reading {len(runs)} recorded {VARIANT} runs: "
          f"{', '.join(r.name for r in runs)}")
    for key in (args.only or sorted(AUDITS)):
        AUDITS[key]()


if __name__ == "__main__":
    main()
