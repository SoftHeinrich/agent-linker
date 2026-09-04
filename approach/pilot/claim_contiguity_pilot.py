"""Level-2 stage pilot: the denotation gate's contiguity instruction, priced.

`s_linker110`'s partial-name denotation prompt ends with one authored line that no
named rule constant holds and `pilot/prompt_defensibility.py` therefore never scored:

    Claim must be a contiguous exact substring of the source sentence.

It has been in that prompt since `s_linker25` and no arm has ever priced it **there**.
The branch's one measurement of this clause is `pilot/design_pilots.py`'s `ClaimChecked`
(`--pilot claim`), which *added* it -- with an enforcing substring check -- to the
`_prompt_validation` judge, the full-name/coreference gate: TP +/-0.0 (p = 1.00),
**FP +1.6 (p = 0.02)**, and the check itself voided 0 verdicts in 25 project-runs. So
the instruction moves verdicts at a gate that is not this one, and the enforcement is
inert everywhere. At this gate the parse-side gate is inert too: `evidence_valid`
requires a non-empty claim and over 238 recorded decisions (3 terra runs x 5 projects)
0 were empty, 236 of 238 claims were already contiguous substrings, and the 2 that were
not (bigbluebutton S49, "recorded events" welded across "recorded, all events") were
`associated` rejections that changed no link.

That is the whole of what level 1 can say, and `pilot/prompt_audit.py` states why it
cannot conclude: **a prohibition has its effect through absence.** 236/238 compliance is
equally consistent with "the rule works" and "the rule is unnecessary". The
counterfactual is a different prompt, so it is a stage arm.

**The arm is the control with a post-processor, not a re-declared prompt builder.** The
measurement policy requires asserting that a re-declared builder renders byte-identically
to the variant's own; deleting the line from the control's rendered prompt makes that
assertion hold by construction, and the pilot fails loudly if the line is not found
exactly once. Nothing else about the call changes -- same scan, same pinned alias table,
same batching, same schema field, same parser.

Candidates come from `s_linker109`'s deterministic scan with a recorded run's alias
table pinned, so the candidate set is byte-identical across arms and samples: the only
thing that varies is the one line. Only teammates and bigbluebutton propose partial-name
candidates at all (mediastore, teastore and jabref propose 0), so a five-project run is
4 denotation calls and the default two-project run is the same 4.

    OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai \\
    OPENAI_MODEL_NAME=gpt-5.6-terra OPENAI_REASONING_EFFORT=none \\
      ../.venv/bin/python pilot/claim_contiguity_pilot.py --samples 3

Both arms run inside one invocation, which is the branch rule: absolute levels drift
between invocation sets, so `--arms` in one run is the comparison.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110  # noqa: E402
from llm_sad_sam.llm_client import LLMBackend  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from nextgen_pilots import DEFAULT_RUN, pinned_knowledge  # noqa: E402
from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402

#: The line under test, exactly as `s_linker110._classify_denotations` renders it.
CONTIGUITY_LINE = "Claim must be a contiguous exact substring of the source sentence."

#: The phase whose prompt carries it. No other call is touched.
DENOTATION_PHASE = "phase_25_partial_denotation"


class NoContiguityLine(SLinker110):
    """The head, with that one line deleted from the denotation prompt.

    Implemented as a post-processor on `_ask` rather than as a re-declared
    `_classify_denotations`, so every other byte of the call is the control's by
    construction. Raises if the line is absent or ambiguous, which is the assertion
    the measurement policy asks for, made unskippable.
    """

    def _ask(self, prompt, **kwargs):
        if kwargs.get("phase") == DENOTATION_PHASE:
            hits = prompt.count(CONTIGUITY_LINE)
            if hits != 1:
                raise AssertionError(
                    f"contiguity line found {hits} times in the denotation prompt; "
                    "the arm can only be trusted when it is present exactly once")
            prompt = prompt.replace(CONTIGUITY_LINE + "\n", "", 1)
            if CONTIGUITY_LINE in prompt:
                raise AssertionError("contiguity line survived removal")
        return super()._ask(prompt, **kwargs)


ARMS = {"control": SLinker110, "nocontig": NoContiguityLine}


def stage(linker, sentences, components, sent_map):
    """The partial-name gate over its own scan's candidates."""
    links, feedback = linker._run_partial_name_linker(sentences, components, sent_map)
    return ({(l.sentence_number, l.component_id) for l in links},
            len(feedback["proposed"]),
            feedback["judge_decisions"])


def permutation(a, b, iterations=20000, seed=0):
    """Paired sign-flip permutation test, `pilot/score_runs.py`'s form."""
    if len(a) != len(b) or not a:
        return 1.0
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(sum(diffs) / len(diffs))
    if len(diffs) <= 20:
        signs = itertools.product([1, -1], repeat=len(diffs))
        outcomes = [abs(sum(s * d for s, d in zip(sign, diffs)) / len(diffs))
                    for sign in signs]
    else:
        rng = random.Random(seed)
        outcomes = [abs(sum(rng.choice((1, -1)) * d for d in diffs) / len(diffs))
                    for _ in range(iterations)]
    hits = sum(1 for o in outcomes if o >= observed - 1e-12)
    return hits / len(outcomes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--datasets", nargs="+",
                        default=["teammates", "bigbluebutton"],
                        help="the two projects whose scan proposes candidates")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dump")
    args = parser.parse_args()

    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown arm(s): {unknown}")

    backend = LLMBackend.OPENAI
    model = os.environ.get("OPENAI_MODEL_NAME", "")
    totals = collections.defaultdict(collections.Counter)
    per_sample = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter))
    claims = collections.defaultdict(list)
    dump: dict = {}

    for project in args.datasets:
        text, repo, gold_path = DATASETS[project]
        components = parse_pcm_repository(str(BENCH / repo))
        sentences = load_sentences(str(BENCH / text))
        sent_map = build_sent_map(sentences)
        gold = gold_pairs(BENCH / gold_path)
        knowledge = pinned_knowledge(args.run, project)
        sent_text = {s.number: s.text for s in sentences}
        print(f"\n=== {project}: {len(sentences)} sentences, {len(components)} "
              f"components, {len(gold)} gold, "
              f"{len(getattr(knowledge, 'aliases', {}) or {})} aliases held fixed ===",
              flush=True)

        sizes = set()
        for sample in range(1, args.samples + 1):
            for arm in args.arms:
                linker = ARMS[arm](backend=backend, model=model)
                linker.doc_knowledge = knowledge
                kept, proposed, decisions = stage(
                    linker, sentences, components, sent_map)
                sizes.add(proposed)
                good = len(kept & gold)
                bad = len(kept) - good
                for key, value in (("kept", len(kept)), ("gold", good),
                                   ("spurious", bad), ("candidates", proposed)):
                    totals[arm][key] += value
                    per_sample[sample][arm][key] += value
                for d in decisions:
                    claim = d.get("claim", "")
                    source = sent_text.get(d["sentence"], "")
                    claims[arm].append({
                        "project": project, "sample": sample,
                        "sentence": d["sentence"], "claim": claim,
                        "denotation": d.get("denotation"),
                        "contiguous": bool(claim) and claim in source,
                        "empty": not claim,
                    })
                dump.setdefault(f"sample{sample}", {}).setdefault(project, {})[arm] = \
                    sorted([list(p) for p in kept])
                print(f"  {arm:<9} sample {sample}: {len(kept):4d} kept, "
                      f"{good:4d} gold, {bad:4d} spurious "
                      f"(of {proposed} candidates)", flush=True)
        if len(sizes) > 1:
            print(f"  !! candidate set differs across arms: {sorted(sizes)} "
                  "-- the pinned scan is supposed to make this impossible")

    runs = args.samples
    print(f"\ndenotation gate on {model or 'unset'}, {runs} samples, "
          f"{len(args.datasets)} projects, per run:")
    print(f"  {'arm':<10}{'candidates':>12}{'kept':>8}{'TP':>8}{'FP':>8}"
          f"{'precision':>11}")
    for arm in args.arms:
        row = totals[arm]
        precision = row["gold"] / row["kept"] if row["kept"] else 0.0
        print(f"  {arm:<10}{row['candidates'] / runs:>12.1f}{row['kept'] / runs:>8.1f}"
              f"{row['gold'] / runs:>8.1f}{row['spurious'] / runs:>8.1f}"
              f"{precision:>11.3f}")

    if len(args.arms) == 2:
        a, b = args.arms
        print(f"\npaired sign-flip permutation, {runs} samples, {b} against {a}:")
        for stat in ("gold", "spurious", "kept"):
            xa = [per_sample[s][a][stat] for s in sorted(per_sample)]
            xb = [per_sample[s][b][stat] for s in sorted(per_sample)]
            delta = (sum(xb) - sum(xa)) / runs
            print(f"  {stat:<10}{a} {list(xa)} -> {b} {list(xb)}"
                  f"   delta {delta:+.1f}  p = {permutation(xb, xa):.2f}")

    print("\nclaim shape (what the line is supposed to govern):")
    for arm in args.arms:
        rows = claims[arm]
        if not rows:
            continue
        n = len(rows)
        noncontig = [r for r in rows if not r["contiguous"] and not r["empty"]]
        print(f"  {arm:<10}{n:4d} claims, {sum(1 for r in rows if r['empty'])} empty, "
              f"{len(noncontig)} not a contiguous substring "
              f"({100 * len(noncontig) / n:.1f}%)")
        for r in noncontig[:5]:
            print(f"      {r['project']} S{r['sentence']} s{r['sample']} "
                  f"{r['denotation']}: {r['claim']!r}")

    if args.dump:
        json.dump({"kept": dump, "claims": claims}, open(args.dump, "w"), indent=1)
        print("\nkept pairs and claims written to", args.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
