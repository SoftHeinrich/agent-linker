# Greedy spans and discarded ambiguous name cases

Date: 2026-09-15

This is a fixed-input design audit, not an end-to-end result. It replays the proposed
deterministic transformation over the six recorded s122 runs (three terra and three
luna) and their own cached alias tables and name-judge decisions. No LLM call is
resampled.

## Design

1. A partial-name candidate is removed when every occurrence of its matched word is
   contained in a longer whole component name written in the sentence.
2. When the same remaining surface proposes several components, all proposals for that
   surface are discarded because the written evidence does not identify one component.
3. The judge therefore receives neither duplicated cases nor a `competitors` field.

The first rule is general span ownership over the runtime catalog. It does not name a
benchmark token. The second is a conservative acceptance rule: ambiguous written
evidence is insufficient to select one catalog component.

## Reproduction

From `approach/`:

```text
../.venv/bin/python pilot/s127_greedy_merge_audit.py
```

Result (exit 0):

```text
CACHE 6 recorded runs; 30 project-runs
candidates 1870 -> 1798
greedy removed 72: gold=0, previously approved=1
merged judge cases 1786; genuinely ambiguous=12
cached accepted-link projection (greedy filter only):
  noanchor_e2e_luna_r1_20260914/bigbluebutton: TP 52->52, FP 14->13
by project:
  mediastore: greedy_removed=0, ambiguous_cases=0
  teastore: greedy_removed=0, ambiguous_cases=0
  teammates: greedy_removed=0, ambiguous_cases=0
  bigbluebutton: greedy_removed=72, ambiguous_cases=12
  jabref: greedy_removed=0, ambiguous_cases=0
impacted/relevant BBB examples (occurrences over cached runs):
   6x bigbluebutton S4 'HTML5' -> HTML5 Server
   6x bigbluebutton S5 'HTML5' -> HTML5 Server
   6x bigbluebutton S8 'HTML5' -> HTML5 Client
   6x bigbluebutton S14 'HTML5' -> HTML5 Server
   6x bigbluebutton S15 'HTML5' -> HTML5 Client
   6x bigbluebutton S27 'redis' -> Redis DB, Redis PubSub
   6x bigbluebutton S31 'redis' -> Redis DB, Redis PubSub
   6x bigbluebutton S46 'Redis' -> Redis DB
   6x bigbluebutton S47 'Redis' -> Redis DB
   6x bigbluebutton S48 'Redis' -> Redis PubSub
   6x bigbluebutton S49 'Redis' -> Redis PubSub
   6x bigbluebutton S60 'redis' -> Redis DB
   6x bigbluebutton S72 'HTML5' -> HTML5 Server
   6x bigbluebutton S79 'Redis' -> Redis DB
cached verdict combinations for merged cases:
   2x bigbluebutton S27: selected=Redis DB, Redis PubSub
   4x bigbluebutton S27: selected=Redis PubSub
   2x bigbluebutton S31: selected=Redis DB, Redis PubSub
   4x bigbluebutton S31: selected=Redis PubSub
```

## Interpretation and limitation

The deterministic removal has zero gold cost on these cached inputs. It removes all
contained-word rivals of the S60 type. The only remaining multi-component surfaces are
BBB S27 and S31 (`redis` reaching `Redis DB` and `Redis PubSub`). Neither sentence has
a gold link to either component, so they are rejection cases, not positive examples.
Projecting the filter through the cached decisions changes one accepted link: a
BigBlueButton false positive in luna run 1 is removed, while cached true positives are
unchanged. On the residual ambiguous cases, the independent-case judge accepted
`Redis PubSub` in all 12 cached instances and additionally accepted `Redis DB` in four;
all 16 accepted pairs are false positives against the benchmark gold standard.

S19 (`bbb-html5`) is not a multi-component case in any individual checkpoint. The
alias table is a mapping from a term to one owner and different runs assign that owner
differently. Merging S19 would require a separate change to alias discovery and its
data model.

Removing the `competitors` rule line and discarding unresolved groups are semantic
changes. This cache replay does not validate their composed effects because accepted
name links suppress later coreference candidates.

## Paired end-to-end evaluation and reporting gate

Run date: 2026-09-16. Each invocation contained `s_linker123` and `s_linker126`, arm
order alternated by repetition, and `OPENAI_SERVICE_TIER=flex`. Both GPT-5.6 backends
ran three repetitions over all five projects:

```text
cd approach
STAMP=20260916 pilot/run_s126_e2e.sh terra 3
STAMP=20260916 pilot/run_s126_e2e.sh luna 3
../.venv/bin/python pilot/score_runs.py \
  --arm s_linker123 ../results/greedymerge_e2e_<model>_r{1,2,3}_20260916 \
  --arm s_linker126 ../results/greedymerge_e2e_<model>_r{1,2,3}_20260916
```

Link-level means (`s126 - s123`, paired by invocation):

```text
backend  control F1/F2  candidate F1/F2  delta F1/F2  delta TP/FP
terra      92.90/93.95     95.06/95.93     +2.16/+1.98   +4.0/-4.7
luna       88.81/91.38     89.74/91.86     +0.93/+0.48   +1.0/-9.3
```

The reporting gate used neutral extracts from the same invocations:

```text
build_alinker_extracts.py -> build_dump.py -> rq12.py
studies/compare_arms.py s126 --base s123gctl \
  --csv evaluation/reports/ARM_COMPARE_s126_vs_s123gctl.csv
```

Terra's doc-code file F2 is `+2.30` with 3/3 positive signs; its file F1 and tail
metrics have positive means but mixed signs. Luna is negative with 3/3 signs on every
decisive doc-code metric: file F1 `-0.86`, file F2 `-1.59`, and worst-component F1
`-2.32`. Harmonic-component F1 is inside noise (`+0.01`, mixed signs). Thus the change
does not clear the cross-backend reporting gate. `s_linker126` remains the experimental
head and `s_linker120` remains the paper arm.

Two initial commands failed before producing scores: the first dump attempt used
`../.venv/bin/python` from the repository root, and the first score attempt ran the
working-directory-relative scorer from the repository root. Both were rerun with the
paths shown above. Flex also produced transient resource/connection retries; all six
final run directories completed.
