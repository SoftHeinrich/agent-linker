# The union-judge audit — one rule, one reply, the evidence carrying the difference

**Question.** `s_linker110` asks three judging questions in three prompts. Two of them
already share a builder (`_prompt_validation`, with `strict=` picking the rubric); the
third, the partial-name denotation judge, is a different question asked of a
**target-blind** case. Can they be *one static link rule*, with the difference between
streams carried by an **evidence bundle computed from the match** — the design law
("facts in code, weighings in the prompt") applied to what the judge is *shown* rather
than to what it is *told*?

**Answer.** Yes, and the audit says it is worth measuring for a reason the union was not
proposed for: **the routing the two rubrics implement is coarser than the facts the code
already has.** The lenient stream is two populations — base rate 0.980 and 0.479 — and
the low one carries **13.8 false positives a run, the largest single bucket in the
pipeline**, under a rubric that says "approve by default". The strictest treatment in
the workflow is spent on a row that costs 1.8. The union is what lets the default follow
the evidence instead of the stream.

Tooling: `approach/pilot/unijudge_audit.py` (U1–U5, **no LLM calls**), the arm
`approach/src/llm_sad_sam/linkers/experimental/s_linker120.py`, its invariants
`approach/pilot/test_s120_union.py` (**1700 checks, five projects, no calls**). Data:
`audit.txt` and the CSVs here. Reproduce from `approach/`:

    ../.venv/bin/python pilot/unijudge_audit.py
    ../.venv/bin/python pilot/test_s120_union.py

## U1 — the census: what may move into the evidence, and what may not

Twelve axes separate the three judging calls, each probed against the module's own
source so the table cannot drift from the code:

| kind | axes | may it move into the evidence bundle? |
| --- | --- | --- |
| **FACT** (5) | the rubric's premise (`named here` / `NOT named`), the target shown, the catalog shown, the evidence bundle, the context window | **yes** — each is computed by code the module already runs |
| **WEIGHING** (4) | the rubric constant, `QUALIFIED_CLAUSE`, `STRICTER_CLAUSE`, the demanded quote | no — this is the rule, and it stays one paragraph |
| **CONTRACT** (3) | the `objection` field, the verdict vocabulary, the batch bound | no — each is a measured refusal (s118 ±0.0, **s119 net −9.0 / −16.0**) |

The premise line is the crux: `LAYERED_ENTITY_RULES` opens with *"the component is named
here"* and `LAYERED_COREF_RULES` with *"which is NOT named in the sentence itself"* —
**both rubrics are already conditioned on a fact `_states_a_name` computes.** The union
does not invent a conditioning variable; it moves an existing one from prose into a
field.

## U2 — the evidence is a sufficient statistic for the routing

All 296 deterministic candidates, five projects, by field:

| field | value | cases | gold | rate |
| --- | --- | ---: | ---: | ---: |
| `naming` | whole name | 172 | 133 | 0.773 |
| | alias | 43 | 21 | 0.488 |
| | word only | 81 | 26 | 0.321 |
| `capitalized` | true | 149 | 122 | **0.819** |
| | false | 147 | 58 | **0.395** |
| `mention` | lowercase, inside qualified name | 28 | 2 | **0.071** |
| `alternatives` | ≥1 competing component | 34 | 28 | 0.824 |
| (today's routing) | stream = full_name | 215 | 154 | 0.716 |
| | stream = partial_name | 81 | 26 | 0.321 |

Two readings. First, `naming` reproduces the stream split **and refines it**: the alias
row sits at 0.488 inside the stream the lenient rubric treats as one 0.716 population.
Second, `capitalized` — a fact the code computes and today only *mentions* in
`STRICTER_CLAUSE` — separates the merged stream better (0.819 / 0.395) than the stream
label does.

## U3 — the row table: what the judges actually do, and what they earn

Six recorded head runs (`../consolidation_e2e_{terra,luna}_r{1,2,3}_20260825`), per
five-project run:

| `naming` | case | gate today | cases | base | kept/run | TP/run | FP/run |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| whole name | capitalized | lenient | 101 | **0.980** | 100.7 | 99.0 | 1.7 |
| whole name | lowercase | lenient | 71 | 0.479 | 44.5 | 30.7 | **13.8** |
| alias | capitalized | lenient | 39 | 0.487 | 24.5 | 19.0 | 5.5 |
| alias | lowercase | lenient | 4 | 0.500 | 2.5 | 1.5 | 1.0 |
| word only | capitalized | target-blind | 9 | 0.444 | 7.2 | 3.8 | 3.3 |
| word only | lowercase | target-blind | 72 | 0.306 | 12.8 | 11.0 | **1.8** |

And what the judge's call is worth on each row, against that row's own default:

| row | approve-all | the judge | the judge earns |
| --- | --- | --- | --- |
| whole name, capitalized | 99 TP / 2 FP | 99.0 TP / 1.7 FP | **+0.0 gold, +0.3 FP killed** |
| whole name, lowercase | 34 / 37 | 30.7 / 13.8 | −3.3 gold, **+23.2 FP killed** |
| alias, capitalized | 19 / 20 | 19.0 / 5.5 | +0.0 gold, **+14.5 FP killed** |
| word only, lowercase | 22 / 50 | 11.0 / 1.8 | −11.0 gold, **+48.2 FP killed** |

**Where the judging work is, and where it is not.** A third of the merged stream (101 of
296 cases) is a row the judge confirms and does not change. The two rows where its call
buys most are one lenient (whole name, lowercase: 23.2 FP killed) and one blind (word
only, lowercase: 48.2 FP killed) — and today they are on opposite sides of the routing,
so no single prompt sees both. **This is not a licence to approve a row in code** —
nothing in the deterministic layer may admit a link, and that is the branch's oldest
invariant — but it is the map of where a unioned prompt's leniency must be spent.

**What blindness is buying, priced against the alternative set.** The word-only stream
split by whether the code can enumerate a competing component:

| | cases | base rate | kept/run | TP/run | FP/run |
| --- | ---: | ---: | ---: | ---: | ---: |
| ambiguous (a rival component shares the matched word) | 17 | **0.765** | 8.5 | 7.8 | 0.7 |
| unique (no rival) | 64 | **0.203** | 11.5 | 7.0 | **4.5** |

This inverts the intuition the blindness was designed around. **Sibling confusion is not
where the word-only errors are**: the ambiguous bucket is mostly gold and is judged
almost perfectly (0.7 FP a run). The stream's errors sit in the *unique* bucket — an
ordinary English word that happens to be a word of exactly one component's name, used in
its ordinary sense. That failure mode has a rule written for it already:
`STRICTER_CLAUSE`, which today is attached to the whole-name gate and **withheld from
the stream that needs it**, because a target-blind case has no target for it to speak
about. Showing the target is what lets that clause reach the 4.5 FP a run it was written
for.

## U4 — the merged prompt, built (not sent)

296 cases, one prompt per batch, `JUDGE_BATCH = 25` unchanged. Two modes, and the
distinction is the point:

| | prompt bytes, five projects | against today |
| --- | ---: | ---: |
| today: full-name judge + denotation judge | 167 174 | — |
| **preserving** (every case keeps its current evidence, plus the new fields) | 148 226 | **−11%** |
| swapped (anchors give way to the recency fact they were read for) | 102 903 | −38% |

**Only the preserving mode is the union arm.** Substituting evidence is a change to what
the judge sees and is measured separately — the head's own `EvidenceBundle` docstring
prices dropping evidence at FP 8.3 against a 4–6 band composed. In preserving mode
bigbluebutton *grows* (+7.7 kB): its word-only cases are now shown the target, its
anchors and its alternatives, which is exactly what the arm is for.

A word-only case, as the union builds it:

    Case 3: "web" -> BBB web
      [prev: "HTML5 client."]
      "The HTML5 client is a single page, responsive web application that is built upon …"
      Evidence: source=partial_name, naming=word only, span="web"
      Anchors (confirmed refs):
        S36: BBB web.

## U5 — cost, and the arm

**Same 14 judging calls a five-project run** (296 cases in one stream fill the batches
that 215 + 81 fill in two), one prompt instead of two, one reply schema instead of two,
−11% prompt bytes.

The arm is registered and built: **`s_linker120`** (`unijudge`), a subclass of
`SLinker110` that changes one stage. Its rule states **one default per `naming` row** —
approve-by-default for `whole name` and `alias`, approve-only-when for `word only` —
so the two defaults survive as rule text keyed to a code fact, which is precisely what
`s_linker119` failed to do by keying them to a schema. Both scans, the parse path, the
decision records, the link sources and every other prompt are the head's;
`test_s120_union.py` pins that at **1700 checks**, including that no case is shown less
than its current stage shows it and that an empty reply keeps nothing.

**What is owed before it is bought.** A stage pilot on fixed recorded candidates, three
samples a side, both models, both arms in one invocation, **read per row and not only in
the total** — a union that trades the 0.980 row against the 0.306 row reads neutral in a
sum and is not neutral. Two predicted failure modes with rungs behind them:

1. *The lenient default leaks onto the word-only row* (s119's mechanism, one level up).
   Rung: split the reply so the word-only row answers a different key in the same call.
2. *The shown target makes the model confirm identity* (s25's −5.5 gold). Rung: keep the
   target but withhold the catalog line, so the case is identified without the catalog
   to confirm against — a fact that is already per-case, not per-prompt.

Level 1 throughout. Nothing here was run; every number is a replay of documents,
catalogs and recorded checkpoints.
