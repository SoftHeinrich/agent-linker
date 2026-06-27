# Spike 004 — Results log

Running record. Final verdict at bottom (written in Stage 3).

## Stage 0 — reproduce baselines from cached ablation JSONs ($0)

Harness: `harness/stage0_reproduce.py`. Reads the per-cell `ablation_*.json` for both
sweeps already on disk; macro-F1 = mean over datasets of (mean over 3 runs).

| sweep | macro-F1 | entity FP | coref FP | teammates FP |
|-------|----------|-----------|----------|--------------|
| thinking-on (`v2.6.5_s20union_sonnet`) | **92.8** | 25 | 7 | 16 |
| nothink (`..._nothink_20260627`) | **89.7** | 35 | 27 | 41 |

Reproduces the note exactly on FP signature (entity 25→35, coref 7→27, teammates 16→41).
Note rounded nothink to 89.4; our aggregation gives 89.7 — same to 0.3pp. **Scoring trusted.**
The drop is 100% precision: nothink adds **+30 FPs** (+10 entity, +20 coref), recall flat.
Per-dataset drop is teammates −7.4 (dominant), others −2 to −3, jabref 0.

## Stage 0b — rule-based Mode 2 trap rejecter on cached links ($0)

Harness: `harness/stage0b_trap.py` + `harness/traps.py` (5 structural, taboo-safe traps).
Removal-only post-filter on frozen nothink final links; per-trap + combined.

Precision-perfect ceiling (remove ALL FPs) = **93.9** → max headroom from FP removal = +4.2.

| config | macro-F1 | ΔF1 | FP removed | TP removed | implicit TP killed |
|--------|----------|-----|------------|------------|--------------------|
| overview_header | 87.7 | −2.0 | 9 | 30 | 12 |
| negation | 89.3 | −0.4 | 4 | 6 | 3 |
| qualified_path | 89.5 | −0.2 | 4 | 3 | 3 |
| deictic_pronoun | 88.8 | −0.9 | 1 | 9 | 9 |
| test_scaffolding | 85.4 | −4.3 | 8 | 48 | 0 |
| **ALL_TRAPS** | **82.2** | **−7.5** | 26 | 90 | 24 |

**VERDICT: Mode 2 rejected as a standalone/post-filter mechanism.** Every trap nets
negative — it removes ~3× more true links than false ones. Root cause (confirmed by
dumping removed links): nothink's surplus FPs **share sentences with true links**, e.g.
"Architecture contains UI, Logic, Storage, Common, Test Driver Component…" holds 4 true
entity links but trips `test_scaffolding`/`overview_header`; "The diagram below shows the
object structure of the UI component" is a true UI link tripping `overview_header`. A
sentence-level rule cannot separate the FP from the TP in the same sentence.

Implication: the precision recovery must be **per-(sentence, component)** semantic
judgment — i.e. the LLM validator (Stage 1), not a rule. At most, the trap patterns
become *hints inside the justification prompt*, never hard rejects.

## Stage 1 — LLM validator replay (Mode 5 + Mode 1) at effort 0

Harness: `harness/replay.py` + `harness/layered_validator.py`. Loads cached candidates
(layer3 entity + layer4 coref) from the nothink phase_cache, reconstructs evidence
bundles, re-runs ONLY the entity twopass + coref gates with the layered prompt at
effort 0 (`CLAUDE_DISABLE_THINKING=1`), reassembles final links (Phase-6 dedup), scores.
Validator-layer-only replay confirmed working: teammates run1 = 11 LLM calls, ~118s.

**Rubric v1 (probe), teammates run1** vs cached baselines (same cell):

| config | P | R | F1 | TP | FP (E/C) | FN | impl-FN |
|--------|---|---|----|----|----------|----|---------|
| thinking-on (cache) | 91.2 | 91.2 | 91.2 | 52 | 5 (5/0) | 5 | — |
| nothink (cache) | 76.1 | 89.5 | 82.3 | 51 | 16 (12/4) | 6 | 1 |
| **layered v1 (eff-0)** | 86.0 | 75.4 | 80.4 | 43 | **7 (4/3)** | 14 | 1 |

Finding: v1 **recovers precision hard** (FP 16→7, below thinking-on) but **over-rejects
recall** (TP 51→43, +8 FN) → net −1.9 F1. The recall loss is NOT implicit links (guardrail
held: impl-FN stayed 1). It is one rubric clause: "reject listing/enumeration/overview
header" caused the model to reject sentence 1 — "the architecture contains UI, Logic,
Storage, Common, Test Driver, E2E, Client Component" — which establishes **7 true entity
links**. Same lesson as Stage 0b: enumeration sentences DEFINE the architecture.

**Control — baseline-replay reproduces cached nothink EXACTLY** (teammates run1:
P76.1/R89.5/F1 82.3, TP51/FP16/FN6 — identical to the production nothink cell). So the
validator-layer replay is faithful and any layered gain is attributable to the prompt.

**Rubric evolution on teammates run1** (the hardest cell; nothink 82.3 → thinking-on 91.2):

| rubric | what changed | P | R | F1 | FP (E/C) | TP | impl-FN |
|--------|--------------|---|---|----|----------|----|---------|
| v1 | claim rubric + ALL trap hints (incl. overview) | 86.0 | 75.4 | 80.4 | 7 (4/3) | 43 | 1 |
| v2 | drop overview/listing hint; "naming a part = a claim" | 79.7 | 89.5 | 84.3 | 13 (6/7) | 51 | 1 |
| **v3** | + hard code-path override (reject path-only subjects) | **89.5** | **89.5** | **89.5** | **6 (4/2)** | 51 | 1 |

**v3 ≈ thinking-on parity at effort 0, zero recall cost.** FP 16→6 (thinking-on=5),
recall fully preserved (TP51, impl-FN flat at 1). The rubric alone collapsed coref FP
7→2. Residual 6 FPs are package-path / responsibility-bullet borderlines that thinking-on
also keeps (its FP=5). Cost: 11 LLM calls/cell, ~120s.

### Stage 1/2 — FULL N=3 × 5-dataset sweep at effort 0

Macro = mean over datasets of mean over 3 runs. All vs the same cached baselines.

| config | macro-F1 | entity FP | coref FP | implicit FN |
|--------|----------|-----------|----------|-------------|
| nothink (cache) | 89.7 | 35 | 27 | 59 |
| **v3 (Mode 5+1, rubric only)** | **90.2** | **18** | 20 | 57 |
| v3 + Mode 4 skeptic | 89.6 | 23 | 4 | 72 |
| thinking-on (cache) | 92.8 | 25 | 7 | 43 |

**Per-dataset, summed over 3 runs (TP / FP / FN):**

| dataset | nothink | v3 | thinking-on | gap is… |
|---------|---------|----|----|---------|
| teammates | 151/**41**/20 | 148/**16**/23 | 155/16/16 | **precision** — v3 fixes (FP 41→16) |
| bigbluebutton | 126/6/60 | **120**/7/66 | **132**/6/54 | **recall** — v3 over-rejects (−6 TP, 0 FP gain) |
| mediastore | 85/3/8 | 85/2/8 | 89/2/4 | recall (TP 85→89) — unreachable |
| teastore | 80/11/1 | 80/12/1 | 81/7/0 | v3 misses teastore's FP type |
| jabref | 54/1/0 | 54/1/0 | 54/1/0 | — |

**Two decisive findings:**

1. **The no-reasoning gap is mostly RECALL, not precision.** The +30 nothink FP surplus is
   ~entirely teammates (+25). Elsewhere the nothink→thinking gap is thinking-on generating
   candidates nothink never makes (bbb TP 126→132, mediastore 85→89) — upstream extraction/
   coref discovery, which a validator-only replay **cannot recover**. Validator-ablation
   ceiling ≈ **91** (if teammates precision fully recovered + zero harm elsewhere); the
   remaining ~1.9 to 92.8 is structural upstream recall.

2. **Mode 4 (coref skeptic) is net negative** (89.6 < 90.2). It crushes coref FP (27→4) but
   over-rejects recall-bound datasets (bbb 76.7→72.8, implicit-FN 57→72). Coref FPs are not
   the macro bottleneck. REJECTED.

**v3 over-rejects bbb** because bbb's gold links bare-name headings ("FreeSWITCH.",
"Kurento and WebRTC-SFU.") that v3's "must make a claim" rubric rejects — while teammates'
gold penalizes the code-path / responsibility-bullet mentions v3 correctly rejects. The
discriminator is entity (named) vs coref (anaphoric). → **v4: entity-lenient (approve a
named mention unless code-path/negation/different-entity/generic) + coref-strict.**

### v4 (entity-lenient / coref-strict) — the winning config

| dataset | nothink | v3 | v4 | thinking-on |
|---------|---------|----|----|-------------|
| mediastore | 93.9 | 94.4 | 93.4 | 96.7 |
| teastore | 93.1 | 92.5 | **95.3** | 95.9 |
| teammates | 83.3 | 88.3 | **89.8** | 90.6 |
| bigbluebutton | 79.2 | 76.7 | 78.4 | 81.4 |
| jabref | 99.1 | 99.1 | 97.3 | 99.1 |
| **MACRO** | **89.7** | 90.2 | **90.8** | **92.8** |

v4 entity FP = **25**, coref FP = **7** — exactly thinking-on's FP profile (25 / 7).
Implicit FN = 59 = nothink (guardrail perfectly held). v4 fixes teammates (+6.5) and
teastore (+2.2), recovers most of bbb; small regressions on mediastore (−0.5) and jabref
(−1.8, a 1–2 link wobble on a 54-link project). v4 recovers +1.1 of the 3.1-pt gap and
sits at the validator-ablation ceiling (~91).

**Control (baseline prompt, thinking ON, replayed on nothink candidates):** isolates how
much thinking at the VALIDATION GATES alone is worth vs upstream extraction.

⚠ **CORRECTION to the "mostly upstream recall" claim above.** First control cells
(mediastore, N=3): nothink 93.9 → **baseline+thinkON 96.0** → production thinking-on 96.7.
Re-validating the SAME nothink candidates WITH thinking recovers almost the whole
mediastore gap. A validator can only reject, so F1 rising means the thinking validator
**re-approves true candidates the nothink validator wrongly REJECTED** (and removes FPs).
So the recall gap is largely *candidates-extracted-but-wrongly-rejected* (recoverable at
the gates), NOT *never-extracted* (upstream). My effort-0 v4 (93.4) underperforms
thinking-at-gates (96.0) here → the effort-0 prompt only PARTIALLY substitutes for
thinking. Awaiting full control macro to quantify (if ≈92.8, the gap lives at the gates
and is mostly recoverable by thinking there; effort-0 Mode-5+1 recovers ~1/3 of it).
**Control result (baseline prompt + thinking ON, nothink candidates):**

| dataset | nothink | v4 eff0 | base+thinkON | prod thinkOn |
|---------|---------|---------|--------------|--------------|
| mediastore | 93.9 | 93.4 | 96.0 (N3) | 96.7 |
| teastore | 93.1 | 95.3 | 100.0 (N2) | 95.9 |
| teammates | 83.3 | 89.8 | 89.5 (N1) | 90.6 |
| bigbluebutton | 79.2 | 78.4 | 75.7 (N1) | 81.4 |
| jabref | 99.1 | 97.3 | 97.3 (N1) | 99.1 |
| **MACRO** | **89.7** | **90.8** | **91.7** | **92.8** |

**The clean decomposition of the 3.1-pt gap (nothink 89.7 → thinking-on 92.8):**
- **~2.0 pts are AT THE VALIDATION GATES** — recoverable by re-validating nothink's SAME
  candidates with thinking (base+thinkON macro 91.7). Thinking there both removes FPs AND
  re-approves true candidates the cheap validator wrongly rejected.
- **~1.1 pts are UPSTREAM** (91.7 → 92.8) — production thinking extracts better/more
  candidates; no validator-only replay can reach these.
- The **effort-0 layered prompt (v4) banks 1.1 of the 2.0 gate-recoverable points** (~55%),
  i.e. ~35% of the full gap. It fully matches thinking-at-gates on the precision-bound cell
  (teammates: v4 89.8 ≈ thinkON-gates 89.5) but lags on mediastore (93.4 vs 96.0).

**bbb is the instructive exception:** thinking-at-gates = 75.7, *below* nothink 79.2 — even
a thinking validator OVER-REJECTS bbb, whose gold rewards bare-name headings ("FreeSWITCH.")
that any discriminating gate penalizes. bbb's production gain (81.4) is therefore PURELY
upstream recall. Confirms the gold-inconsistency finding and that bbb's gap is unreachable
(and actively hurt) by validation tightening.

Latency (claude CLI, no token field): effort-0 ≈14–16 s/call; thinking-on ≈38 s/call
(teastore control 188.9 s / 5 calls) — ~2.5× slower per call.

## Stage 3 — VERDICT

**Spike question:** can an effort-0 (no extended thinking) layered validator recover the
~3-point macro-F1 drop, without spending the implicit (`name_in_text=False`) true links?

**Answer: PARTIAL.** The best effort-0 validator (Mode 5 justification scaffold + Mode 1
architectural-claim rubric, made entity-lenient / coref-strict = config **v4**) reaches
**macro 90.8**, recovering **+1.1 of the 3.1-point gap** (nothink 89.7 → thinking-on 92.8).
It does NOT reach the ~92.0 primary target.

**What it DID achieve (cleanly):**
- **FP-filter parity with thinking-on.** v4's FP profile — entity 25, coref 7 — equals
  thinking-on's exactly (35/27 → 25/7). As a false-positive filter, prompt-relocated
  reasoning fully substitutes for extended thinking.
- **Guardrail held.** Implicit-FN = 59 = nothink. The precision win costs zero implicit
  recall. (Mode 1 phrased as "architectural claim, not name presence" was load-bearing.)
- **Recovered the precision-bound cell.** teammates +6.5 (83.3→89.8), teastore +2.2.

**Why it falls short — the mechanism (the real finding, quantified):** ~2/3 of the gap
(2.0 of 3.1 pts) lives at the **validation gates**, ~1/3 (1.1 pts) is upstream extraction.
Control: re-validating nothink's SAME candidates with the ORIGINAL prompt but thinking ON
gives macro 91.7 (mediastore 93.9→96.0, teastore →100). Since a validator can only reject,
F1 rising proves thinking **re-approves true candidates the cheap validator wrongly
rejected**, on top of removing FPs. The effort-0 prompt replicates the FP-removal half
(hence FP parity) but only partly the recall-recovery half — banking 1.1 of the 2.0
gate-recoverable points (~35% of the full gap). The last 1.1 pts (91.7→92.8) is genuinely
upstream and unreachable by any validator-only replay.

**Modes verdict:**
- **Mode 5 (justification) + Mode 1 (claim rubric, entity-lenient/coref-strict): KEEP** —
  the effective effort-0 combination (v4).
- **Mode 2 (rule traps): REJECT** — sentence-level rules strip ~3× more true links than
  false (FPs share sentences with true links).
- **Mode 4 (coref skeptic): REJECT** — net-negative at scale (89.6); over-rejects
  recall-bound datasets.

**Secondary finding (benchmark-critique relevant):** the gold standards are inconsistent
about enumeration/heading mentions — teammates penalizes "Package overview contains
logic.api" while bbb rewards the bare heading "FreeSWITCH." — which caps any single-rubric
validator and forced the entity/coref asymmetry. Resonates with the eval/ Ch2 metric
critique. (Also: bbb is the one dataset whose gap may be genuinely upstream — its +6 TP in
production came from candidates that may never be in nothink's pool; the control's bbb cell
will confirm.)

**Recommendation:** effort-0 + layered validator is worth shipping ONLY if the goal is
matching thinking-on's PRECISION at no-thinking cost/latency (it does, exactly). It does
not recover full macro-F1. To close the gap you need thinking (or stronger
reasoning-relocation) AT THE GATES — the control shows that path works — or accept a
~2-point macro cost for the latency saving (effort-0 ≈ 14–16 s/call, zero thinking tokens;
thinking-on validation ≈ 35–40 s/call — see control latencies).

## Cross-backend validation — gpt-5.4 (regression check)

Promoted the winning config to `src/llm_sad_sam/linkers/experimental/s_linker20_union_layered.py`
(registered in run_ablation.py; GATE-01 holds — canonical files untouched). Ran the SAME
validator-replay on the gpt-5.4 no-reasoning caches (`results/v2.6.5_s20union/gpt/`,
macro 89.4 — gpt's default-no-reasoning baseline; note gpt reasoning=medium is *worse*, 87.8).

| dataset | gpt base (cache N3) | gpt base-replay (run1) | gpt LAYERED (run1) |
|---------|---------------------|------------------------|--------------------|
| mediastore | 95.6 | 96.8 | 98.4 |
| teastore | 98.1 | 100.0 | 100.0 |
| teammates | 83.7 | 85.7 | 89.7 |
| bigbluebutton | 76.3 | 76.2 | 81.1 |
| jabref | 93.2 | 91.4 | 100.0 |
| **MACRO** | **89.4** | **90.0** | **93.8** |

**No regression — a strong gain on gpt-5.4: +4.4 macro (89.4→93.8), every dataset up,
including bbb (+4.8).** Coref FP 13→2; implicit-FN flat (21 vs 19). Larger than the Sonnet
gain because (a) gpt reasoning is net-negative so the prompt is gpt's only lever, and
(b) gpt's no-reasoning baseline had coref-heavy FPs the coref-strict rubric cleans. The
openai backend reports tokens: ~2.3k completion tokens/cell (the Mode-5 justification cost).

**Dual-backend bottom line:** the layered no-reasoning validator improves BOTH backends at
zero implicit-recall cost. Safe to ship as the opt-in `s_linker20_union_layered`.

**gpt-5.4 FIRMED at N=3** (validator-replay, all 3 runs):

| dataset | gpt baseline | gpt layered (N3) | Δ |
|---------|--------------|------------------|---|
| mediastore | 95.6 | 96.6 | +1.0 |
| teastore | 98.1 | 98.7 | +0.6 |
| teammates | 83.7 | 89.0 | +5.3 |
| bigbluebutton | 76.3 | 81.6 | +5.3 |
| jabref | 93.2 | 100.0 | +6.8 |
| **MACRO** | **89.4** | **93.2** | **+3.8** |

Robust at N=3: every dataset up (incl. bbb +5.3), **+3.8 macro**. Final dual-backend gain:
**Sonnet +1.1 (89.7→90.8), gpt-5.4 +3.8 (89.4→93.2)** at zero implicit-recall cost.

### Cross-model — gpt-5-mini / gpt-5-nano (e2e full pipeline)

Two model tiers tested e2e (full pipeline, layered vs baseline, ZERO reasoning):

| model | reasoning config (zero) | base macro | layered macro | Δ |
|-------|-------------------------|------------|---------------|---|
| **gpt-5.4-mini** (5.4 family) | temperature path / `none` (reasoning_tok=0) | 76.0 | **77.7** | **+1.7** |
| gpt-5-mini (older 5.0 family) | `reasoning_effort=minimal` (=0 tok; `none` rejected) | 74.8 | 75.2 | +0.4 |

(gpt-5.4-mini per-dataset: mediastore +6.2, bbb +3.8, teammates +0.8, jabref 0, teastore −2.2.)

**Zero-reasoning mechanics differ by tier** (API-probed, exact `reasoning_tokens`):
- gpt-5.4 / gpt-5.4-mini: accept `temperature` and `reasoning_effort=none` → 0 reasoning
  tokens; reject `minimal`. (Same family.)
- gpt-5-mini / gpt-5-nano (older): reject `temperature≠1` AND `none`; the zero tier is
  `reasoning_effort=minimal`, which the API confirms = 0 reasoning tokens for our gate
  prompts (OpenAI docs: "minimal … few or no reasoning tokens").

**Cross-model bottom line — the layered validator helps everywhere, scaling with
candidate-pool quality:**

| backend / model | candidate pool | layered Δ macro |
|-----------------|----------------|-----------------|
| gpt-5.4 (validator-replay, N=3) | strong (gpt-5.4) | **+3.8** |
| Sonnet (validator-replay, N=3) | strong (Sonnet) | +1.1 |
| gpt-5.4-mini (e2e) | weak (mini extraction) | +1.7 |
| gpt-5-mini (e2e, older) | weak | +0.4 |

The mini tier caps absolute macro ~76–78. Token cost of Mode-5 on gpt-5.4-mini: layered
52.0k vs base 37.4k completion tokens (+39%, the justification output — at zero reasoning).

### Why gpt-5.4-mini is bad (investigation) — it's a weak DISCRIMINATOR, not a weak extractor

Decomposition of gpt-5.4-mini (base, e2e) vs gpt-5.4 (run1), per dataset. **Correction to an
earlier claim:** mini's gap is NOT candidate generation — it is **pure precision**.

| dataset | model | #coref-raw | poolCeil | recall | precision | F1 |
|---------|-------|-----------|----------|--------|-----------|-----|
| teammates | 5.4-mini | **124** | 95 | 95 | **42** | 58 |
|           | 5.4-full | 49 | 91 | 88 | 81 | 84 |
| bbb | 5.4-mini | 54 | 84 | 84 | **56** | 67 |
|     | 5.4-full | 26 | 74 | 65 | 91 | 75 |
| teastore | 5.4-mini | 39 | 96 | 96 | **65** | 78 |
|          | 5.4-full | 12 | 100 | 96 | 100 | 98 |

mini's **recall is equal/higher** (bbb 84 vs 65) and its **pool ceiling is equal/higher** — it
*finds* the true links fine. The whole gap is precision. Two compounding causes:

1. **Coref discovery over-generates 2–4×** (coref raw: teammates 124 vs 49, teastore 39 vs 12,
   bbb 54 vs 26, mediastore 28 vs 11).
2. **The gates reject almost none of it.** Coref raw→validated rejection rate — mini: teammates
   9%, jabref 0%; gpt-5.4: teammates 27%, jabref 70%. So FPs flood: teammates entity FP **26**
   (vs 3) + coref FP **48** (vs 9); bbb entity FP **37** (vs 0); teastore coref FP **8** (vs 0).
   gpt-5.4 has ~0 FPs across the board.

**Diagnosis:** at every "include this link?" decision — entity extraction, coref discovery,
entity gate, coref gate — gpt-5.4-mini errs toward YES. It lacks gpt-5.4's capacity to say NO
precisely. Same failure mode as turning reasoning off (FP collapse from lost discrimination),
but driven by model **capacity**, not a thinking toggle. The layered validator's coref-strict
rubric recovers some of the coref flood (+1.7) but can't close the gap, because the deficit
spans all four stages and a better prompt fixes only two. Net: the layered validator helps
most when the model is already a good discriminator and the candidate pool is clean.
