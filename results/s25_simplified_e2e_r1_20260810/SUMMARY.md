# s25 post-simplification five-project E2E — 2026-08-10

Three runs after the complexity round (`results/s25_complexity_audit/README.md`):
slim evidence bundle (no constant rationale line, anchors via the lenient name
primitive) and no `antecedent_via_alias` surface. Same model and settings
throughout: `gpt-5.6-terra`, OpenAI backend, reasoning effort `none`.

| | pre-change (N=1) | after design pilots (N=3) | after simplification (N=3) |
|---|---|---|---|
| macro F1 | 94.2 | 94.7 ± 0.8 | **96.8 ± 0.1** |
| pooled F1 | 91.6 | 93.6 ± 1.2 | **95.5 ± 0.1** |
| TP | 179 | 179.7 | **182.3** |
| FP | 17 | 9.3 | **4.3** |
| FN | 16 | 15.3 | **12.7** |

Per project, mean F1 over the three runs:

| Project | after simplification | after design pilots | FP per run now |
|---|---|---|---|
| mediastore | 98.4 | 98.4 | 0 / 0 / 0 |
| teastore | 98.7 | 98.1 | 0 / 0 / 0 |
| teammates | 95.8 | 94.0 | 1 / 1 / 3 |
| bigbluebutton | 91.0 | 88.4 | 3 / 3 / 2 |
| jabref | **100.0** | 94.7 | 0 / 0 / 0 |

No project regressed. Run-to-run spread collapsed as well: macro F1 sd went from
0.8 to 0.1, and mediastore, teastore and jabref returned identical link sets on
every run.

For reference, the S24 role-orchestrator this variant descends from — which
carries an LLM controller, an ambiguity map, alias scopes and two extraction
samples — reports 182 TP / 8 FP / 13 FN, macro 96.07, pooled 94.55. The
simplified s25 reaches 182.3 TP / 4.3 FP / 12.7 FN, macro 96.8, pooled 95.5 with
none of those four mechanisms.

Every change between these columns was adopted only after a paired,
five-run-per-side pilot on a single stage; nothing here was tuned against these
end-to-end numbers.

## Correction (same day, after six runs of identical code)

The `± 0.1` above is not this pipeline's spread. Three later runs of code
verified identical to this configuration (`s25_micro_reverted_confirm_*`; 0
predicate flips over 3697 name/sentence pairs, and 0 mention-label mismatches
over the 170 judged cases recorded in r1's own trace) came in at macro F1
96.2 / 95.8 / 96.2 and macro F2 94.8 / 94.7 / 95.2.

Pooled over all six runs: **macro F1 96.42 ± 0.42, macro F2 95.38 ± 0.58, TP
179–183, FP 4–6**. Quote that band. Two projects carry all of it — teammates
(F1 90.7–97.3) and bigbluebutton (F1 84.0–93.3); mediastore, teastore and jabref
are stable to the link.
