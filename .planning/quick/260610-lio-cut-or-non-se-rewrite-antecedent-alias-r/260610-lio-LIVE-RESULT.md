# Live behavioral confirmation — ANTECEDENT_ALIAS_RULES few-shot (2026-06-10)

N=3 TeaMmates (coref-sensitive), interleaved runs (matched conditions), single variable = the few-shot.

| variant | F1 [range] | coref-FP [per run] | spread |
|---|---|---|---|
| s20 (control, SE few-shot) | 0.837 [0.833–0.840] | [12, 11, 11] | range 1 |
| aliasA (few-shot CUT) | 0.860 [0.813–0.893] | [14, 9, 4] | range 10 |
| aliasB (hardware rewrite) | 0.829 [0.820–0.840] | [14, 13, 9] | range 5 |

## Verdicts
- **aliasB (non-SE hardware rewrite): SAFE.** F1 + coref-FP indistinguishable from control. Removes SE-domain flavor (generality win for the paper) at no behavioral cost. **Recommended ship.**
- **aliasA (cut): mean-safe but VARIANCE-UNSAFE.** Mean F1 fine (nominally higher), but cutting the few-shot blew coref-FP spread from range 1 → range 10. The few-shot anchors STABILITY, not mean accuracy. Do not cut.

## Answer to "does cutting prompt increase LLM variance?"
This is the cleanest controlled test in the v2.6.4/v2.6.5 investigation (matched arch, interleaved, single variable):
**YES for a load-bearing few-shot** — cutting ANTECEDENT_ALIAS_RULES' examples raised coref-FP run-to-run spread ~10× (range 1→10) while leaving the mean ~unchanged. The earlier cross-variant meta-analysis showed "no robust effect" because it was confounded (different cuts/architectures); the single-variable test reveals the effect.
Caveat: N=3; control was unusually tight this batch. Direction is clear but magnitude needs N≥6 to pin. Mechanism: few-shots reduce decision drift on ambiguous coref calls.

## Net recommendation
Ship aliasB (hardware few-shot) if generality-neutral wording is desired; keep the control otherwise. Do NOT cut (aliasA) — it trades stability for nothing.
