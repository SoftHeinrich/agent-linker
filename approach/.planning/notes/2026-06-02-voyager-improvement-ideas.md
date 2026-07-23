---
date: "2026-06-02 11:30"
promoted: false
---

## Voyager improvement ideas (from train/test ceiling analysis)

### 1. Generic disambiguation: aggregate-then-decide
**Current**: per-candidate LLM call in isolation (line 857 s_linker14_voyager.py)
**Fix**: group ALL lowercase mentions of an ambiguous component, show all anchor sentences (confirmed capitals) + all candidates in one call. Ask "given Logic is used as component name in X sentences, which of these Y lowercase mentions also refer to it?" Document-level consistency is much harder to hallucinate wrong.

### 2. Noise floor calibration for probation gate
**Current**: probation commits if delta > 0 — can't distinguish ±3pp variance from real gain
**Fix**: before first distillation pass, run baseline twice on empty bank, record noise_delta = |run1 - run2|. Set commit threshold to delta > noise_floor. Or: run probation twice (with + without patterns, same bank state) for paired comparison.

### 3. ILinker4 coref inflation via role_ref_pat
**Bug**: ILinker4's richer alias generation feeds `_classify_specific_terminals` → more terminal words → `role_ref_pat` matches more sentences → anaphoric_sents balloons → 41 coref links for TM vs 20 with ILinker3. Review terminal classification: are the extra terminals actually specific enough? The inflation is indirect (ILinker alias richness → coref expansion).

### 4. BBB alias coverage gap (root cause of 17 FNs)
ILinker seeds only 65% of BBB gold links. Entity extraction can only find aliases ILinker DID discover (global aliases injected into entity prompt). Coref requires direct-mention antecedent. All three recovery paths are gated on ILinker's alias discovery. The 17 BBB FNs are components whose indirect references (role descriptions like "the signaling bridge" → FSESL) ILinker never aliased.
**Fix direction**: richer role-description alias extraction pass — explicitly ask ILinker to generate aliases from component-role descriptions ("what role does this component play in the architecture?"), not just lexical variants.

### 5. Coref is net-negative for BBB: dataset-configurable threshold
BBB: coref adds ~2 TP, 5 FP. Net −3 FP damage. Should be disabled or use much stricter threshold for datasets with non-ambiguous component names (all BBB components are proper nouns/abbreviations).
**Fix**: add per-dataset coref aggressiveness flag, or auto-detect: if model_knowledge.ambiguous_names is empty/small → reduce coref to pronoun-only (no role_ref_pat).

### 6. Train/test error type mismatch (structural)
TM (train) errors = over-production (FP dominant, 15 FP, 5 FN). BBB (test) errors = under-recall (FN dominant, 9 FP, 17 FN). Every axiom distilled from TM is a precision rule. Applied to BBB these rules are neutral-to-harmful. Voyager has no mechanism to learn recall rules because TM doesn't have recall failures. **The fix requires BBB in training set**, or a separate "recall oracle" that identifies FN patterns.

### 7. Test ceiling estimate
BBB ceiling: ~84-87% (fix FPs + recover 5-8 FNs). JAB: 100% already hit. Test macro ceiling: ~90-93%.
Train macro ceiling: ~93-94% (TM hard cap ~89% due to generic name wall + coref GPT-5.4 limit).
