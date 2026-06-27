---
created: 2026-06-02T14:43:25.166Z
title: Improve prompts_v4_axiom — three root-cause FP fixes for v2.6.1
area: tooling
files:
  - src/llm_sad_sam/linkers/experimental/prompts_v4_axiom.py
  - scripts/ablation_validation_rules.py
  - results/voyager_v5/ablation_validation_rules.json
---

## Problem

After ablation study (2026-06-02), B-variant axiom (+counterparts, 88.99% macro) is committed.
Remaining 16 TM FPs and 11 BBB FPs trace to three distinct root causes, each addressable by
a different axiom slot in prompts_v4_axiom.py.

**Root Cause A — Tier/platform alias (6 FPs: UI×4 + GAE Datastore×2)**
"front-end" alias matched to UI in sentences describing the generic front-end tier ("conceptual
package representing the front-end", "front-end files tested with Jest"). "GAE" alias matched to
GAE Datastore when describing the Google App Engine platform ("GAE production environment").
Pattern: alias names a BROADER TIER or PLATFORM that contains the component, not the component itself.
PCM is flat — no containment hierarchy. Cannot fix via model reasoning; must fix via axiom.

Proposed slot: **DOC_KNOWLEDGE_JUDGE_RULES** (fix at source — prevent tier aliases from entering alias set)
> "An alias is also invalid when it names an architectural tier or technology platform that
> encompasses multiple elements, because it identifies a grouping rather than a single named unit."

**Root Cause B — Code-path prefix leakage (4 FPs: Logic S85, Storage S125/S136, Common S127)**
"logic.api", "storage.entity", "common.datatransfer", "Db classes" — component name appears as
prefix in Java package path or class name. ENTITY_EXTRACTION_RULES already says "Exclude when
the name appears only inside a code-level path" but leaks through because model reasons
"storage.entity is related to Storage → include."

Proposed slot: **ENTITY_EXTRACTION_RULES** (strengthen existing rule)
> Add: "even if the compound identifier is semantically related to the component."

**Root Cause C — Functional alias as workflow subject (BBB Pres.Conversion cluster, 5+ FPs)**
"conversion process" is a valid alias for Presentation Conversion, but sentences like "Files go
through a conversion process", "The conversion process sends progress messages" describe the
WORKFLOW/ACTIVITY, not the component as an architectural unit. The "or activity" variant (A) was
tested and did NOT fix these — "The conversion process sends messages" looks architecturally
participatory even to the model.

Best proposed slot: **SEED_DISAMBIGUATION_RULES** (functional-alias heuristic)
> "When the matched alias is a functional or process description (a phrase describing what the
> component does), apply an additional check: if removing the alias from the sentence still leaves
> an accurate description of a process step or activity, classify as OTHER — the sentence describes
> the activity, not the component. Classify as COMPONENT only when the sentence clearly treats the
> alias as the name of a specific architectural unit."

**Also: negative interaction between A and B variants (already committed)**
"or activity" + "including counterparts" combined gave -3.2pp on TM. B alone (+1.98pp) is correct.
This interaction rules out both changes together in VALIDATION_RULES — document for thesis.

## Solution

Test each root cause fix as isolated ablation (same `scripts/ablation_validation_rules.py` harness):
1. Patch DOC_KNOWLEDGE_JUDGE_RULES for Cause A → run 5 projects empty bank → check TM FP delta
2. Patch ENTITY_EXTRACTION_RULES for Cause B → check TM FP delta
3. Patch SEED_DISAMBIGUATION_RULES for Cause C → check BBB FP delta
4. If all improve or neutral individually → test combined

Success criterion: TM macro improves from 82.26% without regressing MS/TS/JAB.
Expected ceiling without bilateral fix: ~85% macro (BBB HTML5 disambiguation still ~77%).

## Context

Ablation harness: `scripts/ablation_validation_rules.py` + `scripts/run_ablation_variant.py`
All 5-project results cached in `results/voyager_v5/cache/` keyed by axiom hash.
Current B-variant axiom hash: 61e038 (confirmed).
Baseline before any axiom changes: bcae0e (87.01% macro).
B-variant current: 61e038 (88.99% macro).
