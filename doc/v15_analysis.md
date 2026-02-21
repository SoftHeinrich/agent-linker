# V15 Analysis: Why It Underperforms + Data Leakage

## Performance Summary

V15 achieves **88.8-89.0% macro F1** across 2 runs, vs V6 best of **90.5%**. Per-dataset breakdown:

| Dataset | V15 Run1 | V15 Run2 | V6 Best |
|---------|----------|----------|---------|
| mediastore | 86.7% | 86.7% | 96.8% |
| teastore | 86.8% | 86.8% | 94.1% |
| teammates | 91.7% | 91.7% | 86.2% |
| bigbluebutton | 84.1% | 84.1% | 83.2% |
| jabref | 94.7% | 94.7% | 91.9% |

V15 **fixed Phase 1** (deterministic classification — identical both runs) and improved teammates/BBB/jabref, but **regressed mediastore (-10pp) and teastore (-7pp)**.

## Why V15 Doesn't Work

### 1. AMBIGUOUS_WORDS List Is Too Broad (line 53)

The list includes `client, storage, facade, cache, server, web, model, view, controller, store` — but these are legitimate architectural component names in several datasets:

- **mediastore**: `Cache`, `Facade` classified as ambiguous → reduced validated links, Phase 9 scrutinizes them harder
- **teastore**: All components are multi-word or CamelCase so they dodge the list, but the Phase 6 validation becomes stricter for zero ambiguous components (different code path)
- **teammates**: `Client`, `Storage` classified ambiguous → Phase 9 rejects valid links for these

### 2. Phase 9 Judge Still Rejects Valid Links

From the ablation results:
- mediastore: Rejects `DB` (s12, s33) and `Reencoding` (s15, s37) — 3 FP but all high-confidence validated/transarc links
- teastore: Rejects `Registry` (s39, s40) as FP — these are validated links about heartbeat/re-registration
- teammates: 8 FP, all transarc — `Logic` (s22, s79, s117, s119), `Client` (s4), `E2E` (s17, s188), `Test Driver` (s173)

The TransArc immunity fix (line 1287) should protect transarc links, yet teammates shows 8 transarc FP. These are links the system **produced** that are false positives — not links it rejected. So the issue is **over-generation**, not over-rejection.

### 3. Mediastore Regression: -10pp F1

V15 mediastore: P=89.7%, R=83.9%, F1=86.7% (26tp, 3fp, 5fn)
- 5 FN all have `name_in_text: false` and `transarc_had: false` — these are implicit references that V15's Phase 7/8 discourse resolution doesn't catch
- Components: `MediaAccess` (s25), `DB` (s33), `FileStorage` (s33, s35, s36) — all indirect references
- V6 best got these via more aggressive coreference/injection

### 4. Teastore Regression: -7pp F1

V15 teastore: P=88.5%, R=85.2%, F1=86.8% (23tp, 3fp, 4fn)
- 4 FN: `WebUI` (s6, s8) and `Persistence` (s23, s24) — all `name_in_text: false`
- Phase 7 coreference found 5 links but missed these WebUI/Persistence references
- Phase 5b targeted recovery found 9 links for unlinked sub-recommenders but didn't help with the missing WebUI/Persistence links

### 5. Remaining LLM Variance in Later Phases

Despite deterministic Phase 1, run-to-run differences still appear:
- teastore: Run1 produced 29 links, Run2 produced 26 links (Phase 5 found different counts)
- teammates: Run1 produced 69 links, Run2 produced 63 links
- Phases 5, 6, 7 all show variance from LLM non-determinism

## Data Leakage Concerns

V15 has **significant data leakage** in three prompt templates that use benchmark-specific component names as "general" examples:

### 1. Phase 3 Enriched Prompt (lines 486-510)

```
Examples use: "Kurento Media Server (KMS)", "FSESL",
"GAE Datastore", "HTML5 Client/Server"
```

These are actual BBB and teammates component names. Any LLM that has seen the benchmark data during training will have an unfair advantage.

### 2. Phase 6 Validation Prompt (lines 835-853)

```
Examples use: "Logic", "Client", "Storage" with specific
behavioral patterns like "It contains minimal logic beyond
what is directly relevant to CRUD operations"
```

These are verbatim teammates component names with behavioral descriptions that match the teammates SAD text.

### 3. Phase 9 Judge Prompt (lines 1362-1397)

```
Examples: "Logic handles incoming requests",
"The Client component renders the UI",
"It delegates to Storage for persistence"
```

Again teammates-specific component names and behaviors used as "general" examples.

### Impact

The leakage likely **inflates teammates scores** (91.7% — highest per-dataset F1) while not helping mediastore/teastore where the examples don't match. This makes the overall F1 look better than it would be on truly unseen data.

### Recommendation

Replace all benchmark-specific examples with synthetic/generic ones like:
- "AuthenticationService" instead of "FSESL"
- "DataProcessor" instead of "Logic"
- "UserInterface" instead of "Client"

## Summary of Issues

| Issue | Impact | Fix Complexity |
|-------|--------|---------------|
| AMBIGUOUS_WORDS too broad | -10pp mediastore, -7pp teastore | Low — trim list |
| Missing implicit refs (Phase 7/8) | 5 FN mediastore, 4 FN teastore | Medium — improve discourse resolution |
| Over-generation for ambiguous names | 8 FP teammates | Medium — tighten Phase 5 extraction |
| Data leakage in prompts | Inflated teammates score | Low — replace examples |
| LLM variance in Phases 5-9 | ±3-6 links per run | Hard — fundamental LLM issue |
