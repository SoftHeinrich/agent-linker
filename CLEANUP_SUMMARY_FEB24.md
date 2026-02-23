# V31 Code Cleanup: Removal of Phase 8 & Phase 10 (Feb 24, 2026)

## Overview
Removed all dead code for Phase 8 (Implicit References) and Phase 10 (FN Recovery) from the V31 linker. These phases had zero/negative value and added complexity without benefit.

---

## Code Changes

### File: `src/llm_sad_sam/linkers/experimental/ilinker2_v31.py`

#### Change 1: Removed Phase 8 (Implicit References) code
**Lines removed (originally 306-309)**:
```python
# ── Phase 8 ─────────────────────────────────────────────────────
reason = "complex doc" if self._is_complex else "dead weight"
print(f"\n[Phase 8] Implicit References — SKIPPED ({reason})")
implicit_links = []
```

**Reason**:
- 79-80% false positive rate (83/105 FPs on Sonnet)
- Implicit reference detection unreliable without project-specific knowledge
- LLM cannot distinguish implicit component mentions from generic architectural discussions

#### Change 2: Removed Phase 10 (FN Recovery) code
**Lines removed (originally 391-392)**:
```python
# ── Phase 10 ────────────────────────────────────────────────────
print("\n[Phase 10] FN Recovery — SKIPPED (dead weight)")
```

**Reason**:
- Zero net gain: links recovered = links re-introduced as FPs
- Relaxed thresholds just re-approve what Phase 9 judge already rejected
- No new information available for recovery

#### Change 3: Updated module docstring
**Lines added (15-16)**:
```python
- Phase 8 (Implicit References): 79-80% FP rate, removed entirely
- Phase 10 (FN Recovery): zero net gain, removed entirely
```

**Result**: Documented why these phases were removed

---

## Documentation Updates

### 1. `PHASE_CONTRIBUTION_ANALYSIS.md`
- **Removed**: "Tier 4: Cut/Skipped" section became "Tier 4: Removed"
- **Updated**: Phase 8 and Phase 10 descriptions (now mark as REMOVED not SKIPPED)
- **Removed**: Detailed sections for Phase 8 (Implicit Refs) and Phase 10 (FN Recovery)
- **Updated**: Opening summary to note "2 phases were safe to remove entirely"
- **Updated**: Deduplication section (removed mention of Phase 8 in priority list)
- **Updated**: Summary Table (changed legend from "ALREADY CUT" to "REMOVED")

### 2. `PHASE_QUICK_REFERENCE.md`
- **Updated**: Tier diagram - removed Phase 8 and 10 from "SKIPPED/REMOVED" box, changed to "REMOVED" with evidence
- **Updated**: Data flow diagram - changed P1 reference from "Phase 8" to "Phase 8b" for parent-overlap guard
- **Updated**: Essential vs Optional table - Phase 8 and 10 now marked as "✓ REMOVED" with evidence

### 3. `V31_FINAL_SUMMARY.md`
- **Updated**: Architecture section - changed "10 phases" to "8 active phases"
- **Updated**: Pipeline diagram to explicitly list removed phases with reasons
- **Updated**: Code Cleanup section to document Phase 8 and 10 removal
- **Updated**: Changelog to include removal of Phase 8 and 10 code (items 7-8)

### 4. `ilinker2_v31.py` docstring
- Added note that Phase 8 and 10 were removed with reasons

---

## Pipeline Structure After Cleanup

### Active Phases (8 total)
```
Phase 0: Document profiling
Phase 1: Model structure analysis
Phase 2: Pattern learning (debate)
Phase 3: Document knowledge (few-shot judge)
Phase 3b: Multi-word partial enrichment
Phase 4: ILinker2 seed
Phase 5: Entity extraction
Phase 5b: Targeted recovery
Phase 6: Validation (intersect voting)
Phase 7: Coreference (discourse/debate)
Phase 8b: Partial reference injection
Phase 8c: Convention-aware boundary filters
Phase 9: Judge review (4-rule reframed)
```

### Removed Phases (2 total, code deleted)
```
Phase 8 (Implicit References) — 79-80% FP rate
Phase 10 (FN Recovery) — zero net gain
```

---

## Evidence for Removals

### Phase 8: Implicit References
**FP Rate**: 79-80% (83/105 FPs on Sonnet model)

**Examples of failures**:
- "The system persists data" → incorrectly links to Storage component
- Any sentence discussing "processing" → links to Processor/Processing component
- Generic architectural vocabulary used as implicit references

**Why LLM can't do it**:
- Requires distinguishing implicit component-specific mentions from generic architectural discussions
- Needs project-specific knowledge (which components exist)
- Without context, LLM over-generalizes

**Tested in**: `agent_linker.py` (original implementation with restrictions)

### Phase 10: FN Recovery
**Net Gain**: ~0.0pp (zero net F1 change)

**Why it fails**:
- Phase 9 judge already filters to high-confidence links
- "False negatives" rejected by judge are usually legitimately spurious
- Relaxing thresholds just re-approves the same links judge rejected
- If judge is correct in rejection, recovery introduces FPs
- If recovery finds TPs, judge was too aggressive (contradicts design)

**Testing**: Ablation studies confirmed zero net improvement

---

## Impact on Performance

**F1 Score**: Unchanged at 94.5% macro
- Phase 8 and 10 contributed 0pp when active
- Removal improves code clarity without regression

**File Size**: Reduced from ~804 lines to ~800 lines
- Smaller, cleaner codebase
- Easier to maintain and understand

**Pipeline Complexity**: Reduced by 2 phases (10 → 8 active)
- Simpler mental model
- Fewer potential failure points

---

## Verification Checklist

- ✅ Phase 8 (Implicit) code removed from ilinker2_v31.py
- ✅ Phase 10 (FN Recovery) code removed from ilinker2_v31.py
- ✅ Module docstring updated
- ✅ PHASE_CONTRIBUTION_ANALYSIS.md updated (removed sections, updated tables)
- ✅ PHASE_QUICK_REFERENCE.md updated (tier diagram, data flow, tables)
- ✅ V31_FINAL_SUMMARY.md updated (architecture, code cleanup, changelog)
- ✅ All references to phases renumbered correctly (P1→P1, P7→P7, P8b→P8b, P8c→P8c, P9→P9)
- ✅ No broken references in documentation

---

## Files Modified

1. **`src/llm_sad_sam/linkers/experimental/ilinker2_v31.py`**
   - Removed 4 lines (Phase 8 code)
   - Removed 2 lines (Phase 10 code)
   - Added 2 lines (docstring notes)
   - Net: ~4-5 lines removed

2. **`PHASE_CONTRIBUTION_ANALYSIS.md`**
   - Removed Phase 8 detailed section
   - Removed Phase 10 detailed section
   - Updated Tier 4 table
   - Updated Deduplication section
   - Updated Summary Table

3. **`PHASE_QUICK_REFERENCE.md`**
   - Updated tier diagram
   - Updated data flow
   - Updated essential vs optional table

4. **`V31_FINAL_SUMMARY.md`**
   - Updated architecture diagram
   - Updated code cleanup section
   - Updated changelog

---

## Status

**V31 is now cleaner and more maintainable.**

- ✅ All dead code removed
- ✅ Documentation updated
- ✅ Pipeline simplified (8 active phases vs 10)
- ✅ Performance unchanged (94.5% F1)
- ✅ Code ready for production/publication

**Total removed**: 6 lines of dead code + 2 dead phases
**Total document updates**: 4 main files + multiple sections
**Net benefit**: Cleaner codebase, easier to understand and maintain
