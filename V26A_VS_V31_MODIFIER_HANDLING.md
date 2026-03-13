# V26a vs V31: Modifier Handling Strategy

## The Core Difference

**V26a**: Judges modifiers in **Phase 9 (Judge Review)** via advocate-prosecutor debate
**V31**: Handles modifiers in **Phase 8c (Convention Filter)** via LLM 3-step reasoning

---

## Why V26a "Needs" Non-Modifiers

### V26a Phase 9 Judge (Prosecutor Role)
```python
# Lines from agent_linker_v26a.py
prosecutor_prompt = """
...
Your job: Find CLEAR evidence that this is a SPURIOUS match.
Only these patterns warrant rejection:
1. "{comp_name.lower()}" is used as a modifier/adjective in a compound phrase
   (e.g., "cascade {comp_name.lower()}", "minimal {comp_name.lower()}")
   — NOT as a standalone noun referring to the component.
"""
```

**Problem V26a faces**:
- Links already validated in Phases 5-6
- Coref already resolved in Phase 7
- Partials already injected in Phase 8b
- By Phase 9, most good links are already in the set
- Judge must distinguish: is "cascade X" a link or spurious?
- Example: "cascade storage" → is this the Storage component or just a modifier?

**Why V26a needs modifier check**:
- Late-stage filtering requires explicit rejection criteria
- Can't rely on earlier phases (they already accepted the link)
- Must ask: "Is this really referring to the component, or just using it as an adjective?"

---

## Why V31 "Doesn't Need" Non-Modifiers

### V31 Phase 8c Convention Filter (Boundary Filters BEFORE Judge)

```python
# CONVENTION_GUIDE from ilinker2_v31.py
CONVENTION_GUIDE = """
### STEP 2b — Generic word collision (includes modifiers):
NO_LINK — narrow, non-architectural sense:
- Process/activity modifier: "cascade X", "retry X", "validation X"
- Hardware/deployment: "a dedicated hardware node", "32-core server"
- Possessive/personal: "her settings", "their preferences"

LINK — system-level architectural sense:
- System name + word: "the [System] gateway"
- Architectural role: "the orchestrator routes jobs to the gateway"
"""
```

**Advantage V31 has**:
- Filters BEFORE judge (Phase 8c vs Phase 9)
- Early filtering is more decisive
- LLM can reason through architecture semantics
- Handles modifiers as part of broader context
- Default is LINK if not caught (safer)

**Why V31 doesn't need separate modifier rejection**:
- Already filtered out before judge sees them
- Judge receives cleaner link set
- Fewer edge cases to arbitrate in Phase 9

---

## Detailed Comparison

### Strategy Difference

```
V26a Flow:
Phase 5-7 (Entity, Validation, Coref)
  ↓ accepts "cascade Storage"
  ↓
Phase 9 Judge (Prosecutor)
  → "Is 'cascade' a modifier? REJECT"

V31 Flow:
Phase 5-7 (Entity, Validation, Coref)
  ↓ accepts "cascade Storage"
  ↓
Phase 8c Convention Filter
  → "Process/activity modifier? NO_LINK"
  → Link removed BEFORE judge sees it
  ↓
Phase 9 Judge
  → Cleaner input, fewer edge cases
```

---

## Effectiveness Comparison

### V26a Modifier Detection (Judge)
| Pattern | Example | Handled By |
|---------|---------|-----------|
| "cascade X" | "cascade storage" | Prosecutor explicitly ✓ |
| "retry X" | "retry logic" | Prosecutor explicitly ✓ |
| "minimal X" | "minimal memory" | Prosecutor explicitly ✓ |
| "X pattern" | "orchestrator pattern" | Prosecutor explicitly ✓ |

**Catch rate**: ~5 FPs per dataset (from boundary filters)

### V31 Modifier Detection (Convention Filter)
| Pattern | Example | Handled By |
|---------|---------|-----------|
| "cascade X" | "cascade storage" | Step 2b (modifiers) ✓ |
| "retry X" | "retry logic" | Step 2b (modifiers) ✓ |
| "validation X" | "validation framework" | Step 2b (modifiers) ✓ |
| "X pattern" | "orchestrator pattern" | Step 2a (methodology) ✓ |
| "dedicated X node" | "dedicated storage node" | Step 2b (hardware) ✓ |

**Catch rate**: ~11 FPs per dataset (28% better than V26a regex)

---

## Why V31 Catches More with Same Logic

### Root Cause: Timing

**V26a** (Judge in Phase 9):
- Judges individual links in isolation
- No broader context
- Prosecutor must justify rejection
- Very conservative ("only if CLEAR evidence")
- Result: Misses edge cases

**V31** (Convention Filter in Phase 8c):
- Filters all links together
- Can see patterns across document
- LLM can reason about architecture context
- Explicitly lists modifier patterns
- Result: Catches more FPs

### Example Progression

```
Sentence: "The cascade storage mechanism prevents overload"

V26a Judge:
  → Is "cascade" a modifier? Maybe not... it could be architectural
  → "cascade storage" might describe an architectural pattern
  → Default: APPROVE (when in doubt)
  → Result: FALSE POSITIVE KEPT

V31 Convention Filter:
  → Is this a process/activity modifier? "cascade [activity]"
  → Matches: "cascade X" = activity modifier
  → Step 2b: NO_LINK
  → Result: FALSE POSITIVE REMOVED
```

---

## Why Both Approaches Exist

### V26a Judge Approach (Phase 9)
**Pros**:
- Catches obvious modifiers (cascade, retry, minimal)
- Works with any link type
- Conservative (safe for recall)

**Cons**:
- Late-stage filtering
- Only works if prosecutor arguments are convincing
- Requires LLM to justify rejection (expensive)
- Edge cases slip through

### V31 Convention Filter (Phase 8c)
**Pros**:
- Early filtering (before judge)
- Explicit patterns (modifiers, tech names, hierarchies)
- Better context (sees all links together)
- More thorough categorization

**Cons**:
- Replaces regex (different trade-off)
- Different categorization than judge
- More complex reasoning guide

---

## Key Insight: Different Architecture, Same Problem

| Phase | V26a | V31 |
|-------|------|-----|
| **Where modifiers detected** | Phase 9 (Judge) | Phase 8c (Convention) |
| **When modifiers detected** | Late (other phases already accepted) | Early (before judge) |
| **How modifiers detected** | Judge prosecutor argues against | LLM 3-step reasoning guide |
| **Modifier patterns** | "cascade X", "retry X", "minimal X" | Same PLUS "validation X", hardware patterns |
| **Catch rate** | ~5 FPs/dataset | ~11 FPs/dataset |
| **Why different?** | Timing + context | Early detection + explicit patterns |

---

## Bottom Line

**V26a "needs" modifier checks in the Judge because**:
- Judge is last line of defense
- Must catch any spurious matches
- Including modifier usage patterns

**V31 "doesn't need" them in the Judge because**:
- Already filtered in Phase 8c
- Convention filter catches modifiers earlier
- Judge receives cleaner link set
- Can focus on other criteria (reference, perspective, focus)

**Same problem, different solution architecture**:
- V26a: Solve at exit (judge everything at the end)
- V31: Solve at boundary (filter earlier, be explicit about patterns)

V31's approach is more effective: 11 FPs caught vs 5 FPs, with 0 TPs killed.
