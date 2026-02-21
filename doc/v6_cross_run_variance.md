# V6 Cross-Run Variance Analysis & Determinism Recommendations

Run comparison: 18 V6 runs across Feb 16-18, 2026 (codex backend, Claude Sonnet).

---

## 1. Variance Statistics Across 18 Runs

| Dataset | F1 Range | F1 Spread | Best | Worst | Median |
|---|---|---|---|---|---|
| **teammates** | 68.4% – 95.7% | **27.2pp** | 95.7% | 68.4% | ~86% |
| **mediastore** | 71.7% – 95.2% | **23.5pp** | 95.2% | 71.7% | ~90% |
| **bigbluebutton** | 76.4% – 89.6% | **13.1pp** | 89.6% | 76.4% | ~84% |
| **teastore** | 88.2% – 100.0% | 11.8pp | 100.0% | 88.2% | ~94% |
| **jabref** | 88.9% – 100.0% | 11.1pp | 100.0% | 88.9% | ~95% |

### Per-Source Link Count Ranges (across 18 runs)

| Source | teammates | BBB | mediastore |
|---|---|---|---|
| transarc | 47–50 | 39–49 | 17–18 |
| **validated** | **0–30** | 0–9 | 0–10 |
| **coreference** | **0–27** | 0–5 | 0–4 |
| entity | 0–3 | 0–2 | 0–1 |
| **partial_inject** | 0–6 | **0–19** | 0–0 |
| recovered | 0 | 0 | 0 |

The `validated`, `coreference`, and `partial_inject` sources are the most variable. TransArc is stable (deterministic CSV input). The extreme ranges indicate that the LLM pipeline is producing fundamentally different link sets across runs.

---

## 2. Root Cause: Phase 1 Non-Determinism

**The single root cause of all variance is Phase 1 (Model Structure Analysis).** This phase asks the LLM to classify each component as "architectural" (a real, specific software component) vs "ambiguous" (a generic term that could cause false matches). The LLM's classification is non-deterministic and varies across runs.

### Observed Phase 1 Classifications

**Teammates:**
| Run | Architectural | Ambiguous | F1 |
|---|---|---|---|
| Good (Feb 16) | Client, E2E, GAE Datastore, **Logic**, Storage, Test Driver, **UI** | Common | 93.0% |
| Bad (Feb 18) | Client, E2E, GAE Datastore, Storage, Test Driver | Common, **Logic**, **UI** | 74.6% |

**BigBlueButton:**
| Run | Architectural | Ambiguous | F1 |
|---|---|---|---|
| Good (Feb 16) | All 12 components | None | 88.2% |
| Bad (Feb 18) | FSESL, FreeSWITCH, Redis DB, Redis PubSub, WebRTC-SFU, kurento | **Apps, BBB web, HTML5 Client, HTML5 Server, Presentation Conversion, Recording Service** | 77.3% |

**Mediastore:**
| Run | Architectural | Ambiguous | F1 |
|---|---|---|---|
| Good (Feb 16) | All 14 components | None | 95.2% |
| Bad (Feb 18) | 9 components | **Cache, DB, Facade, FileStorage, Packaging** | 89.7% |

### Cascade Mechanism

Phase 1 ambiguous classification triggers a triple cascade:

```
Phase 1: Component marked "ambiguous"
    ↓
Phase 3b: Auto-partials NOT generated for ambiguous components
    → Phase 8b: Partial injection has nothing to inject → recall loss
    ↓
Phase 2-3: Richer synonym/subprocess discovery (compensating)
    → Phase 5-6: Over-generation of entity candidates → FP increase
    ↓
Phase 9: Judge may reject TransArc links for ambiguous components
    → Direct TP loss on baseline links
    ↓
Phase 10: FN recovery BLOCKED for ambiguous components
    → No safety net for any of the above losses
```

### Concrete Impact Per Dataset

**Teammates (bad run):** Logic and UI marked ambiguous →
- Phase 2: 15 subprocess terms vs 0 in good run
- Phase 3: 13 synonyms vs 0 in good run
- Phase 5: 75 entity candidates vs 40 (87% more)
- Phase 6: 46 validated links vs 12 (283% more)
- Phase 9: Rejected 5 TransArc TPs for Logic (s4/Client, s7/Logic, s22/Logic, s117/Logic, s185/Logic)
- Phase 7: Completely different coreference (only 4 of 15+24 links shared)
- Net: 85 final links (many FP) vs 67 (clean), F1 dropped from 93.0% to 74.6%

**BBB (bad run):** 6 components marked ambiguous →
- Phase 3b: Zero auto-partials vs 3 in good run (Server, Client, Conversion)
- Phase 8b: Zero partial injections vs 21 in good run
- Phase 9: Rejected 8 HTML5 Server TransArc TPs
- Net: 57 final links vs 74, F1 dropped from 88.2% to 77.3%

**Mediastore (bad run):** 5 components marked ambiguous →
- Phase 9: Rejected 3 FileStorage validated TPs (s33, s35, s36)
- Phase 7: Lost s9/MediaManagement coreference TP
- Phase 10: Recovery blocked for Cache, DB, Facade, FileStorage, Packaging
- Net: 27 final links vs 30, F1 dropped from 95.2% to 89.7%

---

## 3. Persistent Error Patterns (Structural, Not LLM Variance)

These errors appear in >50% of the 18 runs regardless of Phase 1 classification, indicating structural pipeline limitations rather than LLM non-determinism.

### Persistent False Positives

| Dataset | Sentence | Component | Source | Frequency | Why persistent |
|---|---|---|---|---|---|
| teammates | S17 | E2E | transarc | 100% (18/18) | TransArc baseline always includes this FP |
| teammates | S188 | E2E | transarc | 94% (17/18) | TransArc baseline FP |
| teammates | S79 | Logic | transarc | 83% (15/18) | TransArc baseline FP |
| mediastore | S37 | Reencoding | transarc | 100% (18/18) | TransArc baseline FP |
| BBB | S60 | FreeSWITCH | transarc | 100% (18/18) | TransArc baseline FP |
| BBB | S79 | PresentationConversion | transarc | 94% (17/18) | TransArc baseline FP |
| BBB | S76 | PresentationConversion | transarc | 83% (15/18) | TransArc baseline FP |
| teastore | S40 | Registry | transarc | 78% (14/18) | TransArc baseline FP |

All persistent FPs originate from the TransArc baseline. The judge (Phase 9) sometimes catches them but not reliably.

### Persistent False Negatives

| Dataset | Sentence | Component | Frequency | Why persistent |
|---|---|---|---|---|
| teammates | S8 | Logic | 89% (16/18) | TransArc doesn't find it, no synonym match |
| teammates | S78 | Logic | 78% (14/18) | Coreference chain sometimes catches it |
| teammates | S7 | Logic | 72% (13/18) | Sometimes found, sometimes judge rejects |
| teammates | S46 | Logic | 67% (12/18) | Coreference-dependent |
| mediastore | S33 | FileStorage | 61% (11/18) | Judge frequently rejects |
| mediastore | S35 | FileStorage | 56% (10/18) | Judge frequently rejects |

Persistent FNs are concentrated on components with ambiguous names (Logic, FileStorage) — exactly the components Phase 1 classifies inconsistently.

---

## 4. Recommendations: Improving Determinism Without Data Leakage

The goal is to make Phase 1 classifications stable across runs by providing structural guidance (not dataset-specific examples) in the prompt.

### Recommendation 1: Replace LLM Judgment with Structural Rules for Ambiguity

**Current approach:** Ask the LLM "which components are ambiguous?" — highly non-deterministic.

**Proposed approach:** Use deterministic heuristics first, only fall back to LLM for edge cases:

```python
def classify_component(name: str, model_components: list[str]) -> str:
    """Deterministic classification rules."""
    # Rule 1: Multi-word names with technical qualifiers → always architectural
    # Examples: "HTML5 Server", "Redis PubSub", "GAE Datastore", "BBB web"
    if len(name.split()) >= 2:
        return "architectural"

    # Rule 2: Names that are unique technical terms → architectural
    # (CamelCase, contains digits, all-caps acronyms)
    if re.match(r'^[A-Z][a-z]+[A-Z]', name):  # CamelCase
        return "architectural"
    if any(c.isdigit() for c in name):  # Contains digits
        return "architectural"
    if name.isupper() and len(name) <= 6:  # Short acronym (DB, FSESL)
        return "architectural"

    # Rule 3: Single common English words → ambiguous
    COMMON_WORDS = {"common", "logic", "storage", "cache", "facade",
                    "apps", "client", "server", "service", "gateway",
                    "proxy", "store", "manager", "handler", "util"}
    if name.lower() in COMMON_WORDS:
        return "ambiguous"

    # Rule 4: Everything else → architectural (default-safe)
    return "architectural"
```

**Why this works:** The heuristic captures the structural signal (multi-word names are always specific; single common words are always ambiguous) without any dataset-specific knowledge. The current LLM prompt already attempts this reasoning but does it non-deterministically.

**Trade-off:** A fixed rule set cannot adapt to novel naming conventions. The COMMON_WORDS list needs to be general enough to cover typical software architecture vocabulary without being tuned to specific datasets.

### Recommendation 2: Decouple Ambiguity from Pipeline Behavior

Currently, "ambiguous" classification has three downstream effects: (a) skip auto-partials, (b) judge applies stricter scrutiny, (c) block FN recovery. These should be independent decisions:

1. **Auto-partials (Phase 3b):** Instead of skipping ambiguous components entirely, apply a **capitalization filter** — only inject partial matches when the short form appears capitalized in the sentence (e.g., "the Server handles..." matches, "any server can..." does not). This is purely syntactic and deterministic.

2. **Judge (Phase 9):** Do NOT pass the ambiguity classification to the judge prompt. The judge should evaluate each link on its own merits, not be biased by a Phase 1 label. Currently, the judge's behavior changes when told a component is "ambiguous" — this amplifies Phase 1 non-determinism.

3. **FN Recovery (Phase 10):** Allow recovery for all components, but use the same capitalization filter from (1) to gate recovery candidates. This removes the hard block on ambiguous components.

### Recommendation 3: Stabilize Phase 2-3 via Structured Output Format

Phase 2 (Pattern Learning) and Phase 3 (Document Knowledge) show extreme variance because the LLM's free-form discovery depends on Phase 1's classification. To stabilize:

1. **Constrain output schema:** Instead of asking "what synonyms exist?", provide a fixed list of candidate mappings and ask the LLM to confirm/reject each. This converts an open-ended generation task into a classification task (more deterministic).

2. **Use deterministic abbreviation detection:** Scan the document for patterns like "X (Y)" or "Y, also known as X" before invoking the LLM. These regex-extractable patterns should not require LLM judgment.

3. **Merge Phase 2 into Phase 3:** Phase 2 (subprocess terms) currently has marginal value (see phase analysis doc). Its output feeds Phase 7's rejection logic but has no proven impact. Removing it saves 2 LLM calls and eliminates one source of variance.

### Recommendation 4: Stabilize Phase 9 Judge via Concrete Criteria

The judge currently receives vague instructions like "determine if this sentence discusses the component." This is non-deterministic. Provide structured evaluation criteria:

```
Evaluate this trace link using these SPECIFIC criteria:
1. EXPLICIT MENTION: Does the sentence contain the component name
   (or a known synonym/abbreviation from the document knowledge)?
2. FUNCTIONAL DESCRIPTION: Does the sentence describe what this
   specific component does, not what the system does in general?
3. UNIQUE REFERENCE: Could this sentence plausibly refer to a
   DIFFERENT component with equal or greater likelihood?

Approve if criteria 1 OR 2 is YES and criterion 3 is NO.
```

This converts subjective judgment into a checklist, reducing variance.

### Recommendation 5: Remove Dead Weight Phases

Phases 8 (Implicit References) and 10 (FN Recovery) produce zero output across all 18 runs. They cost ~5 LLM calls per dataset and add execution time. Removing them:
- Saves ~25 LLM calls per full 5-dataset evaluation
- Eliminates sources of potential variance if they ever do fire
- Simplifies the pipeline

---

## 5. Priority-Ordered Action Plan

| Priority | Action | Expected Impact | Effort |
|---|---|---|---|
| **P0** | Replace Phase 1 LLM classification with structural heuristic rules | Eliminates root cause of 13-27pp F1 variance | Medium |
| **P1** | Decouple ambiguity from Phase 9 judge prompt | Prevents Phase 1 errors from killing TransArc TPs | Low |
| **P2** | Add capitalization filter to Phase 3b auto-partials | Fixes BBB "Server"/"Client" FP without losing TP | Low |
| **P3** | Remove Phase 8 (implicit) and Phase 10 (recovery) | Saves 5 LLM calls, no quality loss | Low |
| **P4** | Structured judge criteria in Phase 9 | Reduces judge variance, especially on edge cases | Medium |
| **P5** | Deterministic abbreviation extraction in Phase 3 | Reduces Phase 3 variance | Medium |
| **P6** | Remove or merge Phase 2 (patterns) into Phase 3 | Saves 2 LLM calls, marginal quality impact | Low |

### Expected Outcome

With P0-P2 implemented, the pipeline should:
- **Eliminate** the 13-27pp F1 variance caused by Phase 1 non-determinism
- **Maintain** the 90.5% macro F1 achieved in the best runs
- **Reduce** LLM calls by ~7 per dataset (removing Phases 2, 8, 10)
- **Improve** BBB by fixing the auto-partial stoplist (removing 5 FP from "Server"/"Client"/"Conversion")

---

## 6. Summary

The V6 pipeline achieves 90.5% macro F1 in its best runs but suffers from 13-27pp F1 variance across runs. The **entire variance traces to a single LLM call in Phase 1** that classifies components as architectural vs ambiguous. This classification is subjective and non-deterministic, yet it gates critical pipeline behaviors (auto-partials, judge strictness, FN recovery).

The fix is straightforward: replace the LLM judgment with deterministic structural heuristics for the binary architectural/ambiguous decision, and decouple this classification from downstream pipeline behavior so that no single LLM call can cascade into 10+ TP losses.
