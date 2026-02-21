# V6 Pipeline Phase-by-Phase Value Analysis

Run: V6 with Codex backend, 2026-02-17.
Macro F1: **90.5%** across 5 datasets.

---

## Per-Dataset Results

| Dataset | P | R | F1 | TP | FP | FN | Gold | TransArc baseline F1 |
|---|---|---|---|---|---|---|---|---|
| mediastore | 86.2% | 80.6% | 83.3% | 25 | 4 | 6 | 31 | 69.4% |
| teastore | 100.0% | 88.9% | 94.1% | 24 | 0 | 3 | 27 | 85.1% |
| teammates | 94.8% | 96.5% | 95.7% | 55 | 3 | 2 | 57 | 71.0% |
| bigbluebutton | 84.1% | 85.5% | 84.8% | 53 | 10 | 9 | 62 | 79.3% |
| jabref | 90.0% | 100.0% | 94.7% | 18 | 2 | 0 | 18 | 94.7% |
| **MACRO AVG** | **91.0%** | **90.3%** | **90.5%** | **175** | **19** | **20** | **195** | |

---

## TP/FP Breakdown by Source

| Source | mediastore | teastore | teammates | BBB | jabref | Total TP | Total FP |
|---|---|---|---|---|---|---|---|
| transarc | 17 TP, 1 FP | 20 TP, 0 FP | 47 TP, 3 FP | 44 TP, 5 FP | 18 TP, 2 FP | 146 | 11 |
| validated | 7 TP, 2 FP | 0 | 3 TP, 0 FP | 0 | 0 | 10 | 2 |
| entity | 1 TP, 0 FP | 0 | 0 | 1 TP, 0 FP | 0 | 2 | 0 |
| coreference | 0 TP, 1 FP | 4 TP, 0 FP | 5 TP, 0 FP | 0 | 0 | 9 | 1 |
| partial_inject | 0 | 0 | 0 | 8 TP, 5 FP | 0 | 8 | 5 |
| **Total** | **25 TP, 4 FP** | **24 TP, 0 FP** | **55 TP, 3 FP** | **53 TP, 10 FP** | **18 TP, 2 FP** | **175** | **19** |

---

## Phase Analysis

### Phase 0: Document Profile — GOOD (steering)

Classifies structural complexity based on uncovered ratio (>0.5) and sentences-per-component (>5):

| Dataset | spc | uncovered | Complex? |
|---|---|---|---|
| mediastore | 2.6 | 56.8% | False |
| teastore | 3.9 | 65.1% | False |
| teammates | 24.8 | 62.1% | **True** |
| bigbluebutton | 7.2 | 64.4% | **True** |
| jabref | 2.2 | 23.1% | False |

**Value:** Drives Phase 7 mode selection (discourse vs debate) and Phase 8 skip. All classifications are correct. No direct links but critical routing decision.

**Cost:** Cheap — no LLM call (heuristic only).

---

### Phase 1: Model Structure — GOOD (protective)

Identifies ambiguous component names that downstream phases skip:

| Dataset | Ambiguous | Effect |
|---|---|---|
| teastore | {Registry, Persistence} | Skipped in FN recovery |
| teammates | {Common} | Skipped in FN recovery (previously caused 12 FP in earlier versions) |
| bigbluebutton | {Apps} | Skipped in FN recovery |

**Value:** Prevents FP cascades from generic component names. The `Common` filtering alone justified this phase historically.

**Cost:** 1 LLM call.

---

### Phase 2: Pattern Learning — MARGINAL (indirect)

Learns subprocess terms that are checked in coref (Phase 7) to reject links:

| Dataset | Subprocess terms | Examples |
|---|---|---|
| mediastore | 15 | log-in, registration, reencoding |
| teastore | 15 | scaled image, cache lookup, cookies |
| teammates | 14 | CSS, HTML, JavaScript, AJAX |
| BBB | 5 | backend processes, frontend processes |
| jabref | 0 | — |

**Value:** Indirect — subprocess terms gate coref/implicit rejections. Hard to prove any specific rejection happened because of this.

**Cost:** 2 LLM calls (debate format). Relatively expensive for unproven value.

---

### Phase 3: Document Knowledge — GOOD (enables recall)

Learns abbreviations, synonyms, and partial references:

| Dataset | Abbrev | Syn | Partial | Key mappings |
|---|---|---|---|---|
| mediastore | 0 | 2 | 2 | AudioAccess→MediaAccess, ReEncoder→Reencoding |
| teastore | 1 | 14 | 0 | WebUi→WebUI, "X service"→X patterns |
| teammates | 2 | 6 | 6 | logic.api→Logic, persistence framework→GAE Datastore |
| BBB | 5 | 9 | 8 | KMS→kurento, bbb-html5→HTML5 Server, FreeSWITCH→FreeSWITCH |
| jabref | 0 | 0 | 0 | Nothing to learn (simple document) |

**Value:** Critical for TransArc matching and entity extraction. Without this, "Kurento Media Server" can't map to the component `kurento`, and "persistence framework" can't map to `GAE Datastore`. Enables significant recall.

**Cost:** 2-3 LLM calls (debate + judge). Worth it.

---

### Phase 3b: Multi-word Enrichment — BAD for BBB

Auto-detects when a single word consistently refers to a multi-word component (>=3 standalone mentions):

| Dataset | Auto-partial | Mentions | Outcome |
|---|---|---|---|
| teammates | Datastore→GAE Datastore | 3 | Fine — no FP |
| BBB | Server→HTML5 Server | 9 | **5 FP via partial_inject** |
| BBB | Client→HTML5 Client | 9 | (part of above) |
| BBB | Conversion→Presentation Conversion | 5 | (part of above) |

**Value:** +8 TP for BBB from partial injection of these terms.

**Problem:** "Server" and "Client" are too generic — they match sentences discussing servers/clients in general, not the specific HTML5 Server/Client components. This is **the single largest FP source** in the entire pipeline (5 of 19 total FP across all datasets, all on BBB).

**Recommendation:** Add a stoplist for common English words (server, client, service, store, etc.) or require capitalized matches for single-word auto-partials.

---

### Phase 4: TransArc Baseline — ESSENTIAL

| Dataset | Links | TP | FP |
|---|---|---|---|
| mediastore | 18 | 17 | 1 |
| teastore | 20 | 20 | 0 |
| teammates | 50 | 47 | 3 |
| BBB | 49 | 44 | 5 |
| jabref | 20 | 18 | 2 |
| **Total** | **157** | **146** | **11** |

**Value:** Foundation of the pipeline. 146/175 total TP (83%) originate from TransArc. Not optional.

**Note:** For teammates, raw TransArc produces 81 links (P=60.5%), but after dedup and Phase 9 judge, only 50 remain with 47 TP — the pipeline successfully filters 32+ TransArc FP down to 3.

---

### Phase 5+6: Entity Extraction + Validation — GOOD

| Dataset | Candidates | Validated | Final TP | Final FP |
|---|---|---|---|---|
| mediastore | 29 | 28 | 8 (7 validated + 1 entity) | 2 |
| teastore | 21 | 5 | 0 (all deduped with transarc) | 0 |
| teammates | 47 | 43 | 3 | 0 |
| BBB | 39 | 18 | 1 entity | 0 |
| jabref | 0 | 0 | 0 | 0 |
| **Total** | | | **12** | **2** |

**Value:** +12 TP, +2 FP. Precision 86%. The two-pass voting validation gate works well — teastore rejected 16/21, BBB rejected 21/39 candidates.

**Note:** Many validated candidates are deduped with existing transarc links and appear under the `transarc` source in final output (higher priority). The actual contribution to recall is concentrated on mediastore (+8 TP) and teammates (+3 TP).

---

### Phase 7: Coreference — GOOD

| Dataset | Mode | Links | TP | FP |
|---|---|---|---|---|
| mediastore | discourse | 3 | 0 | 1 |
| teastore | discourse | 4 | 4 | 0 |
| teammates | debate | 5 | 5 | 0 |
| BBB | debate | 0 | 0 | 0 |
| jabref | discourse | 0 | 0 | 0 |
| **Total** | | **12** | **9** | **1** |

**Value:** +9 TP, +1 FP. Excellent precision (90%). Teastore (+4 TP) and teammates (+5 TP) are significant recall gains from pronoun resolution.

**Note:** BBB debate coref found 0 links despite being a complex document. The debate propose+judge pattern may be too conservative for BBB's writing style.

---

### Phase 8: Implicit References — DEAD WEIGHT

| Dataset | Candidates | Detected | Final contribution |
|---|---|---|---|
| mediastore | — | 0 | 0 TP, 0 FP |
| teastore | 8 | 1 | 0 (deduped with existing link) |
| teammates | SKIPPED | — | — |
| BBB | SKIPPED | — | — |
| jabref | — | 0 | — |

**Value:** Zero across all 5 datasets. The conservative filters (active_entity == paragraph_topic, distance <= 2) are so strict that nothing ever survives. Complex documents skip this phase entirely.

**Cost:** Up to 2 LLM calls on non-complex documents.

**Recommendation:** Either remove entirely or significantly relax the pre-filter criteria. Currently this phase burns LLM tokens for zero return.

---

### Phase 8b: Partial Injection — MIXED

| Dataset | Injected | TP | FP |
|---|---|---|---|
| teammates | 6 | 0 (all deduped or filtered by judge) | 0 |
| BBB | 13 | 8 | 5 |
| **Total** | **19** | **8** | **5** |

**Value:** +8 TP, +5 FP. The mechanical injection itself is correct — it just applies known partial references from Phase 3/3b. The 5 FP are entirely caused by Phase 3b's overly aggressive auto-partials ("Server", "Client", "Conversion" in BBB).

**Recommendation:** Fix Phase 3b, not Phase 8b. The injection mechanism is sound.

---

### Phase 9: Judge Review — GOOD (critical for teammates)

| Dataset | Input | Rejected | Effect |
|---|---|---|---|
| mediastore | 30 | 1 | Minor cleanup |
| teastore | 24 | 0 | All clean already |
| teammates | 71 | 13 | **Critical**: brought P from ~60% to 94.8% |
| BBB | 63 | 0 | **Failed**: approved all 5 partial_inject FP |
| jabref | 20 | 0 | All clean already |

**Value:** The judge is the single most important quality gate for noisy datasets. Teammates' TransArc baseline has P=60.5% — without the judge, the pipeline would have ~30+ FP instead of 3.

**Problem:** For BBB, the judge approved all 63 links including 5 FP from partial_inject. The judge struggles with generic-word-as-component-name (e.g., "Server" referring to any server vs. the specific "HTML5 Server" component) when the source is `partial_inject`.

---

### Phase 10: FN Recovery — DEAD WEIGHT

| Dataset | Candidates | To judge | Recovered |
|---|---|---|---|
| mediastore | 0 | — | 0 |
| teastore | 0 | — | 0 |
| teammates | 23 | 11 | 0/11 approved |
| BBB | 7 | — | 0 |
| jabref | 0 | — | 0 |

**Value:** Zero recovery across all datasets. The three-gate system (two-pass voting + judge confirmation) is too strict — recovery candidates that pass both voting rounds still get rejected by the recovery judge.

**Cost:** Up to 3 LLM calls per batch.

**Recommendation:** Either remove this phase (saving LLM cost) or relax the recovery judge to accept candidates that passed both voting rounds. Currently, 0% recovery rate means this is pure overhead.

---

## Summary Table

| Phase | TP | FP | Net | Verdict |
|---|---|---|---|---|
| 0 Profile | — | — | steering | GOOD |
| 1 Model | — | — | protective | GOOD |
| 2 Patterns | — | — | indirect | MARGINAL |
| 3 DocKnowledge | — | — | enables recall | GOOD |
| **3b Multi-word** | ~8 | ~5 | +3 | **BAD** (BBB) |
| 4 TransArc | 146 | 11 | +135 | ESSENTIAL |
| 5+6 Entity+Valid | 12 | 2 | +10 | GOOD |
| 7 Coref | 9 | 1 | +8 | GOOD |
| **8 Implicit** | 0 | 0 | 0 | **DEAD WEIGHT** |
| 8b Partial Inject | 8 | 5 | +3 | MIXED |
| 9 Judge | — | −13 | removes FP | GOOD |
| **10 Recovery** | 0 | 0 | 0 | **DEAD WEIGHT** |

## Key Takeaways

1. **Biggest wins:** Phase 9 judge (+13 FP removed on teammates), Phase 7 coref (+9 TP), Phase 5+6 entity extraction (+12 TP).
2. **Biggest problem:** Phase 3b auto-partials cause 5 of 19 total FP (26%), all on BBB. "Server", "Client", "Conversion" are too generic as auto-partials.
3. **Dead weight:** Phases 8 and 10 produce zero output across all datasets. They cost ~5 LLM calls combined and could be removed or reworked.
4. **BBB is the hardest dataset:** 10 FP (53% of all FP), split between transarc (5) and partial_inject (5). The partial_inject FP are fixable via Phase 3b improvements.
