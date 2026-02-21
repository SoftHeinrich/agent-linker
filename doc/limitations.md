# AgentLinker Limitations and Data Leakage Analysis

Review of `src/llm_sad_sam/linkers/experimental/agent_linker.py`.

---

## Data Leakage Risks

### 1. LLM Training Data Contamination (Critical)

The benchmark datasets (MediaStore, TeaStore, Teammates, BigBlueButton, JabRef) are well-known open-source projects with published gold standards in the ARDoCo research ecosystem. Pre-trained LLMs (GPT-5.2, Claude) have very likely seen:

- The SAD documents themselves
- The PCM repository files
- The gold standard CSV files from ARDoCo's GitHub repos
- Published papers describing expected trace links

The LLM may be recalling memorized gold standard answers rather than reasoning about trace links. There is no way to distinguish genuine inference from memorized recall, making evaluation results potentially unreliable.

**Mitigation options:**
- Evaluate on a held-out project not in common training corpora
- Use a model with a known training cutoff that predates the benchmark publication
- Compare LLM performance on seen vs. unseen projects to quantify contamination

### 2. LLM Sets Its Own Thresholds (Phase 0-1)

In `_learn_thresholds` (lines 723-764), the LLM is asked to recommend confidence thresholds (`coref_threshold`, `validation_threshold`, etc.), and then its own outputs are filtered against those same thresholds. The model is effectively grading its own exam with a rubric it chose. A contaminated model could set permissive thresholds to pass more memorized answers through.

### 3. LLM Classifies Its Own Difficulty (Phase 0)

`_learn_document_profile` (lines 665-721) asks the LLM to decide `recommended_strictness` ("relaxed"/"balanced"/"strict"), which then gates Phase 10 FN recovery entirely (line 162). A contaminated model could choose "relaxed" for projects where it knows the gold standard has many links, enabling FN recovery to inflate recall.

---

## Coverage Limitations (Systematic Blind Spots)

### 4. Entity Extraction Only Covers First 100 Sentences (Critical)

Phase 5 (line 997) sends only the first 100 sentences to the LLM:

```python
{chr(10).join([f"S{s.number}: {s.text}" for s in sentences[:100]])}
```

For larger documents (Teammates has 200+ sentences), everything after sentence 100 is invisible to entity extraction. These sentences can only get links via TransArc baseline, coreference, or implicit detection -- but implicit detection also depends on explicit mentions from earlier phases.

### 5. Judge Reviews Only First 30 Links

Phase 9 (line 591) only reviews the first 30 non-TransArc links:

```python
for i, l in enumerate(non_transarc[:30]):
```

Any links beyond index 30 are auto-approved without review (lines 642-643):

```python
for i, l in enumerate(non_transarc):
    if i not in judged and i >= 30:
        result.append(l)
```

### 6. FN Recovery Only Checks First 12 Candidates

Phase 10 (line 1132) caps recovery at 12 candidates:

```python
cases = [f"S{sn}: ..." for sn, txt, cn in potential_fns[:12]]
```

It also requires both passes to agree (`len(votes) >= 2`), making it very conservative. The hard cap of 12 means systematically missed links in larger documents.

### 7. Pattern Learning Samples Are Incomplete (Phases 2-3)

- Phase 2 pattern debate: first 70 sentences (line 812)
- Phase 3 document knowledge: first 100 sentences (line 885)
- Phase 0 profiling: first 50 sentences (line 677)

Patterns, abbreviations, and synonyms introduced later in documents are never learned.

---

## Algorithmic Limitations

### 8. Paragraph Boundary Detection Is Fragile

`_is_paragraph_boundary` (lines 224-240) only detects boundaries via 7 hardcoded transition words (`however`, `furthermore`, etc.). It misses:

- Actual blank lines or formatting (the sentence loader strips these)
- Section headers, numbered lists
- Topic shifts without transition words

This directly affects discourse context quality, which gates both coreference (Phase 7) and implicit detection (Phase 8).

### 9. Subject Detection Is Brittle

`_is_subject` (lines 242-258) uses character position (`< 60`) and a fixed verb list to guess grammatical subjects:

```python
if comp_pos < 60:  # Magic number
    verbs = ['is', 'are', 'does', ...]
```

No actual NLP parsing is performed. False positives: "Unlike **Component**, the system..." would be tagged as subject. False negatives: component names appearing after position 60 in long sentences are never subjects.

### 10. Context Truncation Loses Information

Throughout prompts, previous sentences are truncated aggressively:

- Coreference: `prev.text[:70]` (line 287)
- Judge: `prev.text[:45]` (lines 598-599)
- Implicit validation: `prev.text[:60]` (line 528)

Component names near the end of long sentences are silently dropped from the LLM's context, leading to missed coreference chains.

### 11. `is_implementation` Has False Positives

`ModelKnowledge.is_implementation` (lines 771-781) uses simple substring matching:

```python
if other != name and len(other) >= 3 and other in name:
```

A component named "DB" (len 2) is skipped, but "Log" in "LoginHandler" would falsely mark "LoginHandler" as an implementation of "Log" (if both exist). The `len(prefix) >= 2` guard helps but does not prevent all cases.

### 12. No Retry at Pipeline Level

If any phase's LLM call fails (returns `None` from `extract_json`), that phase silently produces empty results. There is no retry, no fallback, no error escalation. A single timeout in Phase 5 means zero entity extraction for the entire run.

### 13. Conversation Mode Unused

The `LLMClient` has a full conversation mode implementation, but `AgentLinker` only uses stateless `query()` calls. Each phase starts fresh with no memory of prior phases' findings. The LLM cannot build on its earlier analysis of the document.

---

## Summary

| Category | Risk | Impact |
|----------|------|--------|
| Training data contamination | **High** | Evaluation results may not generalize to unseen projects |
| LLM self-calibrating thresholds | **Medium** | Amplifies contamination effects |
| First-100-sentence coverage cap | **High** | Silent recall loss on large documents |
| Judge only reviews 30 links | **Medium** | FPs auto-approved past the cap |
| Truncated context in prompts | **Low-Medium** | Missed coreference chains |
| No pipeline-level retry | **Medium** | Single failure can zero-out a phase |
