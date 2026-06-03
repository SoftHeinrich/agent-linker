# Linker Checkpoint Guide

How to read, write, and extend the phase-cache checkpoint system.
Relevant for any linker in the `s_linker*` family, especially when adding
validator-contribution logging for ablation studies.

---

## What checkpoints are

Each linker writes intermediate pipeline state to pickle files under
`results/phase_cache/<variant_name>/<dataset>/`.
A checkpoint captures the inputs and outputs of one pipeline stage so that:
- you can inspect intermediate data without re-running,
- post-hoc analysis scripts can reconstruct validator decisions against gold,
- a crashed or interrupted run can be partially resumed.

Checkpoints do **not** change pipeline behaviour — they are pure side-effects.

---

## The `_save_phase` API

```python
self._save_phase(text_path, phase_name, state_dict)
```

| Argument | Type | Notes |
|---|---|---|
| `text_path` | `str` | Path to the architecture document being processed. Used to derive the dataset name and checkpoint directory. |
| `phase_name` | `str` | File stem — saved as `<phase_name>.pkl`. Must be unique within a run. |
| `state_dict` | `dict` | Arbitrary picklable data. Keys are free-form; keep them descriptive. |

**Where files land:**
```
results/phase_cache/<_VARIANT_NAME>/<dataset_stem>/<phase_name>.pkl
```
e.g. `results/phase_cache/s_linker15/mediastore/extraction_passes.pkl`

**Guard pattern** — always wrap in `if self._current_text_path:` to stay safe
when the linker is called in test contexts that never set a text path:

```python
if self._current_text_path:
    self._save_phase(self._current_text_path, "my_phase", {"key": value})
```

---

## Complete checkpoint inventory — s_linker15

| Phase name | Written by | Key contents | Validator(s) covered |
|---|---|---|---|
| `layer1` | `link()` after Tier 1 | `model_knowledge` (incl. `ambiguous_names`), `doc_knowledge` (approved aliases), `raw_seed_links` | K2 classification; L1 input |
| `alias_proposed` | `_learn_document_knowledge_enriched` | `all_mappings` (proposed term→comp), `all_scopes` | **K1** — proposed minus approved = alias-judge rejects |
| `layer2` | `link()` after Tier 2 | `seed_links` (post-L1), `validated` (post-L2/L3/L4), `coref_links` (post-L5) | All Tier 2 outputs |
| `entity_candidates` | `_run_entity_pipeline` | `entity_candidates` (post-L3 intersection), `bundles` | L3 output |
| `extraction_passes` | `_extract_entities_enriched` | `pass1`, `pass2`, `intersected` (all `dict[(snum, comp_id), CandidateLink]`) | **L3** — union minus intersection = consensus rejects |
| `entity_decisions` | `_run_entity_pipeline` | `decisions` dict: per-candidate `path`, `p1`, `p2`, `approved` | **L2** (`path="generic_filter:…"`) and **L4** (`path="twopass_reject"`, check `p1 XOR p2`) |
| `coref_scope` | `_coref_cases_in_context` | `comp_terminals` (set), `pronoun_sents`, `role_ref_sents` (sentence number lists) | **S1** — `role_ref_sents` shows which sentences the terminal filter opened |
| `coref_rejects` | `_coref_cases_in_context` | `rejects`: list of `{snum, comp, reason}` where reason is `"no_antecedent"` or `"citation_fail"` | **L5** — compare against gold to classify TP-killed vs FP-killed |
| `final` | `link()` | `final` (list of `SadSamLink`) | Full pipeline output |

### Deriving per-validator contributions from checkpoints

```
K1 rejects  = alias_proposed["all_mappings"].keys()
              - set(layer1["doc_knowledge"].aliases.keys())

K2 effect   = layer1["model_knowledge"].ambiguous_names
              (components flagged ambiguous; L2 only fires on these)

L1 rejects  = {(s.sentence_number, s.component_id) for s in layer1["raw_seed_links"]}
              - {(s.sentence_number, s.component_id) for s in layer2["seed_links"]}

L2 rejects  = {key for key, d in entity_decisions["decisions"].items()
               if d.get("path", "").startswith("generic_filter")}

L3 rejects  = set(extraction_passes["pass1"]) | set(extraction_passes["pass2"])
              - set(extraction_passes["intersected"])

L4 rejects  = {key for key, d in entity_decisions["decisions"].items()
               if d.get("path") == "twopass_reject"}

L5 rejects  = [(r["snum"], r["comp"]) for r in coref_rejects["rejects"]]

S1 scope    = coref_scope["role_ref_sents"]  # sentences only reachable via terminal filter
```

For each reject set, look up `(snum, comp_id)` in the project gold standard to classify
each reject as a TP-killed (validator hurt recall) or FP-killed (validator helped precision).

---

## Adding a checkpoint to any linker — 3-step recipe

### Step 1 — Identify the data

Find the variable(s) you want to capture and the method they live in.
Prefer saving at the point where a decision is made (after computing something,
before discarding intermediate values), not at the final output.

### Step 2 — Add the save call

```python
# at the decision point, after your variable is fully populated:
if self._current_text_path:
    self._save_phase(self._current_text_path, "descriptive_name", {
        "key1": variable1,
        "key2": variable2,
    })
```

Requirements:
- `self._current_text_path` must be set before the linker processes a document.
  It is set at the top of `link()` — all methods called from `link()` can rely on it.
- The state dict must be **picklable** (`dict`, `list`, `set`, `str`, `int`, custom
  `@dataclass` without open file handles).
- Phase name must be **unique per linker class** — duplicates silently overwrite.

### Step 3 — Update this table

Add a row to the inventory table above with: phase name, method, key contents,
and which validator(s) the checkpoint enables tracking for.

---

## Loading checkpoints

```python
import pickle
from pathlib import Path

def load_checkpoint(variant: str, dataset: str, phase: str,
                    cache_dir: str = "results/phase_cache"):
    path = Path(cache_dir) / variant / dataset / f"{phase}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

# Example: load extraction passes for s_linker15 on mediastore
data = load_checkpoint("s_linker15", "mediastore", "extraction_passes")
pass1 = data["pass1"]      # dict[(snum, comp_id), CandidateLink]
pass2 = data["pass2"]
intersected = data["intersected"]
```

See `s_linker11_checkpoint_viewer.py` in the repo root for a full interactive
viewer pattern (list all phases, pretty-print counts, show rejected links).

---

## Naming conventions

| Pattern | Use for |
|---|---|
| `layer1`, `layer2` | Tier-level summaries (reserved — match `link()` structure) |
| `<stage>_candidates` | Raw candidates before validation (e.g. `entity_candidates`) |
| `<stage>_passes` | Per-pass outputs before aggregation (e.g. `extraction_passes`) |
| `<stage>_decisions` | Per-candidate decision records with reason/path (e.g. `entity_decisions`) |
| `<stage>_rejects` | Rejected candidates with structured reason field (e.g. `coref_rejects`) |
| `<stage>_scope` | Candidate-scope filter outputs — what entered a stage (e.g. `coref_scope`) |
| `<stage>_proposed` | Knowledge-layer proposals before quality gating (e.g. `alias_proposed`) |

---

## Porting to other linkers

`s_linker13_min` and `s_linker13_clean_v3` share the same `_save_phase` /
`_checkpoint_dir` implementation as s_linker15 (copied verbatim per the
standalone-file convention). Adding checkpoints to those linkers follows
the same 3-step recipe — the only difference is that the checkpoint directory
will be `results/phase_cache/s_linker13_min/<dataset>/` etc.

Linkers that predate the `_save_phase` infrastructure (s_linker10 and earlier)
do not have it; backporting is not recommended.
