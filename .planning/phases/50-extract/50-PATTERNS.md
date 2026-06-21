# Phase 50: EXTRACT - Pattern Map

**Mapped:** 2026-06-21
**Files analyzed:** 2 (1 extraction script + 1 faithfulness check; RESEARCH recommends merging the check into the same script's `main()`)
**Analogs found:** 2 / 2 (both exact role + data-flow matches in-repo)

> No CONTEXT.md for this phase. File list extracted from `50-RESEARCH.md`
> (EXTRACT-01/02 = the pickle→JSON extraction script; EXTRACT-03 = the per-cell
> faithfulness oracle). RESEARCH "Architecture Patterns" recommends ONE script
> with `load_cell / to_neutral / faithfulness / main`, so both files map to the
> same primary analog. Planner may keep them as one file or split EXTRACT-03 into
> a thin verifier — the analog is identical either way.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `scripts/extract_s20union_caches.py` (EXTRACT-01/02 — pickle→neutral-JSON extractor) | utility / standalone CLI script | batch + file-I/O + transform (pickle→JSON) | `ablation_evjudge_rest.py` | exact (root-level standalone, .env+src bootstrap, matrix walk, pickle.load from `results/`, JSON writer) |
| Faithfulness check (EXTRACT-03 — re-derive `final.links` set, assert == `*_links.csv`, print per-cell PASS/FAIL) | test / verification | batch + file-I/O + transform (set-equality) | `ablation_evjudge_rest.py` `main()` (tabular per-cell print + JSON dump) + `run_ablation.py` `eval_metrics`/`export_links_csv` | exact (same CSV oracle, same per-cell loop+print idiom) |

**Mirror source (read-only, MUST be matched by the reader, not edited — GATE-01):**
the SAVE side that produced the pickles is `s_linker20_union.py` `_save_phase` /
`_checkpoint_dir` / `_backend_tag` and the Phase-6 final-derivation loop. The new
reader must invert exactly these.

---

## Pattern Assignments

### `scripts/extract_s20union_caches.py` (utility CLI, pickle→JSON batch transform)

**Primary analog:** `ablation_evjudge_rest.py` — a repo-root standalone script that
already does the four things this phase needs: (1) bootstrap env + import path,
(2) declare a project matrix, (3) `pickle.load` per-cell from a `results/...`
path, (4) load the CSV oracle and emit JSON. It is no-LLM in its `baseline_score`
path (loads frozen `final.pkl`), which is exactly Phase 50's mode.

**Env + import-path bootstrap** — copy verbatim shape from `ablation_evjudge_rest.py:10-21`
(also present in `run_ablation.py:18-37`):
```python
# ablation_evjudge_rest.py:10-21
_ENV = Path('/mnt/hostshare/ardoco-home/agent-linker/.env')
if _ENV.exists():
    for line in _ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k, v = line.split('=', 1)
        os.environ[k.strip()] = v.strip()

sys.path.insert(0, str(Path(__file__).parent / "src"))   # repo-root form; from scripts/ use parent.parent / "src"
```
> NOTE for a `scripts/` location: `Path(__file__).parent` must become
> `Path(__file__).resolve().parent.parent` so `src/` resolves. `run_ablation.py:18-19`
> uses `ROOT = Path(__file__).parent` then `ROOT / "src"` — the root-level form.
> A `.env` load is OPTIONAL here (no LLM/network used) but harmless; the
> **load-bearing** line is the `sys.path.insert(..., "src")`.

**The single import that registers all pickled classes** (RESEARCH "Minimal Import
Surface", verified) — pickle resolves `AliasEntry` from the linker module and
transitively pulls `data_types_v2`:
```python
import pickle, json, csv, glob, os
import llm_sad_sam.linkers.experimental.s_linker20_union  # noqa: F401  registers AliasEntry; transitively imports data_types_v2
```
`AliasEntry` is defined in `s_linker20_union.py:245-249` (`@dataclass(frozen=True)`,
fields `component:str, scope:str`). The other pickled classes live in
`data_types_v2.py`: `SadSamLink` (11-18), `CandidateLink` (21-34),
`ModelKnowledge` (37-44, `ambiguous_names: set[str]`), `DocumentKnowledge`
(47-61, `aliases: dict[str,str]` hint but holds `AliasEntry` values at runtime).
Do NOT instantiate `SLinker20Union` (its `__init__` builds an `LLMClient`); a bare
module import has no side effects (RESEARCH-verified).

**Matrix declaration (analog: the `PROJECTS`/`DATASETS` dicts), but encode the TWO
asymmetric roots explicitly** — pattern from `ablation_evjudge_rest.py:23-40`
(`BENCH` + `PROJECTS` dict) and `run_ablation.py:930-961` (`DATASETS`). The
asymmetry (gpt root has an extra `gpt/` level; sonnet does not; subdirs are
`openai/` vs `claude/`) is encoded in the two shell drivers — see
`run_s20union_gpt_n3.sh` (`BASE="results/v2.6.5_s20union/gpt"`, `LLM_BACKEND=openai`)
vs `run_s20union_sonnet_n3.sh` (`BASE="results/v2.6.5_s20union_sonnet"`,
`LLM_BACKEND=claude`). Mirror that as data, NOT a single template:
```python
PROJECTS = ["bigbluebutton", "jabref", "mediastore", "teammates", "teastore"]  # dir names, structural — NOT vocabulary (GATE-06 ok)
RUNS = ["run1", "run2", "run3"]
# (results_root, backend_subdir, friendly_tag)
MATRIX = [
    ("results/v2.6.5_s20union/gpt",         "openai", "gpt"),     # note extra gpt/ level
    ("results/v2.6.5_s20union_sonnet",      "claude", "sonnet"),  # no extra level
]
# cell pkl dir: f"{root}/{run}/phase_cache/s_linker20_union/{backend_subdir}/{project}"
# cell oracle  : f"{root}/{run}/{project}/s_linker20_union_{project}_links.csv"
#                f"{root}/{run}/{project}/ablation_*.json"   (glob; exactly one)
```
Verified on disk this session: `.../openai/mediastore/{layer1..4,final}.pkl` and
`.../run1/mediastore/{s_linker20_union_mediastore_links.csv, ablation_*.json}` exist.

**Pickle-load idiom (no custom unpickler)** — analog `ablation_evjudge_rest.py:63-67`
(`baseline_score`) and `s_linker11_checkpoint_viewer.py:33`:
```python
# ablation_evjudge_rest.py:63-67
def baseline_score(proj, backend):
    p = f'results/phase_cache/s_linker19/{backend}/{proj}/final.pkl'
    if not os.path.exists(p): return None
    d = pickle.load(open(p, 'rb'))
    return score_local(d['final'], proj)
```
Apply per cell for all five files: `layer1.pkl … layer4.pkl, final.pkl`. Each
top-level object is a plain `dict` with `(sentence:int, component_id:str)` tuple keys.

**What the reader must INVERT (the SAVE side in `s_linker20_union.py` — read-only):**
- `_checkpoint_dir` (`s_linker20_union.py:1022-1027`) builds the path the reader walks:
```python
# s_linker20_union.py:1022-1027
def _checkpoint_dir(self, text_path):
    cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
    ds = os.path.splitext(os.path.basename(text_path))[0]
    d = os.path.join(cache_dir, self._VARIANT_NAME, self._backend_tag(), ds)   # <cache>/s_linker20_union/<openai|claude>/<proj>
```
  `_VARIANT_NAME == "s_linker20_union"` (line 263); `_backend_tag()` returns
  `LLMBackend.value` → `"openai"`/`"claude"` (lines 1015-1020). The shell drivers
  set `PHASE_CACHE_DIR="$rdir/phase_cache"`, giving the `<run>/phase_cache/...`
  level. This is why the on-disk leaf is `phase_cache/s_linker20_union/<subdir>/<proj>/`.
- `_save_phase` (`s_linker20_union.py:1029-1034`) is the exact dump the reader mirrors:
```python
# s_linker20_union.py:1029-1034
def _save_phase(self, text_path, phase_name, state):
    d = self._checkpoint_dir(text_path)
    path = os.path.join(d, f"{phase_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(state, f)
```
  The `state` dicts saved are: layer1 (`:511-516`), layer2 (`:523-528`),
  layer3 (`:546-551`), layer4 (`:563-569`), final (`:609-613`). Field-by-field
  schema is in RESEARCH; these line refs are the authoritative key list.

**Final-set derivation to mirror (Phase 6, `s_linker20_union.py:571-599`)** — this is
the authoritative merge; the extractor reads `final.links` directly rather than
re-running it, but must understand it for EXTRACT-03:
```python
# s_linker20_union.py:573-586  (entity FIRST, then coref; dedup keeps first per (s,c))
entity_links = [SadSamLink(c.sentence_number, c.component_id, c.component_name, source="entity") for c in validated]
all_links = entity_links + coref_validated
seen = set(); final = []
for lk in all_links:
    key = (lk.sentence_number, lk.component_id)
    if key not in seen:
        seen.add(key); final.append(lk)
```
LANDMINE (RESEARCH Pitfall 1): re-derive ONLY from the validated **lists** /
`final`, never from the `decisions`/`coref_decisions` dicts (dict collapses dup
`(s,c)` coref keys → off-by-one on `gpt/run2/teammates`, `gpt/run3/teammates`).

**Serialization rule — tuple-key dict → ordered record list** (RESEARCH "Architecture
Patterns"; no in-repo analog exists for this exact transform, it is new logic but trivial):
```python
def keyed_to_records(d):  # d: dict[(s,c), value-dict]; Py3.7+ preserves insertion order
    return [{"s": s, "c": c, **v} for (s, c), v in d.items()]
```
Sets → `sorted(list(...))`; `AliasEntry` → `{"component":..., "scope":...}`;
dataclasses → `dataclasses.asdict`; `coref.raw`/`coref.validated` stay **lists**
(never a keyed map). The exact target JSON shape is in RESEARCH "Proposed Neutral
JSON Schema" (the abridged mediastore/gpt/run1 example).

**JSON writer (house style)** — analogs: `ablation_evjudge_rest.py:111`
(`json.dump(rows, open(out_path, 'w'), indent=2)`) and `run_ablation.py:1328-1331`
(`json.dump(all_results, handle, indent=2, default=str)`). For the NEUTRAL extract,
RESEARCH requires determinism + fail-loud, so extend house style with
`sort_keys=True, allow_nan=False` and do NOT use `default=str` (everything must be
JSON-native already):
```python
with open(out_path, "w") as f:
    json.dump(neutral, f, indent=2, sort_keys=True, allow_nan=False)
```
> Per-cell output filename is STABLE (`<extract_root>/<backend>/<run>/<project>.json`) —
> do NOT timestamp it (RESEARCH determinism check wants byte-identical re-runs).
> This differs from the analogs, which timestamp aggregate reports
> (`run_ablation.py:1328` `ablation_{time.strftime('%Y%m%d_%H%M%S')}.json`,
> `ablation_evjudge_rest.py:109-110`). Reserve timestamping (if any) for an
> optional aggregate PASS/FAIL report, never the cell extracts.

**CLI / `__main__` (analog: `run_ablation.py:1227-1248, 1251, 1335-1336`)** — argparse is
the house pattern for the main runner; `ablation_evjudge_rest.py` uses a bare
`main()`/`__main__` (114-115). For a fixed 30-cell matrix a bare `main()` is
sufficient; add argparse only if `--extract-root` / `--with-raw` flags are wanted
(RESEARCH Open Q2 suggests `--with-raw`):
```python
# run_ablation.py:1335-1336  — the guard to copy
if __name__ == "__main__":
    raise SystemExit(main())
```
`main()` should `exit(nonzero)` if any cell FAILs (RESEARCH "Natural verification
points"); `raise SystemExit(main())` is the established return-code idiom.

---

### Faithfulness check (EXTRACT-03 — verification, set-equality oracle)

**Primary analog:** `ablation_evjudge_rest.py` `main()` (81-112) — the per-cell loop
that loads, compares, and prints a tabular per-row verdict; plus
`run_ablation.py` `export_links_csv` (1079-1093, the CSV the oracle reads) and
`eval_metrics` (1096-1103, the set-arithmetic idiom).

**The CSV oracle to read (exactly what `export_links_csv` wrote)** —
`run_ablation.py:1079-1093`:
```python
# header + row order the faithfulness check must parse
writer.writerow(["sentence", "component_id", "component_name", "confidence", "source"])
# rows sorted by (sentence_number, component_id); confidence formatted f"{c:.2f}" -> "1.00"
```
So the cell's `s_linker20_union_<proj>_links.csv` columns are
`sentence,component_id,component_name,confidence,source`. Build the oracle set on
the `(sentence:int, component_id, source)` triple (RESEARCH EXTRACT-03 definition).

**CSV-read idiom** — analog `ablation_evjudge_rest.py:42-50` (`load_gold_local`):
```python
# ablation_evjudge_rest.py:42-50  — adapt columns: gold is (comp_id=row[0], sentence=row[1]);
#                                   s20union links.csv is (sentence=row[0], component_id=row[1], ..., source=row[4])
with open(csv_path) as f:
    r = csv.reader(f); next(r, None)   # skip header
    for row in r:
        ...
```

**Set-arithmetic + per-cell verdict** — analog `eval_metrics` (`run_ablation.py:1096-1103`)
for the `&`/`-` idiom, and `ablation_evjudge_rest.py:96-107` for the printed
per-row table. EXTRACT-03 is set-EQUALITY (not P/R/F1):
```python
extract_set = {(l["s"], l["c"], l["source"]) for l in neutral["final"]["links"]}
csv_set     = {(int(row[0]), row[1], row[4]) for row in rows}     # sentence, component_id, source
ok = extract_set == csv_set
print(f"{'PASS' if ok else 'FAIL'}  {backend}/{run}/{project}  "
      f"extract={len(extract_set)} csv={len(csv_set)} diff={extract_set ^ csv_set}", flush=True)
```

**Secondary cross-check (optional)** — `ablation_*.json` is glob'd
(`ablation_<TIMESTAMP>.json`, one per cell). Its shape is
`{ "<project>": { "s_linker20_union": {..., "n_links", "sources"{entity,coreference}} } }`
(produced by `run_variant`, `run_ablation.py:1180-1194`, then dumped at
`:1328-1331`). Assert `len(final.links) == n_links` and per-source counts ==
`sources`. RESEARCH marks these SECONDARY (P/R/F1 are gold metrics, not needed for
model-output faithfulness).

---

## Shared Patterns

### Logging / PASS-FAIL output → `print`, never the `logging` module
**Sources:** `ablation_evjudge_rest.py:72,78,96-107,112` (all `print`, several with
`flush=True`); `run_ablation.py:1173-1178,1275-1280` (plain `print`);
`s_linker20_union.py` uses `print(...)` throughout (e.g. `:1034`) and
`print(..., flush=True)` for the loud invariant banner (`:978`).
**Apply to:** the extraction script's per-cell status + the EXTRACT-03 PASS/FAIL
line. Use `print(..., flush=True)` for the verdict lines (matches
`ablation_evjudge_rest.py` `run`/`main` and the s20union loud-warning convention)
so output streams under `tee` like the shell drivers expect.
No `logging` import anywhere in the script family — do not introduce one.

### JSON writing → `json.dump(obj, f, indent=2)` (+ determinism extras for the neutral extract)
**Sources:** `run_ablation.py:1328-1331` (`indent=2, default=str`);
`ablation_evjudge_rest.py:111` (`indent=2`); `s_linker20_union.py:1055,1073`
(`json.dump(..., indent=2, default=str)`).
**Apply to:** neutral cell JSON — `indent=2, sort_keys=True, allow_nan=False`, NO
`default=str` (fail loudly if a non-native type sneaks in). Aggregate report (if
any) may keep the looser house form.

### Env + `src/` path bootstrap for a standalone script
**Sources:** `ablation_evjudge_rest.py:10-18`; `run_ablation.py:18-33` (`ROOT`,
`sys.path.insert(0, str(ROOT/"src"))`, `load_dotenv`).
**Apply to:** the script header. `.env` load is optional (no LLM); the
`sys.path.insert(..., src)` is mandatory so `import llm_sad_sam...` resolves when
run as `python scripts/extract_s20union_caches.py` from repo root.

### Stdout line buffering for long sweeps
**Source:** `run_ablation.py:16` (`sys.stdout.reconfigure(line_buffering=True)`).
**Apply to:** optional but recommended — keeps per-cell PASS/FAIL visible live when
piped to a log, consistent with the shell drivers' `tee`/redirect usage.

---

## No Analog Found

| Concern | Role | Data Flow | Reason / Mitigation |
|---------|------|-----------|---------------------|
| tuple-key dict → ordered `{"s","c",...}` record list | serialization helper | transform | No existing repo code serializes `(s,c)`-keyed dicts to JSON. New logic, but trivial (RESEARCH gives the 1-liner `keyed_to_records`). Self-verified by the per-cell faithfulness assertion. |
| Asymmetric two-root matrix walk | path enumeration | batch | Existing matrices (`DATASETS`, `PROJECTS`) are single-root, symmetric. New code must encode two `(root, subdir, tag)` triples (the shell drivers encode the asymmetry only in bash, not Python). RESEARCH Pitfall 2. |
| `sort_keys=True, allow_nan=False` determinism contract | JSON writer | file-I/O | Analogs use `indent=2 (+ default=str)` only; the determinism/fail-loud extras are new requirements from RESEARCH (PARITY gate), not present in any existing writer. |

## Metadata

**Analog search scope:** repo root (`*.py`, `*.sh`), `scripts/`, `src/llm_sad_sam/linkers/experimental/`, `src/llm_sad_sam/core/`, on-disk `results/v2.6.5_s20union*/`.
**Files scanned:** `run_ablation.py`, `ablation_evjudge_rest.py`, `run_s20union_gpt_n3.sh`, `run_s20union_sonnet_n3.sh`, `s_linker20_union.py`, `data_types_v2.py`, `s_linker11_checkpoint_viewer.py`, `CLAUDE.md`, `50-RESEARCH.md`; matrix cells verified by directory listing.
**Pattern extraction date:** 2026-06-21
</content>
</invoke>
