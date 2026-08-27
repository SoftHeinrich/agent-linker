# Paper Claim Verification — Distributional Inequality

> Audits the paper's GOLD distributional-inequality claims against the Phase-1 engine (`inequality.py`, gate-verified vs `writing/eval.tex` Ch1). Reuse-only; no new measurement. The TransArc actual-error cascade is recorded as SYSTEM-SPECIFIC (out of this gold-only study).

**Summary:** 6 MATCH · 0 MISMATCH · 1 PARTIAL · 1 SYSTEM-SPECIFIC · 3 placeholders deferred → Phase 3.

## Claims

| ID | Claim | Source | Paper value | Computed value | Label |
|----|-------|--------|-------------|----------------|-------|
| C1 | Enrollment expansion factor ranges 1.0x (MediaStore) to 217.6x (JabRef) | `metric.tex:11; writing/eval.tex tab:enrollment` | 1.0x -> 217.6x (35.5x avg) | 1.0x -> 217.6x (avg 35.5x; 525->18660) | **MATCH** |
| C2 | One directory decision expands into hundreds of link-level pairs (JabRef) | `metric.tex:11` | hundreds per directory decision | JabRef max single-component fan-out 972 files; 38 raw -> 8268 enrolled | **MATCH** |
| C3 | Per-component link counts are heavily skewed (long-tail) in BOTH tasks | `metric.tex:14-16; alinker eval.tex:23,25` | long-tail / heavy skew, both tasks (qualitative) | sad-code files-per-component Gini 0.400->0.694; sad-sam per-component Gini 0.179->0.370 (both > 0) | **MATCH** |
| C4 | Per-sentence enrolled gold link Gini ranges 0.331 (MediaStore) to 0.645 (Teammates) | `writing/eval.tex tab:sent_gini (L237-256)` | 0.331 -> 0.645 | 0.331 -> 0.645 | **MATCH** |
| C5 | Three sentences account for ~70% of the entire enrolled gold standard (JabRef) | `writing/eval.tex:258` | 70% | JabRef per-sentence Top-3 share = 70.0% | **MATCH** |
| C6 | SAM-CODE files-per-component Gini 0.400->0.694; JabRef top-3 components = 98.6% of links | `writing/eval.tex tab:samcode_skew (L191-210)` | Gini 0.400 -> 0.694; JabRef Top-3 Conc 98.6% | Gini 0.400->0.694; JabRef Top-3 Conc 98.6% | **MATCH** |
| C7 | The top architectural element per project accounts for 44-48% of the SAD-CODE gold | `writing/eval.tex tab:sadcode_conc (L214-232)` | 44-48% (top AE share of gold links) | top-1 component link share per project: mediastore=38.4%, teastore=43.2%, teammates=22.5%, bigbluebutton=14.5%, jabref=47.0% (max 47.0%) | **PARTIAL** |
| C8 | 36 component-level FPs cascade to 3,457 file-level FPs (96.0x); block correlation | `writing/eval.tex tab:amplification (L156-179)` | 36 -> 3,457 (96.0x) | (TransArc actual-error attribution; not a gold property -- see reports/TRANSARC_EMPIRICAL_STUDY.md) | **SYSTEM-SPECIFIC** |

### Notes

- **C7:** Only JabRef (47.0%) reproduces tab:sadcode_conc. The paper's per-project top-AE share uses a single coarse top-level component per file; this gold-only engine uses the multi-mapped component_suite universe (a file can belong to several AEs), which splits links across sub-components and lowers the per-project top share. The top-AE concentration is confirmed qualitatively (one component dominates) but the exact 44-48% per-project values need the paper's coarser AE grouping.

## Resolved `XX` placeholders (intro.tex)

| intro.tex loc | Placeholder | Resolved value |
|---------------|-------------|----------------|
| `intro.tex:40` | the XX projects of the benchmark | 5 |
| `intro.tex:79` | an evaluation suite of XX complementary metrics | 4 |
| `intro.tex:64` | an XX% concentration of the gold mass on three sentences of one project | 70% (JabRef) |
| `intro.tex:17` | strongest published pipeline ... file-level F1 of XX; roughly XX unrecovered | deferred -> Phase 3 (needs system scores) |
| `intro.tex:54` | \approach ... file-level F1 of XX; improving ... by XX percentage points | deferred -> Phase 3 (needs system scores) |
| `intro.tex:64` | trivial substring-match baseline ... file-level F1 of XX; within XX points; within XX point on one project | deferred -> Phase 3 (baseline scores; MOTIV-01) |

## Excluded (system-specific)

- **Cascade / error amplification** (`writing/eval.tex` `tab:amplification`, 36→3,457, 96.0×; block correlation): a TransArc *actual-error* attribution (real sad-code FPs decomposed by transitive cause — see `reports/TRANSARC_EMPIRICAL_STUDY.md`), NOT a gold/benchmark property. It is intentionally out of scope for this dataset-inequality study and is recorded, not audited as MATCH/MISMATCH.

