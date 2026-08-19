# s25 post-pilot five-project E2E — 2026-08-10

Three independent runs of the linker as it stands after the design pilots
(`results/s25_design_pilots/`). Same model and settings as the pre-change
promoted run `results/s25_cleanup_verify_20260810`: `gpt-5.6-terra`, OpenAI
backend, reasoning effort `none`. Runs are in
`results/s25_postpilot_e2e_r{1,2,3}_20260810/`.

| Project | F1 per run | mean F1 | TP per run | FP per run |
|---|---|---|---|---|
| mediastore | 98.4 / 98.4 / 98.4 | 98.4 | 30 / 30 / 30 | 0 / 0 / 0 |
| teammates | 97.4 / 89.1 / 95.5 | 94.0 | 56 / 49 / 53 | 2 / 4 / 1 |
| teastore | 98.1 / 98.1 / 98.1 | 98.1 | 26 / 26 / 26 | 0 / 0 / 0 |
| bigbluebutton | 85.5 / 89.1 / 90.6 | 88.4 | 53 / 53 / 53 | 9 / 4 / 2 |
| jabref | 94.7 / 94.7 / 94.7 | 94.7 | 18 / 18 / 18 | 2 / 2 / 2 |

| | post-change (N=3) | pre-change (N=1) |
|---|---|---|
| macro F1 | **94.7** ± 0.8 (93.9 / 94.8 / 95.5) | 94.2 |
| pooled F1 | **93.6** ± 1.2 (92.4 / 93.6 / 94.7) | 91.6 |
| TP | 179.7 (183 / 176 / 180) | 179 |
| FP | **9.3** (13 / 10 / 5) | 17 |
| FN | 15.3 (12 / 19 / 15) | 16 |

Recall is flat and precision is the gain, which is what the pilots predicted:
the alias-scope removal was the only recall-positive change (+3.0 TP) and the
subtraction was the precision one (-6.8 FP). Both effects survive composition.

Two caveats. The pre-change column is a single run, so the FP comparison is one
run against three; the per-change evidence is the paired, five-run-per-side
pilots, not this table. And teammates and bigbluebutton carry all of the
run-to-run spread -- mediastore, teastore and jabref returned identical link
sets on all three runs.
