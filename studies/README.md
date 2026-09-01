# studies/ — side analyses that do not write a paper table

`evaluation/` is the table pipeline: everything in it exists because some float in
`paper/{table,appendix}/` is generated from it, and `mini-src/sync_paper.py --check`
guards that bridge. The studies here are the other half — the probes, companions and
audits that informed the paper's *choices* without producing any of its numbers. They
were moved out of `evaluation/` on 2026-08-27 so that directory reads as the pipeline
it is.

| Dir | Question it answered | Status |
|-----|----------------------|--------|
| [`explore-tail/`](explore-tail/README.md) | Is any tail/coverage summary more independent of link-F1 than worst-comp / harmonic-comp F1? | Answered: independence lives in coverage-*counting*, which is why the reported suite carries the component miss rate (CMR, called Silent-Failure Mass here) and dropped component coverage. The count twin (CMC/SFC) was reported beside it until 2026-09-01, then dropped unread. |
| [`mini-depimport/`](mini-depimport/README.md) | Does a component's code footprint predict how much of the codebase depends on it? | Answered: no. Cited as frozen provenance by `paper/figures/jabref_motivation.py`; not recomputable from the benchmark alone. |
| [`mini-inequality/`](mini-inequality/CLAIM_CHECK.md) | Does every distributional-inequality claim in the paper match the computed value? | One-off audit (`claim_check.py`). The engine it audits stayed in `evaluation/mini-inequality/`, because `motivation.py` beside it writes `paper/table/gold_concentration.tex`. |

## Running them

Stdlib-only Python 3, same as `evaluation/`. Each script reaches back into
`evaluation/` for the scorer rather than forking it, so a metric change lands in the
side analyses too:

```bash
python3 studies/explore-tail/explore.py         # -> explore-tail/reports/{cells,corr}_*.csv
python3 studies/mini-depimport/depimport.py     # -> mini-depimport/reports/*.csv
python3 studies/mini-inequality/claim_check.py  # -> mini-inequality/CLAIM_CHECK.md
```

`explore-tail/` resolves its three roots in [`explore-tail/_roots.py`](explore-tail/_roots.py):
`evaluation/mini-src`, `evaluation/mini-rq34`, and the link dump (`$SOTA_LINKS`,
default `sota-links/`). Point `$SOTA_LINKS` elsewhere to score a different dump.

> These are exploratory by design. They are **not** covered by `mini-src/check.py`
> and their frozen findings describe the arms current when they ran (mostly the
> retired s21 arm) — read the numbers as the reasoning behind a decision, not as
> current measurements. Re-run them against s92a before citing any figure.
