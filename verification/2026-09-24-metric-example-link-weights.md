# Metric example link-weight check, 2026-09-24

## Configuration

- Scope: illustrative five-component example in `paper/sections/metric.tex`.
- Illustrative gold-link counts: `(20, 40, 1500, 2800, 3900)`; per-component scores: `(0.2, 0.9, 0.9, 0.9, 0.9)`.
- The counts are rounded and illustrative, not observed component counts. Their top-three share is 99.3%, matching the highest doc-code top-three share in `paper/table/gold_concentration.csv` (JabRef).
- The link-count-weighted arithmetic mean is an illustration of unequal link weights, not an assertion that it always equals pooled link-level $F_\beta$.

## Commands and text results

From the repository root:

```bash
python3 - <<'PY'
import csv
from statistics import harmonic_mean, mean
scores = (0.2, 0.9, 0.9, 0.9, 0.9)
links = (20, 40, 1500, 2800, 3900)
weighted = sum(n * s for n, s in zip(links, scores)) / sum(links)
with open('paper/table/gold_concentration.csv', newline='') as f:
    observed = max(float(row['doc_code_top3_pct']) for row in csv.DictReader(f))
top3 = 100 * sum(sorted(links, reverse=True)[:3]) / sum(links)
print(f'harmonic={harmonic_mean(scores):.6f}')
print(f'arithmetic={mean(scores):.6f}')
print(f'link-count-weighted arithmetic={weighted:.6f}')
print(f'link-count-weighted arithmetic (paper precision)={weighted:.4f}')
print(f'illustrative top-three share={top3:.1f}%; observed maximum={observed:.1f}%')
assert (round(harmonic_mean(scores), 2), round(mean(scores), 2), round(weighted, 4)) == (0.53, 0.76, 0.8983)
assert round(top3, 1) == observed == 99.3
print('PASS: displayed values and concentration comparison')
PY
```

```text
harmonic=0.529412
arithmetic=0.760000
link-count-weighted arithmetic=0.898305
link-count-weighted arithmetic (paper precision)=0.8983
illustrative top-three share=99.3%; observed maximum=99.3%
PASS: displayed values and concentration comparison
```

```bash
git -C paper diff --check
```

```text
(no output; exit 0)
```

```bash
./scripts/build-paper.sh
```

```text
latexmk is required to build the paper (install TeX Live with latexmk).
```

The manuscript build exited 1 because `latexmk` is not installed in this environment.
