# Metric example link-weight check, 2026-09-24

## Configuration

- Scope: illustrative five-component example in `paper/sections/metric.tex`.
- Gold-link counts: `(10, 100, 100, 100, 100)`; per-component scores: `(0.2, 0.9, 0.9, 0.9, 0.9)`.
- The link-count-weighted arithmetic mean is an illustration of unequal link weights, not an assertion that it always equals pooled link-level $F_\beta$.

## Commands and text results

From the repository root:

```bash
python3 - <<'PY'
from statistics import harmonic_mean, mean
scores = (0.2, 0.9, 0.9, 0.9, 0.9)
links = (10, 100, 100, 100, 100)
weighted = sum(n * s for n, s in zip(links, scores)) / sum(links)
print(f'harmonic={harmonic_mean(scores):.6f}')
print(f'arithmetic={mean(scores):.6f}')
print(f'link-count-weighted arithmetic={weighted:.6f}')
assert (round(harmonic_mean(scores), 2), round(mean(scores), 2), round(weighted, 2)) == (0.53, 0.76, 0.88)
print('PASS: all three displayed rounded values')
PY
```

```text
harmonic=0.529412
arithmetic=0.760000
link-count-weighted arithmetic=0.882927
PASS: all three displayed rounded values
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
