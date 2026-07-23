# JabRef component importance vs. code footprint

Graph: 1451 files, 9089 dependency edges (open-source Depends; ArDoCo/jabref @ pinned commit, see data/PROVENANCE.md).

| component | files | Ca | Ca_share | Ce | Instab. | CBO | betw. rank |
|---|--:|--:|--:|--:|--:|--:|--:|
| model | 198 | 716 | 57.1% | 28 | 0.04 | 721 | #1 of 1451 |
| logic | 575 | 428 | 48.9% | 162 | 0.27 | 565 | #9 of 1451 |
| preferences | 18 | 270 | 18.8% | 76 | 0.22 | 342 | #7 of 1451 |
| globals | 1 | 97 | 6.7% | 22 | 0.18 | 117 | #6 of 1451 |
| gui | 641 | 25 | 3.1% | 345 | 0.93 | 352 | #2 of 1451 |
| cli | 5 | 1 | 0.1% | 63 | 0.98 | 64 | #89 of 1451 |

- Boundary alignment: 53.4% of reference weight stays within a component; directed modularity Q = 0.246.
- **Footprint != importance:** `preferences` (few files) and `globals` (one file) are depended on by more code than the largest component `gui`.
- `Ca` (afferent coupling) and `Instability` are the size-independent metrics to cite; `cli` is not depended upon (argue it via role, not coupling).
