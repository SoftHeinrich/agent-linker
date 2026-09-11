# Kubernetes — the second benchmark-style system (descriptive names)

Built 2026-09-04 next to `../gitlab/` (read that README first; same pipeline, same gold
recipe, same scorer — every script there takes `OSS_DIR=$(pwd)`).

## 1. Source and dataset (`build.py`, pinned)

`kubernetes/website` @ `cf96ee6` (2026-09-03): `content/en/docs/concepts/overview/components.md`
(the project's component list with one-line descriptions) plus the eleven pages of
`content/en/docs/concepts/architecture/` — "Cluster Architecture", nodes, controllers,
control-plane/node communication, leases, cloud controller manager, cgroups, garbage
collection, mixed-version proxy, self-healing. User-facing architecture prose about how
the components interact, the BigBlueButton pattern (a docs "Architecture" section).

| | |
|---|---|
| sentences | **576** (Hugo shortcodes resolved, glossary tooltips kept as their text; 45 are definition-list items without a full stop) |
| components | **12**, as `components.md` lists them: kube-apiserver, etcd, kube-scheduler, kube-controller-manager, cloud-controller-manager, kubelet, kube-proxy, Container runtime, DNS, Web UI (Dashboard), Container Resource Monitoring, Cluster-level Logging |
| structural prior | 22 pairs (the per-component subsections of `_index.md`) |

`data/annotator_notes.txt` tells the *annotator* (not the linker) the document's
descriptive names ("the API server" = kube-apiserver, "the scheduler" = kube-scheduler, …)
— these are the page's own definitions and the aliases the linker is expected to discover.

## 2. Gold (`../gitlab/annotate.py`, `../gitlab/label_model.py`)

| | |
|---|---|
| gold / gold_plus pairs | 281 / **297** on 281 sentences (**0.49** of the document), all 12 components |
| κ terra–claude | **0.83**; component-view vs gold κ 0.91 |
| terra pairs reproduced in all 3 runs | 0.93 |
| structural prior confirmed | 22/22 |
| gold_plus pairs naming the component verbatim | **0.32** (name-echo, e.g. "the API server", "the scheduler": 0.25; no surface: 0.43) |
| tiers | gold 281, gold_plus_only 16, silver 91, refers 30 |

Style row (`../tools/style_table.py`): sent w/ gold 0.49, verbatim 0.32, name-echo 0.25,
capitalised names 0.42, hyphenated binary names 0.50, shared-word names 0.58 (`kube` ×4,
`controller` ×2, `container` ×2). Gold density is the benchmark's; the explicit share is
below the benchmark's lowest (BigBlueButton 0.55, name-echo 0.31) because the prose names
the binaries by what they do. Kubernetes is therefore the *alias* case, not the
proper-noun case GitLab is.

## 3. Results (three s110 runs, three one-call runs, one SWATTR run; gold_plus)

`results/oss_scale_k8s_*_20260904`; `run_s110.sh`, `run_onecall.sh`, `../tools/run_swattr.sh kubernetes k8s_arch`.

| arm | links | TP | P | R | F1 | P lenient | R explicit | R implicit |
|---|---|---|---|---|---|---|---|---|
| s110 as shipped | 453 | 190.7 | 0.421 | 0.642 | **0.508** | 0.478 | 0.972 | 0.484 |
| — full-name stage | 173 | 135.7 | 0.784 | 0.457 | 0.577 | 0.934 | 0.962 | 0.216 |
| — partial-name stage | 278 | 52.7 | 0.190 | 0.177 | 0.183 | 0.190 | 0.003 | 0.260 |
| — coreference stage | 2.3 | 2.3 | 1.000 | 0.008 | 0.016 | 1.000 | 0.007 | 0.008 |
| s110 minus partial-name | 175 | 138.0 | 0.787 | 0.465 | **0.584** | 0.935 | 0.969 | 0.224 |
| `s_linker110_onecall` (3 calls) | 187 | 150.0 | 0.803 | 0.505 | **0.619** | 0.927 | 0.983 | 0.277 |
| SWATTR (ArDoCo, deterministic) | 104 | 45.0 | 0.433 | 0.152 | **0.224** | 0.462 | 0.458 | 0.005 |

F1 spread over runs: s110 0.501–0.517, minus partial 0.576–0.597, one-call 0.590–0.665.
Cost: s110 92 calls / ~207 s a run; one-call 3 calls / ~14 s. Under the strict, Claude-only
and three-way golds the ordering is unchanged (minus-partial 0.596 / 0.586 / 0.612; one-call
0.625 / 0.619 / 0.636; SWATTR 0.229 / 0.235 / 0.235).

### 3.1 What differs from GitLab

1. **The partial-name stage is a different animal here: cheap recall, not pure noise.** Its
   278 links carry 53 true ones (P 0.19), 38 of them from the word `controller` (127 links)
   — which is both half of two component names and the document's own term for the things
   `kube-controller-manager` runs. `cluster`, `manager`, `container`, `resource` fire 55
   times with 0 TP. It is the only stage that reaches the no-surface stratum (R implicit
   0.26 vs the full-name stage's 0.22), so dropping it costs 18 pp recall for 37 pp
   precision; F1 still rises (0.51 → 0.58). Same rule as GitLab predicts the losers from
   the catalog (`kube` ×4 names, `controller` ×2, `container` ×2) — and the one word that
   pays is the one the prose uses as a common noun.
2. **The alias step does its job.** The full-name stage hits 0.96 of the explicit stratum
   although only 32% of gold pairs name the binary verbatim: "the API server", "the
   scheduler", "the controller manager", "the container runtime" are resolved from the
   document. The remaining full-name false positives (37 a run) are 26 REFERS and 11 silver
   — mostly `kube-apiserver` mentioned as the thing something talks to.
3. **The implicit tail is again continuation, not reference.** 101 of the 201 implicit
   gold pairs are linked by no run at all; none of them carries a descriptive alias. They
   are lines inside a component's own page ("--register-node - Automatically register with
   the API server." → kubelet; "To update Services, it requires patch and update access to
   the status subresource." → cloud-controller-manager), the same topic-continuation shape
   §9.5/§9.8 of the top-level README measured on rustc. Recall on that half is where every
   arm loses (best: one-call 0.28).
4. **One-call again edges the workflow** (+3.5 pp F1 over minus-partial, +11 pp over s110
   as shipped), with the widest run-to-run spread of any arm (0.59–0.67). It links less
   (187 vs 453) at the same precision as the full-name stage and slightly more implicit
   recall.
5. **SWATTR stays at F1 0.22**: R explicit 0.46 (hyphenated `kube-*` names are half
   noun-phrase, half not), R implicit 0.005.
