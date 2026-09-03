# Candidate: Kubernetes

Measured 2026-09-03 from blobless/sparse clones under `/tmp/oss-case/kubernetes/`
(website @cf96ee6, enhancements @e849163, community @e7a51e1, kubernetes @62a78fd,
design-proposals-archive @acc25e1 [frozen 2021-12-01]). Sentence counts use
`/tmp/oss-case/kubernetes/split.py` (strips front matter, code blocks, headings, tables,
HTML, Hugo shortcodes but keeps `glossary_tooltip` text; splits on `[.!?] + capital`;
drops fragments < 4 words). Per-file outputs are the `*.sents.txt` siblings of each source.

## D1 — Architecture prose

| Source | Lines | Sentences | Licence | Tracked |
|---|---|---|---|---|
| website `content/en/docs/concepts/architecture/_index.md` ("Cluster Architecture") | 216 | 60 | CC-BY-4.0 | yes |
| `.../architecture/nodes.md` | 312 | 99 | CC-BY-4.0 | yes |
| `.../architecture/garbage-collection.md` | 207 | 67 | CC-BY-4.0 | yes |
| `.../architecture/controller.md` | 170 | 55 | CC-BY-4.0 | yes |
| `.../architecture/control-plane-node-communication.md` | 128 | 39 | CC-BY-4.0 | yes |
| `.../architecture/cloud-controller.md` | 217 | 31 | CC-BY-4.0 | yes |
| `.../architecture/leases.md` | 120 | 28 | CC-BY-4.0 | yes |
| `.../architecture/cgroups.md` | 148 | 25 | CC-BY-4.0 | yes |
| `.../architecture/mixed-version-proxy.md` | 121 | 25 | CC-BY-4.0 | yes |
| `.../architecture/self-healing.md` | 52 | 6 | CC-BY-4.0 | yes |
| website `content/en/docs/concepts/overview/components.md` | 90 | 20 | CC-BY-4.0 | yes |
| design-proposals-archive `architecture/architecture.md` (2017-era, archived) | 251 | 80 | Apache-2.0 | yes (frozen) |
| **D1 corpus total** (`/tmp/oss-case/kubernetes/d1-corpus.txt`) | | **537** | | |
| (optional) design-proposals-archive `architecture/architectural-roadmap.md` | 1132 | 135 | Apache-2.0 | frozen |

URLs: https://github.com/kubernetes/website/tree/main/content/en/docs/concepts/architecture ,
https://github.com/kubernetes/website/blob/main/content/en/docs/concepts/overview/components.md ,
https://github.com/kubernetes/design-proposals-archive/blob/main/architecture/architecture.md .
Mean sentence length 20.3 words. Only 3/537 sentences contain a code path (`pkg/`, `cmd/`, `k8s.io/`).

Component-naming sentences: 167/537 (31%) name at least one of the 12 core names
(regex over kube-apiserver|API server|apiserver|scheduler|controller manager|kubelet|kube-proxy|etcd|
container runtime|kubectl|kubeadm). Naming form is **mixed and mostly prose, not binary names**:

| form | sentences | form | sentences |
|---|---|---|---|
| "API server" | 57 | `kube-apiserver` | 15 |
| bare "apiserver" | 16 | "the control plane" | 27 |
| "kubelet" (never "kube-let") | 55 | "controller manager" | 13 |
| `kube-controller-manager` | 7 | "scheduler" | 12 |
| `kube-scheduler` | 3 | "container runtime" | 13 |
| `kube-proxy` | 7 | `etcd` | 4 |

So the doc uses the binary name (`kube-apiserver`) 15 times but the descriptive noun ("the API server")
73 times; a component list keyed on binary names needs alias handling (which the linker's
dynamic alias step is meant to do, but "scheduler" and "controller" are also generic nouns here).

Ten example sentences (file:line in `*.sents.txt`):
1. leases:3 "Kubernetes uses the Lease API to communicate kubelet node heartbeats to the Kubernetes API server."
2. leases:16 "You can inspect Leases owned by each kube-apiserver by checking for lease objects in the `kube-system` namespace with the name `apiserver-`."
3. control-plane-node-communication:6 "The API server is configured to listen for remote connections on a secure HTTPS port (typically 443) with one or more forms of client authentication enabled."
4. control-plane-node-communication:16 "The first is from the API server to the kubelet process which runs on each node in the cluster."
5. control-plane-node-communication:23 "If that is not possible, use SSH tunneling between the API server and kubelet if required to avoid connecting over an untrusted or public network."
6. cloud-controller:6 "You can also run the cloud controller manager as a Kubernetes addon rather than as part of the control plane."
7. nodes (corpus:402) "This period can be configured using the `--node-monitor-period` flag on the `kube-controller-manager` component."
8. _index (corpus:288) "This is used by control plane components like `kube-controller-manager` and `kube-scheduler` in HA configurations, where only one instance of the component should be actively running while the other instances are on stand-by."
9. archive architecture:76 "Each node runs a kube-proxy process which programs `iptables` rules to trap access to service IPs and redirect them to the correct backends."
10. archive architecture:57 "The scheduler watches for unscheduled pods and binds them to nodes via the `/binding` pod subresource API, according to the availability of the requested resources, ..."

## D2 — Component model

| Level | Source | Count | Names match doc strings? |
|---|---|---|---|
| (a) core components, headings of `concepts/architecture/_index.md` + `components.md` | website | **8 binaries** (kube-apiserver, etcd, kube-scheduler, kube-controller-manager, cloud-controller-manager, kubelet, kube-proxy, container runtime) **+ 5 addon categories** (DNS, Web UI, container resource monitoring, cluster-level logging, network plugins) = 13 | yes, verbatim (this is the doc) |
| (b) `cmd/*` in kubernetes/kubernetes | `git ls-tree` | 26 dirs; **10 are shipped binaries** (cloud-controller-manager, kubeadm, kube-apiserver, kube-controller-manager, kubectl, kubectl-convert, kubelet, kubemark, kube-proxy, kube-scheduler); 16 are gen/check tools | binary names = doc's hyphenated form |
| (b') `staging/src/k8s.io/*` library modules | `git ls-tree` | 33 (apiserver, apimachinery, client-go, ...) | no: doc never names modules |
| (c) SIGs in `community/sigs.yaml` | parsed with PyYAML | **24 SIGs** (+8 WGs, 3 committees); 234 subprojects, 159 of which point their OWNERS into k/k | no: doc never says "SIG Node" |
| (d) `area/` labels for k/k in test-infra `label_sync/labels.yaml` | parsed | **78** `area/` labels (many dead: rkt, mesos, federation, os/fedora); only **16** distinct `area/` labels actually appear in k/k OWNERS files | 5 areas are binary names (kubelet, kube-proxy, apiserver, kubectl, kubeadm) |

SIG list (24): api-machinery, apps, architecture, auth, autoscaling, cli, cloud-provider,
cluster-lifecycle, contributor-experience, docs, etcd, instrumentation, k8s-infra, multicluster,
network, node, release, scalability, scheduling, security, storage, testing, ui, windows.

Recommendation: **level (a)+(b) = the 10 shipped binaries plus etcd, container runtime, CoreDNS
(13 components)** as the primary model. It is the only level whose names occur in the prose
(D1 counts above), and it is project-authored (components.md). It is *below* the rubric's
15–80 ideal and only marginally above the benchmark's 6–14; the "large" dimension of Kubernetes
is sentences, not components. The 24 SIGs are a real 20–80-sized organisational model, but they
are teams, not architectural parts, and the prose never names them (0/537 sentences mention "SIG").
A SIG-level run would be a routing experiment, not architecture TLR.

## D3 — Code -> component map

Artifact: 595 `OWNERS` files in kubernetes/kubernetes (https://github.com/kubernetes/kubernetes/blob/master/OWNERS_ALIASES ;
paths in `/tmp/oss-case/kubernetes/k8s-owners-paths.txt`), 366 of which carry a `labels:` field.
Coverage measured by walking each file's nearest-ancestor OWNERS chain (`k8s-tree.txt`, 31,328 files, 17,880 `.go`):

| File set | n | has `sig/` label | has `area/` label |
|---|---|---|---|
| .go excl. vendor | 13,426 | 12,176 (**90.7%**) | 5,667 (42.2%) |
| .go excl. vendor+staging | 5,678 | 4,776 (84.1%) | 2,928 (51.6%) |
| all files excl. vendor | 25,945 | 20,925 (80.7%) | 8,998 (34.7%) |

Top-level dirs: 26 `cmd/*` -> 12 have a sig/area label, 6 OWNERS-without-labels, 8 no OWNERS
(all 8 unlabelled-or-missing are generator tools). 31 `pkg/*` -> 14 labelled, 8 OWNERS-without-labels
(pkg/api, pkg/apis, pkg/registry, pkg/features ...), 9 no OWNERS. Granularity is directory, nested
(pkg/controller = sig/apps, pkg/controller/garbagecollector = sig/api-machinery).

Caveat: OWNERS maps dirs -> **SIG**, not -> binary. Only the 5 binary-named `area/` labels give a
dir -> component map, and they cover 42% of non-vendor Go files. `sigs.yaml` additionally maps
159 k/k OWNERS paths to (SIG, subproject) pairs, e.g. sig-api-machinery/server-binaries ->
cmd/kube-apiserver, sig-node/kubelet -> pkg/kubelet; that is the cleanest project-authored
dir -> component table but it names only ~60 subprojects.

## S1 — Sentence-level self-supervised gold

Measured over the 537-sentence D1 corpus (`grep` on `d1-corpus.txt`):

| Signal | Sentences | Notes |
|---|---|---|
| link to `/docs/concepts/architecture/#<component>` anchor | 10 | all in components.md/self-healing.md bullet lists ("kubelet : Ensures that Pods are running") |
| link to `/docs/reference/command-line-tools-reference/<binary>/` | 3 | 2 of the 3 are feature-gate links, not component links |
| component name in backticks | 11 | e.g. "`kube-apiserver` uses the Lease API" |
| `glossary_tooltip term_id=` on a core-component term (raw source) | 16 of 57 tooltips (kube-apiserver 5, kubelet 4, kube-proxy 2, kube-scheduler/kcm/container-runtime/cloud-provider/cadvisor 1 each) | remaining 41 tooltips are concepts (pod, node, controller, service) |
| code path (`pkg/`, `cmd/`, `k8s.io/`) | 3 | |
| any markdown link | 128 | mostly to concept pages, not components |
| doc-file ownership | 0 | website has 2 OWNERS files under content/en/docs (root + issues-security); no per-page SIG |
| co-change doc line + code dir | n/a | docs and code are in different repos; no same-commit signal |

Union of the component-resolving signals (anchor, reference link, backticks, tooltip) is about
**35 sentences (6.5%)**, and after de-duplication with the anchor-list bullets fewer than 30 are
ordinary prose. Spot-check of 10 (corpus lines 228, 288, 292, 300, 402, 435, 441, 443, 445, 500):
9/10 the linked/backticked component is the true subject; 1/10 (line 300, feature-gate link) is not.
Noise is low but volume is small: this is not enough to score a 537-sentence run; it is a
sanity check at best. The dominant naming form ("the API server") carries no markup.

## S2 — Doc-level gold

KEPs, https://github.com/kubernetes/enhancements/tree/master/keps (Apache-2.0, git-tracked):
- `kep.yaml` files: **656** (654 parseable). Distinct `owning-sig`: **23** (22 real + one `sig-xyz` template artefact).
  Distribution: node 126, api-machinery 81, storage 66, network 61, scheduling 55, cli 35, auth 35,
  apps 34, cloud-provider 31, cluster-lifecycle 28, instrumentation 20, architecture 16, windows 15,
  release 15, autoscaling 9, testing 8, etcd 5, multicluster 5, contrib-ex 3, security 3, docs/ui 1 each.
- KEPs with non-empty `participating-sigs`: **403/656 (61%)**, so the single-label view is a simplification for most.
- KEP `README.md` sentence counts, 5 samples: node-swap 312, pod-scheduling-readiness 112,
  crd-validation-expression-language 291, iptables-cleanup 102, volume-health-monitor 333
  (sum 1,150; 217 of those 1,150 name a core binary). README line-length distribution over all 648:
  min 42, p25 309, median 516, p75 922, max 3,449 lines -> roughly 150k sentences project-wide (extrapolated, not measured).

Proposed SIG -> component mapping and coverage of the 654 KEPs (`/tmp/oss-case/kubernetes/` python calc):
- strict (unambiguous 1 SIG -> 1 binary): node->kubelet, scheduling->kube-scheduler,
  api-machinery->kube-apiserver, cli->kubectl, cluster-lifecycle->kubeadm,
  cloud-provider->cloud-controller-manager, etcd->etcd: **361/654 = 55.2%**
- loose (+ network->kube-proxy, apps and autoscaling->kube-controller-manager): **465/654 = 71.1%**
- unmappable to a binary (storage, auth, instrumentation, architecture, windows, release, testing, ...): 189 = 28.9%.
The loose additions are questionable: sig-network KEPs are mostly Service/EndpointSlice API and
CNI, not kube-proxy; sig-storage spans kubelet volume plugins, KCM controllers and CSI sidecars.
So the honest doc-level gold is **SIG**, not component, for ~45% of KEPs.

## T1 — Downstream tasks

(i) **KEP -> owning SIG routing.** Protocol: feed KEP README sentences + component/SIG list to
the linker, aggregate per-KEP link mass, predict owning-sig; score against `kep.yaml owning-sig`
(23 classes, 654 docs; strict-mapped 7-class subset 361 docs). Gold is authoritative and
git-tracked. Size: 654 docs x ~230 sentences. Caveat: the directory `keps/sig-*/` already
encodes the label, so any input that leaks the path is trivially solved; strip paths.

(ii) **Issue triage against `sig/` and `area/` labels.** GitHub search API, repo kubernetes/kubernetes,
`is:issue` all-time counts: sig/node 8,721; sig/api-machinery 6,762; sig/network 4,170;
sig/scheduling 2,716; area/kubelet 886; area/kube-proxy 318. Open issues 1,857, of which only 85
(4.6%) lack a `sig/` label, so labels are near-complete gold. Caveat: labels are applied by
`/sig` commands and triage bots reacting to the same text a linker would read; area/ labels are
sparse (886 kubelet issues vs 8,721 node) so component-level triage gold is ~10x thinner than SIG-level.

(iii) **Doc staleness across kubernetes/website** (1,717 .md files under content/en/docs, grep for removed things):
dockershim 40 sentences / 13 files; PodSecurityPolicy 25 / 11; kube-dns 28 / 20; Docker Engine 20 / 12;
heapster 4 / 4; rkt 0. But reading them: every dockershim sentence sits in the migration guide,
removal FAQ, or a "no longer" sentence; of the 12 PSP sentences my "not-marked-removed" regex kept,
all are link text or `podsecuritypolicy` API-name fragments. **Kubernetes' docs team purges stale
component references actively, so the staleness task has close to zero positives here** (the
concept page for PSP is itself a removal notice). Do not pitch staleness for this system.

## K — Killer-case pitch

Kubernetes is the largest documented control-plane architecture in OSS, and its 654 design
proposals already carry an authoritative owner label: if sentence-level links from prose to
{kube-apiserver, kubelet, kube-scheduler, ...} are real, they should route a KEP to its SIG
without reading the path, which is a task 24 SIG leads perform by hand every release cycle.

## C — Cost / feasibility

- Architecture run: 537 sentences x 13 components (~7k pairs); ~3x benchmark sentence volume,
  same component count. Extending to KEP READMEs: ~150k sentences x 13 (or x 24 SIGs); one-call
  s110 cost is per sentence, so budget with a 5–20 KEP subsample (~1.2k–5k sentences).
- Licences: website CC-BY-4.0 (attribution only), design-proposals-archive / enhancements /
  community / kubernetes Apache-2.0. All vendorable into the replication package with a
  NOTICE line; no CLA or NC clause.
- Vendoring: 12 D1 source files (~2,000 lines) + `components.md` + sigs.yaml (192 KB) +
  ~650 kep.yaml (small) + selected READMEs. Reproducible from pinned commits above.
- Tooling risk: Hugo shortcodes (`glossary_tooltip`, `skew`, `note`) mangle naive stripping;
  splitter must resolve them or component mentions vanish (my first pass lost 8 of them).

## Verdict

READY-WITH-WORK: prose (537 sentences, CC-BY) and a project-authored 13-component list exist and use compatible names, but sentence-level gold is thin (about 35 marked sentences, 6.5%) and the doc-level gold (KEP owning-sig) maps cleanly to a binary for only 55% of KEPs.
Work needed: hand-label a 150–200 sentence slice of D1 for sentence-level scoring; pick SIG-routing (23-class, 654 docs) as the downstream task and drop staleness (no positives).
Scale story is sentences, not components: 13 components is not the 20–80 the pitch needs, unless the SIG level (24) is used as a second, organisational model.
