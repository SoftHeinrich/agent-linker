#!/usr/bin/env python3
"""Dependency-based component importance for the ARDoCo JabRef benchmark.

Companion to mini-inequality: it shows that a component's *code footprint*
(how many files/model-code links it owns) does not predict its *importance*
(how much of the codebase depends on it). Small-footprint components can be
heavily depended upon, so file/link-level metrics under-weight them.

INPUTS  (no network, no build, stdlib only)
  data/jabref-maindeps-file.json.gz  bundled file-level dependency graph produced by
                                      the open-source `Depends` tool (see data/PROVENANCE.md).
  <benchmark>/jabref/goldstandards/goldstandard_sam_2021-code_2023.csv
                                      the model-code (SAM->code) links -> component<-file map.
                                      Path from $TRANSARC_BENCHMARK or the default sibling layout.

OUTPUTS  reports/COMPONENT_VALUES.csv, reports/IMPORTANCE.csv, reports/DEPIMPORT.md

METRICS  (definitions + citations in README.md / METRICS section)
  Ca, Ce, Instability I=Ce/(Ca+Ce)          Martin 1994   (type/file granularity)
  CBO                                        Chidamber-Kemerer 1994
  Ca_share = Ca / (all other files)          derived intensive ratio (afferent reach)
  PageRank                                   Page & Brin 1999   (power iteration)
  betweenness                                Freeman 1977       (Brandes 2001, unweighted)
  cohesion / directed modularity Q           boundary alignment vs the model-code partition

Usage:  python3 depimport.py            # compute + write reports
        python3 depimport.py --check    # also assert the frozen panel (CI-style)
"""
import csv, gzip, json, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BENCH = "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
BENCH = os.environ.get("TRANSARC_BENCHMARK", DEFAULT_BENCH)
GOLD = os.path.join(BENCH, "jabref", "goldstandards", "goldstandard_sam_2021-code_2023.csv")
GRAPH = os.path.join(HERE, "data", "jabref-maindeps-file.json.gz")
OUT = os.path.join(HERE, "reports")
COMP_ORDER = ["model", "logic", "gui", "cli", "preferences", "globals"]


# ---------------------------------------------------------------- component map
def load_component_rules(gold_path):
    """Derive (exact-file overrides, longest-prefix rules) from the model-code gold standard.

    Only main-source paths are used for the dependency analysis; test/buildSrc paths in the
    gold standard simply never match a graph node (the graph is built over src/main/java).
    """
    exact, prefixes = {}, []
    with open(gold_path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["ae_name"].replace("Component: ", "").strip()
            path = row["ce_ids"].strip()
            if path.endswith(".java"):
                exact[path] = name
            else:
                prefixes.append((path.rstrip("/") + "/", name))
    prefixes.sort(key=lambda p: -len(p[0]))          # longest prefix wins
    return exact, prefixes


def component_of(rel, exact, prefixes):
    if rel in exact:
        return exact[rel]
    for pref, name in prefixes:
        if rel.startswith(pref):
            return name
    return None                                       # unmapped (architecture, migrations, ...)


def norm(path):
    """Absolute node path -> repo-relative path (decouple from the clone location)."""
    key = "src/main/java/"
    i = path.find(key)
    return path[i:] if i >= 0 else path


# ---------------------------------------------------------------- graph loading
def load_graph():
    with gzip.open(GRAPH, "rt") as fh:
        d = json.load(fh)
    nodes = [norm(n) for n in d["variables"]]
    edges = []                                        # (u, v, weight): u depends on v
    for cell in d["cells"]:
        u, v = cell["src"], cell["dest"]
        if u == v:
            continue
        edges.append((u, v, sum(cell["values"].values())))
    return nodes, edges


# ---------------------------------------------------------------- centralities (stdlib)
def pagerank(n, out_adj, alpha=0.85, iters=100, tol=1e-12):
    pr = [1.0 / n] * n
    out_w = [sum(w for _, w in out_adj[i]) for i in range(n)]
    dangling = [i for i in range(n) if out_w[i] == 0]
    for _ in range(iters):
        nxt = [(1 - alpha) / n] * n
        dsum = alpha * sum(pr[i] for i in dangling) / n
        for i in range(n):
            if out_w[i] == 0:
                continue
            share = alpha * pr[i] / out_w[i]
            for j, w in out_adj[i]:
                nxt[j] += share * w
        for j in range(n):
            nxt[j] += dsum
        if sum(abs(nxt[i] - pr[i]) for i in range(n)) < tol:
            pr = nxt
            break
        pr = nxt
    return pr


def betweenness(n, adj):
    """Brandes (2001), unweighted, normalized — matches networkx betweenness_centrality."""
    bc = [0.0] * n
    for s in range(n):
        S, P = [], [[] for _ in range(n)]
        sigma = [0] * n; sigma[s] = 1
        dist = [-1] * n; dist[s] = 0
        Q = collections.deque([s])
        while Q:
            v = Q.popleft(); S.append(v)
            for w, _ in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1; Q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]; P[w].append(v)
        delta = [0.0] * n
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    scale = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0   # directed normalization
    return [b * scale for b in bc]


# ---------------------------------------------------------------- main compute
def compute():
    exact, prefixes = load_component_rules(GOLD)
    nodes, edges = load_graph()
    n = len(nodes)
    comp = [component_of(r, exact, prefixes) for r in nodes]
    files = {c: [i for i in range(n) if comp[i] == c] for c in COMP_ORDER}

    out_adj = [[] for _ in range(n)]
    adj_unw = [[] for _ in range(n)]
    for u, v, w in edges:
        out_adj[u].append((v, w))
        adj_unw[u].append((v, 1))

    # coupling sets at file/type granularity
    ca_files = collections.defaultdict(set); ce_files = collections.defaultdict(set)
    fanin_w = collections.Counter(); fanout_w = collections.Counter()
    w_edge = collections.Counter(); n_fileedge = collections.Counter()   # component x component
    for u, v, w in edges:
        cu, cv = comp[u], comp[v]
        if cu and cv:
            w_edge[(cu, cv)] += w; n_fileedge[(cu, cv)] += 1
        if cu and cv != cu:
            ce_files[cu].add(v); fanout_w[cu] += w
        if cv and cu != cv:
            ca_files[cv].add(u); fanin_w[cv] += w

    pr = pagerank(n, out_adj)
    bc = betweenness(n, adj_unw)
    pr_rank = {i: r + 1 for r, i in enumerate(sorted(range(n), key=lambda i: -pr[i]))}
    bc_rank = {i: r + 1 for r, i in enumerate(sorted(range(n), key=lambda i: -bc[i]))}

    rows = {}
    for c in COMP_ORDER:
        idx = files[c]
        ca, ce = len(ca_files[c]), len(ce_files[c])
        cbo = len(ca_files[c] | ce_files[c])
        inst = ce / (ca + ce) if (ca + ce) else 0.0
        intern = w_edge.get((c, c), 0)
        out_w = sum(v for (a, b), v in w_edge.items() if a == c and b != c)
        in_w = sum(v for (a, b), v in w_edge.items() if b == c and a != c)
        coh = 100.0 * intern / (intern + out_w + in_w) if (intern + out_w + in_w) else 0.0
        rows[c] = dict(
            files=len(idx), ca=ca, ce=ce, cbo=cbo, instab=inst,
            ca_share=100.0 * ca / (n - len(idx)) if n - len(idx) else 0.0,
            fanin_w=int(fanin_w[c]), fanout_w=int(fanout_w[c]),
            internal_w=int(intern), out_w=int(out_w), in_w=int(in_w), cohesion=coh,
            pr_best_rank=min((pr_rank[i] for i in idx), default=0),
            bc_best_rank=min((bc_rank[i] for i in idx), default=0),
        )

    # directed modularity Q of the model-code partition (Leicht-Newman)
    W = sum(w for u, v, w in edges if comp[u] and comp[v])
    kout = collections.Counter(); kin = collections.Counter(); sin = collections.Counter()
    for u, v, w in edges:
        if comp[u] and comp[v]:
            kout[comp[u]] += w; kin[comp[v]] += w
            if comp[u] == comp[v]:
                sin[comp[u]] += w
    Q = sum(sin[c] / W - (kout[c] * kin[c]) / (W * W) for c in COMP_ORDER)
    global_internal = 100.0 * sum(sin.values()) / W

    meta = dict(n_nodes=n, n_edges=len(edges), Q=Q, global_internal=global_internal,
                bc_globals_rank=rows["globals"]["bc_best_rank"], N=n)
    return rows, w_edge, n_fileedge, meta


# ---------------------------------------------------------------- output
def write_reports(rows, w_edge, n_fileedge, meta):
    os.makedirs(OUT, exist_ok=True)
    cols = ["component", "files", "ca", "ce", "cbo", "instab", "ca_share",
            "fanin_w", "fanout_w", "internal_w", "out_w", "in_w", "cohesion",
            "pr_best_rank", "bc_best_rank"]
    with open(os.path.join(OUT, "COMPONENT_VALUES.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for c in COMP_ORDER:
            r = rows[c]
            w.writerow([c] + [round(r[k], 4) if isinstance(r[k], float) else r[k]
                              for k in cols[1:]])
    # importance-focused CSV
    with open(os.path.join(OUT, "IMPORTANCE.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["component", "files", "Ca", "Ca_share_pct", "Instability", "bc_best_rank_of_%d" % meta["N"]])
        for c in sorted(COMP_ORDER, key=lambda x: -rows[x]["ca"]):
            r = rows[c]
            w.writerow([c, r["files"], r["ca"], round(r["ca_share"], 1), round(r["instab"], 3), r["bc_best_rank"]])

    lines = []
    lines.append("# JabRef component importance vs. code footprint\n")
    lines.append(f"Graph: {meta['n_nodes']} files, {meta['n_edges']} dependency edges "
                 f"(open-source Depends; ArDoCo/jabref @ pinned commit, see data/PROVENANCE.md).\n")
    lines.append("| component | files | Ca | Ca_share | Ce | Instab. | CBO | betw. rank |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for c in sorted(COMP_ORDER, key=lambda x: -rows[x]["ca"]):
        r = rows[c]
        lines.append(f"| {c} | {r['files']} | {r['ca']} | {r['ca_share']:.1f}% | {r['ce']} "
                     f"| {r['instab']:.2f} | {r['cbo']} | #{r['bc_best_rank']} of {meta['N']} |")
    lines.append("")
    lines.append(f"- Boundary alignment: {meta['global_internal']:.1f}% of reference weight stays "
                 f"within a component; directed modularity Q = {meta['Q']:.3f}.")
    lines.append("- **Footprint != importance:** `preferences` (few files) and `globals` (one file) "
                 "are depended on by more code than the largest component `gui`.")
    lines.append("- `Ca` (afferent coupling) and `Instability` are the size-independent metrics to cite; "
                 "`cli` is not depended upon (argue it via role, not coupling).")
    with open(os.path.join(OUT, "DEPIMPORT.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


FROZEN = {   # regression panel — regenerate deliberately if the graph/gold changes
    "model": dict(files=198, ca=716, ce=28),
    "logic": dict(files=575, ca=428, ce=162),
    "gui": dict(files=641, ca=25, ce=345),
    "cli": dict(files=5, ca=1, ce=63),
    "preferences": dict(files=18, ca=270, ce=76),
    "globals": dict(files=1, ca=97, ce=22),
}


def check(rows, meta):
    ok = True
    for c, exp in FROZEN.items():
        for k, v in exp.items():
            got = rows[c][k]
            if got != v:
                print(f"  MISMATCH {c}.{k}: got {got}, expected {v}"); ok = False
    if meta["bc_globals_rank"] != 6:
        print(f"  MISMATCH globals betweenness rank: got {meta['bc_globals_rank']}, expected 6"); ok = False
    print("CHECK: PASS" if ok else "CHECK: FAIL")
    return ok


if __name__ == "__main__":
    rows, w_edge, n_fileedge, meta = compute()
    write_reports(rows, w_edge, n_fileedge, meta)
    print(f"wrote reports/ (nodes={meta['n_nodes']}, edges={meta['n_edges']}, "
          f"Q={meta['Q']:.3f}, within-boundary={meta['global_internal']:.1f}%)")
    for c in sorted(COMP_ORDER, key=lambda x: -rows[x]["ca"]):
        r = rows[c]
        print(f"  {c:<12} files={r['files']:>4}  Ca={r['ca']:>4}  Ca_share={r['ca_share']:>4.1f}%  "
              f"Ce={r['ce']:>4}  I={r['instab']:.2f}  betw#={r['bc_best_rank']}")
    if "--check" in sys.argv:
        sys.exit(0 if check(rows, meta) else 1)
