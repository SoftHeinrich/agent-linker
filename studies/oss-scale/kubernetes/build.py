#!/usr/bin/env python3
"""Build the Kubernetes cluster-architecture dataset in the benchmark's shape.

Source: kubernetes/website, `content/en/docs/concepts/overview/components.md` (the
project's component list with one-line descriptions) plus the eleven pages of
`content/en/docs/concepts/architecture/` ("Cluster Architecture": nodes, controllers,
control-plane/node communication, leases, cgroups, garbage collection, ...).  Same genre
as the benchmark texts: an architecture overview naming components and how they interact,
written by the project for its users.  Pinned to website commit cf96ee6 (2026-09-03).

Component model = the 12 entries of components.md (control plane, node, addons), as the
project lists them.  Outputs mirror ../gitlab/build.py: data/sentences.txt,
sentence_meta.json, components.json, gold_structural.csv (sentences under a heading that
is a component's own name — the `_index.md` per-component sections), meta.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent / "gitlab"))
from build import parse, split_sentences  # noqa: E402

COMMIT = "cf96ee6"
LOCAL = Path("/tmp/oss-case/kubernetes/website/content/en/docs/concepts")
RAW = "https://raw.githubusercontent.com/kubernetes/website/{commit}/content/en/docs/concepts/{path}"
PAGES = [
    "overview/components.md",
    "architecture/_index.md",
    "architecture/nodes.md",
    "architecture/control-plane-node-communication.md",
    "architecture/controller.md",
    "architecture/leases.md",
    "architecture/cloud-controller.md",
    "architecture/cgroups.md",
    "architecture/garbage-collection.md",
    "architecture/mixed-version-proxy.md",
    "architecture/self-healing.md",
]
# The project's own component list (components.md), ids = the page anchors it links to.
COMPONENTS = [
    ("kube-apiserver", "kube-apiserver", "control plane"),
    ("etcd", "etcd", "control plane"),
    ("kube-scheduler", "kube-scheduler", "control plane"),
    ("kube-controller-manager", "kube-controller-manager", "control plane"),
    ("cloud-controller-manager", "cloud-controller-manager", "control plane"),
    ("kubelet", "kubelet", "node"),
    ("kube-proxy", "kube-proxy", "node"),
    ("container-runtime", "Container runtime", "node"),
    ("dns", "DNS", "addon"),
    ("web-ui-dashboard", "Web UI (Dashboard)", "addon"),
    ("container-resource-monitoring", "Container Resource Monitoring", "addon"),
    ("cluster-level-logging", "Cluster-level Logging", "addon"),
]


def fetch(path: str, commit: str) -> str:
    local = LOCAL / path
    if local.exists():
        return local.read_text()
    with urllib.request.urlopen(RAW.format(commit=commit, path=path), timeout=60) as r:
        return r.read().decode()


def hugo_clean(t: str) -> str:
    t = re.sub(r'\{\{<\s*glossary_tooltip[^>]*?text="([^"]*)"[^>]*>\}\}', r"\1", t)
    t = re.sub(r'\{\{<\s*glossary_tooltip[^>]*?term_id="([^"]*)"[^>]*>\}\}', r"\1", t)
    t = re.sub(r"\{\{<\s*skew currentVersion\s*>\}\}", "1.34", t)
    t = re.sub(r"\{\{<[^\n]*?>\}\}|\{\{%[^\n]*?%\}\}", "", t)
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"^:\s+", "", t, flags=re.M)  # definition-list bodies are prose
    return t


def descriptions(components_md: str) -> dict[str, str]:
    out = {}
    lines = components_md.splitlines()
    for i, l in enumerate(lines):
        m = re.match(r"\[([^\]]+)\]\(/docs/concepts/architecture/#([^)]+)\)", l)
        if m and i + 1 < len(lines) and lines[i + 1].startswith(":"):
            out[m.group(2)] = re.sub(r"\s+", " ", hugo_clean(lines[i + 1][1:]).strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=COMMIT)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    (DATA / "src").mkdir(exist_ok=True)
    comps = [{"id": i, "name": n, "description": "", "layer": layer, "process": ""} for i, n, layer in COMPONENTS]
    by_name = {c["name"].lower(): c for c in comps}
    by_id = {c["id"]: c for c in comps}
    sentences, meta = [], []
    n_struct = 0
    for page in PAGES:
        md = fetch(page, args.commit)
        (DATA / "src" / page.replace("/", "__")).write_text(md)
        if page.endswith("components.md"):
            for anchor, desc in descriptions(md).items():
                if anchor in by_id:
                    by_id[anchor]["description"] = desc
        for text, heads, line in parse(hugo_clean(md)):
            own = None
            for h in reversed(heads):
                if h and h.lower() in by_name:
                    own = by_name[h.lower()]["id"]
                    break
            for s in split_sentences(text):
                sentences.append(s)
                meta.append({"heading": page + " > " + " > ".join(h for h in heads if h), "own_component": own, "line": line, "page": page})
                n_struct += bool(own)
    (DATA / "sentences.txt").write_text("\n".join(sentences) + "\n")
    json.dump(meta, open(DATA / "sentence_meta.json", "w"), indent=1)
    json.dump(comps, open(DATA / "components.json", "w"), indent=1)
    with open(DATA / "gold_structural.csv", "w") as f:
        f.write("modelElementID,sentence\n")
        for i, m in enumerate(meta, 1):
            if m["own_component"]:
                f.write(f"{m['own_component']},{i}\n")
    (DATA / "no-transarc.csv").write_text("modelElementID,sentence\n")
    info = {"source": "kubernetes/website content/en/docs/concepts (overview/components.md + architecture/*)",
            "commit": args.commit, "pages": PAGES, "sentences": len(sentences), "components": len(comps),
            "structural_pairs": n_struct}
    json.dump(info, open(DATA / "meta.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "pages"}, indent=1))


if __name__ == "__main__":
    main()
