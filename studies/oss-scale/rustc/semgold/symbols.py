"""Public-symbol index: identifier -> crates that define it (pub items + pub modules +
re-exports are ignored). Deterministic code fact used as one labelling function and as
annotator evidence ("`TyCtxt` is defined in rustc_middle")."""
from __future__ import annotations

import collections
import json
import re

from common import OUT, TREE, load_components

PUB_RE = re.compile(r"^\s*pub(?:\([a-z:]+\))?\s+(?:unsafe\s+)?(?:const\s+|async\s+)?(struct|enum|trait|fn|type|const|static|union|mod)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
MACRO_RE = re.compile(r"^\s*macro_rules!\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)


def main() -> None:
    index: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    kinds: dict[str, set] = collections.defaultdict(set)
    for crate in load_components():
        src = TREE / "compiler" / crate / "src"
        if not src.exists():
            continue
        for f in src.rglob("*.rs"):
            text = f.read_text(errors="replace")
            for kind, name in PUB_RE.findall(text):
                index[name][crate] += 1
                kinds[name].add(kind)
            for name in MACRO_RE.findall(text):
                index[name][crate] += 1
                kinds[name].add("macro")
    out = {name: {"crates": dict(c.most_common()), "kinds": sorted(kinds[name])} for name, c in index.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "symbols.json").write_text(json.dumps(out))
    amb = collections.Counter(len(v["crates"]) for v in out.values())
    print(f"identifiers {len(out)}  defined in 1 crate {amb[1]}  2 {amb[2]}  3+ {sum(v for k, v in amb.items() if k >= 3)}")
    for probe in ("TyCtxt", "Diag", "DefId", "Body", "Resolver", "Parser", "Session", "Span", "Symbol", "Obligation"):
        print(f"  {probe}: {out.get(probe, {}).get('crates')}")


if __name__ == "__main__":
    main()
