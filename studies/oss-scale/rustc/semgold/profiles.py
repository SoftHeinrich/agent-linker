"""Component self-descriptions, project-authored: crate-level `//!` docs (lib.rs), README,
Cargo.toml description, top-level public modules, and the most-referenced public items.
This is the evidence an annotator gets that the linker under test never sees."""
from __future__ import annotations

import json
import re
import collections
from pathlib import Path

from common import OUT, TREE, load_components

MOD_RE = re.compile(r"^\s*pub(?:\([a-z]+\))?\s+mod\s+([a-z_][a-z0-9_]*)\s*;", re.M)
PUB_RE = re.compile(r"^\s*pub(?:\([a-z]+\))?\s+(?:unsafe\s+)?(struct|enum|trait|fn|type|const|static|union|macro_rules!)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)


def crate_doc(lib: Path) -> str:
    lines = []
    for line in lib.read_text(errors="replace").splitlines():
        s = line.strip()
        if s.startswith("//!"):
            lines.append(s[3:].strip())
        elif lines and s and not s.startswith("#!") and not s.startswith("//"):
            break
    text = "\n".join(lines)
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\[[^\]]+\]:\s*\S+.*$", "", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def readme(dir_: Path) -> str:
    for name in ("README.md", "Readme.md", "readme.md"):
        p = dir_ / name
        if p.exists():
            t = p.read_text(errors="replace")
            t = re.sub(r"```.*?```", "", t, flags=re.S)
            return t.strip()[:2500]
    return ""


def cargo_description(dir_: Path) -> str:
    p = dir_ / "Cargo.toml"
    if not p.exists():
        return ""
    m = re.search(r'^description\s*=\s*"([^"]*)"', p.read_text(errors="replace"), re.M)
    return m.group(1) if m else ""


def module_docs(src: Path, mods: list[str]) -> dict[str, str]:
    """First `//!` paragraph of each top-level module (mod.rs or <mod>.rs), <=300 chars."""
    out = {}
    cands = [(m, c) for m in mods for c in (src / m / "mod.rs", src / f"{m}.rs")]
    # thin crates: fall back to every source file up to depth 2 (pass files carry `//!` headers)
    seen = {c for _, c in cands}
    for f in sorted(src.glob("*.rs")) + sorted(src.glob("*/*.rs")):
        if f not in seen and f.name not in ("lib.rs", "main.rs", "mod.rs"):
            cands.append((str(f.relative_to(src))[:-3].replace("/", "::"), f))
    for m, cand in cands:
        if len(out) >= 16:
            break
        for cand in (cand,):
            if cand.exists():
                lines = []
                for line in cand.read_text(errors="replace").splitlines():
                    s = line.strip()
                    if s.startswith("//!"):
                        body = s[3:].strip()
                        if not body and lines:
                            break
                        if body:
                            lines.append(body)
                    elif lines:
                        break
                text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", " ".join(lines))
                text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
                if text:
                    out[m] = text[:300]
                break
    return out


def modules_and_items(src: Path) -> tuple[list[str], list[str]]:
    lib = src / "lib.rs"
    mods = MOD_RE.findall(lib.read_text(errors="replace")) if lib.exists() else []
    items = collections.Counter()
    for f in src.rglob("*.rs"):
        for kind, name in PUB_RE.findall(f.read_text(errors="replace")):
            if kind in ("struct", "enum", "trait", "fn", "type", "macro_rules!"):
                items[name] += 1
    # `pub fn new/default/...` are noise; keep names that look specific
    generic = {"new", "default", "from", "into", "get", "set", "len", "is_empty", "iter", "next", "fmt", "clone",
               "hash", "eq", "cmp", "as_str", "index", "visit", "walk", "check", "run", "emit", "build"}
    top = [n for n, _ in items.most_common() if n not in generic][:60]
    return mods, top


def main() -> None:
    profiles = {}
    for crate in load_components():
        dir_ = TREE / "compiler" / crate
        src = dir_ / "src"
        lib = src / "lib.rs"
        if not lib.exists():
            # binary crate (rustc) has src/main.rs
            lib = src / "main.rs"
        doc = crate_doc(lib) if lib.exists() else ""
        mods, items = modules_and_items(src) if src.exists() else ([], [])
        deps = []
        cargo = dir_ / "Cargo.toml"
        if cargo.exists():
            in_deps = False
            for line in cargo.read_text(errors="replace").splitlines():
                if line.startswith("["):
                    in_deps = line.strip() == "[dependencies]"
                elif in_deps:
                    m = re.match(r"^(rustc_[a-z_]+)\s*=", line)
                    if m:
                        deps.append(m.group(1))
        profiles[crate] = {
            "doc": doc, "module_docs": module_docs(src, mods) if src.exists() else {}, "readme": readme(dir_), "description": cargo_description(dir_),
            "modules": mods, "items": items, "deps": deps,
            "files": sorted({str(f.relative_to(src))[:-3] for f in src.rglob("*.rs")} - {"lib", "main"})[:40] if src.exists() else [],
            "n_rs_files": sum(1 for _ in src.rglob("*.rs")) if src.exists() else 0,
            "source_excerpt": (re.sub(r"\s+", " ", " ".join(l for l in lib.read_text(errors="replace").splitlines()
                                                              if l.strip() and not l.strip().startswith(("//", "#!", "use "))))[:400]
                               if lib.exists() and not doc and sum(1 for _ in src.rglob("*.rs")) <= 2 else ""),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "profiles.json").write_text(json.dumps(profiles, indent=1))
    with_doc = sum(1 for p in profiles.values() if len(p["doc"]) > 80)
    with_readme = sum(1 for p in profiles.values() if p["readme"])
    print("module docs:", sum(len(p["module_docs"]) for p in profiles.values()), "across",
          sum(1 for p in profiles.values() if p["module_docs"]), "crates")
    print(f"crates {len(profiles)}  with crate-doc>80ch {with_doc}  with README {with_readme}  "
          f"with Cargo description {sum(1 for p in profiles.values() if p['description'])}")
    for c, p in profiles.items():
        if len(p["doc"]) <= 80 and not p["readme"]:
            print("  thin:", c, "modules:", ",".join(p["modules"][:8]))


if __name__ == "__main__":
    main()
