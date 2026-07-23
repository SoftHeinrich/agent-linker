#!/usr/bin/env python3
"""Context augmentation: an LLM-built one-line ROLE profile per component, per project.

Grounds the discriminative judge so it can tell an external actor ("the user", "the browser")
from the component, and pick the right sibling for a generic phrase ("the datastore" -> the
storage component). Reasoning-off, one call per project, taboo-safe (only the doc's own
component names + its own sentences go in). Cached in profiles.json.
"""
import csv, glob, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
_envf = Path(__file__).resolve().parents[2] / ".env"
if _envf.exists():
    for ln in _envf.read_text().splitlines():
        if "=" in ln and not ln.strip().startswith("#"):
            k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

import run_judges as RJ
HERE = Path(__file__).resolve().parent
CASES = RJ.CASES
BENCH = Path(os.environ.get("TRANSARC_BENCHMARK",
             "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"))
PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]


def sentences(p):
    hits = glob.glob(str(BENCH / p / "text_*" / f"{p}.txt"))
    return [ln.strip() for ln in open(hits[0], errors="replace")] if hits else []


def roster(project):
    names = sorted({c["component"] for c in CASES
                    if c["project"] == project and c.get("component")})
    return names


def main():
    cl = RJ._client()
    out = {}
    for proj in PROJECTS:
        names = roster(proj)
        sents = sentences(proj)
        doc = "\n".join(f"S{i}: {t}" for i, t in enumerate(sents, 1) if t)
        prompt = (
            "Read the architecture document and write a ONE-LINE role for each listed component "
            "— what it is and what it does in this system — grounded in the sentences that name "
            "it. Keep each under 18 words. If a component name is also an ordinary English word, "
            "note what the ordinary word would mean here so it can be told apart from the "
            "component.\n\n"
            f"COMPONENTS: {', '.join(names)}\n\nDOCUMENT:\n{doc}\n\n"
            'Return JSON: {"profiles":[{"component":"Name","role":"one line"}]}\nJSON only:')
        resp = cl.query(prompt, timeout=180)
        prof = {}
        txt = resp.text if resp.success else ""
        a, b = txt.find("{"), txt.rfind("}")
        if a >= 0 and b >= 0:
            try:
                for r in json.loads(txt[a:b + 1]).get("profiles", []):
                    if r.get("component") and r.get("role"):
                        prof[r["component"]] = r["role"]
            except Exception:
                pass
        # ensure every roster name has an entry
        for n in names:
            prof.setdefault(n, "(no role extracted)")
        out[proj] = prof
        print(f"  {proj}: {len(prof)} profiles", file=sys.stderr)
        for n in names:
            print(f"      {n:<20} {prof[n][:70]}", file=sys.stderr)
    (HERE / "profiles.json").write_text(json.dumps(out, indent=1))
    print("wrote profiles.json", file=sys.stderr)


if __name__ == "__main__":
    main()
