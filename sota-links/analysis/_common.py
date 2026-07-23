"""Shared loaders for the Artemis-unique-TP / validator study (model-doc / SAD-SAM).

Standalone stdlib (plus the agent-linker package, only needed to unpickle the
phase cache in 02/03). Paths resolve relative to the recovered-links repo root;
override the benchmark location with the ARDOCO_BENCHMARK env var.
"""
import os, re, csv, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)                                  # recovered-links repo root
APPROACH_SRC = os.path.normpath(os.path.join(RL, "..", "..", "agent-linker", "src"))
BENCHMARK = os.environ.get(
    "ARDOCO_BENCHMARK",
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark",
)

PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]

# project -> (PCM .repository, SAD text) relative to BENCHMARK
_BM = {
    "mediastore":    ("mediastore/model_2016/pcm/ms.repository",        "mediastore/text_2016/mediastore.txt"),
    "teastore":      ("teastore/model_2020/pcm/teastore.repository",    "teastore/text_2020/teastore.txt"),
    "teammates":     ("teammates/model_2021/pcm/teammates.repository",  "teammates/text_2021/teammates.txt"),
    "bigbluebutton": ("bigbluebutton/model_2021/pcm/bbb.repository",    "bigbluebutton/text_2021/bigbluebutton.txt"),
    "jabref":        ("jabref/model_2021/pcm/jabref.repository",        "jabref/text_2021/jabref.txt"),
}


def repo_path(proj):
    return os.path.join(BENCHMARK, _BM[proj][0])


def id_to_name(proj):
    """PCM element id -> entityName (first occurrence wins)."""
    txt = open(repo_path(proj), encoding="utf-8").read()
    d = {}
    for m in re.finditer(r'id="(_[^"]+)"[^>]*?entityName="([^"]*)"', txt):
        d.setdefault(m.group(1), m.group(2))
    return d


def name_to_id(proj):
    d = {}
    for k, v in id_to_name(proj).items():
        d.setdefault(v, k)
    return d


def sentences(proj):
    """1-based sentence index -> text (ArDoCo dumps are one sentence per line)."""
    path = os.path.join(BENCHMARK, _BM[proj][1])
    return {i: l.rstrip("\n") for i, l in enumerate(open(path, encoding="utf-8"), 1)}


def links(path):
    """Load a normalized recovered/gold CSV -> set of (sentence_id, target_id)."""
    s = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            try:
                s.add((int(r["sentence_id"]), r["target_id"]))
            except (ValueError, KeyError):
                pass
    return s


def gold(proj):
    return links(os.path.join(RL, "model-doc", "gold", f"{proj}.csv"))


def our_runs(variant="gpt-5.4_s21"):
    """Per-run recovered-link sets for an aalinker variant (run1..runN)."""
    base = os.path.join(RL, "model-doc", "aalinker", variant)
    out = []
    for d in sorted(os.listdir(base)):
        if d.startswith("run"):
            out.append(links(os.path.join(base, d, f"{{proj}}.csv")))  # template; filled per project
    return base


def our_union(proj, variant="gpt-5.4_s21"):
    base = os.path.join(RL, "model-doc", "aalinker", variant)
    u = set()
    for d in sorted(os.listdir(base)):
        if d.startswith("run"):
            u |= links(os.path.join(base, d, f"{proj}.csv"))
    return u


def artemis(proj):
    return links(os.path.join(RL, "model-doc", f"artemis-{proj}-gpt-5.4.csv"))


# ── lexical classification of a (component-name, sentence) pair ──────────────
_STOP = set("the a an of to in for and or data service interface component".split())


def camel_tokens(name):
    toks = []
    for part in re.split(r"[_\s]+", name):
        toks += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", part)
    return [t.lower() for t in toks if t]


def lexical_class(name, sent):
    """Return (label, evidence). label in LEXICAL-full / LEXICAL-partial / SEMANTIC-none.

    LEXICAL-full   = the component name appears (almost) verbatim, or every name
                     token is present in the sentence.
    LEXICAL-partial= some but not all name tokens present.
    SEMANTIC-none  = no name token present (a non-lexical / semantic match).
    """
    toks = camel_tokens(name)
    sl = sent.lower()
    norm = re.sub(r"[^a-z0-9]", "", sl)
    stoks = set(re.findall(r"[a-z0-9]+", sl))
    joined = "".join(toks)
    verbatim = bool(joined) and joined in norm
    exact = [t for t in set(toks) if re.search(r"\b" + re.escape(t) + r"\b", sl)]
    stem = [t for t in set(toks) if t not in exact and len(t) >= 4
            and any((w.startswith(t) or t.startswith(w)) and len(w) >= 4 for w in stoks)]
    matched, n = len(exact) + len(stem), len(set(toks))
    if verbatim or (n and matched == n):
        label = "LEXICAL-full"
    elif matched > 0:
        label = "LEXICAL-partial"
    else:
        label = "SEMANTIC-none"
    ev = ("[verbatim] " if verbatim else "") + ("exact:" + ",".join(exact) if exact else "")
    if stem:
        ev += " stem:" + ",".join(stem)
    return label, ev.strip() or "—"


def load_phase_cache():
    """Import the agent-linker package so phase-cache pickles can be unpickled."""
    if APPROACH_SRC not in sys.path:
        sys.path.insert(0, APPROACH_SRC)
    import pickle  # noqa: F401  (callers use pickle after this primes sys.path)
    return pickle
