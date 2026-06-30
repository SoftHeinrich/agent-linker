"""Router + direct sentence->code linking (router branch, pilot).

Motivation
----------
The canonical doc-to-code result is COMPOSED:  sentence -> component (our LLM
model-doc linker)  o  component -> code (ArCoTL, deterministic). That transitive
route is powerful but STRUCTURALLY BLIND to documentation sentences that describe
*code organisation* rather than architecture -- package inventories, class names,
config files. Such sentences have no architecture component to route through, so
the composition emits nothing. A feasibility pilot (see ``pilot/``) found ~23% of
the doc-code gold links (all in one project) are unreachable this way, at recall 0.

This module adds a second, DIRECT route and a per-sentence ROUTER that decides
when to use it. Final links = transitive  UNION  direct.

How direct linking works (the design choice)
--------------------------------------------
There is NO raw source tree in the benchmark -- only the ``.acm`` code model. So
direct linking is NOT grep-over-source; it is PACKAGE/CODE-MODEL STRUCTURE match:

  1. parse the ``.acm`` -> one ``CodeUnit`` per CodeCompilationUnit, carrying its
     package path (``pathElements``), class stem (``name``) and extension.
  2. extract code identifiers from the sentence: CamelCase class names
     (``WebApiServlet``), dotted packages (``logic.api``), file names (``web.xml``).
  3. resolve each identifier against an index of the code model and emit the
     matching file path(s). A package identifier expands to every unit under it
     (mirrors how the gold standard enrols directory entries); a class identifier
     resolves to that compilation unit (optionally its ``*Test`` twin).

Only identifiers that RESOLVE to a real compilation unit are emitted, so the
precision gate lives on the linker output -- exactly where the pilot showed it
belongs (router over-fire is free; a conservative direct linker is what bounds FP).

The router (``SentenceRouter``) is a thin LLM classifier (taboo-safe, no component
names, sentence text only) or the free ``rule_route`` (CODE iff the direct linker
finds any resolvable identifier).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# ── code model ────────────────────────────────────────────────────────────────

# Leading build/layout segments that are not part of the logical package and must
# not be treated as a dotted-package token target (so "logic.api" matches the
# package, not "src.main"). Generic only -- no benchmark-derived names (taboo).
_LAYOUT_SEGMENTS = {"src", "main", "java", "test", "resources", "com", "org",
                    "net", "edu", "io"}
_IMPL_PREFIX = "Implementation/"


def _strip_impl(path: str) -> str:
    return path[len(_IMPL_PREFIX):] if path.startswith(_IMPL_PREFIX) else path


@dataclass(frozen=True)
class CodeUnit:
    path: str               # normalized full path (Implementation/ stripped)
    segments: tuple         # lowercased pathElements
    cls: str                # class stem (e.g. "WebApiServlet")
    ext: str                # extension without dot (e.g. "java")


def load_code_units(acm_path: str | Path) -> list[CodeUnit]:
    """All CodeCompilationUnits from a .acm code model."""
    data = json.loads(Path(acm_path).read_text())
    repo = data.get("codeItemRepository", {}).get("repository", {})
    units: list[CodeUnit] = []
    for item in repo.values():
        if item.get("type") != "CodeCompilationUnit":
            continue
        parts = item.get("pathElements", []) or []
        name = item.get("name", "")
        ext = item.get("extension", "") or ""
        if not (parts and name):
            continue
        full = "/".join(parts) + "/" + name + (f".{ext}" if ext else "")
        units.append(CodeUnit(_strip_impl(full),
                              tuple(p.lower() for p in parts), name, ext))
    return units


class CodeIndex:
    """Lookup structures over a code model for direct identifier resolution."""

    def __init__(self, units: list[CodeUnit]):
        self.units = units
        self.by_class: dict[str, set[str]] = defaultdict(set)   # lower stem -> paths
        self.by_file: dict[str, set[str]] = defaultdict(set)    # 'name.ext'/'name' -> paths
        # logical package segments per unit (layout prefixes dropped, in order)
        self._pkg_segments: list[tuple[tuple, str]] = []        # (segments, path)
        for u in units:
            self.by_class[u.cls.lower()].add(u.path)
            self.by_file[(u.cls + ("." + u.ext if u.ext else "")).lower()].add(u.path)
            self.by_file[u.cls.lower()].add(u.path)
            logical = tuple(s for s in u.segments if s not in _LAYOUT_SEGMENTS)
            self._pkg_segments.append((logical, u.path))

    def match_class(self, token: str, include_test: bool = True) -> set[str]:
        t = token.lower()
        out = set(self.by_class.get(t, ()))
        if include_test:
            out |= self.by_class.get(t + "test", set())
            out |= self.by_class.get("test" + t, set())
        return out

    def match_file(self, token: str) -> set[str]:
        return set(self.by_file.get(token.lower(), ()))

    def _match_segs(self, segs: tuple) -> set[str]:
        out: set[str] = set()
        n = len(segs)
        if not n:
            return out
        for logical, path in self._pkg_segments:
            for i in range(len(logical) - n + 1):
                if logical[i:i + n] == segs:
                    out.add(path)
                    break
        return out

    def match_package(self, dotted: str, root_placeholder: bool = False) -> set[str]:
        """A dotted token (e.g. 'logic.api') matches units whose logical package
        contains those segments contiguously.

        Root-placeholder fallback (opt-in): if the literal token does not resolve
        and its first segment is a single character, retry on the suffix. Handles a
        doc that writes a single-letter placeholder for the project root (e.g.
        'x.logic', 'x.search'). OFF by default: it recovers recall on such docs but
        a 1-segment suffix ('x.util' -> 'util') matches every same-named package,
        so it trades file-level precision for recall (see pilot/README.md).
        """
        segs = tuple(s.lower() for s in dotted.split(".") if s)
        if not segs:
            return set()
        out = self._match_segs(segs)
        if root_placeholder and not out and len(segs) > 1 and len(segs[0]) == 1:
            out = self._match_segs(segs[1:])
        return out


# ── identifier extraction ──────────────────────────────────────────────────────

# CamelCase / PascalCase with at least one internal case change, len>=4.
_CAMEL = re.compile(r"\b[A-Za-z][a-z0-9]*(?:[A-Z][a-z0-9]+)+\b")
# dotted lowercase identifier (package/qualified), 2+ segments.
_DOTTED = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:\.[a-zA-Z][a-zA-Z0-9]*)+\b")
# file name with a code/config extension.
_FILE = re.compile(
    r"\b[\w-]+\.(?:java|xml|html|jsp|properties|json|yaml|yml|sql|gradle|cfg)\b",
    re.IGNORECASE)
# dotted tokens whose segments are these are abbreviations/noise, never packages.
_DOTTED_STOP = {"e", "g", "i", "e.g", "i.e", "etc", "vs", "no", "fig"}


def extract_mentions(text: str) -> dict[str, list[str]]:
    files = [m.group(0) for m in _FILE.finditer(text)]
    file_lc = {f.lower() for f in files}
    dotted = []
    for m in _DOTTED.finditer(text):
        tok = m.group(0)
        if tok.lower() in file_lc:                       # already a file token
            continue
        segs = [s for s in tok.lower().split(".") if s]
        if any(s in _DOTTED_STOP for s in segs) or all(len(s) <= 1 for s in segs):
            continue
        dotted.append(tok)
    camel = [m.group(0) for m in _CAMEL.finditer(text) if len(m.group(0)) >= 4]
    return {"camel": camel, "dotted": dotted, "file": files}


# ── direct linker ───────────────────────────────────────────────────────────────

@dataclass
class DirectCodeLinker:
    index: CodeIndex
    include_test: bool = True
    # a single package token can enrol an entire package; cap to avoid a vague
    # token (e.g. a top-level package) dragging in hundreds of files. None = no cap.
    max_files_per_package: Optional[int] = None
    # treat a leading single-char package segment as a project-root placeholder
    # ('x.logic' -> 'logic'). Recall+/precision- tradeoff; off by default.
    root_placeholder: bool = False

    def candidates(self, text: str) -> list[tuple[str, str, frozenset]]:
        """Per-identifier candidates: (identifier, kind, resolved_paths).

        kind in {file, class, package}. This is the unit a judge validates -- one
        verdict per named identifier, not per resolved file.
        """
        ment = extract_mentions(text)
        out: list[tuple[str, str, frozenset]] = []
        for tok in ment["file"]:
            hit = self.index.match_file(tok)
            if hit:
                out.append((tok, "file", frozenset(hit)))
        for tok in ment["camel"]:
            hit = self.index.match_class(tok, self.include_test)
            if hit:
                out.append((tok, "class", frozenset(hit)))
        for tok in ment["dotted"]:
            hit = self.index.match_class(tok.split(".")[-1], self.include_test)
            kind = "class"
            if not hit:
                hit = self.index.match_package(tok, self.root_placeholder)
                kind = "package"
                if self.max_files_per_package and len(hit) > self.max_files_per_package:
                    continue
            if hit:
                out.append((tok, kind, frozenset(hit)))
        return out

    def link_sentence(self, text: str) -> set[str]:
        """Return the set of code file paths this sentence directly names."""
        out: set[str] = set()
        for _ident, _kind, paths in self.candidates(text):
            out |= paths
        return out


# ── router ───────────────────────────────────────────────────────────────────

_ROUTER_PROMPT = (
    "You triage software-documentation sentences for trace-link recovery. "
    "For each sentence choose ONE route:\n"
    '- "ARCH": the sentence describes a high-level component, its responsibility, '
    "or how components interact at the architecture level.\n"
    '- "CODE": the sentence refers to concrete code-level structure such as a '
    "package name, class name, file name, method, exception, or configuration "
    "file (often dotted identifiers, CamelCase names, or file extensions).\n"
    "Choose CODE only when a specific code-level identifier/artifact is named; "
    "otherwise choose ARCH.\n"
    'Reply with ONLY a JSON array of {"id": <int>, "route": "ARCH"|"CODE"}.\n\n'
    "Sentences:\n")

ARCH, CODE = "ARCH", "CODE"


def rule_route(text: str, direct_linker: DirectCodeLinker) -> str:
    """Free router: CODE iff the direct linker resolves >=1 code identifier.

    Self-consistent with the linker -- never routes CODE on a sentence the direct
    linker could not act on, so it adds no FP-risk beyond the linker itself.
    """
    return CODE if direct_linker.link_sentence(text) else ARCH


class SentenceRouter:
    """LLM router (zero-shot, batched). Falls back to rule_route if no client."""

    def __init__(self, client=None, model: str = "gpt-5.4", batch: int = 12,
                 timeout: int = 120):
        self.client = client
        self.model = model
        self.batch = batch
        self.timeout = timeout

    def _client(self):
        if self.client is None:
            from llm_sad_sam.llm_client import LLMClient, LLMBackend
            self.client = LLMClient(backend=LLMBackend.OPENAI, model=self.model,
                                    enable_logging=False)
        return self.client

    def route(self, sentences: dict[str, str]) -> dict[str, str]:
        """sentence_id -> ARCH|CODE for the given {id: text} map."""
        client = self._client()
        ids = list(sentences)
        out: dict[str, str] = {}
        for k in range(0, len(ids), self.batch):
            chunk = ids[k:k + self.batch]
            prompt = _ROUTER_PROMPT + "\n".join(
                f"{i}. {sentences[sid]}" for i, sid in enumerate(chunk))
            resp = client.query(prompt, timeout=self.timeout)
            decisions = self._parse(resp.text if resp.success else "")
            for i, sid in enumerate(chunk):
                out[sid] = decisions.get(i, ARCH)         # default safe = ARCH
        return out

    @staticmethod
    def _parse(txt: str) -> dict[int, str]:
        a, b = txt.find("["), txt.rfind("]")
        if a < 0 or b < 0:
            return {}
        try:
            arr = json.loads(txt[a:b + 1])
        except Exception:
            return {}
        res = {}
        for o in arr:
            try:
                r = str(o["route"]).upper().strip()
                res[int(o["id"])] = CODE if r == CODE else ARCH
            except Exception:
                pass
        return res


# ── direct-link judge ────────────────────────────────────────────────────────

# Mirrors s_linker21's validation pass: claim-before-verdict. The template is
# generic (no benchmark terms); the code identifier is a runtime input, exactly
# as the model-doc validator passes runtime component names.
_JUDGE_PROMPT = (
    "You validate candidate trace links between a documentation sentence and a "
    "named code element. A link is VALID if the sentence states the element is "
    "used, provided, implemented, contained, or described -- INCLUDING when it is "
    'named as a concrete example or instance of what is described ("such as X", '
    '"e.g. X", "including X"). A link is INVALID only if the element appears in an '
    'exclusion -- a negation ("not X", "no X"), a contrast or counter-example '
    '("other than X", "unlike X", "rather than X") -- OR if the token is used as a '
    "product, system, or brand name rather than a reference to that code unit.\n"
    "For each case, FIRST quote the exact words from the sentence that assert (or "
    'exclude) the link, or "none", THEN decide keep true/false based only on that '
    "quote.\n\n"
    "CASES:\n{cases}\n\n"
    'Return JSON: {{"validations":[{{"case":1,"claim":"<exact quote or none>",'
    '"keep":true}}]}}\nJSON only:')


class DirectLinkJudge:
    """LLM keep/reject judge over (sentence, identifier) direct-link candidates."""

    def __init__(self, client=None, model: str = "gpt-5.4", batch: int = 10,
                 timeout: int = 120):
        self.client = client
        self.model = model
        self.batch = batch
        self.timeout = timeout

    def _client(self):
        if self.client is None:
            from llm_sad_sam.llm_client import LLMClient, LLMBackend
            self.client = LLMClient(backend=LLMBackend.OPENAI, model=self.model,
                                    enable_logging=False)
        return self.client

    def judge(self, cases: list[dict]) -> dict[int, bool]:
        """cases: [{text, identifier, kind}]; returns {case_index: keep_bool}.

        Default on parse failure / missing verdict is True (keep) so the judge
        only ever *removes* links it actively rejects.
        """
        client = self._client()
        out: dict[int, bool] = {}
        for k in range(0, len(cases), self.batch):
            chunk = cases[k:k + self.batch]
            body = "\n".join(
                f'{i}. SENTENCE: "{c["text"]}"  CODE ELEMENT: {c["identifier"]} '
                f'({c["kind"]})' for i, c in enumerate(chunk))
            resp = client.query(_JUDGE_PROMPT.format(cases=body), timeout=self.timeout)
            verdicts = self._parse(resp.text if resp.success else "")
            for i in range(len(chunk)):
                out[k + i] = verdicts.get(i, True)
        return out

    @staticmethod
    def _parse(txt: str) -> dict[int, bool]:
        a, b = txt.find("{"), txt.rfind("}")
        if a < 0 or b < 0:
            return {}
        try:
            obj = json.loads(txt[a:b + 1])
        except Exception:
            return {}
        res = {}
        for v in obj.get("validations", []):
            try:
                res[int(v["case"])] = bool(v["keep"])
            except Exception:
                pass
        return res


# ── composition ────────────────────────────────────────────────────────────────

def augment_doc_code(transitive: set[tuple[str, str]],
                     sentences: dict[str, str],
                     direct_linker: DirectCodeLinker,
                     route: dict[str, str]) -> set[tuple[str, str]]:
    """Final doc-code links = transitive UNION direct(CODE-routed sentences).

    ``transitive`` and the return value are sets of (sentence_id, code_path).
    ``route`` maps sentence_id -> ARCH|CODE.
    """
    out = set(transitive)
    for sid, text in sentences.items():
        if route.get(sid) == CODE:
            for path in direct_linker.link_sentence(text):
                out.add((sid, path))
    return out
