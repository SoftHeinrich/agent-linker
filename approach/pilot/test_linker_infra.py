"""`linker_infra`'s invariants: the extracted plumbing is the bytes it replaced.

`s_linker110.py` is the reported arm and a standalone file, so the eleven
infrastructure blocks it shared byte for byte with **72 linker modules** could not be
lifted into a base class -- `pilot/test_s110_shortlist.py` asserts
`SLinker110.__mro__ == (SLinker110, object)` and the paper's supplement is the file.
They moved into `linker_infra` as functions instead, and each method stayed in the
variant as a one-line delegation.

That refactor's equivalence has to be *tested against the polarity it preserves*, not
merely asserted (the `s_linker114` lesson: a stub that answers nothing kept nothing,
and 142/142 compared two empty sets). So every helper here is run against
**`s_linker92`'s untouched copy of the same block** -- the module the ledger records,
which this branch does not edit -- over inputs that exercise each branch:

  1  **the tracing wrapper** records the same row on success, on a usage-less
     response, on `success=False` (which must still raise *after* recording), and on
     an inner exception (`FATAL:`, recorded, re-raised). Phase tagging and the
     `__getattr__` delegation that `describe_backend()`/`extract_json()` ride on.
  2  **the call path** (`ask_json`) under all three success rules x an answering, an
     empty and an unparseable reply, plus the retry count and the rule that a later
     unparseable attempt replaces an earlier rejected payload.
  3  **the run's artifacts**: the checkpoint path and its pickle, the phase-log row,
     both log filenames and their JSON, `CALLS_TRUNCATE_CHARS` on and off, and the
     per-phase metrics -- each compared to what `SLinker92`'s method writes from the
     same state, under a temporary `PHASE_CACHE_DIR`/`LLM_LOG_DIR`.
  4  **the views**: batching, the link view, the decision view, the linker feedback.
  5  **the delegation is live**: `SLinker110`'s own methods return what the helpers
     return, so the wiring is checked and not only the helpers.

No LLM calls.

    ../.venv/bin/python pilot/test_linker_infra.py
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.data_types_v2 import SadSamLink                    # noqa: E402
from llm_sad_sam.linkers.experimental import linker_infra as INFRA       # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92 import (                # noqa: E402
    SLinker92, _TracingLLMClient as HeadTracer,
)
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110      # noqa: E402

CHECKS = []


def check(condition, label):
    CHECKS.append((bool(condition), label))
    if not condition:
        print(f"  FAIL  {label}")


# ── stand-ins ────────────────────────────────────────────────────────────────
# Response and client doubles. Nothing here reaches a backend; the point is that
# both implementations see byte-identical inputs.

class Resp:
    def __init__(self, text="{}", success=True, error=None, usage=None,
                 latency_ms=7, model="m"):
        self.text, self.success, self.error = text, success, error
        self.latency_ms, self.model, self.token_usage = latency_ms, model, usage


class Usage:
    def __init__(self, p=3, c=5, t=8):
        self.prompt_tokens, self.completion_tokens, self.total_tokens = p, c, t


class Inner:
    """A client that hands back a scripted sequence of replies."""

    backend = type("B", (), {"value": "claude"})()

    def __init__(self, script):
        self.script = list(script)
        self.prompts = []

    def query(self, prompt, timeout=180, max_retries=3):
        self.prompts.append((prompt, timeout, max_retries))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def describe_backend(self):
        return "DESCRIBED"

    def extract_json(self, resp):
        try:
            return json.loads(resp.text)
        except Exception:
            return None


VOLATILE = ("ts", "elapsed_s")


def stable(records):
    """A trace with the wall-clock fields dropped -- they cannot be equal."""
    return [{k: v for k, v in r.items() if k not in VOLATILE} for r in records]


def bare(cls, **attrs):
    obj = cls.__new__(cls)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def head_ask(llm, prompt, **kw):
    """`SLinker92._ask` -- the untouched bytes -- driven over a given client."""
    return bare(SLinker92, llm=llm)._ask(prompt, **kw)


# ── 1. the tracing wrapper ───────────────────────────────────────────────────

def tracing():
    cases = {
        "a success with token usage": [Resp('{"a": 1}', usage=Usage())],
        "a success with no usage": [Resp('{"a": 1}')],
        "a failed response": [Resp("", success=False, error="boom")],
        "an inner exception": [RuntimeError("transport died")],
    }
    for label, script in cases.items():
        traces, results = [], []
        for impl in (HeadTracer, INFRA.TracingLLMClient):
            sink = []
            client = impl(Inner(list(script)), sink)
            client.set_phase("phase_x")
            try:
                client.query("PROMPT", timeout=11, max_retries=2)
                results.append("returned")
            except Exception as exc:
                results.append(f"{type(exc).__name__}: {exc}")
            traces.append(sink)
        check(stable(traces[0]) == stable(traces[1]),
              f"tracing: {label} records the head's row")
        check(results[0] == results[1],
              f"tracing: {label} ends the head's way ({results[1]})")
        check(len(traces[1]) == 1, f"tracing: {label} records exactly one row")

    # the phase is read at query time, and defaults when nothing set it
    sink = []
    client = INFRA.TracingLLMClient(Inner([Resp()]), sink)
    INFRA._phase_local.__dict__.pop("phase", None)
    client.query("P")
    check(sink[0]["phase"] == "unknown", "tracing: an untagged call is 'unknown'")
    check(INFRA.current_phase() == "unknown", "current_phase() defaults to 'unknown'")

    # the delegation the two live callers ride on
    client = INFRA.TracingLLMClient(Inner([]), [])
    check(client.describe_backend() == "DESCRIBED",
          "tracing: describe_backend() reaches the inner client")
    check(client.extract_json(Resp('{"k": 2}')) == {"k": 2},
          "tracing: extract_json() reaches the inner client")


# ── 2. the call path ─────────────────────────────────────────────────────────

def call_path():
    answering, empty, junk = '{"links": [1], "other": 0}', "{}", "not json"
    rules = {
        "no rule": {},
        "require=links": {"require": "links"},
        "require_present=links": {"require_present": "links"},
        "require of an absent key": {"require": "missing"},
        "require_present of an absent key": {"require_present": "missing"},
        # both set -- the docstring's priority order. `links` is present but the
        # `require` key is absent, so only require_present-wins reads a success.
        "both, require_present wins": {"require_present": "links",
                                       "require": "missing"},
        "both, the other way round": {"require_present": "missing",
                                      "require": "links"},
    }
    replies = {
        "answering": answering, "empty dict": empty, "unparseable": junk,
    }
    for rule_name, rule in rules.items():
        for reply_name, text in replies.items():
            got, calls = [], []
            for fn in (head_ask, INFRA_ask):
                inner = Inner([Resp(text), Resp(text)])
                got.append(fn(INFRA.TracingLLMClient(inner, []), "P", **rule))
                calls.append(inner.prompts)
            check(got[0] == got[1],
                  f"ask: {rule_name} / {reply_name} returns the head's payload "
                  f"({got[1]!r})")
            check(calls[0] == calls[1],
                  f"ask: {rule_name} / {reply_name} spends the head's calls "
                  f"({len(calls[1])})")

    # the rule the head's comment states: a later unparseable attempt must replace
    # the earlier rejected payload rather than resurrect it
    for fn, who in ((head_ask, "head"), (INFRA_ask, "infra")):
        inner = Inner([Resp('{"other": 1}'), Resp("not json")])
        out = fn(INFRA.TracingLLMClient(inner, []), "P", require="links")
        check(out == {}, f"ask: a rejected payload is not resurrected ({who})")

    # `attempts` is the caller's bound, not a default hidden in the helper
    inner = Inner([Resp("{}")] * 5)
    INFRA.ask_json(INFRA.TracingLLMClient(inner, []), "P", attempts=4,
                   require="links")
    check(len(inner.prompts) == 4, "ask: attempts= is the caller's bound (4 calls)")
    check(SLinker110.ASK_ATTEMPTS == SLinker92.ASK_ATTEMPTS == 2,
          "ask: the variant still declares ASK_ATTEMPTS = 2")

    # timeout and phase travel through
    inner = Inner([Resp('{"links": [1]}')])
    client = INFRA.TracingLLMClient(inner, sink := [])
    INFRA.ask_json(client, "P", attempts=2, timeout=999, phase="phase_9")
    check(inner.prompts[0][1] == 999, "ask: timeout reaches query()")
    check(sink[0]["phase"] == "phase_9", "ask: phase= tags the trace")


def INFRA_ask(llm, prompt, **kw):
    return INFRA.ask_json(llm, prompt, attempts=SLinker92.ASK_ATTEMPTS, **kw)


# ── 3. the run's artifacts ───────────────────────────────────────────────────

CALLS = [
    {"phase": "p1", "elapsed_s": 1.5, "success": True, "prompt": "x" * 40,
     "response_text": "y" * 40, "token_usage": {"total_tokens": 10}},
    {"phase": "p1", "elapsed_s": 0.25, "success": False, "prompt": "short",
     "response_text": None},
    {"phase": "p2", "elapsed_s": 2.0, "success": True, "prompt": None,
     "response_text": "z" * 90, "token_usage": {"total_tokens": None}},
    {"elapsed_s": 0.5, "success": True},
]
LINKS = [SadSamLink(3, "c1", "Alpha", source="full_name"),
         SadSamLink(9, "c2", "Beta", source="coreference")]


def artifacts():
    def build(cls):
        return bare(cls, llm=INFRA.TracingLLMClient(Inner([]), []),
                    _phase_log=[], _llm_calls=list(CALLS))

    head, mine = build(SLinker92), build(SLinker110)
    hv, mv = SLinker92._VARIANT_NAME, SLinker110._VARIANT_NAME

    check(head._backend_tag() == mine._backend_tag() == "claude",
          "backend_tag: the wrapper's inner backend value")
    check(INFRA.backend_tag(object()) == "unknown",
          "backend_tag: no backend at all is 'unknown'")

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["PHASE_CACHE_DIR"] = os.path.join(tmp, "cache")
        os.environ.pop("CALLS_TRUNCATE_CHARS", None)
        text_path = "/somewhere/mediastore.txt"

        hd, md = head._checkpoint_dir(text_path), mine._checkpoint_dir(text_path)
        check(Path(hd).name == Path(md).name == "mediastore",
              "checkpoint_dir: the dataset is the leaf")
        check(Path(hd).parent.name == Path(md).parent.name == "claude",
              "checkpoint_dir: the backend tag is the parent")
        check(Path(md).parts[-3] == mv and Path(hd).parts[-3] == hv,
              f"checkpoint_dir: each variant name is its own ({hv} / {mv})")
        check(Path(md).is_dir(), "checkpoint_dir: the directory is created")

        state = {"doc_knowledge": {"a": 1}, "n": [1, 2, 3]}
        head._save_phase(text_path, "knowledge", state)
        mine._save_phase(text_path, "knowledge", state)
        loaded = [pickle.loads((Path(d) / "knowledge.pkl").read_bytes())
                  for d in (hd, md)]
        check(loaded[0] == loaded[1] == state,
              "save_phase: the same pickle lands at the same name")

        for obj in (head, mine):
            obj._log("s25_summary", {"in": 1}, {"out": 2}, LINKS)
            obj._log("no_links", {"in": 3}, {"out": 4})
        check(stable(head._phase_log) == stable(mine._phase_log),
              "log: the phase rows are the head's")
        check(head._phase_log[0]["links"] == [
            {"s": 3, "c": "Alpha", "src": "full_name"},
            {"s": 9, "c": "Beta", "src": "coreference"}],
            "log: links are reduced to (s, c, src)")
        check("links" not in head._phase_log[1]
              and "links" not in mine._phase_log[1],
              "log: links=None writes no links key")

        #: `{variant}_{backend}_{dataset}_{timestamp}[_calls].json` -- the timestamp
        #: second can tick between the two writes, so the shape is compared without it.
        def shape(name, variant):
            return re.sub(r"_\d{8}_\d{6}", "_TS", name.replace(variant, "V", 1))

        for label, trunc in (("truncate=off", None), ("truncate=20", "20")):
            logs = Path(tmp) / f"logs_{label}"
            os.environ["LLM_LOG_DIR"] = str(logs)
            if trunc is None:
                os.environ.pop("CALLS_TRUNCATE_CHARS", None)
            else:
                os.environ["CALLS_TRUNCATE_CHARS"] = trunc

            written = {}
            for obj, variant in ((head, hv), (mine, mv)):
                obj._save_log(text_path)
                files = sorted(p for p in logs.iterdir()
                               if p.name.startswith(variant + "_"))
                written[variant] = files
                check(len(files) == 2,
                      f"save_log: {variant} writes a phase log and a call trace "
                      f"({label})")

            check([shape(p.name, hv) for p in written[hv]]
                  == [shape(p.name, mv) for p in written[mv]],
                  f"save_log: both name their artifacts the head's way ({label})")

            def payloads(variant):
                out = {}
                for p in written[variant]:
                    key = "calls" if p.name.endswith("_calls.json") else "phases"
                    out[key] = json.loads(p.read_text())
                return out

            h, m = payloads(hv), payloads(mv)
            check(h["calls"] == m["calls"],
                  f"save_log: the call trace is byte-equal to the head's ({label})")
            check(stable(h["phases"]) == stable(m["phases"]),
                  f"save_log: the phase log is byte-equal to the head's ({label})")
            if trunc:
                cut = int(trunc)
                longest = max(len(c.get("prompt") or "") for c in m["calls"])
                check(longest <= cut + len("... [truncated]"),
                      f"save_log: prompts are truncated at {cut} ({longest} chars)")
                check(any((c.get("response_text") or "").endswith("... [truncated]")
                          for c in m["calls"]),
                      "save_log: responses are truncated too")
                check(m["calls"] != CALLS and self_calls_intact(),
                      "save_log: truncation copies rather than mutating the trace")
            else:
                check(m["calls"] == CALLS,
                      "save_log: untruncated is the call list itself")

        os.environ.pop("CALLS_TRUNCATE_CHARS", None)
        os.environ.pop("PHASE_CACHE_DIR", None)
        os.environ.pop("LLM_LOG_DIR", None)

    check(head._compute_phase_metrics() == mine._compute_phase_metrics()
          == INFRA.phase_metrics(CALLS),
          "phase_metrics: the head's per-phase table")
    got = mine._compute_phase_metrics()
    check(got["p1"] == {"calls": 2, "elapsed_s": 1.75, "tokens": 10, "errors": 1},
          f"phase_metrics: errors and tokens accumulate ({got['p1']})")
    check(got["unknown"]["calls"] == 1,
          "phase_metrics: a phaseless call lands under 'unknown'")
    check(got["p2"]["tokens"] == 0,
          "phase_metrics: a null total_tokens counts as 0")

    #: The running sum is re-rounded every call. 0.1 + 0.2 is the case that reads
    #: 0.30000000000000004 unrounded, so a sample that sums exactly cannot see it.
    inexact = [{"phase": "p", "elapsed_s": 0.1}, {"phase": "p", "elapsed_s": 0.2}]
    check(INFRA.phase_metrics(inexact)["p"]["elapsed_s"] == 0.3,
          "phase_metrics: the running sum is rounded, not accumulated raw")
    check(bare(SLinker92, _llm_calls=inexact)._compute_phase_metrics()
          == INFRA.phase_metrics(inexact),
          "phase_metrics: the head rounds it the same way")


def self_calls_intact():
    """The module's own CALLS list is untouched by a truncating write."""
    return len(CALLS[0]["prompt"]) == 40 and len(CALLS[2]["response_text"]) == 90


# ── 4. the views ─────────────────────────────────────────────────────────────

class Sent:
    def __init__(self, number, text):
        self.number, self.text = number, text


def views():
    items = list(range(23))
    for n in (1, 5, 25, 100):
        check(list(SLinker92._iter_batches(items, n))
              == list(SLinker110._iter_batches(items, n))
              == list(INFRA.iter_batches(items, n)),
              f"iter_batches: the head's batches at n={n}")
    check(list(SLinker110._iter_batches([], 5)) == [],
          "iter_batches: nothing to batch yields nothing")

    sent_map = {3: Sent(3, "third"), 9: Sent(9, "ninth")}
    unmapped = LINKS + [SadSamLink(99, "c9", "Gamma", source="partial_name")]
    check(SLinker92._link_view(unmapped, sent_map)
          == SLinker110._link_view(unmapped, sent_map),
          "link_view: the head's rows, unmapped sentences dropped")
    check(len(SLinker110._link_view(unmapped, sent_map)) == 2,
          "link_view: the sentence outside the map is dropped")

    decisions = {(3, "c1"): {"approved": True, "reason": "r"},
                 (9, "c2"): {"approved": False}}
    check(SLinker92._decision_view(decisions)
          == SLinker110._decision_view(decisions),
          "decision_view: the head's rows")

    feedbacks = [
        {"candidates": [{"sentence": 1, "component": "A"},
                        {"sentence": 2, "component": "B"}],
         "accepted": [{"sentence": 1, "component": "A"}]},
        {"proposed": [{"sentence": 4, "component": "C"}], "accepted": []},
        {"accepted": [{"sentence": 5, "component": "D"}]},
        {},
    ]
    for i, fb in enumerate(feedbacks):
        check(SLinker92._linker_feedback(fb) == SLinker110._linker_feedback(fb),
              f"linker_feedback: the head's split, shape {i}")
    check(SLinker110._linker_feedback(feedbacks[0]) == {
        "accepted": [{"sentence": 1, "component": "A"}],
        "rejected": [{"sentence": 2, "component": "B"}]},
        "linker_feedback: a proposal not accepted is rejected")


def main():
    tracing()
    call_path()
    artifacts()
    views()
    passed = sum(1 for ok, _ in CHECKS if ok)
    print(f"\n{passed}/{len(CHECKS)} checks")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
