"""Linker infrastructure: the LLM call path, the run's artifacts, the log views.

Every block here was byte-identical in **72 linker modules** (`s_linker21` through
`s_linker110`), and it is infrastructure in the strict sense the branch means: no
decision rule reads it. Nothing here can change what counts as a link -- it decides
how a request is traced, how an empty reply is retried, where a checkpoint lands and
what a log row looks like. The approach itself -- every prompt, every rule constant,
every scan, the union rule, the pipeline -- stays in the variant's own file.

**Why this is not a base class.** The reported arm is a self-contained file: the
paper's supplement is `s_linker110.py`, so what the arm runs has to be readable
top-to-bottom without walking an MRO (`approach/CLAUDE.md`, and
`pilot/test_s110_shortlist.py` asserts `SLinker110.__mro__ == (SLinker110, object)`).
So this module exports *functions and one wrapper class*, never a mixin: a variant
keeps its method under its own name and the body is a named call. The method still
appears where a reader looks for it; only the plumbing moved.

Each function takes what it needs as an argument rather than reading it off `self`,
which is what keeps a variant's override of a neighbouring method honoured -- a
variant that overrides `_backend_tag` or `_checkpoint_dir` still decides the tag and
the directory, because its own method computes them and passes them in.

Equivalence to the bytes this replaces is tested rather than asserted:
`pilot/test_linker_infra.py` runs every function here against `s_linker92`'s untouched
copy of the same block, which is the reference the ledger records.
"""
from __future__ import annotations

import json
import os
import pickle
import threading
import time

from llm_sad_sam.llm_client import LLMClient, LLMResponse

__all__ = [
    "current_phase",
    "TracingLLMClient",
    "ask_json",
    "backend_tag",
    "checkpoint_dir",
    "save_phase_state",
    "log_entry",
    "write_run_logs",
    "phase_metrics",
    "iter_batches",
    "link_view",
    "decision_view",
    "linker_feedback",
]


# ─────────────────────────────────────────────────────────────────────────────
# Tracing infrastructure — per-LLM-call audit trail
# ─────────────────────────────────────────────────────────────────────────────

_phase_local = threading.local()


def current_phase() -> str:
    return getattr(_phase_local, "phase", "unknown")


class TracingLLMClient:
    """Delegating wrapper that records every query() into a phase-tagged trace."""

    def __init__(self, inner: LLMClient, sink: list[dict]):
        self._inner = inner
        self._sink = sink
        self._sink_lock = threading.Lock()

    def set_phase(self, name: str) -> None:
        _phase_local.phase = name

    def query(self, prompt: str, timeout: int = 180, max_retries: int = 3) -> LLMResponse:
        phase = current_phase()
        t0 = time.time()
        try:
            resp = self._inner.query(prompt, timeout=timeout, max_retries=max_retries)
        except Exception as exc:
            record = {
                "phase": phase, "ts": t0,
                "elapsed_s": round(time.time() - t0, 3),
                "timeout": timeout, "max_retries": max_retries,
                "prompt": prompt,
                "response_text": None,
                "success": False,
                "error": f"FATAL: {exc}",
                "latency_ms": None,
                "model": None,
            }
            with self._sink_lock:
                self._sink.append(record)
            raise
        record = {
            "phase": phase, "ts": t0,
            "elapsed_s": round(time.time() - t0, 3),
            "timeout": timeout, "max_retries": max_retries,
            "prompt": prompt,
            "response_text": getattr(resp, "text", None),
            "success": getattr(resp, "success", None),
            "error": getattr(resp, "error", None),
            "latency_ms": getattr(resp, "latency_ms", None),
            "model": getattr(resp, "model", None),
        }
        usage = getattr(resp, "token_usage", None)
        if usage is not None:
            record["token_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        with self._sink_lock:
            self._sink.append(record)
        # A phase result may only be interpreted after every required request
        # succeeds. Returning a failed response lets extract_json() turn it into
        # None and silently omit an entire batch.
        if not resp.success:
            raise RuntimeError(f"LLM request failed in {phase}: {resp.error}")
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ─────────────────────────────────────────────────────────────────────────────
# The LLM call path
# ─────────────────────────────────────────────────────────────────────────────

def ask_json(
    llm,
    prompt: str,
    *,
    attempts: int,
    timeout: int = 120,
    label: str = "LLM call",
    phase: str | None = None,
    require: str | None = None,
    require_present: str | None = None,
) -> dict:
    """Query the LLM, parse JSON, retry once on empty/incomplete response.

    Success rule, in priority order:
      - require_present=KEY  → KEY must appear in the parsed dict (empty OK)
      - require=KEY          → data[KEY] must be truthy
      - neither              → any non-empty parsed dict succeeds

    ``attempts`` is the caller's ``ASK_ATTEMPTS``: a resource bound, so it stays a
    declared constant on the variant rather than a default hidden in here.
    """
    if phase is not None:
        llm.set_phase(phase)

    def _ok(d: dict | None) -> bool:
        if not d:
            return False
        if require_present is not None:
            return require_present in d
        if require is not None:
            return bool(d.get(require))
        return True

    data: dict = {}
    for attempt in range(attempts):
        parsed = llm.extract_json(llm.query(prompt, timeout=timeout))
        # Each attempt replaces the last. Keeping a previous attempt's dict
        # when a later one fails to parse would return a payload this method
        # already rejected, and callers read it as if it had passed.
        data = parsed if parsed is not None else {}
        if _ok(data):
            return data
        if attempt < attempts - 1:
            print(f"    {label}: empty response, retrying...")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Logging and checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def backend_tag(llm) -> str:
    """The backend's own name, read through the tracing wrapper if there is one."""
    inner = getattr(llm, "_inner", llm)
    backend = getattr(inner, "backend", None)
    if backend is None:
        return "unknown"
    return getattr(backend, "value", str(backend))


def checkpoint_dir(text_path, variant: str, backend: str) -> str:
    cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
    ds = os.path.splitext(os.path.basename(text_path))[0]
    d = os.path.join(cache_dir, variant, backend, ds)
    os.makedirs(d, exist_ok=True)
    return d


def save_phase_state(directory: str, phase_name: str, state) -> None:
    """Pickle one phase's state into an already-resolved checkpoint directory.

    The directory is an argument and not recomputed here, so a variant that
    overrides where its checkpoints land keeps deciding it.
    """
    path = os.path.join(directory, f"{phase_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(state, f)
    print(f"  Checkpoint: {phase_name} saved")


def log_entry(phase, input_summary, output_summary, links=None) -> dict:
    entry = {"phase": phase, "ts": time.time(),
             "in": input_summary, "out": output_summary}
    if links is not None:
        entry["links"] = [
            {"s": l.sentence_number, "c": l.component_name, "src": l.source}
            for l in links
        ]
    return entry


def write_run_logs(text_path, variant: str, backend: str,
                   phase_log: list[dict], llm_calls: list[dict]) -> None:
    """The two artifacts a scored run leaves: the phase log and the call trace."""
    log_dir = os.environ.get("LLM_LOG_DIR", "./results/llm_logs")
    os.makedirs(log_dir, exist_ok=True)
    ds = os.path.splitext(os.path.basename(text_path))[0]
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(log_dir, f"{variant}_{backend}_{ds}_{ts}.json")
    with open(summary_path, "w") as f:
        json.dump(phase_log, f, indent=2, default=str)
    print(f"  Phase log saved: {summary_path}")
    calls_path = os.path.join(log_dir, f"{variant}_{backend}_{ds}_{ts}_calls.json")
    trunc_env = os.environ.get("CALLS_TRUNCATE_CHARS", "").strip()
    trunc = int(trunc_env) if trunc_env.isdigit() else 0
    if trunc > 0:
        calls = []
        for c in llm_calls:
            cc = dict(c)
            if cc.get("prompt") and len(cc["prompt"]) > trunc:
                cc["prompt"] = cc["prompt"][:trunc] + "... [truncated]"
            if cc.get("response_text") and len(cc["response_text"]) > trunc:
                cc["response_text"] = cc["response_text"][:trunc] + "... [truncated]"
            calls.append(cc)
    else:
        calls = llm_calls
    with open(calls_path, "w") as f:
        json.dump(calls, f, indent=2, default=str)
    print(f"  LLM call trace saved: {calls_path} ({len(llm_calls)} calls)")


def phase_metrics(llm_calls: list[dict]) -> dict:
    metrics: dict[str, dict] = {}
    for call in llm_calls:
        ph = call.get("phase", "unknown")
        m = metrics.setdefault(
            ph, {"calls": 0, "elapsed_s": 0.0, "tokens": 0, "errors": 0})
        m["calls"] += 1
        m["elapsed_s"] = round(m["elapsed_s"] + call.get("elapsed_s", 0.0), 3)
        if call.get("success") is False:
            m["errors"] += 1
        usage = call.get("token_usage")
        if usage:
            m["tokens"] += usage.get("total_tokens", 0) or 0
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Batching and the log's views
# ─────────────────────────────────────────────────────────────────────────────

def iter_batches(items, n):
    """Yield (batch_num, batch_slice) — batch_num is 1-indexed."""
    for i, start in enumerate(range(0, len(items), n), start=1):
        yield i, items[start:start + n]


def link_view(links, sent_map):
    return [
        {
            "sentence": link.sentence_number,
            "text": sent_map[link.sentence_number].text,
            "component": link.component_name,
            "source": link.source,
        }
        for link in links
        if link.sentence_number in sent_map
    ]


def decision_view(decisions):
    return [
        {"sentence": sentence, "component_id": component, **decision}
        for (sentence, component), decision in decisions.items()
    ]


def linker_feedback(feedback):
    """Reduce detailed linker evidence to accepted/rejected references."""
    proposed = feedback.get("candidates", feedback.get("proposed", []))
    accepted = feedback.get("accepted", [])
    accepted_keys = {(i["sentence"], i["component"]) for i in accepted}

    def reference(item):
        return {"sentence": item["sentence"], "component": item["component"]}

    return {
        "accepted": [reference(i) for i in accepted],
        "rejected": [
            reference(i) for i in proposed
            if (i["sentence"], i["component"]) not in accepted_keys
        ],
    }
