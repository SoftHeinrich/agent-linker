"""`linker_infra`'s composed invariant: `link()` is byte-identical across the refactor.

`pilot/test_linker_infra.py` checks each extracted helper against `s_linker92`'s
untouched copy of the same block. That is the unit half. This is the composed half,
and it is the one the `s_linker114` lesson demands: a refactor's equivalence test must
exercise the pipeline it preserves, not only the pieces in isolation.

It reconstructs the **pre-refactor `s_linker110.py`** out of git at `BEFORE_REV`,
imports it beside the current file, and runs both through `link()` over all five
projects under one deterministic stubbed client -- replies are a fixed function of the
prompt's own hash, so both files see byte-identical input and every parser branch runs.
Compared per project: the links returned, the call count, the whole call trace, the
phase log, the phase metrics, the workflow history, the five checkpoints, the final
checkpoint's contents and the two log artifacts. Wall-clock fields are dropped, being
the only fields that cannot be equal.

**No LLM calls and no network** -- the stub is the client.

    ../.venv/bin/python pilot/test_infra_refactor_e2e.py [--before REV]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS                                # noqa: E402

#: The commit that carries `s_linker110.py` with the eleven infrastructure blocks
#: still inlined -- the head of the s110 doc round, the refactor's parent.
BEFORE_REV = "d765a027"

EXPERIMENTAL = Path("src/llm_sad_sam/linkers/experimental")
BEFORE_MODULE = "s_linker110_beforeinfra"
VOLATILE_CALL = ("ts", "elapsed_s", "latency_ms")

CHECKS = []


def check(condition, label):
    CHECKS.append((bool(condition), label))
    if not condition:
        print(f"  FAIL  {label}")


# ── the deterministic stand-in ───────────────────────────────────────────────

class Resp:
    def __init__(self, text):
        self.text, self.success, self.error = text, True, None
        self.latency_ms, self.model = 5, "stub"
        self.token_usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 2,
                                          "total_tokens": 3})()


class StubClient:
    """Replies keyed by the prompt's own hash: same prompt, same answer, no network.

    The reply follows the schema each prompt asks for, and the verdicts alternate off
    the hash, so approvals and rejections both occur at every judge -- the polarity a
    stub that answered nothing would never reach.
    """

    backend = type("B", (), {"value": "stub"})()

    def describe_backend(self):
        return "stub backend"

    def query(self, prompt, timeout=180, max_retries=3):
        h = int(hashlib.sha256(prompt.encode()).hexdigest()[:2], 16)
        if '"abbreviations"' in prompt:
            body = '{"abbreviations": [], "aliases": []}'
        elif '"approved"' in prompt:
            body = '{"approved": []}'
        elif '"validations"' in prompt:
            body = json.dumps({"validations": [
                {"case": i, "claim": "c", "approve": (i + h) % 2 == 0}
                for i in range(1, 40)]}) if h % 2 == 0 else '{"validations": []}'
        elif '"judgments"' in prompt:
            body = json.dumps({"judgments": [
                {"case": i, "claim": "c",
                 "denotation": "participant" if (i + h) % 2 else "associated"}
                for i in range(1, 40)]})
        elif '"resolutions"' in prompt:
            body = '{"resolutions": []}'
        else:
            body = "{}"
        return Resp(body)

    def extract_json(self, resp):
        try:
            return json.loads(resp.text)
        except Exception:
            return None


# ── normalizers: everything but the clock ────────────────────────────────────

def norm_links(links):
    return sorted((l.sentence_number, l.component_id, l.component_name, l.source)
                  for l in links)


def norm_trace(calls):
    return [{k: v for k, v in c.items() if k not in VOLATILE_CALL} for c in calls]


def norm_metrics(metrics):
    return {k: {kk: vv for kk, vv in v.items() if kk != "elapsed_s"}
            for k, v in (metrics or {}).items()}


def norm_log(rows):
    out = []
    for row in rows:
        row = {k: v for k, v in row.items() if k != "ts"}
        summary = dict(row.get("out") or {})
        summary.pop("elapsed_s", None)
        if isinstance(summary.get("phase_metrics"), dict):
            summary["phase_metrics"] = norm_metrics(summary["phase_metrics"])
        row["out"] = summary
        out.append(row)
    return out


def run(cls, project, cache, logs):
    """`link()` over one project, with the stub in place of the real client."""
    from llm_sad_sam.linkers.experimental.linker_infra import TracingLLMClient
    text, model, _ = PROJECTS[project]
    os.environ["PHASE_CACHE_DIR"] = cache
    os.environ["LLM_LOG_DIR"] = logs
    obj = cls.__new__(cls)          # __init__ builds a real client; this does not
    obj._llm_calls = []
    obj.llm = TracingLLMClient(StubClient(), obj._llm_calls)
    obj.no_knowledge = False
    obj.doc_knowledge = None
    obj._phase_log, obj._phase_metrics, obj.workflow = [], {}, []
    return obj, obj.link(str(BENCH / text), str(BENCH / model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=BEFORE_REV,
                    help=f"revision carrying the pre-refactor file (default {BEFORE_REV})")
    args = ap.parse_args()

    rel = "approach/src/llm_sad_sam/linkers/experimental/s_linker110.py"
    src = subprocess.run(["git", "show", f"{args.before}:{rel}"],
                         capture_output=True, text=True, cwd="..", check=True).stdout
    check("_TracingLLMClient" in src and "def _save_log" in src,
          f"{args.before} carries the pre-refactor file (blocks still inlined)")

    before_path = EXPERIMENTAL / f"{BEFORE_MODULE}.py"
    before_path.write_text(src.replace("class SLinker110:", "class SLinker110Before:", 1))
    try:
        from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110 as AFTER
        before_mod = __import__(
            f"llm_sad_sam.linkers.experimental.{BEFORE_MODULE}", fromlist=["x"])
        BEFORE = before_mod.SLinker110Before

        for project in PROJECTS:
            with tempfile.TemporaryDirectory() as tmp:
                b, bl = run(BEFORE, project, f"{tmp}/bc", f"{tmp}/bl")
                a, al = run(AFTER, project, f"{tmp}/ac", f"{tmp}/al")

                check(norm_links(bl) == norm_links(al),
                      f"{project}: link() returns the same links ({len(al)})")
                check(len(b._llm_calls) == len(a._llm_calls),
                      f"{project}: the same call count ({len(a._llm_calls)})")
                check(norm_trace(b._llm_calls) == norm_trace(a._llm_calls),
                      f"{project}: the call trace is identical")
                check(norm_log(b._phase_log) == norm_log(a._phase_log),
                      f"{project}: the phase log is identical")
                check(norm_metrics(b._phase_metrics) == norm_metrics(a._phase_metrics),
                      f"{project}: the phase metrics are identical")
                check(b.workflow == a.workflow,
                      f"{project}: the workflow history is identical")

                bcp = sorted(p.name for p in Path(f"{tmp}/bc").rglob("*.pkl"))
                acp = sorted(p.name for p in Path(f"{tmp}/ac").rglob("*.pkl"))
                check(bcp == acp and len(acp) == 5,
                      f"{project}: the same five checkpoints ({acp})")
                bfin = pickle.loads(next(Path(f"{tmp}/bc").rglob("final.pkl")).read_bytes())
                afin = pickle.loads(next(Path(f"{tmp}/ac").rglob("final.pkl")).read_bytes())
                check(norm_links(bfin["final"]) == norm_links(afin["final"])
                      and bfin["workflow"] == afin["workflow"],
                      f"{project}: the final checkpoint is identical")
                check(len(list(Path(f"{tmp}/al").iterdir())) == 2,
                      f"{project}: the two log artifacts are written")
                # non-vacuity: the run has to have gone through the judges
                check(len(a._llm_calls) >= 5,
                      f"{project}: the pipeline actually called ({len(a._llm_calls)})")
                print(f"  {project}: {len(al)} links, {len(a._llm_calls)} stubbed calls,"
                      f" {len(acp)} checkpoints")
    finally:
        before_path.unlink(missing_ok=True)
        for stale in (EXPERIMENTAL / "__pycache__").glob(f"{BEFORE_MODULE}*"):
            stale.unlink(missing_ok=True)
        for key in ("PHASE_CACHE_DIR", "LLM_LOG_DIR"):
            os.environ.pop(key, None)

    passed = sum(1 for ok, _ in CHECKS if ok)
    print(f"\n{passed}/{len(CHECKS)} checks")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
