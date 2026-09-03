# The infra extraction (`linker_infra`) — the plumbing 72 linkers shared, lifted once

Eleven blocks of `s_linker110.py` were **byte-identical in 72 linker modules**
(`s_linker21` through `s_linker110`, measured by AST-splitting every module in
`linkers/experimental/` and hashing each block). They are infrastructure in the strict
sense this branch means — **no decision rule reads them**. Not one can change what
counts as a link; they decide how a request is traced, how an empty reply is retried,
where a checkpoint lands and what a log row looks like.

They now live in `src/llm_sad_sam/linkers/experimental/linker_infra.py`.

    block                     lines  what it is
    _TracingLLMClient           64   the per-call audit trail + the fail-fast rule
    _current_phase / local       6   the phase tag a trace row carries
    _ask                        44   the JSON call path: 3 success rules, 1 retry
    _backend_tag                 9   the backend's own name, through the wrapper
    _checkpoint_dir              7   PHASE_CACHE_DIR/variant/backend/dataset
    _save_phase                  6   one phase's pickle
    _log                        10   a phase-log row
    _save_log                   30   the phase log + the call trace, CALLS_TRUNCATE
    _compute_phase_metrics      15   calls/elapsed/tokens/errors per phase
    _iter_batches                8   1-indexed batching
    _link_view / _decision_view 20   the log's two views
    _linker_feedback            18   accepted/rejected reduction
                              ─────
                                237  of s_linker110.py's 1661

## Why functions and not a mixin

The reported arm is a **self-contained file** — the paper's supplement is
`s_linker110.py`, so what the arm runs has to be readable without walking an MRO
(`approach/CLAUDE.md`), and `pilot/test_s110_shortlist.py` asserts
`SLinker110.__mro__ == (SLinker110, object)`. A mixin would have broken that invariant
and the claim under it.

So `linker_infra` exports **functions and one wrapper class, never a mixin**, and each
method stays in the variant under its own name with a one-line body:

    def _save_log(self, text_path):
        write_run_logs(text_path, self._VARIANT_NAME, self._backend_tag(),
                       self._phase_log, self._llm_calls)

The method is still where a reader looks for it; only the plumbing moved. Each helper
takes what it needs as an argument rather than reading it off `self`, which is what
keeps a variant's override of a neighbouring method honoured — a variant that overrides
`_backend_tag` or `_checkpoint_dir` still decides the tag and the directory, because its
own method computes them and passes them in. **`SLinker110`'s MRO is unchanged and the
approach is untouched**: every prompt, every rule constant, every scan, the union rule,
`N(c)` and the whole three-linker pipeline stay in the variant's own file.

## What was measured

**The extraction is equivalent, and the equivalence is tested against the polarity it
preserves** — the `s_linker114` lesson, which this round had to obey twice.

| level | what | result |
|---|---|---|
| unit | every helper vs. **`s_linker92`'s untouched copy of the same block** — the module the ledger records, which this branch does not edit | **107/107** (`pilot/test_linker_infra.py`) |
| composed | `link()` over all five projects, pre-refactor file reconstructed from git at `d765a027`, one deterministic stubbed client, **no LLM calls** | **51/51** (`pilot/test_infra_refactor_e2e.py`) |
| invariant | the standalone invariants, MRO included | **235/235** (`pilot/test_s110_shortlist.py`, was 224) |
| invariant | the RQ4 floor arm | **27/27** (`pilot/test_s110_onecall.py`) |
| adversarial | 11 deliberate divergences injected into `linker_infra.py` | **11/11 caught** (`mutation_check.log`) |

The composed test compares, per project: the links returned, the call count, the whole
call trace, the phase log, the phase metrics, the workflow history, the five
checkpoints, the final checkpoint's contents and the two log artifacts — everything but
the wall-clock fields, which are the only fields that cannot be equal. It is not
vacuous: 63 stubbed calls and 105 links across the five projects, with approvals *and*
rejections at every judge, because the stub alternates its verdicts off the prompt hash
rather than answering nothing.

**Two of the eleven mutants survived the first version of the unit suite**, and both
were gaps in the test rather than in the helper: no case set `require` and
`require_present` together, so the documented priority order was never exercised; and
the sample's `elapsed_s` values summed exactly, so re-rounding the running sum was
invisible (`0.1 + 0.2` is the case that reads `0.30000000000000004` unrounded). Both are
now covered, which is what took the suite from 93 to 107 checks.

**One real defect was caught before it shipped**: the first draft of
`TracingLLMClient` dropped `__getattr__`, the delegation `describe_backend()` and
`extract_json()` both ride on.

## What the invariant test now checks instead of bytes

The eleven blocks are exempt from `test_s110_shortlist.py`'s byte comparison against
`s_linker92` — they cannot be byte-equal any more. They did not become a hole: each one
gets a **stronger** check, that the block which stayed in the file is a delegation to
its named helper and is no longer than the block it replaced, with behavioural
equivalence proven by the two tests above. Net **+22 / −11 checks**, 224 → 235.

## Reproduce

    cd approach
    ../.venv/bin/python pilot/test_linker_infra.py            # 107/107
    ../.venv/bin/python pilot/test_infra_refactor_e2e.py      #  51/51
    ../.venv/bin/python pilot/test_s110_shortlist.py          # 235/235
    ../.venv/bin/python pilot/test_s110_onecall.py            #  27/27

`pilot/test_infra_refactor_e2e.py --before REV` names the revision the pre-refactor
file is read from; it defaults to `d765a027`, the refactor's parent.

Logs in this directory: `test_linker_infra.log`, `test_infra_refactor_e2e.log`,
`test_s110_shortlist.log`, `test_s110_onecall.log`, `mutation_check.log`.

## Not measured, and why nothing is owed

No prompt, rule constant, batch bound, scan or judge changed, and the composed test
shows `link()` byte-identical over five projects. **No E2E is owed and the head does
not move: `s_linker110` stands.**
