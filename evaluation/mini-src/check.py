#!/usr/bin/env python3
"""Regression check for mini-src/metrics.py (self-contained, stdlib only).

mini-src is the project's sole metrics implementation. The canonical
``src/lib/`` stack it was once reduced from has been retired to ``archive/``
(see mini-src/README.md), so this check no longer cross-scores against it.
Instead it freezes the panel that mini-src produces for the bundled TransArc
results and asserts it reproduces, cell for cell, to 1e-4.

Provenance of the frozen numbers (validated at retirement, 2026-06):
  * sad-code file/component == the then-canonical
    ``metrics_api.compute_sad_code_metrics`` primary panel;
  * ``worst_component_f1`` == ``COMPONENT_SUITE_sad-code.csv`` ``min_comp``
    (swattr/transarc rows) from the interface-dropped component_suite;
  * ``harmonic_component_f1`` == harmonic mean of the per-component F1 set,
    independently reproduced.
  * RE-FROZEN 2026-08-27 for the four worst/harmonic columns only: per-component
    F_beta is now computed over the LINKS whose target file belongs to the component
    (metric.tex eq:worst, read literally) instead of over a (sentence, component)
    projection. Only Teammates and BigBlueButton move -- the other three projects
    have per-component slices the projection did not collapse. Every non-tail column
    was re-asserted unchanged in the same run.
  * CMR/CMC (sad-sam only) added 2026-06-30: Component Miss Rate (%) + Count, the
    doc-model size-aware metric, component--sentence denominator. Frozen from the
    bundled TransArc doc-model results; sad-code goldens are unchanged. (Named
    SFM/SFC until 2026-08-27; the values are unchanged by the rename.)
  * The noise-rate column was dropped 2026-08-27 -- it was never defined in the
    paper and nothing downstream reads it. Every other cell was re-asserted
    unchanged in the same run, so this is a deletion, not a re-freeze.
  * ``component_f2`` / ``worst_component_f2`` / ``harmonic_component_f2`` added
    2026-08-27, when the paper moved to reporting \ftwo beside every \fone. The
    F1 columns above were re-asserted unchanged in the same run, so the three new
    columns are additions to this panel, not a re-freeze of it.

    python3 mini-src/check.py        # -> PASS
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as mini   # noqa: E402  (mini-src/metrics.py)

TOL = 1e-4

# Frozen golden panels for the bundled TransArc results, in PANELS[task] order.
GOLDEN = {
    "sad-code": {
        # Each component F1 is followed by its \ftwo twin, as PANELS["sad-code"] orders them.
        # file_P   file_R   file_F1  file_F2  comp_F1  comp_F2  worst_F1 worst_F2 harm_F1  harm_F2
        "mediastore":    (0.9615, 0.4237, 0.5882, 0.4771, 0.7391, 0.6538, 0.0000, 0.0000, 0.0000, 0.0000),
        "teastore":      (1.0000, 0.7086, 0.8295, 0.7525, 0.8511, 0.7812, 0.6667, 0.5556, 0.8421, 0.7692),
        "teammates":     (0.7531, 0.9024, 0.8211, 0.8680, 0.7158, 0.6501, 0.5366, 0.6548, 0.7570, 0.8156),
        "bigbluebutton": (0.8203, 0.8417, 0.8309, 0.8373, 0.8322, 0.8031, 0.5714, 0.4545, 0.8377, 0.8029),
        "jabref":        (0.8927, 1.0000, 0.9433, 0.9765, 0.9565, 0.9821, 0.8000, 0.9091, 0.9412, 0.9756),
    },
    "sad-sam": {
        # link_P   link_R   link_F1  link_F2  CMR%      CMC
        "mediastore":    (0.9444, 0.5484, 0.6939, 0.5986, 35.4839, 3.0000),
        "teastore":      (1.0000, 0.7407, 0.8511, 0.7812, 0.0000, 0.0000),
        "teammates":     (0.6049, 0.8596, 0.7101, 0.7929, 0.0000, 0.0000),
        "bigbluebutton": (0.8980, 0.7097, 0.7928, 0.7407, 0.0000, 0.0000),
        "jabref":        (0.9000, 1.0000, 0.9474, 0.9783, 0.0000, 0.0000),
    },
}

COMPUTE = {"sad-code": mini.compute_sad_code, "sad-sam": mini.compute_sad_sam}


def run():
    failures = 0
    checked = 0
    for task in ("sad-code", "sad-sam"):
        cols = mini.PANELS[task]
        compute = COMPUTE[task]
        for proj in mini.PROJECTS:
            path = mini.result_path(proj, None, None, task)
            res = mini.load_result(path, task)
            if not res:
                print(f"SKIP  {task:8} {proj:14} (no results at {path})")
                continue
            checked += 1
            row = compute(proj, res)
            gold = GOLDEN[task][proj]
            bad = [(c, row[c], g) for c, g in zip(cols, gold) if abs(row[c] - g) > TOL]
            for c, mv, g in bad:
                failures += 1
                print(f"FAIL  {task:8} {proj:14} {c:22} got={mv:.4f} expected={g:.4f}")
            if not bad:
                print(f"OK    {task:8} {proj:14} "
                      + "  ".join(f"{c}={row[c]:.4f}" for c in cols))

    print()
    if failures:
        print(f"FAILED: {failures} mismatch(es)")
        return 1
    # A run where every cell SKIPped (wrong $TRANSARC_RESULTS_DIR, say) used to print
    # PASS: zero comparisons, zero failures. Demand the full panel instead, so a
    # misconfigured environment can never be mistaken for a green regression check.
    expected = len(mini.PROJECTS) * len(GOLDEN)
    if checked != expected:
        print(f"FAILED: only {checked}/{expected} cells were scored -- the bundled "
              f"TransArc results were not found. Set $TRANSARC_RESULTS_DIR to the "
              f"mini-data root (see HOWTO-REGENERATE-RQ.md).")
        return 1
    print(f"PASS: mini-src/metrics.py reproduces the frozen golden panel "
          f"({checked} cells, sad-code + sad-sam).")
    return 0


if __name__ == "__main__":
    sys.exit(run())
