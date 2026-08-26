#!/usr/bin/env python3
"""Regression check for mini-src/metrics.py (self-contained, stdlib only).

mini-src is the project's sole metrics implementation. The canonical
``src/lib/`` stack it was once reduced from has been retired to ``archive/``
(see mini-src/README.md), so this check no longer cross-scores against it.
Instead it freezes the panel that mini-src produces for the bundled TransArc
results and asserts it reproduces, cell for cell, to 1e-4.

Provenance of the frozen numbers (validated at retirement, 2026-06):
  * sad-code file/component/coverage/noise == the then-canonical
    ``metrics_api.compute_sad_code_metrics`` primary panel;
  * ``worst_component_f1`` == ``COMPONENT_SUITE_sad-code.csv`` ``min_comp``
    (swattr/transarc rows) from the interface-dropped component_suite;
  * ``harmonic_component_f1`` == harmonic mean of the per-component F1 set,
    independently reproduced.
  * SFM/SFC (sad-sam only) added 2026-06-30: Silent-Failure Mass (%) + Count, the
    doc-model size-aware metric, component--sentence denominator. Frozen from the
    bundled TransArc doc-model results; sad-code goldens are unchanged.

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
        # file_P   file_R   file_F1  file_F2 comp_F1  worst_C  harm_C   sent_cov noise
        "mediastore":    (0.9615, 0.4237, 0.5882, 0.4771, 0.7391, 0.0000, 0.0000, 0.6400, 0.0588),
        "teastore":      (1.0000, 0.7086, 0.8295, 0.7525, 0.8511, 0.6667, 0.8421, 0.6957, 0.0000),
        "teammates":     (0.7531, 0.9024, 0.8211, 0.8680, 0.7158, 0.5128, 0.7144, 0.5978, 0.2991),
        "bigbluebutton": (0.8203, 0.8417, 0.8309, 0.8373, 0.8322, 0.7429, 0.8695, 0.8222, 0.2114),
        "jabref":        (0.8927, 1.0000, 0.9433, 0.9765, 0.9565, 0.8000, 0.9412, 1.0000, 0.0823),
    },
    "sad-sam": {
        # link_P   link_R   link_F1  link_F2 sent_cov noise    SFM%      SFC
        "mediastore":    (0.9444, 0.5484, 0.6939, 0.5986, 0.5926, 0.0588, 35.4839, 3.0000),
        "teastore":      (1.0000, 0.7407, 0.8511, 0.7812, 0.6957, 0.0000,  0.0000, 0.0000),
        "teammates":     (0.6049, 0.8596, 0.7101, 0.7929, 0.8444, 0.4470,  0.0000, 0.0000),
        "bigbluebutton": (0.8980, 0.7097, 0.7928, 0.7407, 0.8125, 0.0813,  0.0000, 0.0000),
        "jabref":        (0.9000, 1.0000, 0.9474, 0.9783, 1.0000, 0.1000,  0.0000, 0.0000),
    },
}

COMPUTE = {"sad-code": mini.compute_sad_code, "sad-sam": mini.compute_sad_sam}


def run():
    failures = 0
    for task in ("sad-code", "sad-sam"):
        cols = mini.PANELS[task]
        compute = COMPUTE[task]
        for proj in mini.PROJECTS:
            path = mini.result_path(proj, None, None, task)
            res = mini.load_result(path, task)
            if not res:
                print(f"SKIP  {task:8} {proj:14} (no results at {path})")
                continue
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
    print("PASS: mini-src/metrics.py reproduces the frozen golden panel (sad-code + sad-sam).")
    return 0


if __name__ == "__main__":
    sys.exit(run())
