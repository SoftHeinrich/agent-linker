"""Retained experimental linker exports (s126-only consolidation).

This branch keeps only the paper arm, ``s_linker126`` ("s126"), and the modules
its own validation suite depends on. ``SLinker126`` is standalone — the
``s122 -> s123 -> s125 -> s126`` subclass chain was flattened into it, so it does
not import any other ``s_linkerNNN`` module at runtime. ``s_linker122``,
``s_linker123`` and ``s_linker125`` are kept only because ``pilot/test_s126.py``
and ``pilot/test_s126_standalone.py`` compare s126 against them, and
``s_linker25`` is kept because ``pilot/design_audit.py`` (used by the s126 test
chain via ``pilot/score_runs.py``) imports it. The ablation runner imports
submodules by full path (via ``importlib``), so no namespace-level re-exports
are required for it specifically.
"""

from .s_linker25 import SLinker25
from .s_linker122 import SLinker122
from .s_linker123 import SLinker123
from .s_linker125 import SLinker125
from .s_linker126 import SLinker126

__all__ = [
    "SLinker25",
    "SLinker122",
    "SLinker123",
    "SLinker125",
    "SLinker126",
]
