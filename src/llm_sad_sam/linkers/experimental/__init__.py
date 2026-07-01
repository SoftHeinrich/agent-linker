"""Retained experimental linker exports (s21-only branch).

This branch keeps only the canonical ``s_linker21`` ("s21") linker. ``SLinker21``
is standalone — the former ``s_linker20_union`` ("s20U") pipeline was inlined into
it and s20U removed, so it does not inherit from any other linker. The ablation
runner imports submodules by full path (via ``importlib``), so no namespace-level
re-exports are required here.
"""

from .s_linker21 import SLinker21
from .s_linker21_agentrouter import SLinker21AgentRouter

__all__ = [
    "SLinker21",
    "SLinker21AgentRouter",
]
