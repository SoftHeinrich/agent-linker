"""Retained experimental linker exports (s20U-trimmed branch).

This branch keeps only what is needed to run the ``s_linker20_union`` ("s20U")
sweep. ``SLinker20Union`` is standalone — it does not inherit from any other
linker — so the historical eager imports of the full s_linker/ilinker family
have been removed. The ablation runner imports these submodules by full path
(via ``importlib``), so no namespace-level re-exports are required here.
"""

from .s_linker20_union import SLinker20Union

__all__ = [
    "SLinker20Union",
]
