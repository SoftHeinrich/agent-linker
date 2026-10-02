"""Retained experimental linker exports (s126-only consolidation).

This branch keeps only the paper arm, ``s_linker126`` ("s126"). It is
standalone -- the ``s122 -> s123 -> s125 -> s126`` subclass chain was
flattened into it, so it does not import any other ``s_linkerNNN`` module at
runtime, and nothing else in this package does either.
"""

from .s_linker126 import SLinker126

__all__ = [
    "SLinker126",
]
