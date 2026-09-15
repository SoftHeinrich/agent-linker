"""Call-graph depth inside a linker class: how many hops to read one flow. No calls.

A standalone file is read top to bottom, so the cost of a method is not its length but
how many others a reader must open to follow one path. This prints the longest chain of
self-calls from `link()`, and then the methods that are **pure hops** -- one caller, and
a body that belongs in it.

Not every hop is waste. A prompt builder is one f-string and belongs alone; a primitive
with several callers (`_name_spans`, `_find_exact_form`) is depth worth paying for. What
this catches is the hop that exists because an ancestor had it.

    ../.venv/bin/python pilot/call_chain_audit.py
    ../.venv/bin/python pilot/call_chain_audit.py src/llm_sad_sam/linkers/experimental/s_linker110.py
"""
import ast, sys
from collections import defaultdict

TARGET = sys.argv[1] if len(sys.argv) > 1 else \
    'src/llm_sad_sam/linkers/experimental/s_linker120.py'
src = open(TARGET).read()
tree = ast.parse(src)
cls = [n for n in tree.body if isinstance(n, ast.ClassDef)
       and n.name.startswith('SLinker')][0]
fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

calls = defaultdict(set)
callers = defaultdict(set)
for name, fn in fns.items():
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr in fns:
            if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
                if node.attr != name:
                    calls[name].add(node.attr)
                    callers[node.attr].add(name)

ENTRY = ["link"]


def longest(node, seen=()):
    """Longest simple path of self-calls from `node`."""
    if node in seen:
        return [node]
    best = [node]
    for nxt in sorted(calls.get(node, ())):
        path = [node] + longest(nxt, seen + (node,))
        if len(path) > len(best):
            best = path
    return best


print(f"=== {TARGET.split('/')[-1]}: {len(fns)} methods ===")
for entry in ENTRY:
    path = longest(entry)
    print(f"\ndeepest chain from {entry}(): {len(path)} hops")
    print("   " + "\n     -> ".join(path))

print("\nper-method fan-out (methods it calls) and fan-in (callers):")
rows = sorted(fns, key=lambda n: -len(calls[n]))
for name in rows[:10]:
    print(f"   {name:26} calls {len(calls[name]):>2}   called by {len(callers[name]):>2}")

print("\nmethods on a chain of 3+ that have exactly one caller (pure hops):")
for name in sorted(fns):
    if len(callers[name]) == 1 and calls[name]:
        only = next(iter(callers[name]))
        print(f"   {only} -> {name} -> {sorted(calls[name])}")
