"""Find duplicate / near-duplicate methods in a linker module. No LLM calls.

Written because the union round found `_writes_name` and `_find_exact_form` to be the
same predicate under two names, kept apart only because the ancestor kept them apart.
A byte-identical-copy policy hides that class of duplicate by construction, so a
standalone file wants a check that does not care what anything is called.

Two passes:
  1. structural equality -- each method's AST with docstrings dropped and parameters
     renamed positionally, so `_writes_name` vs `_find_exact_form` (the pair this
     round already deleted) would be caught whatever the argument names were;
  2. behavioural equality -- every 2-argument (text, name)-shaped method run against
     every (sentence, component-name) pair of all five projects, and any two that
     agree on all of them reported. This is what catches a duplicate written
     differently, which pass 1 cannot.

    ../.venv/bin/python pilot/method_dup_audit.py
    ../.venv/bin/python pilot/method_dup_audit.py src/llm_sad_sam/linkers/experimental/s_linker110.py

Pass 2 runs with an EMPTY alias table, so two methods that differ only over discovered
aliases read as near-duplicates here; `_states_a_name` against `_find_exact_form` is
the standing example, and it differs on 43 of 3697 pairs once a recorded table is
loaded. Treat a NEAR line as a question, not a finding.
"""
import ast
import itertools
import sys
from pathlib import Path

ROOT = Path('.').resolve()
sys.path.insert(0, 'src')
sys.path.insert(0, 'pilot')

TARGET = sys.argv[1] if len(sys.argv) > 1 else \
    'src/llm_sad_sam/linkers/experimental/s_linker126.py'


class Norm(ast.NodeTransformer):
    """Rename parameters and locals positionally; drop docstrings."""

    def __init__(self, names):
        self.map = {n: f"v{i}" for i, n in enumerate(names)}

    def visit_Name(self, node):
        node.id = self.map.get(node.id, node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.map.get(node.arg, node.arg)
        return node


def normalise(fn):
    fn = ast.parse(ast.unparse(fn)).body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        fn.body = fn.body[1:]
    params = [a.arg for a in fn.args.args]
    fn.name = "f"
    fn.decorator_list = []
    return ast.dump(Norm(params).visit(fn))


src = open(TARGET).read()
tree = ast.parse(src)
cls = [n for n in tree.body if isinstance(n, ast.ClassDef)
       and n.name.startswith('SLinker')][0]
fns = [n for n in cls.body if isinstance(n, ast.FunctionDef)]

print(f"=== pass 1: structural duplicates among {len(fns)} methods ===")
seen = {}
for fn in fns:
    seen.setdefault(normalise(fn), []).append(fn.name)
dups = [v for v in seen.values() if len(v) > 1]
for group in dups:
    print("  DUPLICATE:", group)
if not dups:
    print("  none")

# ── pass 2 ───────────────────────────────────────────────────────────────────
from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from reading_pilots import BENCH, DATASETS
import importlib
mod = importlib.import_module(
    'llm_sad_sam.linkers.experimental.' + Path(TARGET).stem)
cls_obj = getattr(mod, cls.name)

pairs = []
for project, (text, repo, _) in sorted(DATASETS.items()):
    comps = parse_pcm_repository(str(BENCH / repo))
    sents = load_sentences(str(BENCH / text))
    for s in sents:
        for c in comps:
            pairs.append((s.text, c.name))
print(f"\n=== pass 2: behaviour over {len(pairs)} (sentence, name) pairs ===")

linker = cls_obj.__new__(cls_obj)
linker.doc_knowledge = None
cands = []
for fn in fns:
    args = [a.arg for a in fn.args.args]
    if len(args) == 3 and args[0] in ("self", "cls"):
        cands.append(fn.name)
results = {}
for name in cands:
    fnobj = getattr(linker, name)
    try:
        results[name] = tuple(fnobj(t, n) for t, n in pairs)
    except Exception as exc:
        print(f"  (skipped {name}: {type(exc).__name__})")
print(f"  comparable 2-argument methods: {sorted(results)}")
found = False
for a, b in itertools.combinations(sorted(results), 2):
    if results[a] == results[b]:
        print(f"  BEHAVIOURALLY IDENTICAL: {a} == {b}")
        found = True
    else:
        same = sum(x == y for x, y in zip(results[a], results[b]))
        if same / len(pairs) > 0.99:
            print(f"  NEAR: {a} vs {b} agree on {same}/{len(pairs)}")
            found = True
if not found:
    print("  no behavioural duplicates")
