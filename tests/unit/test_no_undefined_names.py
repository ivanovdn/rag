"""Entry-point scripts are not exercised by any unit test, so a NameError in one
ships. This is the smallest static check that catches it.

eval/run_experiment.py's main() is ~150 lines and nothing runs it. Renaming a
local there on 2026-10-06 (`top_k` -> `candidates`) fixed two of three uses; the
third sat in the agent tiers' metadata dict and raised
`NameError: name 'top_k' is not defined` after the banner had printed and the
run had started -- on a 61-question GPU eval, after the only cheap failure point
(the version assertion) had already passed. The unit tests were green: they
inspect main()'s AST, they never execute it.

No linter is installed and the project adds no dependencies, so this is a small
one: for every top-level function, the names it reads must be bound somewhere in
its own subtree, at module level, or in builtins. A nested function's bindings
count for its parent, which is how a closure reading an enclosing local stays
clean -- deliberately permissive, since a false positive here blocks a commit
while a false negative only returns us to the status quo.
"""

import ast
import builtins
from pathlib import Path

import pytest

SCANNED = sorted(
    p for d in ("eval", "scripts") for p in Path(d).rglob("*.py")
)

_BUILTINS = set(dir(builtins))


def _bindings(node) -> set[str]:
    """Every name bound anywhere in this subtree."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
    return out


def _module_bindings(tree: ast.Module) -> set[str]:
    """Module scope only: recurse through if/for/try/with at module level, but
    NOT into function or class bodies -- a local in one function must not count
    as defined for another."""
    out = set()

    def walk(stmts):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(s.name)
                continue
            out.update(_bindings(s))  # not |=, which would rebind `out` as a local of walk()
            for field in ("body", "orelse", "finalbody"):
                walk(getattr(s, field, []) or [])
            for h in getattr(s, "handlers", []) or []:
                walk(h.body)

    walk(tree.body)
    return out


def undefined_names(source: str) -> dict[str, list[str]]:
    """{function name: sorted undefined names}. Empty when the module is clean."""
    tree = ast.parse(source)
    module = _module_bindings(tree)
    found = {}
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        loaded = {
            n.id for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        undef = loaded - _bindings(fn) - module - _BUILTINS
        if undef:
            found[fn.name] = sorted(undef)
    return found


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: str(p))
def test_no_function_reads_a_name_nothing_binds(path):
    found = undefined_names(path.read_text(encoding="utf-8"))
    assert found == {}, f"{path}: {found}"


# --- the checker itself, against the shape that shipped ---

def test_it_catches_a_renamed_local_that_one_use_kept():
    source = (
        "def main():\n"
        "    candidates = 25\n"
        "    meta = {'candidates': candidates}\n"
        "    other = {'top_k': top_k}\n"
    )
    assert undefined_names(source) == {"main": ["top_k"]}


def test_a_local_in_one_function_does_not_define_another():
    """The permissive version of this check used every binding in the file as
    module scope, which made exactly the shipped bug invisible."""
    source = (
        "def first():\n"
        "    top_k = 20\n"
        "    return top_k\n"
        "def second():\n"
        "    return top_k\n"
    )
    assert undefined_names(source) == {"second": ["top_k"]}


def test_a_closure_reading_an_enclosing_local_is_clean():
    source = (
        "def outer(n):\n"
        "    retrieve_k = n\n"
        "    def inner(x):\n"
        "        return x + retrieve_k\n"
        "    return inner\n"
    )
    assert undefined_names(source) == {}


def test_module_level_names_and_builtins_are_clean():
    source = (
        "import json\n"
        "LIMIT = 25\n"
        "def f():\n"
        "    return len(json.dumps({'a': LIMIT}))\n"
    )
    assert undefined_names(source) == {}
