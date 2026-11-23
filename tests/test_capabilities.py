"""The table of what runs where (vfife/core/capabilities.py):
every reason the compiled step and the compiled relaxation give comes from
it, every row of it is used, and the manuals hold its current rendering."""
import ast
import os
import subprocess
import sys

import pytest

from vfife.core import capabilities

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: the functions that decide a refusal, by file
REFUSERS = {"vfife/core/compiled_step.py": ("_components_refusal", "refusal", "_arrays_refusal", "_fiber_refusal",
                                            "relax_refusal", "_arrays_refusal_relax"),
            "vfife/core/fiber_beam_solver.py": ("fused_kernel_refusal",)}


def _functions(rel, names):
    tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read())
    found = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names}
    assert set(found) == set(names), (rel, set(names) - set(found))
    return found.values()


def _is_refuse(node):
    f = node.func if isinstance(node, ast.Call) else None
    return (isinstance(f, ast.Name) and f.id == "refuse") or (isinstance(f, ast.Attribute) and f.attr == "refuse")


def _has_text(node):
    """A string written here -- a literal, an f-string, one put together or
    formatted -- anywhere in the expression but inside a `refuse(...)` call
    (its key, and the values it fills in)."""
    if isinstance(node, ast.Call) and _is_refuse(node):
        return False
    # a name looked up, or a value compared with -- not words for a user
    if isinstance(node, ast.Compare):
        return False
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr"):
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        # words, not a name (an attribute, a key, a method in a list)
        return " " in node.value
    if isinstance(node, ast.JoinedStr):
        return True
    return any(_has_text(n) for n in ast.iter_child_nodes(node))


def test_every_refusal_is_worded_by_the_table():
    loose = []
    for rel, names in REFUSERS.items():
        for fn in _functions(rel, names):
            for node in ast.walk(fn):
                values = []
                if isinstance(node, ast.Return) and node.value is not None:
                    values.append(node.value)
                # a reason kept in a variable first, under any name
                if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and node.value is not None:
                    values.append(node.value)
                for v in values:
                    if _has_text(v):
                        loose.append(f"{rel}:{node.lineno} in {fn.name}")
    assert not loose, "a reason written in place, not in vfife/core/capabilities.py: " + ", ".join(loose)


def _keys_used():
    """The keys of every `refuse(...)` call in vfife/ (code, not comments)."""
    used = set()
    for root, _dirs, files in os.walk(os.path.join(ROOT, "vfife")):
        for f in files:
            if not f.endswith(".py") or f == "capabilities.py":
                continue
            tree = ast.parse(open(os.path.join(root, f), encoding="utf-8").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _is_refuse(node) and node.args:
                    key = node.args[0]
                    for k in ([key.body, key.orelse] if isinstance(key, ast.IfExp) else [key]):
                        assert isinstance(k, ast.Constant), f"{f}:{node.lineno}: a key that is not written out"
                        used.add(k.value)
    return used


def test_every_key_the_code_asks_for_is_in_the_table_and_every_row_is_asked_for():
    used = _keys_used()
    assert used <= set(capabilities.BY_KEY), used - set(capabilities.BY_KEY)
    assert set(capabilities.BY_KEY) <= used, set(capabilities.BY_KEY) - used


def test_a_row_the_manuals_list_has_both_languages():
    for lim in capabilities.LIMITS:
        assert (lim.feature is None) == (lim.feature_zh is None), lim.key


def test_the_words_did_not_change():
    # the reasons are read by tests and by users' scripts: moving them into
    # the table kept them as they were (1.17)
    assert capabilities.refuse("off") == "VFIFE_COMPILED_STEP is off"
    assert capabilities.refuse("hooked", what="rigid contact") == "the step is hooked (rigid contact)"
    assert capabilities.refuse("own_forces", kind="ShellSolver") == "ShellSolver has its own member forces"
    assert capabilities.refuse("arrays", name="q", shape=(6,)) == "q is not a contiguous float array of shape (6,)"
    assert capabilities.refuse("short_fibre", left=10).startswith("a short fibre run (10 steps)")


def test_the_manuals_tables_are_current():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "capability_tables.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr + "\n(python tools/capability_tables.py rewrites them)"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_table_has_a_column_for_every_path(lang):
    rows = capabilities.table(lang).splitlines()
    assert all(r.count("|") == len(capabilities.PATHS) + 2 for r in rows)
