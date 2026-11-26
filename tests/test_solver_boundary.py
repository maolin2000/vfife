"""The boundary between the solvers and the code that drives them, held as a
ratchet.

The architecture review of 1.17 found the drivers -- the compiled step, the
analysis steps, the implicit solver, restarts -- reading and writing the
solvers' private attributes (`solver._name`), features attached by replacing
a solver's methods on the instance, and 238 probes for optional state
(`getattr(x, "_name", None)`). The refactoring that removes them (P0 of the
review) goes step by step; this test holds every step:

* no NEW private access, method replacement or probe may appear;

What it reads: every .py under vfife/ and tests/ (an untracked file there
counts too: keep scratch elsewhere); a solver is a name `s`, `solver`, `sol`
or an attribute `.solver` -- access through other names (`self.line` in a
composed solver, a benchmark's own variable) is not seen; benchmarks/ and
tools/ are not read.
* the methods the frame and the fibre-beam solver each define for
  themselves under the same name (P1 of the review, v1.18: the
  two copies of one solver) may not grow either;
* one that goes away must also go from the baseline, so the baseline only
  ever shrinks and the numbers in it are the progress.

After a change that removes some, write the new baseline with

    python tests/test_solver_boundary.py --update

and commit `tests/data/solver_boundary.json` with the change. A change that
adds one is the thing this test is for: give the solver a public method or
an explicit field instead.
"""
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASELINE = os.path.join(HERE, "data", "solver_boundary.json")

#: the names a solver goes by outside its own class
SOLVER_NAMES = {"s", "solver", "sol"}
#: the files that define the solver classes; inside them `self._x` is the
#: class's own business and is not counted
SOLVER_FILES = ("beam_solver.py", "fiber_beam_solver.py", "mixed_structural_solver.py", "solver.py",
                "truss_solver.py", "shell_solver.py", "shell_line_solver.py", "coupled_solver.py",
                "connector_solver.py", "orientation.py", "beam_family.py",
                # the fibre solver's mixins (1.22): its sections, fracture and time
                "fiber_sections.py", "fiber_fracture.py", "fiber_time.py")
#: the fibre solver's class and the mixins it is made of
FIBRE_CLASSES = (("fiber_beam_solver.py", "FiberBeamSolver"), ("fiber_sections.py", "FiberSections"),
                 ("fiber_fracture.py", "FiberFracture"), ("fiber_time.py", "FiberTimeDependence"))


def _files(top):
    for root, dirs, files in os.walk(os.path.join(ROOT, top)):
        dirs[:] = [d for d in dirs if d != "__pycache__" and d != "data"]
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(root, f)


def _rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def _solver_methods():
    """Every method a solver class defines: replacing one of these on an
    instance is attaching a feature by patching."""
    names = set()
    for f in SOLVER_FILES:
        tree = ast.parse(open(os.path.join(ROOT, "vfife", "core", f), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                names.update(m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return names


def _is_solver(node):
    if isinstance(node, ast.Name):
        return node.id in SOLVER_NAMES
    # `self.solver._x`, `outcome.solver._x`
    return isinstance(node, ast.Attribute) and node.attr == "solver"


def _private(name):
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def measure():
    methods = _solver_methods()
    access, patched, probes, tests = {}, {}, {}, {}
    for top in ("vfife", "tests"):
        for path in _files(top):
            tree = ast.parse(open(path, encoding="utf-8").read())
            rel = _rel(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and _is_solver(node.value):
                    # a method replaced on a solver, public ones too (the
                    # contacts replaced `nodal_fields`, `checkpoint`, `restore`)
                    if top == "vfife" and isinstance(node.ctx, ast.Store) and node.attr in methods:
                        patched.setdefault(rel, set()).add(node.attr)
                    if not _private(node.attr):
                        continue
                    if top == "tests":
                        tests.setdefault(rel, set()).add(node.attr)
                        continue
                    access.setdefault(rel, set()).add(node.attr)
                elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) >= 2
                      and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
                    name = node.args[1].value
                    if top == "vfife" and node.func.id == "setattr" and _is_solver(node.args[0]) and name in methods:
                        patched.setdefault(rel, set()).add(name)
                    if top == "vfife" and node.func.id in ("getattr", "hasattr") and _private(name):
                        probes[rel] = probes.get(rel, 0) + 1
    sort = lambda d: {k: sorted(v) for k, v in sorted(d.items())}
    return {"private_access": sort(access), "patched_methods": sort(patched),
            "optional_probes": dict(sorted(probes.items())), "tests_private_access": sort(tests),
            "twin_methods": {"BeamSolver/FiberBeamSolver": sorted(_twins())}}


def _class_methods(fname, cls):
    tree = ast.parse(open(os.path.join(ROOT, "vfife", "core", fname), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
    raise LookupError(f"{cls} not in {fname}")


def _twins():
    """The methods both solvers define in their own class body: a shared one
    lives in core/beam_family and is in neither."""
    fibre = set().union(*(_class_methods(f, c) for f, c in FIBRE_CLASSES))
    return _class_methods("beam_solver.py", "BeamSolver") & fibre


def _totals(m):
    return {"private_access": sum(len(v) for v in m["private_access"].values()),
            "patched_methods": sum(len(v) for v in m["patched_methods"].values()),
            "optional_probes": sum(m["optional_probes"].values()),
            "tests_private_access": sum(len(v) for v in m["tests_private_access"].values()),
            "twin_methods": sum(len(v) for v in m["twin_methods"].values())}


def _diff(now, base, key):
    new, gone = [], []
    for f in sorted(set(now[key]) | set(base[key])):
        a, b = set(now[key].get(f, ())), set(base[key].get(f, ()))
        new += [f"{f}: {x}" for x in sorted(a - b)]
        gone += [f"{f}: {x}" for x in sorted(b - a)]
    return new, gone


def _check(key):
    now, base = measure(), json.load(open(BASELINE, encoding="utf-8"))
    new, gone = _diff(now, base, key)
    assert not new, (f"{len(new)} new {key.replace('_', ' ')} -- give the solver a public method or an "
                     "explicit field instead:\n  " + "\n  ".join(new[:40]))
    assert not gone, (f"{len(gone)} {key.replace('_', ' ')} no longer made: write the smaller baseline "
                      "(python tests/test_solver_boundary.py --update):\n  " + "\n  ".join(gone[:40]))


def test_no_new_private_access_to_a_solver():
    _check("private_access")


def test_no_new_feature_attached_by_replacing_a_solver_method():
    _check("patched_methods")


def test_no_new_tests_on_a_solvers_private_state():
    _check("tests_private_access")


def test_no_new_method_in_both_beam_solvers():
    """A method the frame and the fibre-beam solver both need goes into
    core/beam_family once, not into each."""
    _check("twin_methods")


def test_the_optional_state_probes_only_go_down():
    """`getattr(x, "_name", None)` / `hasattr(x, "_name")`, counted per file
    (on the solver or anything else: the habit is the same)."""
    now, base = measure()["optional_probes"], json.load(open(BASELINE, encoding="utf-8"))["optional_probes"]
    more = {f: n - base.get(f, 0) for f, n in now.items() if n > base.get(f, 0)}
    assert not more, ("new getattr/hasattr probe(s) for a private name -- give the state an explicit field "
                      f"with a default instead: {more}")
    fewer = {f: base[f] - now.get(f, 0) for f in base if now.get(f, 0) < base[f]}
    assert not fewer, f"fewer probes than the baseline, write it (--update): {fewer}"


if __name__ == "__main__":
    m = measure()
    if "--update" in sys.argv:
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1, ensure_ascii=False, sort_keys=True)
            f.write("\n")
    print(json.dumps(_totals(m)))
