"""A reading of the state is not a step.

After a step the solvers evaluate the forces again where an output frame or
a progress check falls. That evaluation is of the state the next step starts
from, and whatever it leaves behind -- a member broken, a fracture latched,
a law that has accepted its last strain -- happens a step early when a frame
falls there and on time when none does: the answer depends on
`output.every` and `progress_every`. The fibre beam, the frames and the
truss each had the fault (1.12.1; the truss's rupturing bars v1.13, found
by running every shipped case and every law that keeps a history under four
settings).

So, for EVERY solver family and every kind of history a material keeps: the
same case with a frame after every step, a check after every step, both at
odd intervals, and neither -- and every array the solver holds at the end
must be the same, to the bit. A new solver or a new law belongs in CASES.
"""
import hashlib

import numpy as np
import pytest
import yaml

from vfife.api import run_case

U = {"length": "m", "force": "N", "stress": "Pa"}
CYCLE = [[0, 0], [0.25, 1.0], [0.5, -0.6], [0.75, 1.2], [1.0, 0.0]]
J2 = {"type": "j2", "E": 2.0e11, "nu": 0.3, "rho": 7850.0,
      "hardening": {"source": "linear", "yield_strength": 5.0e7, "H": 2.0e9}}
CDP = {"type": "cdp", "E": 3.0e10, "nu": 0.2, "rho": 2400.0, "l_ref": 0.1, "crack_band": "tension",
       "viscosity": 0.0, "compression": {"source": "gb50010", "fc": 30.0e6},
       "tension": {"source": "fracture_energy", "ft": 3.0e6, "Gf": 40.0}}
ELASTIC = {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2400.0}
STEEL02 = {"type": "steel02", "E": 2.0e11, "Fy": 2.0e8, "b": 0.01, "R0": 20.0, "rho": 7850.0}


def _solid(material, P, backend="numpy", mode="dynamic", bars=None, bond=None, erosion=None, formulation="b-bar"):
    """A cantilever block driven through a load cycle past yield."""
    solver = {"mode": mode, "backend": backend, "damping": {"type": "viscous", "coeff": 20.0}}
    if mode == "dynamic":
        solver["total_time"] = 0.004                    # about 700 steps
    else:
        solver.update({"n_steps": 600, "tolerance": 1e-12, "damping": {"type": "kinetic"}})
    if erosion:
        solver["erosion"] = erosion
    d = {"version": 1, "name": "block", "units": U,
         "mesh": {"dimensions": [1.0, 0.1, 0.1], "divisions": [10, 2, 2]},
         "materials": {"m": material},
         "sections": {"body": {"type": "solid", "material": "m", "formulation": formulation}},
         "assign": [{"region": "all", "section": "body"}],
         "regions": {"x0": {"box": [-1e-3, 1e-3, -1, 1, -1, 1]}, "xL": {"box": [0.999, 1.001, -1, 1, -1, 1]}},
         "constraints": [{"region": "x0", "dofs": [1, 1, 1]}],
         "amplitudes": {"cyc": CYCLE},
         "loads": [{"region": "xL", "vector": [0.3 * abs(P), 0.0, P], "amplitude": "cyc"}],
         "solver": solver, "output": {"fields": ["displacement"]}}
    if bars:
        d["materials"]["steel"] = dict({"type": "rebar", "E": 2.0e11, "rho": 7850.0, "yield_strength": 1.0e8,
                                        "hardening": 2.0e9}, **bars)
        d["sections"]["bars"] = {"type": "bar", "material": "steel", "shape": "circle", "diameter": 0.012}
        d["reinforcement"] = {"long": {"line": {"start": [0.02, 0.03, 0.08], "end": [0.98, 0.03, 0.08],
                                                "segment_length": 0.05,
                                                "repeat": {"count": 2, "offset": [0.0, 0.04, 0.0]}}}}
        d["assign"].append(dict({"section": "bars", "reinforcement": "long", "name": "long"},
                                **({"bond": bond} if bond else {})))
    return d


def _truss(material):
    """Two bars and a mass, shaken through a cycle past yield."""
    return {"version": 1, "name": "truss", "units": U,
            "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "B": [2.0, 0, 0], "C": [1.0, 0, 1.0]},
                     "lines": {"l": {"through": ["A", "C"], "size": 5.0}, "r": {"through": ["B", "C"], "size": 5.0}}},
            "materials": {"m": material},
            "sections": {"bar": {"type": "bar", "material": "m", "shape": "general", "area": 1e-4}},
            "assign": [{"region": "all", "section": "bar"}],
            "nodal_masses": [{"region": "C", "mass": 500.0}],
            "constraints": [{"region": "A", "dofs": [1, 1, 1]}, {"region": "B", "dofs": [1, 1, 1]},
                            {"region": "all", "dofs": [0, 1, 0]}],
            "amplitudes": {"cyc": [[0, 0], [0.25, 1.0], [0.5, -0.8], [0.75, 1.3], [1.0, 0.0]]},
            "loads": [{"region": "C", "vector": [3.0e4, 0, -2.0e4], "amplitude": "cyc"}],
            "solver": {"mode": "dynamic", "backend": "numpy", "total_time": 2.0,
                       "damping": {"type": "viscous", "coeff": 5.0}},
            "output": {"fields": ["displacement"]}}


def _fibre(tmp_path):
    from test_compiled_breaking import _fibre_girder
    d = _fibre_girder(tmp_path, "girder", total=0.02)
    d["loads"][1]["vector"] = [0, 0, -2.0e6]
    return d


def _hinge(tmp_path):
    from test_compiled_breaking import _hinged_cantilever
    return _hinged_cantilever(tmp_path, "hinge", total=0.2)


#: name -> (case, the solver it must come out as, the attribute that shows the history was exercised)
CASES = {
    "solid j2": (lambda t: _solid(J2, -4.0e4), "Solver", "blocks"),
    "solid j2 kinematic, reduced, numba": (
        lambda t: _solid(dict(J2, kinematic=1.0e9), -4.0e4, backend="numba", formulation="reduced"), "Solver", "blocks"),
    "solid cdp, numba": (lambda t: _solid(CDP, -8.0e3, backend="numba"), "Solver", "blocks"),
    "solid cdp with erosion": (
        lambda t: _solid(CDP, -2.0e4, erosion={"damage": 0.9, "strain": 1e-3, "points": 4}), "Solver", "blocks"),
    "solid j2, relaxation": (lambda t: _solid(J2, -4.0e4, mode="relaxation"), "Solver", "blocks"),
    "embedded bars that rupture": (
        lambda t: _solid(ELASTIC, -3.0e4, bars={"rupture_strain": 6e-4}), "Solver", "rebar"),
    "embedded bars that rupture, numba": (
        lambda t: _solid(ELASTIC, -3.0e4, backend="numba", bars={"rupture_strain": 6e-4}), "Solver", "rebar"),
    "bonded bars (bond_cyclic) that rupture": (
        lambda t: _solid(ELASTIC, -3.0e4, bars={"rupture_strain": 6e-4},
                         bond={"type": "bond_cyclic", "fc": 25.6e6}), "Solver", "bonded"),
    "truss steel02": (lambda t: _truss(STEEL02), "TrussSolver", "state"),
    "truss steel02 that ruptures": (lambda t: _truss(dict(STEEL02, rupture_strain=0.004)), "TrussSolver", "state"),
    "truss concrete02": (lambda t: _truss({"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6,
                                           "epscu": 0.0035, "ft": 2.5e6, "Ets": 1.5e9, "rat": 0.1, "rho": 2400}),
                         "TrussSolver", "state"),
    "fibre girder whose bars rupture": (_fibre, "FiberBeamSolver", "_state_s"),
    "frame with a plastic hinge that tears": (_hinge, "MixedStructuralSolver", "_hinge_rotation"),
}
SKIP = {"residuals", "kinetic", "times", "wall_time", "_ke_buf", "_rot_scratch"}


def _arrays(obj, prefix="", depth=0, seen=None):
    """Every numeric array the solver holds, by name: its own, its blocks',
    its reinforcement's, its breaking table's ..."""
    seen = seen if seen is not None else set()
    out = {}
    if id(obj) in seen or depth > 3 or not hasattr(obj, "__dict__"):
        return out
    seen.add(id(obj))
    for k, v in vars(obj).items():
        if k in SKIP:
            continue
        name = prefix + k
        if isinstance(v, np.ndarray):
            if v.dtype.kind in "fiub" and v.size:
                out[name] = v
        elif isinstance(v, (list, tuple)) and 0 < len(v) < 64:
            for i, w in enumerate(v):
                if isinstance(w, np.ndarray) and w.dtype.kind in "fiub" and w.size:
                    out[f"{name}[{i}]"] = w
                elif type(w).__module__.startswith("vfife"):
                    out.update(_arrays(w, f"{name}[{i}].", depth + 1, seen))
        elif type(v).__module__.startswith("vfife") and not callable(v):
            out.update(_arrays(v, name + ".", depth + 1, seen))
    return out


def _end(d, tmp_path, tag, every, progress):
    d = dict(d, output=dict(d["output"], every=every, path=str(tmp_path / f"{tag}.h5")))
    path = tmp_path / f"{tag}.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
    box = {}
    try:
        run_case(str(path), on_ready=lambda s, p: box.setdefault("s", s), progress_every=progress)
    except Exception:                                # a relaxation that runs out of its budget: its state is the test
        pass
    s = box["s"]
    arr = _arrays(s)
    return s, {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arr.items()}, arr


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_end_state_does_not_depend_on_where_the_frames_and_the_checks_fall(tmp_path, name):
    pytest.importorskip("numba")
    make, kind, history = CASES[name]
    d = make(tmp_path)
    ref_solver, ref, arr = _end(d, tmp_path, "none", 10 ** 6, 10 ** 6)
    assert type(ref_solver).__name__ == kind
    assert ref_solver.step >= 200
    # the history the case is here for was exercised: something in it is not zero
    moved = [k for k, v in arr.items() if k.split(".")[0].split("[")[0] == history
             and v.dtype.kind == "f" and np.abs(np.nan_to_num(v)).max() > 0.0]
    assert moved, f"nothing under {history!r} moved: the case does not test what it says"
    for tag, every, progress in (("frames", 1, 10 ** 6), ("checks", 10 ** 6, 1), ("odd", 7, 13)):
        s, got, _ = _end(d, tmp_path, tag, every, progress)
        assert s.step == ref_solver.step
        bad = sorted(k for k in set(got) | set(ref) if got.get(k) != ref.get(k))
        assert not bad, f"{name}: with {tag} these differ from a run with neither: {bad[:12]}"
