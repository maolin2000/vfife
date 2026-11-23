"""Write the 1.18.0 checkpoints for tests/test_restart_across_versions.py.
Run with PYTHONPATH at a 1.18.0 tree, as the tests run (conftest.py):
    OPENBLAS_CORETYPE=Haswell python make_fixtures.py OUTDIR"""
import copy, pathlib, sys, json
import numpy as np, yaml
from vfife import api, __version__
from vfife.core.diagnostics import AnalysisInterrupted
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(pathlib.Path(api.__file__).resolve().parent.parent / "tests"))
from test_member_failure import _write_msh
STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}
def cant(d, n=8, L=2.0, region="bar"):
    _write_msh(d / "c.msh", [(L*i/n, 0, 0) for i in range(n+1)],
               {"root": (0, [0]), "tip": (0, [n]), "bar": (1, [(i, i+1) for i in range(n)])})
    return {"version": 1, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": "c.msh"},
            "materials": {"steel": dict(STEEL)},
            "sections": {"beam": {"type": "beam", "material": "steel", "shape": "rect", "width": 0.1, "height": 0.1}},
            "assign": [{"region": region, "section": "beam", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "root", "dofs": [1]*6}],
            "loads": [{"region": "tip", "vector": [0, 0, -1.0e3]}],
            "output": {"every": 1, "fields": ["displacement"],
                       "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}}}
SC = {
  # region "all": one element type, the frame solver (BeamSolver); region
  # "bar" below is lowered to mixed_line members (MixedStructuralSolver)
  "beam2_dynamic": ({"backend": "numpy", "mode": "dynamic", "total_time": 0.02, "damping": {"type": "none"}},
                    lambda s, local: s.step >= 300, 10),
  "frame_dynamic": ({"backend": "numpy", "mode": "dynamic", "total_time": 0.02, "damping": {"type": "none"}},
                    lambda s, local: s.step >= 300, 10),
  "frame_static": ({"backend": "numpy", "mode": "static", "load_steps": 6, "tolerance": 1e-6},
                   lambda s, level: level >= 3, 1),
}
for name, (sol, stop, every) in SC.items():
    d = out / name; d.mkdir(exist_ok=True)
    c = cant(d, region="all" if name.startswith("beam2") else "bar"); c["solver"] = sol; c["name"] = name
    if sol["mode"] == "dynamic": c["output"]["every"] = 10**6
    c["output"]["path"] = "run.h5"
    p = d / "case.yaml"; p.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    got = {}
    api.run_case(p, output=str(d / "full.h5"), progress_every=every, on_ready=lambda s, _p: got.__setitem__("full", s))
    with __import__("contextlib").suppress(AnalysisInterrupted):
        api.run_case(p, output=str(d / "half.h5"), progress_every=every,
                     on_ready=lambda s, _p: got.__setitem__("half", s),
                     on_progress=lambda local, *_: not stop(got["half"], local))
    ck = got["full"].checkpoint()
    np.savez(d / "full_state.npz", **{k: np.asarray(v) for k, v in ck.items() if np.asarray(v).dtype.kind not in "SUO"})
    (d / "full.h5").unlink()
    for f in d.glob("full*"):
        if f.suffix != ".npz": f.unlink()
    print(name, type(got["full"]).__name__, __version__, "half at", got["half"].step, "full", got["full"].step, sorted(x.name for x in d.iterdir()))
