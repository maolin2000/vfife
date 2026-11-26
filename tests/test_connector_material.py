"""A connector component carrying a one-dimensional law of the material
table (1.21): `laws: {uy: {type: material, material: <name>, area, length}}`,
F = area sigma(d / length). Each check is a closed form or a law the
connector already has."""
import numpy as np
import pytest

from test_connector_laws import K_V, _bearing_case, _run
from vfife.elements import connector
from vfife.materials import create

STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}
K1, K2, FY = 2.0e7, 2.0e6, 5.0e4
U_Y = FY / K1
CYCLE = [[0, 0], [0.2, 5 * U_Y], [0.4, 0], [0.6, -5 * U_Y], [0.8, 0], [1.0, 5 * U_Y]]


def _cycled(tmp_path, name, law, materials=None, backend="numpy", amplitude=CYCLE, n=50):
    d = _bearing_case(tmp_path, {"uy": law}, [K_V, 0, 0, 0, 0, 0])
    if materials:
        d["materials"].update(materials)
    d["constraints"].append({"region": "block", "dofs": [0, 1, 1, 1, 1, 1]})
    d["amplitudes"] = {"cycle": amplitude}
    d["steps"] = [{"constraints": [{"region": "block", "dofs": [1, 0, 0, 0, 0, 0],
                                    "value": [1.0, 0.0, 0.0], "amplitude": "cycle"}],
                   "load_steps": n, "max_steps_per_load": 20000, "tolerance": 1e-8}]
    d["solver"] = {"backend": backend, "mode": "relaxation", "damping": {"type": "kinetic"}, "tolerance": 1e-8}
    d["output"]["monitors"] = {"u": {"kind": "displacement", "region": "block", "dof": "x"},
                               "H": {"kind": "reaction", "region": "anchor", "dof": "x"}}
    out, p = _run(tmp_path, name, d)
    return np.asarray(out.monitors["u"]), -np.asarray(out.monitors["H"]), out, p


# an elastic-perfectly-plastic part beside an elastic one is the bilinear law
# (k1 = E1 + E2, k2 = E2, fy at the plastic part's yield): the force-deformation
# table itself, area = length = 1
EPP = {"type": "piecewise_uniaxial", "E": K1 - K2, "rho": 1.0, "tension": [[0.0, FY * (K1 - K2) / K1], [1.0, FY * (K1 - K2) / K1]]}
SPRING = {"type": "linear_elastic", "E": K2, "nu": 0.0, "rho": 1.0}


def test_a_parallel_of_the_table_is_the_bilinear_law(tmp_path):
    u_b, F_b, _, _ = _cycled(tmp_path, "bil", {"type": "bilinear", "k1": K1, "k2": K2, "fy": FY})
    mats = {"epp": EPP, "spring": SPRING, "iso": {"type": "parallel", "of": ["epp", "spring"]}}
    u_m, F_m, out, _ = _cycled(tmp_path, "mat", {"type": "material", "material": "iso"}, mats)
    assert np.allclose(u_m, u_b, rtol=0, atol=1e-12)
    assert np.allclose(F_m, F_b, rtol=1e-7, atol=1e-3 * FY)
    assert F_m.max() == pytest.approx(FY + K2 * 4 * U_Y, rel=1e-6)


def test_area_and_length_scale_a_steel_law(tmp_path):
    """A brace's core, 3e-3 m2 and 2.5 m long: F = A sigma(d / L) along the
    whole cycle, against steel02 driven by the same strains."""
    A, L = 3.0e-3, 2.5
    core = {"type": "steel02", "E": 2.0e11, "Fy": 2.35e8, "b": 0.01, "rho": 7850.0}
    amp = 6 * 2.35e8 / 2.0e11 * L
    cyc = [[0, 0], [0.25, amp], [0.5, -amp], [0.75, amp / 2], [1.0, 0]]
    u, F, out, _ = _cycled(tmp_path, "core", {"type": "material", "material": "core", "area": A, "length": L},
                           {"core": core}, amplitude=cyc, n=40)
    m = create("steel02", E=2.0e11, Fy=2.35e8, b=0.01, rho=7850.0)
    st = m.init_state(1); ref = []
    for x in u:
        ref.append(A * m.update(np.array([[x / L, 0, 0, 0, 0, 0]]), st, 0.0)[0, 0])
    assert np.allclose(F, ref, rtol=1e-6, atol=1e-6 * A * 2.35e8)
    assert F.max() > A * 2.35e8


def test_the_reference_stiffness_and_the_compiled_kernel():
    """area E / length in the reference matrix (the time step and the
    relaxation masses read it); the compiled connector kernel declines a
    connector with a `material` law, so the solver runs it in Python."""
    from types import SimpleNamespace
    from vfife.core import families
    law = {"type": "material", "material": dict(STEEL), "area": 2.0, "length": 4.0}
    assert connector.law_stiffness(law) == pytest.approx(2.0 * 2.0e11 / 4.0)
    assert connector.has_material_law({"laws": {"uy": law}})
    assert not connector.has_material_law({"laws": {"uy": {"type": "gap", "hi": 0.1, "stiffness": 1.0}}})
    fam = families.ConnectorMembers.__new__(families.ConnectorMembers)
    fam.ids = np.array([0]); fam._kernel = None; fam._declined = False
    s = SimpleNamespace(backend="numba", dofs=SimpleNamespace(stride=6), member_props=[{"laws": {"uy": law}}])
    assert fam.kernel(s) is None


def test_the_law_s_history_rides_in_the_checkpoint(tmp_path):
    """The committed state of the steel is in the checkpoint and comes back
    on restore: the same state steel02 has after the run's strains."""
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver
    from vfife.io.writer import read_checkpoint
    A, L = 3.0e-3, 2.5
    core = {"type": "steel02", "E": 2.0e11, "Fy": 2.35e8, "b": 0.01, "rho": 7850.0}
    amp = 4 * 2.35e8 / 2.0e11 * L
    u, F, out, p = _cycled(tmp_path, "ck", {"type": "material", "material": "core", "area": A, "length": L},
                           {"core": core}, amplitude=[[0, 0], [0.5, amp], [1.0, amp / 2]], n=6)
    ck = read_checkpoint(str(tmp_path / "ck.h5"))
    assert "connector_material_state" in ck
    prep = prepare_case(str(p))
    s = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)
    s.restore(ck)
    m, committed, _work = s.connector_materials.of(0)[1]
    ref = create("steel02", E=2.0e11, Fy=2.35e8, b=0.01, rho=7850.0); st = ref.init_state(1)
    for x in u:
        ref.update(np.array([[x / L, 0, 0, 0, 0, 0]]), st, 0.0)
    assert np.allclose(committed, st, rtol=1e-6, atol=1e-12)
    with pytest.raises(ValueError, match="connector material state"):
        s.connector_materials.restore(np.zeros(3))


def test_a_fibre_column_on_a_material_bearing(tmp_path):
    """The fibre solver's trial discipline with a `material` law: a stiff
    fibre column on a bearing whose uy is the elastic-plastic table pushed
    past yield; the anchor's horizontal reaction stops at the plateau."""
    from test_connector_laws import _write_msh
    _write_msh(tmp_path / "c.msh", [(0, 0, 0), (0, 0, 1), (0, 0, 3)],
               {"anchor": (0, [0]), "foot": (0, [1]), "top": (0, [2]),
                "bearing": (1, [(0, 1)]), "column": (1, [(1, 2)])})
    fy = FY * (K1 - K2) / K1
    d = {
        "units": {"length": "m", "force": "N"},
        "mesh": {"type": "gmsh", "path": "c.msh"},
        "materials": {"steel": STEEL,
                      "epp": {"type": "piecewise_uniaxial", "E": K1, "rho": 1.0, "tension": [[0.0, fy], [1.0, fy]]}},
        "sections": {"col": {"type": "fiber", "material": "steel", "shape": "rect",
                             "width": 0.3, "height": 0.3, "ny": 4, "nz": 4},
                     "bearing": {"type": "connector", "material": "steel", "stiffness": [K_V, 0, 0, 0, 0, 0],
                                 "laws": {"uy": {"type": "material", "material": "epp"}}}},
        "assign": [{"region": "column", "section": "col", "orientation": [1, 0, 0]},
                   {"region": "bearing", "section": "bearing", "orientation": [1, 0, 0]}],
        "constraints": [{"region": "anchor", "dofs": [1, 1, 1, 1, 1, 1]},
                        {"region": "foot", "dofs": [0, 1, 1, 1, 1, 1]},
                        {"region": "top", "dofs": [0, 1, 1, 1, 1, 1]}],
        "amplitudes": {"drag": [[0.0, 0.0], [1.0, 4 * fy / K1]]},
        "steps": [{"constraints": [{"region": "top", "dofs": [1, 0, 0, 0, 0, 0],
                                    "value": [1.0, 0.0, 0.0], "amplitude": "drag"}],
                   "load_steps": 4, "max_steps_per_load": 40000, "tolerance": 1e-7}],
        "solver": {"backend": "numpy", "mode": "relaxation", "damping": {"type": "kinetic"}, "tolerance": 1e-7},
        "output": {"fields": ["displacement", "connector_slip_uy"], "every": 1,
                   "monitors": {"H": {"kind": "reaction", "region": "anchor", "dof": "x"}}},
    }
    out, _ = _run(tmp_path, "fibcol", d)
    H = np.asarray(out.monitors["H"])
    assert abs(H[0]) < fy * 0.999                       # the first quarter is still elastic
    assert H[-1] == pytest.approx(-fy, rel=1e-4)


def test_a_material_law_is_refused_on_a_three_dimensional_law_and_beside_a_stiffness():
    from vfife.io.schema import ConnectorLawSpec
    with pytest.raises(ValueError, match="needs material"):
        ConnectorLawSpec(type="material")
    with pytest.raises(ValueError, match="does not take"):
        ConnectorLawSpec(type="material", material="m", stiffness=1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        connector.law_material({"material": {"type": "j2", "E": 2e11, "nu": 0.3, "rho": 7850.0,
                                             "hardening": {"source": "linear", "yield_strength": 3e8, "H": 1e9}}})


# ------------------------------------------------- built later, in a frame model

def _prop(tmp_path, backend, law_material, name, steps=None):
    from test_connector_birth import _case, STEPS, KS
    d = _case(tmp_path, STEPS if steps is None else steps, backend, name)
    d["materials"]["prop"] = law_material
    d["sections"]["k"]["stiffness"] = [0.0, 1e3, 1e3, 1e6, 1e6, 1e6]
    d["sections"]["k"]["laws"] = {"ux": {"type": "material", "material": "prop"}}
    return d


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_material_prop_built_later_is_born_free_and_yields(tmp_path, backend):
    """test_connector_birth's prop with its axial spring a `material` law: as
    linear_elastic E = KS the numbers of the spring itself; as an
    elastic-plastic table it carries its yield force and no more. The model
    is a frame (mixed_line): under numba its connectors run in Python."""
    if backend == "numba":
        pytest.importorskip("numba")
    from vfife.api import run_case
    from test_connector_birth import KB, KS
    lin = {"type": "linear_elastic", "E": KS, "nu": 0.0, "rho": 1.0}
    out = run_case(_prop(tmp_path, backend, lin, "lin"))
    assert out.status == "CONVERGED", out.message
    w, R = np.asarray(out.monitors["w"]), np.asarray(out.monitors["R"])
    assert abs(R[19]) < 1e-3 * 2.0e4                                    # born free of force
    assert R[-1] == pytest.approx(2.0e4 * KS / (KB + KS), rel=2e-3)
    fy = 5.0e3                                                          # below its elastic share
    epp = {"type": "piecewise_uniaxial", "E": KS, "rho": 1.0, "tension": [[0.0, fy], [1.0, fy]]}
    out = run_case(_prop(tmp_path, backend, epp, "epp"))
    assert out.status == "CONVERGED", out.message
    R = np.asarray(out.monitors["R"])
    assert R[-1] == pytest.approx(fy, rel=1e-4)                         # compression: the table's other side is the same


def test_a_run_stopped_after_the_material_prop_has_yielded_resumes(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_restart_resumes import _same, _split
    from test_connector_birth import KS, STEPS, _load
    epp = {"type": "piecewise_uniaxial", "E": KS, "rho": 1.0, "tension": [[0.0, 5.0e3], [1.0, 5.0e3]]}
    # stopped after the prop has yielded, then unloaded past zero: a state not
    # brought back would unload from the virgin curve, not from the plateau
    d = _prop(tmp_path, "numpy", epp, "r", STEPS + [{"name": "back", "loads": [_load(-4.0e4)]}])
    d.pop("name"); d["output"].pop("path")
    full, half, rest = _split(tmp_path, d, lambda s, local: s.step_index == 2 and local >= 6, progress_every=1)
    _same(full, rest)


def test_in_a_dynamic_step_the_compiled_step_declines_and_the_answer_is_the_numpy_one(tmp_path):
    """Under numba the compiled connector kernel declines a `material` law
    (it has no code for it); the connectors run in Python and the run gives
    the NumPy run's answer."""
    pytest.importorskip("numba")
    from vfife.api import run_case
    from test_connector_birth import KS
    epp = {"type": "piecewise_uniaxial", "E": KS, "rho": 1.0, "tension": [[0.0, 5.0e3], [1.0, 5.0e3]]}
    got = {}
    for backend in ("numpy", "numba"):
        d = _prop(tmp_path, backend, epp, f"dyn_{backend}")
        d["steps"] = []
        d["loads"] = [{"region": "B", "vector": [0, 0, -3.0e4]}]
        d["constraints"][1]["dofs"] = [1, 1, 1, 1, 1, 1]          # the ground point has no mass of its own
        d["materials"]["steel"]["rho"] = 7850.0
        d["solver"] = {"mode": "dynamic", "backend": backend, "total_time": 0.05,
                       "damping": {"type": "rayleigh", "ratio": 0.05, "frequencies": [5.0, 50.0]}}
        out = run_case(d)
        assert out.converged, out.message
        got[backend] = np.asarray(out.monitors["R"])
    assert np.allclose(got["numba"], got["numpy"], rtol=1e-8, atol=1e-6)
    assert np.max(np.abs(got["numpy"])) == pytest.approx(5.0e3, rel=1e-3)
