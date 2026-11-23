"""The fibre section's cracking shear law.

Against closed forms: an elastic law is the Timoshenko cantilever,
PL^3/3EI + PL/GA, to the digit; a law that cracks at a stated shear adds
L (gamma_cr + (P - V_cr)/GA_cr); unloading once cracked runs to the origin;
the truss model gives Park & Paulay's GA_cr; flexural cracking triggers it.
"""
import h5py
import numpy as np
import pytest

from test_member_failure import _run, _write_msh

E = 2.0e11
STEEL = {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 7850.0}
B, H, L, N_EL = 0.2, 0.4, 2.0, 4
I_Y = B * H ** 3 / 12 * (1 - 1 / 10 ** 2)     # ten fibres through the depth (midpoint rule)


def _cantilever(tmp_path, name, shear, P, steps=None, material=STEEL, bars=None, backend="numpy",
                extra_section=None, extra_sections=None, extra_materials=None):
    nodes = [(L * i / N_EL, 0, 0) for i in range(N_EL + 1)]
    _write_msh(tmp_path / f"{name}.msh", nodes,
               {"root": (0, [0]), "tip": (0, [N_EL]), "beam": (1, [(i, i + 1) for i in range(N_EL)])})
    sec = {"type": "fiber", "material": "m", "shape": "rect", "width": B, "height": H,
           "ny": 1, "nz": 10, "integration": "lobatto3"}
    if shear is not None:
        sec["shear"] = shear
    if bars:
        sec["bars"] = bars
    sec.update(extra_section or {})
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": f"{name}.msh"},
        "materials": {"m": material, "s": STEEL, **(extra_materials or {})},
        "sections": {"f": sec, "bar": {"type": "bar", "material": "s", "shape": "general", "area": 3e-4},
                     **(extra_sections or {})},
        "assign": [{"region": "beam", "section": "f", "orientation": [0, 1, 0]}],
        "constraints": [{"region": "root", "dofs": [1] * 6},
                        {"region": "all", "dofs": [0, 1, 0, 1, 0, 1]}],
        "loads": [] if steps else [{"region": "tip", "vector": [0, 0, -P]}],
        # staged: each load level held until equilibrium
        "solver": {"backend": backend, "mode": "relaxation", "n_steps": 100, "tolerance": 1e-9,
                   "load_steps": 1, "max_steps_per_load": 200000},
        "output": {"fields": ["displacement"], "every": 1,
                   "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}},
    }
    if steps:
        d["steps"] = steps
    out, h5 = _run(tmp_path, name, d)
    with h5py.File(h5) as f:
        w = np.asarray(f["monitors/w"])
    return w, out


def test_an_elastic_law_is_the_timoshenko_cantilever(tmp_path):
    GA = 2.0e8
    P = 1.0e4
    w, _ = _cantilever(tmp_path, "el", {"stiffness": GA, "cracked_stiffness": 1.0e8, "cracking": 1.0e12}, P)
    exact = P * L ** 3 / (3 * E * I_Y) + P * L / GA
    assert -w[-1] == pytest.approx(exact, rel=2e-4)
    # the shear is a third of it here: the law is doing the work
    assert P * L / GA > 0.3 * exact


def test_a_law_that_cracks_at_a_stated_shear_adds_the_cracked_flexibility(tmp_path):
    GA0, GAcr, Vcr = 2.0e9, 1.0e8, 5.0e3
    P = 1.5e4
    w, _ = _cantilever(tmp_path, "cr", {"stiffness": GA0, "cracked_stiffness": GAcr, "cracking": Vcr}, P)
    gamma = Vcr / GA0 + (P - Vcr) / GAcr
    exact = P * L ** 3 / (3 * E * I_Y) + L * gamma
    assert -w[-1] == pytest.approx(exact, rel=2e-4)


def test_unloading_once_cracked_runs_to_the_origin(tmp_path):
    """Loaded to 15 kN past a 5 kN crack, then back to 6 kN: the shear
    unloads on the secant through the largest excursion, not the uncracked
    stiffness, and not the cracked envelope."""
    GA0, GAcr, Vcr = 2.0e9, 1.0e8, 5.0e3
    P1, P2 = 1.5e4, 6.0e3
    steps = [{"name": "up", "loads": [{"region": "tip", "vector": [0, 0, -P1]}], "load_steps": 2},
             {"name": "down", "loads": [{"region": "tip", "vector": [0, 0, -P2]}], "load_steps": 2}]
    w, _ = _cantilever(tmp_path, "un", {"stiffness": GA0, "cracked_stiffness": GAcr, "cracking": Vcr},
                       None, steps=steps)
    g1 = Vcr / GA0 + (P1 - Vcr) / GAcr
    secant = P1 / g1
    exact = P2 * L ** 3 / (3 * E * I_Y) + L * P2 / secant
    assert -w[-1] == pytest.approx(exact, rel=2e-4)


def test_the_truss_model_is_park_and_paulay(tmp_path):
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver
    st = {"area": 1.57e-4, "spacing": 0.1, "depth": 0.36, "E": 2.0e11}
    _cantilever(tmp_path, "pp", {"stiffness": 2.0e9, "stirrups": st}, 1.0e4)
    prep = prepare_case(str(tmp_path / "pp.yaml"))
    s = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)
    rho = 1.57e-4 / (B * 0.1)
    n = 2.0e11 / E
    assert s._sl_GAcr[0] == pytest.approx(rho / (1 + 4 * n * rho) * 2.0e11 * B * 0.36, rel=1e-12)


def test_flexural_cracking_is_what_cracks_the_shear(tmp_path):
    """A concrete cantilever with bars: under a load that cracks its root in
    flexure the shear follows the cracked stiffness from the shear it
    carried at that moment; under one that does not, it stays uncracked."""
    conc = {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6, "epscu": 0.005,
            "ft": 2.0e6, "Ets": 1.0e9, "rat": 0.1, "rho": 2400}
    bars = [{"y": 0.05, "z": z, "section": "bar"} for z in (0.15, -0.15)] + \
           [{"y": -0.05, "z": z, "section": "bar"} for z in (0.15, -0.15)]
    law = {"stiffness": 1.0e9, "cracked_stiffness": 5.0e7}
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver
    from vfife.io.writer import read_checkpoint
    # first crack moment ~ f_t I / (H/2) = 2e6 x 1.07e-3 / 0.2 = 10.7 kN m, at P ~ 5.3 kN
    for name, P, cracks in (("lo", 2.0e3, False), ("hi", 2.0e4, True)):
        _cantilever(tmp_path, name, law, P, material=conc, bars=bars)
        prep = prepare_case(str(tmp_path / f"{name}.yaml"))
        s = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                                   prep.mesh, prep.material)
        s.restore(read_checkpoint(str(tmp_path / f"{name}.h5")))
        root = 0
        assert np.isfinite(s._sl_gcr_c[root]) == cracks
        if cracks:
            g = s._sl_gamma[root]
            gcr = s._sl_gcr_c[root]
            assert abs(g) * 1.0e9 < P                       # softer than uncracked ...
            assert 1.0e9 * gcr + 5.0e7 * (abs(g) - gcr) == pytest.approx(P, rel=1e-3)   # ... on the envelope
        else:
            assert abs(s._sl_gamma[root]) * 1.0e9 == pytest.approx(P, rel=1e-3)


def test_what_is_refused(tmp_path):
    from vfife.io.model import FiberShearSpec
    with pytest.raises(ValueError, match="cracked_stiffness` OR `stirrups"):
        FiberShearSpec(stiffness=1e9)
    with pytest.raises(Exception, match="replaces"):
        _cantilever(tmp_path, "both", {"stiffness": 1e9, "cracked_stiffness": 1e8}, 1.0,
                    extra_section={"shear_stiffness": 1e9})
    with pytest.raises(Exception, match="not below"):
        _cantilever(tmp_path, "up", {"stiffness": 1e9, "cracked_stiffness": 2e9}, 1.0)


CONC = {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6, "epscu": 0.005,
        "ft": 2.0e6, "Ets": 1.0e9, "rat": 0.1, "rho": 2400}
BARS = [{"y": 0.05, "z": z, "section": "bar"} for z in (0.15, -0.15)] + \
       [{"y": -0.05, "z": z, "section": "bar"} for z in (0.15, -0.15)]


@pytest.mark.parametrize("P", [2.0e3, 2.0e4])
def test_the_fused_kernel_carries_the_law_as_the_reference_does(tmp_path, P):
    """The cracking shear law is in the fused numba kernel since 1.7
    (it ran on the NumPy reference alone before). The same
    cantilever on both backends, uncracked and cracked: the tip, the shear
    rotation and the crack strain agree to round-off."""
    pytest.importorskip("numba")
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver
    from vfife.io.writer import read_checkpoint
    law = {"stiffness": 1.0e9, "cracked_stiffness": 5.0e7}
    bars = [dict(b, section="sbar") for b in BARS]
    got = {}
    for backend in ("numpy", "numba"):
        name = f"k_{backend}_{int(P)}"
        w, out = _cantilever(tmp_path, name, law, P, material=CONC, bars=bars, backend=backend,
                             extra_materials={"s2": {"type": "steel02", "E": E, "Fy": 4.0e8, "b": 0.01,
                                                     "R0": 20.0, "rho": 7850.0}},
                             extra_sections={"sbar": {"type": "bar", "material": "s2", "shape": "general",
                                                      "area": 3e-4}})
        prep = prepare_case(str(tmp_path / f"{name}.yaml"))
        s = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                                   prep.mesh, prep.material)
        s.restore(read_checkpoint(str(tmp_path / f"{name}.h5")))
        got[backend] = (w[-1], s._sl_gamma.copy(), s._sl_gcr_c.copy(), s._sl_gmax_c.copy(), out.status)
        if backend == "numba":
            assert s._kernel is not None, s.kernel_status
    a, b = got["numpy"], got["numba"]
    assert a[4] == b[4] == "CONVERGED"
    assert b[0] == pytest.approx(a[0], rel=1e-7)
    assert np.allclose(b[1], a[1], rtol=1e-6, atol=1e-15)
    assert np.array_equal(np.isfinite(a[2]), np.isfinite(b[2]))
    assert np.allclose(b[2][np.isfinite(a[2])], a[2][np.isfinite(a[2])], rtol=1e-6)
    assert np.allclose(b[3], a[3], rtol=1e-6, atol=1e-15)
    assert np.isfinite(a[2][0]) == (P > 1.0e4)
