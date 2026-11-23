"""Response-spectrum analysis.

A two-mass chain along x -- ground, spring k1, mass m1, spring k2, mass
m2 -- whose modes, participation and CQC/SRSS peaks are a 2 x 2 eigenproblem
done here with NumPy, independently of the library.
"""
import h5py
import numpy as np
import pytest

from test_member_failure import _run, _write_msh

E, A = 2.0e11, 1.0e-3
L1, L2 = 2.0, 1.0
M1, M2 = 2.0e4, 1.0e4
TABLE = [[0.0, 3.0], [0.1, 6.0], [0.5, 6.0], [2.0, 1.5], [5.0, 0.6]]


def _chain(tmp_path, name, rs, mode="relaxation"):
    _write_msh(tmp_path / f"{name}.msh", [(0, 0, 0), (L1, 0, 0), (L1 + L2, 0, 0)],
               {"base": (0, [0]), "m1": (0, [1]), "m2": (0, [2]), "free": (0, [1, 2]),
                "s1": (1, [(0, 1)]), "s2": (1, [(1, 2)])})
    light = {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 1.0e-3}
    beam = {"type": "beam", "material": "light", "shape": "general", "area": A, "iy": 1e-6, "iz": 1e-6, "j": 2e-6}
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": f"{name}.msh"},
        "materials": {"light": light},
        "sections": {"b": beam},
        "assign": [{"region": "s1", "section": "b", "orientation": [0, 0, 1]},
                   {"region": "s2", "section": "b", "orientation": [0, 0, 1]}],
        "nodal_masses": [{"region": "m1", "mass": M1}, {"region": "m2", "mass": M2}],
        "constraints": [{"region": "base", "dofs": [1, 1, 1, 1, 1, 1]},
                        {"region": "free", "dofs": [0, 1, 1, 1, 1, 1]}],
        "loads": [],
        "solver": {"backend": "numpy", "mode": mode, "n_steps": 50, "tolerance": 1e-8},
        "output": {"fields": ["displacement"], "every": 10, "monitors": {}},
        "response_spectrum": rs,
    }
    if mode == "dynamic":
        d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 0.01}
    out, h5 = _run(tmp_path, name, d)
    with h5py.File(h5) as f:
        g = f["response_spectrum"]
        return {k: np.asarray(g[k]) for k in g}, dict(g.attrs)


def _closed_form(zeta, rule):
    k1, k2 = E * A / L1, E * A / L2
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    M = np.diag([M1, M2])
    w2, phi = np.linalg.eigh(np.diag(M.diagonal() ** -0.5) @ K @ np.diag(M.diagonal() ** -0.5))
    phi = np.diag(M.diagonal() ** -0.5) @ phi                     # mass-normalised
    w = np.sqrt(w2)
    T = 2 * np.pi / w
    t = np.asarray(TABLE)
    Sa = np.interp(T, t[:, 0], t[:, 1])
    gam = phi.T @ M @ np.ones(2)
    U = phi * (gam * Sa / w2)[None, :]                             # (dof, mode)
    base = k1 * U[0]                                               # base shear per mode
    s2 = k2 * (U[1] - U[0])                                        # spring 2 force per mode
    if rule == "cqc":
        r = w[1] / w[0]
        rho12 = 8 * zeta ** 2 * (1 + r) * r ** 1.5 / ((1 - r * r) ** 2 + 4 * zeta ** 2 * r * (1 + r) ** 2)
    else:
        rho12 = 0.0
    rho = np.array([[1, rho12], [rho12, 1]])
    comb = lambda v: float(np.sqrt(v @ rho @ v))
    return {"T": T, "u1": comb(U[0]), "u2": comb(U[1]), "base": comb(base), "s2": comb(s2),
            "eff": gam ** 2 / (M1 + M2)}


@pytest.mark.parametrize("rule", ["cqc", "srss"])
def test_a_two_mass_chain_matches_its_eigenproblem(tmp_path, rule):
    zeta = 0.05
    rs = {"spectra": [{"direction": [1, 0, 0], "table": TABLE}], "combination": rule, "damping": zeta,
          "modes": 2}
    out, attrs = _chain(tmp_path, f"chain_{rule}", rs)
    ref = _closed_form(zeta, rule)
    assert attrs["n_modes"] == 2
    assert out["period"] == pytest.approx(ref["T"], rel=1e-6)
    assert out["effective_mass"][:, 0] == pytest.approx(ref["eff"], rel=1e-6)
    assert out["displacement"][1, 0] == pytest.approx(ref["u1"], rel=1e-5)
    assert out["displacement"][2, 0] == pytest.approx(ref["u2"], rel=1e-5)
    # the base force is combined from the modal SUMS, and it equals the
    # first spring's force, member end action by member end action
    assert out["base_force"][0] == pytest.approx(ref["base"], rel=1e-5)
    assert out["reaction"][0, 0] == pytest.approx(ref["base"], rel=1e-5)
    assert out["axial_force"][0] == pytest.approx(ref["base"], rel=1e-5)
    assert out["axial_force"][1] == pytest.approx(ref["s2"], rel=1e-5)
    # nothing moves across the direction it is shaken in
    assert np.abs(out["displacement"][:, 1:]).max() < 1e-12


def test_left_to_itself_it_keeps_the_modes_that_reach_the_mass_fraction(tmp_path):
    # mode 1 of this chain carries 99.3% of the mass: 0.9 needs one mode,
    # 0.999 needs both
    eff = _closed_form(0.05, "srss")["eff"]
    assert eff[0] > 0.99
    a, attrs_a = _chain(tmp_path, "auto", {"spectra": [{"direction": [1, 0, 0], "table": TABLE}]})
    b, attrs_b = _chain(tmp_path, "all", {"spectra": [{"direction": [1, 0, 0], "table": TABLE}],
                                          "mass_fraction": 0.999})
    assert attrs_a["n_modes"] == 1 and attrs_b["n_modes"] == 2
    assert a["mass_reached"][0] == pytest.approx(eff[0], rel=1e-6)


def test_two_directions_combine_by_srss_and_a_scale_multiplies(tmp_path):
    # the same spectrum in x and, scaled by two, along a direction that is
    # x again: the chain only moves in x, so the peaks are sqrt(1 + 4) times one
    one = {"spectra": [{"direction": [1, 0, 0], "table": TABLE}]}
    two = {"spectra": [{"direction": [1, 0, 0], "table": TABLE},
                       {"direction": [2, 0, 0], "table": TABLE, "scale": 2.0}]}
    a, _ = _chain(tmp_path, "one", one)
    b, _ = _chain(tmp_path, "two", two)
    assert b["displacement"][2, 0] == pytest.approx(np.sqrt(5.0) * a["displacement"][2, 0], rel=1e-9)


def test_a_period_outside_the_table_is_refused_unless_held(tmp_path):
    short = [[0.0, 3.0], [0.02, 4.0]]
    with pytest.raises(ValueError, match="outside its table"):
        _chain(tmp_path, "short", {"spectra": [{"direction": [1, 0, 0], "table": short}]})
    out, _ = _chain(tmp_path, "held", {"spectra": [{"direction": [1, 0, 0], "table": short}],
                                        "beyond_table": "hold"})
    assert np.all(out["spectral_acceleration"] == 4.0)


def test_the_spectrum_is_refused_where_it_would_mean_nothing():
    from vfife.io.schema import ResponseSpectrumSpec, SpectrumSpec
    with pytest.raises(ValueError, match="table` OR a `path"):
        SpectrumSpec(direction=[1, 0, 0])
    with pytest.raises(ValueError, match="increase"):
        SpectrumSpec(direction=[1, 0, 0], table=[[0.5, 1.0], [0.1, 2.0]])
    with pytest.raises(ValueError, match="cannot be zero"):
        SpectrumSpec(direction=[0, 0, 0], table=TABLE)
    assert ResponseSpectrumSpec(spectra=[{"direction": [0, 1, 0], "table": TABLE}]).combination == "cqc"


def test_a_dynamic_run_is_refused(tmp_path):
    with pytest.raises(Exception, match="relaxation"):
        _chain(tmp_path, "dyn", {"spectra": [{"direction": [1, 0, 0], "table": TABLE}]}, mode="dynamic")


def test_cqc_correlation_is_one_on_the_diagonal_and_falls_with_spacing():
    from vfife.spectrum import cqc_rho
    rho = cqc_rho(np.array([1.0, 1.05, 2.0]), 0.05)
    assert np.allclose(np.diag(rho), 1.0) and np.allclose(rho, rho.T)
    assert rho[0, 1] > 0.5 > rho[0, 2]


def test_case_35_meets_its_contract(tmp_path):
    """The shipped case against its closed-form contract, through the same
    `source: spectrum` reader `vfife verify` uses."""
    from pathlib import Path
    from vfife.api import run_case
    from vfife.verification import verify
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "case_35.h5"
    run_case(str(root / "cases" / "case_35_spectrum_cantilever.yaml"), output=str(out))
    report = verify(root / "verification" / "case_35_spectrum_cantilever.yaml", out)
    assert report["status"] == "PASS", [c for c in report["checks"] if c["status"] != "PASS"]
