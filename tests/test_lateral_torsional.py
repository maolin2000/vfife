"""Lateral-torsional and torsional buckling in `vfife buckling` (1.5).

K_G now carries the moments' coupling of twist with bending out of their
plane, and the Wagner term N Ip/A theta'^2 (vfife/buckling.py). Checked
against the classical results for a doubly symmetric member with
Saint-Venant torsion and no warping, which is what a vfife member is:

* a simply supported beam on forks under a uniform moment:
  M_cr = pi/L sqrt(E Iz G J), and its second mode at twice that;
* the same beam under a central point load and under a uniform load at the
  centroid, and a cantilever with an end load: P_cr = gamma sqrt(E Iz G J)
  / L^2 with gamma = 16.94, 28.31, 4.013 -- each computed here from the
  governing equation theta'' + M(x)^2 / (E Iz G J) theta = 0 by shooting,
  independent of vfife;
* a column on forks whose torsional load G J A / Ip is below its Euler
  load: it twists there;
* a beam-column, the uniform moment with an axial thrust:
  M^2 = (Ip/A) (P_E - P) (P_T - P);
* the same answers from a mixed line model, from a beam along global y,
  and from a fibre section;
* Argyris' right-angle frame and the cantilever against the nonlinear
  model itself: its co-rotational tangent, at the static equilibrium under
  the load, turns singular within 1-2% of each factor;
* the 1.1 results (flexural buckling) unchanged: their tests stand as they
  were.
"""
import numpy as np
import pytest
import yaml
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from vfife import buckling
from vfife.api import prepare_case
from vfife.cli import _with_element, make_solver

E, NU = 2.0e11, 0.3
G = E / (2.0 * (1.0 + NU))
B, H = 0.02, 0.4                                   # a narrow rectangle, deep along local z
A = B * H
IY, IZ = B * H ** 3 / 12.0, H * B ** 3 / 12.0
J = H * B ** 3 / 3.0 * (1.0 - 0.63 * B / H)
L = 6.0
ROOT = np.sqrt(E * IZ * G * J)
#: Kirchhoff's factor for the in-plane bending before the member buckles:
#: the critical moment of the linear theory over sqrt((1 - Iz/Iy)(1 - GJ/EIy))
KIRCHHOFF = 1.0 / np.sqrt((1.0 - IZ / IY) * (1.0 - G * J / (E * IY)))
FORK = [{"region": "A", "dofs": [1, 1, 1, 1, 0, 0]}, {"region": "B", "dofs": [0, 1, 1, 1, 0, 0]}]


def _gamma(m, end, lo, hi):
    """gamma of theta'' + (gamma m(xi))^2 theta = 0 on [0, 1], theta(0) = 0
    and theta(1) = 0 (end "theta") or theta'(1) = 0 (end "slope")."""
    def miss(g):
        s = solve_ivp(lambda x, y: [y[1], -(g * m(x)) ** 2 * y[0]], (0.0, 1.0), [0.0, 1.0],
                      rtol=1e-12, atol=1e-14)
        return s.y[0, -1] if end == "theta" else s.y[1, -1]
    return brentq(miss, lo, hi, xtol=1e-12)


def _case(tmp_path, name, loads, constraints, n=32, iz=IZ, j=J, axis="x", extra=None, fibre=False):
    if axis == "x":
        pts = {"A": [0, 0, 0], "M": [L / 2, 0, 0], "B": [L, 0, 0]}
        orient = [0, 1, 0]                            # local z = global z
    else:
        pts = {"A": [0, 0, 0], "M": [0, L / 2, 0], "B": [0, L, 0]}
        orient = [1, 0, 0]                            # local z = -global z
    if fibre:
        sec = {"type": "fiber", "material": "s", "shape": "rect", "width": B, "height": H,
               "ny": 2, "nz": 40, "torsion_constant": j}
    else:
        sec = {"type": "beam", "material": "s", "shape": "general", "area": A, "iy": IY, "iz": iz, "j": j}
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": pts, "lines": {"g": {"through": ["A", "M", "B"], "size": L / n}}},
         "materials": {"s": {"type": "linear_elastic", "E": E, "nu": NU, "rho": 7850}},
         "sections": {"c": sec},
         "assign": [{"region": "g", "section": "c", "orientation": orient}],
         "constraints": [dict(c) for c in constraints], "loads": [dict(x) for x in loads],
         "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 1000},
         "output": {"fields": ["displacement"]}}
    if extra:
        extra(d)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    prep = prepare_case(str(p))
    return make_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)


def _first(s, n_modes=1):
    return buckling.analyse(s, n_modes=n_modes).factors


def test_a_beam_on_forks_under_a_uniform_moment(tmp_path):
    M = 1.0e3
    loads = [{"region": "A", "moment": [0, M, 0]}, {"region": "B", "moment": [0, -M, 0]}]
    mcr = np.pi / L * ROOT * KIRCHHOFF            # exact with the pre-buckling bending
    for n, tol in ((16, 2e-3), (32, 5e-4)):
        lam = _first(_case(tmp_path, f"um{n}", loads, FORK, n=n), n_modes=2)
        assert lam[0] * M / mcr == pytest.approx(1.0, abs=tol)
        assert lam[1] / lam[0] == pytest.approx(2.0, rel=5e-3)        # the second half-wave, n M_cr


@pytest.mark.parametrize("what, gamma", [
    ("central", _gamma(lambda x: 0.5 * min(x, 1.0 - x), "theta", 10, 25)),
    ("uniform", _gamma(lambda x: 0.5 * x * (1.0 - x), "theta", 20, 40)),
    ("cantilever", _gamma(lambda x: 1.0 - x, "slope", 2, 6)),
])
def test_a_beam_under_transverse_loads_at_its_centroid(tmp_path, what, gamma):
    P = 1.0e3
    if what == "central":
        loads, cons, total = [{"region": "M", "vector": [0, 0, -P]}], FORK, P
    elif what == "uniform":
        loads, cons, total = [{"region": "g", "line_load": [0, 0, -P]}], FORK, P * L
    else:
        loads, cons, total = [{"region": "B", "vector": [0, 0, -P]}], [{"region": "A", "dofs": [1] * 6}], P
    assert gamma == pytest.approx({"central": 16.94, "uniform": 28.31, "cantilever": 4.013}[what], abs=6e-3)
    lam = _first(_case(tmp_path, what, loads, cons))
    assert lam[0] * total * L ** 2 / ROOT == pytest.approx(gamma * KIRCHHOFF, rel=1.5e-3)


def test_a_column_twists_at_its_torsional_load(tmp_path):
    j = 1.0e-6                                       # G J A / Ip below the Euler load
    s = _case(tmp_path, "tc", [{"region": "B", "vector": [-1.0e3, 0, 0]}], FORK, iz=IY, j=j)
    pt = G * j * A / (2 * IY)
    assert pt < np.pi ** 2 * E * IY / L ** 2
    r = buckling.analyse(s, n_modes=2)
    assert r.factors[0] * 1.0e3 / pt == pytest.approx(1.0, rel=2e-4)
    assert r.rotational[0]                           # a twist, no translation


def test_a_beam_column_under_moment_and_thrust(tmp_path):
    P, M = 1.0e3, 1.0e3
    loads = [{"region": "A", "moment": [0, M, 0]}, {"region": "B", "moment": [0, -M, 0]},
             {"region": "B", "vector": [-P, 0, 0]}]
    rp2, pe, pt = (IY + IZ) / A, np.pi ** 2 * E * IZ / L ** 2, G * J * A / (IY + IZ)
    Me = M / KIRCHHOFF                                # the moment's share, as K_G scales it
    a, b, c = Me ** 2 - rp2 * P ** 2, rp2 * P * (pe + pt), -rp2 * pe * pt
    lam = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    assert 0.3 < lam * P / pe < 0.95                  # the thrust matters, and so does the moment
    assert _first(_case(tmp_path, "bc", loads, FORK))[0] / lam == pytest.approx(1.0, abs=5e-4)


def test_the_same_from_a_mixed_model_a_beam_along_y_and_a_fibre_section(tmp_path):
    P = 1.0e3
    gamma = _gamma(lambda x: 0.5 * min(x, 1.0 - x), "theta", 10, 25)
    ref = _first(_case(tmp_path, "ref", [{"region": "M", "vector": [0, 0, -P]}], FORK))[0]
    assert ref * P * L ** 2 / ROOT == pytest.approx(gamma * KIRCHHOFF, rel=1.5e-3)

    def tie(d):                                      # a bar along x at B: the model is mixed
        d["mesh"]["points"]["C"] = [L + 1.0, 0, 0]
        d["mesh"]["lines"]["t"] = {"through": ["B", "C"], "size": 1.0}
        d["sections"]["bar"] = {"type": "bar", "material": "s", "shape": "general", "area": 1e-4}
        d["assign"].append({"region": "t", "section": "bar"})
        d["constraints"].append({"region": "C", "dofs": [1] * 6})
    s = _case(tmp_path, "mixed", [{"region": "M", "vector": [0, 0, -P]}], FORK, extra=tie)
    assert type(s).__name__ == "MixedStructuralSolver"
    assert _first(s)[0] == pytest.approx(ref, rel=1e-6)

    fork_y = [{"region": "A", "dofs": [1, 1, 1, 0, 1, 0]}, {"region": "B", "dofs": [1, 0, 1, 0, 1, 0]}]
    assert _first(_case(tmp_path, "y", [{"region": "M", "vector": [0, 0, -P]}], fork_y, axis="y"))[0] \
        == pytest.approx(ref, rel=1e-6)

    s = _case(tmp_path, "fib", [{"region": "M", "vector": [0, 0, -P]}], FORK, fibre=True)
    assert type(s).__name__ == "FiberBeamSolver"
    # the fibres' Iz is that of 2 x 40 cells (b^3 h / 48, not / 12): the
    # closed form with the section's own Iz and Ip
    iy, iz = float(s.Iy[0]), float(s.Iz[0])
    root = np.sqrt(E * iz * G * J) / np.sqrt((1.0 - iz / iy) * (1.0 - G * J / (E * iy)))
    assert _first(s)[0] * P * L ** 2 / root == pytest.approx(gamma, rel=3e-3)


# ------------------------------------------- against the nonlinear model itself
def _tangent_min_at(tmp_path, d, lam):
    """The smallest eigenvalue of the co-rotational model's own tangent at
    its static equilibrium under lam times the loads (`mode: static`)."""
    from vfife.api import run_case
    from vfife.testing import invariants
    from vfife.testing.state import probe as make_probe
    d = yaml.safe_load(yaml.safe_dump(d))
    for x in d["loads"]:
        x["vector"] = [float(lam * c) for c in x["vector"]]
    d["solver"] = {"backend": "numpy", "mode": "static", "tolerance": 3e-6, "load_steps": 4}
    d["output"] = {"path": str(tmp_path / f"nl{lam:.4f}.h5"), "fields": ["displacement"], "every": 4}
    box = {}
    out = run_case(d, on_ready=lambda s, _p: box.setdefault("s", s))
    assert out.converged, out.message
    p = make_probe(box["s"])
    K, _ = invariants.tangent(p, p.get(), h=1e-7 * L, external=True)
    free = np.flatnonzero(~p.fixed)
    return float(np.linalg.eigvalsh(0.5 * (K + K.T)[np.ix_(free, free)])[0])


def _argyris(tmp_path, name):
    """Argyris' right-angle frame: two members in the x-y plane, clamped at
    A, deep in that plane, pushed at the tip along -x -- the load bends both
    about their strong axes and compresses the first; the frame buckles out
    of its plane, the moment turned round the corner from one member to the
    other."""
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "C": [L, 0, 0], "D": [L, L, 0]},
                  "lines": {"g1": {"through": ["A", "C"], "size": L / 16},
                            "g2": {"through": ["C", "D"], "size": L / 16}}},
         "materials": {"s": {"type": "linear_elastic", "E": E, "nu": NU, "rho": 7850}},
         "sections": {"c": {"type": "beam", "material": "s", "shape": "general", "area": A,
                            "iy": IY, "iz": IZ, "j": J}},
         "assign": [{"region": "g1", "section": "c", "orientation": [0, 0, -1]},
                    {"region": "g2", "section": "c", "orientation": [0, 0, 1]}],
         "constraints": [{"region": "A", "dofs": [1] * 6}],
         "loads": [{"region": "D", "vector": [-1.0e3, 0, 0]}],
         "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 1000},
         "output": {"fields": ["displacement"]}}
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    prep = prepare_case(str(p))
    return d, make_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)


def test_argyris_frame_buckles_where_the_nonlinear_model_does(tmp_path):
    """No closed form (it depends on how the corner turns the moment); the
    reference is the solver's own: its co-rotational tangent, at the static
    equilibrium under the load, turns singular within 1% of each factor --
    pushed (+), and pulled (-) where the first member is in tension. The
    continuum terms alone (the 1.5 draft) were 4.5% high and 7.7% low."""
    d, s = _argyris(tmp_path, "arg")
    r = buckling.analyse(s, n_modes=1)
    for lam in (float(r.factors[0]), float(r.first_negative)):
        assert _tangent_min_at(tmp_path, d, 0.99 * lam) > 0.0 > _tangent_min_at(tmp_path, d, 1.01 * lam), lam
    assert r.first_negative == pytest.approx(-1.596 * r.factors[0], rel=2e-3)


def test_a_cantilever_buckles_where_the_nonlinear_model_does(tmp_path):
    P = 1.0e3
    s = _case(tmp_path, "cnl", [{"region": "B", "vector": [0, 0, -P]}], [{"region": "A", "dofs": [1] * 6}], n=16)
    lam = float(_first(s)[0])
    d = yaml.safe_load((tmp_path / "cnl.yaml").read_text(encoding="utf-8"))
    # the pre-buckling deflection raises it by about (1 - Iz/Iy)^-1/2 (1 - GJ/EIy)^-1/2 -- 0.3%
    assert _tangent_min_at(tmp_path, d, 0.99 * lam) > 0.0 > _tangent_min_at(tmp_path, d, 1.02 * lam)


def test_a_stockier_section_buckles_where_the_nonlinear_model_does(tmp_path):
    """Iz = Iy / 4: the bending before buckling raises the factor by 17%
    over the linear theory (Kirchhoff's factor), and that is where the
    nonlinear model's tangent turns singular. Without the factor (the 1.5
    draft) it was 1.05 to 1.3 times the linear one; at Iz = Iy / 2, 1.43."""
    s = _case(tmp_path, "k25", [{"region": "B", "vector": [0, 0, -1.0e3]}], [{"region": "A", "dofs": [1] * 6}],
              n=16, iz=0.25 * IY, j=0.05 * IY)
    lam = float(_first(s)[0])
    d = yaml.safe_load((tmp_path / "k25.yaml").read_text(encoding="utf-8"))
    assert _tangent_min_at(tmp_path, d, 0.99 * lam) > 0.0 > _tangent_min_at(tmp_path, d, 1.05 * lam)


def test_a_square_section_and_a_beam_bent_about_its_minor_axis_do_not_buckle_laterally(tmp_path):
    """Iz = Iy, or the moment about the minor axis: no lateral-torsional
    mode exists (Kirchhoff's factor is 0); the nonlinear model's tangent
    stiffens. The 1.5 draft found one -- the Bathe 45-degree bend
    (case_18, a square) at 0.86 of the load it carries."""
    P = 1.0e3
    cant = [{"region": "A", "dofs": [1] * 6}]
    for name, iz in (("sq", IY), ("minor", 4.0 * IY)):
        r = buckling.analyse(_case(tmp_path, name, [{"region": "B", "vector": [0, 0, -P]}], cant, iz=iz, j=0.5 * IY),
                             n_modes=1)
        assert not len(r.factors) and "no positive factor" in r.note, (name, r.factors)


def test_kirchhoff_s_factor_is_zero_about_the_minor_axis_even_when_gj_is_larger(tmp_path):
    """The narrow rectangle has G J > E Iz: about its minor axis both
    brackets are negative, and their product made a factor of 14 that took
    an inclined beam's critical moment 8% below the nonlinear model's (at
    30 degrees; 36% at 60). Each bracket is clamped now. A uniform moment
    about an axis at 30 degrees to the major one, on forks: the minor
    plane's share drops out, and the major one's buckles the beam at
    pi/L sqrt(E Iz G J) times Kirchhoff's factor (the nonlinear model's
    tangent turns singular 0.8% above it, in the review)."""
    s = _case(tmp_path, "kf", [{"region": "B", "vector": [0, 0, -1.0]}], FORK)
    cy, cz = buckling._kirchhoff(s, 0)
    assert cy == pytest.approx(1.0 / KIRCHHOFF, rel=1e-12) and cz == 0.0
    b, M = np.radians(30.0), 1.0e3
    mvec = [0.0, float(M * np.cos(b)), float(M * np.sin(b))]
    loads = [{"region": "A", "moment": mvec}, {"region": "B", "moment": [-m for m in mvec]}]
    lam = _first(_case(tmp_path, "incl", loads, FORK))[0]
    assert lam * M * np.cos(b) == pytest.approx(np.pi / L * ROOT * KIRCHHOFF, rel=1e-3)


def test_a_fine_mesh_of_a_narrow_section_is_not_a_mechanism(tmp_path):
    """The cantilever on 64 elements (and an uneven mesh with 0.05 m elements
    beside 0.4 m ones) was refused as a mechanism: its weakest stiffness is
    3e-11 of its largest, under the relative threshold. A mechanism is now a
    stiffness below the finite difference's own noise (the tangent's
    asymmetry), and a pinned column is still refused (test_v11_buckling)."""
    P = 1.0e3
    cant = [{"region": "A", "dofs": [1] * 6}]
    ref = _first(_case(tmp_path, "m32", [{"region": "B", "vector": [0, 0, -P]}], cant, n=32))[0]
    assert _first(_case(tmp_path, "m64", [{"region": "B", "vector": [0, 0, -P]}], cant, n=64))[0] \
        == pytest.approx(ref, rel=1e-3)

    def uneven(d):
        d["mesh"]["lines"] = {"g1": {"through": ["A", "M"], "size": 0.05}, "g2": {"through": ["M", "B"], "size": 0.4}}
        d["mesh"]["points"]["M"] = [0.2, 0, 0]
        d["assign"] = [{"region": "g1", "section": "c", "orientation": [0, 1, 0]},
                       {"region": "g2", "section": "c", "orientation": [0, 1, 0]}]
    assert _first(_case(tmp_path, "mu", [{"region": "B", "vector": [0, 0, -P]}], cant, extra=uneven))[0] \
        == pytest.approx(ref, rel=2e-3)


def test_a_no_tension_fibre_column_is_told_to_ask_from_its_loaded_state(tmp_path):
    """Concrete01 has no tension: at the reference configuration every fibre
    sits on the kink between E and 0, K_b averages them, and the column
    comes out near half its Euler load. The note says so and where to ask
    instead (review of 1.5); at the loaded state the section carries its
    compression stiffness (and the lateral load's cracking)."""
    Lc = 10.0
    d = {"version": 1, "name": "nt", "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "B": [0, 0, Lc]},
                  "lines": {"g": {"through": ["A", "B"], "size": Lc / 10}}},
         "materials": {"c": {"type": "concrete01", "fpc": 30e6, "epsc0": 0.002, "fpcu": 6e6, "epscu": 0.0035,
                             "rho": 2400}},
         "sections": {"s": {"type": "fiber", "material": "c", "shape": "rect", "width": 0.4, "height": 0.4,
                            "ny": 8, "nz": 8, "torsion_constant": 3.6e-3}},
         "assign": [{"region": "g", "section": "s", "orientation": [1, 0, 0]}],
         "constraints": [{"region": "A", "dofs": [1] * 6}],
         "loads": [{"region": "B", "vector": [0, 0, -1.0e5]}],
         "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 1000},
         "output": {"fields": ["displacement"]}}
    p = tmp_path / "nt.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    prep = prepare_case(str(p))
    s = make_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)
    Ec = 2.0 * 30e6 / 0.002
    euler = np.pi ** 2 * Ec * 0.4 ** 4 / 12 / (4 * Lc ** 2) / 1.0e5
    r = buckling.analyse(s, n_modes=1)
    assert "no-tension" in r.note and "--result" in r.note
    assert r.factors[0] == pytest.approx(0.5 * euler, rel=0.03)
    # from the loaded state (static run, pure compression): the full stiffness
    from vfife.api import run_case
    d["solver"] = {"backend": "numpy", "mode": "static", "tolerance": 1e-8}
    d["output"]["path"] = str(tmp_path / "nt.h5")
    box = {}
    out = run_case(d, on_ready=lambda sv, _p: box.setdefault("s", sv))
    assert out.converged
    s2 = box["s"]
    r2 = buckling.analyse(s2, n_modes=1, state=s2.q.copy())
    assert "no-tension" not in r2.note
    # lambda is on the load ADDED to the state: total = (1 + lambda) P
    assert (1.0 + r2.factors[0]) == pytest.approx(euler, rel=0.03)


# A SECTION SET OFF ITS NODES ACROSS THE MEMBER (end_offsets, 1.21) under a
# thrust at the nodes: the thrust is eccentric by e, and the column is a
# beam-column under the uniform moment P e. Its lever's own term in K_G,
# (d theta x r) x N, is what the explicit end moments do not have (1.23:
# the reference solution the review of 1.21 asked for)
E_OFF = 0.3


def _set_off(d):
    d["assign"][0]["end_offsets"] = {"i": [0, 0, E_OFF], "j": [0, 0, E_OFF]}


def test_a_column_set_off_its_nodes_on_forks_is_the_beam_column_of_its_eccentricity(tmp_path):
    P = 1.0e3
    rp2, pe, pt = (IY + IZ) / A, np.pi ** 2 * E * IZ / L ** 2, G * J * A / (IY + IZ)
    Me = P * E_OFF / KIRCHHOFF
    a, b, c = Me ** 2 - rp2 * P ** 2, rp2 * P * (pe + pt), -rp2 * pe * pt
    lam = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    s = _case(tmp_path, "off_fork", [{"region": "B", "vector": [-P, 0, 0]}], FORK, extra=_set_off)
    assert _first(s)[0] / lam == pytest.approx(1.0, abs=5e-4)


def test_a_column_set_off_its_nodes_buckles_where_the_nonlinear_model_does(tmp_path):
    """One end free to twist: the lever turns with the node, and the factor
    is no longer the fixed end moments' (0.8% below it here). The model's own
    co-rotational tangent at its static equilibrium turns singular within
    +-0.5% of the factor."""
    P = 1.0e3
    free = [{"region": "A", "dofs": [1, 1, 1, 1, 0, 0]}, {"region": "B", "dofs": [0, 1, 1, 0, 0, 0]}]
    box = {}

    def grab(d):
        _set_off(d)
        box["d"] = yaml.safe_load(yaml.safe_dump(d))
    lam = _first(_case(tmp_path, "off_free", [{"region": "B", "vector": [-P, 0, 0]}], free, extra=grab))[0]
    moments = [{"region": "A", "moment": [0, -P * E_OFF, 0]}, {"region": "B", "moment": [0, P * E_OFF, 0]},
               {"region": "B", "vector": [-P, 0, 0]}]
    fixed = _first(_case(tmp_path, "mom_free", moments, free))[0]
    assert lam / fixed - 1.0 > 5e-3
    assert _tangent_min_at(tmp_path, box["d"], 0.995 * lam) > 0.0
    assert _tangent_min_at(tmp_path, box["d"], 1.005 * lam) < 0.0
