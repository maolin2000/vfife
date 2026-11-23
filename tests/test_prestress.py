"""Pretensioning: a bar that is already strained when the model is not.

THE MODEL, IN ONE LINE
    `pre.prestrain` is added to the mechanical strain before the constitutive
    law is called, so at u = 0 the bar carries E_p eps_p A and the structure is
    out of balance by exactly the jacking force. Relaxing it IS transfer.

WHY THAT IS THE PRIMITIVE AND NOT A STRESS OFFSET
    Two things fall out of it that an offset added to the bar's residual would
    have got wrong, and both are pinned below:

      * yield is judged on the TOTAL strain. A strand jacked to 0.7 f_py is
        elastic and hardens from there. An offset on the force would have let
        the bar reach f_py PLUS the prestress -- 60% too much tendon capacity.
      * `host_deduction` is taken on the MECHANICAL strain only. The concrete
        the bar displaces is strained by the structure, not by the tensioning
        the bar had before it was cast in.

WHAT THE PROGRAM COMPUTES AND WHAT IT DOES NOT
    It computes the ELASTIC SHORTENING loss, and it computes it by
    equilibrating rather than by applying a formula -- which is why the prism
    below can be checked against the closed form to 1e-5. It computes no other
    loss: friction along a duct, anchor set, relaxation and shrinkage/creep
    are the analyst's, and `prestress` is the stress the tendon HAS after
    them. Post-tensioning (jacking against concrete that already exists) and
    curved tendons are not modelled at all.
"""
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case
from vfife.core.diagnostics import UserError
from vfife.core.solver import Solver
from vfife.elements.truss import internal_force, precompute
from vfife.materials import create
from vfife.materials.points import initialise_points

EC, EP, AP = 30.0e9, 195.0e9, 200.0e-6
FPY = 1670.0e6
STRAND = {"type": "rebar", "E": EP, "rho": 7850.0, "yield_strength": FPY,
          "hardening": 2.0e9}


# --------------------------------------------------------------------------- #
#  The primitive, without a solve
# --------------------------------------------------------------------------- #

def _bar(prestrain, host_deduction=0.0, material=None, L=1.0):
    nodes = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
    conn = np.array([[0, 1]], dtype=np.int32)
    pre = precompute(nodes, conn, AP)
    pre.prestrain = prestrain
    pre.host_deduction = host_deduction
    mat = material or create("rebar", **{k: v for k, v in STRAND.items()
                                         if k != "type"})
    state, _pts = initialise_points(mat, 1, kinematics="uniaxial",
                                    backend="numpy",
                                    char_length=pre.char_len)
    def N(eps_mech):
        u_e = np.zeros((1, 2, 3))
        u_e[0, 1, 0] = eps_mech * L
        f, sig = internal_force(pre, nodes[conn], u_e, mat, state, 1.0,
                                want_stress=True)
        return float(f[0, 1, 0]), float(sig[0, 0, 0])
    return N


def test_a_prestrained_bar_carries_its_jacking_force_at_zero_displacement():
    eps_p = 1000.0e6 / EP
    force, stress = _bar(eps_p)(0.0)
    assert stress == pytest.approx(1000.0e6, rel=1e-12)
    assert force == pytest.approx(1000.0e6 * AP, rel=1e-12)


def test_zero_prestrain_is_the_old_answer_exactly():
    plain, pres = _bar(0.0), _bar(0.0)
    assert plain(1.0e-3) == pres(1.0e-3)
    assert plain(0.0)[0] == 0.0


def test_yield_arrives_early_by_the_prestrain():
    """A strand jacked to 1000 MPa has 670 MPa of elastic range left, not
    1670. Reaching yield at eps_y regardless of the prestress is the mistake
    a force offset would have made, and it is worth 60% of the capacity."""
    eps_p = 1000.0e6 / EP
    N = _bar(eps_p)
    eps_y = FPY / EP
    just_below = N(eps_y - eps_p - 1.0e-5)[1]
    just_above = N(eps_y - eps_p + 2.0e-4)[1]
    assert just_below == pytest.approx(EP * (eps_y - 1.0e-5), rel=1e-9)
    # past yield the tangent is the hardening modulus, not E
    assert just_above < EP * (eps_y + 2.0e-4 - 1e-5)
    assert just_above == pytest.approx(FPY + 2.0e9 * 2.0e-4, rel=1e-3)


def test_host_deduction_is_taken_on_the_mechanical_strain_only():
    """At u = 0 the concrete around the bar carries nothing, so there is
    nothing to deduct -- the bar's force must be the full jacking force.
    Deducting E_c eps_p A would have removed 15% of it, concrete force that
    never existed."""
    eps_p = 1000.0e6 / EP
    force, _ = _bar(eps_p, host_deduction=EC)(0.0)
    assert force == pytest.approx(1000.0e6 * AP, rel=1e-12)
    # and it IS still taken once the bar moves
    moved, _ = _bar(eps_p, host_deduction=EC)(1.0e-4)
    assert moved == pytest.approx((EP * (eps_p + 1e-4) - EC * 1e-4) * AP, rel=1e-9)


# --------------------------------------------------------------------------- #
#  sigma_pe -> eps_p, and what is refused
# --------------------------------------------------------------------------- #

class _NoModulus:
    name = "stub"
    E = 0.0


def test_prestress_is_converted_with_the_bar_modulus():
    mat = create("rebar", E=EP, rho=7850.0, yield_strength=FPY)
    assert Solver._prestrain("t", {"prestress": 1000.0e6}, mat) == \
        pytest.approx(1000.0e6 / EP)


@pytest.mark.parametrize("spec", [{}, {"prestress": 0.0}, {"prestress": None}])
def test_no_prestress_is_no_offset(spec):
    mat = create("rebar", E=EP, rho=7850.0, yield_strength=FPY)
    assert Solver._prestrain("t", spec, mat) == 0.0


def test_prestress_at_or_past_yield_is_refused():
    """eps_p = sigma/E is the right strain only while the bar is elastic, and
    a tendon jacked to yield has no elastic range left to load. Both halves of
    that are modelling errors, so this is an error and not a warning."""
    mat = create("rebar", E=EP, rho=7850.0, yield_strength=FPY)
    with pytest.raises(UserError) as exc:
        Solver._prestrain("tendon", {"prestress": FPY}, mat)
    assert exc.value.code == "PRESTRESS_YIELD"
    assert "1670" in str(exc.value) and "tendon" in str(exc.value)


def test_prestress_on_a_material_with_no_modulus_is_refused():
    with pytest.raises(UserError) as exc:
        Solver._prestrain("t", {"prestress": 1.0e9}, _NoModulus())
    assert exc.value.code == "PRESTRESS_MATERIAL"


def test_compression_is_allowed_and_keeps_its_sign():
    """Not prestressing, but the offset is a general initial strain and a
    negative one is a meaningful thing to write. Refusing it would buy
    nothing."""
    mat = create("rebar", E=EP, rho=7850.0, yield_strength=FPY)
    assert Solver._prestrain("t", {"prestress": -100.0e6}, mat) < 0.0


# --------------------------------------------------------------------------- #
#  The closed form: elastic shortening
# --------------------------------------------------------------------------- #

L, B, H = 0.5, 0.1, 0.1
SIG = 1000.0e6


def _prism(tmp_path, prestress, bond=None, anchorage=None,
           divisions=(10, 2, 2), tol=1.0e-6):
    """A free prism with one central tendon. Nothing is loaded and nothing is
    held but rigid-body motion, so the only thing that happens is transfer."""
    assign = {"section": "tendon", "reinforcement": "t", "name": "t",
              "segment_length": 0.025}
    if prestress:
        assign["prestress"] = prestress
    if bond:
        assign["bond"] = bond
        if anchorage:
            assign["anchorage"] = anchorage
    case = {
        "name": "pt", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [L, B, H], "divisions": list(divisions)},
        "materials": {
            "conc": {"type": "linear_elastic", "E": EC, "nu": 0.2, "rho": 2400.0},
            "strand": dict(STRAND)},
        "sections": {
            "body": {"type": "solid", "material": "conc", "formulation": "b-bar"},
            "tendon": {"type": "bar", "material": "strand", "shape": "circle",
                       "diameter": 0.016, "area": AP}},
        "reinforcement": {"t": {"line": {"start": [0.0, B / 2, H / 2],
                                         "end": [L, B / 2, H / 2]}}},
        "assign": [{"section": "body", "region": "all"}, assign],
        "regions": {
            "left": {"box": [-1e-3, 1e-3, -0.01, B + 0.01, -0.01, H + 0.01]},
            "A": {"box": [-1e-3, 1e-3, B / 2 - 1e-3, B / 2 + 1e-3,
                          H / 2 - 1e-3, H / 2 + 1e-3]},
            "Bn": {"box": [-1e-3, 1e-3, B / 2 - 1e-3, B / 2 + 1e-3,
                           H - 1e-3, H + 1e-3]}},
        "constraints": [{"region": "left", "dofs": [1, 0, 0]},
                        {"region": "A", "dofs": [0, 1, 1]},
                        {"region": "Bn", "dofs": [0, 1, 0]}],
        "steps": [{"name": "transfer", "load_steps": 1,
                   "max_steps_per_load": 60000, "tolerance": tol}],
        "solver": {"backend": "numpy", "mode": "relaxation", "mass": "underwood",
                   "n_steps": 60000, "tolerance": tol,
                   "relaxation_safety": 0.3, "damping": {"type": "kinetic"}},
        "output": {"path": str(tmp_path / "pt.h5"), "every": 5000,
                   "monitors": {"R": {"kind": "reaction", "region": "left",
                                      "dof": "x"}}},
    }
    p = tmp_path / "pt.yaml"
    p.write_text(yaml.safe_dump(case), encoding="utf-8")
    prepared = prepare_case(str(p))
    solver = Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
                    prepared.material, log=lambda *_: None)
    solver.run(on_progress=lambda *_: None, progress_every=1000000)
    return solver


def _section(solver, x):
    m = np.abs(solver.mesh.nodes[:, 0] - x) < 1e-9
    assert m.any()
    return float(solver.u[m, 0].mean())


def _closed_form():
    """Equilibrium of a free composite section:

        E_p A_p (eps_p + eps) + E_c A_g eps = 0

    A_g is the GROSS area: `host_deduction` is off, so the concrete the bar
    displaces is integrated by the solid element as well (elements/truss.py,
    THE OVERLAP), and the closed form has to say the same thing the model does.
    """
    eps_p = SIG / EP
    eps = -EP * AP * eps_p / (EP * AP + EC * B * H)
    return eps_p, eps, EP * (eps_p + eps)


@pytest.mark.slow
def test_a_pretensioned_prism_matches_the_elastic_shortening_closed_form(tmp_path):
    """Measured on a gauge length, NOT on the end face.

    A perfectly bonded tendon dumps its whole force into one host element at
    each end, so the end faces sit in a St Venant zone: u_x across the right
    face of this prism runs -235 to -397 um about a mean of -264, and the last
    two bar segments read 816 and 558 MPa instead of 885. That is a real
    anchorage zone and not an error -- but averaging over it measures the
    disturbance, not the shortening. Between x = 0.1 L and 0.4 L the strain is
    the closed form to nine parts in a million.
    """
    eps_p, eps, sigma = _closed_form()
    s = _prism(tmp_path, SIG)

    gauge = (_section(s, 0.4 * L) - _section(s, 0.1 * L)) / (0.3 * L)
    assert gauge == pytest.approx(eps, rel=2e-4)

    bar = np.asarray(s.rebar[0].stress(s.u)).reshape(s.rebar[0].n_elem, -1)[:, 0]
    assert float(np.median(bar)) == pytest.approx(sigma, rel=2e-4)
    # the loss is real and is about 11.5% here: a 2% tendon ratio in a small
    # section. Asserted loosely, so this stays a statement about the physics
    # rather than a second copy of the formula above.
    assert 0.10 < 1.0 - float(np.median(bar)) / SIG < 0.13

    # nothing pushes on the prism, so no reaction: the state is self-equilibrated
    assert abs(float(np.array(s.monitors["R"].values)[-1])) < 1.0e-6


@pytest.mark.slow
def test_a_bonded_tendon_builds_its_force_over_a_transfer_length(tmp_path):
    """The reason pretensioning wants the bonded element.

    A perfectly bonded bar transfers everything at its end node. A bar with
    its own dofs and an interface law transfers through bond stress, so its
    force grows from the free end inwards over the length the bond needs --
    the transfer length, out of the bond law rather than out of an assumption.

    Measured on this 0.5 m prism, 16 mm strand jacked to 1000 MPa, CEB bond at
    fc = 40 MPa, tendon stress along the bar:

        x (m)  0.013  0.063  0.113  0.163  0.213  0.263 ... 0.488
        MPa     45.0  206.1  343.0  455.1  538.9  566.8       44.6

    THE MEMBER IS SHORTER THAN ITS OWN TRANSFER LENGTH, and that is the
    result, not a defect: the tendon peaks at 567 MPa in the middle and never
    reaches the 885 MPa the perfectly bonded prism carries. A 0.5 m
    pretensioned member really does not develop a 16 mm strand -- fib MC2010
    puts the transfer length near 50 diameters, 0.8 m, twice this specimen.
    A perfectly bonded model would have said 885 MPa everywhere, which is the
    error this element exists to avoid.

    The gradient is the check that ties it to the bond law: dN/dx over the bar
    perimeter IS the bond stress, and it must not exceed tau_max.
    """
    s = _prism(tmp_path, SIG, bond={"type": "bond_ceb", "fc": 40.0e6})
    bt = s.bonded[0]
    _fb, _fh, sig = bt.force(s.u, want_stress=True)
    sig = np.asarray(sig).reshape(bt.n_elem, -1)[:, 0]
    n = bt.n_elem
    x = bt.bar_nodes[bt.bar_conn].mean(axis=1)[:, 0]

    # free at both ends, full in the middle, and symmetric about it
    assert sig[0] < 0.10 * SIG and sig[-1] < 0.10 * SIG
    assert sig[n // 2] > 6.0 * sig[0]
    assert sig[:n // 2] == pytest.approx(sig[::-1][:n // 2], rel=0.02)

    # monotone from each end towards the middle
    assert np.all(np.diff(sig[:n // 2]) > 0.0)
    assert np.all(np.diff(sig[n // 2:]) < 0.0)

    # and the slope is a bond stress, bounded by the law's own peak
    tau = np.abs(np.diff(sig[:n // 2]) * bt.area / np.diff(x[:n // 2])) / bt.perimeter
    tau_max = s.bonded[0].bond.tau_max
    assert tau.max() < 1.05 * tau_max, (
        f"tendon force builds at {tau.max()/1e6:.1f} MPa of bond stress, past "
        f"the law's peak {tau_max/1e6:.1f} MPa")
    assert tau.max() > 0.5 * tau_max, (
        "the bond is barely working; the profile is not a transfer length")


# --------------------------------------------------------------------------- #
#  The closed form: camber
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_an_eccentric_tendon_gives_P_e_L2_over_8EI_of_camber(tmp_path):
    """The prism says the FORCE is right; this says it lands in the right place.

    3.0 x 0.2 x 0.3 m beam on two supports, straight tendon 0.10 m below the
    centroid, nothing else applied. The prestress is a constant moment P e, so
    the camber is P e L^2 / (8 EI) with no shear term at all.

    P is taken from the model's own tendon stress rather than from the jacking
    stress, which keeps this a test of the moment and not a second test of the
    elastic-shortening loss. I is the transformed second moment: the tendon
    sits 0.1 m off the centroid and is worth 2.4% of it.

    CONVERGENCE, because the tolerance below is a discretisation error and it
    is worth saying which one. Through-depth elements against the transformed
    closed form, b-bar hexahedra:

         6 layers, L = 3 m     1.4830 mm    -4.40%
        12 layers, L = 3 m     1.5297 mm    -1.30%
         6 layers, L = 6 m     5.9587 mm    -3.97%

    The error halves and halves again with depth refinement and does not move
    with span, so it is bending stiffness through the depth, not the tendon.
    """
    L, B, H = 3.0, 0.2, 0.3
    e = 0.10
    case = {
        "name": "cam", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [L, B, H], "divisions": [36, 2, 12]},
        "materials": {
            "conc": {"type": "linear_elastic", "E": EC, "nu": 0.2, "rho": 2400.0},
            "strand": dict(STRAND)},
        "sections": {
            "body": {"type": "solid", "material": "conc", "formulation": "b-bar"},
            "tendon": {"type": "bar", "material": "strand", "shape": "circle",
                       "diameter": 0.016, "area": AP}},
        "reinforcement": {"t": {"line": {"start": [0.0, B / 2, H / 2 - e],
                                         "end": [L, B / 2, H / 2 - e]}}},
        "assign": [{"section": "body", "region": "all"},
                   {"section": "tendon", "reinforcement": "t", "name": "t",
                    "segment_length": 0.05, "prestress": SIG}],
        "regions": {
            "sl": {"box": [-1e-3, 1e-3, -0.01, B + 0.01, -1e-3, 1e-3]},
            "sr": {"box": [L - 1e-3, L + 1e-3, -0.01, B + 0.01, -1e-3, 1e-3]},
            "y0": {"box": [-1e-3, L + 1e-3, -1e-3, 1e-3, -1e-3, 1e-3]},
            "mid": {"box": [L / 2 - 1e-3, L / 2 + 1e-3, -0.01, B + 0.01,
                            -1e-3, 1e-3]}},
        "constraints": [{"region": "sl", "dofs": [1, 0, 1]},
                        {"region": "sr", "dofs": [0, 0, 1]},
                        {"region": "y0", "dofs": [0, 1, 0]}],
        "steps": [{"name": "transfer", "load_steps": 1,
                   "max_steps_per_load": 80000, "tolerance": 1.0e-6}],
        "solver": {"backend": "numpy", "mode": "relaxation", "mass": "underwood",
                   "n_steps": 80000, "tolerance": 1.0e-6,
                   "relaxation_safety": 0.3, "damping": {"type": "kinetic"}},
        "output": {"path": str(tmp_path / "cam.h5"), "every": 10000,
                   "monitors": {"uz": {"kind": "displacement", "region": "mid",
                                       "dof": "z"}}},
    }
    path = tmp_path / "cam.yaml"
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    prepared = prepare_case(str(path))
    s = Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
               prepared.material, log=lambda *_: None)
    s.run(on_progress=lambda *_: None, progress_every=1000000)

    emb = s.rebar[0]
    sig = np.asarray(emb.stress(s.u)).reshape(emb.n_elem, -1)[:, 0]
    P = float(np.median(sig)) * AP
    It = B * H ** 3 / 12 + (EP / EC - 1.0) * AP * e ** 2
    expected = P * e * L ** 2 / (8.0 * EC * It)

    camber = float(np.array(s.monitors["uz"].values)[-1])
    assert camber > 0.0, "the beam sagged: the prestress moment has the wrong sign"
    assert camber == pytest.approx(expected, rel=0.03)


# --------------------------------------------------------------------------- #
#  The input model
# --------------------------------------------------------------------------- #

def test_the_input_model_lowers_prestress_onto_the_reinforcement_entry(tmp_path):
    from vfife.io.model import InputCase
    case = {
        "materials": {"c": {"type": "linear_elastic", "E": EC, "nu": 0.2,
                            "rho": 2400.0}, "s": dict(STRAND)},
        "sections": {"b": {"type": "solid", "material": "c"},
                     "t": {"type": "bar", "material": "s", "shape": "circle",
                           "diameter": 0.016, "area": AP}},
        "reinforcement": {"t": {"line": {"start": [0.0, 0.05, 0.05],
                                         "end": [0.5, 0.05, 0.05]}}},
        "assign": [{"section": "b", "region": "all"},
                   {"section": "t", "reinforcement": "t",
                    "segment_length": 0.025, "prestress": 900.0e6}],
        "mesh": {"dimensions": [0.5, 0.1, 0.1], "divisions": [4, 2, 2]},
    }
    lowered = InputCase(**case).lower()
    assert lowered["reinforcement"][0]["prestress"] == 900.0e6


# --------------------------------------------------------------------------- #
#  Preflight says it back as a force, and says what it leaves out
# --------------------------------------------------------------------------- #

def test_preflight_reports_the_prestress_force_and_the_losses_it_omits():
    """The number an analyst checks is a force per tendon and a total, not a
    stress -- one multiplication that catches an area in mm2, a stress in MPa
    and a tendon count that is not what was drawn, before a two-hour run.

    The second line exists because "prestress" carries losses with it in every
    design code, and this solver produces exactly one of them.
    """
    from pathlib import Path
    from vfife.cli import build
    from vfife.io import schema
    from vfife.preflight import check

    path = Path(__file__).resolve().parents[1] / "cases" / "case_13_pretensioned_beam.yaml"
    case, _raw = schema.load(path)
    mesh, _material, element = build(case, str(path))
    report = check(case, mesh, element)
    issues = {i["code"]: i for i in report["issues"]}

    assert report["status"] == "PASS"
    # four strands of 150 mm2 at 1000 MPa: 150 kN each, 600 kN in total, and
    # the `repeat: {count: 4}` has to be counted or the total is a quarter of it
    assert issues["PRESTRESS"]["details"]["lines"] == 4
    assert issues["PRESTRESS"]["details"]["force"] == pytest.approx(150.0e3)
    assert issues["PRESTRESS_LOSSES"]["details"]["total"] == pytest.approx(600.0e3)
    assert "friction" in issues["PRESTRESS_LOSSES"]["message"]


def test_preflight_is_silent_when_nothing_is_prestressed():
    from pathlib import Path
    from vfife.cli import build
    from vfife.io import schema
    from vfife.preflight import check

    path = Path(__file__).resolve().parents[1] / "cases" / "case_12_ualberta_sf0b.yaml"
    case, _raw = schema.load(path)
    mesh, _material, element = build(case, str(path))
    codes = {i["code"] for i in check(case, mesh, element)["issues"]}
    assert "PRESTRESS" not in codes and "PRESTRESS_LOSSES" not in codes
