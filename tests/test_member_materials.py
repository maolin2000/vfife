"""A member group with its own material.

Until this stage `mixed_line` carried ONE material for every member, and
`io/model.py` refused the second one by name. That refusal is what made a
cable-stayed or hybrid cable-supported bridge -- steel girder, concrete
tower, two different cable steels -- impossible to write down at all.
The refusal was at least loud, and the workaround was
exact for what it could buy: scale every member's `area`/`iy`/`iz`/`j` by
E_i/E_ref so the products EA and EI come out right, then top the mass back
up with `nodal_masses`. That is the arithmetic this file pins, on both
sides:

  * the real two-material model and the E-scaled one agree to the BIT on
    stiffness and on translational mass -- so nothing that used the
    workaround was wrong, and the workaround is now unnecessary;
  * they DISAGREE on rotary inertia, because `nodal_masses` adds translation
    only. That difference is the thing this stage actually bought, and it is
    asserted here rather than argued.

The other half of the contract is that a model with ONE material is what it
always was: same object on every member, no `material` key written by the
input model, same numbers.
"""
import pathlib

import numpy as np
import pytest
import scipy.sparse.linalg as sla

from vfife.api import prepare_case
from vfife.core.structural import make_structural_solver
from vfife.io import model as input_model


E1, R1, A1, I1 = 2.10e11, 7850.0, 0.01, 1.0e-4        # a steel girder
E2, R2, A2, I2 = 3.60e10, 2650.0, 0.50, 5.0e-2        # a C60 tower
N, L, P = 8, 0.5, 1.0e4
K = E2 / E1                                            # the E-scaling factor


def _mesh(td):
    d = pathlib.Path(td); d.mkdir(parents=True, exist_ok=True)
    msh = d / "m.msh"
    msh.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n"
        f"{N+1}\n" + "\n".join(f"{i+1} {i*L} 0 0" for i in range(N + 1))
        + "\n$EndNodes\n$Elements\n" + f"{N}\n"
        + "\n".join(f"{i+1} 1 2 1 1 {i+1} {i+2}" for i in range(N))
        + "\n$EndElements\n", encoding="utf-8")
    return msh


def _case(td, materials, sections, assigns, extra="", alpha=""):
    msh = _mesh(td)
    p = pathlib.Path(td) / "c.yaml"
    p.write_text(f"""
name: mm
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
{materials}
sections:
{sections}
assign:
{assigns}
regions:
  ra:   {{box: [-0.01, {4*L+0.01}, -0.01, 0.01, -0.01, 0.01]}}
  rb:   {{box: [{4*L-0.01}, {8*L+0.01}, -0.01, 0.01, -0.01, 0.01]}}
  root: {{box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}}
  tip:  {{box: [{8*L-0.01}, {8*L+0.01}, -0.01, 0.01, -0.01, 0.01]}}
{extra}constraints: [{{region: root, dofs: [1, 1, 1, 1, 1, 1]}}]
loads: [{{region: tip, vector: [0, 0, {-P}]}}]
steps: [{{name: push, load_steps: 1}}]
solver: {{backend: numpy, mode: relaxation, tolerance: 1.0e-7, n_steps: 2000}}
output:
  path: {pathlib.Path(td)/'o.h5'}
  monitors: {{w: {{kind: displacement, region: tip, dof: z}}}}
""", encoding="utf-8")
    return str(p)


def _solver(path):
    prep = prepare_case(path)
    return make_structural_solver(prep.case.model_dump(mode="python"),
                                  prep.mesh, prep.material)


def _tip(sol):
    """Tip deflection from the DECLARED tangent -- this file is about what the
    model IS, not about how fast it relaxes."""
    F = np.zeros(sol.K.shape[0]); F[6*N + 2] = -P
    free = np.ones(sol.K.shape[0], bool); free[:6] = False
    u = np.zeros_like(F)
    u[free] = sla.spsolve(sol.K.tocsr()[free][:, free], F[free])
    return u[6*N + 2]


TWO_MATS = (f"  steel: {{type: linear_elastic, E: {E1}, nu: 0.3, rho: {R1}}}\n"
            f"  c60:   {{type: linear_elastic, E: {E2}, nu: 0.2, rho: {R2}}}")
ONE_MAT = f"  steel: {{type: linear_elastic, E: {E1}, nu: 0.3, rho: {R1}}}"
SPLIT = ("  - {region: ra, section: a, orientation: [0, 0, 1]}\n"
         "  - {region: rb, section: b, orientation: [0, 0, 1]}")


def _real(td):
    return _case(td, TWO_MATS,
                 f"  a: {{type: beam, material: steel, shape: general, area: {A1}, "
                 f"iy: {I1}, iz: {I1}, j: {2*I1}}}\n"
                 f"  b: {{type: beam, material: c60, shape: general, area: {A2}, "
                 f"iy: {I2}, iz: {I2}, j: {2*I2}}}", SPLIT)


def _scaled(td):
    lump = (R2*A2 - R1*A2*K) * 4*L
    return _case(td, ONE_MAT,
                 f"  a: {{type: beam, material: steel, shape: general, area: {A1}, "
                 f"iy: {I1}, iz: {I1}, j: {2*I1}}}\n"
                 f"  b: {{type: beam, material: steel, shape: general, area: {A2*K}, "
                 f"iy: {I2*K}, iz: {I2*K}, j: {2*I2*K}}}", SPLIT,
                 extra=f"nodal_masses: [{{region: rb, mass: {lump}}}]\n")


# --------------------------------------------------------------------------- #
#  a single-material model is what it always was
# --------------------------------------------------------------------------- #
def test_one_material_writes_no_material_key_and_shares_one_object(tmp_path):
    """The whole point of leaving `material` off a group that does not need
    one: `self.material` is still THE object every member reads, so a case
    from before this stage cannot have moved."""
    path = _case(tmp_path, ONE_MAT,
                 f"  a: {{type: beam, material: steel, shape: general, area: {A1}, "
                 f"iy: {I1}, iz: {I1}, j: {2*I1}}}",
                 "  - {region: ra, section: a, orientation: [0, 0, 1]}\n"
                 "  - {region: rb, section: a, orientation: [0, 0, 1]}")
    prep = prepare_case(path)
    for m in prep.case.members:
        assert m.material is None
    sol = _solver(path)
    assert sol.one_material
    assert all(m is sol.material for m in sol.member_material)
    assert sol.summary().splitlines()[2].strip().startswith("material      linear_elastic;")


def test_two_sections_naming_the_same_material_still_write_nothing(tmp_path):
    """Two sections, one material: the comparison is on the RESOLVED spec, so
    naming the same material twice is not a second material."""
    path = _case(tmp_path, ONE_MAT,
                 f"  a: {{type: beam, material: steel, shape: general, area: {A1}, "
                 f"iy: {I1}, iz: {I1}, j: {2*I1}}}\n"
                 f"  b: {{type: beam, material: steel, shape: general, area: {A2}, "
                 f"iy: {I2}, iz: {I2}, j: {2*I2}}}", SPLIT)
    prep = prepare_case(path)
    assert [m.material for m in prep.case.members] == [None, None]
    assert _solver(path).one_material


# --------------------------------------------------------------------------- #
#  the input model
# --------------------------------------------------------------------------- #
def test_the_second_material_travels_down_on_its_own_group(tmp_path):
    prep = prepare_case(_real(tmp_path))
    groups = {m.region: m for m in prep.case.members}
    assert groups["ra"].material is None                 # the model's
    assert groups["rb"].material["E"] == pytest.approx(E2)
    assert groups["rb"].material["rho"] == pytest.approx(R2)
    # `material` stays the FIRST line section's, for every single-material
    # reader downstream
    assert prep.case.material["E"] == pytest.approx(E1)


# --------------------------------------------------------------------------- #
#  the arithmetic: against the E-scaling workaround, and against closed form
# --------------------------------------------------------------------------- #
def _closed_form_tip():
    x = np.linspace(0, N*L, 400001)
    EI = np.where(x < 4*L, E1*I1, E2*I2)
    kap = -P*(N*L - x)/EI
    th = np.concatenate([[0], np.cumsum((kap[1:] + kap[:-1])/2*np.diff(x))])
    return np.concatenate([[0], np.cumsum((th[1:] + th[:-1])/2*np.diff(x))])[-1]


def test_two_materials_and_the_e_scaled_twin_agree_on_axial_and_bending(tmp_path):
    """EA and EI are products, so scaling the section by E_i/E_ref reproduces
    the two-material model's axial and bending stiffness exactly (to rounding).
    That is why the workaround worked, and why nothing built on it was wrong."""
    a = _solver(_real(tmp_path / "real"))
    b = _solver(_scaled(tmp_path / "scaled"))
    assert _tip(a) == pytest.approx(_tip(b), rel=1e-14)
    assert _tip(a) == pytest.approx(_closed_form_tip(), rel=1e-5)
    # axial rows (dof 0 of each node) and the bending rows are the same matrix
    ka, kb = a.K.toarray(), b.K.toarray()
    rows = [6*n + c for n in range(N + 1) for c in (0, 1, 2, 4, 5)]
    assert ka[np.ix_(rows, rows)] == pytest.approx(kb[np.ix_(rows, rows)], rel=1e-12)


def test_but_the_e_scaled_twin_gets_torsion_wrong_when_poisson_differs(tmp_path):
    """G = E / 2(1 + nu), and the workaround could only scale E. With the
    girder at nu = 0.3 and the C60 at nu = 0.2 the scaled twin's concrete
    members are 2.6/2.4 = 8.33% soft in torsion -- silently, because nothing
    in the model says which nu a scaled section was meant to have. A real
    material per group is the only way to state it."""
    a = _solver(_real(tmp_path / "real"))
    b = _solver(_scaled(tmp_path / "scaled"))
    rx = [6*n + 3 for n in range(N + 1)]
    ta = np.abs(a.K.toarray()[np.ix_(rx, rx)]).sum()
    tb = np.abs(b.K.toarray()[np.ix_(rx, rx)]).sum()
    # the steel half is identical; only the concrete half moves
    gj_real = E2/(2*1.2) * 2*I2
    gj_scaled = E1/(2*1.3) * 2*I2*K
    assert gj_real / gj_scaled == pytest.approx(1.3/1.2, rel=1e-14)
    assert ta > tb                      # the scaled twin really is the soft one
    # four concrete elements, each contributing |GJ/L| on four matrix entries
    assert (ta - tb) / (gj_real - gj_scaled) == pytest.approx(4*4/L, rel=1e-12)


def test_they_agree_on_translational_mass_and_disagree_on_rotary(tmp_path):
    """`nodal_masses` adds translation only, so the workaround could buy back
    the missing weight and never the missing rotary inertia. That gap is what
    this stage bought: 0.535 of the true value, on this model."""
    a = _solver(_real(tmp_path / "real"))
    b = _solver(_scaled(tmp_path / "scaled"))
    ma = a.physical_mass.reshape(-1, 6)
    mb = b.physical_mass.reshape(-1, 6)
    true_m = R1*A1*4*L + R2*A2*4*L
    assert ma[:, 0].sum() == pytest.approx(true_m, rel=1e-12)
    assert mb[:, 0].sum() == pytest.approx(true_m, rel=1e-12)

    # about a transverse axis (ry: the members lie along x) the lumped
    # rho A L^3/24; about their own axis the torsional rho Ip L/2, Ip = 2 I
    # (v1.0: it was rho A L^3/24 about all three)
    true_rot = 2*(R1*A1*L**3/24*4 + R2*A2*L**3/24*4)
    assert ma[:, 4].sum() == pytest.approx(true_rot, rel=1e-12)
    assert mb[:, 4].sum() / ma[:, 4].sum() == pytest.approx(0.5353, abs=5e-4)
    assert ma[:, 3].sum() == pytest.approx(R1*2*I1*L*4 + R2*2*I2*L*4, rel=1e-12)


def test_the_summary_names_every_material_it_carries(tmp_path):
    line = [l for l in _solver(_real(tmp_path)).summary().splitlines()
            if "material" in l][0]
    assert "2 linear_elastic" in line
    assert "(4 el)" in line


# --------------------------------------------------------------------------- #
#  cables: a group's own E drives its force AND its unstressed length
# --------------------------------------------------------------------------- #
CABLE_MATS = ("  a: {type: linear_elastic, E: 2.0e+11, nu: 0.3, rho: 7850.0}\n"
              "  b: {type: linear_elastic, E: 1.0e+11, nu: 0.3, rho: 7850.0}")
CABLE_SECS = ("  ca: {type: bar, material: a, shape: general, area: 0.01}\n"
              "  cb: {type: bar, material: b, shape: general, area: 0.01}")


def test_a_cables_axial_force_uses_its_own_modulus(tmp_path):
    path = _case(tmp_path, CABLE_MATS, CABLE_SECS,
                 "  - {region: ra, section: ca}\n  - {region: rb, section: cb}")
    sol = _solver(path)
    # the same 1% strain on one member of each group
    for e, E in ((0, 2.0e11), (N - 1, 1.0e11)):
        N_e = sol._truss_force(e, sol.member_Lu[e] * 1.01)
        assert N_e == pytest.approx(E * 0.01 * 0.01, rel=1e-12)


def test_prestress_is_a_stress_so_the_unstressed_length_uses_the_groups_e(tmp_path):
    """L_u = L_0 / (1 + sigma/E): the strain is this cable's steel's, and with
    one E for the model the softer group was cut to the wrong length."""
    path = _case(tmp_path, CABLE_MATS, CABLE_SECS,
                 "  - {region: ra, section: ca, prestress: 1.0e+8}\n"
                 "  - {region: rb, section: cb, prestress: 1.0e+8}")
    sol = _solver(path)
    assert sol.member_Lu[0] == pytest.approx(L / (1 + 1.0e8/2.0e11), rel=1e-14)
    assert sol.member_Lu[N-1] == pytest.approx(L / (1 + 1.0e8/1.0e11), rel=1e-14)


# --------------------------------------------------------------------------- #
#  temperature: alpha is the member's
# --------------------------------------------------------------------------- #
def test_each_member_expands_at_its_own_alpha(tmp_path):
    """A steel deck at 1.2e-5 on concrete at 1.0e-5. Before this stage both
    took the model material's number, and which number that was depended on
    which section was written first."""
    mats = ("  steel: {type: linear_elastic, E: 2.1e+11, nu: 0.3, rho: 7850.0, "
            "thermal_expansion: 1.2e-5}\n"
            "  conc:  {type: linear_elastic, E: 3.6e+10, nu: 0.2, rho: 2650.0, "
            "thermal_expansion: 1.0e-5}")
    path = _case(tmp_path, mats,
                 f"  a: {{type: beam, material: steel, shape: general, area: {A1}, "
                 f"iy: {I1}, iz: {I1}, j: {2*I1}}}\n"
                 f"  b: {{type: beam, material: conc, shape: general, area: {A1}, "
                 f"iy: {I1}, iz: {I1}, j: {2*I1}}}", SPLIT)
    sol = _solver(path)
    d = sol._thermal_load(np.arange(sol.nn), {"uniform": 50.0})
    assert d[0, 6] == pytest.approx(1.2e-5 * 50.0 * L, rel=1e-14)
    assert d[N-1, 6] == pytest.approx(1.0e-5 * 50.0 * L, rel=1e-14)


def test_one_alpha_takes_the_same_path_it_always_did(tmp_path):
    """The grouping by alpha exists so a single-material model calls
    `local_deformation` once, with the argument it always got."""
    mats = ("  steel: {type: linear_elastic, E: 2.1e+11, nu: 0.3, rho: 7850.0, "
            "thermal_expansion: 1.2e-5}")
    path = _case(tmp_path, mats,
                 f"  a: {{type: beam, material: steel, shape: general, area: {A1}, "
                 f"iy: {I1}, iz: {I1}, j: {2*I1}}}",
                 "  - {region: ra, section: a, orientation: [0, 0, 1]}\n"
                 "  - {region: rb, section: a, orientation: [0, 0, 1]}")
    sol = _solver(path)
    d = sol._thermal_load(np.arange(sol.nn), {"uniform": 50.0})
    from vfife.core.temperature import local_deformation
    assert np.array_equal(d, np.vstack(
        [local_deformation(1.2e-5, {"uniform": 50.0}, sol.length)]))


# --------------------------------------------------------------------------- #
#  refusals
# --------------------------------------------------------------------------- #
def test_a_member_group_may_not_smuggle_in_a_nonlinear_material(tmp_path):
    """`BeamSolver` demands linear_elastic of the model's material; a group is
    not a door around that."""
    mats = (ONE_MAT + "\n  c: {type: concrete01, fpc: 3.0e7, epsc0: 0.002, "
            "fcu: 1.5e7, epscu: 0.02, rho: 2400}")
    path = _case(tmp_path, mats,
                 f"  a: {{type: beam, material: steel, shape: general, area: {A1}, "
                 f"iy: {I1}, iz: {I1}, j: {2*I1}}}\n"
                 f"  b: {{type: beam, material: c, shape: general, area: {A1}, "
                 f"iy: {I1}, iz: {I1}, j: {2*I1}}}", SPLIT)
    with pytest.raises(Exception, match="linear_elastic only"):
        _solver(path)


# --------------------------------------------------------------------------- #
#  the bridge this was for
# --------------------------------------------------------------------------- #
def test_the_five_materials_of_a_hybrid_cable_supported_bridge(tmp_path):
    """Xihoumen's Table 1: steel girder, C60 tower, C40
    pier, stay cable, main cable. This combination is what the old refusal
    made impossible to write down, and it is the reason this stage exists."""
    xs = [(2.10e11, 7850.0), (3.60e10, 2650.0), (3.25e10, 2550.0),
          (1.81e11, 7810.0), (2.00e11, 8400.0)]
    names = ["girder", "tower", "pier", "stay", "main"]
    mats = "\n".join(f"  {n}: {{type: linear_elastic, E: {E}, nu: 0.3, rho: {r}}}"
                     for n, (E, r) in zip(names, xs))
    secs = "\n".join(f"  s{i}: {{type: beam, material: {n}, shape: general, "
                     f"area: {A1}, iy: {I1}, iz: {I1}, j: {2*I1}}}"
                     for i, n in enumerate(names))
    # five regions, one element each in the first five; the rest take girder
    regions = "".join(
        f"  r{i}: {{box: [{i*L-0.01}, {(i+1)*L+0.01}, -0.01, 0.01, -0.01, 0.01]}}\n"
        for i in range(5))
    assigns = "\n".join(f"  - {{region: r{i}, section: s{i}, orientation: [0, 0, 1]}}"
                        for i in range(5))
    assigns += ("\n  - {region: rest, section: s0, orientation: [0, 0, 1]}")
    regions += f"  rest: {{box: [{5*L-0.01}, {8*L+0.01}, -0.01, 0.01, -0.01, 0.01]}}\n"
    sol = _solver(_case(tmp_path, mats, secs, assigns, extra=regions))
    assert not sol.one_material
    assert sorted(set(np.round(sol.member_E, 6))) == sorted({E for E, _ in xs})
    assert sorted(set(np.round(sol.member_rho, 6))) == sorted({r for _, r in xs})
    assert "5 linear_elastic" in sol.summary()


def test_a_connector_group_is_told_it_has_no_material():
    with pytest.raises(Exception, match="connector2 has no material"):
        input_model.schema_member_group_probe = None
        from vfife.io.schema import MemberGroupSpec
        MemberGroupSpec(region="link", type="connector2",
                        stiffness=[1e9]*6,
                        material={"type": "linear_elastic", "E": 2e11, "nu": 0.3, "rho": 7850})
