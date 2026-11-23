"""Temperature on a beam (cases/case_27).

A temperature field is a FREE STRAIN, not a load, and everything below is one
consequence of that distinction:

  * a member free to take it carries nothing, and says so in its own reported
    actions -- the equivalent-nodal-load form of this gets the displacements
    right and reports N = EA alpha dT on a member with no stress in it;
  * a member held carries exactly EA alpha dT;
  * a section under a NONLINEAR profile carries self-equilibrated stress even
    with nothing holding it, because plane sections stay plane;
  * and the compiled kernels agree with the reference paths to the bit.
"""
import pathlib
import sys

import numpy as np
import pytest

from vfife.api import _with_element, prepare_case
from vfife.cli import make_solver
from vfife.core.temperature import linear_field, local_deformation, section_field
from vfife.io import schema

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))
ROOT = pathlib.Path(__file__).resolve().parents[1]
E, NU, RHO, ALPHA = 2.1e11, 0.3, 7850.0, 1.2e-5
AREA, IY, IZ, SPAN = 0.10, 9.70e-4, 9.70e-4, 20.0


def _case(loads, constraints=None, **solver):
    """A 20 m beam on the moving-load mesh, section y standing up."""
    return {
        "name": "thermal_unit",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": str(ROOT / "benchmarks/assets/moving_load_beam_20.msh")},
        "materials": {"steel": {"type": "linear_elastic", "E": E, "nu": NU,
                                "rho": RHO, "thermal_expansion": ALPHA}},
        "sections": {"beam": {"type": "beam", "material": "steel", "shape": "general",
                              "area": AREA, "iy": IY, "iz": IZ, "j": 1.94e-3}},
        "assign": [{"region": "all", "section": "beam", "orientation": [0, 0, 1]}],
        "constraints": constraints or [{"region": "left", "dofs": [1, 1, 1, 1, 0, 0]},
                                       {"region": "right", "dofs": [0, 1, 1, 1, 0, 0]}],
        "loads": loads,
        "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 20000,
                   "load_steps": 10, "max_steps_per_load": 20000,
                   "damping": {"type": "kinetic"}, "tolerance": 1e-7, **solver},
        "output": {"path": "results/_unit_thermal.h5", "every": 1,
                   "fields": ["displacement"], "monitors": {}},
    }


def _run(case):
    prep = prepare_case(case)
    s = make_solver(_with_element(prep.case, prep.element, prep.case_path),
                    prep.mesh, prep.material)
    s.run(on_progress=lambda *a, **k: None)
    return s


def test_a_material_carries_its_own_thermal_expansion():
    """`alpha` is a property of the substance, not of the constitutive model,
    so it is set centrally and defaults to zero -- a model that never mentions
    it is unchanged, and a temperature load on materials that never mention it
    is refused rather than silently doing nothing."""
    from vfife.materials import create
    assert create("linear_elastic", E=E, nu=NU, rho=RHO).thermal_expansion == 0.0
    assert create("linear_elastic", E=E, nu=NU, rho=RHO,
                  thermal_expansion=1e-5).thermal_expansion == 1e-5
    assert create("concrete02", fpc=-3e7, epsc0=-0.002, fpcu=-6e6, epscu=-0.008,
                  ft=2.5e6, Ets=2e8, thermal_expansion=1e-5).thermal_expansion == 1e-5
    # and `alpha` still means what bond_ceb's own curve means by it
    with pytest.raises(ValueError):
        create("bond_ceb", tau_max=1.0e7, alpha=1.5)


def test_the_field_is_read_the_way_the_case_states_it():
    y = np.array([-0.5, 0.0, 0.1, 0.3, 0.5]); z = np.zeros(5)
    assert np.allclose(section_field({"uniform": 12.0}, y, z), 12.0)
    assert np.allclose(section_field({"gradient": [10.0, 0.0]}, y, z), 10.0*y)
    assert np.allclose(section_field({"gradient": [0.0, 4.0]}, y, np.ones(5)), 4.0)
    p = {"profile": [[-0.5, 0.0], [0.1, 0.0], [0.5, 20.0]]}
    assert np.allclose(section_field(p, y, z), [0, 0, 0, 10, 20])
    # held constant beyond its ends: a code profile says nothing about outside
    assert section_field(p, np.array([-9.0]), np.zeros(1))[0] == 0.0
    assert section_field(p, np.array([+9.0]), np.zeros(1))[0] == 20.0
    # and it can be stated along z, which is where `rect`/`box` put the depth
    assert np.allclose(section_field({**p, "axis": "z"}, z, y), [0, 0, 0, 10, 20])
    # a profile has no linear reduction without the section's area distribution
    assert linear_field(p) is None
    assert linear_field({"uniform": 3.0, "gradient": [1.0, 2.0]}) == (3.0, 1.0, 2.0)


def test_the_free_thermal_deformation_is_a_constant_curvature():
    """The end rotations `local_deformation` produces must be the ones the
    element's own curvature row reads back as the intended kappa -- checked
    against `precompute_shape_derivatives`, which is what the fibre beam and
    the kernels use."""
    from vfife.kernels.numba_fiber_beam import precompute_shape_derivatives
    L = np.array([2.5])
    d = local_deformation(ALPHA, {"uniform": 30.0, "gradient": [7.0, -3.0]}, L)[0]
    assert d[6] == pytest.approx(ALPHA*30.0*2.5)          # axial elongation
    Bky, Bkz = precompute_shape_derivatives(L, np.array([0.0, 0.5, 1.0]))
    for gi in range(3):
        kz = Bkz[0, gi] @ d[[1, 5, 7, 11]]
        ky = Bky[0, gi] @ d[[2, 4, 8, 10]]
        # eps = eps0 - y kz + z ky must equal alpha (dT + gy y + gz z)
        assert kz == pytest.approx(-ALPHA*7.0)
        assert ky == pytest.approx(-ALPHA*3.0)


def test_a_beam_free_to_expand_carries_nothing_and_says_so():
    """The member ACTIONS, not only the displacements. An equivalent-nodal-load
    implementation passes the first half of this test and fails the second:
    the beam comes out the right length and reports EA alpha dT of axial force
    it does not have."""
    s = _run(_case([{"region": "all", "temperature": {"uniform": 50.0}}]))
    q = s.dofs.view(s.q)
    assert q[s.mesh.regions["right"][0], 0] == pytest.approx(ALPHA*50.0*SPAN, rel=1e-4)
    _, actions = s.internal_force()
    assert np.abs(actions[:, 0]).max() < 1e-3 * E*AREA*ALPHA*50.0


def test_a_beam_held_at_both_ends_carries_exactly_EA_alpha_dT():
    s = _run(_case([{"region": "all", "temperature": {"uniform": 50.0}}],
                   [{"region": "left", "dofs": [1, 1, 1, 1, 0, 0]},
                    {"region": "right", "dofs": [1, 1, 1, 1, 0, 0]}]))
    _, actions = s.internal_force()
    assert actions[:, 0].max() == pytest.approx(E*AREA*ALPHA*50.0, rel=1e-9)


def test_a_gradient_on_one_span_bows_it_and_leaves_it_unstressed():
    """kappa L^2 / 8, and no moment anywhere: a determinate beam takes the
    curvature for free. Two spans cannot -- that is case_27."""
    g = 10.0
    s = _run(_case([{"region": "all", "temperature": {"gradient": [g, 0.0]}}]))
    q = s.dofs.view(s.q)
    assert q[s.mesh.regions["mid"][0], 2] == pytest.approx(ALPHA*g*SPAN**2/8, rel=2e-3)
    _, actions = s.internal_force()
    assert np.abs(actions[:, 5]).max() < 1e-3 * E*IZ*ALPHA*g


def test_the_numba_frame_kernel_matches_the_reference_to_the_bit():
    g = 10.0
    load = [{"region": "all", "temperature": {"uniform": 20.0, "gradient": [g, 0.0]}}]
    a = _run(_case(load))
    b = _run(_case(load, backend="numba"))
    assert np.abs(a.q - b.q).max() == 0.0


def test_a_general_section_refuses_a_profile_rather_than_reducing_it():
    """Reducing a profile to a mean and a slope needs the section's area
    distribution. A `general` section states area and second moments and has
    no shape, so the reduction would be an invention."""
    with pytest.raises(ValueError, match="fibres to integrate"):
        _run(_case([{"region": "all",
                     "temperature": {"profile": [[-0.5, 0.0], [0.5, 20.0]]}}]))


def test_a_load_is_one_kind_and_a_temperature_says_how_it_varies():
    with pytest.raises(ValueError, match="uniform, gradient or profile"):
        schema.LoadSpec(region="all", temperature={})
    with pytest.raises(ValueError, match="not both"):
        schema.LoadSpec(region="all", temperature={"uniform": 1.0,
                                                   "profile": [[0, 1], [1, 2]]})
    with pytest.raises(ValueError, match="at least two"):
        schema.LoadSpec(region="all", temperature={"profile": [[0, 1]]})
    with pytest.raises(ValueError, match="strictly increasing"):
        schema.LoadSpec(region="all", temperature={"profile": [[1, 1], [0, 2]]})
    with pytest.raises(ValueError, match="exactly one of vector"):
        schema.LoadSpec(region="all", vector=[0, 0, -1], temperature={"uniform": 1.0})
    assert schema.LoadSpec(region="all", temperature={"gradient": 5.0}).temperature.gradient \
        == [5.0, 0.0]


def test_a_nonlinear_profile_stresses_a_section_that_nothing_is_holding():
    """The whole reason a fibre section is needed for a code gradient. Solved
    here on the section alone -- the two numbers that null the resultants --
    against the same section as the solver builds it, so what is compared is
    the physics and not a second copy of the same loop.
    """
    from generate_thermal_beam import eigenstress
    prep = prepare_case(str(ROOT / "cases/case_27b_thermal_eigenstress.yaml"))
    s = make_solver(_with_element(prep.case, prep.element, prep.case_path),
                    prep.mesh, prep.material)
    fixed, _moving, _pre, groups, ground = s.step_data(0)
    s._force(groups, ground, 1.0)
    e = s.nel // 2
    y, a, imposed = s.fiber_y[e], s.fiber_area[e], s._imposed[e]
    assert np.any(imposed), "the fibres carry no imposed strain at all"
    # the imposed strain IS -alpha T: a fibre held while it is heated is in
    # compression, which is the sign a restrained member must produce
    # the top FIBRE sits at y = 0.475, not at the 0.5 the profile names, so
    # it is 18.75 K warm, not 20 -- the section is what the profile is read on
    assert imposed.min() == pytest.approx(-1.0e-5 * 20.0 * (0.475 - 0.1)/0.4, rel=1e-9)
    assert imposed.max() == 0.0

    Ec = 3.45e10
    eps0 = -(imposed*a).sum()/a.sum()
    kappa = (imposed*y*a).sum()/(a*y*y).sum()
    sigma = Ec*(eps0 - y*kappa + imposed)
    assert abs((sigma*a).sum()) < 1e-6 * abs(Ec*imposed*a).sum()      # no net force
    assert abs((-sigma*y*a).sum()) < 1e-6 * abs(Ec*imposed*y*a).sum()  # no net moment
    assert sigma.max() > 1.5e6 and sigma.min() < -2.0e6                # and real stress
    closed = eigenstress(n=4000)
    assert eps0 == pytest.approx(closed["eps0"], rel=2e-3)
    assert kappa == pytest.approx(closed["kappa"], rel=5e-3)


def test_the_fibre_kernel_carries_an_imposed_strain_on_concrete_too():
    """Until v0.4 the fused kernel had an imposed-strain array for STEEL
    fibres only, while the NumPy reference added `fiber_prestrain` to EVERY
    fibre. The two agreed because nothing ever set one on concrete; a
    temperature field does, so the kernel grew `cpre` -- and this is the test
    that the two paths still give the same force.

    On case_11's section, which has a kernel (concrete02 + steel02; a
    `linear_elastic` fibre has no scalar twin, so case_27b itself runs on the
    reference path).
    """
    import yaml
    case = yaml.safe_load((ROOT / "cases/case_11_fiber_rc_column_cyclic.yaml").read_text(encoding="utf-8"))
    case["mesh"]["path"] = str(ROOT / "cases" / case["mesh"]["path"])
    for m in case["materials"].values():
        m["thermal_expansion"] = 1.0e-5
    # the anchorage scales the bars' strain, and a temperature strain with it:
    # the two are refused together since v1.0, so the section is bonded here
    for a in case["assign"]:
        a.pop("anchorage", None)
    case["loads"] = (case.get("loads") or []) + [
        {"region": "all", "temperature": {"gradient": [40.0, 0.0]}}]
    prep = prepare_case(case)
    s = make_solver(_with_element(prep.case, prep.element, None), prep.mesh, prep.material)
    if getattr(s, "_kernel", None) is None:
        pytest.skip(f"no fused kernel here: {s.kernel_status}")
    data = s.step_data(0)
    s._force(data[3], data[4], 1.0)
    assert np.any(s._imposed), "no imposed strain to compare"
    assert np.any(s._cpre), "the kernel was handed no concrete imposed strain"
    s.q[:] = np.linspace(0.0, 1e-4, s.ndof)          # some deformation, not just the field
    f_ref, a_ref = s._internal_force_true()
    f_ker, a_ker = s._internal_force_kernel()
    scale = max(np.abs(f_ref).max(), 1.0)
    assert np.abs(f_ref - f_ker).max() < 1e-9 * scale
    assert np.abs(a_ref - a_ker).max() < 1e-9 * max(np.abs(a_ref).max(), 1.0)
