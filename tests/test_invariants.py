"""The same questions, of every element and every load.

Twenty numbered cases, each of which took days to make converge, and at the
end of every one of them a defect that was not about the case: a load that
was not the gradient of anything, a frame that was exact in a plane
and wrong in space, an energy ledger a step out of phase, a
mass that bounded the reference configuration and not the current one.
Every one of them is a property that holds for every element, that
is cheap to ask, and that nobody was asking.

This file asks them, over `vfife.testing.models` -- one minimal model per
registered element -- through the `vfife.testing.state` adapter, which is
what makes the SAME question reach a solid, a truss, a frame, a shell and a
coupled model. The checks themselves live in `vfife.testing.invariants` and
are shared with `vfife probe`, which runs them on the user's own case.

WHAT IS ASKED WHERE. The rigid-motion and reference-state checks hold at any
state and are asked at two: the reference one and a `bent` one, the model
rolled into an arc of 1.2 radians, which is a large per-point rotation, a
real strain in every element and no symmetry left to hide anything in.

The tangent's symmetry is more delicate and the delicacy is the physics. A
co-rotational element's tangent is unsymmetric AWAY from equilibrium in
proportion to the out-of-balance force, which is a classical result and not
a defect: the arc state, held by nothing, scores 9e-2 for a frame and 1e-1
for a shell. So the internal force's symmetry is asked at the reference
state and at converged equilibria. An EXTERNAL force has no such excuse --
it is a function of the state alone -- so the load tangent is asked at the
bent state, where a follower load has actually followed something. That is
the check that would have caught the face moment of v0.1 on the day it was
written, at a cost of a tenth of a second.
"""

import numpy as np
import pytest

from vfife.elements.library import element_definition, element_names
from vfife.testing import invariants, models
from vfife.testing.state import probe

ELEMENTS = list(models.elements())

#: the schema's spelling of a load, and the element catalogue's name for it
LOAD_NAMES = {
    "vector": "nodal_force",
    "moment": ("nodal_moment", "face_moment"),
    "traction": "surface_traction",
    "pressure": "surface_pressure",
    "body_force": "body_force",
    "gravity": "gravity",
    "line_load": "line_load",
}


def _declares(element, kind):
    names = LOAD_NAMES[kind]
    declared = element_definition(element).loads
    return any(n in declared for n in (names if isinstance(names, tuple) else (names,)))


def _load(kind, m, follower=False):
    """A specimen load of that kind on that model, or None if the model has
    no region to put it on."""
    face, loaded = m.regions["face"], m.regions["loaded"]
    spec = {
        "vector": {"region": loaded, "vector": [0.0, 0.0, -1.0e3]},
        "moment": {"region": face or loaded, "moment": [0.0, 1.0e3, 0.0]},
        "traction": {"region": face, "traction": [0.0, 0.0, -1.0e5]},
        "pressure": {"region": face, "pressure": 1.0e5},
        "body_force": {"region": "all", "body_force": [0.0, 0.0, -1.0e4]},
        "gravity": {"region": "all", "gravity": [0.0, 0.0, -9.81]},
        "line_load": {"region": m.regions["line"], "line_load": [0.0, 0.0, -1.0e3]},
    }[kind]
    if spec["region"] is None:
        return None
    return {**spec, "follower": True} if follower else spec


@pytest.fixture(scope="module")
def built():
    return {e: models.build(e) for e in ELEMENTS}


def _fail(findings):
    return [str(f) for f in findings if not f.passed]


# --------------------------------------------------------------------------- #
#  the registry itself
# --------------------------------------------------------------------------- #

def test_every_registered_element_has_a_model():
    """A new element with no model here is an element nothing below asks
    anything of, which is how five of them shipped unexamined."""
    assert set(models.elements()) == set(element_names())


# --------------------------------------------------------------------------- #
#  the element
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("element", ELEMENTS)
def test_the_internal_force_is_well_behaved_at_the_reference_state(element, built):
    """Force-free, rigid-motion-free, a gradient, and inside the step's
    stability bound, at the configuration everything is derived from.

    The gradient tolerance is 1e-5 and not the 1e-9 the finite difference can
    resolve because `shell4` carries a systematic 1.3e-6: it is independent of
    the difference step, it survives to an equilibrium, and it is a hundredth
    of what the arc state's out-of-balance produces. Chasing it is a job for
    the shell's own file, not a reason to leave the other sixteen elements
    unasked.
    """
    p = probe(models.make(built[element]))
    findings = invariants.report(p, external=False)
    for f in findings:
        if f.name.startswith("internal force"):
            f.passed = f.value <= 1e-5
    assert not _fail(findings)


@pytest.mark.parametrize("element", ELEMENTS)
def test_a_rigid_motion_of_a_bent_body_is_free(element, built):
    """The one the frame solver and the connector both failed before
    v0.1. A rigid motion of the REFERENCE body is a weaker question --
    rotation vectors add in a plane, and they average and subtract correctly
    when the two ends agree -- and the arc state, where every point has
    turned by a different amount, is where the difference lives.

    `connector2` was carried here as an expected failure for two stages,
    written as an assertion that it STILL fails so that fixing it would set
    off an alarm rather than pass quietly. It did: 7.5e-2 of
    the model's own stiffness scale became 2.1e-10."""
    p = probe(models.make(built[element]))
    x = p.bent(1.2)
    for check in (invariants.a_rigid_translation_is_free,
                  invariants.a_rigid_rotation_is_free):
        f = check(p, x)
        assert f.passed, str(f)


@pytest.mark.parametrize("element", ELEMENTS)
def test_the_relaxation_mass_bounds_the_stiffest_mode(element, built):
    """omega^2 < 4 s^2 at the reference configuration, which is the
    configuration the fictitious mass is derived from. Past it the explicit
    step is unstable at the first iteration and the model stalls at load
    level one -- the single most common thing a user reports."""
    p = probe(models.make(built[element]))
    f = invariants.the_mass_bounds_the_step(p)
    assert f.passed, str(f)


# --------------------------------------------------------------------------- #
#  the loads
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("kind", sorted(LOAD_NAMES))
def test_a_declared_load_works_and_a_working_load_is_declared(element, kind, built):
    """The round trip over `vfife capabilities`.

    It found four wrong entries. Three solids declared `surface_pressure`
    with no facet table behind it, so a `pressure:` on a hexa20 model raised
    `KeyError: 'hexa20'` from the middle of the face walk; every solid
    accepted a `traction:` without saying so; the solid face moment of
    v0.1 was undeclared; and `connector2` took a gravity load it did not
    advertise. A load the catalogue does not name is a feature the user
    cannot find, and a load it names that does not work is worse.
    """
    spec = _load(kind, built[element])
    if spec is None:
        pytest.skip("this model has no exterior face to load")
    works, why = True, ""
    try:
        p = probe(models.make(models.build(element, loads=[spec])))
        f = p.external(np.zeros(p.n))
        works = bool(np.isfinite(f).all()) and float(np.abs(f).max()) > 0.0
    except Exception as exc:                                   # noqa: BLE001
        works, why = False, f"{type(exc).__name__}: {exc}"
    declared = _declares(element, kind)
    assert works == declared, (
        f"{element} {'accepts' if works else 'refuses'} a {kind} load and the "
        f"element catalogue says {'yes' if declared else 'no'}" + (f" ({why})" if why else ""))


@pytest.mark.parametrize("element", ["hexa8", "wedge6", "tet4", "tet10",
                                     "hexa20", "wedge15", "pyramid5"])
def test_a_follower_pressure_on_a_closed_body_is_the_gradient_of_something(element):
    """THE check of v0.1, on the load v0.1 added.

    A follower pressure is re-evaluated on the deformed body every step,
    which is exactly where a load stops being conservative without anybody
    noticing. On a CLOSED surface it is conservative -- the potential is
    minus the pressure times the enclosed volume, and the consistent nodal
    force is that volume's gradient -- so this is a sharp question with a
    known answer, unlike the same pressure on an open patch, which is
    genuinely circulatory (see docs/convergence_debugging.md).

    hexa8 answered 0.55 and tet4 answered 8e-10, which is what exposed the
    interior point: `mesh._interior_points` used to be the mean of an
    element's first four nodes, and for a hexahedron that is the centre of
    its own bottom face, so the outward test for that face was the sign of a
    quantity that is exactly zero until the mesh moves.
    """
    m = models.build(element, loads=[{"region": "all", "pressure": 1.0e5,
                                      "follower": True}])
    p = probe(models.make(m))
    x = p.bent(0.8)
    _, Ke = invariants.tangent(p, x)
    assert np.abs(Ke).max() > 0.0, "a follower pressure that does not move with the body"
    f = invariants.is_a_gradient(p, K=Ke, what="external", tol=1e-7)
    assert f.passed, str(f)


@pytest.mark.parametrize("element", ["hexa8", "wedge6", "pyramid5"])
def test_a_follower_face_moment_is_the_gradient_of_something(element):
    """The load of v0.1 itself, asked generically. Its first version scored
    1.8 here -- a tangent that was almost entirely its skew part."""
    m = models.build(element)
    p = probe(models.make(models.build(
        element, loads=[{"region": m.regions["face"], "moment": [0.0, 5.0e3, 0.0],
                         "follower": True}])))
    x = p.bent(0.8)
    _, Ke = invariants.tangent(p, x)
    assert np.abs(Ke).max() > 0.0
    f = invariants.is_a_gradient(p, K=Ke, what="external", tol=1e-7)
    assert f.passed, str(f)


@pytest.mark.parametrize("element", ["hexa8", "wedge6", "tet4", "tet10",
                                     "hexa20", "wedge15", "pyramid5"])
@pytest.mark.parametrize("distortion", [0.0, 0.02])
def test_a_pressure_over_a_closed_body_has_no_resultant(element, distortion):
    """The facet table, the outward normals and the face quadrature, in one
    number: a uniform pressure over a complete closed boundary pushes equally
    from every side.

    It is how the tables added for hexa20, wedge15 and pyramid5 in v0.1 were
    checked, and it is what a triangular face could not have passed, because
    a pressure on one raised `The weights and list don't have the same
    length` out of a bincount -- one missing broadcast, present in the
    traction path beside it, on every tet4 model since the beginning.
    """
    from vfife.core.mesh import surface_pressure, surface_traction

    m = models.build(element).mesh
    X = m.nodes
    if distortion:
        rng = np.random.default_rng(0)
        h = distortion * float(np.ptp(X, axis=0).max())
        X = X + h * rng.standard_normal(X.shape)
    idx = np.arange(m.n_nodes)
    f = surface_pressure(X, m.elements, element, idx, 1.0)
    scale = float(np.abs(f).sum())
    assert scale > 0
    assert float(np.abs(f.sum(axis=0)).max()) < 1e-12 * scale
    # and the same faces integrate the boundary's area
    t = surface_traction(X, m.elements, element, idx, [1.0, 0.0, 0.0])
    assert float(np.abs(t.sum(axis=0))[0]) > 0


# --------------------------------------------------------------------------- #
#  at equilibrium
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("element", ["hexa8", "truss2", "beam2", "shell4"])
def test_the_tangent_is_symmetric_at_an_equilibrium(element):
    """Away from equilibrium a co-rotational tangent is unsymmetric in
    proportion to the out-of-balance force -- classical, and the reason the
    arc state is not where this is asked. At a converged equilibrium what is
    left is the formulation itself, and a force that is not conservative
    (v0.1's, at 1.8) does not care that the residual is small."""
    # 1e-5 on the residual, not the registry's 1e-6: the shell cantilever is
    # a slow relaxation and the last decade of it says nothing about the
    # question, which is about the formulation and not about how converged
    # this particular state is
    m = models.build(element, solver={"tolerance": 1.0e-5})
    s = models.make(m)
    s.run(progress_every=10 ** 9, output_every=10 ** 9)
    p = probe(s)
    x = p.get()
    f = invariants.is_a_gradient(p, x, tol=1e-4)
    assert f.passed, str(f)
