"""Every registered material, every element-independent question.

The companion of `test_invariants.py`, one layer down: that file asks the
questions of the ELEMENTS, this one asks them of the constitutive laws. Both
are breadth, not depth -- `test_cdp.py` and `test_bond.py` are worth more than
anything here for the law they were written for, and neither of them would
notice a NEW law arriving with the same defect.

Two halves:

  * the sweep -- `materials.report(name)` over every registered law, with the
    open findings listed as REVERSE assertions, so a fix trips the test on the
    day it lands rather than quietly making the suite greener;

  * the planted defects -- a handful of deliberately broken laws, one per
    check, so that "everything passes" means the questions can still bite. A
    check nobody has ever seen fail is a check nobody has tested.
"""

import numpy as np
import pytest

from vfife.materials import base as mat_base
from vfife.materials.elastic import LinearElastic
from vfife.testing import materials as M


# --------------------------------------------------------------------------- #
#  the sweep
# --------------------------------------------------------------------------- #

#: Findings that are open, with the number measured on the day they were
#: recorded. Each is asserted to STILL fail: see v0.1.
_OPEN = {
    ("cdp", "the_crack_band_energy_does_not_depend_on_the_element_size"): 0.05,
}


@pytest.mark.parametrize("name", M.materials())
def test_every_registered_material_says_how_to_drive_it(name):
    """A law with no entry in `_DRIVE` cannot be asked anything, so the
    registry and the harness are kept in step here rather than by remembering.
    """
    s = M.spec(name)
    assert s.amplitude != 0.0
    assert s.kinematics in ("solid_3d", "uniaxial", "bond")
    assert abs(np.linalg.norm(s.direction) - 1.0) < 1e-12


@pytest.mark.parametrize("name", M.materials())
def test_the_material_answers_every_question_that_applies(name):
    findings = M.report(name)
    assert len(findings) == len(M.CHECKS)
    bad = []
    for f in findings:
        if (name, f.name) in _OPEN:
            assert not f.passed, (
                f"{name}: {f.name} now PASSES at {f.value:.3e}. It is listed as "
                "an open finding in v0.1; if it is fixed, delete the "
                "entry from _OPEN and say so in the note.")
            continue
        if not f.passed:
            bad.append(str(f))
    assert not bad, f"{name}\n" + "\n".join(bad)


def test_vfife_probe_asks_the_material_questions_of_the_case(capsys):
    """The user-facing half. The material block a case declares is not the
    harness's specimen -- case_12's `cdp` carries `viscosity: 1.0` and a
    `Gf` of its own -- so the questions are asked on the analyst's numbers.
    """
    from vfife.cli import main

    main(["probe", "cases/case_09_connector_support.yaml",
          "--bend", "0", "--max-dofs", "0"])
    out = capsys.readouterr().out
    assert "the case material: linear_elastic" in out
    assert "the_virgin_state_is_free" in out
    assert "rotating_the_strain_rotates_the_stress" in out


def test_a_viscous_case_is_not_accused_of_reading_the_time_step():
    """`cdp` is rate-independent at `viscosity: 0` and not at `viscosity: 1`,
    which is a PARAMETER and cannot live in a table keyed by the material
    name. Asked of the built law, case_12 skips two checks with a reason
    instead of failing them at 0.93 and 1.19.
    """
    props = {"type": "cdp", "E": 2.985e10, "nu": 0.2, "rho": 2400,
             "viscosity": 1.0, "l_ref": 0.05, "damage_max": 0.8,
             "compression": {"source": "gb50010", "fc": 4.4e7},
             "tension": {"source": "fracture_energy", "ft": 3.42e6, "Gf": 140}}
    findings = {f.name: f for f in M.report_spec(M.spec_for(props))}
    for name in ("the_time_step_does_not_change_a_rate_independent_law",
                 "unloading_is_no_stiffer_than_the_first_loading"):
        assert findings[name].skipped, findings[name]
        assert "viscosity = 1" in findings[name].detail
    library, _ = M.make("cdp")
    assert not M.rate_dependence(library, M.spec("cdp"))


def test_steel02s_envelope_stays_under_its_own_asymptote():
    """Closed in v1.0 (open since v0.4 at 3.45% of Fy).

    The first-loading branch used to put the yield point at `Fy / (E q)` with
    `q = b + (1-b)/2^(1/R)`, so that the CURVE passed through `Fy`; the
    envelope then ran `(1-b)(1/q - 1)` over `Fy + bE(eps - eps_y)`. The law is
    now OpenSees' Steel02, which puts the ASYMPTOTE INTERSECTION at `Fy / E`
    and lets the rounded curve pass below it. The price recorded in v0.4:
    about 2.6% of flexural capacity on every section whose bars sit on their
    plateau (case_31: 58.98 -> 57.50 kN m).
    """
    material, s = M.make("steel02")
    f = M.the_envelope_stays_under_its_own_hardening_asymptote(material, s)
    assert f.passed, f


def test_an_envelope_over_its_own_asymptote_is_caught():
    """The check, shown biting on a law that is not the one it found."""
    class TooStrong(_Elastic):
        """Elastic to yield, then a hardening line 5% of Fy too high. Written
        out rather than wrapped around `LinearElastic` because that law answers
        a UNIAXIAL STRAIN with the constrained modulus (1.35 E at nu = 0.3),
        which is not an overshoot -- it is a different question."""
        Fy, b = 4.0e8, 0.01

        def update(self, strain, state, dt):
            e, ey = strain[:, 0], self.Fy / self.E
            out = np.zeros_like(strain)
            out[:, 0] = np.where(
                np.abs(e) <= ey, self.E * e,
                np.sign(e) * (1.05 * self.Fy
                              + self.b * self.E * (np.abs(e) - ey)))
            return out

    f = M.the_envelope_stays_under_its_own_hardening_asymptote(
        TooStrong(), _spec(kinematics="uniaxial", amplitude=6.0e-3))
    assert not f.passed and f.value == pytest.approx(0.05, rel=0.02), f


def test_a_law_that_declares_no_hardening_pair_is_skipped_not_passed():
    f = M.the_envelope_stays_under_its_own_hardening_asymptote(
        _Elastic(), _spec(kinematics="uniaxial"))
    assert f.skipped and f.passed


def test_the_open_finding_is_the_size_it_was_measured_at():
    """`cdp` dissipates more fracture energy per unit crack area on a bigger
    element: 8.6% per doubling, measured at l_ref and 2 l_ref, and confirmed on
    the material's own uniaxial-stress driver (not an artefact of the harness's
    fixed lateral strain). Pinned so that a change to the crack band which
    makes it WORSE is also caught.
    """
    material, spec = M.make("cdp")
    f = M.the_crack_band_energy_does_not_depend_on_the_element_size(material, spec)
    assert 0.05 < f.value < 0.15, f


# --------------------------------------------------------------------------- #
#  planted defects: each check, shown biting
# --------------------------------------------------------------------------- #

def _spec(**kw):
    base = dict(name="linear_elastic", props={"E": 2.0e11, "nu": 0.3, "rho": 7850.0},
                kinematics="solid_3d", amplitude=1.0e-3)
    base.update(kw)
    return M.Spec(**base)


class _Elastic(LinearElastic):
    """The control: the harness's own reference law, unbroken."""

    def __init__(self):
        super().__init__(E=2.0e11, nu=0.3, rho=7850.0)


def test_the_control_passes_every_check():
    material, spec = _Elastic(), _spec()
    for check in M.CHECKS:
        f = check(material, spec)
        assert f.passed, f


def test_a_residual_stress_is_caught():
    class Prestressed(_Elastic):
        def update(self, strain, state, dt):
            out = super().update(strain, state, dt)
            out[:, 0] += 1.0e6          # a precompute that was never zeroed
            return out

    f = M.the_virgin_state_is_free(Prestressed(), _spec())
    assert not f.passed and f.value > 1e-3, f


def test_a_jump_in_the_law_is_caught():
    """The `bond_ceb` shape: a branch that switches on the current strain, so
    the stress moves a finite amount for an infinitesimal step."""
    class Jumpy(_Elastic):
        def update(self, strain, state, dt):
            out = super().update(strain, state, dt)
            out[strain[:, 0] > 5.0e-4, 0] += 1.0e7
            return out

    f = M.the_stress_is_continuous_in_the_strain(Jumpy(), _spec())
    # a 10 MPa jump inside a step of 2.4e-5 strain, against E0 = 2.7e11: the
    # measured slope is m/|de| times the jump height, ten times the limit
    assert not f.passed and f.value > 10.0, f


def test_the_fracture_exemption_forgives_the_rupture_and_nothing_else():
    """A brittle latch IS a stress discontinuity, so a law that declares one
    would fail the continuity check for doing exactly what it says it does --
    `rebar` at a rupture strain the path reaches reports 137 E0. The exemption
    covers the interval in which the failure fraction reaches one, and that is
    all: this law jumps at a THIRD of its rupture strain as well, and the
    check has to still report it.
    """
    class LatchesAndAlsoJumps(_Elastic):
        n_state = 1
        state_names = {"f": slice(0, 1)}

        def update(self, strain, state, dt):
            out = super().update(strain, state, dt)
            e = strain[:, 0]
            out[e > 2.0e-4, 0] += 1.0e7            # a defect, nothing to do with fracture
            state[:, 0] = np.maximum(state[:, 0], (e >= 6.0e-4).astype(float))
            out[state[:, 0] > 0.5] = 0.0           # the declared rupture
            return out

        def failure(self, state):
            return state[:, 0]

    class LatchesOnly(LatchesAndAlsoJumps):
        def update(self, strain, state, dt):
            out = LinearElastic.update(self, strain, state, dt)
            state[:, 0] = np.maximum(state[:, 0], (strain[:, 0] >= 6.0e-4).astype(float))
            out[state[:, 0] > 0.5] = 0.0
            return out

    spec = _spec(amplitude=1.0e-3)
    assert M.the_stress_is_continuous_in_the_strain(LatchesOnly(), spec).passed
    f = M.the_stress_is_continuous_in_the_strain(LatchesAndAlsoJumps(), spec)
    assert not f.passed and f.value > 10.0, f


def test_the_rupturing_bar_records_that_it_was_forgiven():
    """The harness's own `rebar` specimen carries a rupture strain, so the
    finding is a PASS with a note -- an exemption that leaves no trace is an
    exemption nobody can audit."""
    findings = {f.name: f for f in M.report("rebar")}
    f = findings["the_stress_is_continuous_in_the_strain"]
    assert f.passed, f
    assert "declared fracture" in f.detail, f
    assert findings["failure_is_irreversible"].passed
    assert not findings["failure_is_irreversible"].skipped


def test_energy_created_round_a_cycle_is_caught():
    class Pump(_Elastic):
        """Stiffer going up than coming down: a hysteresis loop traversed the
        wrong way round, which is a free energy source."""

        def update(self, strain, state, dt):
            out = super().update(strain, state, dt)
            # only in the first quadrant: scaling every unloading branch
            # scales the third leg too, and the two halves of the cycle then
            # cancel to 8e-17 -- which is what the first version of this
            # planted defect did, and it made the check look asleep.
            fake = (strain[:, 0] < state[:, 0]) & (strain[:, 0] > 0.0)
            state[:, 0] = strain[:, 0]
            out[fake] *= 1.02
            return out

        n_state = 1
        state_names = {"prev": slice(0, 1)}

    f = M.a_closed_cycle_does_not_create_energy(Pump(), _spec())
    assert not f.passed and f.value < -1e-3, f


def test_a_stiffer_unloading_branch_is_caught():
    class Stiffening(_Elastic):
        n_state = 1
        state_names = {"peak": slice(0, 1)}

        def update(self, strain, state, dt):
            out = super().update(strain, state, dt)
            past = strain[:, 0] < state[:, 0]
            state[:, 0] = np.maximum(state[:, 0], strain[:, 0])
            out[past] *= 1.5
            return out

    f = M.unloading_is_no_stiffer_than_the_first_loading(Stiffening(), _spec())
    assert not f.passed and f.value > 1.1, f


def test_points_talking_to_each_other_are_caught():
    class Gossip(_Elastic):
        """The classic: a reduction with no `axis=`, so the worst-strained
        point in the batch sets the response of all of them."""

        def update(self, strain, state, dt):
            out = super().update(strain, state, dt)
            return out * (1.0 - 0.1 * np.abs(strain).max() / 1.0e-3)

    f = M.the_points_do_not_talk_to_each_other(Gossip(), _spec())
    assert not f.passed and f.value > 1e-3, f


def test_a_rate_effect_in_a_rate_independent_law_is_caught():
    class Creeps(_Elastic):
        def update(self, strain, state, dt):
            return super().update(strain, state, dt) * (1.0 + 0.01 * dt)

    f = M.the_time_step_does_not_change_a_rate_independent_law(Creeps(), _spec())
    assert not f.passed and f.value > 1e-4, f


def test_an_axis_assumption_is_caught():
    class Preferred(_Elastic):
        """An axis assumed somewhere in a return map -- here, bluntly, a law
        that is 5% stiffer along x."""

        def update(self, strain, state, dt):
            out = super().update(strain, state, dt)
            out[:, 0] *= 1.05
            return out

    f = M.rotating_the_strain_rotates_the_stress(Preferred(), _spec())
    assert not f.passed and f.value > 1e-3, f


def test_healing_damage_is_caught():
    class Heals(_Elastic):
        n_state = 1
        state_names = {"d": slice(0, 1)}

        def update(self, strain, state, dt):
            state[:, 0] = np.abs(np.sin(3.0e3 * strain[:, 0]))   # rises and falls
            return super().update(strain, state, dt)

        def damage_measure(self, state):
            return state[:, 0]

    f = M.damage_does_not_heal(Heals(), _spec())
    assert not f.passed and f.value > 1e-3, f


def test_a_failure_that_heals_is_caught():
    """A failure fraction is a latch. Damage may recover --
    `cdp` restores stiffness when a crack closes, and `damage_does_not_heal`
    is asked on a monotonic path for exactly that reason -- but what is dead
    stays dead, and this one is asked on the closed cycle.
    """
    class Waver(_Elastic):
        n_state = 1
        state_names = {"f": slice(0, 1)}

        def update(self, strain, state, dt):
            state[:, 0] = np.abs(np.sin(3.0e3 * strain[:, 0]))
            return super().update(strain, state, dt)

        def failure(self, state):
            return state[:, 0]

    f = M.failure_is_irreversible(Waver(), _spec())
    assert not f.passed and f.value > 1e-3, f


def test_a_law_with_no_failure_measure_is_skipped_not_passed():
    f = M.failure_is_irreversible(_Elastic(), _spec())
    assert f.skipped, f


def test_an_unregularised_softening_law_is_caught():
    """The crack-band check needs a REGULARISED law to be asked at all, so the
    planted defect is a law that declares the regularisation and does not do
    it: its softening branch is the same in strain at every element size, so
    the energy per unit area scales with the element.
    """
    @mat_base.register("_planted_unregularised", regularised=True,
                       backends=("numpy",), verification="unit")
    class Unregularised(LinearElastic):
        E0, FT = 3.0e10, 3.0e6

        def __init__(self, E=3.0e10, nu=0.2, rho=2400.0, **_):
            super().__init__(E=E, nu=nu, rho=rho)
            self.l_ref = 0.2
            self.sigma_t0 = self.FT

        def bind_points(self, char_length):
            return None                      # the whole defect, in one line

        def update(self, strain, state, dt):
            out = super().update(strain, state, dt)
            e = strain[:, 0]
            peak = self.FT / self.E0
            soft = np.clip(1.0 - (e - peak) / (9.0 * peak), 0.0, 1.0)
            out[:, 0] = np.where(e > peak, self.FT * soft, out[:, 0])
            return out

    try:
        spec = M.Spec(name="_planted_unregularised",
                      props={"E": 3.0e10, "nu": 0.2, "rho": 2400.0},
                      kinematics="solid_3d", amplitude=1.0e-3)
        f = M.the_crack_band_energy_does_not_depend_on_the_element_size(
            Unregularised(), spec)
        assert not f.passed and f.value > 0.3, f
    finally:
        mat_base._REGISTRY.pop("_planted_unregularised", None)
        mat_base._DEFINITIONS.pop("_planted_unregularised", None)
