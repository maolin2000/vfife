"""Steel02 verified qualification tests: monotonic, step_convergence, energy, unload_reverse."""
import numpy as np
import pytest
from vfife.materials.base import create, definition
from vfife.catalog import library


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """Run in a folder of the test's own: these tests write the default
    `results.h5` (or results/...) beside the working directory, and parallel
    workers (pytest -n) wrote the same file at once."""
    monkeypatch.chdir(tmp_path)

def _follow(mat, history, state=None):
    state = mat.init_state(1) if state is None else state
    stresses = []
    for eps in history:
        strain = np.zeros((1, 6)); strain[0,0] = eps
        stresses.append(mat.update(strain, state, 1.0)[0,0])
    return np.asarray(stresses), state

def test_steel02_metadata_is_verified_uniaxial():
    meta = definition("steel02")
    assert meta.kinematics == ("uniaxial",)
    assert "steel02" in library()["materials"]
    assert "steel02" in library()["elements"]["beam2_fiber"]["compatible_materials"]
    assert meta.nonlinear is True
    # numpy AND numba: this law has `numba_uniaxial_fn` + `numba_params`, and
    # the fused fibre kernel inlines them -- every shipped fibre case has been
    # running steel02 on the kernel since v0.1. The declaration said numpy
    # only until v0.4, so `backend: numba` was refused by preflight for the
    # thing it was already doing. NOT warp: `kernels/warp_materials.py` has
    # elastic, j2, cdp, chaboche, rebar and the two bond laws, not this one.
    assert meta.backends == ("numpy", "numba")

def test_monotonic_tension_reaches_Fy_and_hardening():
    m = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01, R0=20)
    # monotonic to 0.004
    hist = np.linspace(0, 0.004, 41)
    stresses, _ = _follow(m, hist)
    # at 2 eps_y the curve is just under its asymptote Fy + bE(eps - eps_y)
    asym = 400e6 + 0.01 * 200e9 * (0.004 - 0.002)
    assert 0.99 * asym < stresses[-1] <= asym
    # compressive side symmetric
    m2 = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01)
    comp, _ = _follow(m2, np.linspace(0, -0.004, 41))
    assert comp[-1] == pytest.approx(-stresses[-1], rel=1e-12)

def test_step_partition_does_not_change_rate_independent_result():
    target = 0.003
    m1 = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01)
    m2 = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01)
    coarse, s_coarse = _follow(m1, [target])
    fine, s_fine = _follow(m2, np.linspace(0, target, 51))
    # stresses should be close; state hysteretic energy also close-ish
    assert coarse[-1] == pytest.approx(fine[-1], rel=0.05)
    # direction states should agree
    assert s_coarse[0,4] == pytest.approx(s_fine[0,4])

def test_unload_reverse_shows_Bauschinger():
    m = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01, R0=18)
    hist = [0.0, 0.003, 0.001, -0.001]
    stresses, state = _follow(m, hist)
    # After tension yielding, reverse compression stress should be lower (tension) than virgin?
    # Check that state has direction flipped and reversal captured
    # stress at -0.001 after prior tension should be more compressive than virgin elastic
    m_virgin = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01)
    virgin, _ = _follow(m_virgin, [ -0.001 ])
    # Bauschinger: after tensile plastic, compressive yielding earlier => stress less negative? Actually more negative due to kinematic shift
    # We check that histories differ
    assert stresses[-1] != pytest.approx(virgin[-1], rel=0.2)

def test_closed_plastic_cycle_has_positive_energy_and_elastic_has_zero():
    m = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01)
    # elastic small cycle
    e_state = m.init_state(1)
    _follow(m, [0, 0.0005, 0], e_state)
    e_energy = m.output_fields(e_state)["hysteretic_energy"][0]
    # plastic cycle
    m2 = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01)
    p_state = m2.init_state(1)
    _follow(m2, [0, 0.004, 0, -0.004, 0, 0.004, 0], p_state)
    p_energy = m2.output_fields(p_state)["hysteretic_energy"][0]
    assert e_energy == pytest.approx(0.0, abs=1e2)
    assert p_energy > 1e3

def test_fiber_compatibility_with_concrete01():
    from vfife.api import prepare_case, run_case
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        msh = pathlib.Path(td)/"m.msh"
        msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n2\n1 0 0 0\n2 1 0 0\n$EndNodes\n$Elements\n1\n1 1 2 1 1 1 2\n$EndElements\n", encoding="utf-8")
        yp = pathlib.Path(td)/"c.yaml"
        yp.write_text(f"""materials:
  steel: {{type: steel02, E: 200.0e9, Fy: 400.0e6, b: 0.01, rho: 7850}}
sections:
  rc: {{type: fiber, material: steel, shape: rect, width: 0.05, height: 0.10, ny: 6, nz: 6}}
assign:
  - {{region: all, section: rc}}
mesh:
  type: gmsh
  path: {msh}
regions:
  fixed: {{box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}}
  tip: {{box: [0.99, 1.01, -0.01, 0.01, -0.01, 0.01]}}
constraints:
  - region: fixed
    dofs: [true, true, true, true, true, true]
loads:
  - region: tip
    vector: [50000, 0, 0]
solver:
  backend: numpy
  mode: relaxation
  n_steps: 600
output:
  monitors:
    tip_disp: {{region: tip, dof: x, kind: displacement}}
""", encoding="utf-8")
        prep = prepare_case(str(yp))
        assert prep.preflight["status"] == "PASS"
        out = run_case(prep)
        assert out.converged


# --------------------------------------------------------------------------- #
#  inherited from the retired `rebar_gmp`
#
#  That law was this one, uncompiled: over 18 parameter sets and 2 100 steps of
#  cyclic strain the two agreed to 0.42% of the peak stress, while `rebar_gmp`
#  cost 4.5 us a point against steel02's 147 ns and had no scalar twin -- so a
#  fibre section that named it dropped off the fused kernel. It was removed
#  rather than kept as an alias. Its two real assertions live on here, because
#  retiring a law must not retire the questions it was answering.
# --------------------------------------------------------------------------- #

def test_the_reversal_is_smooth_and_has_no_stress_jump():
    m = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01, R0=20.0)
    state = m.init_state(1)
    tension, _ = _follow(m, np.linspace(0.0, 0.004, 41), state)
    reverse, _ = _follow(m, np.linspace(0.004, -0.004, 161), state)
    assert tension.max() > 0.99 * 400e6
    # the Bauschinger rounding (R falls from 20 toward 1.5) keeps the reverse
    # branch under Fy for a while: -391 MPa at -0.004
    assert reverse.min() < -0.95 * 400e6
    assert np.isfinite(reverse).all()
    assert abs(reverse[1] - tension[-1]) < 20e6, "a step at the reversal point"


def test_compression_yields_early_after_tension():
    """The Bauschinger effect, as a number rather than as 'the histories
    differ': a bar taken past yield in tension yields in compression before it
    reaches the virgin compressive yield strain of sigma_y/E."""
    m = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01, R0=20.0)

    def departs(path, state):
        """The stress at which the branch leaves its elastic line by 2% Fy."""
        sig, _ = _follow(m, path, state)
        line = sig[0] + 200e9 * (np.asarray(path) - path[0])
        return sig[np.argmax(np.abs(sig - line) > 0.02 * 400e6)]

    state = m.init_state(1)
    _follow(m, np.linspace(0.0, 0.004, 41), state)
    after_tension = departs(np.linspace(0.004, -0.004, 161), state)
    virgin = departs(np.linspace(0.0, -0.004, 81), m.init_state(1))
    # measured v1.0: -7.9 MPa against -381 MPa
    assert -0.1 * 400e6 < after_tension < 0.0
    assert virgin < -0.9 * 400e6


def test_the_retired_name_says_where_the_law_went():
    from vfife.core.diagnostics import UserError

    with pytest.raises(UserError) as excinfo:
        create("rebar_gmp", E=200e9, rho=7850, yield_strength=400e6)
    assert excinfo.value.code == "RETIRED_MATERIAL"
    assert "steel02" in excinfo.value.hint and "0.42%" in excinfo.value.hint
    assert "rebar_gmp" not in library()["materials"]
    assert "rebar_gmp" not in library()["elements"]["beam2_fiber"]["compatible_materials"]


# --------------------------------------------------------------------------- #
#  the OpenSees law
# --------------------------------------------------------------------------- #

class _OpenSeesSteel02:
    """`Steel02::setTrialStrain` from the OpenSees source, transcribed
    independently of vfife/materials/steel02.py (with a1..a4), committed at
    every call."""

    def __init__(s, Fy, E0, b, R0=20.0, cR1=0.925, cR2=0.15, a1=0.0, a2=1.0, a3=0.0, a4=1.0):
        s.Fy, s.E0, s.b, s.R0, s.cR1, s.cR2 = Fy, E0, b, R0, cR1, cR2
        s.a1, s.a2, s.a3, s.a4 = a1, a2, a3, a4
        s.epsmin = s.epsmax = s.epspl = s.epss0 = s.sigs0 = s.epsr = s.sigr = 0.0
        s.kon = 0
        s.epsP = s.sigP = 0.0

    def step(s, eps):
        Fy, E0, b = s.Fy, s.E0, s.b
        Esh, epsy = b * E0, Fy / E0
        deps = eps - s.epsP
        if s.kon in (0, 3):
            if abs(deps) < 10 * 2.2e-16:
                s.kon = 3
                return 0.0
            s.epsmax, s.epsmin = epsy, -epsy
            if deps < 0:
                s.kon, s.epss0, s.sigs0, s.epspl = 2, s.epsmin, -Fy, s.epsmin
            else:
                s.kon, s.epss0, s.sigs0, s.epspl = 1, s.epsmax, Fy, s.epsmax
        if s.kon == 2 and deps > 0:
            s.kon, s.epsr, s.sigr = 1, s.epsP, s.sigP
            s.epsmin = min(s.epsmin, s.epsP)
            d1 = (s.epsmax - s.epsmin) / (2.0 * (s.a4 * epsy))
            shft = 1.0 + s.a3 * d1 ** 0.8
            s.epss0 = (Fy * shft - Esh * epsy * shft - s.sigr + E0 * s.epsr) / (E0 - Esh)
            s.sigs0 = Fy * shft + Esh * (s.epss0 - epsy * shft)
            s.epspl = s.epsmax
        elif s.kon == 1 and deps < 0:
            s.kon, s.epsr, s.sigr = 2, s.epsP, s.sigP
            s.epsmax = max(s.epsmax, s.epsP)
            d1 = (s.epsmax - s.epsmin) / (2.0 * (s.a2 * epsy))
            shft = 1.0 + s.a1 * d1 ** 0.8
            s.epss0 = (-Fy * shft + Esh * epsy * shft - s.sigr + E0 * s.epsr) / (E0 - Esh)
            s.sigs0 = -Fy * shft + Esh * (s.epss0 + epsy * shft)
            s.epspl = s.epsmin
        xi = abs((s.epspl - s.epss0) / epsy)
        R = s.R0 * (1 - (s.cR1 * xi) / (s.cR2 + xi))
        r = (eps - s.epsr) / (s.epss0 - s.epsr)
        sig = b * r + (1 - b) * r / (1 + abs(r) ** R) ** (1 / R)
        sig = sig * (s.sigs0 - s.sigr) + s.sigr
        s.epsP, s.sigP = eps, sig
        return sig


@pytest.mark.parametrize("iso", [dict(), dict(a1=0.04, a2=1.0, a3=0.04, a4=1.0)])
def test_steel02_is_opensees_steel02_over_a_random_history(iso):
    rng = np.random.default_rng(7)
    path = np.cumsum(rng.normal(0.0, 4e-4, 3000))
    m = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01, R0=20, **iso)
    ref = _OpenSeesSteel02(400e6, 200e9, 0.01, R0=20, **iso)
    ours, _ = _follow(m, path)
    theirs = np.array([ref.step(e) for e in path])
    assert np.max(np.abs(ours - theirs)) <= 1e-6 * 400e6


def test_a_small_wobble_does_not_lose_the_envelope():
    """The old law aimed every new branch at +-Fy_eff and forgot the
    hardening envelope at every reversal: a 2e-4 wobble at 2% took the stress
    at 3% from 469.8 to 440.3 MPa. OpenSees gives the same stress with or
    without the wobble."""
    def at3(path):
        m = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01, R0=20)
        return _follow(m, path)[0][-1]
    straight = at3(np.linspace(0, 0.03, 301))
    wobble = at3(list(np.linspace(0, 0.02, 201)) + [0.0198] + list(np.linspace(0.02, 0.03, 101)))
    assert wobble == pytest.approx(straight, rel=0.01)


def test_fifty_small_cycles_do_not_walk_the_stress_down_to_Fy():
    m = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01, R0=20)
    st = m.init_state(1)
    _follow(m, np.linspace(0, 0.02, 201), st)
    for _k in range(50):
        _follow(m, [0.0198, 0.02], st)
    # the old law ended at Fy; OpenSees comes back onto the envelope
    on, _ = _follow(m, np.linspace(0.02, 0.03, 101), st)
    straight, _ = _follow(m, np.linspace(0, 0.03, 301))
    assert on[-1] == pytest.approx(straight[-1], rel=1e-3)
    assert on[-1] == pytest.approx(456e6, rel=1e-3)


def test_a_repeated_strain_leaves_stress_and_state_alone():
    m = create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01, R0=20)
    st = m.init_state(1)
    _follow(m, [0.0, 0.004, 0.001], st)
    before = st.copy()
    s1, _ = _follow(m, [0.001], st)
    assert np.array_equal(st, before)
    assert s1[-1] == before[0, 1]


def test_hardening_equal_to_one_is_hookes_law():
    m = create("steel02", E=200e9, rho=7850, Fy=400e6, b=1.0)
    s, _ = _follow(m, [0.0, 0.01, -0.01])
    assert s.tolist() == pytest.approx([0.0, 2e9, -2e9])


@pytest.mark.parametrize("kw", [dict(cR1=1.0), dict(cR2=0.0), dict(a2=0.0), dict(a4=-1.0)])
def test_parameters_that_would_break_the_law_are_refused(kw):
    with pytest.raises(ValueError):
        create("steel02", E=200e9, rho=7850, Fy=400e6, b=0.01, **kw)
