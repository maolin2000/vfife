"""The bar's length scale: a segment length, and a strain at which it breaks.

WHY THIS FILE EXISTS
Embedded bars are perfectly bonded, so a crack of opening w crossing a segment
of length L puts strain w/L into that segment with nothing to spread it. The
segment length is therefore physical, and it was the hidden variable behind a
long investigation: refining the solid RC column's mesh from 100 to 50 mm while
letting `divisions` follow it halved the drift the model survived, 32.0 -> 18.0
mm, at a bar plastic strain that stayed put (4.83% and 4.67%). Holding the
segments at 100 mm and refining only the concrete put the drift back to 32.0
exactly. `divisions` is a count, and a count is what people scale with a mesh.

And the bar had no end: `rebar` hardened for ever, so nothing stopped those
strains at 19% either.
"""
import numpy as np
import pytest

from vfife.elements.truss import bars_from_spec
from vfife.materials import create


# --------------------------------------------------------------------------- #
#  segment_length
# --------------------------------------------------------------------------- #

def _line(**kw):
    return {"lines": [dict(start=[0.0, 0.0, 0.0], end=[0.0, 0.0, 1.6], **kw)]}


def test_a_segment_length_gives_the_segments_that_length_implies():
    _, conn = bars_from_spec(_line(segment_length=0.1))
    assert len(conn) == 16


def test_a_segment_length_that_does_not_divide_evenly_rounds():
    """1.6 / 0.209 is 7.66, and 8 segments of 200 mm is the nearest whole
    answer. Rounding is right here: the input is a physical smearing length,
    known to maybe 10%, not a number that has to divide the member."""
    _, conn = bars_from_spec(_line(segment_length=0.209))
    assert len(conn) == 8


def test_segment_length_and_divisions_together_are_refused():
    """Two different answers to the same question; one would silently win."""
    with pytest.raises(ValueError, match="not both"):
        bars_from_spec(_line(divisions=16, segment_length=0.1))


def test_an_unset_division_count_still_means_one_segment():
    """The schema emits `divisions: None` for an unset field, so the key is
    present and a get(..., 1) default never fires. int(None) would raise."""
    _, conn = bars_from_spec(_line(divisions=None, segment_length=None))
    assert len(conn) == 1


def test_segment_length_survives_the_repeat_form():
    """Stirrups are written as one leg plus a spacing; the two features have to
    compose, because transverse steel is exactly where a short segment matters."""
    spec = {"lines": [{"start": [0.0, 0.0, 0.1], "end": [0.4, 0.0, 0.1],
                       "segment_length": 0.1,
                       "repeat": {"count": 3, "offset": [0.0, 0.0, 0.1]}}]}
    _, conn = bars_from_spec(spec)
    assert len(conn) == 4 * 3


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_a_non_positive_segment_length_is_refused(bad):
    with pytest.raises(ValueError, match="segment_length"):
        bars_from_spec(_line(segment_length=bad))


# --------------------------------------------------------------------------- #
#  rupture
# --------------------------------------------------------------------------- #

def _bar(rupture=None):
    return create("rebar", E=2.0e11, rho=7850.0, yield_strength=4.74e8,
                  hardening=1.0e9, kinematic_fraction=0.5,
                  rupture_strain=rupture)


def _pull(mat, eps_max=0.08, n=400):
    st = mat.init_state(1)
    out = []
    for e in np.linspace(0.0, eps_max, n):
        x = np.zeros((1, 6))
        x[0, 0] = e
        out.append(float(mat.update(x, st, 1.0)[0, 0]))
    return np.array(out), st


def test_without_a_rupture_strain_the_bar_hardens_for_ever():
    """The behaviour every existing case relies on, pinned so the new parameter
    cannot change it by default."""
    sig, _ = _pull(_bar())
    assert sig[-1] > 4.74e8
    assert np.all(np.diff(sig) > 0)


def test_past_the_rupture_strain_the_bar_carries_nothing():
    sig, _ = _pull(_bar(rupture=0.02))
    assert sig[-1] == 0.0
    assert sig.max() > 4.74e8            # it did carry load before it broke


def test_rupture_is_irreversible():
    """Measured on accumulated plastic strain, which only grows -- so a broken
    bar cannot come back on an elastic step. Unloading after rupture must still
    give zero."""
    mat = _bar(rupture=0.02)
    st = mat.init_state(1)
    for e in list(np.linspace(0.0, 0.06, 200)) + list(np.linspace(0.06, 0.0, 200)):
        x = np.zeros((1, 6))
        x[0, 0] = e
        s = float(mat.update(x, st, 1.0)[0, 0])
    assert s == 0.0


def test_rupture_counts_accumulated_plastic_strain_not_peak_strain():
    """Cycling accumulates it, so a cycled bar breaks earlier than a pulled one.
    That is the documented meaning and it is worth pinning, because reading the
    parameter as a tension-test eps_u would be wrong in the unsafe direction."""
    limit = 0.02
    pulled = _bar(rupture=limit)
    st = pulled.init_state(1)
    for e in np.linspace(0.0, 0.012, 200):        # peak strain below the limit
        x = np.zeros((1, 6)); x[0, 0] = e
        s_pull = float(pulled.update(x, st, 1.0)[0, 0])
    assert s_pull > 0.0                            # a monotonic pull survives

    cycled = _bar(rupture=limit)
    st = cycled.init_state(1)
    for _ in range(4):
        for e in list(np.linspace(0.0, 0.012, 60)) + list(np.linspace(0.012, -0.012, 120)):
            x = np.zeros((1, 6)); x[0, 0] = e
            s_cyc = float(cycled.update(x, st, 1.0)[0, 0])
    assert s_cyc == 0.0                            # the same peaks, cycled, do not


@pytest.mark.parametrize("bad", [0.0, -0.01])
def test_a_non_positive_rupture_strain_is_refused(bad):
    with pytest.raises(ValueError, match="rupture_strain"):
        _bar(rupture=bad)


def test_the_compiled_twin_agrees_with_the_reference_through_rupture():
    """Three implementations carry this law -- numpy, numba and warp -- and a
    rupture that fires in one and not another is a silent backend split."""
    pytest.importorskip("numba")
    for rup in (None, 0.02):
        ref, fast = _bar(rup), _bar(rup)
        s_ref, s_fast = ref.init_state(1), fast.init_state(1)
        point = fast.numba_point_fn()
        path = np.concatenate([np.linspace(0, 0.03, 150),
                               np.linspace(0.03, -0.01, 100),
                               np.linspace(-0.01, 0.06, 200)])
        worst = 0.0
        for e in path:
            x = np.zeros((1, 6)); x[0, 0] = e
            a = float(ref.update(x, s_ref, 1.0)[0, 0])
            out = np.zeros(6)
            point(np.array([e, 0.0, 0.0, 0.0, 0.0, 0.0]), s_fast[0], 1.0, 1.0, out)
            worst = max(worst, abs(a - out[0]) / max(abs(a), 1.0e3))
        assert worst < 1e-12, f"rupture={rup} split by {worst:.2e}"


def test_the_warp_parameter_block_carries_the_rupture_limit():
    """The warp twin reads prm[5]; a params array that forgot it would leave the
    GPU path hardening for ever while the CPU path broke."""
    prm, _ = _bar(rupture=0.037).warp_params()
    assert len(prm) == 6
    assert prm[5] == pytest.approx(0.037)
    prm_off, _ = _bar().warp_params()
    assert prm_off[5] == 0.0             # 0 means off, and alpha is never < 0
