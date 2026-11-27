"""`Gc` on the compression curve: the crushing band, regularised.

WHY
    The tension curve has been built from a fracture energy over a reference
    length since the crack band was written, so the energy dissipated per
    unit CRACK AREA does not depend on the element size. The compression
    curve had no such thing: a code curve (GB 50010) is a strain curve, and
    crushing localises into one element row whatever its size, so the energy
    that row dissipates per unit area is the curve's energy per volume times
    the element size -- 9.6 N/mm at 75 mm and 2.4 N/mm at 18.75 mm for the
    Alberta beam's concrete.

    Measured on that beam, refining only the depth (4 / 8 / 12 / 16 rows,
    everything else identical): the peak load fell 334 / 330 / 321 / 300 kN
    and the descent steepened, while the pre-peak branch did not move.
    Not confinement, not the load line, not spalling -- the
    compression band had no length scale.

    `Gc` (J/m2) is that length scale: the post-peak branch is stretched so a
    band `l_ref` thick dissipates Gc per unit area, and the crack-band factor
    k = l_e/l_ref then rescales it per element, exactly as on the tension
    side. Only the post-peak branch moves (Feenstra & de Borst).
"""
import numpy as np
import pytest

from vfife.materials import create
from vfife.materials import curves as cv
from vfife.base.compat import trapezoid

FC, E0, L_REF, GC = 44.0e6, 2.985e10, 0.0375, 35.0e3


def _dissipated(curve, l_e, l_ref=L_REF):
    """Energy per unit crushed AREA an element of size `l_e` dissipates."""
    k = l_e / l_ref
    ep = np.linspace(0.0, curve.ep[-1] / k, 8000)
    sig, _, d = curve.evaluate(ep, k)
    cauchy = sig * (1.0 - d)
    i0 = int(np.argmax(cauchy))
    g = float(trapezoid(cauchy[i0:], ep[i0:]))
    return g * l_e


def test_without_Gc_the_energy_per_area_scales_with_the_element_size():
    """The state of every version before this one, measured rather than
    asserted -- and the reason Gc exists."""
    plain = cv.gb50010_compression(fc=FC, E0=E0)
    at = {l: _dissipated(plain, l, l_ref=l) for l in (0.075, 0.0375, 0.01875)}
    assert at[0.075] / at[0.0375] == pytest.approx(2.0, rel=0.02)
    assert at[0.0375] / at[0.01875] == pytest.approx(2.0, rel=0.02)


def test_with_Gc_the_energy_per_area_is_the_same_at_every_element_size():
    c = cv.gb50010_compression(fc=FC, E0=E0, Gc=GC, l_ref=L_REF)
    at = [_dissipated(c, l) for l in (0.075, 0.0375, 0.01875)]
    assert max(at) / min(at) < 1.001
    # and it IS Gc, to the few percent the elastic recovery inside the
    # Cauchy integral and the no-expansion cap account for
    assert at[1] == pytest.approx(GC, rel=0.05)


def test_the_ascending_branch_is_untouched():
    """Only the post-peak branch is regularised: the peak stress, the strain
    it occurs at and the initial stiffness are the code's."""
    plain = cv.gb50010_compression(fc=FC, E0=E0)
    reg = cv.gb50010_compression(fc=FC, E0=E0, Gc=GC, l_ref=L_REF)
    ep = np.linspace(0.0, plain.ep[int(np.argmax(plain.sig * (1 - plain.dmg)))], 50)
    for e in ep:
        a = [x[0] if np.ndim(x) else x for x in plain.evaluate(np.array([e]), 1.0)]
        b = [x[0] if np.ndim(x) else x for x in reg.evaluate(np.array([e]), 1.0)]
        assert a[0] == pytest.approx(b[0], rel=1e-9)
        assert a[2] == pytest.approx(b[2], rel=1e-9)
    assert reg.sigma0 == pytest.approx(plain.sigma0, rel=1e-12)


def test_a_bigger_Gc_is_a_longer_descent():
    soft = cv.gb50010_compression(fc=FC, E0=E0, Gc=2 * GC, l_ref=L_REF)
    hard = cv.gb50010_compression(fc=FC, E0=E0, Gc=GC, l_ref=L_REF)
    # 2.04, not 2.000: the elastic recovery inside the Cauchy integral is a
    # fixed subtraction that the doubled branch pays once over twice the path
    assert _dissipated(soft, L_REF) == pytest.approx(2 * _dissipated(hard, L_REF), rel=0.05)
    assert soft.ep[-1] > 1.9 * hard.ep[-1]


def _cdp(**over):
    kw = dict(E=E0, nu=0.2, rho=2400.0, crack_band="both", l_ref=L_REF,
              compression={"source": "gb50010", "fc": FC, "Gc": GC},
              tension={"source": "fracture_energy", "ft": 3.42e6, "Gf": 140.0})
    kw.update(over)
    return create("cdp", **kw)


def test_the_material_passes_its_reference_length_to_the_curve():
    m = _cdp()
    assert _dissipated(m.curve_c, L_REF) == pytest.approx(GC, rel=0.05)


def test_Gc_without_the_compression_crack_band_is_refused():
    """The curve would be built on l_ref and never rescaled, so it would be
    right only where the element size happens to equal l_ref."""
    with pytest.raises(ValueError, match="crack_band: both"):
        _cdp(crack_band="tension")
