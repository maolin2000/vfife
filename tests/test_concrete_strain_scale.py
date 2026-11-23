"""`concrete02` takes a share of the section's strain.

The exact counterpart of `steel02.strain_scale` -- same mechanism, different
reason. There it is the anchorage taking part of a bar's elongation; here it
is CREEP. The law responds to the INSTANTANEOUS strain, and a time station
that must carry the creep of the interval it is arriving in solves

    eps_el = (eps_section - eps_free) / (1 + phi_cur)

with `eps_free` in the fibre's imposed-strain slot and this factor the
`1 / (1 + phi_cur)`. Writing the station's own creep as a SCALE rather than
folding it into the imposed strain is what makes the step implicit; v0.4
measured the explicit version to be marginally unstable on a held strain.

The defining property is the same one `tests/test_strain_penetration.py`
pins for steel: a material with `strain_scale = k` must answer at `eps`
exactly what an unscaled one answers at `k eps`, over a full path and not at
one point, because both laws are path dependent.
"""
import numpy as np
import pytest

from vfife.materials.base import create

PROPS = dict(fpc=3.0e+7, epsc0=0.002, fcu=6.0e+6, epscu=0.004,
             ft=3.0e+6, Ets=1.5e+9, rat=0.1, rho=2400)


def _path():
    """Compression past the peak, unload through zero into cracking, back."""
    p = [0.0]
    for peak in (-5.0e-4, -2.2e-3, 3.0e-4, -3.5e-3):
        p += list(np.linspace(p[-1], peak, 40))[1:]
    return np.array(p)


def _walk(material, strains):
    st = material.init_state(1)
    return np.array([float(material.update(np.array([[e]]), st, 1.0)[0, 0])
                     for e in strains])


@pytest.mark.parametrize("k", [0.25, 0.5, 0.8, 0.95])
def test_a_scaled_fibre_answers_at_eps_what_a_plain_one_answers_at_k_eps(k):
    path = _path()
    plain = _walk(create("concrete02", **PROPS), k * path)
    scaled = _walk(create("concrete02", strain_scale=k, **PROPS), path)
    scale = max(float(np.max(np.abs(plain))), 1.0)
    worst = float(np.max(np.abs(scaled - plain))) / scale
    assert worst < 1e-12, f"the two paths differ by {worst:.2e}"


def test_the_default_changes_nothing():
    path = _path()
    a = _walk(create("concrete02", **PROPS), path)
    b = _walk(create("concrete02", strain_scale=1.0, **PROPS), path)
    assert np.array_equal(a, b)


def test_a_scale_softens_the_fibre_in_proportion_while_it_is_linear():
    """A scale k IS a modulus k E0 wherever the law is linear -- which is what
    makes `1 / (1 + phi_cur)` the effective modulus of the station.

    IN TENSION, not compression: the compression branch is the parabola
    `fc r (2 - r)`, whose initial SLOPE is E0 but which is not proportional at
    any finite strain. The tension branch is straight to `ft / E0 = 1e-4`.
    """
    e = 5.0e-5
    full = _walk(create("concrete02", **PROPS), [e])[0]
    half = _walk(create("concrete02", strain_scale=0.5, **PROPS), [e])[0]
    assert full == pytest.approx(e * 2.0 * PROPS["fpc"] / PROPS["epsc0"], rel=1e-12)
    assert half == pytest.approx(0.5 * full, rel=1e-12)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_a_scale_outside_its_range_is_refused(bad):
    with pytest.raises(ValueError, match="strain_scale must lie"):
        create("concrete02", strain_scale=bad, **PROPS)


def test_the_compiled_and_reference_paths_agree_with_a_scale():
    """The fused fibre-beam kernel inlines the compiled point function, so a
    scale that only one of the two applied would be invisible until a case
    ran -- and it would look like a creep result."""
    m = create("concrete02", strain_scale=0.6, **PROPS)
    path = _path()
    n = 8
    fast_state, ref_state = m.init_state(n), m.init_state(n)
    strain = np.zeros((n, 6))
    worst = 0.0
    for e in path:
        strain[:, 0] = e * (1.0 + 0.05 * np.arange(n) / n)
        fast = m.update(strain, fast_state, 1.0)[:, 0].copy()
        ref = m._update_reference(strain, ref_state, np.zeros((n, 6)))[:, 0]
        worst = max(worst, float(np.max(np.abs(fast - ref))))
    assert worst == 0.0, f"compiled and reference disagree by {worst:.3e} Pa"
