"""sections.properties: shape in, integrated properties out, against hand values."""
import math

import pytest

from vfife.sections import properties


def test_solid_circle():
    p = properties({"shape": "circle", "diameter": 0.016})
    r = 0.008
    assert p["area"] == pytest.approx(math.pi * r * r)
    assert p["iy"] == pytest.approx(math.pi * r ** 4 / 4) == p["iz"]
    assert p["j"] == pytest.approx(math.pi * r ** 4 / 2)
    assert p["perimeter"] == pytest.approx(math.pi * 0.016)


def test_tube():
    p = properties({"shape": "circle", "d": 0.2, "t": 0.01})
    ro, ri = 0.1, 0.09
    assert p["area"] == pytest.approx(math.pi * (ro ** 2 - ri ** 2))
    assert p["j"] == pytest.approx(math.pi * (ro ** 4 - ri ** 4) / 2)


def test_rectangle_second_moments_and_roark_torsion():
    p = properties({"shape": "rect", "b": 0.3, "h": 0.5})
    assert p["area"] == pytest.approx(0.15)
    assert p["iy"] == pytest.approx(0.3 * 0.5 ** 3 / 12)      # about y: depth h
    assert p["iz"] == pytest.approx(0.5 * 0.3 ** 3 / 12)
    # Roark for a square: J = 0.1406 a^4 (a = side); series gives 0.1406 to 0.3%
    sq = properties({"shape": "rect", "b": 0.2, "h": 0.2})
    assert sq["j"] == pytest.approx(0.1406 * 0.2 ** 4, rel=0.005)
    assert p["ay"] == pytest.approx(5 / 6 * 0.15)


def test_hollow_box_uses_bredt():
    p = properties({"shape": "rect", "b": 0.3, "h": 0.5, "t": 0.01})
    assert p["area"] == pytest.approx(0.3 * 0.5 - 0.28 * 0.48)
    am = 0.29 * 0.49
    assert p["j"] == pytest.approx(4 * am * am * 0.01 / (2 * (0.29 + 0.49)))


def test_i_section_against_hand_values():
    # a UB-like section: bf 200, tf 12, hw 400 (clear), tw 8 -> h 424 mm
    p = properties({"shape": "I", "bf": 0.2, "tf": 0.012, "hw": 0.4, "tw": 0.008})
    assert p["area"] == pytest.approx(2 * 0.2 * 0.012 + 0.4 * 0.008)
    assert p["iy"] == pytest.approx((0.2 * 0.424 ** 3 - 0.192 * 0.4 ** 3) / 12)
    assert p["iz"] == pytest.approx((2 * 0.012 * 0.2 ** 3 + 0.4 * 0.008 ** 3) / 12)
    assert p["j"] == pytest.approx((2 * 0.2 * 0.012 ** 3 + 0.4 * 0.008 ** 3) / 3)
    assert p["az"] == pytest.approx(0.4 * 0.008)


def test_general_passes_values_through_and_explicit_values_override():
    p = properties({"shape": "general", "area": 0.01, "iy": 1e-4, "iz": 1e-4, "j": 2e-4})
    assert p["area"] == 0.01 and p["j"] == 2e-4 and p["ay"] is None
    q = properties({"shape": "rect", "b": 0.3, "h": 0.5, "j": 1.234e-3})
    assert q["j"] == 1.234e-3 and q["area"] == pytest.approx(0.15)


@pytest.mark.parametrize("spec, match", [
    ({"shape": "hexagon"}, "unknown section shape"),
    ({"shape": "rect", "b": 0.3}, "needs h"),
    ({"shape": "circle", "d": 0.2, "t": 0.15}, "must lie in"),
    ({"shape": "rect", "b": 0.3, "h": -0.5}, "came out"),
])
def test_bad_shapes_say_what_is_wrong(spec, match):
    with pytest.raises(ValueError, match=match):
        properties(spec)
