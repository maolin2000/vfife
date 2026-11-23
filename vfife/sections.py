"""Cross-section geometry: shape parameters in, integrated properties out.

The input model names sections by what they ARE -- `{shape: rect, b: 0.3,
h: 0.5}` -- and the solver needs what they DO: area, second moments, torsion
constant, shear areas, perimeter. This module is the one place that
conversion lives, so a bar's perimeter for bond and a beam's Iy come from
the same numbers the analyst wrote.

Local axes follow the beam convention of this project: y and z are the
section axes, x the member axis; `iy` bends about y (fibres vary in z),
`iz` about z.

SHAPES
    circle   d (or diameter)             (solid), d + t (tube)
    rect     b (along y), h (along z);  + t (hollow box)
    I        bf, tf, hw, tw             flanges along y, web along z
    general  area, iy, iz, j, [ay, az, perimeter]   -- given, not derived

TORSION
    Closed thin-walled sections use Bredt; the solid rectangle uses the
    Roark series truncated at the term that matters (0.3% at b = h); the
    I-section uses the open-section sum of b t^3 / 3, which is what a
    hand check gives and what a beam2 with St-Venant torsion expects.
    None of these is exact for thick walls, and the docstring says so
    rather than the answer looking exact.

SHEAR AREAS
    Timoshenko `ay`, `az` are the areas the shear force acts on: kappa * A
    with the shape factor kappa (5/6 rectangle, 0.9 circle, web area for an
    I-section), unless the analyst gives them.
"""
from __future__ import annotations

import math
from typing import Any, Dict


_DERIVED = ("area", "iy", "iz", "j", "ay", "az", "perimeter")


def properties(spec: Dict[str, Any]) -> Dict[str, float]:
    """Integrated properties of a section shape spec.

    Returns a dict with area, iy, iz, j, ay, az, perimeter. Any of those
    given explicitly in `spec` overrides the derived value, so an analyst
    with a better torsion constant can say so without switching to
    `general`.
    """
    # the case file's names as well as the formula letters
    for word, letter in (("width", "b"), ("height", "h"), ("diameter", "d")):
        if spec.get(word) is not None and spec.get(letter) is None:
            spec = {**spec, letter: spec[word]}
    shape = spec.get("shape", "general")
    if shape in ("box_girder", "t_girder"):
        from .section_shapes import properties as girder_properties
        out = girder_properties(spec)
        for key in ("area", "iy", "iz", "j", "ay", "az"):
            if spec.get(key) is not None:
                out[key] = float(spec[key])
        return out
    fn = _SHAPES.get(shape)
    if fn is None:
        raise ValueError(f"unknown section shape {shape!r}; one of "
                         + ", ".join(sorted(_SHAPES)))
    out = fn(spec)
    for key in _DERIVED:
        if spec.get(key) is not None:
            out[key] = float(spec[key])
    for key in ("area", "iy", "iz", "j"):
        if out.get(key) is not None and out[key] <= 0.0:
            raise ValueError(f"section {shape!r}: {key} came out {out[key]:.4g}; "
                             "check the dimensions")
    return out


def _need(spec, *keys):
    missing = [k for k in keys if spec.get(k) is None]
    if missing:
        # the case file's names, not the formula letters
        said = {"b": "width", "h": "height", "d": "diameter"}
        raise ValueError(f"section shape {spec.get('shape')!r} needs "
                         + ", ".join(said.get(k, k) for k in missing))
    return [float(spec[k]) for k in keys]


def _circle(spec):
    if spec.get("d") is None and spec.get("diameter") is not None:
        spec = {**spec, "d": spec["diameter"]}
    (d,) = _need(spec, "d")
    t = spec.get("t")
    if t is None:
        r = d / 2.0
        area = math.pi * r * r
        i = math.pi * r ** 4 / 4.0
        return {"area": area, "iy": i, "iz": i, "j": 2.0 * i,
                "ay": 0.9 * area, "az": 0.9 * area, "perimeter": math.pi * d}
    t = float(t)
    if not 0.0 < t < d / 2.0:
        raise ValueError(f"circle: wall t = {t:.4g} must lie in (0, d/2)")
    ro, ri = d / 2.0, d / 2.0 - t
    area = math.pi * (ro * ro - ri * ri)
    i = math.pi * (ro ** 4 - ri ** 4) / 4.0
    # thin-tube shear factor 0.5; exact for t << d, 3% high at t = 0.1 d
    return {"area": area, "iy": i, "iz": i, "j": 2.0 * i,
            "ay": 0.5 * area, "az": 0.5 * area, "perimeter": math.pi * d}


def _rect(spec):
    b, h = _need(spec, "b", "h")
    t = spec.get("t")
    if t is None:
        area = b * h
        iy = b * h ** 3 / 12.0          # bending about y: fibres vary along z (h)
        iz = h * b ** 3 / 12.0
        # Roark: J = a b^3 (16/3 - 3.36 b/a (1 - b^4/(12 a^4))) / 16, a >= b halves
        a2, b2 = max(b, h) / 2.0, min(b, h) / 2.0
        j = a2 * b2 ** 3 * (16.0 / 3.0 - 3.36 * (b2 / a2) * (1.0 - b2 ** 4 / (12.0 * a2 ** 4)))
        return {"area": area, "iy": iy, "iz": iz, "j": j,
                "ay": 5.0 / 6.0 * area, "az": 5.0 / 6.0 * area,
                "perimeter": 2.0 * (b + h)}
    t = float(t)
    if not 0.0 < t < min(b, h) / 2.0:
        raise ValueError(f"rect: wall t = {t:.4g} must lie in (0, min(b, h)/2)")
    bi, hi = b - 2.0 * t, h - 2.0 * t
    area = b * h - bi * hi
    iy = (b * h ** 3 - bi * hi ** 3) / 12.0
    iz = (h * b ** 3 - hi * bi ** 3) / 12.0
    # Bredt for the closed thin wall on the mid-line
    am = (b - t) * (h - t)
    j = 4.0 * am * am * t / (2.0 * ((b - t) + (h - t)))
    # shear carried by the two walls parallel to the force
    return {"area": area, "iy": iy, "iz": iz, "j": j,
            "ay": 2.0 * t * (b - 2.0 * t), "az": 2.0 * t * (h - 2.0 * t),
            "perimeter": 2.0 * (b + h)}


def _i_section(spec):
    bf, tf, hw, tw = _need(spec, "bf", "tf", "hw", "tw")
    # hw is the clear web height between flanges; total depth h = hw + 2 tf
    h = hw + 2.0 * tf
    area = 2.0 * bf * tf + hw * tw
    iy = (bf * h ** 3 - (bf - tw) * hw ** 3) / 12.0
    iz = (2.0 * tf * bf ** 3 + hw * tw ** 3) / 12.0
    j = (2.0 * bf * tf ** 3 + hw * tw ** 3) / 3.0
    return {"area": area, "iy": iy, "iz": iz, "j": j,
            "ay": 5.0 / 3.0 * bf * tf,     # flanges carry the y-shear (2 * 5/6 * bf tf)
            "az": hw * tw,                  # the web carries the z-shear
            "perimeter": 4.0 * bf - 2.0 * tw + 4.0 * tf + 2.0 * hw}


def _general(spec):
    out = {k: (None if spec.get(k) is None else float(spec[k])) for k in _DERIVED}
    return out


_SHAPES = {"circle": _circle, "rect": _rect, "I": _i_section, "general": _general}
