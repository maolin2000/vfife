"""Defaults that carry a unit, in the units the case declares.

vfife converts nothing a case WRITES: every number is in the one consistent
system the case declares (`units:`). A DEFAULT is different -- it is a number
the user never wrote, so it cannot be "in the user's system" unless it is put
there. Before v1.0 each was an SI constant whatever the case said: in a
mm / N / MPa case a vehicle weighed under g = 9.80665 mm/s^2 (a thousandth of
its weight, with no warning), the water around a pier had a density of 1025
t/mm^3, a stirrup modulus of 2e11 MPa.

`fill_defaults(d)` writes each such default into a COPY of the case dict,
converted from SI to the declared units, wherever the case leaves it out. A
case that declares no units, or SI, is returned unchanged (the defaults are
SI). The consistent mass unit is force x time^2 / length (a tonne for
mm / N / s), the stress unit force / length^2.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

#: how many metres / newtons / seconds one declared unit is
LENGTH_IN_M = {"m": 1.0, "metre": 1.0, "meter": 1.0,
               "mm": 1.0e-3, "millimetre": 1.0e-3, "millimeter": 1.0e-3,
               "cm": 1.0e-2, "km": 1.0e3,
               "in": 0.0254, "inch": 0.0254, "ft": 0.3048, "foot": 0.3048}
FORCE_IN_N = {"n": 1.0, "kn": 1e3, "mn": 1e6, "lbf": 4.4482216, "kip": 4448.2216}
TIME_IN_S = {"s": 1.0, "sec": 1.0, "second": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0}

#: dimension -> (exponents of length, force, time)
DIMENSIONS = {
    "length": (1, 0, 0),
    "per_length": (-1, 0, 0),
    "acceleration": (1, 0, -2),
    "stress": (-2, 1, 0),
    "density": (-4, 1, 2),
}

#: the material laws' unit-carrying defaults: {law: [(parameter, SI default,
#: dimension, names that also give it)]}. A parameter given under any of its
#: names is the user's and is left alone.
MATERIAL_DEFAULTS = {
    "concrete01": [("rho", 2400.0, "density", ()), ("fpc", 30.0e6, "stress", ("fc",))],
    "concrete02": [("rho", 2400.0, "density", ()), ("fpc", 30.0e6, "stress", ("fc",))],
    "steel02": [("rho", 7850.0, "density", ()), ("E", 200.0e9, "stress", ("E0",)),
                ("Fy", 400.0e6, "stress", ("yield_strength", "fy"))],
    "steel_cyclic": [("rho", 7850.0, "density", ()), ("E", 206.0e9, "stress", ())],
    "bond_ceb": [("rho", 7850.0, "density", ())],
    # the same MC2010 envelope with cyclic degradation (BondCyclic inherits
    # BondCEB's defaults): converted alike, or switching law changed the bond
    # a thousandfold in a mm case (1.0 re-review)
    "bond_cyclic": [("rho", 7850.0, "density", ())],
}

#: the bond laws whose peak and slips default from MC2010 in SI
BOND_LAWS = ("bond_ceb", "bond_cyclic")


#: how many pascals one declared `units.stress` is
STRESS_IN_PA = {"pa": 1.0, "kpa": 1e3, "mpa": 1e6, "gpa": 1e9, "psi": 6894.757, "ksi": 6894757.0,
                "n/m2": 1.0, "n/mm2": 1e6}


def _named(table, value):
    """The first name in `table` for an SI factor (None when there is none)."""
    for name, v in table.items():
        if abs(v - value) <= 1e-9 * abs(value):
            return name
    return None


def complete(units: Optional[Dict[str, Any]]) -> Tuple[Dict[str, str], List[str], List[str]]:
    """(the declaration with what it implies filled in, [inferred keys],
    [missing keys]) (1.0 re-review).

    The system is length, force and time; the stress unit is force/length^2.
    A declaration with `stress` but without `length` or `force` names the
    third through the other two: `{force: N, stress: MPa}` is mm, `{length:
    m, stress: kPa}` is kN. It used to be read as the SI default for the key
    left out -- m beside a declared MPa -- so nothing was converted and the
    defaults went in as SI into a mm model, with preflight PASS. When the
    two do not imply a unit this knows (`{force: kN, stress: MPa}` is a
    31.6 mm length), or `stress` stands alone, the key is MISSING: preflight
    refuses the case (UNITS_INCOMPLETE) -- except `stress: Pa` alone, which is
    m and N. Without `stress`, a key left out is the SI one (m, N, s), as
    before."""
    units = dict(units or {})
    out = {k: str(v) for k, v in units.items()}
    inferred: List[str] = []
    missing: List[str] = []
    S = STRESS_IN_PA.get(str(units.get("stress", "")).lower()) if "stress" in units else None
    if S is not None:
        has_L, has_F = "length" in units, "force" in units
        if has_F and not has_L:
            F = FORCE_IN_N.get(str(units["force"]).lower())
            name = _named(LENGTH_IN_M, (F / S) ** 0.5) if F else None
            if name:
                out["length"] = name; inferred.append("length")
            else:
                missing.append("length")
        elif has_L and not has_F:
            L = LENGTH_IN_M.get(str(units["length"]).lower())
            name = _named(FORCE_IN_N, S * L * L) if L else None
            if name:
                out["force"] = {"n": "N", "kn": "kN", "mn": "MN"}.get(name, name); inferred.append("force")
            else:
                missing.append("force")
        elif not has_L and not has_F and abs(S - 1.0) > 1e-12:
            # `stress: Pa` alone is the SI pair m and N; any other stress
            # alone names neither
            missing += ["length", "force"]
    return out, inferred, missing


def mass_unit(units) -> str:
    """The name of the consistent mass unit (force x time^2 / length): kg for
    N / m / s, t for N / mm / s and kN / m / s; "" when it has none here."""
    sc = scales(units)
    if sc.get("density") is None or sc.get("length") is None:
        return ""
    m = sc["density"] * sc["length"] ** 3
    return {1.0: "kg", 1.0e3: "t", 1.0e-3: "g"}.get(
        next((k for k in (1.0, 1.0e3, 1.0e-3) if abs(m - k) <= 1e-9 * k), None), "")


def scales(units: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """SI value of one case unit of each dimension, on the declaration as
    `complete` fills it in (a length or force named by `stress` and the
    other). A key the declaration leaves out is otherwise the SI one (m, N,
    s); None where it names a unit this does not know (`force: tonf`) or is
    missing (`complete`), so that dimension cannot be converted."""
    units, _inferred, missing = complete(units)
    L = (None if "length" in missing else
         LENGTH_IN_M.get(str(units.get("length", "")).lower()) if "length" in units else 1.0)
    F = (None if "force" in missing else
         FORCE_IN_N.get(str(units.get("force", "")).lower()) if "force" in units else 1.0)
    T = TIME_IN_S.get(str(units.get("time", "s")).lower())
    out = {}
    for dim, (a, b, c) in DIMENSIONS.items():
        parts = [(L, a), (F, b), (T, c)]
        if any(v is None and e for v, e in parts):
            out[dim] = None
        else:
            s = 1.0
            for v, e in parts:
                if e:
                    s *= v ** e
            out[dim] = s
    return out


def is_si(units) -> bool:
    return all(s is not None and abs(s - 1.0) < 1e-12 for s in scales(units).values())


def to_case(value_si: float, dim: str, units) -> Optional[float]:
    """An SI value in the case's units (None when the declaration does not fix
    the unit of `dim`)."""
    s = scales(units)[dim]
    return None if s is None else float(value_si) / s


def fill_defaults(d: Dict[str, Any]) -> Tuple[Dict[str, Any], List[tuple]]:
    """(the case dict with every unit-carrying default written in the declared
    units, [(where, value, dimension)] of what was written). Returns `d`
    itself, untouched, when the case is in SI or declares no units."""
    units = d.get("units") if isinstance(d, dict) else None
    if not isinstance(units, dict) or not units or is_si(units):
        return d, []
    sc = scales(units)
    out = copy.deepcopy(d)
    filled: List[tuple] = []

    def put(block, key, value_si, dim, where):
        if not isinstance(block, dict) or block.get(key) is not None:
            return
        if sc.get(dim) is None:
            filled.append((where, None, dim))       # not convertible: SI default kept
            return
        block[key] = float(value_si) / sc[dim]
        filled.append((where, block[key], dim))

    def law(props, where):
        if not isinstance(props, dict):
            return
        kind = props.get("type")
        for key, si, dim, others in MATERIAL_DEFAULTS.get(kind, ()):
            if any(props.get(o) is not None for o in others):
                continue
            put(props, key, si, dim, f"{where}.{key}")
        if kind == "steel_cyclic" and props.get("grade") is not None:
            from ..materials.steel_cyclic import GRADES
            table = GRADES.get(props["grade"], {})
            for key in ("fy", "fu"):
                if key in table:
                    put(props, key, table[key], "stress", f"{where}.{key}")
        if kind in BOND_LAWS:
            _bond_defaults(props, where, sc, filled)
        if kind in ("parallel", "series"):
            # a part written in place takes the defaults a named one does
            # (found in review: an inline steel02 in a mm case kept 7850
            # and 400e6 -- 1e12 too heavy, and it never yielded)
            for i, part in enumerate(props.get("of") or ()):
                law(part, f"{where}.of[{i}]")

    for name, props in (out.get("materials") or {}).items():
        law(props, f"materials.{name}")
    for i, a in enumerate(out.get("assign") or []):
        if isinstance(a, dict) and isinstance(a.get("bond"), dict):
            b = a["bond"]
            if b.get("type", "bond_ceb") in BOND_LAWS:
                b2 = dict(b, type=b.get("type", "bond_ceb"))
                law(b2, f"assign[{i}].bond")
                b2.pop("type")
                if "type" in b:
                    b2["type"] = b["type"]
                a["bond"] = b2
    for name, sec in (out.get("sections") or {}).items():
        shear = sec.get("shear") if isinstance(sec, dict) else None
        if isinstance(shear, dict) and isinstance(shear.get("stirrups"), dict):
            put(shear["stirrups"], "E", 2.0e11, "stress", f"sections.{name}.shear.stirrups.E")
    for i, h in enumerate(out.get("hydrodynamic_mass") or []):
        put(h, "density", 1025.0, "density", f"hydrodynamic_mass[{i}].density")
    loads = [(f"loads[{i}]", ld) for i, ld in enumerate(out.get("loads") or [])]
    for s, st in enumerate(out.get("steps") or []):
        if isinstance(st, dict):
            loads += [(f"steps[{s}].loads[{i}]", ld) for i, ld in enumerate(st.get("loads") or [])]
    for where, ld in loads:
        v = ld.get("vehicle") if isinstance(ld, dict) else None
        if not isinstance(v, dict):
            continue
        put(v, "gravity", 9.80665, "acceleration", f"{where}.vehicle.gravity")
        psd = (v.get("roughness") or {}).get("psd") if isinstance(v.get("roughness"), dict) else None
        if isinstance(psd, dict):
            for key, si in (("n0", 0.1), ("n_min", 0.01), ("n_max", 3.0)):
                put(psd, key, si, "per_length", f"{where}.vehicle.roughness.psd.{key}")
    return out, filled


def _bond_defaults(props, where, sc, filled):
    """MC2010's bond law is written in MPa and mm (and here in Pa and m):
    its peak from fc and its slips are found in SI and brought back."""
    if sc.get("stress") is None or sc.get("length") is None:
        return
    keys = ("tau_max", "s1", "s2", "s3")
    if all(props.get(k) is not None for k in keys):
        return
    from ..materials.bond import BondCEB
    args = {k: props[k] for k in ("condition", "failure_mode", "confinement", "tau_max_coeff",
                                  "alpha", "tau_f_ratio") if props.get(k) is not None}
    if props.get("fc") is not None:
        args["fc"] = float(props["fc"]) * sc["stress"]
    if props.get("tau_max") is not None:
        args["tau_max"] = float(props["tau_max"]) * sc["stress"]
    for k in ("s1", "s2", "s3"):
        if props.get(k) is not None:
            args[k] = float(props[k]) * sc["length"]
    try:
        si = BondCEB(**args)
    except Exception:
        return                                   # the law itself will say what is wrong
    for k, dim in (("tau_max", "stress"), ("s1", "length"), ("s2", "length"), ("s3", "length")):
        if props.get(k) is None:
            props[k] = getattr(si, k) / sc[dim]
            filled.append((f"{where}.{k}", props[k], dim))
