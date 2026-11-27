"""Section properties and material weights of a Midas model (see the package's docstring)."""
from __future__ import annotations

import math
import re
from typing import Dict, Optional, Tuple


from .. import midas

G = 9.80665
#: weight density (N/m^3) of a grade the file names without giving its values
_GRADE_WEIGHT = (("strand", 78.5e3), ("wire", 78.5e3), ("q", 76.98e3), ("c", 25.0e3))


def section_properties(sec: midas.Section, length: float, t: float = 0.5) -> Dict[str, float]:
    """{area, iy, iz, j} in m^2 and m^4 of a Midas section, with local z
    the depth direction (Midas H) and local y the width; `note` when the
    numbers are an approximation. `length` is metres per file length unit.

    Properties the file STATES are taken as stated (1.15): a
    VALUE section's, and what `*SECT-PSCVALUE` gives for a PSC, a general or
    a composite section -- Midas' own AREA, Ixx (torsion), Iyy, Izz. A
    TAPERED section is one prismatic member here: its properties at the
    fraction `t` of the way from its I end to its J end, linear between the
    two ends' (the member's mid-length when it runs over one element)."""
    def stated(v, cent=()):
        out = {"area": v[0] * length ** 2, "j": v[3] * length ** 4,
               "iy": v[4] * length ** 4, "iz": v[5] * length ** 4}
        if len(cent) >= 4:                           # the centroid below the top, above the bottom
            out.update(czp=cent[2] * length, czm=cent[3] * length)
        if v[1] > 0.0 and v[2] > 0.0:                # ASy, ASz: the areas the shear acts on
            out.update(ay=v[1] * length ** 2, az=v[2] * length ** 2)
        return out
    if sec.kind == "TAPERED":
        if sec.props and sec.props_j:
            a, b = stated(sec.props, sec.cent), stated(sec.props_j, sec.cent_j)
        elif sec.dims and sec.dims_j:
            import dataclasses
            # its sizes are linear along it, and the properties are those sizes' (not linear: I goes with the depth cubed)
            ends = [list(sec.dims), list(sec.dims_j)]
            if sec.shape == "H":                     # a bottom flange written as 0 is the top one's: at each end, first
                for e_ in ends:
                    for i_, j_ in ((4, 1), (5, 3)):
                        if len(e_) > i_ and e_[i_] <= 0.0:
                            e_[i_] = e_[j_]
            mid = tuple((1.0 - t) * x + t * y for x, y in zip(*ends))
            out = section_properties(dataclasses.replace(sec, kind="DBUSER", dims=mid, dims_j=(), props=()), length)
            out["note"] = "a tapered section taken prismatic, with its sizes at the member's place along it"
            return out
        else:
            raise ValueError(f"section {sec.name!r} (TAPERED {sec.shape} {sec.db}): its two ends are not stated "
                             "in the file in a form this translation reads")
        out = {q: (1.0 - t) * a[q] + t * b[q] for q in ("area", "iy", "iz", "j", "czp", "czm", "ay", "az") if q in a and q in b}
        out["note"] = "a tapered section taken prismatic, its properties linear between its two ends"
        return out
    if sec.props:
        return stated(sec.props, sec.cent)
    d = [x * length for x in sec.dims]
    shape = sec.shape
    note = ""
    if sec.kind == "SRC" and shape == "EPC" and sec.lines:
        # a concrete-filled pipe: its steel pipe alone (the fill comes with a
        # later stage, which this translation does not follow)
        p = [x.strip() for x in sec.lines[0].split(",")]
        D, t = float(p[3]) * length, float(p[4]) * length
        shape, d, note = "P", [D, t], "the steel pipe of a concrete-filled section, without its fill"
    if sec.kind == "SOD" and shape == "SOD-B" and sec.lines:
        p = [float(x) for x in sec.lines[0].split(",")[1:12]]
        B, H, t = (p[0] + p[1] + p[2]) * length, p[6] * length, p[7] * length
        shape, d, note = "B", [H, B, t, t], "a built-up box taken as a plain box of its outer sizes"
    if shape == "H" and not sec.dims and sec.db:
        m = re.search(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", sec.db)
        if not m:
            raise ValueError(f"section {sec.name!r}: cannot read the sizes of {sec.db!r}")
        H, B, tw, tf = (float(x) * length for x in m.groups())
        d = [H, B, tw, tf]
        note = f"sizes read from the database name {sec.db.split(',')[-1].strip()} (no root fillets)"
    if shape == "P":
        D, t = d[0], d[1]
        di = D - 2 * t
        a = math.pi / 4 * (D * D - di * di)
        i = math.pi / 64 * (D ** 4 - di ** 4)
        out = {"area": a, "iy": i, "iz": i, "j": 2 * i, "czp": D / 2, "czm": D / 2, "ay": 0.5 * a, "az": 0.5 * a}
    elif shape == "SR":
        D = d[0]
        i = math.pi / 64 * D ** 4
        out = {"area": math.pi / 4 * D * D, "iy": i, "iz": i, "j": 2 * i, "czp": D / 2, "czm": D / 2,
               "ay": 0.9 * math.pi / 4 * D * D, "az": 0.9 * math.pi / 4 * D * D}
    elif shape == "SB":
        H, B = d[0], d[1]                                # a solid rectangle
        lo, hi = min(H, B), max(H, B)
        j = hi * lo ** 3 * (1.0 / 3.0 - 0.21 * lo / hi * (1.0 - lo ** 4 / (12.0 * hi ** 4)))
        out = {"area": H * B, "iy": B * H ** 3 / 12, "iz": H * B ** 3 / 12, "j": j, "czp": H / 2, "czm": H / 2,
               "ay": 5.0 / 6.0 * H * B, "az": 5.0 / 6.0 * H * B}
    elif shape == "B":
        H, B, tw, tf = d[0], d[1], d[2], d[3]
        hi, bi = H - 2 * tf, B - 2 * tw
        a = B * H - bi * hi
        iy = (B * H ** 3 - bi * hi ** 3) / 12            # bending in the depth
        iz = (H * B ** 3 - hi * bi ** 3) / 12
        am = (B - tw) * (H - tf)                         # Bredt, on the wall centre lines
        j = 4 * am * am / (2 * (B - tw) / tf + 2 * (H - tf) / tw)
        out = {"area": a, "iy": iy, "iz": iz, "j": j, "czp": H / 2, "czm": H / 2,
               "ay": 2.0 * tf * bi, "az": 2.0 * tw * hi}                 # the flanges carry the y shear, the webs the z
    elif shape == "H":
        # H, B1, tw, tf1, B2, tf2: the top flange, and the bottom one where it differs
        H, B, tw, tf = d[0], d[1], d[2], d[3]
        B2 = d[4] if len(d) > 4 and d[4] > 0.0 else B
        tf2 = d[5] if len(d) > 5 and d[5] > 0.0 else tf
        hw = H - tf - tf2
        parts = ((B * tf, H - tf / 2, B * tf ** 3 / 12), (hw * tw, tf2 + hw / 2, tw * hw ** 3 / 12),
                 (B2 * tf2, tf2 / 2, B2 * tf2 ** 3 / 12))                   # (area, height of its centre, its own I)
        a = sum(q[0] for q in parts)
        zc = sum(q[0] * q[1] for q in parts) / a
        iy = sum(q[2] + q[0] * (q[1] - zc) ** 2 for q in parts)
        iz = (tf * B ** 3 + tf2 * B2 ** 3 + hw * tw ** 3) / 12
        j = (B * tf ** 3 + B2 * tf2 ** 3 + hw * tw ** 3) / 3
        out = {"area": a, "iy": iy, "iz": iz, "j": j, "czp": H - zc, "czm": zc,
               "ay": 5.0 / 6.0 * (B * tf + B2 * tf2), "az": hw * tw}
    elif shape == "2C":
        H, B, tw, tf, C = d[0], d[1], d[2], d[3], d[4]   # two channels, C between their webs' backs
        hw = H - 2 * tf
        a1 = hw * tw + 2 * B * tf
        iy1 = tw * hw ** 3 / 12 + 2 * (B * tf ** 3 / 12 + B * tf * ((H - tf) / 2) ** 2)
        e = (hw * tw * tw / 2 + 2 * B * tf * B / 2) / a1              # centroid from the web's back
        iz1 = hw * tw ** 3 / 12 + hw * tw * (e - tw / 2) ** 2 + 2 * (tf * B ** 3 / 12 + B * tf * (B / 2 - e) ** 2)
        out = {"area": 2 * a1, "iy": 2 * iy1, "iz": 2 * (iz1 + a1 * (C / 2 + e) ** 2),
               "j": 2 * (hw * tw ** 3 + 2 * B * tf ** 3) / 3}
        note = "two channels taken back to back at their spacing, flanges outward"
    else:
        raise ValueError(f"section {sec.name!r} ({sec.kind} {sec.shape}): a shape this translation does not compute")
    if note:
        out["note"] = note
    return out


def _weight(mat: midas.Material, force: float, length: float) -> Tuple[float, str]:
    """(weight density N/m^3, note) of a material."""
    if mat.weight_density is not None:
        return mat.weight_density * force / length ** 3, ""
    grade = (mat.standard.split()[-1] if mat.standard else "").lower()
    for key, w in _GRADE_WEIGHT:
        if grade.startswith(key):
            if key == "q" and "D64" in mat.standard:
                # the steel of JTG D64 weighs 78.5 kN/m3 in Midas (a real truss: its girder stage's
                # reactions, 5723 kN, are this to four digits, and 2% more than with 76.98)
                w = 78.5e3
            return w, f"weight density {w / 1e3:g} kN/m3 assumed for grade {mat.standard.split()[-1]}"
    return 76.98e3, "weight density of steel assumed (the file gives none)"


def _expansion(mat: midas.Material) -> Tuple[Optional[float], str]:
    """(thermal coefficient 1/K, note) of a material: the file's, or for a
    grade the file names without its values the codes' (JTG 3362 / D62:
    concrete 1.0e-5; JTG D64 and GB 50017: steel, strand and wire 1.2e-5),
    with a note; None for anything else."""
    if mat.alpha:
        return mat.alpha, ""
    if not mat.standard:
        return None, ""
    a = 1.0e-5 if mat.kind == "CONC" else 1.2e-5 if mat.kind == "STEEL" else None
    if a is None:
        return None, ""
    return a, f"thermal coefficient {a:g}/K assumed for grade {mat.standard.split()[-1]}"


def _poisson(mat: midas.Material) -> float:
    """Poisson's ratio of a material the file names by grade only: Midas'
    database has 0.2 for concrete (it sets the shear modulus, and with it a
    girder's shear deformation and its twist), 0.3 for steel."""
    grade = (mat.standard.split()[-1] if mat.standard else "").lower()
    return 0.2 if (mat.kind or "").upper().startswith("CONC") or (grade.startswith("c") and grade[1:2].isdigit()) else 0.3
