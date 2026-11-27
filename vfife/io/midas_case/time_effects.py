"""Midas' creep, shrinkage and ageing of concrete (`*TDM-TYPE`,
`*TDM-ELAST`, `*TDM-LINK`, `*ELEM-DEPMATL`, `*STAGE-CTRL` line 4) as the
materials' `creep`, `shrinkage` and `ageing` (1.22).

THE CODES TRANSLATED are the two Chinese highway ones the bridge models use:
JTG D62-2004 (`JTG`) and JTG 3362-2018 (`China(JTG3362-2018)`). Their
creep and shrinkage are CEB-FIP Model Code 1990's (D62's annex F, 3362's
annex C), with the mean strength from the characteristic CUBE strength,

    fcm = 0.8 fcu,k + 8 MPa,

so each becomes the solver's `mc90` with that fcm, the relative humidity
(HU), the notional size h = 2A/u (MSIZE, or the element's own from
`*ELEM-DEPMATL`), beta_sc (BSC, as `cement`) and the age drying starts
(AGE) -- beta_sc in the shrinkage only: JTG's creep has no adjustment of
the loading age for the cement, which MC90's `cement` makes. The modulus
is the material's, as the 28-day one (`E28`), unless `*TDM-ELAST` gives its
growth by CEB-FIP 2010 and the stage control takes it (bVAR) -- then
`ageing: {s, of: strength}`, Ec(t) = beta_cc(t)^0.5 Ec28, with s by its
cement class. 3362's
fly-ash adjustment (FLYASH) is not followed; a material with it is counted.
Every other code, and every other modulus law, is counted and left out.

Whether the stages take time-dependent effects at all is the stage
control's (`*STAGE-CTRL` line 4: bINC-TDE; bCNS, the "Creep & Shrinkage" box, with
its TYPE -- CREEP, SHRINKAGE or BOTH; and bVAR); a
model without a stage control takes none, and says so.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

#: the codes read as MC90, and how many fields they have after CODE
CODES = {"JTG": 5, "CHINA(JTG3362-2018)": 6}
#: CEB-FIP 2010's s by Midas' cement type iCTYPE (MC2010 5.1.9.1). Midas' MCT
#: quick reference numbers them from 1: "1: Rapid hardening high strength
#: cement" (RS, s 0.20), 2 normal or rapid hardening (N R, 0.25), 3 slowly
#: hardening (SL, 0.38), 4 with fly ash (not read). It was read from 0 until
#: the reference was found (1.23), every type one step too slow.
CEB2010_S = {1: 0.20, 2: 0.25, 3: 0.38}


def _f(x) -> Optional[float]:
    try:
        return float(str(x).strip())
    except ValueError:
        return None


def specs(model, f: float, length: float, leave, notes) -> Dict[int, dict]:
    """{material id: {"creep": ..., "shrinkage": ..., "ageing": ...,
    "h0": notional size in mm}} of the materials whose concrete creeps or
    shrinks in this model's stages; empty when nothing does."""
    if not model.time_links:
        return {}
    if model.time_effects is None:
        notes.append("the model has creep and shrinkage (*TDM-TYPE) and no stage control (*STAGE-CTRL): "
                     "time-dependent effects not taken")
        return {}
    on, which, growing = model.time_effects
    if not on:
        return {}
    creep_on = which in ("CREEP", "BOTH", "")
    shrink_on = which in ("SHRINKAGE", "BOTH", "")
    stress = f / length ** 2
    if not (creep_on or shrink_on or growing):
        notes.append("the stage control takes no creep and shrinkage (bCNS off) and no growth of the modulus")
        return {}
    out: Dict[int, dict] = {}
    for mid, (tname, ename) in sorted(model.time_links.items()):
        if mid not in model.materials or not tname:
            continue
        tm = model.time_models.get(tname)
        if tm is None:
            leave(f"*TDM-LINK names a creep model {tname!r} that *TDM-TYPE does not define")
            continue
        code, p = tm[0].strip().upper(), tm[1]
        if code not in CODES:
            leave(f"creep and shrinkage by code {tm[0].strip()} (not followed)")
            continue
        vals = [_f(x) for x in p[:CODES[code]]]
        if len(vals) < 5 or any(v is None for v in vals[:5]):
            leave(f"creep model {tname!r}: fields not read")
            continue
        fcu, rh, h, bsc, ts = vals[:5]
        if code == "CHINA(JTG3362-2018)" and len(vals) > 5 and vals[5]:
            leave("JTG 3362-2018 creep: the fly-ash adjustment (FLYASH) not followed")
        if bsc not in (4.0, 5.0, 8.0):
            leave(f"creep model {tname!r}: BSC {bsc:g} is none of MC90's 4, 5, 8 (not followed)")
            continue
        if not (40.0 <= rh <= 100.0) or h <= 0.0 or ts <= 0.0 or fcu <= 0.0:
            leave(f"creep model {tname!r}: humidity, notional size, drying age or strength out of "
                  "MC90's range (not followed)")
            continue
        fcm = 0.8 * fcu * stress + 8.0e6
        h0 = h * length * 1000.0
        # beta_sc is the shrinkage's alone in JTG (D62 F.2, 3362 C.2): its
        # creep has no adjustment of the loading age for the cement, which
        # MC90's `cement` makes -- the creep takes the normal class's none
        common = {"model": "mc90", "fcm": round(fcm, 3), "rh": rh, "notional_size": round(h0, 3),
                  "cement": 5}
        spec: dict = {"h0": h0}
        ageing = None
        if ename and growing:
            em = model.time_moduli.get(ename)
            s = None
            if em is not None and em[0] == "CODE" and em[1].strip().upper() == "CEB-FIP(2010)":
                ict = _f(em[2][1]) if len(em[2]) > 1 else None
                s = CEB2010_S.get(int(ict)) if ict is not None and ict == int(ict) else None
            if s is None:
                leave(f"modulus growth {ename!r} (*TDM-ELAST) not read: the modulus is the 28-day one")
            else:
                ageing = {"s": s, "of": "strength"}
                spec["ageing"] = ageing
        if creep_on:
            spec["creep"] = dict(common)
            if ageing is None:
                spec["creep"]["E28"] = None             # the material's modulus, filled in by the caller
        if shrink_on:
            spec["shrinkage"] = dict(common, cement=int(bsc), drying_start=ts)
        out[mid] = spec
    if out:
        notes.append(f"creep and shrinkage of {len(out)} material(s) as MC90 (JTG: fcm = 0.8 fcu,k + 8 MPa), "
                     f"each stage's duration a step of time; elements' own notional sizes from *ELEM-DEPMATL")
    return out


def apply(materials: dict, key: str, spec: dict, h0: Optional[float] = None) -> None:
    """Put `spec`'s laws on `materials[key]`, at the notional size `h0` (mm)
    when given."""
    m = materials[key]
    for law in ("creep", "shrinkage"):
        if law in spec:
            m[law] = dict(spec[law])
            if h0 is not None:
                m[law]["notional_size"] = round(h0, 3)
            if m[law].get("E28", 0) is None:
                m[law]["E28"] = m["E"]
    if "ageing" in spec:
        m["ageing"] = dict(spec["ageing"])


def notional_key(h0_mm: float) -> str:
    """The suffix of a material at its own notional size: whole millimetres."""
    return f"h{int(round(h0_mm))}"


def stage_ages(model, born) -> Dict[Tuple[int, str], float]:
    """(stage, structure group) -> the age (days) Midas gives the group's
    members when that stage builds them."""
    out: Dict[Tuple[int, str], float] = {}
    for k, st in enumerate(model.stages):
        for g, age in st.activate_elements:
            out.setdefault((k, g), float(age))
    return out
