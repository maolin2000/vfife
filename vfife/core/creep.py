"""Creep and shrinkage of concrete: strain that arrives with time.

The third step of the same staircase v0.4 started. A temperature is a free
strain that depends on nothing but the field you impose; SHRINKAGE is a free
strain that depends on nothing but TIME; CREEP is a free strain that depends
on the whole STRESS HISTORY. All three are subtracted from the deformation --
`f = k (d - d_imposed)` -- for the reason `core/temperature` spells out: a
member's reported actions must be the ones it carries.

WHAT MAKES CREEP DIFFERENT is that the imposed strain is not an input. Under a
stress applied at age `t0` and held, concrete goes on straining to

    eps(t) = sigma(t0) [1 + phi(t, t0)] / Ec(t0)

and for a stress history it is the SUPERPOSITION of that over every increment
(the principle of superposition, McHenry 1943; linear in stress, which is the
assumption every code model makes and which holds to about 0.4 f'c):

    eps_cr(t) = sum_j  phi(t, t_j) * d_elastic_j

with `d_elastic_j` the elastic (mechanical) deformation INCREMENT that arrived
at age `t_j`. Written that way creep needs no new element and no new material
law: at every time station the solver adds up that sum and hands it to the
same slot a temperature uses.

TWO CONSEQUENCES WORTH SAYING OUT LOUD.

  * A determinate structure creeps and carries no extra force: the deflection
    grows by `phi` and the moments do not move. A model that reports changed
    reactions on a simply supported beam has the sign of the thing backwards,
    exactly as with temperature.
  * A restrained one RELAXES. Hold a bar at constant strain and its stress
    decays, because the creep strain eats the elastic one; hold a shrinking
    member and the force it develops is far less than `EA eps_sh`, because it
    creeps out of most of it. Those two are the closed forms this is checked
    against, and they are the reason creep matters at all for a bridge: it is
    what lets 8000 t of double cantilever settle into a frame without tearing
    itself apart.

AGEING. Concrete gets stiffer and stronger as it hydrates, so `Ec` is a
function of age and `phi` of the age AT LOADING as well as of the elapsed
time. Both are here; a model that ignores ageing over-predicts the creep of
anything loaded late by a factor of two.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

#: the models this module knows, for the error message when a name is wrong
CREEP_MODELS = ("mc90", "aci209", "curve")
SHRINKAGE_MODELS = ("mc90", "aci209", "curve")


# --------------------------------------------------------------------------- #
#  Ageing of the modulus
# --------------------------------------------------------------------------- #
def modulus_factor(t: float, s: float = 0.25, of: str = "strength") -> float:
    """Ec(t) / Ec(28) at concrete age `t` days.

    MC90 gives the STRENGTH gain as `exp[s (1 - sqrt(28/t))]` with s = 0.20,
    0.25 or 0.38 by cement type, and E follows the square root of it, so
    `of: "strength"` (the default, and the code's own reading) applies s/2 to
    the modulus. `of: "modulus"` applies s directly, which is what an analyst
    who has FITTED s to measured moduli means -- Ghali et al. (2000) do
    exactly that for the Confederation Bridge, with s = 0.25 and
    Ec(28) = 38.7 GPa.
    """
    if t <= 0.0:
        raise ValueError(f"concrete age t = {t} must be positive")
    e = s if of == "modulus" else 0.5 * s
    return math.exp(e * (1.0 - math.sqrt(28.0 / t)))


# --------------------------------------------------------------------------- #
#  Creep coefficient
# --------------------------------------------------------------------------- #
def creep_coefficient(spec: Dict[str, Any], t: float, t0):
    """phi(t, t0): the creep strain as a multiple of the elastic strain.

    `t` and `t0` are concrete ages in days. phi(t0, t0) = 0 by construction --
    the elastic part is not creep -- and phi is non-decreasing in t. `t0` may
    be an array, which is what makes the superposition affordable: a history
    of n stations costs n vectorised calls and not n^2 scalar ones.
    """
    if np.ndim(t0):
        a = np.asarray(t0, float)
        out = np.zeros_like(a)
        live = a < t
        if live.any():
            out[live] = _dispatch(spec, t, a[live])
        return out
    if t <= t0:
        return 0.0
    return float(_dispatch(spec, t, t0))


def _dispatch(spec, t, t0):
    model = spec.get("model", "mc90")
    if model == "mc90":
        return _mc90_creep(spec, t, t0)
    if model == "aci209":
        return _aci209_creep(spec, t, t0)
    if model == "curve":
        return _curve_creep(spec, t, t0)
    raise ValueError(f"unknown creep model {model!r}; one of "
                     + ", ".join(CREEP_MODELS))


def _notional_size(spec: Dict[str, Any]) -> float:
    """h0 = 2 Ac / u in mm: the drying thickness a section behaves as."""
    h0 = spec.get("notional_size")
    if h0 is None:
        raise ValueError("this model needs `notional_size` (h0 = 2 Ac / u, mm): "
                         "how thick the member is for drying, which is what "
                         "decides how long it takes")
    return float(h0)


def _mc90_creep(spec, t, t0):
    """CEB-FIP Model Code 1990, 2.1.6.4.

    phi = phi_RH beta(fcm) beta(t0,adj) * [(t-t0)/(betaH + t-t0)]^0.3

    `cement` (4 slow, 5 normal, 8 rapid high-strength -- the beta_sc of the
    shrinkage expression) adjusts the loading age (MC90 2.1.6.4.3),
    t0,adj = t0 [9 / (2 + t0^1.2) + 1]^alpha >= 0.5 d with alpha = -1, 0, 1:
    it was read by shrinkage only and did nothing to phi.

    MC90's phi is referenced to the 28-DAY modulus, J = 1/Ec(t0) + phi/Ec28.
    A spec bound to its material (`bind_reference`) returns
    phi Ec(t0)/Ec28 -- the creep as a multiple of the ELASTIC strain at t0,
    which is what the superposition multiplies. It returned the code's phi
    against Ec(t0), 17-22% low on concrete loaded late.
    """
    fcm = float(spec.get("fcm", 38.0e+6)) / 1e+6          # MPa
    rh = float(spec.get("rh", 70.0))
    h0 = _notional_size(spec)
    phi_rh = 1.0 + (1.0 - rh / 100.0) / (0.46 * (h0 / 100.0) ** (1.0 / 3.0))
    beta_fcm = 16.8 / math.sqrt(fcm)
    t0a = np.asarray(t0, float)
    alpha = _cement_alpha(spec)
    if alpha:
        t0a = np.maximum(t0a * (9.0 / (2.0 + t0a ** 1.2) + 1.0) ** alpha, 0.5)
    beta_t0 = 1.0 / (0.1 + t0a ** 0.20)
    beta_h = min(150.0 * (1.0 + (1.2 * rh / 100.0) ** 18) * h0 / 100.0 + 250.0,
                 1500.0)
    d = t - np.asarray(t0, float)
    phi = phi_rh * beta_fcm * beta_t0 * (d / (beta_h + d)) ** 0.3
    return phi * _reference_ratio(spec, t0)


#: MC90's cement classes by the shrinkage coefficient beta_sc, and the
#: loading-age exponent alpha each one takes (MC90 2.1.6.4.3)
CEMENT_ALPHA = {4.0: -1, 5.0: 0, 8.0: 1}


def _cement_alpha(spec):
    c = spec.get("cement", 5.0)
    try:
        return CEMENT_ALPHA[float(c)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"mc90 `cement` is 4 (slow), 5 (normal) or 8 (rapid "
                         f"high-strength), MC90's beta_sc; not {c!r}") from None


def _reference_ratio(spec, t0):
    """Ec(t0) / Ec28 of a spec `bind_reference` has tied to its material; 1
    for one that has not been (the code's own phi, as the tests read it)."""
    ag = spec.get("_ageing")
    if ag is not None:
        s_ = float(ag.get("s", 0.25))
        e = s_ if str(ag.get("of", "strength")) == "modulus" else 0.5 * s_
        t0a = np.asarray(t0, float)
        return np.exp(e * (1.0 - np.sqrt(28.0 / t0a)))
    return float(spec.get("_ec_ratio", 1.0))


def check_reference(spec: Optional[Dict[str, Any]], ageing, who: str) -> None:
    """An `mc90` creep spec has to say what Ec28 is, one way and only one."""
    if spec is None or spec.get("model", "mc90") != "mc90":
        return
    if ageing is not None and spec.get("E28") is not None:
        raise ValueError(
            f"{who}: `creep` has `E28` and the material declares `ageing`; mc90's "
            "phi is referenced to Ec28 and only one of them may say what it is -- "
            "drop `E28` (the ageing law gives Ec(t)/Ec28) or drop `ageing`")
    if ageing is None and spec.get("E28") is None:
        raise ValueError(
            f"{who}: `creep: {{model: mc90}}` needs the 28-day modulus. MC90's phi "
            "is referenced to Ec28, J(t, t0) = 1/Ec(t0) + phi/Ec28, and E here is the "
            "modulus at loading: declare `ageing` on the material (Ec(t)/Ec28 then "
            "follows from it) or give `E28` in `creep`")


def bind_reference(spec: Optional[Dict[str, Any]], E: float, ageing, who: str):
    """The creep spec as a solver uses it: an `mc90` one carries how Ec(t0)
    compares with Ec28, which its phi is referenced to.

    From the material's `ageing` when it has one (Ec(t)/Ec28 is then that
    law's own ratio); otherwise from `E28` in the creep spec, the declared E
    being the modulus at loading. Neither is refused: guessing Ec28 = E is
    the 17-22% this fixes. Other models are returned as they are -- ACI 209's
    phi and a measured curve are already multiples of the strain at loading.
    """
    if spec is None or spec.get("model", "mc90") != "mc90":
        return spec
    check_reference(spec, ageing, who)
    out = dict(spec)
    if ageing is not None:
        out["_ageing"] = dict(ageing)
    else:
        out["_ec_ratio"] = float(E) / float(spec["E28"])
    return out


def _aci209_creep(spec, t, t0):
    """ACI 209R-92: phi = (t-t0)^0.6 / (10 + (t-t0)^0.6) * phi_u * gamma_la.

    `ultimate` is phi_u; 2.35 times the product of the code's correction
    factors OTHER than the loading-age one, and 2.35 itself is the default
    because a model whose only input is time should say what it assumed.

    gamma_la(t0) = 1.25 t0^-0.118 moist cured, 1.13 t0^-0.094 steam cured
    (ACI 209R-92; `curing`) -- 0.994 at 7 days moist. It is applied here
    and not folded into `ultimate`, because in the superposition every
    increment has its own loading age; without it phi did not depend on t0
    and a late increment crept up to 81% too much.
    """
    phi_u = float(spec.get("ultimate", 2.35))
    t0a = np.asarray(t0, float)
    if spec.get("curing", "moist") == "steam":
        gamma = 1.13 * t0a ** -0.094
    else:
        gamma = 1.25 * t0a ** -0.118
    d = (t - t0a) ** 0.6
    return phi_u * gamma * d / (10.0 + d)


def _curve_creep(spec, t, t0):
    """A MEASURED creep function, `[[t - t0, phi], ...]` at one loading age,
    or `{t0: [[t - t0, phi], ...], ...}` at several.

    Linear in log(t - t0) between the points and held at the ends -- creep
    data is read on a log axis and interpolating it linearly in t puts the
    first decade in one segment. With several loading ages the two bracketing
    curves are blended linearly in log(t0).
    """
    table = spec.get("curve")
    if table is None:
        raise ValueError("a `curve` creep model needs `curve`: [[t - t0, phi], ...]")
    if isinstance(table, dict):
        if np.ndim(t0):
            return np.array([_curve_creep(spec, t, float(v)) for v in t0])
        ages = sorted(float(k) for k in table)
        lo = max([a for a in ages if a <= t0], default=ages[0])
        hi = min([a for a in ages if a >= t0], default=ages[-1])
        v_lo = _interp_log(table[_key(table, lo)], t - t0)
        if hi == lo:
            return v_lo
        v_hi = _interp_log(table[_key(table, hi)], t - t0)
        w = (math.log(t0) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return v_lo + w * (v_hi - v_lo)
    return _interp_log(table, t - t0)


def _key(table, value):
    for k in table:
        if float(k) == value:
            return k
    raise KeyError(value)


def _interp_log(table, d):
    """Linear in log(duration) between the rows; from the origin to the first
    row with a positive duration, linear in the duration itself.

    A table whose first row was `[1, 0.5]` read 0.5 from the instant of
    loading -- log(0) has no place on the axis, so the value was held back to
    it. The origin (0, 0) is implicit: nothing has crept or
    shrunk at zero duration, and a `[0, 0]` row says the same thing.
    """
    a = np.asarray(table, float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("a creep/shrinkage curve is [[duration, value], ...]")
    a = a[a[:, 0] > 0.0]
    d = np.asarray(d, float)
    if not len(a):
        v = np.zeros_like(d)
        return v if v.ndim else float(v)
    d1, v1 = a[0]
    v = np.interp(np.log(np.maximum(d, d1)), np.log(a[:, 0]), a[:, 1])
    v = np.where(d < d1, v1 * np.maximum(d, 0.0) / d1, v)
    return v if v.ndim else float(v)


# --------------------------------------------------------------------------- #
#  Shrinkage
# --------------------------------------------------------------------------- #
def shrinkage_strain(spec: Dict[str, Any], t: float, ts: Optional[float] = None) -> float:
    """eps_sh(t): free shrinkage strain, NEGATIVE (the member gets shorter) --
    except `mc90` at RH >= 99%, which is MC90's swelling and positive.

    `ts` is the age at which drying starts; it defaults to the spec's own
    `drying_start` and then to 7 days. Before `ts` the strain is zero: a
    sealed member does not dry.
    """
    ts = float(spec.get("drying_start", 7.0) if ts is None else ts)
    if t <= ts:
        return 0.0
    model = spec.get("model", "mc90")
    if model == "mc90":
        return _mc90_shrinkage(spec, t, ts)
    if model == "aci209":
        return _aci209_shrinkage(spec, t, ts)
    if model == "curve":
        return -abs(_interp_log(spec["curve"], t - ts))
    raise ValueError(f"unknown shrinkage model {model!r}; one of "
                     + ", ".join(SHRINKAGE_MODELS))


def _mc90_shrinkage(spec, t, ts):
    fcm = float(spec.get("fcm", 38.0e+6)) / 1e+6
    rh = float(spec.get("rh", 70.0))
    h0 = _notional_size(spec)
    bsc = float(spec.get("cement", 5.0))          # 4 slow, 5 normal, 8 rapid HS
    eps_s = (160.0 + 10.0 * bsc * (9.0 - fcm / 10.0)) * 1e-6
    # MC90 2.1.6.4.4: -1.55 [1 - (RH/100)^3] for 40 <= RH < 99, and +0.25 --
    # SWELLING, the member lengthens -- at 99 and above. The sign was thrown
    # away (`-abs`), so a member in water shrank. The jump at
    # 99% is the code's own: -0.046 just below it, +0.25 at it.
    beta_rh = 0.25 if rh >= 99.0 else -1.55 * (1.0 - (rh / 100.0) ** 3)
    d = t - ts
    beta_s = math.sqrt(d / (350.0 * (h0 / 100.0) ** 2 + d))
    return eps_s * beta_rh * beta_s


def _aci209_shrinkage(spec, t, ts):
    eps_u = abs(float(spec.get("ultimate", 780.0e-6)))
    a = 55.0 if spec.get("curing") == "steam" else 35.0
    d = t - ts
    return -eps_u * d / (a + d)


# --------------------------------------------------------------------------- #
#  What a spec may say
#
#  The specs are plain dicts on the material, and every key was read with
#  `.get(key, default)`: `RH: 40` for `rh` ran at the default 70%, and
#  `curing` on aci209 creep was read by nothing. Each model now names the keys
#  it reads and a key outside them is refused by name.
# --------------------------------------------------------------------------- #
CREEP_KEYS = {"mc90": {"notional_size", "rh", "fcm", "cement", "E28"},
              "aci209": {"ultimate", "curing"},
              "curve": {"curve"}}
SHRINKAGE_KEYS = {"mc90": {"notional_size", "rh", "fcm", "cement", "drying_start"},
                  "aci209": {"ultimate", "curing", "drying_start"},
                  "curve": {"curve", "drying_start"}}
AGEING_KEYS = {"s", "of", "reference_age"}


def _unknown(spec, known, who):
    import difflib
    bad = sorted(str(k) for k in spec if k not in known)
    if bad:
        lines = []
        for k in bad:
            near = difflib.get_close_matches(k, sorted(known), n=1, cutoff=0.5) or \
                [m for m in known if m.lower() == k.lower()]
            lines.append(repr(k) + (f" (did you mean {near[0]!r}?)" if near else ""))
        raise ValueError(f"{who} does not read " + ", ".join(lines)
                         + "; it reads " + ", ".join(sorted(known)))


def _positive(spec, key, who):
    v = spec.get(key)
    if v is not None and not float(v) > 0.0:
        raise ValueError(f"{who}: `{key}` must be positive, not {v!r}")


def _check_table(table, who):
    if isinstance(table, dict):
        if not table:
            raise ValueError(f"{who}: `curve` names no loading age")
        for k, v in table.items():
            if not float(k) > 0.0:
                raise ValueError(f"{who}: a `curve` loading age must be positive, not {k!r}")
            _check_table(v, f"{who} at t0 = {k}")
        return
    a = np.asarray(table, float)
    if a.ndim != 2 or a.shape[1] != 2 or not len(a):
        raise ValueError(f"{who}: `curve` is [[duration, value], ...]")
    if np.any(a[:, 0] < 0.0) or np.any(np.diff(a[:, 0]) <= 0.0):
        raise ValueError(f"{who}: `curve` durations must be >= 0 and increase")
    if a[0, 0] == 0.0 and a[0, 1] != 0.0:
        raise ValueError(f"{who}: `curve` at duration 0 must be 0 -- nothing has happened "
                         "yet at the instant of loading (or of drying)")


def _check_mc90(spec, who):
    rh = spec.get("rh")
    if rh is not None and not 40.0 <= float(rh) <= 100.0:
        raise ValueError(f"{who}: `rh` = {rh} is outside MC90's range, 40-100 % "
                         "(a relative humidity in percent)")
    fcm = spec.get("fcm")
    # MC90 covers C12 to C80: fck 12-80 MPa, fcm = fck + 8 = 20-88 MPa. Above
    # it the shrinkage expression's base strain (160 + 10 beta_sc (9 -
    # fcm/10)) 1e-6 turns negative near 110 MPa and the member swelled at 70%
    # humidity; 8-150 MPa was accepted
    if fcm is not None and not 20.0e+6 <= float(fcm) <= 88.0e+6:
        raise ValueError(f"{who}: `fcm` = {fcm} is outside MC90's range: it is read in Pa "
                         "(38.0e+6 for 38 MPa), and MC90 is for C12-C80 concrete, fck 12-80 MPa, "
                         "fcm = fck + 8 = 20-88 MPa")
    _positive(spec, "notional_size", who)
    _cement_alpha(spec)


def validate_creep(spec: Dict[str, Any], who: str = "creep") -> None:
    """Refuse now what the first time station would misread or ignore."""
    model = spec.get("model", "mc90")
    if model not in CREEP_KEYS:
        raise ValueError(f"{who}: unknown creep model {model!r}; one of "
                         + ", ".join(CREEP_MODELS))
    _unknown(spec, CREEP_KEYS[model] | {"model"}, f"{who} (model {model})")
    if model == "mc90":
        _check_mc90(spec, who)
        _positive(spec, "E28", who)
    elif model == "aci209":
        _positive(spec, "ultimate", who)
        if spec.get("curing", "moist") not in ("moist", "steam"):
            raise ValueError(f"{who}: `curing` is moist or steam, not {spec['curing']!r}")
    else:
        if spec.get("curve") is None:
            raise ValueError(f"{who}: a `curve` creep model needs `curve`: [[t - t0, phi], ...]")
        _check_table(spec["curve"], who)


def validate_shrinkage(spec: Dict[str, Any], who: str = "shrinkage") -> None:
    model = spec.get("model", "mc90")
    if model not in SHRINKAGE_KEYS:
        raise ValueError(f"{who}: unknown shrinkage model {model!r}; one of "
                         + ", ".join(SHRINKAGE_MODELS))
    _unknown(spec, SHRINKAGE_KEYS[model] | {"model"}, f"{who} (model {model})")
    _positive(spec, "drying_start", who)
    if model == "mc90":
        _check_mc90(spec, who)
    elif model == "aci209":
        if spec.get("curing", "moist") not in ("moist", "steam"):
            raise ValueError(f"{who}: `curing` is moist or steam, not {spec['curing']!r}")
    else:
        if spec.get("curve") is None:
            raise ValueError(f"{who}: a `curve` shrinkage model needs `curve`: [[t - ts, eps], ...]")
        _check_table(spec["curve"], who)


def validate_ageing(spec: Dict[str, Any], who: str = "ageing") -> None:
    _unknown(spec, AGEING_KEYS, who)
    _positive(spec, "s", who)
    _positive(spec, "reference_age", who)
    if str(spec.get("of", "strength")) not in ("strength", "modulus"):
        raise ValueError(f"{who}: `of` is strength or modulus, not {spec['of']!r}")


# --------------------------------------------------------------------------- #
#  The clock
#
#  Two solvers run on it -- BeamSolver on member deformations and
#  FiberBeamSolver on fibre strains -- and the two would drift apart if each
#  kept its own copy of these three lines. They are here, and
#  tests/test_creep.py asserts the two solvers put their stations at the same
#  ages and see the same phi sequence for the same step.
# --------------------------------------------------------------------------- #
def read_step_ages(step: Dict[str, Any], current: float, default: float,
                   latest: Optional[float] = None):
    """`(age0, age1 or None, spacing)` for one step of the case file.

    `age` defaults to where the previous step left off, so a run reads as one
    clock and not as a list of independent intervals.

    The clock only runs forward. `latest` is the age of the
    last frame already recorded: a step whose `age` is earlier than that
    un-crept the member, and a `to_age` earlier than the step's own `age` was
    silently a step at one instant. Both are refused by name.
    """
    a0 = float(step.get("age") if step.get("age") is not None else current
               if current is not None else default)
    a1 = step.get("to_age")
    name = step.get("name") or "?"          # the dump carries None
    if latest is not None and a0 < float(latest) - 1e-9 * max(1.0, abs(float(latest))):
        raise ValueError(
            f"step {name!r}: `age` = {a0:g} days is earlier than {float(latest):g}, where "
            "the analysis already is; concrete does not get younger. Start a separate run, "
            "or give this step an age at or after the last one")
    if a1 is not None and float(a1) < a0:
        raise ValueError(
            f"step {name!r}: `to_age` = {float(a1):g} is earlier than the step's own age "
            f"{a0:g} days; the clock only runs forward")
    return (a0, None if a1 is None else float(a1),
            str(step.get("time_spacing", "log")))


def refuse_moving_clock(step: Dict[str, Any], ages, current: Optional[float], who: str) -> None:
    """A dynamic run with `creep`: allowed while the concrete clock stands
    still, refused by name on a step that moves it.

    The history is fed at time stations and a dynamic run has none: with
    `to_age`, or an `age` other than where the clock is, the age advanced
    and nothing crept. On a clock that stands still, the
    history a `solver.initial_state` restored is the concrete's memory, a
    frozen imposed strain -- the dynamics of a crept structure. Refusing
    those too left the only way to run them dropping `creep`, and
    the member sprang back to its elastic length."""
    a0, a1, _ = ages
    moved = a1 is not None and float(a1) != float(a0)
    if current is not None and abs(float(a0) - float(current)) > 1e-9 * max(1.0, abs(float(current))):
        moved = True
    if moved:
        raise ValueError(
            f"step {step.get('name') or '?'!r}: {who} declares `creep`, and this dynamic step moves "
            f"the concrete clock ({float(current) if current is not None else float(a0):g} -> "
            f"{float(a1 if a1 is not None else a0):g} days); a dynamic run has no time stations, "
            "so the age would advance and nothing would creep. Run the creep in relaxation "
            "steps (`to_age` with `load_steps`) and the dynamics from its result "
            "(`solver.initial_state`) with no `age` / `to_age` in the dynamic steps: the "
            "creep reached is then held as a frozen imposed strain")


def age_at(step_ages, progress: float) -> float:
    """The concrete age at this point of the current step.

    LOG spacing by default, because that is the axis creep happens on: a step
    from 28 days to 10 000 in ten linear stations puts nine of them after the
    first thousand, where nothing is left to happen, and one across the decade
    where most of it does.
    """
    a0, a1, spacing = step_ages
    if a1 is None or a1 <= a0:
        return a0
    if spacing == "linear":
        return a0 + (a1 - a0) * progress
    return a0 * (a1 / a0) ** progress


def station_phi(spec, age: float, prev_station: Optional[float]) -> float:
    """The creep coefficient of the increment that ARRIVES AT this station.

    The increment is taken to arrive at the geometric middle of the interval
    it belongs to -- the classical step-by-step rule, and the one
    `relaxation()` uses -- so by the time the station is solved it has already
    crept by `phi(age, sqrt(prev * age))`. The first station of an analysis
    has no interval behind it and gets zero, which makes the elastic step
    exact.
    """
    if spec is None or prev_station is None or age <= prev_station:
        return 0.0
    return float(creep_coefficient(spec, age, math.sqrt(prev_station * age)))


def station_date(age: float, prev_station: Optional[float]) -> float:
    """The date an increment recorded at `age` is filed under: the geometric
    middle of its interval, the same date `station_phi` assumed while the
    station was being solved. The two go together and NEITHER WORKS ALONE --
    mid-dating without the station's own creep in the equation is pure
    over-relaxation and diverges; the station's own creep with end-dating is
    the explicit scheme, which on a held strain wobbles and gets worse as the
    axis is refined."""
    return age if prev_station is None else math.sqrt(prev_station * age)


# --------------------------------------------------------------------------- #
#  The superposition
# --------------------------------------------------------------------------- #
class CreepHistory:
    """The elastic deformation increments a member has seen, and their ages.

    `add(age, d_elastic)` records the state at a time station; `strain(age)`
    returns the creep deformation at a later one,

        d_cr(t) = sum_j phi(t, t_j) * (d_el_j - d_el_{j-1}),

    which is the superposition integral with the history discretised at the
    stations the analysis already stops at. It is recomputed from the whole
    history at every station rather than accumulated, so a refined time axis
    converges to the integral instead of drifting.
    """

    def __init__(self, spec: Dict[str, Any], groups=None):
        self.spec = dict(spec)
        #: [(row indices, spec), ...] when the rows do not all creep by one
        #: law -- a frame whose members have their own materials; rows in no
        #: group do not creep at all. None: `spec` for all.
        self.groups = None if groups is None else [
            (np.asarray(r, dtype=np.int64), dict(g)) for r, g in groups]
        self.ages: list = []
        self.increments: list = []
        self._last = None

    def add(self, age: float, d_elastic: np.ndarray) -> None:
        d = np.asarray(d_elastic, float).copy()
        inc = d if self._last is None else d - self._last
        self.ages.append(float(age))
        self.increments.append(inc)
        self._last = d

    def strain(self, age: float) -> Optional[np.ndarray]:
        if not self.ages:
            return None
        if self.groups is not None:
            out, hit = np.zeros_like(self.increments[0]), False
            for rows, spec in self.groups:
                for t_j, inc in zip(self.ages, self.increments):
                    phi = creep_coefficient(spec, age, t_j)
                    if phi != 0.0:
                        out[rows] += phi * inc[rows]
                        hit = True
            return out if hit else None
        out = None
        for t_j, inc in zip(self.ages, self.increments):
            phi = creep_coefficient(self.spec, age, t_j)
            if phi == 0.0:
                continue
            out = phi * inc if out is None else out + phi * inc
        return out


def shifted_strain(history: "CreepHistory", age: float, age_groups) -> Optional[np.ndarray]:
    """`history.strain(age)` with every member on its own clock (1.2):
    `age_groups(rows)` splits rows into (rows, shift, floor), and a member's
    phi is taken at (age + shift, max(t_j + shift, floor)) -- a segment born
    at age `floor` has no increment dated before it."""
    if not history.ages:
        return None
    out = np.zeros_like(history.increments[0])
    hit = False
    groups = history.groups if history.groups is not None else [
        (np.arange(len(history.increments[0])), history.spec)]
    windows = getattr(history, "windows", None)
    if windows is None or len(windows) != len(history.ages):
        windows = [None] * len(history.ages)
    for rows, spec in groups:
        for sub, sh, floor in age_groups(rows):
            t = age + sh
            for t_j, win, inc in zip(history.ages, windows, history.increments):
                if win is not None:
                    # dated on the member's own clock, as `station_phi` did
                    # while the station was solved: sqrt(max(prev, floor) * age)
                    prev, end = win
                    e = max(end + sh, floor)
                    t0 = e if prev is None else math.sqrt(max(prev + sh, floor) * e)
                else:
                    t0 = max(t_j + sh, floor)
                if t <= t0:
                    continue
                phi = creep_coefficient(spec, t, t0)
                if phi != 0.0:
                    out[sub] += phi * inc[sub]
                    hit = True
    return out if hit else None


def relaxation(spec: Dict[str, Any], t: float, t0: float, n: int = 400) -> float:
    """R(t, t0) / Ec(t0): the stress left in a member held at the strain it
    took at age t0, as a fraction of what it started with.

    Solved by the step-by-step method on a log time axis -- the same
    superposition the solver runs, with the strain held instead of the load --
    so a case that relaxes can be checked against something that is not the
    solver. Ageing of Ec is NOT included here: this is the relaxation of the
    creep law alone, which is what `tests/test_creep.py` compares against.
    """
    if t <= t0:
        return 1.0
    grid = np.unique(np.r_[t0, t0 * np.exp(np.linspace(0.0, math.log(t / t0), n))])
    sig = np.zeros(len(grid))
    sig[0] = 1.0                                   # the unit stress at t0
    # Each increment is taken to arrive at the MIDDLE of its interval (the
    # geometric middle, since the grid is logarithmic), which is the classical
    # step-by-step rule. Putting it at the end instead -- so that the current
    # increment has not crept at all -- makes the recursion explicit and
    # OSCILLATE: 24 stations over this decay returned a negative stress.
    mid = np.r_[grid[0], np.sqrt(grid[:-1] * grid[1:])]
    for i in range(1, len(grid)):
        dsig = np.r_[sig[0], np.diff(sig[:i])]
        s_ = float(dsig @ creep_coefficient(spec, grid[i], mid[:i]))
        phi_i = creep_coefficient(spec, grid[i], mid[i])
        sig[i] = (1.0 - s_ + sig[i - 1] * phi_i) / (1.0 + phi_i)
    return float(sig[-1])


# --------------------------------------------------------------------------- #
#  Where the clock runs
# --------------------------------------------------------------------------- #
TIME_KEYS = ("creep", "shrinkage", "ageing", "relaxation")
#: the element types whose solvers run the concrete clock: the frame path
#: (member by member) and the fibre path. `relaxation` needs a prestrained
#: member there -- a bar with `prestress`, or a fibre tendon.
CLOCKED = frozenset(("beam2", "beam2_timoshenko", "mixed_line", "beam2_fiber"))


def refuse_unclocked(problem: Dict[str, Any], material) -> None:
    """Creep on a solid, relaxation on an embedded tendon, shrinkage on a
    standalone truss: each was read by nothing, and the run converged on the
    answer without it. Refused by name instead."""
    kind = (problem.get("element") or {}).get("type")
    if kind in CLOCKED:
        return
    found = []

    def look(m, where):
        if m is None:
            return
        for k in TIME_KEYS:
            v = m.get(k) if isinstance(m, dict) else getattr(m, k, None)
            if v is not None and (k, where) not in found:
                found.append((k, where))

    look(material, "the model's material")
    look(problem.get("material"), "the model's material")
    for key, what in (("blocks", "a solid block"), ("members", "a line member"),
                      ("reinforcement", "embedded reinforcement")):
        for b in problem.get(key) or []:
            if isinstance(b, dict):
                look(b.get("material"), what)
    if found:
        said = ", ".join(f"`{k}` on {w}" for k, w in found)
        raise ValueError(
            f"{said}: the {kind!r} solver has no concrete clock, so it would be read by "
            "nothing and the run would converge without it. Creep, shrinkage, ageing and "
            "relaxation run on frame members (beam, and bar sections assigned to named "
            "regions -- the mixed-line path) and on fibre sections (docs/creep.md)")
