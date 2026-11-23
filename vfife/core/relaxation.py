"""Relaxation of prestressing steel: the stress a tendon loses while its
strain is held.

The third free strain, and the one that is not concrete's. Creep and shrinkage
belong to the concrete and arrive as a strain; relaxation belongs to the steel
and arrives as a LOSS OF PRESTRAIN, which is the same thing said in the units
a tendon is described in:

    sigma_p = E_p (eps_section + eps_pre)

so a stress loss `d_sigma_pr` at held section strain is `eps_pre` falling by
`d_sigma_pr / E_p`, i.e. `eps_pre(t) = eps_pre(0) [1 - r(t)]` with
`r = d_sigma_pr / sigma_pi`. That is the whole coupling: it rides the
imposed-strain slot a tendon already uses, and nothing else in the fibre
solver has to know about it.

WHY THERE IS NO `mc90` OR `ec2` MODEL HERE, and this is a deliberate absence
rather than a gap left for later. The code expressions are short, and several
plausible-looking forms of them are in circulation; the coefficients could not
be sourced verbatim while this was written, and a recalled version was checked
against its own definition and FAILED it -- at the stress ratio where the
expression must reduce to `rho_1000` it returned 0.96% for a declared 2.5%.
A relaxation law written from memory would be a number with the shape of a
code answer and none of its authority, and it would be invisible: it would
come out as a plausible prestress loss. So this module carries the two shapes
that a case can state honestly, and the case file has to say where its numbers
came from.

    power   rho(t) = rho_1000 (t / 1000 h)^k
            The shape MC90, EC2 and the PCI family all reduce to at one stress
            level. Both coefficients are REQUIRED: there is no default, because
            a default here would be exactly the invented number above.
    curve   measured data [[t_hours, r], ...], read linearly in log t --
            relaxation data is plotted on a log axis for the same reason creep
            data is, and interpolating it linearly in t puts the whole first
            decade in one segment. `interpolation: linear` reads it linearly
            in t instead, as a table of ratios against days is meant.
            Before the first row the loss grows linearly
            from zero; after the last it stays.

`rho_1000` and `k` belong to ONE initial stress ratio. The dependence on
`sigma_pi / f_pk` is strong in every code, and this module does not model it:
a case that changes the stress ratio has to change the coefficients.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

RELAXATION_MODELS = ("power", "curve")


def relaxation_loss(spec: Dict[str, Any], t_hours: float) -> float:
    """`d_sigma_pr / sigma_pi` at `t_hours` after the tendon was stressed.

    Zero at and before t = 0 -- the instant a tendon is stressed it has not
    relaxed, which is what makes the first station of an analysis exact, the
    same property `creep.station_phi` has.
    """
    d = float(t_hours)
    if d <= 0.0:
        return 0.0
    model = str(spec.get("model", "power"))
    if model == "power":
        rho = spec.get("rho_1000")
        k = spec.get("k")
        if rho is None or k is None:
            raise ValueError(
                "relaxation model `power` needs BOTH `rho_1000` (the loss "
                "fraction at 1000 hours, e.g. 0.031) and `k` (the exponent). "
                "Neither has a default on purpose: the value depends on the "
                "steel class AND on the stress ratio it is held at, and a "
                "default would be a number nobody chose. Say where yours came "
                "from in the case file")
        out = float(rho) * (d / 1000.0) ** float(k)
    elif model == "curve":
        out = _curve(spec, d)
    else:
        raise ValueError(f"unknown relaxation model {model!r}; one of "
                         + ", ".join(RELAXATION_MODELS))
    out *= float(spec.get("reduction", 1.0))
    if out >= 1.0:
        # the whole prestrain gone and more: the tendon would push
        raise ValueError(f"relaxation loss {out:.3g} of the initial stress at {d:g} h: "
                         "the law has left the range it describes (rho_1000 and k are "
                         "fractions and small)")
    return out


def _curve(spec: Dict[str, Any], d: float) -> float:
    table = np.asarray(spec.get("curve"), dtype=float)
    if table.ndim != 2 or table.shape[1] != 2 or len(table) < 2:
        raise ValueError("relaxation `curve` is [[t_hours, loss], ...] with at "
                         "least two rows")
    t, r = table[:, 0], table[:, 1]
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("relaxation `curve` times must increase")
    if d <= t[0]:
        return float(r[0] * d / t[0]) if t[0] > 0.0 else float(r[0])
    if d >= t[-1]:
        return float(r[-1])
    if spec.get("interpolation", "log") == "linear":
        return float(np.interp(d, t, r))
    lo = np.searchsorted(t, d) - 1
    f = (math.log(d) - math.log(t[lo])) / (math.log(t[lo + 1]) - math.log(t[lo]))
    return float(r[lo] + f * (r[lo + 1] - r[lo]))


#: the keys each model reads; anything else is refused by name
RELAXATION_KEYS = {"power": {"rho_1000", "k"}, "curve": {"curve", "interpolation"}}
COMMON_KEYS = {"model", "reduction", "tensioned_at"}
#: a loss FRACTION above this is not a low- or normal-relaxation steel at any
#: stress ratio a code covers; it is a percentage written as a number
MAX_LOSS = 0.2


def validate(spec: Dict[str, Any], who: str = "relaxation") -> None:
    """Raise now rather than at the first time station."""
    model = str(spec.get("model", "power"))
    if model not in RELAXATION_MODELS:
        raise ValueError(f"{who}: unknown relaxation model {model!r}; one of "
                         + ", ".join(RELAXATION_MODELS))
    known = RELAXATION_KEYS[model] | COMMON_KEYS
    bad = sorted(str(k) for k in spec if k not in known)
    if bad:
        raise ValueError(f"{who} (model {model}) does not read "
                         + ", ".join(repr(k) for k in bad) + "; it reads "
                         + ", ".join(sorted(known)))
    if float(spec.get("reduction", 1.0)) <= 0.0:
        raise ValueError("relaxation `reduction` must be positive; it is the "
                         "factor a code applies because a tendon in a member "
                         "that is itself shortening relaxes less than one held "
                         "at constant strain (about 0.8). This library does "
                         "not compute it -- state it or leave it at 1.0 and "
                         "know the answer is the intrinsic loss")
    if spec.get("tensioned_at") is not None and not float(spec["tensioned_at"]) > 0.0:
        raise ValueError(f"{who}: `tensioned_at` is a concrete age in days, and positive")
    if model == "power":
        rho, k = spec.get("rho_1000"), spec.get("k")
        # `rho_1000: 3.1` for 3.1% took 310% off the prestrain and put the
        # tendon in compression
        if rho is not None and not 0.0 <= float(rho) <= MAX_LOSS:
            raise ValueError(f"{who}: `rho_1000` = {rho} is the loss at 1000 h as a FRACTION "
                             f"(0.031 for 3.1%), between 0 and {MAX_LOSS}")
        if k is not None and not 0.0 < float(k) < 1.0:
            raise ValueError(f"{who}: `k` = {k} is the exponent of t/1000 h, between 0 and 1 "
                             "(about 0.2 for strand)")
    else:
        if spec.get("interpolation", "log") not in ("log", "linear"):
            raise ValueError(f"{who}: `interpolation` is `log` (the default: linear in log t, "
                             "as relaxation data is plotted) or `linear` (linear in t, as a "
                             "table of ratios against days is read)")
        table = np.asarray(spec.get("curve"), dtype=float)
        if table.ndim == 2 and table.shape[1] == 2 and (
                np.any(table[:, 1] < 0.0) or np.any(table[:, 1] > MAX_LOSS)):
            raise ValueError(f"{who}: `curve` losses are FRACTIONS of the initial stress "
                             f"(0.031 for 3.1%), between 0 and {MAX_LOSS}")
    relaxation_loss(spec, 1000.0)


def stressed_at(spec: Dict[str, Any], age0: float, steps, who: str) -> float:
    """The concrete age the tendon was stressed at, where its relaxation
    clock starts: `tensioned_at`, or by default `solver.age` -- the age the
    analysis (and its concrete) starts at -- or the first step's `age` when
    that is earlier.

    v1.0 made the default the first step's `age`, and a continuation from
    `initial_state` whose first step is at 444.7 days restarted the
    relaxation there: the tendon got back the 1.4 kN it had already lost.
    (A continuation also carries the start in its checkpoint, `relax_t0`.)

    Not later than the analysis starts: the tendon carries its prestress
    from the first station, and a tendon stressed mid-run is a construction
    stage whose steel would carry the strain from before it was stressed."""
    first = (steps[0].get("age") if steps else None)
    start = float(first if first is not None else age0)
    t0 = float(spec.get("tensioned_at", min(float(age0), start)))
    if t0 > start + 1e-9:
        raise ValueError(
            f"{who}: `tensioned_at` = {t0:g} days is after the analysis starts "
            f"({start:g}); the tendon carries its prestress from the first station. "
            "Start the analysis at the age it is stressed, or later")
    return t0
