"""The accelerogram behind a `ground_motion` load.

Ten lines with two reasons to exist.

THE FIRST IS AN ERROR MESSAGE. `np.loadtxt` raises `FileNotFoundError` with
`filename` unset, and the CLI's handler prints `exc.filename` -- so a case
pointing at a record that is not there reported

    error: no such file: None

which says neither what was missing nor which of the case's several paths it
was. A mesh in the same position reports its own name, because `open()` sets
`filename` and numpy does not. Read once by a user who has just waited for a
restart to load.

THE SECOND is that both solvers were unpacking the two columns, the scale and
the direction themselves, in lines that had to agree and were checked by
nothing. The fibre solver had grown a cache for it -- a fracture rebuilds the
step's data, and re-reading a 20 000-line record at every split is a cost with
no answer attached -- and the beam solver had not.
"""
from __future__ import annotations

import os

import numpy as np


def read_record(spec: dict, cache: dict | None = None):
    """``(t, a, direction)`` for one `ground_motion` load.

    `a` is already multiplied by `scale`, which is what carries the record's
    unit: the El Centro file shipped in `benchmarks/assets` is in **g**, so
    its cases say `scale: 9.80665`. Getting that wrong is not an error
    anywhere -- the structure is simply shaken an order of magnitude too
    gently, and the answer looks reassuring.
    """
    path = spec["path"]
    if cache is None or path not in cache:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ground_motion record not found: {path} (a relative path is "
                "resolved against the case file, like a mesh)")
        data = np.loadtxt(path, float, encoding="utf-8")
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(
                f"ground_motion record {path} must have two columns "
                f"(time, acceleration); this one is {data.shape}")
        if np.any(np.diff(data[:, 0]) <= 0.0):
            raise ValueError(
                f"ground_motion record {path} has a time column that does not "
                "increase; `np.interp` would read it backwards and report "
                "nothing")
        if cache is not None:
            cache[path] = data
    else:
        data = cache[path]
    return (data[:, 0], data[:, 1] * float(spec.get("scale", 1.0)),
            np.asarray(spec.get("direction", [1, 0, 0]), float))


def pchip_slopes(t, v):
    """Fritsch-Carlson knot slopes of the monotone piecewise-cubic Hermite
    interpolant through (t, v) (as scipy's PchipInterpolator): zero at a
    local extremum of the samples, the weighted harmonic mean of the two
    secants elsewhere, and a shape-preserving three-point slope at each end.
    Between two equal samples the curve is flat; it never leaves the range
    of the two samples it joins (1.0 re-review)."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    h = np.diff(t); s = np.diff(v) / h
    m = np.zeros(len(v))
    if len(s) == 1:
        m[:] = s[0]
        return m
    for k in range(1, len(v) - 1):
        a, b = s[k - 1], s[k]
        if a * b > 0.0:
            w1 = 2.0 * h[k] + h[k - 1]; w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / a + w2 / b)

    def end(h0, h1, s0, s1):
        d = ((2.0 * h0 + h1) * s0 - h0 * s1) / (h0 + h1)
        if np.sign(d) != np.sign(s0):
            return 0.0
        if np.sign(s0) != np.sign(s1) and abs(d) > abs(3.0 * s0):
            return 3.0 * s0
        return d

    m[0] = end(h[0], h[1], s[0], s[1])
    m[-1] = end(h[-1], h[-2], s[-1], s[-2])
    return m


class RecordMotion:
    """A support's own record as a factor of the run's progress
    -- the same interface as an amplitude curve, so the
    prescribed-motion path the constraints already have drives it, with the
    record's time t = progress * duration - delay. Zero before the record
    starts and after it ends (a displacement record holds its last value:
    a support that has moved stays where it went).

    `rate` is the time derivative of the prescribed quantity at the same
    instant: a support driven by displacement moves with that velocity, not
    at rest between steps (the damping force and the kinetic energy of the
    support see it)."""

    def __init__(self, spec: dict, duration: float, delay: float = 0.0, cache: dict | None = None):
        path = spec["path"]
        if cache is None or path not in cache:
            if not os.path.exists(path):
                raise FileNotFoundError(f"support motion record not found: {path} (a relative path is "
                                        "resolved against the case file, like a mesh)")
            data = np.loadtxt(path, float, ndmin=2, encoding="utf-8")
            if data.shape[1] < 2 or len(data) < 2:
                raise ValueError(f"support motion record {path} must have two columns (time, value) "
                                 f"and two rows or more; this one is {data.shape}")
            if np.any(np.diff(data[:, 0]) <= 0.0):
                raise ValueError(f"support motion record {path} has a time column that does not increase")
            if cache is not None:
                cache[path] = data
        else:
            data = cache[path]
        self.name = f"record:{os.path.basename(path)}"
        self.t = data[:, 0]
        self.v = data[:, 1] * float(spec.get("scale", 1.0))
        self.quantity = spec.get("quantity", "displacement")
        self.duration = float(duration)
        self.delay = float(spec.get("delay", 0.0)) + float(delay)
        # the slope of each segment: the exact derivative of the linear
        # interpolation the value is read with
        self.slope = np.diff(self.v) / np.diff(self.t)
        self.hold = self.quantity == "displacement"
        # A DISPLACEMENT RECORD IS READ C1: a cubic Hermite through the
        # samples, so the support's velocity is continuous.
        # Read linearly, the velocity jumped at every sample -- an impulse on
        # the support every 0.005 s -- and the high-frequency pier modes it
        # rang tripled the pier base shear against the same motion given as
        # an acceleration record (2266 against 708 kN), with the drifts equal.
        # The knot slopes are Fritsch-Carlson's (PCHIP), which keep the record
        # monotone where its samples are: the three-point slope overshot a
        # coarse step 0, 0, 1, 1 by 7.4% each side, the support stepping back
        # before it stepped forward (1.0 re-review).
        self.knot_slope = None
        if self.hold:
            self.knot_slope = pchip_slopes(self.t, self.v)

    def _time(self, progress):
        return float(progress) * self.duration - self.delay

    def _hermite(self, t):
        """(value, derivative) of the C1 displacement interpolation at t,
        inside the record."""
        k = min(max(int(np.searchsorted(self.t, t, side="right")) - 1, 0), len(self.slope) - 1)
        h = self.t[k + 1] - self.t[k]
        u = (t - self.t[k]) / h
        y0, y1 = self.v[k], self.v[k + 1]
        m0, m1 = self.knot_slope[k] * h, self.knot_slope[k + 1] * h
        u2, u3 = u * u, u * u * u
        val = (2*u3 - 3*u2 + 1) * y0 + (u3 - 2*u2 + u) * m0 + (-2*u3 + 3*u2) * y1 + (u3 - u2) * m1
        der = ((6*u2 - 6*u) * y0 + (3*u2 - 4*u + 1) * m0 + (-6*u2 + 6*u) * y1 + (3*u2 - 2*u) * m1) / h
        return float(val), float(der)

    def __call__(self, progress):
        t = self._time(progress)
        if self.knot_slope is not None:
            if t <= self.t[0]:
                return 0.0 if t < self.t[0] else float(self.v[0])
            if t >= self.t[-1]:
                return float(self.v[-1])
            return self._hermite(t)[0]
        return float(np.interp(t, self.t, self.v, left=0.0, right=self.v[-1] if self.hold else 0.0))

    def rate(self, progress):
        t = self._time(progress)
        if t < self.t[0] or t >= self.t[-1]:
            return 0.0
        if self.knot_slope is not None:
            return self._hermite(t)[1]
        return float(self.slope[min(int(np.searchsorted(self.t, t, side="right")) - 1, len(self.slope) - 1)])

    def motion(self, progress):
        """(displacement, velocity) of a velocity or acceleration record,
        integrated EXACTLY from its piecewise-linear interpolation -- as a
        displacement record is read, exact and independent of the step. The
        support used to be advanced q += dt v (+ dt^2 a / 2), first order in
        dt. Zero before the record starts; after it ends a
        velocity record holds its displacement, an acceleration record
        coasts at its last velocity."""
        t, x, s = self.t, self.v, self.slope
        if getattr(self, "_U", None) is None:
            h = np.diff(t)
            if self.quantity == "velocity":
                self._V = None
                self._U = np.concatenate([[0.0], np.cumsum(x[:-1] * h + 0.5 * s * h * h)])
            else:
                V = np.concatenate([[0.0], np.cumsum(x[:-1] * h + 0.5 * s * h * h)])
                self._V = V
                self._U = np.concatenate([[0.0], np.cumsum(V[:-1] * h + 0.5 * x[:-1] * h * h + s * h ** 3 / 6.0)])
        U, V = self._U, self._V
        tt = self._time(progress)
        if tt <= t[0]:
            return 0.0, 0.0
        if tt >= t[-1]:
            if V is None:
                return float(U[-1]), 0.0
            return float(U[-1] + V[-1] * (tt - t[-1])), float(V[-1])
        k = min(int(np.searchsorted(t, tt, side="right")) - 1, len(s) - 1)
        tau = tt - t[k]
        if V is None:
            return float(U[k] + x[k] * tau + 0.5 * s[k] * tau * tau), float(x[k] + s[k] * tau)
        return (float(U[k] + V[k] * tau + 0.5 * x[k] * tau * tau + s[k] * tau ** 3 / 6.0),
                float(V[k] + x[k] * tau + 0.5 * s[k] * tau * tau))


def support_prescriptions(c: dict, idx, node_dofs, coords, duration: float, cache: dict | None = None):
    """The prescribed-motion entries ``(dofs, 1.0, RecordMotion, motion)``
    of one constraint with a `record`. One entry for the
    whole region, or, under a travelling wave, one per node with the node's
    own delay (x - origin) . e / c."""
    rec = c["record"]
    motion = rec.get("quantity", "displacement")
    comps = [d for d, on in enumerate(c["dofs"][:3]) if on]
    wave = rec.get("wave")
    if wave is None:
        a = RecordMotion(rec, duration, 0.0, cache)
        return [(node_dofs(idx, d), 1.0, a, motion) for d in comps]
    e = np.asarray(wave["direction"], float)
    if not np.linalg.norm(e) > 0:
        raise ValueError("a travelling wave needs a direction that is not zero")
    e = e / np.linalg.norm(e)
    lag = (np.asarray(coords, float)[idx] - np.asarray(wave.get("origin", [0, 0, 0]), float)) @ e \
        / float(wave["speed"])
    if np.any(lag < -1e-12):
        raise ValueError("a travelling wave reaches part of the region before its origin: put `origin` "
                         "where the wave arrives first (every delay must be zero or more)")
    out = []
    for n, dt in zip(np.asarray(idx), np.maximum(lag, 0.0)):
        a = RecordMotion(rec, duration, float(dt), cache)
        out += [(node_dofs(np.array([n]), d), 1.0, a, motion) for d in comps]
    return out
