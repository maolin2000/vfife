"""A vehicle with mass, suspension and tyres, riding the deck.

`moving_load` is a moving FORCE: the axle loads keep their size,
nothing bounces, and the deck's deflection does not reach back into the
vehicle. That is the right first model and the one with a closed form; it
cannot give an impact factor that depends on the suspension, a vehicle's
own acceleration (the comfort question), or the road roughness, which on a
short span is what excites a bridge most.

This is the planar vehicle of the vehicle-bridge-interaction literature
(Yang & Yau 1997, Yang, Yau & Wu 2004): a rigid body -- the sprung mass --
with heave and, when it has more than one axle, pitch; each axle an unsprung
mass under a suspension (spring + dashpot) standing on a tyre (spring +
dashpot) that touches the road. The road is the deck's deflection at the
contact point plus a roughness profile. A tyre pushes and never pulls: when
the wheel leaves the road the contact force is zero, and the number of such
steps is counted and reported.

    body       m_b z_b''  = sum_i S_i - m_b g,   I theta'' = sum_i a_i S_i
    wheel i    m_u z_u,i'' = -S_i + P_i - m_u g
    S_i = -k_s e_i - c_s e_i',  e_i = z_b + a_i theta - z_u,i        (suspension)
    P_i = max(0, k_t d_i + c_t d_i') if d_i > 0 else 0,
          d_i = w_i + r(s_i) - z_u,i                                (tyre)

z is measured UP (against `direction`, the way gravity pulls) from the
vehicle's unloaded ride height; a_i is the axle's position ahead of the body's
centre of mass. The deck feels -P_i along `direction`... i.e. P_i pushing
down, at the contact point, by the same Hermite shape functions `moving_load`
uses -- and w_i, the deck's upward displacement there, is read back through
the transpose of the same functions, so the coupling does no spurious work.

Coupling in time: the vehicle is integrated by the same central difference
and the same step as the structure, from the structure's state at the start
of the step, so the pair advances as one explicit system. The vehicle starts
in static equilibrium on the road at its first station (flat deck, the
roughness there), so it does not drop onto the bridge.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .moving_load import LoadPath


def roughness_profile(spec: Optional[dict], s_min: float, s_max: float):
    """A callable r(s) (metres, up) from `roughness`, or None.

    `profile`: a two-column file (station m, elevation m), linear in between,
    zero outside. `psd`: a one-sided displacement PSD G_d(n) = g0 (n / n0)^-w
    (m^3, n in cycles/m) sampled as sum_k sqrt(2 G_d(n_k) dn) cos(2 pi n_k s +
    phi_k), phases from `seed` -- the harmonic superposition every VBI paper
    uses. No roughness class table lives here: g0 is the user's (a road
    standard's class is a number to look up, not a law)."""
    if not spec:
        return None
    if spec.get("profile") is not None:
        data = np.loadtxt(spec["profile"], float, ndmin=2, encoding="utf-8")
        if data.shape[1] < 2 or np.any(np.diff(data[:, 0]) <= 0):
            raise ValueError(f"roughness profile {spec['profile']}: two columns (station, "
                             "elevation), stations increasing")
        s, r = data[:, 0], data[:, 1] * float(spec.get("scale", 1.0))
        return lambda x: np.interp(x, s, r, left=0.0, right=0.0)
    p = spec["psd"]
    n0 = float(p.get("n0", 0.1)); w = float(p.get("w", 2.0)); g0 = float(p["g0"])
    n_lo, n_hi = float(p.get("n_min", 0.01)), float(p.get("n_max", 3.0))
    count = int(p.get("count", 400))
    rng = np.random.default_rng(int(p.get("seed", 0)))
    edges = np.linspace(n_lo, n_hi, count + 1)
    n = 0.5 * (edges[1:] + edges[:-1]); dn = np.diff(edges)
    # `scale` multiplies the elevation, as it does a profile's
    amp = float(spec.get("scale", 1.0)) * np.sqrt(2.0 * g0 * (n / n0) ** (-w) * dn)
    phi = rng.uniform(0.0, 2.0 * np.pi, count)
    # the sum itself, not an interpolated table: four points per shortest
    # wavelength lost 7.8% of the slope variance a tyre feels.
    # A few axles times `count` cosines per step costs nothing.
    kn = 2.0 * np.pi * n
    try:
        from ..kernels.numba_loads import psd_road as _psd
    except ImportError:                                   # pragma: no cover
        _psd = None

    def road(x):
        x = np.asarray(x, float)
        flat = np.ascontiguousarray(x.reshape(-1))
        out = np.empty(flat.shape)
        if _psd is not None:
            # compiled, station by station (1.6): the compiled step asks for
            # a chunk of steps at once and must read what the loop reads
            _psd(flat, kn, phi, amp, out)
        else:
            for i in range(0, len(flat), 4096):
                c = flat[i:i + 4096]
                out[i:i + 4096] = np.cos(np.outer(c, kn) + phi[None, :]) @ amp
        return out.reshape(x.shape) if x.ndim else float(out[0])
    return road


class Vehicle:
    """One vehicle on a `LoadPath`, called like a `MovingLoad`: with the
    step's progress and time it returns the nodal forces of its tyres, and
    advances its own state by one step."""

    def __init__(self, spec: dict, path: LoadPath, solver, dofs, mesh, length, R0,
                 total_time: float, dt: float):
        self.name = spec.get("name") or "vehicle"
        self.path, self.dofs, self.mesh = path, dofs, mesh
        self.length, self.R0 = length, R0
        self.solver = solver
        self.dt = float(dt)
        self.total_time = float(total_time)
        self.speed = float(spec["speed"])
        self.start = float(spec.get("start", 0.0))
        d = np.asarray(spec.get("direction", [0.0, 0.0, -1.0]), float)
        if not np.isfinite(d).all() or float(np.linalg.norm(d)) == 0.0:
            raise ValueError(f"vehicle {self.name!r}: direction must be a nonzero vector "
                             "(the way gravity pulls)")
        self.down = d / np.linalg.norm(d)
        self.g = float(spec.get("gravity", 9.80665))
        body = spec["body"]
        axles = spec["axles"]
        self.na = len(axles)
        way = -1.0 if self.speed < 0 else 1.0
        self.offsets = way * np.array([float(a["offset"]) for a in axles])
        cg = way * float(body.get("cg", 0.0))
        self.arm = self.offsets - cg                     # ahead of the centre of mass, in s
        self.arm_travel = way * self.arm                 # ahead, in the direction of travel
        self.pitch = self.na > 1
        if self.pitch and body.get("pitch_inertia") is None:
            raise ValueError(f"vehicle {self.name!r}: a body on {self.na} axles pitches; give "
                             "body.pitch_inertia (kg m^2)")
        nb = 2 if self.pitch else 1
        self.nb = nb
        # an axle with a tyre is a wheel of its own (a degree of freedom); an
        # axle without one rides the road rigidly -- the "moving sprung mass"
        # of the literature -- and passes its suspension force straight on
        self.flex = np.array([a.get("tyre") is not None for a in axles])
        self.fi = np.flatnonzero(self.flex)                 # flexible axles
        nf = len(self.fi)
        n = nb + nf
        m = np.zeros(n)
        m[0] = float(body["mass"])
        if self.pitch:
            m[1] = float(body["pitch_inertia"])
        mu = np.array([float(a.get("unsprung_mass") or 0.0) for a in axles])
        if np.any(mu[self.flex] <= 0.0):
            raise ValueError(f"vehicle {self.name!r}: an axle on a tyre needs an unsprung_mass > 0 "
                             "(the wheel and axle the tyre carries)")
        if np.any(mu[~self.flex] > 0.0):
            raise ValueError(f"vehicle {self.name!r}: an axle without a tyre rides the road rigidly "
                             "and carries no unsprung mass of its own; give it a tyre, or drop the mass")
        m[nb:] = mu[self.fi]
        self.m = m
        ks = np.array([float(a["suspension"]["stiffness"]) for a in axles])
        cs = np.array([float(a["suspension"].get("damping", 0.0)) for a in axles])
        self.ks, self.cs = ks, cs
        self.kt = np.array([float((a.get("tyre") or {}).get("stiffness", 0.0)) for a in axles])
        self.ct = np.array([float((a.get("tyre") or {}).get("damping", 0.0)) for a in axles])
        # suspension extensions e = B x - R s (s: the road under a rigid axle)
        B = np.zeros((self.na, n))
        B[:, 0] = 1.0
        if self.pitch:
            B[:, 1] = self.arm_travel
        B[self.fi, nb + np.arange(nf)] = -1.0
        self.B = B
        self.rigid = (~self.flex).astype(float)
        self.weight = np.zeros(n)
        self.weight[0] = m[0] * self.g
        self.weight[nb:] = m[nb:] * self.g
        # the explicit step against the vehicle's own stiffest mode
        Kfull = B.T @ (ks[:, None] * B)
        Kfull[nb + np.arange(nf), nb + np.arange(nf)] += self.kt[self.fi]
        w_max = float(np.sqrt(np.max(np.abs(Kfull).sum(axis=1) / m)))
        if w_max * self.dt > 1.0:
            raise ValueError(
                f"vehicle {self.name!r}: its stiffest mode (about {w_max / (2 * np.pi):.3g} Hz) needs a "
                f"step below {1.0 / w_max:.3g} s; the structure's step is {self.dt:.3g} s. Lower "
                "solver.dt_safety, or check the tyre and suspension stiffnesses and the masses")
        # roughness over every station any axle will touch
        s0 = self.start + self.offsets
        s1 = s0 + self.speed * self.total_time
        self.rough = roughness_profile(spec.get("roughness"), float(min(s0.min(), s1.min())),
                                       float(max(s0.max(), s1.max())))
        # static equilibrium on the road at the first station, flat deck
        surf = self._surface_rigid(self.start)
        self.x = self._static(surf)
        self.v = np.zeros(n)
        self.a = np.zeros(n)
        self.contact = self._forces(self.x, self.v, surf, np.zeros(self.na))[1]
        self._surf_prev = None
        self._t_done = None
        self._out = None
        self.lift_off = 0                  # axle-steps not pressing the road
        self.lift_tyre = 0                 # of those: a tyred wheel off the road
        self.tension_rigid = 0             # of those: a rigid axle pulling the deck up
        self.lift_first = float("inf")     # when it first happened; inf = never
        self._prepared = False
        self.member_load = None
        self.flex_mask = self.flex.astype(np.bool_)
        self.up = np.ascontiguousarray(-self.down)

    def _forces(self, x, v, surf, surf_rate):
        """(generalised force on x, the force each axle presses the road
        with) for the road `surf` under the axles."""
        e = self.B @ x - self.rigid * surf
        de = self.B @ v - self.rigid * surf_rate
        S = -self.ks * e - self.cs * de                    # on the body, up
        f = self.B.T @ S - self.weight
        P = S.copy()                                        # a rigid axle passes S on
        if len(self.fi):
            u = self.nb + np.arange(len(self.fi))
            d = surf[self.fi] - x[u]                        # tyre compression
            dd = surf_rate[self.fi] - v[u]
            # a tyre pushes only while it is compressed: its dashpot does not
            # reach up to a wheel above the road, and it never pulls
            # (v1.0: 31 kN at a 0.08 mm gap before)
            Pt = np.where(d > 0.0, np.maximum(self.kt[self.fi] * d + self.ct[self.fi] * dd, 0.0), 0.0)
            P[self.fi] = Pt
            f[u] += Pt
        return f, P

    # -------------------------------------------------------------- geometry
    def stations(self, t: float) -> np.ndarray:
        return self.start + self.speed * float(t) + self.offsets

    def _surface_rigid(self, ref: float) -> np.ndarray:
        s = ref + self.offsets
        return self.rough(s) if self.rough is not None else np.zeros(self.na)

    def _static(self, surface: np.ndarray) -> np.ndarray:
        nf = len(self.fi)
        K = self.B.T @ (self.ks[:, None] * self.B)
        u = self.nb + np.arange(nf)
        K[u, u] += self.kt[self.fi]
        f = -self.weight.copy() + self.B.T @ (self.ks * self.rigid * surface)
        f[u] += self.kt[self.fi] * surface[self.fi]
        return np.linalg.solve(K, f)

    def _prepare(self):
        el = self.path.elements
        self._R = np.ascontiguousarray(np.asarray(self.R0, float)[el])
        self._L = np.ascontiguousarray(np.asarray(self.length, float)[el])
        self._edofs = np.ascontiguousarray(
            np.stack([self.dofs.element_dofs(self.mesh.elements[e][:2]) for e in el]), dtype=np.int64)
        rel = getattr(self, "releases", None)
        self._has_rel = rel is not None
        self._rel = np.ascontiguousarray(rel, dtype=float) if self._has_rel else np.zeros((1, 12, 12))
        self._rel_mask = np.full(len(el), self._has_rel, dtype=np.bool_)
        self._prepared = True

    def _kernel_state(self):
        """The state arrays `kernels.numba_loads.vehicle_step` advances in
        place (1.6), made contiguous float arrays of their own once: a
        restart hands them back as whatever the checkpoint held."""
        for f in ("x", "v", "a", "contact"):
            arr = getattr(self, f)
            if not (isinstance(arr, np.ndarray) and arr.dtype == float and arr.flags.c_contiguous
                    and arr.ndim == 1):
                setattr(self, f, np.ascontiguousarray(np.asarray(arr, float).reshape(-1)))
        sp = self._surf_prev
        buf = getattr(self, "_surf_buf", None)
        if buf is not None and sp is buf:
            # our own buffer: `_has_prev` says whether a step has filled it
            return buf, self._has_prev
        if sp is None:
            self._surf_buf = np.zeros(self.na)
            self._has_prev = np.zeros(1, np.uint8)
        else:
            self._surf_buf = np.ascontiguousarray(np.asarray(sp, float).reshape(-1))
            self._has_prev = np.ones(1, np.uint8)
        self._surf_prev = self._surf_buf
        return self._surf_buf, self._has_prev

    def _shape(self, s: np.ndarray):
        """For the axles on the path: (mask, element dofs, the global
        consistent vector of a unit force UP at each contact point). The
        member-local vectors and the members are kept for the end actions."""
        on = (s >= 0.0) & (s <= self.path.total)
        if not on.any():
            self._unit_local = None
            return on, None, None
        ss = s[on]
        up = np.broadcast_to(-self.down, (len(ss), 3))
        k, fl, fg = self.path.shape(ss, self._R, self._L, up, getattr(self, "releases", None))
        self._unit_local = (self.path.elements[k], fl)
        return on, self._edofs[k], fg

    # ------------------------------------------------------------------ step
    def __call__(self, progress: float, t: Optional[float], out: np.ndarray) -> np.ndarray:
        """The tyre forces on the deck for this step, and the vehicle moved
        through it. `t` is the START of the step (the loop reads its loads
        there since v1.0), where the deck's state is: the pair
        advances as one explicit system."""
        if t is None:
            raise ValueError(f"vehicle {self.name!r}: a vehicle needs a clock (solver.mode: dynamic)")
        if not self._prepared:
            self._prepare()
        if self._t_done is not None and abs(float(t) - self._t_done) < 1e-12 * max(1.0, abs(t)):
            out += self._out                                # asked twice at one instant
            return out
        # t, not t - dt: since v1.0 the loop asks at the step's start, and
        # `t - dt` left the vehicle one step behind
        t0 = float(t)
        s = self.stations(t0)
        try:
            from ..kernels.numba_loads import vehicle_step as _vs
        except ImportError:                                   # pragma: no cover
            _vs = None
        if _vs is not None:
            # the compiled version (1.6), which the compiled step runs too
            on = (s >= 0.0) & (s <= self.path.total)
            k, a = self.path.place(s)
            rough = self.rough(s) if self.rough is not None else np.zeros(self.na)
            surf_prev, has_prev = self._kernel_state()
            fl = np.zeros((self.na, 12)); fg = np.zeros((self.na, 12))
            force = np.zeros_like(out)
            tyres, rigid = _vs(np.ascontiguousarray(self.solver.q, dtype=float), self._R, self._L,
                               self._rel, self._rel_mask, self._edofs,
                               np.ascontiguousarray(k, dtype=np.int64), np.ascontiguousarray(a, dtype=float),
                               np.ascontiguousarray(on, dtype=np.bool_), np.ascontiguousarray(rough, dtype=float),
                               self.up, int(self.nb), np.ascontiguousarray(self.fi, dtype=np.int64),
                               self.flex_mask, self.m, self.B, self.rigid, self.ks, self.cs, self.kt, self.ct,
                               self.weight, self.dt, self.x, self.v, self.a, self.contact, surf_prev, has_prev,
                               fl, fg, force, np.zeros((6, self.na)), np.zeros(len(self.m)))
            if tyres or rigid:
                self._contact_lost(t0 + float(getattr(self, "clock_offset", 0.0)), int(tyres), int(rigid))
            self._latch(force, on, k, fl, t0)
            out += force
            return out
        on, edofs, fg = self._shape(s)
        q = self.solver.q
        w = np.zeros(self.na)
        if edofs is not None:
            w[on] = np.einsum("nk,nk->n", fg, q[edofs])
        surf = w + (self.rough(s) if self.rough is not None else 0.0)
        rate = np.zeros(self.na) if self._surf_prev is None else (surf - self._surf_prev) / self.dt
        self._surf_prev = surf
        f, P = self._forces(self.x, self.v, surf, rate)
        # a tyre does not pull (`_forces` gives it 0 off the road). A rigid
        # axle cannot leave the road in this model; its tension is passed on
        # and counted -- the count says the model is outside its range. Both
        # go to the run's diagnostics and the log
        tyre_off = self.flex & (P <= 0.0)
        rigid_pull = ~self.flex & (P < 0.0)
        if tyre_off.any() or rigid_pull.any():
            # on the run's clock, as member failures are: `t0` is the step's
            # own
            self._contact_lost(t0 + float(getattr(self, "clock_offset", 0.0)),
                               int(tyre_off.sum()), int(rigid_pull.sum()))
        self.contact = P
        self.a = f / self.m
        self.v = self.v + self.a * self.dt
        self.x = self.x + self.v * self.dt
        force = np.zeros_like(out)
        if edofs is not None:
            # each tyre pushes the deck down by P: minus P along "up"
            np.add.at(force, edofs.ravel(), (-P[on][:, None] * fg).ravel())
            els, unit = self._unit_local
            self.member_load = (els, -P[on][:, None] * unit)
        else:
            self.member_load = None
        self._out = force
        self._t_done = float(t)
        out += force
        return out

    def _latch(self, force, on, k, fl, t0):
        """What a step leaves behind besides the state: the force (asked
        twice at one instant, it is handed back), the members' share of the
        tyre loads for their end actions, and the instant."""
        if on.any():
            self.member_load = (self.path.elements[k[on]], -self.contact[on][:, None] * fl[on])
        else:
            self.member_load = None
        self._out = force
        self._t_done = float(t0)

    def _contact_lost(self, t: float, tyres: int, rigid: int):
        self.lift_off += tyres + rigid
        self.lift_tyre += tyres
        self.tension_rigid += rigid
        first = not np.isfinite(self.lift_first)
        if first:
            self.lift_first = float(t)
        diag = getattr(self.solver, "diagnostics", None)
        if isinstance(diag, dict):
            diag.setdefault("vehicle_contact", {})[self.name] = self.contact_report()
        if first:
            what = ("a tyre left the road" if tyres else
                    "a rigid axle pulled the deck up (the model is outside its range)")
            log = getattr(self.solver, "log", None)
            if callable(log):
                log(f"  vehicle       {self.name!r}: {what} at t = {t:.6g} s; counted in the "
                    "run's diagnostics (vehicle_contact)")

    def contact_report(self) -> dict:
        """What `lift_off` counts, for the run report."""
        return {"tyre_lift_off_steps": int(self.lift_tyre),
                "rigid_axle_tension_steps": int(self.tension_rigid),
                "first_time": float(self.lift_first)}

    # --------------------------------------------------------------- report
    def value(self, field: str, axle: int = 0) -> float:
        if field == "heave":
            return float(self.x[0])
        if field == "pitch":
            return float(self.x[1]) if self.pitch else 0.0
        if field == "acceleration":
            return float(self.a[0])
        if field == "contact_force":
            return float(self.contact[axle])
        if field == "wheel":
            if not self.flex[axle]:
                return float(self._surf_prev[axle]) if self._surf_prev is not None else 0.0
            return float(self.x[self.nb + int(np.searchsorted(self.fi, axle))])
        if field == "station":
            return float(self.stations(self._t_done or 0.0)[axle])
        raise ValueError(f"vehicle monitor field {field!r}: heave, pitch, acceleration, "
                         "contact_force, wheel or station")
