"""Modal analysis: the natural frequencies and mode shapes of a model at a
state, from the finite-difference tangent and the physical mass.

    K phi = omega^2 M phi

K is the tangent of the internal force at the state asked for -- the
central finite difference of `testing.invariants.tangent`, one column per
degree of freedom -- and M is the lumped physical mass the dynamic mode
integrates with (the fictitious relaxation mass is not a mass). Taking K
at a DEFORMED, LOADED state is the whole point for a bridge: the geometric
stiffness of a cable's tension, of a deck's axial force, of a prestressed
girder is in the tangent of the internal force and nowhere else, so the
modes of a suspension bridge are asked at its dead-load state
(`--result`), not at the drawing.

The generalised problem is solved densely: M is diagonal, so with
D = M^(-1/2) the problem is the symmetric one (D K D) psi = omega^2 psi.
Degrees of freedom held by a constraint are removed; degrees of freedom
that carry no mass (a truss node's rotation, a spare particle) are
condensed out statically when they carry stiffness and dropped when they
carry none. A finite-difference tangent costs 2 n force evaluations; this
is for models of a few thousand dofs, which every bridge in the library
is, and it says so past `max_dofs` rather than running for an hour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .testing import invariants
from .io.writer import open_result
from .testing.state import probe as make_probe


@dataclass
class ModalResult:
    frequencies: np.ndarray          #: Hz, ascending
    periods: np.ndarray              #: s
    shapes: np.ndarray               #: (n_modes, ndof), zero on held dofs, unit max amplitude
    participation: np.ndarray        #: (n_modes, 3) effective modal mass fraction in x, y, z
    n_free: int = 0
    n_condensed: int = 0
    n_dropped: int = 0
    total_mass: np.ndarray = field(default_factory=lambda: np.zeros(3))
    note: str = ""
    #: (n_modes, 3) participation factors phi^T M r of the MASS-NORMALISED
    #: modes, signed; participation = gamma^2 / total_mass
    gamma: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: (n_modes,) the mass-normalised mode is `norm * shapes[i]` (the shapes
    #: are scaled to a unit largest translation for drawing)
    norm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: the angular frequencies, rad/s
    omega: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: (n_modes,) modes with no translation (a torsion mode): their shape is
    #: scaled to a unit largest ROTATION
    rotational: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    def table(self) -> str:
        lines = ["  mode    f [Hz]       T [s]     eff. mass x     y       z"]
        rot = self.rotational if len(self.rotational) == len(self.frequencies) else [False] * len(self.frequencies)
        for i, (f, T, part, r) in enumerate(zip(self.frequencies, self.periods, self.participation, rot), 1):
            lines.append(f"  {i:4d}  {f:10.4f}  {T:10.4f}     {part[0]:6.3f}  {part[1]:6.3f}  {part[2]:6.3f}"
                         + ("   rotation only" if r else ""))
        return "\n".join(lines)


def physical_mass(solver, p):
    """The lumped physical mass per dof, (n,)."""
    m = getattr(solver, "physical_mass", None)
    if m is None and hasattr(solver, "_lumped_physical_mass"):
        m = solver._lumped_physical_mass()
    if m is None:
        raise ValueError(f"{type(solver).__name__} carries no physical mass to ask modes of")
    m = np.asarray(m, float).reshape(-1)
    if m.size != p.n:
        raise ValueError(f"physical mass has {m.size} entries for {p.n} dofs")
    return m


def analyse(solver, n_modes=6, x=None, max_dofs=6000, h=None) -> ModalResult:
    """The lowest `n_modes` modes of `solver` at state `x` (its current
    state when None)."""
    from .core import constraints
    ties = constraints.of(solver)
    p = make_probe(solver)
    if p.n > max_dofs:
        raise ValueError(f"{p.n} dofs is past max_dofs = {max_dofs}: the finite-difference "
                         f"tangent would need {2 * p.n} force evaluations")
    x = p.get() if x is None else np.asarray(x, float)
    K, _ = invariants.tangent(p, x, h=h, external=False)
    K = 0.5 * (K + K.T)
    m = physical_mass(solver, p)
    free = np.flatnonzero(~p.fixed)
    Kf = K[np.ix_(free, free)]
    mf = m[free]
    # the multi-point constraints (rigid floors: they were ignored here,
    # every floor node vibrating on its own; rigid links, 1.21)
    T = ties.free_map(p, free, m) if ties is not None and not ties.empty() else None
    import scipy.sparse as sp
    # the mass of members set off their nodes, where it is (1.21): the
    # coupling of each carrying node's translation and rotation
    # (`constraints.CarriedMass`); the mass is then not diagonal
    C = None
    carried = ties.carried if ties is not None else None
    if carried is not None:
        from .testing.state import _orient_of
        r, c, v = carried.coupling(_orient_of(solver, x))
        if len(v):
            C = sp.coo_matrix((v, (r, c)), shape=(p.n, p.n)).tocsr()[free][:, free]
            if T is None:
                T = sp.identity(len(free), format="csr")
    Mf = None
    Q = None
    if T is not None:
        Kf = np.asarray(T.T @ np.asarray((T.T @ Kf).T))
        Kf = 0.5 * (Kf + Kf.T)
        Mfull = sp.diags(mf) if C is None else sp.diags(mf) + C
        Mr = (T.T @ (Mfull @ T)).toarray()
        # THE REDUCED MASS IS NOT DIAGONAL, AND MAY BE SINGULAR: a point mass
        # on a rigid arm gives its master m [[1, a], [a, a^2]] in (w, theta),
        # a direction with no mass that the diagonal does not show, and the
        # generalised eigenproblem then lost 4% on the fundamental (found
        # with the rigid links, 1.21). In the mass's own axes, u_red = Q z,
        # the mass is diagonal and its null directions are massless dofs
        # like any other: condensed below.
        lam, Q = np.linalg.eigh(0.5 * (Mr + Mr.T))
        if lam.min() > 1e-12 * max(float(lam.max()), 1e-300):
            # a reduced mass with mass in every direction (the floors):
            # the generalised eigenproblem below, as before 1.21
            Q = None
            Mf = Mr
            mf = np.diag(Mf).copy()
        else:
            Kf = Q.T @ Kf @ Q
            Kf = 0.5 * (Kf + Kf.T)
            mf = np.maximum(lam, 0.0)
    scale = float(np.abs(Kf).max()) or 1.0
    # massless dofs: condense those with stiffness, drop those without
    massless = mf <= 1e-12 * max(float(mf.max()), 1e-300)
    n_cond = n_drop = 0
    keep = ~massless
    r = np.flatnonzero(keep)
    if massless.any():
        s = np.flatnonzero(massless)
        Kss = Kf[np.ix_(s, s)]
        live = np.abs(Kss).max(axis=1) > 1e-12 * scale
        s_live = s[live]; n_drop = int((~live).sum()); n_cond = int(live.sum())
        if n_cond:
            Kss = Kf[np.ix_(s_live, s_live)]
            Ksr = Kf[np.ix_(s_live, r)]
            Kf = Kf[np.ix_(r, r)] - Ksr.T @ np.linalg.solve(Kss, Ksr)
        else:
            Kf = Kf[np.ix_(r, r)]
        mf = mf[r]
    if Mf is None:
        d = 1.0 / np.sqrt(mf)
        A = d[:, None] * Kf * d[None, :]
        A = 0.5 * (A + A.T)
        w, v = np.linalg.eigh(A)
        n_modes = int(min(n_modes, len(w)))
        order = np.argsort(w)[:n_modes]
        w = w[order]; psi = v[:, order]
        phi = d[:, None] * psi                     # mass-normalised: phi^T M phi = 1
        wscale = scale * float(np.max(d) ** 2)
    else:
        import scipy.linalg
        Mr = Mf[np.ix_(r, r)]
        w, v = scipy.linalg.eigh(Kf, 0.5 * (Mr + Mr.T))
        n_modes = int(min(n_modes, len(w)))
        order = np.argsort(w)[:n_modes]
        w = w[order]; phi = v[:, order]             # phi^T M phi = 1
        wscale = scale / float(np.min(mf))
    omega = np.sqrt(np.maximum(w, 0.0))
    # the shapes over the reduced dofs, then over every dof
    red = np.zeros((n_modes, len(massless)))
    red[:, r] = phi.T
    # THE CONDENSED DOFS MOVE TOO. A massless dof with
    # stiffness is condensed out of the eigenproblem, but it is not held: it
    # follows the kept ones statically, u_s = -K_ss^-1 K_sr u_r. Until this
    # stage the shape was left zero there, which no frequency shows and every
    # animation does -- a node snapping back to where it started.
    if n_cond:
        red[:, s_live] = -np.linalg.solve(Kss, Ksr @ phi).T
    shapes = np.zeros((n_modes, p.n))
    shapes[:, free] = red if T is None else np.asarray(T @ (red.T if Q is None else Q @ red.T)).T
    # participation over the free translational dofs, phi^T M r per
    # direction, in the full coordinates (a floor's slaves included)
    part = np.zeros((n_modes, 3))
    gam = np.zeros((n_modes, 3))
    total = np.zeros(3)
    comp = np.full(p.n, -1)
    comp[p.translation] = np.tile(np.arange(3), p.nn)
    mfull = np.zeros(p.n)
    mfull[free] = m[free]
    for k in range(3):
        rk = (comp == k).astype(float) * mfull
        total[k] = float(rk.sum())                    # r^T M r: the coupling adds none
        if C is not None:
            # M r: a carried mass's translation pulls on its node's rotation too
            rk[free] += C @ (comp[free] == k).astype(float)
        if total[k] > 0:
            gam[:, k] = shapes @ rk                    # phi^T M r
            part[:, k] = gam[:, k] ** 2 / total[k]
    # drawing scale: a unit largest translation -- unless the mode has none
    # (a torsion mode of a column): then the largest component, and the mode
    # is flagged. Dividing by a translation of 1e-16 wrote rotations of 1e13
    norm = np.ones(n_modes)
    rotational = np.zeros(n_modes, bool)
    for i in range(n_modes):
        sh = shapes[i]
        et = float(np.sum(mfull[p.translation] * sh[p.translation] ** 2)) if p.translation.size else 0.0
        e = float(np.sum(mfull * sh ** 2))
        rotational[i] = bool(p.rotation.size) and et <= 1e-8 * e
        pick = sh if rotational[i] else sh[p.translation]
        amp = float(np.abs(pick).max()) if pick.size else 1.0
        norm[i] = amp or 1.0
        shapes[i] = sh / norm[i]
    freq = omega / (2.0 * np.pi)
    with np.errstate(divide="ignore"):
        periods = np.where(freq > 0, 1.0 / np.where(freq > 0, freq, 1.0), np.inf)
    note = ""
    if np.any(w[:n_modes] < -1e-8 * wscale):
        note = "negative eigenvalue(s): the state is not a stable equilibrium (or not converged)"
    return ModalResult(frequencies=freq, periods=periods, shapes=shapes, participation=part,
                       n_free=len(free), n_condensed=n_cond, n_dropped=n_drop,
                       total_mass=total, note=note, gamma=gam, norm=norm, omega=omega,
                       rotational=rotational)


def frame_key(frames, frame, result):
    """The stored name of frame `frame` -- given as its step number, padded
    or not (`3338` or `00003338`); the last frame when None. `--frame 3338`
    was refused because the names are zero-padded."""
    if frame is None or frame == "":
        return frames[-1]
    by_number = {int(k): k for k in frames}
    try:
        return by_number[int(str(frame).strip())]
    except (ValueError, KeyError):
        from .core.diagnostics import UserError
        raise UserError("NO_SUCH_FRAME",
                        f"frame {frame!r} is not in {result}; it has {len(frames)} frames, "
                        f"{int(frames[0])} to {int(frames[-1])}",
                        "give a frame's step number, e.g. --frame " + str(int(frames[-1]))) from None


def state_from_result(p, result, frame=None):
    """The probe state of one frame of a result file (displacement +
    rotation), for asking modes at a loaded, deformed configuration."""
    with open_result(result, "r") as h:
        frames = sorted(h["steps"], key=int)
        frame = frame_key(frames, frame, result)
        g = h["steps"][frame]
        u = g["displacement"][...].astype(float)
        rot = g["rotation"][...].astype(float) if (p.rotation.size and "rotation" in g) else None
    x = p.get()
    x[p.translation] = u.ravel()
    if rot is not None:
        x[p.rotation] = rot.ravel()
    return x, frame, rot is None and p.rotation.size > 0


def frame_time(result, frame) -> float:
    """The `time` a stored frame was written at."""
    with open_result(result, "r") as h:
        return float(h["steps"][frame].attrs.get("time", 0.0))


def use_step_at(solver, t) -> int:
    """Put a freshly built solver on the analysis step a frame at time `t`
    belongs to -- that step's supports, not the first's.
    A step spans (k, k + 1] in a relaxation or static run and
    (k T, (k + 1) T] in a dynamic one. Returns k."""
    steps = getattr(solver, "steps", None) or [{}]
    span = float(getattr(solver, "total_time", 1.0) or 1.0) if getattr(solver, "mode", "") == "dynamic" else 1.0
    k = int(min(max(int(np.ceil(float(t) / span - 1e-9)) - 1, 0), len(steps) - 1))
    if hasattr(solver, "apply_step"):
        solver.apply_step(k)
    elif hasattr(solver, "step_data"):
        solver.step_index = k
        solver.current_fixed = solver.step_data(k)[0]
    return k


def restore_state(solver, case, element, case_path, result, frame=None):
    """Put `solver` in the state of one frame of `result`, for asking modes
    or invariants there. Returns (x, frame, step index,
    full, missing_rotation, note): `x` the probe state to ask at, `full`
    whether the material state came across too.

    When the frame is the one the result's checkpoint holds (the run's last
    state, by its step counter), EVERY state comes across, as
    `solver.initial_state` takes it: positions, rotations, fibre and
    connector-law histories, the creep history and the tendons' relaxed
    prestrain -- and the creep reached is then held at the instantaneous
    modulus (`hold_time_strain`). Before, only the displacement and rotation
    fields were read and every material started virgin: on a bridge on
    friction bearings `vfife modal` gave T1 = 0.868 s where the spectrum of
    the same run, at the same state, had 0.508 s, and `probe` reported a
    relative residual of 66 at a converged frame. Any other frame has no
    stored material state: its positions are used, and the note says so."""
    from types import SimpleNamespace
    from .api import start_from
    from .io import fingerprint as _fp
    from .io.writer import load_checkpoint
    from .core.diagnostics import UserError
    p = make_probe(solver)
    x, frame, missing_rot = state_from_result(p, result, frame)
    held = None
    try:
        held = load_checkpoint(result)
    except UserError:
        held = None
    step_of_frame = int(frame)
    if held is not None and hasattr(solver, "restore") \
            and int(np.asarray(held.data.get("step", [-1])).ravel()[0]) == step_of_frame:
        prepared = SimpleNamespace(case=case, element=element, case_path=case_path,
                                   mesh=solver.mesh)
        fp = None
        try:
            fp = _fp.model_fingerprint(solver, prepared)
        except Exception:                      # a model the fingerprint cannot read
            fp = None
        start_from(solver, result, fingerprint=fp)
        k = use_step_at(solver, frame_time(result, frame))
        if getattr(solver, "time_dependent", False):
            age = float(getattr(solver, "_current_age", 0.0))
            if hasattr(solver, "_step_ages"):
                # any later force evaluation (`probe`'s loads) stays at the age
                # the state was reached at, not the step's starting age
                solver._step_ages = (age, None, "log")
            if hasattr(solver, "hold_time_strain"):
                solver.hold_time_strain()
        return make_probe(solver).get(), frame, k, True, False, None
    from .core import constraints
    ties = constraints.of(solver)
    if ties is not None and ties.links is not None and ties.links.staged():
        # which links tie at this frame, and from where, is the checkpoint's
        # alone (1.21)
        raise ValueError("this model makes or releases rigid links in its steps: ask at the result's last "
                         "state (the frame its checkpoint holds), where the links are known")
    k = use_step_at(solver, frame_time(result, frame))
    note = ("only positions and rotations are read for this frame: the material, creep and "
            "connector histories are stored for the result's last state only (its checkpoint)"
            if held is not None else
            "the result has no checkpoint beside it: only positions and rotations are read, "
            "every material starts virgin")
    return x, frame, k, False, missing_rot, note


def connector_kinks(solver):
    """Connector laws that sit ON a kink at the solver's state -- a friction
    component sliding (|F| = mu N), a bilinear one on its yield surface, a
    gap at its stop. There the central finite-difference
    tangent the modes and the spectrum use is the MEAN of the two branches
    (half the stick stiffness of a sliding bearing), which is neither state;
    a spectrum on such a bearing wants an effective linear stiffness the
    analyst chooses. Returns one line per component."""
    from .elements import connector as _c
    ids = getattr(solver, "_connector_ids", None)
    if ids is None or not len(ids) or not hasattr(solver, "_conn_committed"):
        return []
    out = []
    for e in ids:
        props = solver._connector_props[int(solver._connector_of[e])]
        laws = props.get("laws") or {}
        if not laws:
            continue
        conn = solver.mesh.elements[e][:2]
        dof = solver.dofs.element_dofs(conn)
        qe = np.asarray(solver.q)[dof].reshape(2, 6)
        d = _c.deformation(qe[:, :3], solver.orient[conn], solver.mesh.nodes[conn],
                           solver.R0[e], props)
        K = _c.matrix(props.get("stiffness"))
        slip = np.asarray(solver._conn_committed[e], float)
        for name, law in laws.items():
            i = _c.COMPONENTS.index(name)
            t = law["type"]
            if t in ("friction", "pendulum"):
                k = float(law["stiffness"])
                fy = float(law["mu"]) * max(float(_c._normal_force(law, K, d)), 0.0)
                trial = k * (float(d[i]) - float(slip[i]))
                if fy <= 0.0 or abs(trial) >= fy * (1.0 - 1e-6):
                    out.append(f"element {e} {name} ({t}) is sliding (|F| = mu N = {fy:.4g})")
            elif t == "bilinear":
                k1, k2, fy = float(law["k1"]), float(law["k2"]), float(law["fy"])
                H = k1 * k2 / (k1 - k2) if k1 > k2 else 0.0
                xi = k1 * (float(d[i]) - float(slip[i])) - H * float(slip[i])
                if abs(xi) >= fy * (1.0 - 1e-6):
                    out.append(f"element {e} {name} (bilinear) is on its yield surface (fy = {fy:.4g})")
            elif t == "gap":
                for bound in (law.get("lo"), law.get("hi")):
                    if bound is not None and abs(float(d[i]) - float(bound)) <= 1e-6 * max(abs(float(bound)), 1e-3):
                        out.append(f"element {e} {name} (gap) is at its stop ({float(bound):g})")
    return out


#: the modes file's layout (docs/result_format.md)
MODAL_FORMAT = 1


def write_modes(path, result: ModalResult, nn, stride, times=None):
    """The modes into an HDF5 file: frequencies, periods, participation and
    the shapes as (n_modes, nn, stride) node fields."""
    import h5py
    with h5py.File(path, "w") as h:
        # a file of its own kind, versioned like a result
        from . import __version__
        h.attrs["modal_format"] = MODAL_FORMAT
        h.attrs["vfife_version"] = __version__
        h.attrs["shape_layout"] = "mode,node,dof (dof: ux uy uz [rx ry rz])"
        h.create_dataset("frequency", data=result.frequencies)
        h.create_dataset("period", data=result.periods)
        h.create_dataset("participation", data=result.participation)
        h.create_dataset("shape", data=result.shapes.reshape(len(result.frequencies), nn, stride))
        h.create_dataset("rotation_only", data=np.asarray(result.rotational, bool))
        h.attrs["n_free"] = result.n_free
        h.attrs["note"] = result.note
    return path
