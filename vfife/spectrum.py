"""Response-spectrum analysis.

At the state a relaxation run ends in -- the loaded equilibrium, or the
drawing when nothing is loaded -- the modes of the tangent (`modal.analyse`,
with the geometric stiffness of whatever is prestressed or hanging), and
for every mode i and every spectrum d:

    U_id = phi_i  Gamma_id  S_a,d(T_i) / omega_i^2,   Gamma_id = phi_i^T M r_d

with phi_i mass-normalised and r_d the rigid-body unit vector of the
direction. Any response R linear in the displacement has the modal peak
R_id = R(U_id); the peaks are combined per component

    R_d = sqrt( sum_i sum_j rho_ij R_id R_jd )          (CQC)
    rho_ij = 8 z^2 (1 + r) r^1.5 / ((1 - r^2)^2 + 4 z^2 r (1 + r)^2),  r = w_j / w_i

(Der Kiureghian 1981, one damping ratio z; SRSS is rho = identity) and the
directions by SRSS, R = sqrt(sum_d R_d^2).

What a response is. A displacement is the mode itself. The member end
actions and the support reactions come from the model: the internal force
is evaluated at the state moved by +-h along the mode, the difference is
the response per unit of the mode -- exactly the tangent the modes were
found with, in the solver's own coordinates (rotations as a spin), so a
cable's tension stiffening or a deck's axial force is in the member forces
the same way it is in the frequencies. A reaction is the internal force on
a held dof; the BASE force is summed over the held dofs mode by mode BEFORE
the combination (the combination of a sum is not the sum of combinations).

The spectrum is the user's table; no code spectrum lives here. Periods
outside the table are refused unless `beyond_table: hold`. The modes kept
are the lowest `modes`, or as many as bring every excited direction to
`mass_fraction` of the free translational mass -- and when they do not
reach it (a stiff support region, massless dofs), the fraction reached is
reported, not corrected: there is no missing-mass term.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import modal
from .io.writer import open_result
from .testing.state import probe as make_probe


def cqc_rho(omega: np.ndarray, zeta: float) -> np.ndarray:
    """Der Kiureghian's modal correlation for equal damping, (n, n)."""
    w = np.asarray(omega, float)
    r = w[None, :] / np.maximum(w[:, None], 1e-300)
    z = float(zeta)
    num = 8.0 * z * z * (1.0 + r) * r ** 1.5
    den = (1.0 - r * r) ** 2 + 4.0 * z * z * r * (1.0 + r) ** 2
    rho = np.where(den > 0, num / np.maximum(den, 1e-300), 1.0)
    np.fill_diagonal(rho, 1.0)
    return rho


def combine(R: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """sqrt(R^T rho R) per component, R of shape (n_modes, ...)."""
    R = np.asarray(R, float)
    flat = R.reshape(R.shape[0], -1)
    q = np.einsum("ik,ij,jk->k", flat, rho, flat)
    return np.sqrt(np.maximum(q, 0.0)).reshape(R.shape[1:])


def read_file(path: Path) -> np.ndarray:
    """A spectrum file, checked as strictly as a `table`:
    period and acceleration in the first two columns of each row (spaces,
    tabs or commas; further columns -- Sd, Sv -- are not read),
    `#` comments, periods from 0 or more and increasing, no negative or
    non-finite value. It used to be read after the run, one column read as
    an IndexError and a negative acceleration accepted."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"spectrum file {path} does not exist")
    rows = []
    for k, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.split("#", 1)[0].replace(",", " ").split()
        if not line:
            continue
        try:
            vals = [float(v) for v in line]
        except ValueError:
            raise ValueError(f"spectrum file {path}, line {k}: {' '.join(line)!r} is not two numbers") from None
        if len(vals) < 2:
            raise ValueError(f"spectrum file {path}, line {k}: {len(vals)} value(s); a row is "
                             "`period acceleration`, and further columns are not read")
        # the first two columns: a file with Sd or Sv beside Sa was read so
        # before v1.0 and refused by it
        rows.append(vals[:2])
    t = np.asarray(rows, float).reshape(-1, 2)
    if len(t) < 2:
        raise ValueError(f"spectrum file {path}: {len(t)} row(s); a spectrum is two or more")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"spectrum file {path}: a value is not finite")
    if t[0, 0] < 0 or np.any(np.diff(t[:, 0]) <= 0):
        raise ValueError(f"spectrum file {path}: periods must start at 0 or more and increase")
    if np.any(t[:, 1] < 0):
        raise ValueError(f"spectrum file {path}: spectral accelerations cannot be negative "
                         f"(row {int(np.argmax(t[:, 1] < 0)) + 1})")
    return t


def resolve(spec: Dict[str, Any], base: Optional[Path] = None) -> Path:
    path = Path(spec["path"])
    return base / path if (not path.is_absolute() and base is not None) else path


def check_files(case, case_path=None) -> None:
    """Every spectrum `path` read and checked before the run, not after it."""
    rs = getattr(case, "response_spectrum", None)
    if rs is None:
        return
    base = Path(case_path).resolve().parent if case_path else Path.cwd()
    for sp in rs.spectra:
        if sp.path is not None:
            read_file(resolve({"path": sp.path}, base))


def spectrum_table(spec: Dict[str, Any], base: Optional[Path] = None) -> np.ndarray:
    """(n, 2) periods and accelerations (already scaled)."""
    if spec.get("table") is not None:
        t = np.asarray(spec["table"], float)
    else:
        t = read_file(resolve(spec, base))
    t = t.copy()
    t[:, 1] *= float(spec.get("scale", 1.0))
    return t


def _moved(p, x, u, h):
    """The state x moved by h along the mode u: translations added, the
    rotation part applied as a spin on every point's triad -- the
    coordinates `modal.analyse` found u in."""
    y = np.asarray(x, float).copy()
    y[p.translation] += h * u[p.translation]
    if p.rotation.size:
        from .base.rotations import exp_map
        turn = exp_map(h * u[p.rotation].reshape(p.nn, 3))
        p._rotate(y, None, turn)
    return y


def _bar_axial(solver, y):
    """A truss model's bar axial forces at state y, without side effects --
    it keeps no end actions, and its spectrum wrote none."""
    from .elements import truss
    s = solver
    ue = np.asarray(y, float).reshape(s.nn, 3)[s.mesh.elements]
    f_e, _ = truss.internal_force(s.pre, s.x0_e, ue, s.material, s.state.copy(), s.dt)
    d = (s.x0_e[:, 1] + ue[:, 1]) - (s.x0_e[:, 0] + ue[:, 0])
    return (f_e[:, 1] * d).sum(axis=1) / np.maximum(np.linalg.norm(d, axis=1), 1e-300)


def _responses(solver, p, x, shape, h):
    """Per unit of the mode: the internal force on every dof (for the
    reactions) and the member end actions, by a central difference."""
    if p.family == "truss":
        yp, ym = _moved(p, x, shape, h), _moved(p, x, shape, -h)
        df = (p.internal(yp) - p.internal(ym)) / (2.0 * h)
        return df, (_bar_axial(solver, yp) - _bar_axial(solver, ym)) / (2.0 * h)
    fp = p.internal(_moved(p, x, shape, h))
    ep = np.array(getattr(solver, "end_actions", np.zeros((0, 12))), float, copy=True)
    fm = p.internal(_moved(p, x, shape, -h))
    em = np.array(getattr(solver, "end_actions", np.zeros((0, 12))), float, copy=True)
    df = (fp - fm) / (2.0 * h)
    de = (ep - em) / (2.0 * h) if ep.shape == em.shape and ep.size else None
    return df, de


def analyse(solver, spec: Dict[str, Any], base: Optional[Path] = None, log=print) -> Dict[str, Any]:
    """The response-spectrum peaks at the solver's current state."""
    # the creep reached held at the instantaneous modulus: the last station's
    # 1/(1 + phi_cur) is a sustained-load stiffness, and the modes taken from
    # it were 20-35% long and moved with the station count
    if hasattr(solver, "hold_time_strain"):
        solver.hold_time_strain()
    # a connector law ON its kink is linearised at the mean of its branches
    kinks = modal.connector_kinks(solver)
    for line in kinks:
        log(f"  warning       connector {line}: the spectrum's tangent there is the mean of "
            "its two branches, which is neither state; give that bearing an effective "
            "linear stiffness for a spectrum")
    p = make_probe(solver)
    x = p.get()
    keep_mode = getattr(solver, "mode", None)
    # the modal responses are those of the tangent: nothing may break,
    # yield or be judged while the model is moved along a mode
    if keep_mode is not None:
        solver.mode = "relaxation"
    try:
        res = modal.analyse(solver, n_modes=p.n, x=x, max_dofs=int(spec.get("max_dofs", 6000)))
        dirs = []
        tables = []
        for sp in spec["spectra"]:
            e = np.asarray(sp["direction"], float)
            dirs.append(e / np.linalg.norm(e))
            tables.append(spectrum_table(sp, base))
        dirs = np.asarray(dirs)
        omega = res.omega
        # participation of each mode in each spectrum's direction
        gam = res.gamma @ dirs.T                              # (n_all, n_dir)
        mass_dir = np.array([float(res.total_mass @ (e * e)) for e in dirs])
        eff = gam ** 2 / np.maximum(mass_dir[None, :], 1e-300)
        n_all = len(omega)
        if spec.get("modes"):
            n = min(int(spec["modes"]), n_all)
        else:
            target = float(spec.get("mass_fraction", 0.9))
            cum = np.cumsum(eff, axis=0)
            excited = mass_dir > 0
            reach = [int(np.searchsorted(cum[:, d], target - 1e-12)) + 1
                     for d in range(len(dirs)) if excited[d]]
            n = min(max(reach) if reach else n_all, n_all)
        omega = omega[:n]; gam = gam[:n]; eff = eff[:n]
        if np.any(omega <= 0.0):
            bad = np.flatnonzero(omega <= 0.0) + 1
            raise ValueError(f"mode(s) {', '.join(map(str, bad))} have no stiffness (a mechanism or "
                             "a free body): a spectrum cannot be applied to a rigid-body motion")
        T = 2.0 * np.pi / omega
        Sa = np.zeros((n, len(dirs)))
        for d, t in enumerate(tables):
            out = (T < t[0, 0]) | (T > t[-1, 0])
            if out.any() and spec.get("beyond_table", "refuse") == "refuse":
                bad = np.flatnonzero(out)
                raise ValueError(
                    f"spectrum {d + 1}: the period(s) of mode(s) {', '.join(str(i + 1) for i in bad[:8])}"
                    f"{' ...' if len(bad) > 8 else ''} ({', '.join(f'{T[i]:.3g}' for i in bad[:4])} s) "
                    f"are outside its table ({t[0, 0]:g} to {t[-1, 0]:g} s): extend the table, or "
                    "write `beyond_table: hold` to take the end values")
            Sa[:, d] = np.interp(T, t[:, 0], t[:, 1])
        cum = eff.sum(axis=0)
        # modal amplitude of each mode's unit-amplitude shape, per direction
        amp = (res.norm[:n, None] * gam * Sa) / (omega[:, None] ** 2)
        h = 1e-7 * max(1.0, float(np.abs(p.nodes).max()))
        nn = p.nn
        U = np.zeros((n, len(x)))
        F = np.zeros((n, len(x)))
        E = None
        for i in range(n):
            shape = res.shapes[i]
            U[i] = shape
            df, de = _responses(solver, p, x, shape, h)
            F[i] = df
            if de is not None:
                if E is None:
                    E = np.zeros((n,) + de.shape)
                E[i] = de
        rho = cqc_rho(omega, float(spec.get("damping", 0.05))) if spec.get("combination", "cqc") == "cqc" \
            else np.eye(n)
        fixed = p.fixed
        comp = np.full(len(x), -1)
        comp[p.translation] = np.tile(np.arange(3), nn)
        out: Dict[str, Any] = {"period": T, "frequency": omega / (2 * np.pi), "spectral_acceleration": Sa,
                               "effective_mass": eff, "modal_mass": gam ** 2,
                               "mass_reached": cum, "directions": dirs,
                               "n_modes": n, "n_modes_available": n_all}
        disp2 = np.zeros(len(x)); react2 = np.zeros(len(x)); base2 = np.zeros(3)
        ends2 = None if E is None else np.zeros(E.shape[1:])
        for d in range(len(dirs)):
            a = amp[:, d]
            disp2 += combine(U * a[:, None], rho) ** 2
            Fd = F * a[:, None]
            react2 += combine(np.where(fixed[None, :], Fd, 0.0), rho) ** 2
            base_modal = np.stack([Fd[:, fixed & (comp == k)].sum(axis=1) for k in range(3)], axis=1)
            base2 += combine(base_modal, rho) ** 2
            if E is not None:
                ends2 += combine(E * a.reshape((-1,) + (1,) * (E.ndim - 1)), rho) ** 2
        disp = np.sqrt(disp2); react = np.sqrt(react2)
        out["displacement"] = disp[p.translation].reshape(nn, 3)
        if p.rotation.size:
            out["rotation"] = disp[p.rotation].reshape(nn, 3)
        r = np.where(fixed, react, 0.0)
        out["reaction"] = r[p.translation].reshape(nn, 3)
        out["base_force"] = np.sqrt(base2)
        if ends2 is not None and ends2.ndim == 1:
            out["axial_force"] = np.sqrt(ends2)          # a truss's bars
        elif ends2 is not None:
            ea = np.sqrt(ends2)
            out["end_actions"] = ea
            names = ("axial_force", "shear_y", "shear_z", "torsion", "bending_y", "bending_z")
            for k, name in enumerate(names):
                out[name] = np.maximum(ea[:, k], ea[:, 6 + k])
        for d in range(len(dirs)):
            msg = (f"  spectrum      {d + 1}: direction {np.round(dirs[d], 3).tolist()}, {n} modes "
                   f"reach {cum[d]:.1%} of the mass")
            if not spec.get("modes") and cum[d] < float(spec.get("mass_fraction", 0.9)) - 1e-9:
                msg += f" (all {n_all} available; no missing-mass term)"
            log(msg)
        log(f"  combination   {spec.get('combination', 'cqc').upper()}"
            + (f", damping {float(spec.get('damping', 0.05)):g}" if spec.get("combination", "cqc") == "cqc" else "")
            + f"; base force {np.round(out['base_force'], 1).tolist()}")
        if res.note:
            log(f"  note          {res.note}")
        if kinks:
            out["connector_kinks"] = list(kinks)
        return out
    finally:
        if keep_mode is not None:
            solver.mode = keep_mode
        p.set(x)


def write(path, out: Dict[str, Any], spec: Dict[str, Any]) -> None:
    """Into the result file's `response_spectrum` group."""
    with open_result(path, "a") as h:
        if "response_spectrum" in h:
            del h["response_spectrum"]
        g = h.create_group("response_spectrum")
        # what each axis of each table is: a bare (20, 12)
        # array did not say which column was which
        dims = {"period": "mode", "frequency": "mode", "spectral_acceleration": "mode,direction",
                "effective_mass": "mode,direction", "modal_mass": "mode,direction",
                "mass_reached": "direction", "directions": "direction,xyz",
                "displacement": "node,xyz", "rotation": "node,xyz", "reaction": "node,xyz",
                "base_force": "xyz", "end_actions": "element,action"}
        ends = ",".join(f"{n}_{e}" for e in ("i", "j") for n in
                        ("axial_force", "shear_y", "shear_z", "torsion", "bending_y", "bending_z"))
        for k, v in out.items():
            if isinstance(v, np.ndarray):
                d = g.create_dataset(k, data=v)
                d.attrs["dims"] = dims.get(k, "element")
                if k == "end_actions":
                    d.attrs["columns"] = ends
                if k == "effective_mass":
                    d.attrs["meaning"] = "ratio of the model's mass in that direction"
        g.attrs["n_modes"] = int(out["n_modes"])
        g.attrs["combination"] = str(spec.get("combination", "cqc"))
        g.attrs["damping"] = float(spec.get("damping", 0.05))
        if out.get("connector_kinks"):
            # connector laws linearised ON their kink
            g.attrs["connector_kinks"] = "; ".join(out["connector_kinks"])
