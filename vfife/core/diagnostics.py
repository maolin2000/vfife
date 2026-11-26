"""Unified analysis termination and energy-audit records."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..base.errors import (AnalysisFailure, AnalysisInterrupted, ConvergenceFailure,  # noqa: F401
                           NonFiniteFailure, StopAnalysis, UnstableFailure, UserError)


@dataclass
class EnergyLedger:
    available: bool = True
    reason: str | None = None
    external_work: float = 0.0
    internal_work: float = 0.0
    dissipated: float = 0.0
    _u: np.ndarray | None = None
    _fext: np.ndarray | None = None
    _fint: np.ndarray | None = None
    history: list = field(default_factory=list)

    def update(self, u, fext, fint):
        """Trapezoidal work of `fext` and `fint` over the last increment of
        `u`. THE THREE ARGUMENTS MUST BE OF ONE INSTANT: the forces evaluated
        AT `u`, i.e. the call goes before the position update. Fed after it,
        the forces of step n are paired with the increment of step n+1, and
        that one-step lag drifts on every oscillating mode -- the continuum
        solver lost 18 J of a 91 J budget on the floating beam,
        and the frame, fibre and truss solvers kept the lag until v0.4,
        where it was 15% of the external work of a 3-D seismic run by 6.5 s."""
        if not self.available: return
        u = np.asarray(u, float).reshape(-1); fe = np.asarray(fext, float).reshape(-1); fi = np.asarray(fint, float).reshape(-1)
        if self._u is not None:
            du = u-self._u
            self.external_work += float(np.dot(.5*(self._fext+fe), du))
            self.internal_work += float(np.dot(.5*(self._fint+fi), du))
        self._u, self._fext, self._fint = u.copy(), fe.copy(), fi.copy()

    def damping(self, fdamp, du):
        """The work a velocity-proportional force did over one step:
        `fdamp . du`, where `du` is the step's displacement increment and
        `fdamp` the damping force that produced it. It goes into
        `dissipated`, beside what kinetic damping removes.

        UNTIL v0.4 NO SOLVER CALLED THIS, so Rayleigh and viscous
        damping were booked as ERROR: the Hangzhou Bay seismic runs reported
        a relative energy error of 0.54 and 0.77, and 93-100% of it was the
        alpha M v term's work. Fed with the step's own damping force and
        displacement increment (and the forces of the right instant, see
        `update`), 7 s of the 3-D run stays within 4e-4 and ends at 1e-7."""
        if not self.available: return
        w = float(np.dot(np.asarray(fdamp, float).reshape(-1), np.asarray(du, float).reshape(-1)))
        if np.isfinite(w): self.dissipated += w

    def dissipate(self, value):
        if self.available and np.isfinite(value): self.dissipated += max(float(value), 0.0)

    def checkpoint(self) -> np.ndarray:
        """The running totals and the last instant, flat, for a restart: a
        restarted run began a new ledger at zero, and its audit reported the
        kinetic energy it resumed with as an imbalance -- relative error 1.0."""
        if not self.available:
            return np.zeros(0)
        n = 0 if self._u is None else self._u.size
        head = [self.external_work, self.internal_work, self.dissipated, float(n)]
        tail = [] if self._u is None else [self._u, self._fext, self._fint]
        return np.concatenate([np.asarray(head, float)] + tail)

    def resume(self, arr) -> None:
        arr = np.asarray(arr, float).reshape(-1)
        if not self.available or arr.size < 4:
            return
        self.external_work, self.internal_work, self.dissipated = map(float, arr[:3])
        n = int(arr[3])
        if n and arr.size == 4 + 3 * n:
            self._u, self._fext, self._fint = (arr[4:4 + n].copy(), arr[4 + n:4 + 2 * n].copy(),
                                               arr[4 + 2 * n:].copy())

    def snapshot(self, step, kinetic):
        if not self.available: return
        balance = self.external_work-self.internal_work-float(kinetic)-self.dissipated
        scale = max(abs(self.external_work), abs(self.internal_work)+abs(float(kinetic))+abs(self.dissipated), 1e-30)
        self.history.append({"step": int(step), "external_work": self.external_work,
                             "internal_work": self.internal_work, "kinetic_energy": float(kinetic),
                             "dissipated_energy": self.dissipated,
                             "energy_imbalance": balance, "relative_energy_error": abs(balance)/scale})


def start(mode, backend, resident=False):
    return {"status": "RUNNING", "converged": False, "mode": mode, "backend": backend,
            "step": 0, "load_step": None, "residual": None, "kinetic_energy": None,
            "message": None, "location": {},
            "energy": EnergyLedger(not resident, "GPU-resident energy audit would require per-step field transfers" if resident else None)}


def resume_ledger(solver) -> None:
    """Seed a new run's ledger from the checkpoint it was restored from."""
    saved = getattr(solver, "_restored_energy", None)
    solver._restored_energy = None
    if saved is not None and getattr(solver, "diagnostics", None):
        solver.diagnostics["energy"].resume(saved)
        # and the events the run had recorded: a restarted result had no
        # member_splits / member_failures though its members were broken
        if getattr(solver, "splits", None):
            solver.diagnostics["member_splits"] = solver.splits
        b = getattr(solver, "breaking", None)
        if b is not None and hasattr(b, "diagnostics"):
            b.diagnostics(solver.diagnostics)


def finish(record, status, *, step, residual=None, kinetic=None, load_step=None,
           message=None, location=None, eroded=None):
    record.update(status=status, converged=status in {"CONVERGED", "COMPLETED_DYNAMIC"},
                  step=int(step), residual=None if residual is None else float(residual),
                  kinetic_energy=None if kinetic is None else float(kinetic), load_step=load_step,
                  message=message, location=location or {})
    # how much of the model was removed on the way to this answer -- a result
    # computed on 252 of 256 elements is a different result, and the report
    # is where that belongs
    if eroded is not None:
        record["eroded"] = int(eroded)


def first_nonfinite(**arrays):
    for name, value in arrays.items():
        a = np.asarray(value); bad = np.argwhere(~np.isfinite(a))
        if len(bad): return {"array": name, "index": bad[0].tolist()}
    return {}


def worst_residual(residual, free=None, dofs_per_node=3):
    """Locate the largest unconstrained residual component for failure reports."""
    a = np.asarray(residual, float).reshape(-1)
    candidates = np.arange(a.size) if free is None else np.flatnonzero(np.asarray(free).reshape(-1))
    if not len(candidates):
        return {}
    local = int(np.argmax(np.abs(a[candidates])))
    index = int(candidates[local])
    return {"array": "residual", "index": [index], "node": index // dofs_per_node,
            "dof": index % dofs_per_node, "value": float(a[index])}


def serialise(record):
    out = {k: v for k, v in record.items() if k != "energy"}
    ledger = record["energy"]
    out["energy_available"] = ledger.available; out["energy_unavailable_reason"] = ledger.reason
    out["energy_history"] = ledger.history
    return out


