"""Form-finding: the unstressed lengths that make the mesh the dead-load state.

A cable bridge is drawn at its design geometry -- the deck level, the cable
sag, the hanger lines -- and that geometry is where the structure must sit
UNDER ITS DEAD LOAD. What is not drawn is how long each cable was cut: its
unstressed length, which fixes the tension it carries there. Form-finding
is the search for those lengths.

THE ITERATION is the target-length one every cable-structure text gives, and
it fits this solver because the solver is a relaxation: relax the model
under the dead-load step with the current unstressed lengths, read every
cable element's equilibrium length L, and reset its unstressed length so
that the SAME tension would hold at the mesh length L0 --

    L_u  <-  L_u * L0 / L

-- then relax again. A member that ended too long (L > L0, hanging low) is
cut shorter; one that ended too short is let out. When every node is back
on the mesh to within the tolerance, the lengths are found. Convergence is
geometric on the systems tried here (five to ten relaxations for a
suspension bridge started from L_u = L0); it can stall on a system whose
dead-load geometry is not a state the cables alone can hold, and then the
tolerance is simply not reached and the caller is told so.

The beams are not touched: a deck at zero displacement under dead load
carries no bending, which is the classic assumption that the girder was
built to profile and the hangers took the weight as it went.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field

import numpy as np

from . import api
from .core.diagnostics import AnalysisFailure, UserError
from .core.structural import make_structural_solver


@dataclass
class FormFound:
    #: group region name -> unstressed length per element (element order)
    unstressed_lengths: dict = field(default_factory=dict)
    #: group region name -> axial force per element at the found state
    forces: dict = field(default_factory=dict)
    iterations: int = 0
    max_displacement: float = float("inf")
    history: list = field(default_factory=list)
    converged: bool = False
    #: the tolerance searched to, m
    tol: float | None = None
    #: the inner solve: "relaxation" (explicit, to equilibrium) or "static"
    #: (Newton) -- what the header and the command line call the iterations
    mode: str = "relaxation"

    @property
    def solves(self) -> str:
        """"relaxations" or "static solutions", for the messages."""
        return "static solutions" if self.mode == "static" else "relaxations"


def _cable_groups(problem, groups):
    members = problem.get("members")
    if not members:
        raise ValueError("form-finding needs a mixed_line model (bar sections on named line "
                         "regions alongside the beams); this case lowers to a single element block")
    by_region = {m["region"]: k for k, m in enumerate(members)}
    out = []
    for g in groups:
        if g not in by_region:
            raise ValueError(f"no member group {g!r}; the case has: {', '.join(by_region)}")
        if members[by_region[g]].get("type") != "truss2":
            raise ValueError(f"member group {g!r} is {members[by_region[g]]['type']}, not a "
                             "cable (truss2)")
        out.append(by_region[g])
    return out


def _elements_of(sol, region):
    """The group's elements in ascending order -- the order `unstressed_lengths`
    is written and read in."""
    return np.asarray([e for e in range(sol.nel) if sol.member_props[e].get("region") == region],
                      dtype=np.int64)


def _relax(sol, problem, it) -> None:
    """One inner run to equilibrium, or a UserError saying it was not one:
    an exhausted budget was a traceback, and a run that ended unconverged
    was taken as the dead-load state (found in review)."""
    try:
        if problem["solver"].get("static_run"):
            from . import implicit
            implicit.run(sol, output_every=10 ** 9, progress_every=10 ** 9)
        else:
            sol.run(output_every=10 ** 9, progress_every=10 ** 9)
    except AnalysisFailure as exc:
        raise UserError("FORMFIND_NOT_RELAXED",
                        f"{'static solution' if problem['solver'].get('static_run') else 'relaxation'} {it} did not reach equilibrium ({exc.status}): {exc}",
                        "give the dead-load step a larger budget (max_steps_per_load, "
                        "n_steps) or check the model with `vfife run` first") from None
    d = getattr(sol, "diagnostics", None) or {}
    if not d.get("converged"):
        raise UserError("FORMFIND_NOT_RELAXED",
                        f"{'static solution' if problem['solver'].get('static_run') else 'relaxation'} {it} ended {d.get('status', 'unconverged')}, not at equilibrium",
                        "give the dead-load step a larger budget (max_steps_per_load, "
                        "n_steps) or check the model with `vfife run` first")


def find(source, groups, tol=1.0e-3, max_iter=25, step_index=0, log=print) -> FormFound:
    """Find the unstressed lengths of `groups` (member region names) so the
    dead-load step leaves the mesh where it is, to `tol` metres.

    `source` is a case path, dict or Case; `step_index` picks the step that
    IS the dead load (the first by default) -- later steps (live load, an
    earthquake) are left out of the search.
    """
    prep = api.prepare_case(source)
    # a dynamic run "converged" by finishing its time: every inner run is a
    # relaxation to equilibrium, or the lengths are fitted to a transient
    # (found in review)
    mode = prep.case.solver.mode
    if mode not in ("relaxation", "static"):
        raise UserError("FORMFIND_MODE",
                        f"form-finding relaxes the dead-load step to equilibrium; this case "
                        f"has solver.mode = {mode}",
                        "set `solver: {mode: relaxation}` (or static) in the case, then form-find")
    base = api._with_element(prep.case, prep.element, prep.case_path)
    idx = _cable_groups(base, groups)
    n_steps = len(base.get("steps") or []) or 1
    if not 0 <= int(step_index) < n_steps:
        raise UserError("FORMFIND_STEP",
                        f"step {step_index}: the case has {n_steps} step(s), numbered 0 to "
                        f"{n_steps - 1}", "--step names the dead-load step by its index")
    problem = copy.deepcopy(base)
    problem["steps"] = [base["steps"][step_index]] if base.get("steps") else None
    if problem["steps"] is None:
        problem.pop("steps")
    if any(m.get("jack") is not None for m in problem.get("members") or []):
        raise UserError("FORMFIND_JACK", "a cable with `jack` is stressed to a force in an analysis step; "
                        "form-finding finds lengths", "give the cables to be found `prestress` (a first guess) "
                        "instead, or keep `jack` and run the steps")
    result = FormFound(tol=float(tol), mode=mode)
    Lu = {}                                     # group index -> array of unstressed lengths
    for it in range(1, max_iter + 1):
        for g, lu in Lu.items():
            m = problem["members"][g]
            for key in ("prestress", "unstressed_length"):
                m.pop(key, None)
            m["unstressed_lengths"] = [float(v) for v in lu]
        sol = make_structural_solver(problem, copy.deepcopy(prep.mesh), prep.material)
        # the inner solver says nothing: the static path's Newton lines went
        # to stdout whatever `log` was; a failure is
        # reported by _relax with its reason
        sol.log = lambda *a, **k: None
        _relax(sol, problem, it)
        u = sol.dofs.view(sol.q)[:, :3]
        moved = float(np.max(np.linalg.norm(u, axis=1)))
        x = sol.mesh.nodes + u
        for g in idx:
            ids = _elements_of(sol, problem["members"][g]["region"])
            L = np.linalg.norm(x[sol.mesh.elements[ids, 1]] - x[sol.mesh.elements[ids, 0]], axis=1)
            L0 = sol.length[ids]
            lu_now = sol.member_Lu[ids]
            Lu[g] = lu_now * L0 / L
            result.forces[problem["members"][g]["region"]] = [float(sol.end_actions[e, 6]) for e in ids]
        result.history.append(moved)
        log(f"  formfind {it:2d}: max displacement {moved:.4e} m")
        if moved <= tol:
            result.converged = True
            # the lengths that PRODUCED this state, not the next guess
            for g in idx:
                ids = _elements_of(sol, problem["members"][g]["region"])
                result.unstressed_lengths[problem["members"][g]["region"]] = [
                    float(v) for v in sol.member_Lu[ids]]
            result.iterations = it
            result.max_displacement = moved
            return result
    result.iterations = max_iter
    result.max_displacement = result.history[-1]
    for g in idx:
        result.unstressed_lengths[problem["members"][g]["region"]] = [float(v) for v in Lu[g]]
    return result


def _rebase(node, src_dir: str, out_dir: str, top: bool = True):
    """Relative file paths in a case dict (mesh, records, road profiles,
    spectra, initial_state), moved from `src_dir` to `out_dir`, in place.
    Only strings under `path`, `profile` or `initial_state`; `output` is
    skipped (its path is relative to the working directory)."""
    if src_dir == out_dir:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if top and k == "output":
                continue
            if k in ("path", "profile", "initial_state") and isinstance(v, str) \
                    and v and not os.path.isabs(v):
                full = os.path.normpath(os.path.join(src_dir, v))
                try:
                    node[k] = os.path.relpath(full, out_dir).replace(os.sep, "/")
                except ValueError:          # Windows, another drive
                    node[k] = full
            else:
                _rebase(v, src_dir, out_dir, top=False)
    elif isinstance(node, list):
        for v in node:
            _rebase(v, src_dir, out_dir, top=False)


def write_found_case(source_path: str, found: FormFound, out_path: str):
    """The input-form case with `unstressed_lengths` on the found cables'
    assignments (and any `prestress` / `unstressed_length` there removed),
    written with the compact formatter. Comments do not survive the
    round trip; the header says where the numbers came from."""
    import yaml
    from .io.yamlfmt import dump_compact
    with open(source_path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    assigns = doc.get("assign") or []
    done = set()
    for a in assigns:
        region = a.get("region")
        if region in found.unstressed_lengths:
            a.pop("prestress", None); a.pop("unstressed_length", None)
            a["unstressed_lengths"] = found.unstressed_lengths[region]
            done.add(region)
    missing = set(found.unstressed_lengths) - done
    if missing:
        raise ValueError(f"no `assign` entry targets region(s) {sorted(missing)} in {source_path}")
    # written into another folder, the case's relative mesh, records and
    # initial state were copied verbatim and no longer found (found in
    # review); `output.path` is relative to the working
    # directory and is left as it is
    _rebase(doc, os.path.dirname(os.path.abspath(source_path)),
            os.path.dirname(os.path.abspath(out_path)))
    if found.converged:
        header = (f"# Form-found from {source_path} (vfife formfind): the\n"
                  f"# `unstressed_lengths` below leave the mesh at its dead-load geometry to\n"
                  f"# {found.max_displacement:.2e} m after {found.iterations} {found.solves}.\n")
    else:
        # it said the lengths reproduce the mesh when they had not
        tol = f" (tolerance {found.tol:.2e} m)" if found.tol is not None else ""
        header = (f"# NOT CONVERGED -- form-finding from {source_path} (vfife formfind) stopped\n"
                  f"# after {found.iterations} {found.solves} with the mesh still "
                  f"{found.max_displacement:.2e} m from its\n"
                  f"# dead-load geometry{tol}. The `unstressed_lengths` below are the last\n"
                  f"# estimate, not form-found lengths.\n")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(header + dump_compact(doc))
    return out_path
