"""The Python API, version 1: what a script, a notebook or a
front end uses to run vfife.

The names in `__all__` are the API. Within 1.x they keep their signatures
and meaning -- arguments may be added, with defaults, and nothing is taken
away; everything else in the package is internal and may change. The
reference is generated from these docstrings (`python -m vfife reference
--api`, docs/api_reference.md), and a test keeps it in step.

What the functions raise. A mistake in the input -- a name
that is not there, a refused value, a file that is not a case -- is a
`UserError` (a `ValueError` with `code`, `message`, `hint`; the codes are in
docs/error_codes.md). A case file that does not fit the schema raises
pydantic's `ValidationError` (also a `ValueError`); a missing file
`FileNotFoundError`; a law refusing its own numbers a plain `ValueError`.
Catching `ValueError` and `OSError` catches every input mistake. A run that
fails numerically is not raised: `run_case` returns it as an outcome; a run
stopped on purpose raises `AnalysisInterrupted`.

    from vfife import api
    case = api.load_case("cases/case_24g_hzb_static_pushover.yaml")
    rows = api.study(case, {"sections.girder.j": [20.0, 29.5, 40.0]},
                     output_dir="results/j_study",
                     extract=lambda o: {"k_peak": 20 * (max(o.times) - 1)})
"""
from __future__ import annotations

import os
import csv
import itertools
import re
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union
import time
import yaml

__all__ = [
    "API_VERSION", "load_case", "CaseDraft", "prepare_case", "run_case", "study",
    "PreparedAnalysis", "AnalysisOutcome", "case_schema", "read_monitors", "verify",
    # what `run_case` raises when `on_progress` returns False or on Ctrl-C;
    # it was raised and not exported (found in review)
    "AnalysisInterrupted",
    # what every function here raises for a mistake in the input, with a
    # stable `code` (docs/error_codes.md); raised and not exported until
    # v1.0
    "UserError",
]

#: the version of this interface
API_VERSION = 1

from .assemble import build, make_solver, _with_element
from .core.diagnostics import (AnalysisFailure, AnalysisInterrupted, StopAnalysis, UserError,
                               finish as finish_diagnostics, serialise)
from .io import schema
from .io.writer import ResultWriter, read_checkpoint, open_result   # noqa: F401  (read_checkpoint: reachable as vfife.api.read_checkpoint since 1.0)
from .preflight import check as preflight_check


def case_schema() -> dict:
    """JSON Schema of a case file as it is written -- materials, sections,
    assign and the rest -- for GUI property editors, external case
    generators and text editors (`vfife schema` writes it to a file);
    every shipped case validates against it. A number written
    `2e11` is a string to YAML 1.1 and a number to vfife: validate what
    `yaml.safe_load` returns after turning such strings into numbers."""
    # it was the INTERNAL, lowered form (element / material / members),
    # which every loader refuses (found in review); with the
    # field comments as descriptions for an editor
    from .reference import generate_schema
    return generate_schema()


def _input_case_model():
    """The input form (io/model.InputCase) with the keys it hands to the
    internal case unchanged spelled out, and nothing else allowed."""
    from pydantic import ConfigDict, create_model
    from .io.model import InputCase
    internal = {"element", "material", "members", "blocks"}      # made by the lowering
    passthrough = {n: (f.annotation, f) for n, f in schema.Case.model_fields.items()
                   if n not in internal and n not in InputCase.model_fields}

    class _Closed(InputCase):
        model_config = ConfigDict(extra="forbid")

    return create_model("CaseFile", __base__=_Closed, __doc__=InputCase.__doc__, **passthrough)


@dataclass
class PreparedAnalysis:
    """A case validated, its mesh imported and preflighted, no solver built
    yet (`prepare_case`). `preflight["status"]` is PASS or FAIL (warnings
    leave it PASS) and `preflight["issues"]` lists every error and warning; hand the object to `run_case` so the
    mesh is not read twice."""
    case: schema.Case
    raw_case: str
    case_path: str | None
    mesh: Any
    material: Any
    element: str
    preflight: dict


@dataclass
class AnalysisOutcome:
    """What `run_case` returns: never an exit, whatever happened.

    `status` is the diagnostics' word (CONVERGED, COMPLETED_DYNAMIC,
    MAX_STEPS, NONFINITE, PREFLIGHT_FAILED, ...), `converged` whether the
    result is an answer; `result_path` the HDF5 file (written even for a
    failed run, with the diagnostics and the last state), `monitors` every
    monitor's record by name and `times` the instant (or load factor) of
    each record; `message` the failure, if any."""
    status: str
    converged: bool
    result_path: str | None
    sidecars: list[str]
    preflight: dict
    diagnostics: dict | None = None
    monitors: dict[str, list[float]] | None = None
    wall_time: float | None = None
    message: str | None = None
    times: list[float] | None = None

    def as_dict(self):
        """Every field as plain data, for JSON."""
        return asdict(self)


# ----------------------------------------------------------------- drafts
_PART = re.compile(r"([^.\[\]]+)|\[(-?\d+)\]")


def _parts(key: str) -> List[Union[str, int]]:
    """`sections.girder.j`, `loads[0].vector[2]` -> ['sections', 'girder',
    'j'] / ['loads', 0, 'vector', 2]. A bare number after a dot is an index
    too (`loads.0.vector.2`)."""
    out: List[Union[str, int]] = []
    for name, index in _PART.findall(key):
        if index:
            out.append(int(index))
        elif name.lstrip("-").isdecimal():
            out.append(int(name))
        else:
            out.append(name)
    if not out or "".join(str(p) for p in out) == "":
        raise KeyError(f"{key!r} names nothing")
    return out


def _plain(x: Any) -> Any:
    """A copy that shares nothing and holds only YAML's own types.

    NOT `copy.deepcopy`: that keeps within the copy whatever the source
    shared, and `yaml.safe_load` shares every `&anchor`/`*alias` and `<<:`
    merge -- so changing one tendon's friction in case_24d changed all 66,
    and one section's prestress in case_30 changed its sibling's (found in
    review). numpy scalars and arrays become Python numbers
    and lists, which the result file's YAML can hold."""
    if isinstance(x, Mapping):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    tolist = getattr(x, "tolist", None)            # numpy scalar or array
    if tolist is not None and type(x).__module__ == "numpy":
        return _plain(tolist())
    return x


def _lookup(node: Any, part: Union[str, int]) -> Any:
    """node[part], and a mapping's string key when `part` came out a number
    (a section called `100`)."""
    try:
        return node[part]
    except (KeyError, IndexError, TypeError):
        if isinstance(node, Mapping) and isinstance(part, int) and str(part) in node:
            return node[str(part)]
        raise


class CaseDraft:
    """A case as data: read from a file, changed by dotted keys, run again.

    It remembers the folder the case came from, so a changed copy still
    finds its mesh, records and initial state by the paths the file wrote
    -- which a plain dict handed to `run_case` does not (its relative paths
    resolve against the working directory). `output.path` is the one path
    that is not the case's: as for `vfife run`, it is relative to the
    working directory.

        case = load_case("pc_beam.yaml")             # the manual's chapter 3 case
        case.get("materials.concrete.fpc")           # 4.0e7
        stiffer = case.with_changes({"materials.concrete.fpc": 5.0e7,
                                     "loads[0].line_load[2]": -3.0e4})
        run_case(stiffer, output="results/pc_beam_fpc50.h5")

    Keys are dotted paths through the YAML; `[i]` (or `.i`) indexes a list.
    Only the LAST part of a key may be new -- `solver.threads` on a solver
    block that does not have one -- so a misspelt section or material name
    is an error here, naming what exists, and a misspelt field is refused
    by the schema when the draft is prepared.
    """

    def __init__(self, data: Mapping[str, Any], base: Union[str, Path, None] = None,
                 name: Optional[str] = None):
        self.data: Dict[str, Any] = _plain(dict(data))
        self.base = Path(base).resolve() if base is not None else Path.cwd()
        self.name = name or str(self.data.get("name") or "case")

    @property
    def path(self) -> Path:
        """Where the case would be, for resolving its relative paths: in
        `base`, under the case's own name. The file need not exist. The
        name is only a label: a `/` in it no longer moves the folder."""
        stem = re.sub(r"[\\/:]", "_", self.name) or "case"
        return self.base / f"{stem}.yaml"

    def get(self, key: str) -> Any:
        """The value at a dotted key."""
        node: Any = self.data
        walked = []
        for part in _parts(key):
            try:
                node = _lookup(node, part)
            except (KeyError, IndexError, TypeError):
                raise KeyError(self._missing(key, walked, part, node)) from None
            walked.append(part)
        return node

    def set(self, key: str, value: Any) -> CaseDraft:
        """Change one value in place; returns the draft, so calls chain."""
        parts = _parts(key)
        node: Any = self.data
        for i, part in enumerate(parts[:-1]):
            try:
                node = _lookup(node, part)
            except (KeyError, IndexError, TypeError):
                raise KeyError(self._missing(key, parts[:i], part, node)) from None
        last = parts[-1]
        if isinstance(node, dict) and isinstance(last, int):
            last = str(last) if str(last) in node or last not in node else last
        if isinstance(last, int):
            if not isinstance(node, list) or not -len(node) <= last < len(node):
                raise KeyError(self._missing(key, parts[:-1], last, node))
        elif not isinstance(node, dict):
            raise KeyError(f"{key!r}: {'.'.join(map(str, parts[:-1])) or 'the case'} is not a mapping")
        node[last] = _plain(value)
        return self

    def with_changes(self, changes: Mapping[str, Any], name: Optional[str] = None) -> CaseDraft:
        """A copy with `changes` ({dotted key: value}) applied; this draft is
        left as it was."""
        new = CaseDraft(self.data, self.base, name or self.name)
        for k, v in changes.items():
            new.set(k, v)
        if name:
            new.data["name"] = name
        return new

    def to_yaml(self) -> str:
        """The draft as YAML text (comments are not kept: it is data)."""
        return yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True)

    def write(self, path: Union[str, Path]) -> Path:
        """The draft as a case file. Relative paths in it are written as
        they are, so write it into `base` (or a folder beside the mesh) if
        it is to be run from the file."""
        path = Path(path)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    @staticmethod
    def _missing(key, walked, part, node) -> str:
        where = ".".join(map(str, walked)) or "the case"
        if isinstance(node, dict):
            have = ", ".join(map(str, list(node)[:12])) + (" ..." if len(node) > 12 else "")
            return f"{key!r}: {where} has no {part!r} (it has: {have})"
        if isinstance(node, list):
            return f"{key!r}: {where} has {len(node)} entries; [{part}] is not one of them"
        return f"{key!r}: {where} is a value, not a mapping or a list"

    def __repr__(self):
        return f"CaseDraft({self.name!r}, base={str(self.base)!r})"


def load_case(path: Union[str, Path]) -> CaseDraft:
    """A case file as a `CaseDraft`, to change and run."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise UserError("NOT_A_CASE", f"{path} is not a case (no mapping at the top level)",
                        "a case file is YAML with `mesh`, `materials`, `sections`, ...")
    return CaseDraft(data, path.parent, data.get("name") or path.stem)


def prepare_case(source: str | Path | dict | schema.Case | CaseDraft) -> PreparedAnalysis:
    """Validate, import the mesh and run preflight, without building a
    solver. `source` is a case file, a `CaseDraft`, a dict (its relative
    paths then resolve against the working directory) or a `schema.Case`."""
    if isinstance(source, CaseDraft):
        case = schema.from_dict(source.data, str(source.base))
        raw = source.to_yaml()
        path = str(source.path)
        mesh, material, element = build(case, path)
        report = preflight_check(case, mesh, element)
        return PreparedAnalysis(case, raw, path, mesh, material, element, report)
    if isinstance(source, schema.Case):
        case=source; raw=yaml.safe_dump(case.as_dict(),sort_keys=False); path=None
    elif isinstance(source, dict):
        case=schema.from_dict(source); raw=yaml.safe_dump(source,sort_keys=False); path=None
    else:
        path=str(Path(source).resolve()); case,raw=schema.load(path)
    mesh,material,element=build(case,path)
    report=preflight_check(case,mesh,element)
    return PreparedAnalysis(case,raw,path,mesh,material,element,report)


def available_fields(solver) -> set[str]:
    """The field names THIS model can actually emit.

    Asked of the solver rather than tabulated. There are eight `nodal_fields`
    implementations -- solid, truss, beam, fibre beam, shell, shell+line,
    coupled, mixed structural -- and what each offers depends on its element
    AND its material (`output_fields` differs per material). Any static list
    would be a ninth place to forget, which is how `wedge15` was mis-spelled
    for months.

    So the model is asked directly: evaluate one frame's fields on zeroed
    stress, and take the keys. It costs one pass over the elements, once,
    before the first step.
    """

    names: set[str] = set()
    with _as_if_running(solver):
        for recover in (False, True):
            try:
                sig = _zero_stress(solver)
                names |= set(solver.nodal_fields(sig, recover_stress=recover))
            except Exception:
                # a solver that cannot be probed must not block the run; the
                # writer still drops anything unmatched, as it always did
                continue
    return names


@contextmanager
def _as_if_running(solver):
    """Lend the solver the state it only acquires once stepping.

    Reaction output is gated on a `last_reaction*` attribute that every solver
    creates when it first computes reactions -- i.e. during the run, after this
    probe. Probing the bare solver therefore reported `reaction_force` as
    unavailable and would have refused a case that produces it perfectly well,
    which is worse than the silent drop it replaced.

    So the probe walks the same branch the run will, then puts the solver back
    exactly as it was.
    """
    import numpy as _np

    lent = []                               # (attribute, had it, its value)

    def lend(attr, value):
        lent.append((attr, hasattr(solver, attr), getattr(solver, attr, None)))
        setattr(solver, attr, value)

    nn = getattr(solver, "nn", None)
    if nn:
        for attr in ("last_reaction", "last_reaction_force", "last_reaction_moment"):
            if not hasattr(solver, attr):
                lend(attr, _np.zeros((nn, 3)))
    nel = getattr(solver, "nel", None)
    if nel and _is_line_solver(solver) and not _is_shell(solver):
        # the line solvers' fields read the members' end actions, which the
        # first force evaluation creates; without them the probe raised,
        # was swallowed, and the output-field check was skipped for every
        # beam, fibre and mixed model (found in review)
        for attr, shape in (("end_actions", (nel, 12)), ("connector_deformation", (nel, 6))):
            if getattr(solver, attr, None) is None:
                lend(attr, _np.zeros(shape))
    # the composite solvers report their parts' fields only once the parts
    # have results: a shell's stiffeners (`line_actions`) and a coupled
    # model's solid (`solid_stress`)
    from contextlib import ExitStack
    with ExitStack() as parts:
        for part, attr in (("line", "line_actions"), ("solid", "solid_stress")):
            sub = getattr(solver, part, None)
            if sub is not None and getattr(solver, attr, None) is None and hasattr(sub, "nodal_fields"):
                parts.enter_context(_as_if_running(sub))
                lend(attr, _zero_stress(sub))
        try:
            yield
        finally:
            for attr, had, old in reversed(lent):
                if had:
                    setattr(solver, attr, old)
                else:
                    try:
                        delattr(solver, attr)
                    except AttributeError:
                        pass


def _is_shell(solver) -> bool:
    return any(c.__name__ == "ShellSolver" for c in type(solver).__mro__)


def _is_line_solver(solver) -> bool:
    return type(solver).__name__ != "TrussSolver" and (hasattr(solver, "member_kind") or type(solver).__name__ in (
        "BeamSolver", "FiberBeamSolver", "MixedStructuralSolver"))


def _zero_stress(solver):
    """A stress argument shaped the way this solver's own force call returns."""
    import numpy as _np

    if _is_shell(solver) and getattr(solver, "nel", None):
        return _np.zeros((solver.nel, 8))                 # membrane, moment, shear resultants
    if _is_line_solver(solver) and getattr(solver, "nel", None):
        return _np.zeros((solver.nel, 6))                 # member actions

    blocks = getattr(solver, "blocks", None)
    if blocks:
        sig = [_np.zeros((b.n_elem, b.n_gp, 6)) for b in blocks]
        return sig[0] if len(sig) == 1 else sig
    n_gp = getattr(solver, "n_gp", 1)
    return _np.zeros((getattr(solver, "nel", 1), n_gp, 6))


#: what a run writes when its case names no `output.fields` (1.1), each where
#: the model produces it. Until 1.1 the default was the solid's [displacement,
#: von_mises] for every model, so a frame, a fibre bridge or a truss that
#: named no fields wrote its displacements and nothing else -- no rotations,
#: no member forces, no reactions -- and the forces were found missing after
#: the run. Signed member forces, not end actions or envelopes; no fibre
#: stresses (a field per fibre per frame is asked for by name).
_DEFAULT_FIELDS = ("displacement", "rotation", "von_mises", "von_mises_top", "von_mises_bottom",
                  "reaction_force", "reaction_moment",
                  "axial_force", "shear_y", "shear_z", "torsion", "bending_y", "bending_z",
                  "hinge_rotation", "hinge_usage")
#: written by default only when the model has the thing they describe
_MEMBER_DEFAULTS = ("axial_force", "shear_y", "shear_z", "torsion", "bending_y", "bending_z")
_CONNECTOR_DEFAULTS = ("connector_force", "connector_moment", "connector_deformation")
_FAILURE_DEFAULTS = ("member_failed", "capacity_ratio")


def _output_fields(solver, case) -> List[str]:
    """The fields a run of `case` on `solver` writes: the case's own list when
    it names one, otherwise `_DEFAULT_FIELDS` and, where the model has them,
    its connectors' forces and its members' failure state (1.1)."""
    output = getattr(case, "output", None)
    if output is None:
        return list(_DEFAULT_FIELDS)
    if "fields" in output.model_fields_set:
        return list(output.fields)
    have = available_fields(solver)
    if not have:                       # could not probe: the 1.0 default
        return list(output.fields)
    out = [f for f in _DEFAULT_FIELDS if f in have]
    if "axial_force" not in have:
        # a shell's `shear_y` is a resultant of its own group, not a member
        # force: written by itself it came without `shear_x`
        out = [f for f in out if f not in _MEMBER_DEFAULTS]
    kinds = getattr(solver, "member_kind", None)
    if kinds is not None and "connector2" in set(kinds):
        out += [g for g in _CONNECTOR_DEFAULTS if g + "_x" in have or g + "_ux" in have]
    breaking = getattr(solver, "breaking", None)
    if breaking is not None and (breaking.can_break or bool(getattr(breaking, "hinged", [False]).any())):
        out += [f for f in _FAILURE_DEFAULTS if f in have]
    if "member_built" in have:
        out.append("member_built")         # a model whose steps build and remove (1.2)
    return out


def check_output_fields(solver, requested) -> None:
    """Refuse a field name the run could never produce.

    `output.fields` was filtered by name at write time and anything unmatched
    was dropped in silence. Writing `von_misses` for `von_mises` gave a preflight
    PASS with no warnings, a full analysis, and a result file with the field
    quietly absent -- discovered in ParaView, after the run. On a long analysis
    that is an hour spent on a typo, and it breaks this project's own rule that
    a mistake is refused rather than ignored.
    """
    import difflib

    output = getattr(requested, "output", None)
    monitor_fields = {}
    if output is not None:
        # a case: only the fields its author wrote are checked. The default
        # list (displacement, von_mises) is the solid's, and a frame that
        # never named its fields must not be refused for it
        requested = output.fields if "fields" in output.model_fields_set else []
        # an element monitor on a line model reads one of these fields:
        # a name the model does not produce is refused
        # here, not recorded as NaN for the whole run
        if getattr(solver, "_monitor_members", None):
            monitor_fields = {n: m.field for n, m in (output.monitors or {}).items()
                              if m.kind == "element"}
    requested = [f for f in (requested or [])]
    if not requested and not monitor_fields:
        return
    from .elements.library import FIELD_GROUPS
    have = available_fields(solver)
    if not have:                       # could not probe; leave it to the writer
        return
    bad = [f"monitor {n!r}: field {f!r}" for n, f in monitor_fields.items()
           if not f or f not in have or f in FIELD_GROUPS]
    if bad:
        raise UserError(
            "UNKNOWN_MONITOR_FIELD",
            "; ".join(bad) + " -- an element monitor reads one element field this model writes",
            "fields this model writes: " + ", ".join(sorted(have)))
    if not requested:
        return
    # a group name (`end_actions`) is available when any of its fields is
    have |= {g for g, members in FIELD_GROUPS.items() if have.intersection(members)}
    unknown = [f for f in requested if f not in have]
    if not unknown:
        return
    lines = []
    for f in unknown:
        close = difflib.get_close_matches(f, have, n=1)
        lines.append(f"{f!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    raise UserError(
        "UNKNOWN_OUTPUT_FIELD",
        "output.fields names " + ", ".join(lines)
        + ", which this model does not produce",
        "available for this element and material: "
        + ", ".join(sorted(have))
        + ".  `vfife describe <element|material>` lists what each contributes.",
    )


def start_from(solver, path, fingerprint: Optional[dict] = None) -> None:
    """Put `solver` in the final state of the result at `path`, at rest, at
    t = 0 (`solver.initial_state`).

    This is `--restart` with the clock and the velocities taken away. A
    restart RESUMES a run -- same mode, same step counter, its velocities --
    while this STARTS one from where another ended, which is what a dynamic
    run from a relaxed dead state needs: the relaxation's velocities are
    fictitious and its counter counts iterations, not time. Positions,
    rotations and every material state come across (the checkpoint is the one
    `--restart` uses); the mesh must be the same mesh, or the state would be
    laid onto different particles."""
    import numpy as np
    from .io.writer import load_checkpoint, checkpoint_path
    from .io import fingerprint as _fp
    path = str(path)
    held = load_checkpoint(path)
    if held is None:
        raise UserError("NO_CHECKPOINT",
                        f"initial_state {path!r} holds no final state: there is no checkpoint "
                        f"file {os.path.basename(checkpoint_path(path))} beside it",
                        "the checkpoint is written at the end of every run, beside its result; "
                        "copy the two files together")
    ck = held.data
    if not (hasattr(solver, "restore") and hasattr(solver, "v")):
        raise ValueError(f"{type(solver).__name__} cannot start from an initial_state")
    ours = np.asarray(solver.mesh.nodes)
    if held.legacy:
        # a result of format 2 or older: the checkpoint has no fingerprint,
        # and the result's mesh is the evidence
        with open_result(path, "r") as f:
            nodes = np.asarray(f["mesh/nodes"]) if "mesh/nodes" in f else None
        if nodes is None or nodes.shape[1] != ours.shape[1] or len(nodes) > len(ours) \
                or not np.allclose(nodes, ours[:len(nodes)], rtol=0.0,
                                   atol=1e-9 * max(1.0, float(np.ptp(ours)))):
            raise ValueError(f"initial_state {path!r} was computed on a different mesh "
                             "than this case's; a state is a set of particles and cannot "
                             "be moved onto others")
    if fingerprint is not None:
        _fp.check(held.meta, ck, fingerprint, _fp.own_state(solver), "initial_state", path, nodes=ours)
    solver.restore(ck)
    solver.v = np.zeros_like(np.asarray(solver.v))
    for name in ("step", "step_start", "step_index", "n_resets"):
        if hasattr(solver, name):
            setattr(solver, name, 0)
    if hasattr(solver, "_half"):
        solver._half = True             # the first step takes half an acceleration
    if hasattr(solver, "ke_prev"):
        solver.ke_prev = 0.0
    # at rest means every particle: a bonded bar's own velocity came across
    for bt in getattr(solver, "bonded", None) or []:
        bt.v_bar[...] = 0.0
    # a new run, not the old one resumed: no step to pick up mid-way, no
    # ledger to carry on, no stage or level to re-enter
    for name in ("_resuming", "_resume"):
        if hasattr(solver, name):
            setattr(solver, name, False if name == "_resuming" else None)
    for name in ("_restored_stage", "_restored_bc_start", "_restored_energy"):
        if hasattr(solver, name):
            setattr(solver, name, None)
    staging = getattr(solver, "staging", None)
    if staging is not None and getattr(staging, "resumed_time", None) is not None:
        staging.resumed_time = None              # the step a resumed run was on
    solver.initial_state = path


def _finite(ck) -> bool:
    """The motion a checkpoint holds (positions, velocities, orientations)
    is finite. Only those: a material or member array may hold NaN by design
    (a member without a capacity)."""
    import numpy as np
    for key in ("q", "v", "orient"):
        if key in ck:
            a = np.asarray(ck[key])
            if a.dtype.kind in "fc" and a.size and not np.all(np.isfinite(a)):
                return False
    return True


@contextmanager
def _stop_on_signals(stop: dict, log=print):
    """SIGINT and SIGTERM set `stop["signal"]`, which the progress callback
    turns into a stop AT A STEP BOUNDARY: a Ctrl-C used to
    raise inside whatever bytecode was running, and the checkpoint of a
    half-finished step resumed slightly wrong; a SIGTERM (a cluster's time
    limit) killed the process with the result unclosed. A second signal
    stops at once. Only in the main thread, where Python delivers signals."""
    import signal
    import sys
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old = {}

    def handler(signum, _frame):
        if stop.get("signal"):
            raise KeyboardInterrupt
        stop["signal"] = signum
        print("\n  stopping at the next progress check (the checkpoint is then exact); "
              "interrupt again to stop at once", file=sys.stderr, flush=True)
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            old[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):
            pass
    try:
        yield
    finally:
        for sig, h in old.items():
            try:
                signal.signal(sig, h)
            except (ValueError, OSError):
                pass


def _monitor_unit(case, name) -> str:
    """The unit of a monitor in the case's declared units ('' when they are
    not declared or the quantity has none): `vfife plot` labels with it.
    It said [m] and [N] whatever the case was in."""
    units = dict(getattr(case, "units", None) or {})
    spec = (getattr(case.output, "monitors", None) or {}).get(name)
    if spec is None or not units:
        return None              # not recorded: a reader assumes the SI the manual recommends
    # the length or force `stress` implies (core/units.complete, 1.0 re-review)
    from .core.units import complete
    units = complete(units)[0]
    kind, field = spec.kind, (spec.field or "")
    # a member force of a line model, its envelope or an end action
    # (1.0 re-review): force for the forces, force x length for the moments
    base = field
    for suffix in ("_max_abs", "_i", "_j"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    moment = units.get("moment") or (f"{units['force']}*{units['length']}"
                                     if units.get("force") and units.get("length") else "")
    if kind == "element" and base in ("axial_force", "shear_y", "shear_z"):
        return units.get("force", "")
    if kind == "element" and base in ("torsion", "bending_y", "bending_z"):
        return moment
    if kind == "displacement":
        return units.get("length", "")
    if kind == "reaction":
        return units.get("force", "")
    if field in ("stress", "von_mises", "axial_stress"):
        return units.get("stress", "")
    if field in ("axial_force", "contact_force", "force"):
        return units.get("force", "")
    if kind == "vehicle":
        if field in ("wheel", "station", "heave"):
            return units.get("length", "")
        if field == "pitch":
            return "rad"
        if field == "acceleration" and units.get("length") and units.get("time"):
            return f"{units['length']}/{units['time']}^2"
    return ""


_RETIRED_SWITCHES = ("VFIFE_COMPILED_BARS", "VFIFE_COMPILED_CONNECTORS", "VFIFE_TANGENT_MEMBERS")


def _warn_retired_switches():
    """A switch that went in 1.18 no longer does anything:
    say so once, rather than let a comparison silently compare a path with
    itself."""
    import warnings
    for name in _RETIRED_SWITCHES:
        if name in os.environ:
            warnings.warn(f"{name} no longer has an effect (removed in 1.18): the compiled path is the only one; "
                          "`backend: numpy` is the NumPy reference", stacklevel=3)


def run_case(source, *, output: str | None = None, restart: str | None = None,
             on_ready: Callable | None = None,
             on_progress: Callable | None = None, on_frame: Callable | None = None,
             progress_every: int = 500) -> AnalysisOutcome:
    """Run one analysis; returns an `AnalysisOutcome`, never exits.

    `source` is anything `prepare_case` takes, or its `PreparedAnalysis`.
    `output` overrides the case's `output.path`; `restart` continues from a
    result's checkpoint. `on_ready(solver, prepared)` is called once the
    solver is built, `on_progress(step, n, residual, kinetic, extra)` every
    `progress_every` steps (return False to stop, keeping the result), and
    `on_frame(step, t, fields)` at every output frame. A numerical failure
    comes back as the outcome's status and message, with its diagnostic
    result written. A stop -- `on_progress` returning False, Ctrl-C or
    SIGTERM, honoured at the next progress check -- raises
    `AnalysisInterrupted` after the result and checkpoint are saved;
    its `result_path`, `sidecars` and `report` (the diagnostics) say what
    was kept, and `--restart` / `restart=` resumes from it. The checkpoint
    is its own file beside the result, `<result>.ckpt.h5`,
    and a restart or an initial_state from another model is refused
    (`CHECKPOINT_MISMATCH`)."""
    _warn_retired_switches()
    # Accepting a prepared model lets a CLI/GUI show preflight information
    # before the run without importing and checking the mesh a second time.
    prepared = source if isinstance(source, PreparedAnalysis) else prepare_case(source)
    if prepared.preflight["status"] != "PASS":
        return AnalysisOutcome("PREFLIGHT_FAILED",False,None,[],prepared.preflight,
                               message="model preflight failed")
    from .io.writer import load_checkpoint, checkpoint_path, refuse_checkpoint_name
    refuse_checkpoint_name(output or prepared.case.output.path)
    p=_with_element(prepared.case,prepared.element,prepared.case_path)
    solver=make_solver(p,prepared.mesh,prepared.material)
    # A RESTART IGNORES `initial_state`: the checkpoint is the state, and the
    # case that started from a relaxed dead state is the case its stopped run
    # resumes with. It was refused ("not both"), and the stop message's own
    # `--restart` failed for every scenario started from a dead state
    # (found in review)
    init = None if restart else prepared.case.solver.initial_state
    from .io import fingerprint as _fp
    static_run = bool(p["solver"].get("static_run"))
    # the model this run is, before anything is restored or split:
    # its checkpoints carry it, and a restore compares it
    fingerprint = _fp.model_fingerprint(solver, prepared, static_run)
    out_path = os.path.abspath(output or prepared.case.output.path)
    restarted_at = None
    restarted_time = None
    restarted_from = None

    def _same(a, b):
        a = os.path.abspath(str(a))
        return a == os.path.abspath(str(b)) or (
            os.path.exists(a) and os.path.exists(str(b)) and os.path.samefile(a, str(b)))
    if restart:
        held = load_checkpoint(restart)
        if _same(restart, out_path) or (held is not None and _same(held.path, checkpoint_path(out_path))):
            # the writer opens its file anew, so the history up to the
            # checkpoint was erased and a contract read the tail as the whole
            # run; and its first checkpoint would replace the
            # one being resumed
            raise UserError("RESTART_OVERWRITES",
                            "the restarted run would write over the result it restarts from, "
                            "erasing the history before the checkpoint",
                            "give the continuation its own file: `vfife run <case> --restart "
                            "<old.h5> -o <new.h5>` (api: output=...)")
        if held is None:
            raise UserError("NO_CHECKPOINT",
                            f"no checkpoint for {restart}: there is no "
                            f"{os.path.basename(checkpoint_path(restart))} beside it",
                            "a run writes its checkpoint beside its result at the end (and every "
                            "output.checkpoint_every steps); copy the two files together")
        if held.legacy:
            raise UserError("RESTART_OLD_CHECKPOINT",
                            f"{restart} keeps its checkpoint inside the result (result format 2 or "
                            "older), with no record of which model it belongs to, so it cannot be "
                            "resumed safely",
                            "start a new run from it with `solver.initial_state: <that result>` "
                            "(the state at rest, t = 0), or run the case again from the start")
        _fp.check(held.meta, held.data, fingerprint, _fp.own_state(solver), "restart", held.path)
        solver.restore(held.data)
        restarted_at = int(getattr(solver, "step", 0))
        # the clock of the event tables (member_failures, member_splits),
        # which only a dynamic run fills: step x dt, its global time
        restarted_time = (float(restarted_at) * float(getattr(solver, "dt", 0.0) or 0.0)
                          if getattr(solver, "mode", "") == "dynamic" else None)
        # the RESULT resumed, also when its checkpoint was named
        restarted_from = os.path.abspath(str(restart))
        if restarted_from.endswith(".ckpt.h5") and held.meta.get("result"):
            restarted_from = os.path.join(os.path.dirname(restarted_from), str(held.meta["result"]))
    elif init:
        base = Path(prepared.case_path).parent if prepared.case_path else Path.cwd()
        init_path = init if Path(init).is_absolute() else base / init
        if _same(init_path, out_path) or _same(checkpoint_path(init_path), checkpoint_path(out_path)):
            # the writer opens its file anew: the state this run starts from
            # was erased by the run
            raise UserError("INITIAL_STATE_OVERWRITES",
                            f"output.path {out_path} is the result this run starts from "
                            f"(solver.initial_state): writing it would erase that state",
                            "give this run its own output.path (or -o <new.h5>)")
        start_from(solver, init_path, fingerprint=fingerprint)
    check_output_fields(solver, prepared.case)
    if prepared.case._tendon_report:
        # each tendon's bars, one member each, before the result is opened
        from .io.writer import tendon_rows
        try:
            tendon_rows(prepared.case._tendon_report, prepared.mesh)
        except ValueError as exc:
            raise UserError("HIGH_LEVEL_BLOCK", str(exc), "see the manual, 11.14 (`tendons`)") from None
    if on_ready:
        on_ready(solver, prepared)
        from .core.protocol import refuse_old_names
        refuse_old_names(solver)
    path=output or prepared.case.output.path
    writer=ResultWriter(path,prepared.mesh,case_yaml=prepared.raw_case,
        fields=set(_output_fields(solver, prepared.case)),element=prepared.element,rebar=solver.rebar,
        time_axis=(solver.staging.time_axis or
                   ("time" if getattr(solver,"mode","")=="dynamic" and not static_run else "load_factor")),
        units=dict(prepared.case.units or {}),
        restarted_from=restarted_from,
        restarted_at_step=restarted_at, restarted_at_time=restarted_time)
    # the tendons of `tendons:`, for `vfife tendons`
    if prepared.case._tendon_report:
        writer.write_tendons(prepared.case._tendon_report, prepared.mesh,
                             step_names=[st.name or f"step {i + 1}" for i, st in enumerate(prepared.case.steps or [])])
    # what the model weighs and carries, for `vfife summary` (1.1)
    try:
        from .core.model_totals import totals as _totals
        for key, value in _totals(solver, p).items():
            if value is not None:
                import numpy as _np
                writer.f.attrs[f"model_{key}"] = _np.asarray(value, dtype=_np.float64)
    except Exception:
        pass
    every = int(prepared.case.output.checkpoint_every or 0)
    ck_state = {"last": int(getattr(solver, "step", 0) or 0)}

    def checkpoint_due(s):
        # COUNTED IN STEPS: it used to be tested only at an
        # output frame whose step number divided by it, so with sparse frames
        # or a staged relaxation's irregular frame numbers no checkpoint was
        # written before the end. Taken at a frame or a progress check --
        # the points where a stop is taken too, so a checkpoint is a state a
        # restart resumes exactly
        # never in the middle of the static solver's relaxation fallback,
        # whose state is no increment's (1.9)
        if every and not getattr(s, "_in_fallback", False) and int(s.step) - ck_state["last"] >= every:
            writer.write_checkpoint(s.checkpoint(), fingerprint)
            ck_state["last"] = int(s.step)

    def frame(s,step,t,sig):
        fields=s.nodal_fields(sig,recover_stress="stress_nodal" in prepared.case.output.fields)
        extra = {"age": float(s._current_age)} if (getattr(s, "time_dependent", False)
                                                    and getattr(s, "_current_age", None) is not None) else None
        if s.plan is not None:
            # the analysis step this frame belongs to, and its procedure (1.2)
            extra = dict(extra or {}, analysis_step=int(getattr(s, "step_index", 0) or 0),
                         procedure=str(s.plan.procedure[int(getattr(s, "step_index", 0) or 0)]))
        writer.write_frame(step,t,fields,equilibrium=getattr(s,"frame_equilibrium",None),attrs=extra)
        writer.write_rebar_frame(step,t,s.u)
        checkpoint_due(s)
        if on_frame: on_frame(step,t,fields)
    stop = {}

    def progress(*a):
        checkpoint_due(solver)
        if stop.get("signal"):
            return False              # a stop at a step boundary
        if on_progress is not None and on_progress(*a) is False:
            return False
        return None
    if every:
        progress_every = max(1, min(int(progress_every), every))
    started=time.perf_counter(); failure=None; interrupted=False; hard=False
    try:
        with _stop_on_signals(stop, getattr(solver, "log", print)):
            if static_run:
                # the implicit static solver
                from . import implicit
                # A FRAME AT EVERY INCREMENT unless `output.every` is written:
                # the default 200 counts increments here, so
                # a 90-position influence sweep wrote only its last frame,
                # where a staged relaxation writes one per level
                out_every = (prepared.case.output.every
                             if "every" in getattr(prepared.case.output, "model_fields_set", {"every"})
                             else 1)
                implicit.run(solver,on_output=frame,output_every=out_every,
                             on_progress=progress,progress_every=progress_every)
            else:
                solver.run(on_output=frame,output_every=prepared.case.output.every,
                           on_progress=progress,progress_every=progress_every)
    except AnalysisFailure as exc:
        failure=exc
        if solver.diagnostics.get("status") == "RUNNING":
            finish_diagnostics(solver.diagnostics,exc.status,step=solver.step,
                residual=solver.residuals[-1] if solver.residuals else None,
                kinetic=solver.kinetic[-1] if solver.kinetic else None,
                message=str(exc),location=exc.location)
    except (KeyboardInterrupt, StopAnalysis) as exc:
        # Everything below this still runs: monitors, diagnostics, the final
        # checkpoint and writer.close().  Without it a Ctrl-C threw away the
        # whole run -- an hour of solving lost because the file was never
        # closed -- and left a truncated HDF5 behind.
        interrupted=True; failure=exc
        # a KeyboardInterrupt is a second Ctrl-C, or one outside the signal
        # handler: it can land inside a step, and a checkpoint of a half-
        # updated state resumed slightly wrong. The last
        # clean checkpoint is kept instead.
        hard = isinstance(exc, KeyboardInterrupt)
        import signal as _signal
        why = "SIGTERM" if stop.get("signal") == getattr(_signal, "SIGTERM", object()) else "the user's request"
        finish_diagnostics(solver.diagnostics,"INTERRUPTED",step=solver.step,
            residual=solver.residuals[-1] if solver.residuals else None,
            kinetic=solver.kinetic[-1] if solver.kinetic else None,
            message=(f"stopped by {why}; the checkpoint is the last state reached and "
                     "`--restart` will resume from it") if not hard else
                    ("stopped at once (a second interrupt); the step under way was not finished, "
                     "so the checkpoint is the last one written before it, if any"))
    report=serialise(solver.diagnostics)
    series={k:v.values for k,v in solver.monitors.items()}
    meta={k:{"kind":v.kind,"regions":",".join(v.regions),
             "dof":"xyz"[v.dof] if 0<=v.dof<3 else str(v.dof),
             "units":_monitor_unit(prepared.case,k),
             # a fiber monitor's fibre: element, station, coordinates, kind, law
             **(getattr(v,"read",None) or {})}
          for k,v in solver.monitors.items()}
    # A creep analysis' own axis is the concrete age in days, and the step
    # counter is not it: the stations are log-spaced, so `time` runs 1.08,
    # 1.17, ... while the concrete goes 38 days, 51, 69. The solver keeps the
    # age of every frame when the model is time-dependent, and it is written
    # beside the monitors exactly as `time` is -- so `vfife plot --x age`
    # draws a creep curve and a contract can say `at, against: age`.
    ages=getattr(solver,"ages",None)
    if ages and len(ages)==len(solver.times):
        series["age"]=list(ages)
        meta["age"]={"kind":"age","regions":"","dof":"days","units":"d"}
    writer.write_monitors(solver.times,series,
                          residuals=solver.residuals,kinetic=solver.kinetic,meta=meta)
    writer.write_diagnostics(report)
    if not hard:
        final = solver.checkpoint()
        if _finite(final):
            writer.write_checkpoint(final, fingerprint)
        else:
            # NOT OVER THE LAST GOOD ONE: a run that went non-finite replaced
            # its last `checkpoint_every` checkpoint with the NaN state, and a
            # restart resumed the NaN (found in review)
            kept = writer.checkpoint_file
            getattr(solver, "log", print)(
                "  checkpoint    the final state is not finite and was not saved; "
                + (f"{os.path.basename(kept)} keeps the last finite one" if kept else
                   "no earlier checkpoint was written (output.checkpoint_every)"))
    if getattr(writer, "checkpoint_error", None):
        getattr(solver, "log", print)(f"  checkpoint    NOT written ({writer.checkpoint_error})")
    writer.close()
    rs = prepared.case.response_spectrum
    if rs is not None and not interrupted:
        # response spectrum at the state the run ended in;
        # only an equilibrium is a state to linearise about
        say = getattr(solver, "log", print)
        if report.get("converged"):
            from . import spectrum
            base = Path(prepared.case_path).parent if prepared.case_path else Path.cwd()
            spec = rs.model_dump()
            spectrum.write(path, spectrum.analyse(solver, spec, base=base, log=say), spec)
        else:
            say("  spectrum      skipped: the run did not converge to an equilibrium")
    stem=str(Path(path).with_suffix("")); sidecars=[stem+".xdmf",stem+"_last.xdmf"]
    if prepared.element in ("coupled","shell_line"): sidecars.append(stem+"_members.xdmf")
    if solver.rebar: sidecars.append(stem+"_rebar.xdmf")
    if writer.checkpoint_file and os.path.exists(writer.checkpoint_file):
        sidecars.append(writer.checkpoint_file)          # the checkpoint
    if interrupted:
        raise AnalysisInterrupted(str(path), sidecars, report, signal=stop.get("signal"))
    return AnalysisOutcome(report["status"],bool(report["converged"]),str(path),sidecars,
        prepared.preflight,report,{k:list(v.values) for k,v in solver.monitors.items()},
        time.perf_counter()-started,str(failure) if failure else None,
        [float(t) for t in solver.times])


# ------------------------------------------------------------------ study
def study(case: Union[str, Path, CaseDraft], changes, *, output_dir: Union[str, Path],
          combine: str = "product", extract: Optional[Callable[[AnalysisOutcome], Any]] = None,
          on_run: Optional[Callable[[int, Dict[str, Any]], None]] = None,
          quiet: bool = True) -> List[Dict[str, Any]]:
    """Run a case once per set of parameter values.

    `changes` is {dotted key: [values]} -- every combination is run
    (`combine="product"`), or the lists are taken side by side
    (`combine="zip"`, equal lengths) -- or an explicit list of
    {dotted key: value} dicts, one per run. Each run writes
    `output_dir/<name>_<nnn>.h5`; its row records the index, the values,
    the status, whether it converged, the result path, the wall time and,
    with `extract(outcome)`, whatever that returns (a dict is merged into
    the row, anything else goes under "value"). A run that raises is
    recorded with its message and the study goes on; an `extract` that
    raises leaves the run's own columns as they were and puts its message
    under "extract_error". A misspelt key or field name is refused before
    the first run. The rows are returned
    and written to `output_dir/study.csv`; `pandas.DataFrame(rows)` makes
    them a table. `on_run(index, values)` is called before each run; with
    `quiet` (the default) the solver's progress lines are not printed.
    """
    base = case if isinstance(case, CaseDraft) else load_case(case)
    if isinstance(changes, Mapping):
        keys = list(changes)
        lists = [list(changes[k]) for k in keys]
        if combine == "product":
            runs = [dict(zip(keys, vals)) for vals in itertools.product(*lists)]
        elif combine == "zip":
            if len({len(v) for v in lists}) > 1:
                raise ValueError("combine='zip' needs lists of one length: "
                                 + ", ".join(f"{k} {len(v)}" for k, v in zip(keys, lists)))
            runs = [dict(zip(keys, vals)) for vals in zip(*lists)]
        else:
            raise ValueError("combine is 'product' or 'zip'")
    else:
        runs = [dict(r) for r in changes]
    for r in runs:                    # a misspelt key is refused before the first run
        _check_field_names(base.with_changes(r), r)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", base.name)
    rows: List[Dict[str, Any]] = []
    for i, values in enumerate(runs):
        row: Dict[str, Any] = {"index": i, **values}
        path = out_dir / f"{stem}_{i:03d}.h5"
        if on_run:
            on_run(i, values)
        try:
            draft = base.with_changes(values)
            outcome = _run_quietly(draft, str(path)) if quiet else run_case(draft, output=str(path))
            row.update(status=outcome.status, converged=outcome.converged,
                       result=outcome.result_path, wall_time=outcome.wall_time,
                       message=outcome.message)
            if extract is not None:
                # the run succeeded whatever `extract` does: it was recorded
                # as an ERROR row with no result (found in review,
                # v1.0)
                try:
                    got = extract(outcome)
                except Exception as exc:
                    row["extract_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    if isinstance(got, Mapping):
                        row.update(got)
                    else:
                        row["value"] = got
        except AnalysisInterrupted:
            # Ctrl-C stops the STUDY, not only the run it caught: the rows so
            # far are written, then the interrupt goes on up (it used to be
            # recorded as an ERROR row and the next run started -- found in
            # review)
            row.update(status="INTERRUPTED", converged=False, result=str(path), wall_time=None,
                       message="interrupted")
            rows.append(row)
            _write_rows(out_dir, rows)
            raise
        except KeyboardInterrupt:
            rows.append(dict(row, status="INTERRUPTED", converged=False, result=None,
                             wall_time=None, message="interrupted"))
            _write_rows(out_dir, rows)
            raise
        except Exception as exc:                     # the study goes on
            row.update(status="ERROR", converged=False, result=None, wall_time=None,
                       message=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    _write_rows(out_dir, rows)
    return rows


def _check_field_names(draft: CaseDraft, values: Mapping[str, Any]) -> None:
    """A field name no schema reads (`solver.n_stepz`), refused before any
    run: the draft only checks that the path up to the last part exists,
    so a new last part went through, and every run then failed on it
    (found in review). Only the names are checked here -- a
    bad VALUE is the study's business, recorded on its own row."""
    from pydantic import ValidationError
    try:
        schema.from_dict(draft.data, str(draft.base))
    except ValidationError as exc:
        extra = [e for e in exc.errors() if e.get("type") == "extra_forbidden"]
        if extra:
            where = ".".join(str(p) for p in extra[0]["loc"])
            raise UserError("STUDY_UNKNOWN_FIELD",
                            f"{where}: the case has no such field (from the study's change "
                            f"{', '.join(map(repr, values))})",
                            "check the spelling against docs/yaml_reference.md") from None
    except Exception:
        pass                               # a value the run itself will refuse
    # a material's keys are the law's parameters, which the schema leaves
    # open; they are asked of the law itself (materials/base)
    from .materials.base import _refuse_unknown_parameters
    for name, props in (draft.data.get("materials") or {}).items():
        if not isinstance(props, Mapping) or not isinstance(props.get("type"), str):
            continue
        try:
            _refuse_unknown_parameters(props["type"], {k: v for k, v in props.items() if k != "type"})
        except UserError as exc:
            if exc.code not in ("UNKNOWN_MATERIAL_PARAMETER", "MATERIAL_ALIAS_CONFLICT"):
                continue
            raise UserError("STUDY_UNKNOWN_FIELD", f"materials.{name}: {exc.message}",
                            exc.hint) from None
        except Exception:
            continue
        # ...and the time-dependent blocks' own keys (`creep.curvee`), which
        # the law leaves to their models: they passed, and every run was
        # refused on them. Only a NAME is refused here; a
        # value the model rejects is that run's own row.
        from .materials.base import _check_slow
        slow = {k: props[k] for k in ("creep", "shrinkage", "ageing", "relaxation") if k in props}
        if slow:
            try:
                _check_slow(name, slow)
            except ValueError as exc:
                if "does not read" in str(exc):
                    raise UserError("STUDY_UNKNOWN_FIELD", f"materials.{name}: {exc}",
                                    "check the spelling against docs/yaml_reference.md") from None
            except Exception:
                pass
    # `units` is an open mapping to the schema, and a misspelt key is a
    # preflight ERROR since v1.0: every run failed on it
    from .preflight import _UNIT_KEYS
    units = draft.data.get("units")
    if isinstance(units, Mapping):
        bad = [k for k in units if k not in _UNIT_KEYS]
        if bad:
            raise UserError("STUDY_UNKNOWN_FIELD",
                            f"units.{bad[0]}: `units` has no such key (it takes "
                            f"{', '.join(_UNIT_KEYS)})",
                            "check the spelling against docs/yaml_reference.md")


def _write_rows(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(out_dir / "study.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _run_quietly(draft: CaseDraft, output: str) -> AnalysisOutcome:
    """run_case with the solver's progress lines swallowed."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return run_case(draft, output=output)


def read_monitors(path):
    """Every monitor of a result file, {name: Series}; a `Series` has
    `name`, `values` (a float array), `kind`, `regions`, `dof`, `unit` (the
    dataset's `units` attribute; None when the case declared no units) and
    `units` (the same, "" when none). `time`, `residual` and `kinetic_energy` (and `age`)
    are Series too. The result's format is checked first."""
    from .post import read_monitors as _read
    return _read(str(path))


def verify(contract, result) -> dict:
    """A result checked against a verification contract (a YAML file):
    {"status": PASS|FAIL, "checks": [...]} -- see docs/user_manual.md 13.3.
    The contract comes first; the two given the other way round are refused
    as `VERIFY_ARGUMENTS` (it was a UnicodeDecodeError)."""
    import h5py
    from .verification import verify as _verify
    for p in (contract, result):
        if not os.path.exists(str(p)):
            raise UserError("NO_SUCH_FILE", f"no such file: {p}",
                            "verify(contract, result): the contract YAML first, then the result .h5")
    if h5py.is_hdf5(str(contract)) or not h5py.is_hdf5(str(result)):
        raise UserError("VERIFY_ARGUMENTS",
                        f"verify(contract, result) was given contract = {contract!r} and result = "
                        f"{result!r}: the contract is the YAML file and the result the HDF5 file",
                        "call verify(contract_yaml, result_h5)")
    return _verify(contract, result)
