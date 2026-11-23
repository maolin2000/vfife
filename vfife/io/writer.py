"""
Step groups are named by the solver's cumulative iteration count, zero padded.
The width matters: a name only sorts lexicographically the way it sorts
numerically while every name is the same length, and at five digits a run that
passes 99999 iterations starts producing six-character names that sort BEFORE
the five-character ones. case_11 does that -- its frames came back shuffled,
which looked for a while like the internal forces disagreeing with statics.
Eight digits keeps every realistic run at one width; readers that sort should
still use `key=int` (see `io.read_steps`).
HDF5 result file + XDMF sidecar, and checkpoint/restart.

`.mat` does not scale: at 10^6 elements with 30 output frames the raw field
data is ~10 GB, and ParaView cannot read it.  HDF5 stores the mesh once,
appends each frame as a chunked, gzip-compressed dataset, and the XDMF sidecar
makes the whole time series openable directly in ParaView or PyVista.

The full case YAML is embedded in the file, so a result can always be traced
back to the input that produced it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import h5py

from ..elements.library import xdmf_topologies, expand_fields
import numpy as np
from datetime import datetime, timezone
from .. import __version__

#: the result-file layout this version writes, and the newest it reads.
#: Files without the attribute predate it and are read.
#: 3: the checkpoint left the result for its own file
#: (`checkpoint_path`), `equilibrium` is always a string, units and the
#: frame's record index / load factor are attributes. docs/result_format.md
RESULT_FORMAT = 3

#: the checkpoint file's own layout; a reader refuses a newer one
CHECKPOINT_FORMAT = 1


def tendon_rows(report: Dict[str, Any], mesh) -> Dict[str, list]:
    """{tendon: [the element row of each bar, along it]};
    a bar that is not one member of the mesh is refused."""
    out = {}
    for name, t in report.items():
        rows = []
        for region, _p in t.get("segments") or ():
            got = ((mesh.element_regions or {}).get(region) or {}).get("truss2")
            if got is None or len(got) != 1:
                raise ValueError(f"tendon {name!r}: its bar {region!r} is not one member of the mesh")
            rows.append(int(got[0]))
        out[name] = rows
    return out


def checkpoint_path(result_path) -> str:
    """The checkpoint file beside a result: `run.h5` -> `run.ckpt.h5`.

    Its own file, replaced whole (`write_checkpoint_file`), so a run killed
    while writing it leaves the previous checkpoint intact -- inside the
    result it shared the result's fate, and a SIGKILL left neither readable."""
    p = str(result_path)
    if p.endswith(".ckpt.h5"):
        return p
    root, ext = os.path.splitext(p)
    # only `.h5` drops its extension: `run.hdf5` and `run.h5` side by side
    # shared `run.ckpt.h5`, and the second run deleted the first one's
    return (root if ext.lower() == ".h5" else p) + ".ckpt.h5"


def refuse_checkpoint_name(result_path) -> None:
    """A result may not be named like a checkpoint: `-o run.ckpt.h5` was its
    own checkpoint path, so the writer deleted the open result as a stale
    checkpoint and the run's end replaced it with the checkpoint -- every
    frame and monitor lost (found in review)."""
    if str(result_path).endswith(".ckpt.h5"):
        from ..core.diagnostics import UserError
        raise UserError("OUTPUT_IS_CHECKPOINT_NAME",
                        f"output path {result_path} ends in .ckpt.h5, the name of a checkpoint file",
                        "name the result `<name>.h5`; its checkpoint is written beside it as "
                        "`<name>.ckpt.h5`")


def write_checkpoint_file(path, ck: Dict[str, np.ndarray], meta: Optional[dict] = None) -> str:
    """Write a checkpoint to `path` atomically: into `<path>.tmp`, closed,
    then `os.replace`d over the old one. A process killed at any moment
    leaves either the previous checkpoint or this one, never half of one."""
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["checkpoint_format"] = CHECKPOINT_FORMAT
            f.attrs["vfife_version"] = __version__
            f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            for k, v in (meta or {}).items():
                if isinstance(v, np.ndarray) and v.ndim:
                    # an array of the fingerprint (the reference nodes)
                    f.create_dataset(f"fingerprint/{k}", data=np.ascontiguousarray(v))
                elif v is not None:
                    f.attrs[k] = v
            g = f.create_group("checkpoint")
            for k, v in ck.items():
                g.create_dataset(k, data=np.ascontiguousarray(v))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


class Checkpoint:
    """A checkpoint read back: its arrays, the model fingerprint it was
    written with (empty for a legacy one), where it came from, and whether
    it is a LEGACY checkpoint -- a group inside a result of format 2 or
    older, which carries no fingerprint."""

    __slots__ = ("data", "meta", "path", "legacy")

    def __init__(self, data, meta, path, legacy):
        self.data, self.meta, self.path, self.legacy = data, meta, path, legacy


def _read_checkpoint_file(side) -> Checkpoint:
    with h5py.File(side, "r") as f:
        v = f.attrs.get("checkpoint_format")
        if v is not None and int(v) > CHECKPOINT_FORMAT:
            from ..core.diagnostics import UserError
            raise UserError(
                "CHECKPOINT_TOO_NEW",
                f"{side} is checkpoint format {int(v)}; this vfife reads format {CHECKPOINT_FORMAT}",
                f"it was written by vfife {f.attrs.get('vfife_version', '?')}: restart with that version")
        meta = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        if "fingerprint" in f:
            meta.update({k: np.array(v) for k, v in f["fingerprint"].items()})
        data = {k: np.array(v) for k, v in f["checkpoint"].items()} if "checkpoint" in f else {}
    return Checkpoint(data, meta, side, False)


def load_checkpoint(path) -> Optional[Checkpoint]:
    """The checkpoint of a result (or a `.ckpt.h5` named directly): the
    file beside it, or, for a result of format 2 or older, the group inside
    it (`legacy`). None when there is none.

    A result of format 3 says which checkpoint is its own: it names the file
    (`checkpoint_file`) and both carry the run's `run_id`. A checkpoint
    beside it that is not its own -- left by an earlier run of the same name
    when it could not be deleted (Windows keeps a file another program holds
    open), or a file renamed onto another's -- is refused by name rather than
    resumed: it resumed the other run's state (found in review,
    v1.0). A `.ckpt.h5` copied without its result is read as it is."""
    from ..core.diagnostics import UserError
    path = str(path)
    if path.endswith(".ckpt.h5"):
        return _read_checkpoint_file(path) if os.path.exists(path) else None
    side = checkpoint_path(path)
    if not os.path.exists(path):
        return _read_checkpoint_file(side) if os.path.exists(side) else None
    with open_result(path, "r") as f:
        fmt = int(f.attrs.get("result_schema_version", 0) or 0)
        if fmt < 3:
            if "checkpoint" not in f:
                return None
            return Checkpoint({k: np.array(v) for k, v in f["checkpoint"].items()}, {}, path, True)
        named = f.attrs.get("checkpoint_file")
        run_id = f.attrs.get("run_id")
    named = named.decode() if isinstance(named, bytes) else named
    if not named:
        if os.path.exists(side):
            raise UserError("CHECKPOINT_NOT_THIS_RUN",
                            f"the checkpoint beside {path} ({os.path.basename(side)}) belongs to another "
                            "run: this result never wrote one (it was stopped at once, or the old file "
                            "could not be replaced)",
                            "resume from the result that wrote that checkpoint, or run the case again")
        return None
    cand = side if os.path.exists(side) else os.path.join(os.path.dirname(os.path.abspath(path)), named)
    if not os.path.exists(cand):
        return None
    held = _read_checkpoint_file(cand)
    if run_id and held.meta.get("run_id") != run_id:
        raise UserError("CHECKPOINT_NOT_THIS_RUN",
                        f"the checkpoint beside {path} ({os.path.basename(cand)}) belongs to another run "
                        "(its run_id is not the result's)",
                        "copy a result together with its own checkpoint; resume from the result that "
                        "wrote this one, or run the case again")
    return held


def check_result_format(f, path) -> None:
    """Refuse a result written in a newer layout than this vfife knows,
    by name, instead of failing somewhere inside on a missing group."""
    v = f.attrs.get("result_schema_version")
    if v is not None and int(v) > RESULT_FORMAT:
        from ..core.diagnostics import UserError
        raise UserError(
            "RESULT_TOO_NEW",
            f"{path} is result format {int(v)}; this vfife reads format {RESULT_FORMAT} and older",
            f"it was written by vfife {f.attrs.get('vfife_version', '?')}: read it with that "
            "version or a newer one")


def open_result(path, mode: str = "r"):
    """`h5py.File` on a vfife result, its format checked first. Every reader
    opens results through this, so a newer file is refused in one place."""
    f = h5py.File(path, mode)
    try:
        check_result_format(f, path)
    except Exception:
        f.close()
        raise
    return f

_XDMF = """<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
{grids}
    </Grid>
  </Domain>
</Xdmf>
"""

# Older ParaView builds route a temporal collection through vtkXdmfReader and
# may warn that its vtkMultiBlockDataSet does not match an UnstructuredGrid.
# The normal sidecar remains the complete time history; this companion exposes
# its final equilibrium frame as one Uniform grid for that reader.
_XDMF_STATIC = """<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
{grid}  </Domain>
</Xdmf>
"""

# XDMF topology names, and the node count each expects.  This used to be
# hard-coded to Hexahedron, so a tet or wedge run wrote a file whose topology
# did not match its connectivity -- ParaView reads it without complaint and
# draws the wrong cells.
# XDMF spellings come from the element library, which is where the element
# is defined.  This used to be a hand-written copy, and it drifted: wedge15
# was filed as "Wedge", the 6-node topology, so ParaView misread it.
_TOPOLOGY = xdmf_topologies()

#: XDMF cell-type codes, for the flat connectivity a `Mixed` topology needs.
#: Keyed by XDMF topology NAME rather than by element, so it is a fact about
#: the XDMF specification and cannot drift as elements are added -- the library
#: still owns which topology each element is written as.
_MIXED_CODE = {
    "Polyvertex": 1, "Polyline": 2, "Triangle": 4, "Quadrilateral": 5,
    "Tetrahedron": 6, "Pyramid": 7, "Wedge": 8, "Hexahedron": 9,
    "Tetrahedron_10": 38, "Wedge_15": 40, "Wedge_18": 41,
    "Hexahedron_20": 48, "Hexahedron_27": 50,
}


#: nodes per cell of each XDMF topology name
_TOPOLOGY_N = {topo: n for topo, n, _extra in _TOPOLOGY.values()}


def mixed_centroids(nodes, stream) -> np.ndarray:
    """Element centres from a flat XDMF `Mixed` connectivity stream."""
    code_to_n = {code: _TOPOLOGY_N[topo] for topo, code in _MIXED_CODE.items() if topo in _TOPOLOGY_N}
    out, i, stream = [], 0, np.asarray(stream, dtype=np.int64)
    while i < len(stream):
        n = code_to_n[int(stream[i])]
        out.append(nodes[stream[i + 1:i + 1 + n]].mean(axis=0))
        i += 1 + n
    return np.array(out)


def mixed_connectivity(blocks):
    """Flatten several element blocks into one XDMF `Mixed` connectivity.

    XDMF stores a mixed topology as a single stream of
    ``[type code, node ids..., type code, node ids..., ...]``.  The blocks are
    taken in the order given, which is the same block-major order the solver
    assembles and numbers elements in, so cell data written per element lines
    up with the cells without any further bookkeeping.
    """
    parts = []
    for name, conn in blocks.items():
        topo = _TOPOLOGY[name][0]
        code = _MIXED_CODE[topo]
        conn = np.asarray(conn, dtype=np.int64)
        row = np.empty((conn.shape[0], conn.shape[1] + 1), dtype=np.int64)
        row[:, 0] = code
        row[:, 1:] = conn
        parts.append(row.ravel())
    return np.concatenate(parts)

_MIXED_GRID = """      <Grid Name="step{i}" GridType="Uniform">
        <Time Value="{t}"/>
        <Topology TopologyType="Mixed" NumberOfElements="{nel}">
          <DataItem Dimensions="{nconn}" NumberType="Int" Format="HDF">{h5}:/mesh/elements</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{nn} 3" Format="HDF">{h5}:/mesh/nodes</DataItem>
        </Geometry>
{attrs}      </Grid>
"""

_GRID = """      <Grid Name="step{i}" GridType="Uniform">
        <Time Value="{t}"/>
        <Topology TopologyType="{topo}"{topo_extra} NumberOfElements="{nel}">
          <DataItem Dimensions="{nel} {nne}" NumberType="Int" Format="HDF">{h5}:/mesh/elements</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{nn} 3" Format="HDF">{h5}:/mesh/nodes</DataItem>
        </Geometry>
{attrs}      </Grid>
"""

_REBAR_GRID = """      <Grid Name="rebar{i}" GridType="Uniform">
        <Time Value="{t}"/>
        <Topology TopologyType="Polyline" NodesPerElement="2" NumberOfElements="{nel}">
          <DataItem Dimensions="{nel} 2" NumberType="Int" Format="HDF">{h5}:/rebar/elements</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{nn} 3" Format="HDF">{h5}:/rebar/nodes</DataItem>
        </Geometry>
{attrs}      </Grid>
"""

_ATTR = """        <Attribute Name="{name}" AttributeType="{atype}" Center="{center}">
          <DataItem Dimensions="{dims}" Format="HDF">{h5}:/steps/{i:08d}/{name}</DataItem>
        </Attribute>
"""


_COMPONENT_ATTR = """        <Attribute Name="{name}_{comp}" AttributeType="Scalar" Center="{center}">
          <DataItem ItemType="HyperSlab" Dimensions="{n} 1" Type="HyperSlab">
            <DataItem Dimensions="3 2" Format="XML">0 {col} 1 1 {n} 1</DataItem>
            <DataItem Dimensions="{n} 6" Format="HDF">{h5}:/steps/{i:08d}/{name}</DataItem>
          </DataItem>
        </Attribute>
"""

#: the fields that belong to the nodes, whatever their length; everything
#: else a frame holds is per element. Decided by the field, not by its length:
#: a 3-bar triangle truss has as many elements as nodes, and its axial forces
#: were drawn on the nodes.
NODE_FIELDS = {"displacement", "rotation", "velocity", "acceleration", "reaction_force",
               "reaction_moment", "stress_nodal", "temperature"}


class ResultWriter:
    def __init__(self, path: str, mesh, case_yaml: str = "", fields=None,
                 compression: str | None = "gzip", element: str = "hexa8",
                 rebar=None, time_axis: str = "time", units=None,
                 restarted_from: Optional[str] = None, restarted_at_step: Optional[int] = None,
                 restarted_at_time: Optional[float] = None):
        self.path = path
        #: where a continuation resumed, in the time of its event tables: the
        #: rows at or before it are the earlier file's
        self.restarted_at_time = None if restarted_at_time is None else float(restarted_at_time)
        #: what a frame's `time` is: "time" (seconds, dynamic) or
        #: "load_factor" (completed steps + lambda, relaxation and static).
        #: The XDMF clock of a load-factor result is the record index: lambda
        #: goes back on an arc-length path, and ParaView orders by its clock:
        self.time_axis = "load_factor" if time_axis == "load_factor" else "time"
        self.checkpoint_file = None
        self.coupled = element in {"coupled", "shell_line"}
        self.member_conn = None
        if self.coupled:
            solid = [k for k in mesh.cell_blocks if k != "truss2"]
            if len(solid) != 1: raise ValueError("coupled output requires one solid block")
            self.element = solid[0]; self.member_conn = mesh.cell_blocks["truss2"]
        else:
            self.element = element
        # A mixed mesh is written as one XDMF `Mixed` grid: ParaView shows the
        # hexahedra and the tetrahedra as one body, with one set of fields,
        # rather than two grids the user has to overlay.
        cb = getattr(mesh, "cell_blocks", None) or {}
        solid_blocks = {k: v for k, v in cb.items()
                        if k in _TOPOLOGY and k != "truss2"}
        self.mixed = None if self.coupled or len(solid_blocks) < 2 else solid_blocks
        refuse_checkpoint_name(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.f = h5py.File(path, "w")
        # the result is new: a checkpoint left beside it by an earlier run of
        # the same name is not this run's, and a restart must not find it.
        # Where it cannot be deleted (Windows: another program holds it
        # open), the run id below tells it apart
        try:
            os.remove(checkpoint_path(path))
        except OSError:
            pass
        import uuid
        #: this run, in the result and in every checkpoint it writes
        self.run_id = uuid.uuid4().hex
        self.f.attrs["run_id"] = self.run_id
        self.f.attrs["result_schema_version"] = RESULT_FORMAT
        self.f.attrs["vfife_version"] = __version__
        self.f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        self.f.attrs["element_family"] = element
        self.f.attrs["time_axis"] = self.time_axis
        # the case's declared units, one attribute each (`unit_length`, ...):
        # the numbers are in them, and a reader should not have to parse the
        # embedded case to label an axis
        for key, value in sorted((units or {}).items()):
            self.f.attrs[f"unit_{key}"] = str(value)
        if restarted_from:
            # a continuation holds only the tail of the run: say so, and of what
            self.f.attrs["restarted_from"] = str(restarted_from)
            if restarted_at_step is not None:
                self.f.attrs["restarted_at_step"] = int(restarted_at_step)
            if restarted_at_time is not None:
                self.f.attrs["restarted_at_time"] = float(restarted_at_time)
        self.comp = compression
        self.fields = None if fields is None else expand_fields(fields)
        self.nn = mesh.n_nodes
        g = self.f.create_group("mesh")
        g.create_dataset("nodes", data=mesh.nodes.astype(np.float64))
        if self.mixed:
            flat = mixed_connectivity(self.mixed)
            self.nel = int(sum(c.shape[0] for c in self.mixed.values()))
            self.nconn = int(flat.size)
            g.create_dataset("elements", data=flat.astype(np.int32))
            g.attrs["blocks"] = ",".join(self.mixed)
        else:
            self.nel = mesh.n_elem
            g.create_dataset("elements", data=mesh.elements.astype(np.int32))
        if self.member_conn is not None:
            mg = self.f.create_group("members")
            mg.create_dataset("elements", data=self.member_conn.astype(np.int32))
        self.f.attrs["case_yaml"] = case_yaml
        self.f.create_group("steps")
        self._frames = []
        self._member_frames = []
        self._rebar_frames = []
        self._setup_rebar(rebar or [])

    # ------------------------------------------------------------------ #
    #  reinforcement
    # ------------------------------------------------------------------ #

    def _setup_rebar(self, blocks):
        """Write the bar geometry once, as ONE polyline grid over all blocks.

        The bars go to their own file rather than into a spatial collection
        beside the concrete.  ParaView's handling of a temporal collection with
        spatial collections nested inside it is unreliable -- some builds show
        only the first block, others lose the time steps -- and two files that
        each open cleanly are worth more than one that sometimes does.  Opening
        both and leaving the concrete translucent is the normal way to look at
        this anyway.

        The blocks are concatenated with an offset because a ParaView user wants
        one "rebar" object to colour and tube, not five.  Which block a segment
        came from stays available as the `block` cell field.
        """
        self.rebar = list(blocks)
        self.n_bar = 0
        if not self.rebar:
            return
        pts, conn, block, base = [], [], [], 0
        for b, emb in enumerate(self.rebar):
            # the bar's own reference geometry is the HOST-INTERPOLATED position
            # (elements/truss.py), which is what the solver actually integrates
            pts.append(emb.x0_e.reshape(-1, 3))
            n = emb.n_elem
            conn.append(base + np.arange(2 * n, dtype=np.int32).reshape(n, 2))
            block.append(np.full(n, b, dtype=np.float32))
            base += 2 * n
        self.bar_nodes = np.vstack(pts)
        self.bar_conn = np.vstack(conn)
        self.bar_block = np.concatenate(block)
        self.n_bar = self.bar_conn.shape[0]
        g = self.f.create_group("rebar")
        g.create_dataset("nodes", data=self.bar_nodes.astype(np.float64))
        g.create_dataset("elements", data=self.bar_conn)
        g.create_dataset("block", data=self.bar_block)
        for b, emb in enumerate(self.rebar):
            g.attrs[f"block{b}"] = getattr(emb, "label", f"rebar{b}")

    def write_rebar_frame(self, index: int, t: float, u: np.ndarray):
        """Bar displacement, axial stress and plastic strain for one frame.

        Displacement is the HOST INTERPOLATION at the bar ends, so Warp By
        Vector in ParaView moves the bars with the concrete.  Without it the
        reinforcement sits at its undeformed position while the beam around it
        bends, which looks like a bug and is merely a missing field.
        """
        if not self.n_bar:
            return
        disp, sig, epsp = [], [], []
        for emb in self.rebar:
            disp.append(emb.gather(u).reshape(-1, 3))
            sig.append(emb.stress(u))          # read-only: see EmbeddedTruss.stress
            epsp.append(emb.state[:, 1])
        g = self.f.require_group("rebar_steps").create_group(f"{index:08d}")
        g.attrs["time"] = float(t)
        data = {
            "displacement": np.vstack(disp).astype(np.float32),
            "axial_stress": np.concatenate(sig).astype(np.float32),
            "plastic_strain": np.concatenate(epsp).astype(np.float32),
            "block": self.bar_block,
        }
        kept = {}
        for name, arr in data.items():
            g.create_dataset(name, data=arr, compression=self.comp)
            kept[name] = arr.shape
        self._rebar_frames.append((index, float(t), kept))

    def write_frame(self, index: int, t: float, fields: Dict[str, np.ndarray], equilibrium=None,
                    attrs: Optional[dict] = None):
        g = self.f["steps"].create_group(f"{index:08d}")
        g.attrs["time"] = float(t)
        # ALWAYS A STRING: "true", "false" or "not_applicable" (a dynamic
        # frame). It was a bool or a string by mode, and a reader had to
        # handle both
        g.attrs["equilibrium"] = ("not_applicable" if equilibrium is None
                                  else "true" if bool(equilibrium) else "false")
        g.attrs["record"] = len(self._frames)
        if self.time_axis == "load_factor":
            g.attrs["load_factor"] = float(t)
        for key, value in (attrs or {}).items():
            if value is not None:
                g.attrs[key] = value
        kept = {}
        line_names = {"axial_force", "shear_y", "shear_z", "torsion", "bending_y", "bending_z",
                      "member_type", "beam_formulation", "connector_flag"}
        line_names |= {f"{n}_{e}" for n in tuple(line_names) for e in ("i", "j", "max_abs")}
        member_data = {}
        for name, arr in fields.items():
            # the members' connectivity is not a field the analyst picks:
            # once a member has split it is the only record of which
            # particles carry it (fiber_beam_solver.nodal_fields)
            if self.fields is not None and name not in self.fields and name != "connectivity":
                continue
            # the connectivity is integers: particle numbers in float32 are
            # exact only below 2^24, and the XDMF reads it as a topology
            a = np.ascontiguousarray(arr, dtype=np.int32 if name == "connectivity" else np.float32)
            if self.coupled and (name in line_names or name.startswith("connector_")):
                member_data[name] = a; continue
            g.create_dataset(name, data=a, compression=self.comp)
            kept[name] = a.shape
        self._frames.append((index, float(t), kept, dict(g.attrs)))
        if self.coupled:
            mg = self.f.require_group("member_steps").create_group(f"{index:08d}")
            mg.attrs["time"] = float(t); mkept = {}
            for name, a in member_data.items():
                mg.create_dataset(name, data=a, compression=self.comp); mkept[name] = a.shape
            for name in ("displacement", "rotation"):
                if name in fields:
                    a = np.ascontiguousarray(fields[name], dtype=np.float32)
                    mg.create_dataset(name, data=a, compression=self.comp); mkept[name] = a.shape
            self._member_frames.append((index, float(t), mkept))
        # on disk at every frame: a run killed later leaves a readable file
        # with every frame up to here
        self.f.flush()

    def write_monitors(self, times, monitors: Dict[str, list],
                       residuals=None, kinetic=None, meta=None):
        g = self.f.require_group("monitors")
        g.create_dataset("time", data=np.asarray(times, dtype=np.float64))
        for name, vals in monitors.items():
            d = g.create_dataset(name, data=np.asarray(vals, dtype=np.float64))
            # What the series IS, stored beside it.  A bare column of numbers
            # cannot label its own axis, so anything reading the file back --
            # `vfife plot`, `vfife export`, a GUI -- had to be told separately
            # what it was looking at, or guess from the name the user chose.
            for key, value in (meta or {}).get(name, {}).items():
                if value is not None:
                    d.attrs[key] = value
        g["time"].attrs["kind"] = self.time_axis
        if residuals is not None:
            d = g.create_dataset("residual", data=np.asarray(residuals, dtype=np.float64))
            d.attrs["sampling"] = "progress_check"
        if kinetic is not None:
            d = g.create_dataset("kinetic_energy", data=np.asarray(kinetic, dtype=np.float64))
            d.attrs["sampling"] = "progress_check"

    def write_diagnostics(self, report):
        """Persist termination status and the CPU energy audit."""
        import json
        g = self.f.require_group("diagnostics")
        for key in ("status", "converged", "mode", "backend", "step", "load_step",
                    "residual", "kinetic_energy", "message", "energy_available",
                    "energy_unavailable_reason"):
            value = report.get(key)
            if value is not None: g.attrs[key] = value
        g.attrs["location_json"] = json.dumps(report.get("location") or {})
        history = report.get("energy_history") or []
        length, force = self.f.attrs.get("unit_length"), self.f.attrs.get("unit_force")
        energy_unit = f"{force}*{length}" if length and force else ""
        for name in ("step", "external_work", "internal_work", "kinetic_energy",
                     "dissipated_energy", "energy_imbalance", "relative_energy_error"):
            d = g.create_dataset(name, data=np.asarray([row[name] for row in history], dtype=float))
            # as on the monitors: absent when the case declared no units
            # (it was "" then)
            if any(k.startswith("unit_") for k in self.f.attrs):
                d.attrs["units"] = ("" if name in ("step", "relative_energy_error") else energy_unit)
        # ends moved onto particles of their own: one row each,
        # [t, element, the particle it was on, the particle it moved to]
        splits = report.get("member_splits")
        if splits:
            d = g.create_dataset("member_splits", data=np.asarray(splits, dtype=float).reshape(-1, 4))
            d.attrs["columns"] = "time,element,parent_particle,new_particle"
            self._mark_restored_rows(d)
        # members that broke or were removed: one row each
        failures = report.get("member_failures")
        if failures:
            d = g.create_dataset("member_failures", data=np.asarray(failures, dtype=float).reshape(-1, 4))
            d.attrs["columns"] = "time,element,capacity_ratio,cause"
            d.attrs["cause_codes"] = "0=capacity,1=timed removal,2=what a massless member connected is gone,3=twisted a quarter turn"
            self._mark_restored_rows(d)
        # tyres that left the road and rigid axles that pulled, per vehicle:
        # counts and the first time
        contact = report.get("vehicle_contact")
        if contact:
            g.attrs["vehicle_contact_json"] = json.dumps(contact)
        self.f.attrs["analysis_status"] = report.get("status", "UNKNOWN")
        self.f.attrs["analysis_converged"] = bool(report.get("converged", False))

    def _mark_restored_rows(self, d):
        """A continuation's event table holds the WHOLE history (the
        checkpoint carries it); `rows_before_restart` counts the rows that
        are already in the file it resumed, so a reader joining the two
        files takes the earlier file's table and this one's rows from there
        on. 0 in a run that is not a continuation."""
        rows = np.asarray(d[...], float).reshape(-1, d.shape[-1] if d.ndim > 1 else 1)
        n = 0
        if self.restarted_at_time is not None and rows.size:
            n = int(np.count_nonzero(rows[:, 0] <= self.restarted_at_time + 1e-12 * max(1.0, abs(self.restarted_at_time))))
        d.attrs["rows_before_restart"] = n

    def write_checkpoint(self, ck: Dict[str, np.ndarray], meta: Optional[dict] = None):
        """The checkpoint goes to its own file (`checkpoint_path`), replaced
        whole; the result records its name and is flushed with it."""
        target = checkpoint_path(self.path)
        try:
            path = write_checkpoint_file(target, ck, dict(meta or {}, result=os.path.basename(self.path),
                                                          run_id=self.run_id))
        except OSError as exc:
            # another program holds the old checkpoint open (Windows locks
            # it): the run and its result go on, the previous checkpoint
            # stays, and the caller is told (`checkpoint_error`)
            self.checkpoint_error = f"{target}: {exc}"
            try:
                os.remove(target + ".tmp")
            except OSError:
                pass
            return None
        self.checkpoint_error = None
        self.checkpoint_file = path
        self.f.attrs["checkpoint_file"] = os.path.basename(path)
        self.f.flush()
        return path

    def write_tendons(self, report: Dict[str, Any], mesh, step_names=()) -> None:
        """The `tendons:` of the case: one group each under
        `tendons`, in the order of the case -- its name, strands, area, the
        jacking force at each end and the step that stresses it; and its
        bars in order along it, as the element rows of the result
        (`elements`, the rows of `axial_force`), their regions and the force
        each was jacked to (`jack`). `vfife tendons` reads it."""
        if not report:
            return
        # (a frame model only -- `tendons` expand onto a mesh of Line2 members
        # alone -- so a row of the mesh is a row of `steps/*/axial_force`)
        assert not self.mixed and not self.coupled, "tendons are for a frame model"
        rows_of = tendon_rows(report, mesh)
        g = self.f.create_group("tendons")
        g.attrs["step_names_json"] = json.dumps([str(n) for n in step_names], ensure_ascii=False)
        for k, (name, t) in enumerate(report.items()):
            rows = rows_of[name]
            jack = [float(p) for _r, p in t.get("segments") or ()]
            regions = [str(r) for r, _p in t.get("segments") or ()]
            tg = g.create_group(f"t{k + 1:05d}")
            tg.attrs["name"] = str(name)
            tg.attrs["count"] = int(t.get("count", 1))
            tg.attrs["area"] = float(t.get("area", 0.0))
            tg.attrs["jacking"] = np.asarray(t.get("jacking") or [0.0, 0.0], dtype=np.float64)
            tg.attrs["stress_step"] = int(t.get("step", 0))
            tg.attrs["regions_json"] = json.dumps(regions, ensure_ascii=False)
            tg.create_dataset("elements", data=np.asarray(rows, dtype=np.int64))
            tg.create_dataset("jack", data=np.asarray(jack, dtype=np.float64))
        self.f.flush()

    def close(self):
        self._write_xdmf()
        self._write_member_xdmf()
        self._write_rebar_xdmf()
        self.f.close()

    def _write_member_xdmf(self):
        if not self._member_frames: return
        h5=os.path.basename(self.path); grids=[]; nel=len(self.member_conn)
        # the frame's own scalars (load factor, age) as on the main grid
        fattrs_of = {fr[0]: (fr[3] if len(fr) > 3 else {}) for fr in self._frames}
        for i,t,kept in self._member_frames:
            attrs=self._grid_values(fattrs_of.get(i) or {})
            for name,shape in kept.items():
                if len(shape) > 2 or name == "connectivity": continue
                atype="Vector" if len(shape)==2 and shape[1]==3 else "Scalar"
                attrs += _ATTR.format(name=name, atype=atype,
                    center="Node" if (name in NODE_FIELDS or (shape[0]==self.nn and self.nn!=nel)) else "Cell",
                    dims=" ".join(map(str,shape)),h5=h5,i=i).replace("/steps/","/member_steps/")
            grid=_GRID.format(i=i,t=self._clock(i,t),nel=nel,nn=self.nn,h5=h5,attrs=attrs,
                topo="Polyline",topo_extra=' NodesPerElement="2"',nne=2)
            grid=grid.replace("/mesh/elements","/members/elements")
            grids.append(grid)
        with open(os.path.splitext(self.path)[0]+"_members.xdmf","w",encoding="utf-8") as fh:
            fh.write(_XDMF.format(grids="".join(grids)))

    def _node_centred(self, name, shape):
        if name in NODE_FIELDS:
            return True
        if shape[0] != self.nn:
            return False
        # a field of node length the table does not know: by length, unless
        # the element count is the same number, when it is an element field
        return self.nn != self.nel

    def _clock(self, index, t) -> float:
        """The XDMF `Time` of a frame: seconds for a dynamic result, the
        record index (0, 1, 2, ...) for a load-factor one -- lambda runs
        backwards on an arc-length path and ParaView orders frames by this
        value; lambda itself goes beside it (`_grid_values`)."""
        if self.time_axis == "time":
            return t
        records = getattr(self, "_record_of", None)
        if records is None or len(records) != len(self._frames):
            records = self._record_of = {fr[0]: k for k, fr in enumerate(self._frames)}
        return records.get(index, 0)

    @staticmethod
    def _grid_values(fattrs) -> str:
        """Per-frame scalars as Grid-centred attributes (field data in
        ParaView): the load factor and the concrete age, where they apply."""
        out = ""
        for key in ("load_factor", "age"):
            if key in fattrs:
                out += (f'        <Attribute Name="{key}" AttributeType="Scalar" Center="Grid">\n'
                        f'          <DataItem Dimensions="1" Format="XML">{float(fattrs[key])!r}</DataItem>\n'
                        f'        </Attribute>\n')
        return out

    def _write_xdmf(self):
        grids = [self._frame_grid(*fr) for fr in self._frames]
        stem = os.path.splitext(self.path)[0]
        with open(stem + ".xdmf", "w", encoding="utf-8") as fh:
            fh.write(_XDMF.format(grids="".join(grids)))
        if grids:
            # Keep this tiny static sidecar beside the full temporal file.
            # It references the same HDF5 datasets, so it adds no field-data
            # duplication and is a reliable ParaView fallback. Without the
            # six-column arrays: the older reader it is for (vtkXdmfReader)
            # reads a `Matrix` attribute as ONE tuple of 6n components; the
            # six components by name are there
            with open(stem + "_last.xdmf", "w", encoding="utf-8") as fh:
                fh.write(_XDMF_STATIC.format(grid=self._frame_grid(*self._frames[-1], matrix=False)))

    def _frame_grid(self, i, t, kept, fattrs=None, matrix=True) -> str:
        """One frame's XDMF grid (`matrix=False`: without the six-column arrays)."""
        h5 = os.path.basename(self.path)
        attrs = ""
        for name, shape in kept.items():
            if len(shape) > 2 or name == "connectivity":
                # per-fibre arrays and the members' connectivity live in
                # the HDF5 for `vfife sweep`; XDMF has no attribute for them
                continue
            node_centered = self._node_centred(name, shape)
            center = "Node" if node_centered else "Cell"
            if len(shape) == 2 and shape[1] == 3:
                atype = "Vector"
            elif len(shape) == 2 and shape[1] == 6:
                # NOT Tensor6. The solver's order is [xx, yy, zz, xy, yz,
                # xz]; VTK reads a Tensor6 as [xx, xy, xz, yy, yz, zz], so
                # ParaView showed sigma_xy under sigma_yy and sigma_xz
                # under sigma_zz. The array goes as six
                # plain components in the stored order, and each
                # component by name, read in place through a hyperslab.
                atype = "Matrix"
                attrs += "".join(_COMPONENT_ATTR.format(
                    name=name, comp=c, col=k, n=shape[0], center=center, h5=h5, i=i)
                    # `zx`, the name the stored order, the monitors and the
                    # contracts use (it was `stress_xz` here)
                    for k, c in enumerate(("xx", "yy", "zz", "xy", "yz", "zx")))
                if not matrix:
                    continue
            else:
                atype = "Scalar"
            dims = " ".join(str(s) for s in shape)
            attrs += _ATTR.format(
                name=name, atype=atype, center=center,
                dims=dims, h5=h5, i=i,
            )
        attrs += self._grid_values(fattrs or {})
        clock = self._clock(i, t)
        if self.mixed:
            return _MIXED_GRID.format(
                i=i, t=clock, nel=self.nel, nconn=self.nconn, nn=self.nn,
                h5=h5, attrs=attrs)
        topo, nne, topo_extra = _TOPOLOGY.get(self.element,
                                              ("Hexahedron", 8, ""))
        grid = _GRID.format(
            i=i, t=clock, nel=self.nel, nn=self.nn, h5=h5, attrs=attrs,
            topo=topo, topo_extra=topo_extra, nne=nne,
        )
        if "connectivity" in kept:
            # members have moved onto other particles: this frame's
            # own connectivity, or ParaView draws a fragment still
            # pinned to the particle it broke away from
            grid = grid.replace(f"{h5}:/mesh/elements", f"{h5}:/steps/{i:08d}/connectivity")
        return grid

    def _write_rebar_xdmf(self):
        if not self._rebar_frames:
            return
        h5 = os.path.basename(self.path)
        nbn = self.bar_nodes.shape[0]
        grids = []
        for i, t, kept in self._rebar_frames:
            attrs = ""
            for name, shape in kept.items():
                atype = "Vector" if len(shape) == 2 and shape[1] == 3 else "Scalar"
                attrs += _ATTR.format(
                    name=name, atype=atype,
                    center="Node" if shape[0] == nbn else "Cell",
                    dims=" ".join(str(s) for s in shape), h5=h5, i=i,
                ).replace("/steps/", "/rebar_steps/")
            grids.append(_REBAR_GRID.format(
                i=i, t=self._clock(i, t), nel=self.n_bar, nn=nbn, h5=h5, attrs=attrs))
        out = os.path.splitext(self.path)[0] + "_rebar.xdmf"
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(_XDMF.format(grids="".join(grids)))


def read_checkpoint(path: str) -> Optional[Dict[str, np.ndarray]]:
    """The arrays of a result's checkpoint (`load_checkpoint`), or None."""
    if os.path.exists(str(path)) and not str(path).endswith(".ckpt.h5"):
        with open_result(path, "r"):          # a newer result is refused by name
            pass
    ck = load_checkpoint(path)
    return None if ck is None else ck.data
