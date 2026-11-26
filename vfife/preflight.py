"""Solver-independent engineering model checks.

The checker runs after mesh import and region resolution, but before material
kernels or CUDA are initialized. Deterministic defects are errors; mesh-quality
and large-model rank limitations are explicit warnings.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from types import SimpleNamespace

from .core import selectors
from .catalog import compatibility
from .base.dof import repeated_rotation


_LINE_TYPES = {"truss2", "beam2", "beam2_timoshenko", "beam2_fiber", "connector2", "mixed_line"}
_BEAM_TYPES = {"beam2", "beam2_timoshenko", "beam2_fiber"}
_SIX_DOF_TYPES = _BEAM_TYPES | {"connector2", "mixed_line", "shell4", "shell3"}


class PreflightError(RuntimeError):
    pass


def _issue(items, severity, code, message, **details):
    items.append({"severity": severity, "code": code, "message": message,
                  "details": details or None})


def _regions(case, mesh, element, items):
    predefined = dict(mesh.regions)
    predefined.setdefault("all", np.arange(mesh.n_nodes, dtype=np.int64))
    lazy = {}
    if element in _LINE_TYPES or element in {"shell4","shell3"}:
        # a line or surface mesh is all surface, and that is cheap to say
        predefined.setdefault("surface", np.arange(mesh.n_nodes, dtype=np.int64))
    else:
        # offered, not computed -- see selectors.resolve_all
        lazy["surface"] = selectors.surface_provider(mesh, element)
    try:
        specs = {k: v.model_dump(mode="python", exclude_none=True) for k, v in case.regions.items()}
        notes = []
        mesh.regions = selectors.resolve_all(mesh.nodes, specs, predefined, lazy=lazy, notes=notes)
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        _issue(items, "ERROR", "REGION_RESOLUTION", str(exc))
        return
    for note in notes:
        # a snap is the ordinary case for a support that does not sit on a
        # node line; it is INFO so the coordinate used is on the record
        _issue(items, "INFO", "REGION_SNAPPED", note)
    references = []
    for obj in list(case.constraints) + list(case.loads) + list(case.nodal_masses) \
            + list(getattr(case, "hydrodynamic_mass", [])):
        references.append(obj.region)
    for step in case.steps:
        references += [x.region for x in step.constraints] + [x.region for x in step.loads]
    references += [m.region for m in case.members]
    # the fibre model's own regions (found in review)
    el = case.element
    references += [fs.get("region") for fs in (getattr(el, "fiber_sections", None) or []) if fs.get("region")]
    references += [fc.get("region") for fc in (getattr(el, "fiber_connectors", None) or []) if fc.get("region")]
    if getattr(el, "fiber_region", None):
        references.append(el.fiber_region)
    references += [d.region for d in getattr(case, "diaphragms", [])]
    for m in (case.output.monitors or {}).values():
        references += ([m.region] if m.region else []) + list(m.regions or []) + ([m.relative_to] if m.relative_to else [])
    for name in sorted(set(references)):
        if name not in mesh.regions:
            _issue(items, "ERROR", "UNKNOWN_REGION", f"region {name!r} is referenced but unavailable")
        elif len(mesh.regions[name]) == 0:
            _issue(items, "ERROR", "EMPTY_REGION", f"region {name!r} contains no nodes")


def _diaphragms(case, mesh, items):
    """A floor cannot be both held by a support and moved by a diaphragm, and
    a diaphragm of one node is not a floor. The solver refuses
    both; preflight says so first, before anything is assembled."""
    ndof = 6
    fixed = _constraint_mask(case, mesh, ndof)
    for d in getattr(case, "diaphragms", []):
        idx = mesh.regions.get(d.region)
        if idx is None or len(idx) == 0:
            continue                       # _regions has already said so
        if len(idx) < 2:
            _issue(items, "ERROR", "DIAPHRAGM_TOO_SMALL",
                   f"diaphragm region {d.region!r} has one node; a rigid floor needs at least two",
                   region=d.region)
        n = "xyz".index(d.normal); a, b = [k for k in range(3) if k != n]
        held = fixed[idx][:, (a, b, 3 + n)]
        if held.any():
            _issue(items, "ERROR", "DIAPHRAGM_OVER_CONSTRAINED",
                   f"diaphragm region {d.region!r} contains nodes whose in-plane degrees of "
                   "freedom are also constrained; a floor is either held by a support or moved "
                   "by a diaphragm, not both",
                   region=d.region, nodes=int(held.any(axis=1).sum()))


def _linked_pairs(case, mesh):
    """The (master, slave) node pairs of the case's rigid links (1.21): a
    slave on its master's position is an equalDOF joint, not two parts
    left unjoined."""
    out = set()
    links = list(getattr(case, "rigid_links", None) or []) + [
        lk for st in (getattr(case, "steps", None) or []) for lk in getattr(st, "rigid_links", None) or []]
    for lk in links:
        if lk.master in mesh.regions and lk.slaves in mesh.regions:
            for m in np.asarray(mesh.regions[lk.master]).ravel():
                for n in np.asarray(mesh.regions[lk.slaves]).ravel():
                    out.add(frozenset((int(m), int(n))))
    return out


def _mesh_checks(mesh, element, items, joined=frozenset()):
    x, conn = np.asarray(mesh.nodes), np.asarray(mesh.elements)
    if not np.isfinite(x).all():
        _issue(items, "ERROR", "NONFINITE_COORDINATE", "mesh contains non-finite coordinates")
    if conn.ndim != 2 or conn.size == 0:
        _issue(items, "ERROR", "EMPTY_MESH", "mesh contains no elements"); return
    if conn.min() < 0 or conn.max() >= len(x):
        _issue(items, "ERROR", "CONNECTIVITY_RANGE", "element connectivity references an invalid node")
        return
    repeated_local = np.array([len(np.unique(row)) != len(row) for row in conn])
    if repeated_local.any():
        _issue(items, "ERROR", "REPEATED_ELEMENT_NODE", "elements repeat a node locally",
               count=int(repeated_local.sum()), first=np.flatnonzero(repeated_local)[:8].tolist())
    canonical = np.sort(conn, axis=1)
    _, counts = np.unique(canonical, axis=0, return_counts=True)
    duplicate = int(np.sum(counts - 1))
    if duplicate:
        _issue(items, "ERROR", "DUPLICATE_ELEMENT", "duplicate elements have identical node sets", count=duplicate)
    used = np.zeros(len(x), dtype=bool)
    for block in _solid_blocks(mesh):      # a node the hexahedra use is not an
        used[np.unique(block)] = True      # orphan just because no tet uses it
    if (~used).any():
        _issue(items, "ERROR", "ORPHAN_NODE", "mesh contains nodes unused by every element",
               count=int((~used).sum()), first=np.flatnonzero(~used)[:8].tolist())
    diag = max(float(np.linalg.norm(np.ptp(x, axis=0))), 1.0)
    # WITHIN A TOLERANCE, NOT BY ROUNDING: two volumes meshed without being
    # fragmented have interface nodes 1e-9 apart, and rounding to 1e-12 of
    # the model let them through as a silently disconnected interface while
    # an exact duplicate was refused
    from scipy.spatial import cKDTree
    pairs = cKDTree(x).query_pairs(r=1e-8 * diag, output_type="ndarray")
    if len(pairs) and joined:
        pairs = np.array([p for p in pairs if frozenset((int(p[0]), int(p[1]))) not in joined]).reshape(-1, 2)
    if len(pairs):
        _issue(items, "ERROR", "DUPLICATE_NODE",
               f"distinct nodes share the same position (within {1e-8 * diag:.3g}): "
               f"{len(pairs)} pair(s), e.g. nodes {int(pairs[0][0])} and {int(pairs[0][1])}. "
               "Two parts meshed separately and not joined leave such pairs on their interface",
               extra=int(len(pairs)))
    if element in _LINE_TYPES:
        lengths = np.linalg.norm(x[conn[:, 1]] - x[conn[:, 0]], axis=1)
        bad = lengths <= diag * 1e-12
        if bad.any():
            _issue(items, "ERROR", "ZERO_LENGTH", "line elements have zero length",
                   count=int(bad.sum()), first=np.flatnonzero(bad)[:8].tolist())
        if lengths.size and lengths.min() > 0 and lengths.max()/lengths.min() > 100:
            _issue(items, "WARNING", "LINE_LENGTH_RANGE", "line length ratio exceeds 100",
                   ratio=float(lengths.max()/lengths.min()))
        return
    if element in ("shell4", "shell3", "shell_line"):
        # SHELLS ARE NOT COVERED BY 1.0 (the user's decision):
        # a third review found the triangle locking in shear and the quad's
        # stabilisation stiffening bending by an amount refinement does not
        # remove. Said on every shell run, not only in the manual.
        _issue(items, "WARNING", "SHELL_EXPERIMENTAL",
               ("shell3 (triangles) locks in shear: a thin plate deflects 0.2-20% of the true value; "
                "mesh with quads" if element == "shell3" else
                "shell4 is experimental: its hourglass stabilisation stiffens bending (2-30%, more "
                "on elongated elements, not removed by refinement) and distorted quads lock; "
                "compare against a reference before relying on it"))
    if element == "shell4":
        from .elements import shell
        bad=[]; quality=[]
        for e,row in enumerate(conn):
            try:
                _R,_c,local=shell.frame(x[row])
                det=[]
                for xi in (-1/np.sqrt(3),1/np.sqrt(3)):
                    for eta in (-1/np.sqrt(3),1/np.sqrt(3)):
                        _N,d=shell.shape(xi,eta); det.append(np.linalg.det(d@local[:,:2]))
                if min(det)<=0: bad.append(e)
                else: quality.append(min(det)/max(det))
            except ValueError: bad.append(e)
        if bad: _issue(items,"ERROR","INVALID_SHELL_FACE","shell4 faces are degenerate, inverted or excessively warped",count=len(bad),first=bad[:8])
        if quality and min(quality)<.05: _issue(items,"WARNING","DISTORTED_ELEMENT","shell4 Jacobian ratio is below 0.05",minimum=float(min(quality)))
        return
    if element == "shell3":
        from .elements import shell
        bad=[]
        for e,row in enumerate(conn):
            try:
                _R,_c,local=shell.frame_tri(x[row])
                _N,d = shell.shape_tri(1/3,1/3)
                det=np.linalg.det(d@local[:,:2])
                if det<=0: bad.append(e)
            except ValueError: bad.append(e)
        if bad: _issue(items,"ERROR","INVALID_SHELL_FACE","shell3 faces are degenerate or inverted",count=len(bad),first=bad[:8])
        return
    from .elements import library
    definition = library.get(element)
    points, _ = definition.gauss("full")
    dref = definition.dN_dxi(points)
    J = np.einsum("gik,ekj->egij", dref, x[conn])
    det = np.linalg.det(J)
    bad = np.any(det <= 0, axis=1)
    if bad.any():
        _issue(items, "ERROR", "NONPOSITIVE_JACOBIAN", "elements have a non-positive integration-point Jacobian",
               count=int(bad.sum()), first=np.flatnonzero(bad)[:8].tolist())
    positive = np.where(det > 0, det, np.nan)
    quality = np.nanmin(positive, axis=1) / np.nanmax(positive, axis=1)
    distorted = np.isfinite(quality) & (quality < 0.05)
    if distorted.any():
        _issue(items, "WARNING", "DISTORTED_ELEMENT", "element Jacobian ratio is below 0.05",
               count=int(distorted.sum()), minimum=float(np.nanmin(quality)))
    corners = {"tet4": 4, "tet10": 4, "hexa8": 8, "hexa20": 8, "wedge6": 6, "wedge15": 6, "pyramid5": 5}[element]
    xc = x[conn[:, :corners]]
    distances = np.linalg.norm(xc[:, :, None] - xc[:, None, :], axis=-1)
    distances[distances == 0] = np.nan
    aspect = np.nanmax(distances, axis=(1, 2)) / np.nanmin(distances, axis=(1, 2))
    if np.any(aspect > 20):
        _issue(items, "WARNING", "HIGH_ASPECT_RATIO", "element corner-distance ratio exceeds 20",
               count=int(np.sum(aspect > 20)), maximum=float(np.max(aspect)))


def _constraint_mask(case, mesh, ndof, soil=False):
    mask = np.zeros((mesh.n_nodes, ndof), dtype=bool)
    specs = list(case.constraints)
    if case.steps: specs += list(case.steps[0].constraints)
    for spec in specs:
        if spec.region not in mesh.regions: continue
        dofs = np.asarray(spec.dofs, dtype=bool)[:ndof]
        mask[np.asarray(mesh.regions[spec.region]), :len(dofs)] |= dofs
    # soil springs hold a pile sideways below the ground, as a support does:
    # a pile standing in soil is not a free body. For the
    # rigid-mode check only: counted as a constraint everywhere, a buried
    # pile cap's diaphragm read as over-constrained and a pile under a
    # ground motion as fully restrained (review of 1.2)
    for spec in (getattr(case, "soil_springs", None) or []) if soil else []:
        if spec.region not in mesh.regions:
            continue
        up = "xyz".index(spec.vertical)
        nodes = np.asarray(mesh.regions[spec.region])
        nodes = nodes[mesh.nodes[nodes, up] <= float(spec.ground)]
        axes = [i for i in range(3) if i != up] if spec.direction == "horizontal" else ["xyz".index(spec.direction)]
        for a in axes:
            mask[nodes, a] = True
    # A RIGID LINK OF SOME DIRECTIONS (1.21) holds its slave in them from its
    # master, as a support does: a stay hung on a tower by its translations
    # alone is no free body. (One of all six joins its nodes to their master's
    # part instead, `_components`.) For the rigid-mode check only.
    links = list(getattr(case, "rigid_links", None) or []) + [
        lk for st in (getattr(case, "steps", None) or []) for lk in getattr(st, "rigid_links", None) or []]
    for lk in links if soil else []:
        if lk.slaves not in mesh.regions or all(lk.dofs):
            continue
        dofs = np.asarray(lk.dofs, dtype=bool)[:ndof]
        mask[np.asarray(mesh.regions[lk.slaves]), :len(dofs)] |= dofs
    return mask


def _import_notes(mesh, items):
    """Say what the reader had to repair on the way in.

    Silently correcting a file is how a user ends up trusting a mesh that is
    not the one they asked Gmsh for.  A cell listed under two entities is a
    modelling choice and only worth an INFO; a cell repeated inside ONE entity
    is a meshing failure upstream, and the user has to hear about it even
    though the repair itself is unambiguous.
    """
    for note in getattr(mesh, "import_notes", None) or []:
        if note["across_entities"]:
            _issue(items, "INFO", "MERGED_SHARED_ELEMENTS",
                   f"{note['across_entities']} {note['element']} cells were "
                   "listed under more than one entity and have been merged into "
                   "one element each, belonging to every group that named them",
                   element=note["element"], count=note["across_entities"])
        if note["within_entity"]:
            _issue(items, "WARNING", "REPEATED_ELEMENTS_IN_ONE_VOLUME",
                   f"{note['within_entity']} {note['element']} cells were "
                   "written more than once inside a single volume and have been "
                   "merged; this is not something a physical group can explain, "
                   "so the mesher produced it",
                   element=note["element"], count=note["within_entity"],
                   hint="check the Gmsh log for errors -- `Recombine Surface "
                        "{:}` recombines internal interfaces too, which Gmsh "
                        "reports as `non-manifold quad boundaries not supported "
                        "yet` and then writes the file anyway. Recombine only "
                        "the outer surfaces.")


def _solid_blocks(mesh):
    """Every element block in the model, as connectivity arrays.

    `mesh.elements` is the largest block only. Anything that reasons about the
    model as a body -- connectivity, support, the outer surface -- has to see
    all of them or it is describing a different structure than the one that
    will be solved.
    """
    from .elements.library import ELEMENTS
    blocks = getattr(mesh, "cell_blocks", None) or {}
    solid = [v for k, v in sorted(blocks.items()) if k in ELEMENTS]
    return solid or [np.asarray(mesh.elements)]


def _components(mesh, blocks=None, case=None):
    """Connected components over `blocks`, or over the solid blocks.

    `blocks` is how a mixed-dimensional model asks the question it actually
    has: the walls of a building are 55 disjoint quad patches and its frame is
    what joins them, so the components of the shell block alone are not the
    components of the structure.
    """
    parent = np.arange(mesh.n_nodes)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    # every block, not just the largest: a mixed mesh whose hexahedra are the
    # minority block would otherwise report each of their nodes as its own
    # component and raise hundreds of false RIGID_BODY_MODE errors
    for conn in (blocks if blocks is not None else _solid_blocks(mesh)):
        for row in conn:
            root = find(int(row[0]))
            for node in row[1:]:
                a, b = root, find(int(node))
                if a != b: parent[b] = a
    # a rigid link of all six components joins its slaves to its master
    # (1.21), as a member does; a partial one leaves the slave's part its
    # own mechanism, which the check below is to find
    # (a step's too: the members a step builds are joined here as well -- the
    # check is of the model, not of one stage; 1.21)
    links = [] if case is None else list(getattr(case, "rigid_links", None) or []) + [
        lk for st in (getattr(case, "steps", None) or []) for lk in getattr(st, "rigid_links", None) or []]
    for lk in links:
        if lk.master not in mesh.regions or lk.slaves not in mesh.regions or not all(lk.dofs):
            continue
        m = np.asarray(mesh.regions[lk.master]).ravel()
        if not len(m):
            continue
        # a master of several nodes is refused by the solver, with its own
        # message; joined here to its first so that message is the one met
        for node in np.asarray(mesh.regions[lk.slaves]).ravel():
            a, b = find(int(m[0])), find(int(node))
            if a != b: parent[b] = a
    groups = {}
    for node in range(mesh.n_nodes): groups.setdefault(find(node), []).append(node)
    return [np.asarray(nodes, dtype=int) for nodes in groups.values()]


_MODE_NAMES = ("translation x", "translation y", "translation z",
               "rotation about x", "rotation about y", "rotation about z")


def _free_rigid_modes(modes, restrained, xyz, nodes, fixed, ndof):
    """Name each unsuppressed rigid mode and propose one constraint that kills it.

    The check used to say "5 of 6" and stop, leaving the analyst to work out
    which mode and where to pin it -- on case_12 that took a round trip
    through the solver. The null space of the restrained mode matrix says
    which; for each null vector the node where that mode moves most, and the
    dof it moves it along, is a constraint that removes it and, being at the
    largest lever arm, disturbs the model least. The suggestion is printed
    as the YAML the case needs.
    """
    lines = []
    if restrained.size:
        _, sv, vt = np.linalg.svd(restrained, full_matrices=True)
        null = [vt[i] for i in range(6) if i >= len(sv) or sv[i] < 1e-10]
    else:
        null = list(np.eye(6))
    for v in null:
        v = v / np.linalg.norm(v)
        name = " + ".join(_MODE_NAMES[i] for i in np.argsort(-np.abs(v))[:2]
                          if abs(v[i]) > 0.3) or _MODE_NAMES[int(np.argmax(np.abs(v)))]
        motion = modes[:, :3, :] @ v                       # (n_nodes, 3)
        amp = np.linalg.norm(motion, axis=1)
        amp[fixed[nodes].any(axis=1)] *= 0.5               # prefer a fresh node
        k = int(np.argmax(amp))
        d = int(np.argmax(np.abs(motion[k])))
        dofs = [0] * ndof
        dofs[d] = 1
        if ndof == 6 and modes.shape[1] == 6:
            # a six-dof model can hold a rotation directly: the rotational
            # dof of a node that spins with the mode (v1.0 -- the
            # advice was a translation, three entries long, at the node already
            # pinned, for a straight member's twist about its own axis)
            spin = modes[:, 3:, :] @ v
            s_amp = np.linalg.norm(spin, axis=1)
            if s_amp.max() > 1e-12 and np.linalg.norm(v[3:]) > 0.3:
                s_amp[fixed[nodes].any(axis=1)] *= 0.5
                k = int(np.argmax(s_amp))
                dofs = [0] * 6
                dofs[3 + int(np.argmax(np.abs(spin[k])))] = 1
        x, y, z = xyz[k]
        lines.append(f"free: {name}. One fix: add region "
                     f"{{where: {{x: {x:.4g}, y: {y:.4g}, z: {z:.4g}}}}} and constrain it with "
                     f"dofs: {dofs}")
    return lines


def _held_by_contact(case) -> bool:
    """A model with a rigid contact surface may have a free
    body that the surface holds: the mechanism checks then warn rather than
    refuse, because the solver's contact force is what restrains it."""
    contact = getattr(case, "contact", None)
    return bool(contact and (getattr(contact, "rigid", None) or getattr(contact, "members", None)))


def _rotating_nodes(case, mesh, element):
    """(n_nodes,) bool: the nodes a member with rotational stiffness meets,
    for a mixed_line model whose member groups say which are bars; None when
    every node rotates (any other model, or groups that cannot be placed)."""
    members = getattr(case, "members", None) or []
    if element != "mixed_line" or not members:
        return None
    spin = np.zeros(mesh.n_nodes, dtype=bool)
    elements = np.asarray(mesh.elements)[:, :2]
    for m in members:
        if getattr(m, "type", None) == "truss2":
            continue
        ids = mesh.region_elements(m.region, "truss2", elements)
        if ids is None:
            return None
        spin[elements[np.asarray(ids, dtype=np.int64)].ravel()] = True
    return spin


def _rigid_modes(case, mesh, element, items, blocks=None):
    ndof = 6 if element in _SIX_DOF_TYPES else 3
    fixed = _constraint_mask(case, mesh, ndof, soil=True)
    components = _components(mesh, blocks, case)
    for c, nodes in enumerate(components):
        centre = mesh.nodes[nodes].mean(axis=0); r = mesh.nodes[nodes] - centre
        modes = np.zeros((len(nodes), ndof, 6))
        modes[:, 0, 0] = 1; modes[:, 1, 1] = 1; modes[:, 2, 2] = 1
        axes = np.eye(3)
        for j, axis in enumerate(axes):
            modes[:, :3, 3+j] = np.cross(axis, r)
            if ndof == 6: modes[:, 3:, 3+j] = axis
        if ndof == 6:
            # a node only bars meet carries no rotation a member could resist:
            # its rotational dofs are not part of any rigid motion. A straight
            # bar chain of a mixed_line model, held transversely at every
            # node, was refused for "rotation about x"
            spin = _rotating_nodes(case, mesh, element)
            if spin is not None:
                modes[~spin[nodes], 3:, :] = 0.0
        # A collinear truss has no nodal displacement for rotation about its
        # own axis, so its observable rigid-mode space is rank 5 rather than 6.
        required = int(np.linalg.matrix_rank(modes.reshape(-1, 6), tol=1e-10))
        restrained = modes[fixed[nodes]]
        rank = int(np.linalg.matrix_rank(restrained, tol=1e-10)) if restrained.size else 0
        if rank < required:
            free = _free_rigid_modes(modes, restrained, mesh.nodes[nodes], nodes, fixed, ndof)
            # Under relaxation a free body has no equilibrium to relax to, so
            # this is an error. Under dynamics with a physical mass it is a
            # legitimate model -- the particle equations of motion need no
            # support, and a free flight (Wu & Ting 2020, example 1) is the
            # test of exactly that -- so it is a warning that names the
            # modes, in case the analyst did not mean the body to fly.
            dynamic = case.solver.mode == "dynamic"
            held = _held_by_contact(case)
            _issue(items, "WARNING" if (dynamic or held) else "ERROR", "RIGID_BODY_MODE",
                   f"connected component {c} suppresses only {rank} of {required} observable rigid-body modes"
                   + ("; under mode = dynamic it is free to fly, and will unless the loads self-equilibrate"
                      if dynamic else "")
                   + ("; a rigid contact surface may hold it -- an equilibrium exists only while it "
                      "rests on the surface" if held else "")
                   + "".join("\n    " + line for line in free),
                   component=c, nodes=len(nodes), unconstrained_modes=required-rank)
    if element == "truss2":
        free = ~fixed[:, :3].reshape(-1)
        nfree = int(free.sum())
        if nfree <= 2000:
            R = np.zeros((mesh.n_elem, mesh.n_nodes*3))
            for e, (i, j) in enumerate(mesh.elements):
                d = mesh.nodes[j]-mesh.nodes[i]; length = np.linalg.norm(d)
                if length > 0:
                    n = d/length; R[e, 3*i:3*i+3] = -n; R[e, 3*j:3*j+3] = n
            rank = int(np.linalg.matrix_rank(R[:, free], tol=1e-10))
            mechanisms = nfree-rank
            if mechanisms:
                held = _held_by_contact(case)
                _issue(items, "WARNING" if held else "ERROR", "TRUSS_MECHANISM",
                       f"truss rigidity matrix has {mechanisms} free mechanism DOFs"
                       + ("; a rigid contact surface may hold them" if held else ""),
                       free_dofs=nfree, rank=rank)
        else:
            _issue(items, "WARNING", "TRUSS_RANK_SKIPPED", "exact truss mechanism rank skipped above 2000 free DOFs",
                   free_dofs=nfree)


def _line_assignments(case, mesh, element, items):
    if element != "mixed_line": return
    owner = np.zeros(mesh.n_elem, dtype=int)
    for group in case.members:
        if group.region not in mesh.regions and group.region not in mesh.element_regions:
            continue
        ids = mesh.region_elements(group.region, "truss2", mesh.elements)
        if ids is None: continue
        selected = np.zeros(mesh.n_elem, dtype=bool); selected[ids] = True
        owner += selected
        if group.type in _BEAM_TYPES | {"connector2"} and group.orientation is not None:
            d = mesh.nodes[mesh.elements[selected, 1]]-mesh.nodes[mesh.elements[selected, 0]]
            if len(d):
                n = d/np.linalg.norm(d, axis=1)[:, None]
                o = np.asarray(group.orientation, dtype=float); o /= np.linalg.norm(o)
                if np.any(np.abs(n@o) > 1-1e-8):
                    code = "BEAM_ORIENTATION" if group.type in _BEAM_TYPES else "CONNECTOR_ORIENTATION"
                    _issue(items, "ERROR", code, f"member group {group.region!r} orientation is parallel to its Line2 axis")
    if np.any(owner == 0):
        _issue(items, "ERROR", "UNASSIGNED_MEMBER", "mixed-line elements lack a member assignment", count=int(np.sum(owner == 0)))
    if np.any(owner > 1):
        _issue(items, "ERROR", "OVERLAPPING_MEMBER", "mixed-line elements have multiple member assignments", count=int(np.sum(owner > 1)))


def _block_regions(case, mesh, element, items):
    """Every block's region must exist, and every element must land in one.
    The solver refuses the same things; preflight says them first, in the
    report, with the counts. Runs after `_regions` has resolved them."""
    from .elements.library import ELEMENTS
    from .core.solver import partition_elements

    regions = mesh.regions
    for b in case.blocks:
        if b.region not in regions:
            _issue(items, "ERROR", "UNKNOWN_REGION",
                   f"section {b.name!r} is assigned to region {b.region!r}, which does not exist",
                   region=b.region, available=sorted(regions))
            return
    blocks = getattr(mesh, "cell_blocks", None) or {}
    conns = [v for k, v in sorted(blocks.items()) if k in ELEMENTS] or [mesh.elements]
    specs = [{"region": b.region} for b in case.blocks]
    counts = np.zeros(len(specs), dtype=int)
    lost = 0
    for conn in conns:
        owner, overrides = partition_elements(conn, mesh.n_nodes, specs, lambda r: regions[r])
        lost += int((owner < 0).sum())
        for i in range(len(specs)):
            counts[i] += int((owner == i).sum())
        for j, i, n in overrides:
            _issue(items, "INFO", "SECTION_OVERRIDE",
                   f"section {case.blocks[i].name!r} takes {n} element(s) also covered by "
                   f"{case.blocks[j].name!r} (later assignment wins)", count=n)
    if lost:
        _issue(items, "ERROR", "ELEMENTS_UNASSIGNED",
               f"{lost} element(s) are covered by no assigned section; an element is in a "
               "section only when every one of its nodes is in its region", count=lost)
    for i, b in enumerate(case.blocks):
        if counts[i] == 0:
            _issue(items, "ERROR", "SECTION_EMPTY",
                   f"section {b.name!r} on region {b.region!r} holds no whole element", section=b.name)
        else:
            _issue(items, "INFO", "SECTION_ELEMENTS",
                   f"section {b.name!r}: {int(counts[i])} element(s), material {b.material.get('type')}",
                   section=b.name, count=int(counts[i]))


#: how many metres one unit of `units.length` is. A system this does not
#: know is not checked rather than guessed at.
_LENGTH_IN_M = {"m": 1.0, "metre": 1.0, "meter": 1.0,
                "mm": 1.0e-3, "millimetre": 1.0e-3, "millimeter": 1.0e-3,
                "cm": 1.0e-2, "km": 1.0e3,
                "in": 0.0254, "inch": 0.0254, "ft": 0.3048, "foot": 0.3048}

#: dilatational wave speeds, m/s: 100 is softer than saturated soil, 20 000
#: faster than any structural solid (steel 5900, concrete 3500, timber 5000,
#: diamond 18 000).
_WAVE_SPEED_M_S = (100.0, 20000.0)


# one table each, shared with the conversion of defaults (core/units.py)
from .core.units import FORCE_IN_N as _FORCE_IN_N, STRESS_IN_PA as _STRESS_IN_PA  # noqa: E402
_UNIT_KEYS = ("length", "force", "stress", "moment", "mass", "time", "temperature", "density")


def _units_keys(case, units, items):
    """The declaration itself: a misspelt key was read as
    no key, and a stress unit that is not force / length^2 passed."""
    import difflib
    for k in units:
        if k not in _UNIT_KEYS:
            # an ERROR: as a warning the case still PASSED,
            # and `units: {lenght: mm}` ran with its length unit undeclared
            # -- the wave-speed check against it silently skipped
            near = difflib.get_close_matches(k, _UNIT_KEYS, n=1) or \
                [u for u in _UNIT_KEYS if u == str(k).lower()]
            _issue(items, "ERROR", "UNITS_UNKNOWN_KEY",
                   f"units.{k} is not a key `units` takes"
                   + (f" (did you mean units.{near[0]}?)" if near else "")
                   + "; it takes " + ", ".join(_UNIT_KEYS), key=k)
    # the system as the declaration completes it (core/units.complete,
    # 1.0 re-review): `{force: N, stress: MPa}` is mm -- the checks below read
    # the length and force it implies; a stress beside which the other two
    # name no unit is refused
    from .core.units import complete
    declared = units
    units, inferred, missing = complete(units)
    for k in inferred:
        _issue(items, "INFO", "UNITS_INFERRED",
               f"units.{k} is not declared: {units[k]!r} is what units.stress = "
               f"{declared['stress']!r} with units.{'force' if k == 'length' else 'length'} = "
               f"{units['force' if k == 'length' else 'length']!r} implies, and it is read so",
               key=k, unit=units[k])
    for k in missing:
        _issue(items, "ERROR", "UNITS_INCOMPLETE",
               f"units.{k} is missing: units.stress = {declared.get('stress')!r} is declared, and "
               + (f"with units.{'force' if k == 'length' else 'length'} = "
                  f"{declared.get('force' if k == 'length' else 'length')!r} it names no {k} unit"
                  if ('force' if k == 'length' else 'length') in declared else
                  "without units.length or units.force it fixes neither")
               + "; the defaults that carry a unit (densities, moduli, g) cannot be written in "
               "the case's system", key=k,
               hint=f"declare units.{k} (mm / N / MPa or m / kN / kPa are consistent systems)")
    L = _LENGTH_IN_M.get(str(units.get("length", "")).lower())
    F = _FORCE_IN_N.get(str(units.get("force", "")).lower())
    S = _STRESS_IN_PA.get(str(units.get("stress", "")).lower())
    if L and F and S and abs(np.log(S / (F / L ** 2))) > 1e-3:
        _issue(items, "WARNING", "UNITS_INCONSISTENT",
               f"units.stress = {units['stress']!r} is not units.force / units.length^2 "
               f"({units['force']}/{units['length']}^2); the solver takes one consistent "
               "system, so the stresses and the forces cannot both be in the declared units")
    if L:
        g_expected = 9.80665 / L
        loads = list(case.loads) + [l for st in case.steps for l in st.loads]
        for ld in loads:
            g = getattr(ld, "gravity", None)
            if g is None:
                continue
            mag = float(np.linalg.norm(np.asarray(g, float)))
            if mag > 0 and abs(np.log(mag / g_expected)) > np.log(3.0):
                _issue(items, "WARNING", "GRAVITY_MAGNITUDE",
                       f"a gravity load of {mag:.4g} with units.length = {units.get('length')!r}: "
                       f"g is {g_expected:.6g} in that unit (9.81 m/s^2 is 9810 mm/s^2)",
                       gravity=mag)
                break


def _shell_normals(case, mesh, items):
    """A pressure on shells whose normals disagree.

    A shell's pressure acts along its normal, and the normal is its node
    order. Two Gmsh surfaces oriented oppositely gave a net force of 0 under
    a uniform pressure, with nothing said. Neighbours sharing an edge are
    consistently oriented when they run that edge in opposite directions."""
    blocks = {k: v for k, v in (getattr(mesh, "cell_blocks", None) or {}).items()
              if k in ("shell3", "shell4")}
    if not blocks:
        return
    loads = list(case.loads) + [l for st in case.steps for l in st.loads]
    for ld in loads:
        if getattr(ld, "pressure", None) is None:
            continue
        region = mesh.regions.get(ld.region)
        if region is None:
            continue
        inside = np.zeros(mesh.n_nodes, dtype=bool)
        inside[np.asarray(region, dtype=np.int64)] = True
        edges = {}
        bad = 0
        for conn in blocks.values():
            conn = np.asarray(conn)
            for row in conn[inside[conn].all(axis=1)]:
                n = len(row)
                for a in range(n):
                    i, j = int(row[a]), int(row[(a + 1) % n])
                    if (i, j) in edges:
                        bad += 1                 # the same edge, the same way
                    edges[(i, j)] = True
        if bad:
            _issue(items, "ERROR", "SHELL_NORMALS_INCONSISTENT",
                   f"the pressure on {ld.region!r} acts on shells whose normals point different "
                   f"ways ({bad} shared edge(s) run the same way by both neighbours), so parts "
                   "of it cancel",
                   hint="orient the surfaces alike (Gmsh: ReverseMesh / Reverse Surface), or "
                        "apply a traction vector instead", region=ld.region)


def _units_check(case, items):
    """Do the case's own numbers agree with the units it declares?

    `units:` was stored and never read by anything, so a case written in
    mm/tonne/MPa while declaring m/N/Pa ran happily and produced numbers that
    were wrong by powers of a thousand, in silence. There is no unit
    conversion here and there should not be -- the solver takes one
    consistent system, whichever it is -- but "consistent" is checkable, and
    the check is a physical invariant the solver already computes for its own
    time step: the wave speed sqrt(E/rho), read in the DECLARED length unit.

        SI throughout                E 2.1e11, rho 7850   -> 5170 m/s   ok
        mm/tonne/MPa throughout      E 2.1e5, rho 7.85e-9 -> 5170 m/s   ok
        MPa with kg/m3, declaring m  E 2.1e5, rho 7850    ->    5 m/s   caught
        SI numbers declaring mm      E 2.1e11, rho 7850   ->    5 m/s   caught

    A warning, not an error: an unusual material (a foam, a soft soil) can
    sit outside the band honestly, and this cannot tell that from a mistake.
    What it can do is refuse to be silent about it.
    """
    units = case.units or {}
    _units_keys(case, units, items)
    from .core.units import complete
    units = complete(units)[0]          # a length `stress` and `force` imply (1.0 re-review)
    length = str(units.get("length", "")).lower()
    factor = _LENGTH_IN_M.get(length)
    if factor is None:
        if length:
            _issue(items, "INFO", "UNKNOWN_LENGTH_UNIT",
                   f"units.length = {length!r} is not one this checker knows, so the "
                   "case's magnitudes are not checked against it",
                   unit=length)
        return
    mats = [("", case.material)] + [(b.name, b.material) for b in case.blocks]
    mats += [(f"reinforcement {r.name or i + 1}", r.material)
             for i, r in enumerate(case.reinforcement)]
    lo, hi = _WAVE_SPEED_M_S
    for label, m in mats:
        try:
            E = float(m.get("E")); rho = float(m.get("rho"))
        except (TypeError, ValueError):
            continue
        if not (E > 0 and rho > 0):
            continue
        c = float(np.sqrt(E / rho)) * factor          # declared units -> m/s
        if lo <= c <= hi:
            continue
        who = f"{label}: " if label else ""
        # Which of the two systems anyone actually uses would reconcile these
        # numbers? Only m and mm are offered: `units.length: km` also puts a
        # MPa-with-kg/m3 mixture inside the band, arithmetically, and telling
        # someone that would be worse than saying nothing.
        raw = float(np.sqrt(E / rho))
        fits = [(abs(np.log(raw * f / 4000.0)), n, raw * f)
                for n, f in (("m", 1.0), ("mm", 1.0e-3))
                if lo <= raw * f <= hi and n != length]
        if fits:
            _, name, speed = min(fits)
            fix = (f". Read as `units.length: {name}` the same numbers give "
                   f"{speed:.0f} m/s, so it is the DECLARATION that is wrong")
        else:
            # no length unit reconciles them: the mixture is inside the
            # material, and this is the pair people mix
            fix = (". No length unit reconciles them, so it is E or rho that is in "
                   "another system: SI is E ~ 2.1e11 Pa with rho ~ 7850 kg/m3, and "
                   "mm/tonne/MPa is E ~ 2.1e5 with rho ~ 7.85e-9")
        _issue(items, "WARNING", "UNIT_MAGNITUDE",
               f"{who}E = {E:.4g} and rho = {rho:.4g} give a wave speed of {c:.4g} m/s "
               f"with units.length = {length!r}, outside the {lo:g}-{hi:g} m/s any "
               f"structural solid has{fix}. The solver does not convert units: every "
               "number in the case must be in ONE consistent system",
               wave_speed=c, E=E, rho=rho, length_unit=length)


def _crack_band_checks(case, mesh, element, items):
    """What the CDP compression band factor will be on this mesh, before the
    solver refuses it (materials/cdp.py bind_points has the reasoning)."""
    from .elements.library import ELEMENTS
    if element not in ELEMENTS:
        return
    # with blocks, `material` is block 0's again; list each section once
    mats = ([(b.name, b.material) for b in case.blocks] if case.blocks
            else [("", case.material)])
    cdp = [(n, m) for n, m in mats if m.get("type") == "cdp"]
    # A code compression curve is a STRAIN curve: without a crushing energy
    # its band dissipates an energy per unit area that scales with the
    # element size, and the peak load moves with the mesh.
    # Not an error -- it is what every version before that did, and it is
    # defensible on a fixed mesh -- but never silent.
    for label, m in cdp:
        comp = m.get("compression") or {}
        if comp.get("source") in ("gb50010", "table") and comp.get("Gc") is None:
            who = f"section {label!r}" if label else "the material"
            _issue(items, "INFO", "COMPRESSION_UNREGULARISED",
                   f"{who}: the compression curve has no `Gc`, so its crushing band is "
                   "not regularised and the peak load will move with the element size; "
                   "state a crushing energy (J/m2) to fix the band's energy per area",
                   source=comp.get("source"))
    mats = [(n, m) for n, m in cdp if m.get("crack_band") == "both"]
    if not mats:
        return
    from .elements import continuum
    from .materials.cdp import COMPRESSION_BAND_MAX, COMPRESSION_BAND_MIN
    blocks = getattr(mesh, "cell_blocks", None) or {}
    conns = [(k, v) for k, v in sorted(blocks.items()) if k in ELEMENTS] or [(element, mesh.elements)]
    try:
        lengths = np.concatenate([
            continuum.precompute(mesh.nodes, conn, element=name, formulation="reduced").char_len
            for name, conn in conns])
    except Exception:
        return                       # the mesh checks report a bad element
    l_min, l_max = float(lengths.min()), float(lengths.max())
    for label, m in mats:
        l_ref = m.get("l_ref", 1.0)
        l_ref = l_min if str(l_ref).lower() == "auto" else float(l_ref)
        k_lo, k_hi = l_min / l_ref, l_max / l_ref
        who = f"section {label!r}" if label else "the material"
        if k_lo < COMPRESSION_BAND_MIN or k_hi > COMPRESSION_BAND_MAX:
            _issue(items, "ERROR", "COMPRESSION_BAND_RATIO",
                   f"{who}: crack_band: both with l_e/l_ref = {k_lo:.3g}..{k_hi:.3g} "
                   f"(elements {l_min:.4g}..{l_max:.4g} m, l_ref {l_ref:.4g} m) is outside "
                   f"[{COMPRESSION_BAND_MIN:g}, {COMPRESSION_BAND_MAX:g}]; the compression "
                   "curve has no reference length, so write `l_ref: auto`, or "
                   "`crack_band: tension` on a graded mesh",
                   k_min=k_lo, k_max=k_hi, l_ref=l_ref)
        else:
            _issue(items, "INFO", "CRACK_BAND",
                   f"{who}: l_e/l_ref = {k_lo:.3g}..{k_hi:.3g} on both curves "
                   f"(l_ref {l_ref:.4g} m)", k_min=k_lo, k_max=k_hi, l_ref=l_ref)


def _line_points(ln):
    """The route a reinforcement line entry describes, as a point array.

    `start`/`end` is a two-point route; `points` is the general one; `closed`
    returns to the first point. One entry is ONE bar -- io/schema.BarLineSpec
    says why -- so everything measured per bar is measured on this.
    """
    if getattr(ln, "points", None):
        p = np.asarray(ln.points, dtype=float)
        return np.vstack([p, p[:1]]) if getattr(ln, "closed", False) else p
    return np.asarray([ln.start, ln.end], dtype=float)


def _route_length_and_turn(pts):
    d = np.diff(pts, axis=0)
    L = np.linalg.norm(d, axis=1)
    if not np.all(L > 0):
        return 0.0, 0.0
    t = d / L[:, None]
    if len(t) < 2:
        return float(L.sum()), 0.0
    cos = np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1.0, 1.0)
    return float(L.sum()), float(np.arccos(cos).sum())


def _tendon_run(spec):
    """(length, total turn angle) of the LONGEST bar in a set.

    The two numbers a friction calculation needs, and the solver does not do
    friction -- so the useful thing preflight can do with a curved tendon is
    hand the analyst x and theta for the sigma_con (1 - e^-(mu theta + k x))
    they have to apply themselves. A set with `repeat` is identical copies, so
    one of them describes all of them.
    """
    runs = [_route_length_and_turn(_line_points(ln)) for ln in (spec.lines or [])]
    return max(runs, default=(0.0, 0.0))


def _split_route_check(case, items):
    """Bonded bars drawn as several touching entries are several bars.

    Every line entry gets its own nodes, so two entries that meet at a point
    are not joined there. A perfectly bonded bar does not care -- coincident
    bar nodes take the same host displacement -- but a bonded one does: each
    piece slips on its own and pulls out at tau_max, and `anchorage: ends`
    ties the ends of every piece instead of the two ends of the bar. `points`
    lays the same route as one bar.
    """
    for i, spec in enumerate(case.reinforcement or []):
        if spec.bond is None:
            continue
        lines = spec.lines or []
        touching = sum(1 for a, b in zip(lines, lines[1:])
                       if a.end is not None and b.start is not None
                       and np.allclose(a.end, b.start, atol=1e-9))
        if touching:
            name = spec.name or f"rebar{i + 1}"
            _issue(items, "WARNING", "SPLIT_BAR_ROUTE",
                   f"bonded reinforcement {name!r} has {touching + 1} line(s) that "
                   "meet end to start but are laid as separate bars, each free to "
                   "slip and pull out on its own; draw the route as one line with "
                   "`points: [[x,y,z], ...]`",
                   joins=touching)


def _prestress_check(case, items):
    """Say the prestress back as a force, and say what it does not include.

    `prestress` is a stress in a YAML full of stresses, and the thing an
    analyst actually checks a prestressed model against is a FORCE per tendon
    and a total. Reporting sigma * A here is one multiplication that catches
    an area in mm2, a stress in MPa, and a tendon count that is not what was
    drawn -- all before a two-hour run.

    The second line exists because the word "prestress" carries losses with it
    in every design code. This solver produces the elastic-shortening loss and
    no other, so the number the analyst writes has to already be net of
    friction, anchor set, relaxation and creep, and that is worth saying at
    the point they can still change it.
    """
    total = 0.0
    for i, spec in enumerate(case.reinforcement or []):
        sigma = getattr(spec, "prestress", None)
        if not sigma:
            continue
        name = spec.name or f"rebar{i + 1}"
        # one ENTRY is one bar, and a `repeat` on it is that many. Counting
        # segments instead would report a parabolic tendon sampled into 32
        # chords as 32 tendons, with 32 times the force actually in the member
        # -- exactly the arithmetic this report exists to catch.
        n_bars = sum((ln.repeat.count if ln.repeat else 1)
                     for ln in (spec.lines or [])) or 1
        force = float(sigma) * float(spec.area)
        total += force * n_bars
        mat = spec.material or {}
        sy = mat.get("yield_strength")
        ratio = f", {float(sigma) / float(sy):.2f} f_y" if sy else ""
        _issue(items, "INFO", "PRESTRESS",
               f"reinforcement {name!r}: {float(sigma) / 1e6:.0f} MPa{ratio} "
               f"on {float(spec.area) * 1e6:.0f} mm2 = {force / 1e3:.1f} kN "
               f"per bar, {n_bars} bar(s), {force * n_bars / 1e3:.1f} kN "
               "before elastic shortening",
               stress=float(sigma), force=force, lines=n_bars)
        length, theta = _tendon_run(spec)
        fr = spec.friction
        if fr is not None:
            # `prestress` is the JACKING stress here, and the far-end value is
            # the one an analyst checks against the design -- so say it, rather
            # than making them read exp() by eye.
            far = float(sigma) * math.exp(-(fr.mu * theta + fr.k * length))
            _issue(items, "INFO", "PRESTRESS_PROFILE",
                   f"reinforcement {name!r}: {float(sigma) / 1e6:.0f} MPa AT THE JACK "
                   f"({fr.jack}), {length:.3g} m long, turning {theta:.4g} rad; "
                   f"friction alone leaves {far / 1e6:.0f} MPa at the far end, "
                   f"a {(1.0 - far / float(sigma)) * 100:.1f}% loss"
                   + (f", before {fr.anchor_set * 1e3:.0f} mm of anchor set"
                      if fr.anchor_set else ""),
                   length=length, angle=theta, far=far)
        elif theta > 0.01:
            _issue(items, "INFO", "PRESTRESS_PROFILE",
                   f"reinforcement {name!r} is a curved or harped tendon: "
                   f"{length:.3g} m long, turning {theta:.4g} rad in total. "
                   "Friction is not computed here -- those are the x and theta "
                   "for the sigma_con (1 - exp(-(mu theta + k x))) you have to "
                   "apply before writing `prestress`, or add a `friction` block "
                   "and let the solver do it",
                   length=length, angle=theta)
    if total:
        computed = "friction, anchor set and elastic shortening" if any(
            s.friction is not None for s in (case.reinforcement or [])) else \
            "the elastic-shortening loss"
        left = ("relaxation and shrinkage/creep" if computed.startswith("friction")
                else "friction, anchor set, relaxation and shrinkage/creep")
        _issue(items, "INFO", "PRESTRESS_LOSSES",
               f"{total / 1e3:.1f} kN of prestress before losses; the solver computes "
               f"{computed} and nothing else -- `prestress` must already be net of "
               f"{left}",
               total=total)


def _fp32_strain_resolution(x, conn):
    """Strain error left by storing these reference coordinates in fp32.

    A segment vector is a difference of two coordinates. Store them in fp32
    and the difference keeps only what the seven significant digits left over
    after the cancellation, so the error in `(L - L0)/L0` scales with
    (coordinate magnitude)/(segment length) -- the model's extent divided by
    its element size, which is a number nobody thinks about when choosing a
    mesh.
    """
    a, b = np.asarray(x, float)[conn[:, 0]], np.asarray(x, float)[conn[:, 1]]
    L64 = np.linalg.norm(b - a, axis=1)
    ok = L64 > 0
    if not ok.any():
        return 0.0, 0.0
    a32, b32 = a[ok].astype(np.float32), b[ok].astype(np.float32)
    L32 = np.linalg.norm((b32 - a32).astype(np.float32), axis=1).astype(np.float64)
    err = (L32 - L64[ok]) / L64[ok]
    return float(np.sqrt(np.mean(err ** 2))), float(np.abs(err).max())


def _cracking_strain(case):
    """ft/E of the first material that has both, or None."""
    mats = ([b.material for b in case.blocks] if case.blocks else [case.material])
    for m in mats:
        ft = ((m.get("tension") or {}).get("ft")) if isinstance(m, dict) else None
        E = m.get("E") if isinstance(m, dict) else None
        if ft and E:
            return float(ft) / float(E)
    return None


def _mixed_precision_check(case, mesh, items):
    """What `precision: mixed` costs on a long model with small elements.

    THE RUN THIS EXISTS FOR. The post-tensioned T-beam (cases/case_15) is
    8.5 m long with 50 mm elements, a ratio of 170. On the warp backend at
    `precision: mixed` its transfer step spent 150000 iterations with 44581
    kinetic resets -- one every 3.4 steps -- and ended at a relative residual
    of 2.2e-1, where the same model converges to 1e-5 in 34000 iterations on
    the CPU. The tendon was the worst of it: 1.5e-5 of strain error, an eighth
    of the cracking strain, which at 195 GPa is 2.9 MPa of stress noise that
    relaxation cannot remove because it is not out of balance -- it is the
    arithmetic. That took an afternoon and two hours of card time to find, and
    every ingredient of it is known before the run starts.
    """
    sol = case.solver
    if getattr(sol, "backend", None) != "warp" or getattr(sol, "precision", "mixed") != "mixed":
        return
    worst_rms = worst_max = 0.0
    where = []
    conn = np.asarray(mesh.elements)
    if conn.ndim == 2 and conn.shape[1] >= 2:
        r, m = _fp32_strain_resolution(mesh.nodes, conn[:, :2])
        worst_rms, worst_max = max(worst_rms, r), max(worst_max, m)
        where.append(f"mesh {r:.1e} rms")
    from .elements import truss
    for i, spec in enumerate(case.reinforcement or []):
        try:
            bn, bc = truss.bars_from_spec(
                {"lines": [ln.model_dump(exclude_none=True) for ln in (spec.lines or [])]}
                if spec.lines else
                {"nodes": spec.nodes, "elements": spec.elements})
        except Exception:
            continue
        r, m = _fp32_strain_resolution(bn, bc)
        worst_rms, worst_max = max(worst_rms, r), max(worst_max, m)
        where.append(f"{spec.name or f'rebar{i + 1}'} {r:.1e} rms")
    if worst_max <= 0.0:
        return

    eps_cr = _cracking_strain(case)
    ratio = (worst_max / eps_cr) if eps_cr else None
    tail = (f", {ratio * 100:.1f}% of the cracking strain {eps_cr:.2e}"
            if ratio is not None else "")
    body = (f"`precision: mixed` keeps the element reference coordinates in fp32; "
            f"on this model that leaves {worst_max:.1e} of strain error at worst"
            f"{tail} ({'; '.join(where)})")
    if (ratio is not None and ratio > 0.05) or (ratio is None and worst_max > 2.0e-6):
        _issue(items, "WARNING", "MIXED_PRECISION_RESOLUTION",
               body + ". Relaxation cannot remove it -- it is not out of balance, "
               "it is the arithmetic -- and it shows up as kinetic damping "
               "resetting every few iterations and a residual that stops "
               "falling. Use `precision: fp64`, or coarsen nothing and expect "
               "a floor",
               rms=worst_rms, worst=worst_max, ratio=ratio)
    else:
        _issue(items, "INFO", "MIXED_PRECISION_RESOLUTION", body,
               rms=worst_rms, worst=worst_max, ratio=ratio)


def _mass_checks(case, mesh, element, items):
    rho = float(case.material.get("rho", 0.0))
    # A MEMBER GROUP MAY HAVE ITS OWN MATERIAL, and with it
    # its own density. `case.material` is only the first line section's, so a
    # model whose weight is all in the OTHER groups would be refused here for
    # having no mass at all.
    rho = max([rho] + [float((m.material or {}).get("rho", 0.0)) for m in case.members])
    carried = any(getattr(ld, "with_mass", False) for ld in case.loads)
    if case.solver.mode == "dynamic" and rho <= 0 and not case.nodal_masses and not carried:
        _issue(items, "ERROR", "ZERO_DYNAMIC_MASS", "dynamic analysis has no positive material or nodal mass")
    if case.solver.mode == "dynamic":
        # A LINE LOAD WEIGHS AND HAS NO INERTIA unless it says so (1.1): said
        # once, as information -- a live load or a wind load is rightly a
        # force alone, a superimposed dead load is not
        plain = [i for i, ld in enumerate(case.loads)
                 if ld.line_load is not None and not getattr(ld, "with_mass", False)]
        if plain:
            _issue(items, "INFO", "LINE_LOAD_WITHOUT_MASS",
                   f"loads {plain} {'is a line_load' if len(plain) == 1 else 'are line_loads'} without "
                   "with_mass: a force with no inertia in this dynamic run. If a load is the weight of "
                   "something (superimposed dead load, surfacing), `with_mass: true` gives it its mass |q|/g",
                   loads=plain)
    carriers = [i for i, ld in enumerate(case.loads) if getattr(ld, "with_mass", False)]
    if carriers:
        from .core.units import to_case
        if to_case(9.80665, "acceleration", case.units or {}) is None:
            _issue(items, "WARNING", "UNIT_DEFAULT_SI",
                   f"loads {carriers} carry their mass |q|/g with g = 9.80665 (SI): the declared "
                   "`units` name a unit this cannot convert an acceleration to, so the mass is in "
                   "error unless the case is in m and s", where=f"loads[{carriers[0]}].with_mass")
    if element in _BEAM_TYPES | {"connector2", "mixed_line"}:
        groups = case.members if element == "mixed_line" else [case.element]
        for group in groups:
            if getattr(group, "type", "beam2") in _BEAM_TYPES | {"connector2"} and group.orientation is not None:
                o = np.asarray(group.orientation, dtype=float)
                if not np.isfinite(o).all() or np.linalg.norm(o) <= 1e-14:
                    code = "BEAM_ORIENTATION" if getattr(group,"type",None) in _BEAM_TYPES else "CONNECTOR_ORIENTATION"
                    _issue(items, "ERROR", code, "member orientation must be a finite non-zero vector")
                elif element in _BEAM_TYPES | {"connector2"}:
                    n = mesh.nodes[mesh.elements[:, 1]]-mesh.nodes[mesh.elements[:, 0]]
                    n /= np.linalg.norm(n, axis=1)[:, None]; unit_o = o/np.linalg.norm(o)
                    if np.any(np.abs(n@unit_o) > 1-1e-8):
                        code = "BEAM_ORIENTATION" if getattr(group,"type",None) in _BEAM_TYPES else "CONNECTOR_ORIENTATION"
                        _issue(items, "ERROR", code, "orientation is parallel to a member axis")
            releases = set(getattr(group, "releases", []))
            for end in ("i", "j"):
                if {f"{end}_rx", f"{end}_ry", f"{end}_rz"} <= releases:
                    _issue(items, "WARNING", "FULL_ROTATIONAL_RELEASE",
                           f"beam {end}-end releases all three rotations; verify this does not create a mechanism")
            if getattr(group, "type", None) == "connector2":
                damping = np.asarray(group.connector_damping if group.connector_damping is not None else np.zeros(6), float)
                if case.solver.mode == "relaxation" and np.any(damping):
                    _issue(items, "ERROR", "CONNECTOR_PHYSICAL_DAMPING_IN_RELAXATION",
                           "connector_damping is physical-time damping and cannot act in relaxation pseudo-time")
    if element == "connector2" and case.solver.mode == "dynamic" and not case.nodal_masses:
        _issue(items, "ERROR", "CONNECTOR_DYNAMIC_MASS",
               "a standalone dynamic connector model needs nodal_masses; connectors carry no physical mass")


def _connector_rank(case, mesh, element, items):
    """Detect unconstrained connector components before pseudo-mass hides them."""
    if element != "connector2": return
    from .core.beam_solver import BeamSolver
    from .elements import connector
    ndof=6*mesh.n_nodes; K=np.zeros((ndof,ndof)); props=case.element.model_dump(mode="python")
    probe=object.__new__(BeamSolver)
    for conn in mesh.elements:
        if np.linalg.norm(mesh.nodes[conn[1]]-mesh.nodes[conn[0]]) < 1e-12:
            # a zero-length connector has no chord to build axes from: the
            # ZERO_LENGTH check reports it; this one raised a traceback before
            # it could
            continue
        reference=props.get("orientation")
        if reference is None:
            axis=mesh.nodes[conn[1]]-mesh.nodes[conn[0]]; axis/=np.linalg.norm(axis)
            basis=np.eye(3); reference=basis[np.argmin(np.abs(basis@axis))]
        R,_=probe._frame(mesh.nodes[conn[0]],mesh.nodes[conn[1]],reference)
        ke=connector.reference_stiffness(mesh.nodes[conn],R,props)
        if case.solver.mode=="dynamic":
            C=connector.damping_matrix(props)
            J=connector.jacobian(np.zeros((2,3)),None,mesh.nodes[conn],R,props)
            ke=ke+J.T@C@J
        dof=np.r_[6*conn[0]+np.arange(6),6*conn[1]+np.arange(6)]
        K[np.ix_(dof,dof)]+=ke
    fixed=_constraint_mask(case,mesh,6).reshape(-1); free=~fixed
    if free.sum()>2000:
        _issue(items,"WARNING","CONNECTOR_RANK_SKIPPED",
               "exact connector mechanism rank skipped above 2000 free DOFs",free_dofs=int(free.sum()))
        return
    A=K[np.ix_(free,free)]
    scale=max(float(np.abs(A).max(initial=0.0)),1.0)
    rank=int(np.linalg.matrix_rank(A,tol=scale*1e-11))
    if rank<len(A):
        held=_held_by_contact(case)
        _issue(items,"WARNING" if held else "ERROR","CONNECTOR_MECHANISM",
               f"connector model has {len(A)-rank} unstiffened/unrestrained DOF(s); constrain inactive components or connect them to physical members"
               + ("; a rigid contact surface may hold them" if held else ""),
               free_dofs=len(A),rank=rank)


def _ground_motion_direction(case, mesh, element, items):
    """A base acceleration along an axis nothing is free to move in is a
    thirty-five-minute run that reports nothing happened.

    `ground_motion` enters in the relative frame as `-m a_g(t)` on every
    node's TRANSLATION dofs. If the direction lies in dofs that are
    constrained everywhere, the whole of it goes into the reactions and the
    structure does not move -- exactly, not approximately. Nothing fails, the
    solve converges, every monitor is flat, and the natural reading is that
    the model is wrong somewhere.

    The case this was written for is a PLANE FRAME: `case_24f` and its
    seismic twin carry `{region: all, dofs: [0, 1, 0, 1, 0, 1]}`, so `uy` is
    locked at all 57 of 57 nodes. `direction: [0, 1, 0]` -- the transverse
    component of a real record, and the obvious second run to make -- excites
    nothing at all.
    """
    dof_names = ("ux", "uy", "uz")
    ndof = 6 if element in _SIX_DOF_TYPES else 3
    mask = _constraint_mask(case, mesh, ndof)
    loads = list(case.loads) + [x for s in case.steps for x in s.loads]
    for ld in loads:
        gm = getattr(ld, "ground_motion", None)
        if not gm:
            continue
        d = np.asarray(gm.get("direction", [1, 0, 0]), dtype=float)
        if not np.any(np.abs(d) > 0):
            _issue(items, "ERROR", "GROUND_MOTION_NO_DIRECTION",
                   "ground_motion direction is the zero vector")
            continue
        live = [k for k in range(min(3, ndof))
                if abs(d[k]) > 1e-12 and not mask[:, k].all()]
        if live:
            continue
        locked = [dof_names[k] for k in range(min(3, ndof)) if abs(d[k]) > 1e-12]
        _issue(items, "ERROR", "GROUND_MOTION_FULLY_RESTRAINED",
               f"ground_motion direction {d.tolist()} lies entirely in "
               f"{', '.join(locked)}, which is constrained at EVERY node: the "
               "base acceleration would go straight into the reactions and "
               "nothing would move. A plane frame cannot be shaken out of its "
               "plane -- release the DOF or shake in the plane.",
               direction=d.tolist(), locked=locked)


def _record_and_ground_motion(case, items):
    """A support's own `record` and a global `ground_motion` along the same
    direction in one step: both were applied -- the base acceleration as
    -M a_g in the relative frame AND the support's absolute motion -- and
    the response doubled. One or the other."""
    steps = list(case.steps) or [None]
    for k, st in enumerate(steps):
        loads = list(case.loads) + (list(st.loads) if st is not None else [])
        cons = list(case.constraints) + (list(st.constraints) if st is not None else [])
        gms = [np.asarray((getattr(l, "ground_motion", None) or {}).get("direction", [1, 0, 0]), float)
               for l in loads if getattr(l, "ground_motion", None)]
        for c in cons:
            if getattr(c, "record", None) is None or not gms:
                continue
            driven = [j for j, on in enumerate(list(c.dofs)[:3]) if on]
            hit = sorted({"xyz"[j] for d in gms for j in driven if abs(d[j]) > 1e-12})
            if hit:
                _issue(items, "ERROR", "GROUND_MOTION_AND_RECORD",
                       f"the support {c.region!r} follows its own `record` in {', '.join(hit)} and a "
                       f"`ground_motion` shakes the model along {', '.join(hit)} too"
                       + (f" (step {k + 1})" if st is not None else "")
                       + ": both would apply and the response double. Use `record` on every support "
                       "(multiple-support excitation) or one `ground_motion` (uniform), not both",
                       region=c.region, components=hit)
                return


def _fiber_connector_hosts(case, mesh, items):
    """A connector in a fibre mesh has to hang off something.

    `element.fiber_connectors` lets a `beam2_fiber` model carry members that
    are not fibre beams -- bearings. A connector contributes ONLY its six
    declared stiffnesses, and those are routinely zero: a pot bearing releases
    its rotation, a sliding one its longitudinal component. So a node that
    touches connectors and nothing else has a zero row in K wherever its
    connectors release, and the FIFE pseudo-mass of a zero row is zero. The
    run does not fail; it produces a NaN at the first step, or -- worse, and
    this is why it is an ERROR and not a WARNING -- it produces a plausible
    number with one node flying off on its own.

    A fully restrained node is fine: it is the ground, which is exactly how a
    bearing is meant to be modelled when the substructure is not.
    """
    specs = getattr(case.element, "fiber_connectors", None)
    if not specs:
        return
    is_connector = np.zeros(mesh.n_elem, bool)
    for m in specs:
        region = m["region"] if isinstance(m, dict) else getattr(m, "region", None)
        idx = mesh.regions.get(region)
        if idx is None:
            continue
        mark = np.zeros(mesh.n_nodes, bool); mark[idx] = True
        is_connector |= mark[mesh.elements].all(axis=1)
    if not is_connector.any():
        return
    held = np.zeros(mesh.n_nodes, bool)
    other = mesh.elements[~is_connector]
    if len(other):
        held[np.unique(other)] = True
    fixed = _constraint_mask(case, mesh, 6).all(axis=1)
    touched = np.unique(mesh.elements[is_connector])
    lonely = touched[~held[touched] & ~fixed[touched]]
    if len(lonely):
        _issue(items, "ERROR", "CONNECTOR_NODE_UNHELD",
               f"{len(lonely)} node(s) carry only connectors and are not fully "
               "restrained; a released connector component leaves a zero stiffness "
               "row there and the pseudo-mass of a zero row is zero. Attach a "
               "member to the node or constrain all six of its DOFs.",
               count=int(len(lonely)), first=lonely[:8].tolist())


def _fiber_thermal_expansion(case, items, mesh=None):
    """A heated fibre section some of whose laws state no `thermal_expansion`.
    Those fibres expand by zero, so a free member under a
    uniform +30 K puts +52 MPa in bars whose steel block simply forgot the
    coefficient -- a self-equilibrated stress nobody asked for. Refusing it
    would refuse a deliberate choice; saying it costs one line.

    Only a section on members a temperature load reaches (its region shares
    an element with the load's, model-level or in any step): a temperature
    on the deck warned about the piers (found in review)."""
    loads = list(case.loads or [])
    for st in case.steps or []:
        loads += list(getattr(st, "loads", None) or [])
    heated_regions = [getattr(ld, "region", None) or "all" for ld in loads
                      if getattr(ld, "temperature", None) is not None]
    if not heated_regions:
        return

    def members(region):
        """The elements wholly in `region`, or None when that cannot be said."""
        if mesh is None or region is None:
            return None
        conn = np.asarray(mesh.elements)[:, :2]
        if region == "all":
            return np.ones(len(conn), bool)
        idx = mesh.regions.get(region)
        if idx is None:
            return None
        mark = np.zeros(mesh.n_nodes, bool); mark[np.asarray(idx, dtype=int)] = True
        return mark[conn].all(axis=1)

    heated = np.zeros(0, bool)
    for r in heated_regions:
        m = members(r)
        if m is None:
            heated = None
            break
        heated = m if not heated.size else (heated | m)

    def reached(region):
        if heated is None:
            return True                  # cannot tell: say it rather than miss it
        m = members(region if region is not None else "all")
        return True if m is None else bool((m & heated).any())
    el = case.element
    default = dict(case.material or {})

    def laws(sec, cover, rebars, tendons):
        out = [("material", cover)]
        if sec.get("core_material"):
            out.append(("core_material", sec["core_material"]))
        for k, p in enumerate(sec.get("patches") or []):
            if isinstance(p.get("material"), dict):
                out.append((f"patch {k}", p["material"]))
        out += [(f"bar {k}", b.get("material") or {}) for k, b in enumerate(rebars or [])]
        out += [(f"tendon {k}", t.get("material") or {}) for k, t in enumerate(tendons or [])]
        return out

    if getattr(el, "fiber_sections", None):
        sections = [(m.get("region", f"#{i}"), laws(m.get("fiber_section") or {}, m.get("material") or default,
                                                    m.get("fiber_rebars"), m.get("fiber_tendons")))
                    for i, m in enumerate(el.fiber_sections) if reached(m.get("region"))]
    else:
        sections = [("the fibre section", laws(getattr(el, "fiber_section", None) or {}, default,
                                               getattr(el, "fiber_rebars", None),
                                               getattr(el, "fiber_tendons", None)))]
    for name, ls in sections:
        has = [w for w, m in ls if float(m.get("thermal_expansion") or 0.0) > 0.0]
        lacks = [w for w, m in ls if m.get("thermal_expansion") is None]     # an explicit 0 is a choice
        if has and lacks:
            _issue(items, "WARNING", "THERMAL_EXPANSION_PARTIAL",
                   f"{name}: a temperature load acts, and {', '.join(has)} state(s) "
                   f"`thermal_expansion` but {', '.join(lacks)} do(es) not -- those fibres "
                   "expand by zero and the section stresses itself against them. State the "
                   "coefficient on every law (steel ~1.2e-5, concrete ~1.0e-5), or 0 if meant",
                   section=str(name), without=lacks)


def _coupled_rotation_check(case, line, items):
    """Exact linear rotational-rank check with continuum translations supported."""
    from .core.beam_solver import BeamSolver
    from .elements import connector
    owner = np.full(line.n_elem, -1, int)
    specs = list(case.members)
    for g, spec in enumerate(specs):
        if spec.region not in line.regions: continue
        mark = np.zeros(line.n_nodes, bool); mark[line.regions[spec.region]] = True
        owner[mark[line.elements].all(axis=1)] = g
    rows=[]; cols=[]; data=[]; probe=object.__new__(BeamSolver)
    probe.material=SimpleNamespace(E=float(case.material["E"]), nu=float(case.material["nu"]))
    probe.p={"element": {}}
    for e,conn in enumerate(line.elements):
        if owner[e] < 0 or specs[owner[e]].type not in _BEAM_TYPES | {"connector2"}: continue
        props=specs[owner[e]].model_dump(mode="python",exclude_none=True)
        reference=props.get("orientation")
        if props["type"]=="connector2" and reference is None:
            axis=line.nodes[conn[1]]-line.nodes[conn[0]]; axis/=np.linalg.norm(axis)
            basis=np.eye(3); reference=basis[np.argmin(np.abs(basis@axis))]
        R,L=probe._frame(line.nodes[conn[0]],line.nodes[conn[1]],reference)
        if props["type"]=="connector2":
            kg=connector.reference_stiffness(line.nodes[conn],R,props)
        else:
            kl=probe._local_stiffness(L,props); T=repeated_rotation(R); kg=T.T@kl@T
        rd=np.r_[6*conn[0]+np.arange(3,6),6*conn[1]+np.arange(3,6)]
        block=kg[np.ix_([3,4,5,9,10,11],[3,4,5,9,10,11])]
        ii,jj=np.nonzero(block); rows.extend(rd[ii]); cols.extend(rd[jj]); data.extend(block[ii,jj])
    if not data: return
    K=sparse.coo_matrix((data,(rows,cols)),shape=(6*line.n_nodes,6*line.n_nodes)).tocsr()
    rd=(6*np.arange(line.n_nodes)[:,None]+np.arange(3,6)).ravel()
    row=np.asarray(abs(K[rd]).sum(axis=1)).ravel(); active=row>max(row.max(initial=0),1)*1e-13
    fixed=_constraint_mask(case,line,6)[:,3:].reshape(-1); free=active&~fixed
    if np.any(free):
        A=K[rd][:,rd][free][:,free].toarray(); rank=np.linalg.matrix_rank(A,tol=max(np.abs(A).max(),1)*1e-11)
        if rank < len(A):
            _issue(items,"ERROR","BEAM_ROTATION_MECHANISM",
                   f"coupled beam rotational subsystem has {len(A)-rank} mechanism DOF(s); add physical rotations/releases or use truss members for a pinned connection")


_SOLID_TYPES = {"tet4", "tet10", "hexa8", "hexa20", "wedge6", "wedge15", "pyramid5"}
#: the monitor kinds each solver records: anything else was
#: recorded as a mean displacement -- a `fiber` stress as 0.0, an `element`
#: von Mises as the tip deflection
_MONITOR_KINDS = {
    "solid": {"reaction", "displacement", "element", "bar"},
    "truss2": {"reaction", "displacement", "element"},
    "beam2_fiber": {"reaction", "displacement", "fiber", "vehicle", "element"},
    "frame": {"reaction", "displacement", "vehicle"},
}
#: frame models whose solver records member-force monitors (`kind: element`,
#: v1.0); the shells do not
_ELEMENT_MONITOR_FRAMES = {"beam2", "beam2_timoshenko", "mixed_line", "coupled"}


def _scope(case, element, items):
    """Settings that validate but that this element, solver or mode does
    not use. Each was accepted and silently dropped -- the
    analyst believed a setting was active that was not -- until a review
    went looking for them."""
    family = ("solid" if element in _SOLID_TYPES else element if element in ("truss2", "beam2_fiber")
              else "frame")
    kinds = set(_MONITOR_KINDS[family])
    if family == "frame" and element in _ELEMENT_MONITOR_FRAMES:
        kinds.add("element")
    # what a monitor of each kind is, for the `reduce` refusal: the old text
    # called every kind "the mean displacement"
    what = {"displacement": "the mean displacement of its regions' nodes",
            "reaction": "the summed reaction of its regions' nodes",
            "fiber": "one point of one section",
            "vehicle": "one quantity of one vehicle",
            "bar": "a reinforcement set's segments, on solid models only",
            "element": "the members its regions own"}
    for name, m in (case.output.monitors or {}).items():
        if m.kind not in kinds:
            _issue(items, "ERROR", "MONITOR_UNSUPPORTED",
                   f"monitor {name!r}: a {element} model records {', '.join(sorted(kinds))} monitors, "
                   f"not {m.kind!r}", monitor=name, kind=m.kind)
        if m.reduce is not None and m.kind not in ("element", "bar"):
            _issue(items, "ERROR", "MONITOR_REDUCE_UNUSED",
                   f"monitor {name!r}: `reduce` applies to element monitors (and bar monitors on "
                   f"solids); a {m.kind} monitor reads {what.get(m.kind, 'one value')}, with nothing to reduce",
                   monitor=name)
    from .elements.library import takes, taking
    if getattr(case, "diaphragms", None) and (not takes(element, "diaphragms")
                                               or case.solver.backend == "warp"):
        _issue(items, "ERROR", "DIAPHRAGM_UNSUPPORTED",
               f"diaphragms are applied to beam, fibre-beam, shell and coupled models on the numpy/numba backends; "
               f"this is a {element} model on {case.solver.backend}, where they would be ignored")
    loads = list(case.loads) + [l for st in case.steps for l in st.loads]
    if element not in _SOLID_TYPES and any(getattr(l, "follower", False) for l in loads):
        _issue(items, "ERROR", "FOLLOWER_UNSUPPORTED",
               f"`follower: true` is implemented for pressure on solid elements; on a {element} model "
               "the load keeps its reference direction")
    if case.solver.erosion is not None and element not in _SOLID_TYPES:
        _issue(items, "ERROR", "EROSION_UNSUPPORTED",
               f"solver.erosion removes solid elements; a {element} model loses members through "
               "`failure`, section `fracture` or `removals`")
    if case.solver.mode == "static" and not takes(element, "static"):
        _issue(items, "ERROR", "STATIC_UNSUPPORTED",
               f"solver.mode = static runs frame and fibre-beam models ({', '.join(taking('static'))}); this is a {element} "
               "model. Use mode: relaxation")
    # SETTINGS WRITTEN WHERE THEY DO NOTHING (found in review):
    # accepted without a word, as `damping` in a static run once was in
    # every mode. Warnings, not errors: the answer is the
    # same with or without them, but the analyst thought otherwise.
    written = case.solver.model_fields_set
    if case.solver.mode == "static" and "damping" in written:
        _issue(items, "WARNING", "STATIC_IGNORES_DAMPING",
               "solver.damping has no effect under mode: static (Newton has no time and no velocity); "
               "remove it, or use mode: relaxation / dynamic")
    if case.solver.mode != "static" and "static" in written:
        _issue(items, "WARNING", "STATIC_BLOCK_IGNORED",
               f"solver.static (Newton, arc length) is read only under mode: static; this run is "
               f"mode: {case.solver.mode} and ignores it")
    arc = case.solver.static.arc_length if case.solver.mode == "static" else None
    names = [st.name for st in case.steps or []]
    if arc is not None and "line_search" in case.solver.static.model_fields_set and (
            not arc.steps or (names and set(names) <= set(arc.steps))):
        # Only when EVERY step runs by arc length: a load-controlled step
        # uses the line search fully. Even then it is not ignored -- the
        # equilibrium at lambda = 0 and the landing on lambda = 1 are Newton
        # solves -- so the note says where it acts (v1.0: it warned
        # on `line_search: true` beside a load-controlled dead step).
        _issue(items, "INFO", "ARC_NO_LINE_SEARCH",
               "every step here runs by arc length, whose corrector has no line search (its step "
               "length is set by the arc constraint): static.line_search acts only in each step's two "
               "load-controlled solves, the equilibrium at load factor 0 and the last increment "
               "landing on load factor 1")
    if element == "truss2" and case.solver.mode == "relaxation" and (
            case.solver.load_steps or any(st.load_steps for st in case.steps)):
        _issue(items, "ERROR", "LOAD_STEPS_UNSUPPORTED",
               "a truss2 model relaxes each step continuously: `load_steps` would only change the "
               "amplitude, not stage the load (use a mixed_line model, or steps)")
    # amplitudes: every name used exists, every table reads, none runs past 1
    from .core import amplitude as amp
    try:
        amps = amp.build(case.amplitudes)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        _issue(items, "ERROR", "AMPLITUDE_INVALID", f"an amplitude does not read: {exc}")
        amps = None
    if amps is not None:
        used = set()
        users = list(case.loads) + list(case.constraints)
        for st in case.steps:
            users += list(st.loads) + list(st.constraints)
        for u in users:
            name = getattr(u, "amplitude", None)
            if name:
                used.add(name)
                if name not in amps:
                    _issue(items, "ERROR", "AMPLITUDE_UNDEFINED",
                           f"amplitude {name!r} is used but not defined (available: "
                           f"{', '.join(sorted(amps))})", amplitude=name)
        for name in sorted(case.amplitudes or {}):
            if name not in used:
                _issue(items, "WARNING", "AMPLITUDE_UNUSED", f"amplitude {name!r} is defined but not used",
                       amplitude=name)
            a = amps.get(name)
            if a is not None and len(getattr(a, "p", [])) and float(max(a.p)) > 1.0 + 1e-9:
                _issue(items, "WARNING", "AMPLITUDE_PAST_PROGRESS",
                       f"amplitude {name!r} runs to {float(max(a.p)):g}, but its first column is the "
                       "step's PROGRESS, 0 to 1: the part past 1 is never reached (a table written in "
                       "seconds?)", amplitude=name)
    if len(case.steps) > 1 and amps is not None:
        for l in case.loads:
            a = amps.get(l.amplitude) if l.amplitude else None
            # only an amplitude that does not end where it starts reads as a
            # repeat: a constant one (a dead load held from 0) is the same
            # load in every step
            if a is not None and abs(a(0.0) - a(1.0)) > 1e-12:
                _issue(items, "WARNING", "AMPLITUDE_REPEATS",
                       f"a model-level load on {l.region!r} has amplitude {l.amplitude!r}: it is applied "
                       "again from the amplitude's start in every step, not held after the first. "
                       "Leave `amplitude` out to have it held, or put the load in the first step")
                break
    if len(case.steps) > 1 and case.solver.mode == "dynamic":
        for l in case.loads:
            if l.ground_motion is not None:
                # every step's clock starts at 0, and so does the record: a
                # pulse then a "free decay" step shook the model twice
                _issue(items, "WARNING", "GROUND_MOTION_REPEATS",
                       "a model-level ground_motion is replayed from its start in every step; put it "
                       "in the step it belongs to")
                break
    # a removal the run never reaches: accepted, and nothing was removed.
    # Each dynamic step lasts solver.total_time.
    if case.solver.mode == "dynamic" and case.solver.total_time and getattr(case, "removals", None):
        end = float(case.solver.total_time) * max(len(case.steps), 1)
        for r in case.removals:
            # AFTER the end, strictly: one at the end itself is reached by the
            # last frame's force evaluation and is logged (v1.0 --
            # it was warned of as never happening, and then happened)
            if float(r.time) > end * (1.0 + 1e-9):
                _issue(items, "WARNING", "REMOVAL_AFTER_END",
                       f"the removal of {r.region!r} is at t = {float(r.time):g} s, after the run ends "
                       f"at {end:g} s ({max(len(case.steps), 1)} step(s) of solver.total_time): it never "
                       "happens", region=r.region, time=float(r.time), end=end)


def _other_materials(case, element, items):
    """The materials beyond the model's own: line member groups, embedded
    reinforcement and every fibre law. Only the first was checked, so the
    verdict depended on which assignment came first -- a steel02 cable
    after a beam passed, the same cable first was refused -- and the rest
    surfaced as a solver error after PASS (found in review)."""
    from .catalog import material_definition

    def kin(t):
        try:
            return material_definition(t).kinematics
        except (KeyError, ValueError):
            return None                   # an unknown law is reported elsewhere

    if element in ("mixed_line", "coupled", "shell_line", "connector2"):
        for m in case.members:
            t = (m.material or {}).get("type")
            if t and t != "linear_elastic":
                _issue(items, "ERROR", "CAPABILITY_MISMATCH",
                       f"member group {m.region!r} names material {t!r}; line members support "
                       "linear_elastic only (nonlinear members: a beam2_fiber model)",
                       element=element, material=t)
    for r in case.reinforcement:
        t = (r.material or {}).get("type")
        k = kin(t) if t else None
        if k is not None and "uniaxial" not in k:
            _issue(items, "ERROR", "CAPABILITY_MISMATCH",
                   f"reinforcement {getattr(r, 'name', None) or ''!s}: material {t!r} is a "
                   "three-dimensional law; bars need a one-dimensional one (rebar, steel02, ...)",
                   material=t)
    if element == "beam2_fiber":
        el = case.element
        entries = ([{"fiber_section": el.fiber_section, "fiber_rebars": el.fiber_rebars,
                     "fiber_tendons": el.fiber_tendons}] if el.fiber_section else []) + list(el.fiber_sections or [])
        for e in entries:
            sec = e.get("fiber_section") or {}
            laws = [e.get("material"), sec.get("core_material")]
            laws += [q.get("material") for q in sec.get("patches") or []]
            laws += [b.get("material") for b in (e.get("fiber_rebars") or [])]
            laws += [t.get("material") for t in (e.get("fiber_tendons") or [])]
            for law in laws:
                t = (law or {}).get("type") if isinstance(law, dict) else None
                if not t:
                    continue
                # the backend too: the numba fibre kernel
                # takes only laws with a uniaxial scalar twin
                ok, reasons = compatibility(element, t, backend=case.solver.backend, mode=None)
                if not ok:
                    _issue(items, "ERROR", "CAPABILITY_MISMATCH",
                           f"a fibre of {e.get('region') or 'the section'!s}: " + "; ".join(reasons),
                           element=element, material=t)


def check(case, mesh, element: str) -> dict[str, Any]:
    items = []
    _scope(case, element, items)
    # old spellings: read, but said once, with the new name
    # what a tendon block followed only roughly (vfife/prep)
    for who, text in case._expansion_notes:
        _issue(items, "WARNING", "TENDON_APPROXIMATE", f"{who}: {text}", where=who)
    for where, old, new in getattr(case, "_deprecated", None) or []:
        _issue(items, "WARNING", "DEPRECATED_FIELD",
               f"{where}: `{old}` is the old spelling of `{new}`; it is still read, and "
               f"`vfife migrate <case> --in-place` rewrites it",
               where=where, old=old, new=new)
    # unit-carrying defaults written in the declared units
    for where, value, dim in getattr(case, "_unit_defaults", None) or []:
        if value is None:
            _issue(items, "WARNING", "UNIT_DEFAULT_SI",
                   f"{where} is left out and its default is SI; the declared `units` name a unit "
                   f"this cannot convert a {dim} to, so write {where.rsplit('.', 1)[-1]} "
                   "in the case's units", where=where)
        else:
            _issue(items, "INFO", "UNIT_DEFAULT_CONVERTED",
                   f"{where} is left out: its default is written in the declared units as {value:.6g}",
                   where=where, value=value)
    # every material the mesh carries: the whole-mesh one, or each block's
    for label, mat in [("", case.material)] + [(b.name, b.material) for b in case.blocks]:
        material = mat.get("type")
        if not material:
            continue
        ok, reasons = compatibility(
            element, material, backend=case.solver.backend,
            # the implicit static solver runs what relaxation runs
            mode="relaxation" if case.solver.mode == "static" else case.solver.mode,
        )
        if not ok:
            _issue(items, "ERROR", "CAPABILITY_MISMATCH",
                   (f"section {label!r}: " if label else "") + "; ".join(reasons),
                   element=element, material=material,
                   backend=case.solver.backend, mode=case.solver.mode)
    _other_materials(case, element, items)
    # The element declares which formulations it supports; asking for another
    # is an error, and it must be reported here rather than surfacing as a
    # traceback from deep in precompute after preflight has said PASS.
    # A mixed mesh has one formulation setting and several element types, so
    # the check is against EVERY block: `b-bar` is legal for the hexahedra and
    # not for the tetrahedra, and the run would have failed inside precompute
    # for the second block after preflight said PASS for the first.
    from .elements.library import ELEMENTS
    _form = getattr(case.element, "formulation", None)
    _blocks = [k for k in (getattr(mesh, "cell_blocks", None) or {})
               if k in ELEMENTS] or [element]
    for _name in _blocks:
        _elem = ELEMENTS.get(_name)
        if _elem is None or _form is None or _form in _elem.formulations:
            continue
        _shared = [f for f in _elem.formulations
                   if all(f in ELEMENTS[b].formulations for b in _blocks)]
        _issue(items, "ERROR", "FORMULATION_UNSUPPORTED",
               f"element {_name!r} does not support formulation {_form!r}; "
               f"it supports {', '.join(_elem.formulations)}"
               + (f". {_elem.caveat}" if _elem.caveat else ""),
               element=_name, formulation=_form,
               hint=(f"set element.formulation to {_shared[0]!r}" if _shared
                     else f"set element.formulation to {_elem.formulations[0]!r}"))

    if element == "shell_line":
        from .core.mesh import Mesh
        if set(mesh.cell_blocks)!={"shell4","truss2"}:
            _issue(items,"ERROR","SHELL_LINE_TOPOLOGY","shell_line requires Shell4 and selected Line2 blocks")
        else:
            surface=Mesh(mesh.nodes,mesh.cell_blocks["shell4"],dict(mesh.regions))
            line=Mesh(mesh.nodes,mesh.cell_blocks["truss2"],dict(mesh.regions),
                      element_regions=dict(mesh.element_regions))
            # NEITHER BLOCK OWNS EVERY NODE in a mixed-dimensional model, so
            # the orphan check belongs to the union and is dropped from both
            # halves rather than reported twice from two half-views.
            surface_items=[]; _mesh_checks(surface,"shell4",surface_items)
            items.extend(x for x in surface_items if x["code"]!="ORPHAN_NODE")
            line_items=[]; _mesh_checks(line,"mixed_line",line_items); items.extend(x for x in line_items if x["code"]!="ORPHAN_NODE")
            used=np.zeros(mesh.n_nodes,dtype=bool)
            for blk in mesh.cell_blocks.values(): used[np.unique(blk)]=True
            if (~used).any():
                _issue(items,"ERROR","ORPHAN_NODE","mesh contains nodes unused by every element",
                       count=int((~used).sum()),first=np.flatnonzero(~used)[:8].tolist())
            _regions(case,surface,"shell4",items); mesh.regions=surface.regions; line.regions=dict(surface.regions)
            # THE WHOLE MODEL, not the shell block: the walls of a building
            # are separate quad patches held together by its frame, so asking
            # the shells alone which rigid modes are restrained reports every
            # panel as its own free body (263 of them on the first real
            # building this saw -- v0.1).
            _rigid_modes(case,surface,"shell4",items,
                         blocks=list(mesh.cell_blocks.values()))
            _diaphragms(case,surface,items)
            _line_assignments(case,line,"mixed_line",items)
            _mass_checks(case,line,"mixed_line",items)
        counts={s:sum(i["severity"]==s for i in items) for s in ("ERROR","WARNING","INFO")}
        return {"schema_version":1,"case":case.name,"element":element,
                "mesh":{"nodes":mesh.n_nodes,"elements":sum(len(v) for v in mesh.cell_blocks.values())},
                "status":"FAIL" if counts["ERROR"] else "PASS","counts":counts,"issues":items}
    if element == "coupled":
        from .core.mesh import Mesh
        solids = [k for k in mesh.cell_blocks if k != "truss2"]
        if len(solids) != 1 or "truss2" not in mesh.cell_blocks:
            _issue(items, "ERROR", "COUPLED_TOPOLOGY", "coupled model requires one solid topology and Line2 members")
        else:
            solid_type = solids[0]
            solid = Mesh(mesh.nodes, mesh.cell_blocks[solid_type], dict(mesh.regions))
            line = Mesh(mesh.nodes, mesh.cell_blocks["truss2"], dict(mesh.regions),
                        element_regions=dict(mesh.element_regions))
            _mesh_checks(solid, solid_type, items)
            line_items = []; _mesh_checks(line, "mixed_line", line_items)
            items.extend(x for x in line_items if x["code"] != "ORPHAN_NODE")
            _regions(case, solid, solid_type, items); mesh.regions = solid.regions
            _rigid_modes(case, solid, solid_type, items)
            _diaphragms(case, solid, items)
            _line_assignments(case, line, "mixed_line", items)
            solid_nodes = np.zeros(mesh.n_nodes, bool); solid_nodes[np.unique(solid.elements)] = True
            for component in _components(line):
                used = component[np.isin(component, np.unique(line.elements))]
                if len(used) and not solid_nodes[used].any():
                    _issue(items, "ERROR", "UNCONNECTED_MEMBER", "a Line2 member network shares no node with the solid",
                           first_nodes=used[:8].tolist())
            _mass_checks(case, line, "mixed_line", items)
            _coupled_rotation_check(case, line, items)
            def transfers_moment(g):
                if g.type in _BEAM_TYPES: return True
                if g.type != "connector2": return False
                K=np.asarray(g.stiffness if g.stiffness is not None else np.zeros(6),float)
                C=np.asarray(g.connector_damping if g.connector_damping is not None else np.zeros(6),float)
                rotational=(np.diag(K)[3:] if K.ndim==2 else K[3:])
                rotational_c=(np.diag(C)[3:] if C.ndim==2 else C[3:])
                return bool(np.any(rotational) or np.any(rotational_c)
                            or set(g.curves).intersection({"rx","ry","rz"}))
            beam_regions = [g.region for g in case.members if transfers_moment(g) and g.region in mesh.regions]
            beam_nodes = np.unique(np.concatenate([mesh.regions[r] for r in beam_regions])) if beam_regions else np.empty(0, int)
            shared_beam = beam_nodes[solid_nodes[beam_nodes]] if len(beam_nodes) else beam_nodes
            if len(shared_beam):
                _issue(items, "WARNING", "BEAM_SOLID_MOMENT_TRANSFER",
                       "a shared node transfers beam forces but a point solid node has no rotational DOF; restrain the beam rotation or use a future MPC/rigid connection when moment transfer is required",
                       count=len(shared_beam), first=shared_beam[:8].tolist())
        counts = {s: sum(i["severity"] == s for i in items) for s in ("ERROR", "WARNING", "INFO")}
        return {"schema_version": 1, "case": case.name, "element": element,
                "mesh": {"nodes": mesh.n_nodes, "elements": sum(len(v) for v in mesh.cell_blocks.values())},
                "status": "FAIL" if counts["ERROR"] else "PASS", "counts": counts, "issues": items}
    from .core.mesh import Mesh as _Mesh
    for _name in _blocks:
        if _name == element:
            _mesh_checks(mesh, element, items, _linked_pairs(case, mesh))
        else:
            # each block gets its own Jacobian / degeneracy check; ORPHAN_NODE
            # is dropped because a node is only an orphan of the WHOLE model,
            # and every block legitimately leaves most of the mesh untouched
            _sub = []
            _mesh_checks(_Mesh(mesh.nodes, mesh.cell_blocks[_name],
                               dict(mesh.regions)), _name, _sub)
            for it in _sub:
                if it["code"] != "ORPHAN_NODE":
                    it["element"] = _name
                    items.append(it)
    _import_notes(mesh, items)
    _units_check(case, items)
    _regions(case, mesh, element, items)
    if element in ("shell3", "shell4", "shell_line"):
        _shell_normals(case, mesh, items)
    if case.blocks:
        _block_regions(case, mesh, element, items)
    _crack_band_checks(case, mesh, element, items)
    _prestress_check(case, items)
    _mixed_precision_check(case, mesh, items)
    _split_route_check(case, items)
    _rigid_modes(case, mesh, element, items)
    if element in _SIX_DOF_TYPES:
        _diaphragms(case, mesh, items)
    _line_assignments(case, mesh, element, items)
    _mass_checks(case, mesh, element, items)
    _connector_rank(case, mesh, element, items)
    _ground_motion_direction(case, mesh, element, items)
    _record_and_ground_motion(case, items)
    if element == "beam2_fiber":
        _fiber_connector_hosts(case, mesh, items)
        _fiber_thermal_expansion(case, items, mesh)
    counts = {s: sum(i["severity"] == s for i in items) for s in ("ERROR", "WARNING", "INFO")}
    return {"schema_version": 1, "case": case.name, "element": element,
            "mesh": {"nodes": mesh.n_nodes, "elements": mesh.n_elem},
            "status": "FAIL" if counts["ERROR"] else "PASS", "counts": counts, "issues": items}


def print_summary(report):
    c = report["counts"]
    print(f"  preflight     {report['status']} ({c['ERROR']} errors, {c['WARNING']} warnings)")
    for issue in report["issues"]:
        print(f"    [{issue['severity']}] {issue['code']}: {issue['message']}")


def write_json(report, path):
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def require_pass(report):
    if report["status"] != "PASS":
        raise PreflightError(f"model preflight failed with {report['counts']['ERROR']} error(s)")
