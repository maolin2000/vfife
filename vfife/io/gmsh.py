"""Gmsh `.msh` import: geometry, node ordering, and physical groups as regions.

WHY THIS IS NOT `meshio`
------------------------
`meshio` reads these files perfectly well, and for a one-off conversion it is
the right tool.  This reader exists for two things it would not give us:

* **Node ordering is not a detail we can delegate.**  Gmsh's 10-node
  tetrahedron numbers its mid-edge nodes (0,1) (1,2) (0,2) (0,3) (2,3) (1,3);
  the VTK/Abaqus convention `elements/library.py` is built on uses (0,1) (1,2)
  (2,0) (0,3) (1,3) (2,3).  The last two are swapped.  On the straight-sided
  tets Gmsh produces, omitting that swap happens to be caught downstream --
  `precompute` reports a non-positive Jacobian -- but the message it gives is
  about determinants, which is true and useless when the fix is a permutation,
  and a curved or nearly degenerate element need not trip it at all.
  `verify_ordering()` below checks the mapping from where the nodes actually
  are.  It permits the modest chord deviation of a genuinely curved
  second-order CAD edge, but still rejects a mid-side node assigned to a
  different tetrahedron edge.

* **Orphan nodes and orientation.**  A Gmsh file carries nodes that belong to
  curves and points as well as volumes, and its tags are sparse.  A node that no
  volume element references would arrive with zero mass and no stiffness -- a
  free particle in the middle of the model.  And a mesh with inverted elements
  is rejected downstream by `precompute` with a message about Jacobians, which
  is true but unhelpful when the fix is a node swap at import.

FORMATS
-------
MSH 4.1 ASCII and binary, and MSH 2.2 ASCII.  4.1 is what current Gmsh writes;
2.2 is still what a great deal of existing tooling emits.  4.1 binary matters
because this project's target size is 10^6 elements, where the ASCII file is
several hundred megabytes.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np

from ..core.diagnostics import UserError
from ..core.mesh import Mesh
from ..elements.library import (gmsh_types as _lib_gmsh_types,
                                families as _lib_families)

#: Gmsh element type -> (our element name, permutation from Gmsh to our order).
#: The permutation is `ours[i] = gmsh[PERM[i]]`.
# Gmsh type ids and node permutations are declared on each element in
# elements/library.py; splitting them by dimension here keeps the reader's
# "is this a volume mesh" test working.
_ALL_GMSH = _lib_gmsh_types()
_VOLUME = {t: v for t, v in _ALL_GMSH.items()
           if _lib_families()[v[0]] == "continuum"}


# A two-node line is a standalone structural element, not a lower-dimensional
# boundary of a solid.  It deliberately shares the same importer so Gmsh
# Physical Point/Curve groups remain the user-facing support/load regions.
_STRUCTURAL = {t: v for t, v in _ALL_GMSH.items()
               if _lib_families()[v[0]] != "continuum"}

#: lower-dimensional types, read only so that a physical group defined on a
#: surface can be turned into a node set
_OTHER_NODE_COUNT = {1: 2, 2: 3, 3: 4, 8: 3, 9: 6, 15: 1, 16: 8}

_UNSUPPORTED = {
    19: "13-node pyramid", 12: "27-node hexahedron",
}

#: mid-edge slots of a tet10, in OUR order -- used by verify_ordering
_TET10_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


# --------------------------------------------------------------------------- #
#  public
# --------------------------------------------------------------------------- #


def read(path: str, scale: float = 1.0, verify: bool = True, overlay=None) -> Tuple[Mesh, str]:
    """Read a `.msh` file. Returns (mesh, element name).

    `scale` multiplies every coordinate, because meshing a structure in
    millimetres and analysing it in metres is the normal case, not the exception.

    Physical groups arrive as named entries in `mesh.regions`, already resolved
    to node index arrays -- so a case file can say `region: left_face` and never
    write a bounding box.
    """
    raw = open(path, "rb").read()
    version, binary = _format(raw)
    if version >= 4.0:
        names, ent_phys, nodes, tags, elems = (
            _parse_41_binary(raw) if binary else _parse_41_ascii(raw)
        )
    else:
        if binary:
            raise NotImplementedError(
                "MSH 2.2 binary is not supported. Re-export as 4.1 (binary is "
                "fine) or as 2.2 ASCII -- in Gmsh, Mesh.MshFileVersion = 4.1."
            )
        names, ent_phys, nodes, tags, elems = _parse_22_ascii(raw)

    if scale != 1.0:
        nodes = nodes * float(scale)
    if overlay is not None:
        # what a case's high-level blocks add (vfife/prep):
        # nodes after the file's, and Line2 groups after its own
        names, ent_phys, nodes, tags, elems = apply_overlay(names, ent_phys, nodes, tags, elems, overlay)

    blocks, groups, import_notes, element_regions = _assemble(names, ent_phys, tags, elems)
    used_tags = _used_tags(tags, blocks)
    nodes, blocks, groups = _prune(nodes, blocks, groups)
    blocks = {k: _fix_orientation(nodes, v, k) for k, v in blocks.items()}
    blocks = {k: np.ascontiguousarray(v, dtype=np.int32)
              for k, v in blocks.items()}

    # The returned name stays a REAL element name -- the largest block's --
    # rather than something like "hexa8+tet4".  Every caller that asks the mesh
    # what element it holds (the schema Literal, the scaffold, the preflight
    # element checks, `vfife mesh`) can then keep treating it as one, and
    # `mesh.cell_blocks` is the single authority on what is actually in the
    # model.  The solver builds its blocks from cell_blocks, not from this name.
    element = max(blocks, key=lambda k: len(blocks[k]))
    mesh = Mesh(nodes=np.ascontiguousarray(nodes),
                elements=blocks[element],
                cell_blocks=blocks, element_regions=element_regions)
    mesh.regions.update(groups)
    # each row's tag in the file, and the file's largest: what an overlay
    # names an existing node by (`apply_overlay`)
    mesh.node_tags = used_tags
    mesh.max_node_tag = int(tags.max()) if tags.size else 0
    # carried to the preflight rather than printed here: the reader is called
    # from the GUI and the API too, and each decides how to show it
    mesh.import_notes = import_notes
    if verify and "tet10" in blocks:
        verify_ordering(mesh.nodes, blocks["tet10"])
    return mesh, element


def write_expanded(source: str, out: str, scale: float = 1.0, overlay=None) -> None:
    """`source` with `overlay` added, written as MSH 2.2 ASCII at `out`, the
    coordinates scaled and exact (`vfife expand`). Read back
    it is the mesh `read(source, scale, overlay=overlay)` gives -- checked
    here, since MSH 2.2 has one physical group per element and groups its
    elements by group, which a 4.1 file need not have done."""
    raw = open(source, "rb").read()
    version, binary = _format(raw)
    if version >= 4.0:
        names, ent_phys, nodes, tags, elems = (_parse_41_binary(raw) if binary else _parse_41_ascii(raw))
    else:
        if binary:
            raise NotImplementedError("MSH 2.2 binary is not supported")
        names, ent_phys, nodes, tags, elems = _parse_22_ascii(raw)
    nodes = nodes * float(scale) if scale != 1.0 else nodes
    if overlay is not None:
        names, ent_phys, nodes, tags, elems = apply_overlay(names, ent_phys, nodes, tags, elems, overlay)
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    for (dim, phys), name in sorted(names.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if '"' in name:
            raise ValueError(f"a group name with a quote cannot be written to MSH: {name!r}")
        lines.append(f'{dim} {phys} "{name}"')
    lines += ["$EndPhysicalNames", "$Nodes", str(len(tags))]
    lines += [f"{int(t)} {x!r} {y!r} {z!r}" for t, (x, y, z) in zip(tags.tolist(), nodes.tolist())]
    rows = []
    for dim, ent, etype, conn in elems:
        ps = ent_phys.get((dim, ent), [])
        if len(ps) > 1:
            raise ValueError("an entity in two physical groups cannot be written as MSH 2.2")
        p = ps[0] if ps else 0
        for r in conn.tolist():
            rows.append(f"{etype} 2 {p} {p} " + " ".join(str(int(v)) for v in r))
    lines += ["$EndNodes", "$Elements", str(len(rows))]
    lines += [f"{i + 1} {r}" for i, r in enumerate(rows)]
    lines += ["$EndElements", ""]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    want, _ = read(source, scale=scale, overlay=overlay, verify=False)
    got, _ = read(out, verify=False)
    if not same_mesh(want, got):
        raise ValueError(f"{out} would not read back as the expanded mesh (its groups are not one "
                         "physical group per entity, in order); write the source mesh as MSH 2.2 first")


def join_groups(mesh: Mesh, groups) -> None:
    """`mesh.groups` of a Gmsh mesh: each a group of the
    file's physical groups, its elements and its nodes theirs -- what the
    file would have held as one physical group of several pieces."""
    taken = set(mesh.regions) | set(mesh.element_regions or {}) | {"all"}
    for name, parts in groups.items():
        if name in taken:
            raise ValueError(f"group {name!r} has the name of a group of the mesh (or `all`)")
        missing = [p for p in parts if p not in mesh.regions and p not in (mesh.element_regions or {})]
        if missing:
            raise ValueError(f"group {name!r} lists {missing[0]!r}, which is not a group of the mesh")
        nodes = [np.asarray(mesh.regions[p]) for p in parts if p in mesh.regions]
        mesh.regions[name] = np.unique(np.concatenate(nodes)) if nodes else np.zeros(0, np.int64)
        per_block: dict = {}
        for p in parts:
            for b, rows in (mesh.element_regions or {}).get(p, {}).items():
                per_block.setdefault(b, []).append(np.asarray(rows))
        if per_block:
            mesh.element_regions[name] = {b: np.unique(np.concatenate(v)) for b, v in per_block.items()}
        taken.add(name)


def same_mesh(a: Mesh, b: Mesh) -> bool:
    """Every array of two meshes equal, bit for bit."""
    if not np.array_equal(a.nodes, b.nodes) or set(a.cell_blocks) != set(b.cell_blocks):
        return False
    if any(not np.array_equal(a.cell_blocks[k], b.cell_blocks[k]) for k in a.cell_blocks):
        return False
    if set(a.regions) != set(b.regions) or any(not np.array_equal(a.regions[k], b.regions[k]) for k in a.regions):
        return False
    ea, eb = a.element_regions or {}, b.element_regions or {}
    if set(ea) != set(eb):
        return False
    for k in ea:
        if set(ea[k]) != set(eb[k]) or any(not np.array_equal(ea[k][q], eb[k][q]) for q in ea[k]):
            return False
    return True


def read_coupled(path: str, scale: float = 1.0, verify: bool = True,
                 line_regions=None) -> Tuple[Mesh, str]:
    """Read one solid topology plus Line2 members with shared node numbering."""
    raw = open(path, "rb").read(); version, binary = _format(raw)
    if version >= 4.0:
        names, ent_phys, nodes, tags, elems = (_parse_41_binary(raw) if binary else _parse_41_ascii(raw))
    else:
        if binary:
            raise NotImplementedError("MSH 2.2 binary is not supported")
        names, ent_phys, nodes, tags, elems = _parse_22_ascii(raw)
    nodes = nodes * float(scale)
    lut = _tag_lookup(tags); blocks, group_nodes = {}, {}
    # region name -> block name -> element indices within that block. A
    # building's beams and columns share every node, so which ELEMENTS a
    # physical group holds is the only thing that separates them
    # (core/mesh.py::element_regions).
    group_elems: dict = {}
    counts: dict = {}
    for dim, ent, etype, rows in elems:
        mapped = lut(rows) if rows.size else rows
        phys_names={names.get((dim,p),f"physical_{p}") for p in ent_phys.get((dim,ent),[])}
        keep_line=(etype!=1 or line_regions is None or bool(phys_names.intersection(line_regions)))
        kept = None
        if (etype in _VOLUME or etype==3 or etype==1) and keep_line:
            name, perm = (_VOLUME | _STRUCTURAL)[etype]
            blocks.setdefault(name, []).append(mapped[:, perm])
            first = counts.get(name, 0); counts[name] = first + len(mapped)
            kept = (name, np.arange(first, first + len(mapped), dtype=np.int64))
        for phys in ent_phys.get((dim, ent), []):
            label = names.get((dim, phys), f"physical_{phys}")
            group_nodes.setdefault(label, []).append(mapped.ravel())
            if kept is not None:
                group_elems.setdefault(label, {}).setdefault(kept[0], []).append(kept[1])
    volume_names={v[0] for v in _VOLUME.values()}
    if any(k in volume_names for k in blocks):
        blocks.pop("shell4",None)  # Quad4 faces are boundaries of the volume.
    primaries = [k for k in blocks if k in volume_names | {"shell4"}]
    if len(primaries) != 1 or "truss2" not in blocks:
        raise ValueError("coupled import requires exactly one supported solid/shell topology and selected Line2 members")
    blocks = {k: np.vstack(v) for k, v in blocks.items()}
    used = np.unique(np.concatenate([v.ravel() for v in blocks.values()]))
    remap = np.full(len(nodes), -1, dtype=np.int64); remap[used] = np.arange(len(used))
    groups = {k: np.unique(remap[np.concatenate(v)]) for k, v in group_nodes.items()}
    groups = {k: v[v >= 0] for k, v in groups.items() if np.any(v >= 0)}
    nodes = nodes[used]; blocks = {k: remap[v].astype(np.int32) for k, v in blocks.items()}
    solid = primaries[0]; blocks[solid] = _fix_orientation(nodes, blocks[solid], solid)
    element_regions = {name: {b: np.concatenate(v) for b, v in per_block.items()}
                       for name, per_block in group_elems.items()}
    mesh = Mesh(nodes=np.ascontiguousarray(nodes), elements=blocks[solid], regions=groups,
                cell_blocks=blocks, element_regions=element_regions)
    if verify and solid == "tet10": verify_ordering(nodes, blocks[solid])
    return mesh, ("shell_line" if solid=="shell4" else "coupled")


def survey(path: str) -> dict:
    """What element types the file contains, without requiring only one.

    `read()` refuses a mixed mesh, which is right -- but it left `vfife mesh`,
    whose whole job is to say what is in a file, unable to answer for the one
    file you most need answered.
    """
    raw = open(path, "rb").read()
    version, binary = _format(raw)
    if version >= 4.0:
        names, _ent, nodes, _tags, elems = (
            _parse_41_binary(raw) if binary else _parse_41_ascii(raw))
    else:
        names, _ent, nodes, _tags, elems = _parse_22_ascii(raw)
    known = _VOLUME | _STRUCTURAL
    census: dict = {}
    for _dim, _ent2, etype, rows in elems:
        if etype in known:
            census[known[etype][0]] = census.get(known[etype][0], 0) + rows.shape[0]
        elif etype in _UNSUPPORTED:
            census[_UNSUPPORTED[etype]] = (census.get(_UNSUPPORTED[etype], 0)
                                           + rows.shape[0])
    return {"counts": census, "n_nodes": int(nodes.shape[0]),
            "physical_names": sorted(set(names.values()))}


def verify_ordering(nodes: np.ndarray, conn: np.ndarray,
                    curved_edge_limit: float = 0.25):
    """Check that every tet10 mid-edge node really is on the edge it claims.

    This is what makes the permutation table above trustworthy rather than
    remembered.  A straight-sided tet has zero error.  A second-order mesh on a
    curved CAD boundary deliberately puts its mid-side nodes off the corner-node
    chord, so exact midpoint equality would reject valid meshes.  The default
    permits a 25% chord deviation (already a very coarse curved edge), while a
    node on a different tet edge is typically O(1) edge lengths away.
    """
    x = nodes[conn]
    worst, worst_slot = 0.0, -1
    scale = np.linalg.norm(x[:, 1] - x[:, 0], axis=1).mean()
    for k, (i, j) in enumerate(_TET10_EDGES):
        err = np.linalg.norm(x[:, 4 + k] - 0.5 * (x[:, i] + x[:, j]), axis=1).max()
        if err > worst:
            worst, worst_slot = err, 4 + k
    if worst > curved_edge_limit * scale:
        raise ValueError(
            f"tet10 node ordering is wrong: node {worst_slot} is {worst:.3e} "
            f"away from the midpoint of edge {_TET10_EDGES[worst_slot - 4]} "
            f"(mean edge length {scale:.3e}, curved-edge limit "
            f"{curved_edge_limit:.2f}). The Gmsh-to-VTK permutation in "
            "io/gmsh.py does not match this file's convention."
        )
    return worst


# --------------------------------------------------------------------------- #
#  format dispatch
# --------------------------------------------------------------------------- #


def _format(raw: bytes) -> Tuple[float, bool]:
    m = re.search(rb"\$MeshFormat\s*\n\s*([\d.]+)\s+(\d+)\s+(\d+)", raw)
    if not m:
        raise ValueError("not a Gmsh .msh file: no $MeshFormat block")
    return float(m.group(1)), int(m.group(2)) == 1


def _section(text: str, name: str) -> str:
    # User-generated MSH 2.2 files commonly use Windows CRLF.  Matching a
    # literal LF immediately after `$Nodes` misses those sections and later
    # reports an unrelated ``int('')`` parsing error instead of the useful
    # mesh diagnostic from _assemble().
    m = re.search(rf"\${name}\r?\n(.*?)\r?\n\$End{name}", text, re.S)
    return m.group(1) if m else ""


def _physical_names(text: str) -> Dict[Tuple[int, int], str]:
    """{(dimension, physical tag): name}.

    BY DIMENSION AND TAG: Gmsh numbers physical groups per dimension, and
    `Physical Volume(1)` beside `Physical Surface(1)` is legal and common in
    older .geo files. Keyed by the tag alone, the two became one group: the
    volume vanished and the surface took its elements."""
    body = _section(text, "PhysicalNames")
    out = {}
    if not body:
        return out
    for line in body.strip().split("\n")[1:]:
        m = re.match(r'\s*(\d+)\s+(\d+)\s+"(.*)"', line)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = m.group(3)
    return out


#: the topological dimension of each Gmsh element type this reader meets
_TYPE_DIM = {15: 0, 1: 1, 8: 1, 2: 2, 3: 2, 9: 2, 16: 2, 10: 2}


def _type_dim(etype: int) -> int:
    return 3 if etype in _VOLUME else _TYPE_DIM.get(etype, 2)


# --------------------------------------------------------------------------- #
#  MSH 4.1
# --------------------------------------------------------------------------- #


def _parse_41_ascii(raw: bytes):
    text = raw.decode("utf-8", "replace")
    names = _physical_names(text)
    ent_phys = _entities_41_ascii(_section(text, "Entities"))

    body = _section(text, "Nodes").strip().split("\n")
    n_blocks, n_nodes = (int(v) for v in body[0].split()[:2])
    tags = np.empty(n_nodes, dtype=np.int64)
    coords = np.empty((n_nodes, 3))
    i, at = 1, 0
    for _ in range(n_blocks):
        _dim, _tag, _par, nb = (int(v) for v in body[i].split())
        i += 1
        if _par:
            raise UserError("MESH_PARAMETRIC_NODES",
                            "the mesh file stores parametric node coordinates, which this "
                            "reader does not take",
                            "export again with Mesh.SaveParametric = 0 (the default)")
        # whole block at a time: a per-node Python loop here is the difference
        # between reading a 10^6-element mesh in a second and in half a minute
        tags[at:at + nb] = _numbers(body[i:i + nb], np.int64)
        i += nb
        coords[at:at + nb] = _numbers(body[i:i + nb]).reshape(nb, 3)
        i += nb
        at += nb

    body = _section(text, "Elements").strip().split("\n")
    n_blocks = int(body[0].split()[0])
    elems = []
    i = 1
    for _ in range(n_blocks):
        dim, ent, etype, nb = (int(v) for v in body[i].split())
        i += 1
        nn = _n_nodes(etype)
        rows = _numbers(body[i:i + nb], np.int64).reshape(nb, 1 + nn)
        i += nb
        elems.append((dim, ent, etype, rows[:, 1:]))
    return names, ent_phys, coords, tags, elems


def _numbers(lines, dtype=np.float64):
    """Parse a block of whitespace-separated numbers in one go."""
    return np.fromstring(" ".join(lines), sep=" ", dtype=np.float64).astype(
        dtype, copy=False)


def _entities_41_ascii(body: str) -> Dict[Tuple[int, int], list]:
    """{(dim, entity tag): [physical tags]} -- the 4.1 indirection.

    In 4.1 a physical group is attached to a geometric ENTITY, and elements name
    the entity, not the group.  2.2 put the physical tag on the element itself.
    """
    out = {}
    if not body:
        return out
    lines = body.strip().split("\n")
    counts = [int(v) for v in lines[0].split()]
    i = 1
    for dim, n in enumerate(counts):
        for _ in range(n):
            p = lines[i].split()
            i += 1
            # points carry 3 coordinates, everything else a 6-value bounding box
            off = 1 + (3 if dim == 0 else 6)
            n_phys = int(p[off])
            out[(dim, int(float(p[0])))] = [int(v) for v in
                                            p[off + 1:off + 1 + n_phys]]
    return out


def _parse_41_binary(raw: bytes):
    text_head = raw[:raw.index(b"$EndMeshFormat")].decode("ascii", "replace")
    del text_head
    names = _physical_names(_ascii_sections(raw))
    pos = raw.index(b"$EndMeshFormat") + len(b"$EndMeshFormat")

    ent_phys, pos = _read_entities_41_binary(raw, pos)
    coords, tags, pos = _read_nodes_41_binary(raw, pos)
    elems = _read_elements_41_binary(raw, pos)
    return names, ent_phys, coords, tags, elems


def _ascii_sections(raw: bytes) -> str:
    """$PhysicalNames is ASCII even in a binary file."""
    m = re.search(rb"\$PhysicalNames\n.*?\n\$EndPhysicalNames", raw, re.S)
    return m.group(0).decode("utf-8", "replace") if m else ""


def _seek(raw: bytes, marker: bytes, start: int) -> int:
    i = raw.index(marker, start) + len(marker)
    while i < len(raw) and raw[i:i + 1] in (b"\n", b"\r"):
        i += 1
    return i


def _read_entities_41_binary(raw, pos):
    i = _seek(raw, b"$Entities", pos)
    counts = np.frombuffer(raw, np.uint64, 4, i)
    i += 32
    out = {}
    for dim, n in enumerate(int(v) for v in counts):
        for _ in range(n):
            tag = int(np.frombuffer(raw, np.int32, 1, i)[0])
            i += 4
            i += 24 if dim == 0 else 48                # 3 or 6 doubles
            n_phys = int(np.frombuffer(raw, np.uint64, 1, i)[0])
            i += 8
            out[(dim, tag)] = [int(v) for v in
                               np.frombuffer(raw, np.int32, n_phys, i)]
            i += 4 * n_phys
            if dim > 0:                                # bounding entities
                nb = int(np.frombuffer(raw, np.uint64, 1, i)[0])
                i += 8 + 4 * nb
    return out, i


def _read_nodes_41_binary(raw, pos):
    i = _seek(raw, b"$Nodes", pos)
    n_blocks, n_nodes = (int(v) for v in np.frombuffer(raw, np.uint64, 2, i))
    i += 32
    tags = np.empty(n_nodes, dtype=np.int64)
    coords = np.empty((n_nodes, 3))
    at = 0
    for _ in range(n_blocks):
        if int(np.frombuffer(raw, np.int32, 1, i + 8)[0]):
            raise UserError("MESH_PARAMETRIC_NODES",
                            "the mesh file stores parametric node coordinates, which this "
                            "reader does not take",
                            "export again with Mesh.SaveParametric = 0 (the default)")
        nb = int(np.frombuffer(raw, np.uint64, 1, i + 12)[0])
        i += 20
        tags[at:at + nb] = np.frombuffer(raw, np.uint64, nb, i)
        i += 8 * nb
        coords[at:at + nb] = np.frombuffer(raw, np.float64, 3 * nb, i
                                           ).reshape(nb, 3)
        i += 24 * nb
        at += nb
    return coords, tags, i


def _read_elements_41_binary(raw, pos):
    i = _seek(raw, b"$Elements", pos)
    n_blocks = int(np.frombuffer(raw, np.uint64, 1, i)[0])
    i += 32
    elems = []
    for _ in range(n_blocks):
        dim, ent, etype = (int(v) for v in np.frombuffer(raw, np.int32, 3, i))
        nb = int(np.frombuffer(raw, np.uint64, 1, i + 12)[0])
        i += 20
        nn = _n_nodes(etype)
        block = np.frombuffer(raw, np.uint64, nb * (1 + nn), i
                              ).reshape(nb, 1 + nn)
        i += 8 * nb * (1 + nn)
        elems.append((dim, ent, etype, block[:, 1:].astype(np.int64)))
    return elems


def _n_nodes(etype: int) -> int:
    if etype in _VOLUME:
        return len(_VOLUME[etype][1])
    if etype in _OTHER_NODE_COUNT:
        return _OTHER_NODE_COUNT[etype]
    if etype in _UNSUPPORTED:
        raise ValueError(
            f"Gmsh element type {etype} ({_UNSUPPORTED[etype]}) is not "
            "supported. Supported: tet4, tet10, hexa8, hexa20, wedge6, wedge15, pyramid5."
        )
    raise ValueError(f"unknown Gmsh element type {etype}")


# --------------------------------------------------------------------------- #
#  MSH 2.2
# --------------------------------------------------------------------------- #


def _parse_22_ascii(raw: bytes):
    text = raw.decode("utf-8", "replace")
    names = _physical_names(text)

    body = _section(text, "Nodes").strip().split("\n")
    n = int(body[0])
    block = _numbers(body[1:1 + n]).reshape(n, 4)
    tags = block[:, 0].astype(np.int64)
    coords = np.ascontiguousarray(block[:, 1:4])

    body = _section(text, "Elements").strip().split("\n")
    n = int(body[0])
    # 2.2 puts the physical tag on the element, so group by (type, physical)
    # and present the same shape the 4.1 path produces
    buckets: Dict[Tuple[int, int], list] = {}
    for line in body[1:1 + n]:
        p = [int(v) for v in line.split()]
        etype, n_tags = p[1], p[2]
        phys = p[3] if n_tags >= 1 else 0
        buckets.setdefault((etype, phys), []).append(p[3 + n_tags:])
    elems = []
    for (etype, phys), rows in buckets.items():
        dim = _type_dim(etype)
        # entity tag doubles as the physical tag here; _assemble reads
        # ent_phys[(dim, tag)] and gets [phys] back
        elems.append((dim, phys, etype, np.array(rows, dtype=np.int64)))
    ent_phys = {(d, t): ([t] if t else []) for d, t, _e, _r in elems}
    return names, ent_phys, coords, tags, elems


# --------------------------------------------------------------------------- #
#  assembly
# --------------------------------------------------------------------------- #


def _assemble(names, ent_phys, tags, elems):
    """Pick the volume element type, remap tags, and build the node groups."""
    lut = _tag_lookup(tags)

    found = {}
    for _dim, _ent, etype, rows in elems:
        if etype in _VOLUME or etype in _STRUCTURAL:
            found.setdefault(etype, 0)
            found[etype] += rows.shape[0]
        elif etype in _UNSUPPORTED:
            raise ValueError(
                f"the file contains {rows.shape[0]} {_UNSUPPORTED[etype]} "
                "elements, which are not supported. Supported: truss2, shell4, tet4, "
                "tet10, hexa8, hexa20, wedge6, wedge15, pyramid5."
            )
    # A volume mesh normally also stores Quad4 boundary faces, and a shell
    # mesh normally stores Line2 boundary edges.  Select the highest supported
    # topological dimension; lower-dimensional blocks remain region sources.
    if any(t in _VOLUME for t in found):
        found = {t:n for t,n in found.items() if t in _VOLUME}
    elif 2 in found and 3 in found:
        # a quad mesh with triangles left in it (an incomplete Recombine):
        # the triangles were dropped and the model ran on the quads alone
        raise UserError(
            "MIXED_SHELL_ELEMENTS",
            f"the mesh has {found[3]} quadrilateral and {found[2]} triangular shell "
            "elements; one shell model takes one of the two",
            "recombine every surface (Mesh.RecombineAll = 1, or Recombine Surface{:}), "
            "or mesh with triangles only")
    elif 3 in found:
        found = {3:found[3]}
    elif 2 in found:
        # a triangle mesh with a Physical Curve for its support edge was read
        # as a truss of the curve's lines
        found = {2:found[2]}
    elif 1 in found:
        found = {1:found[1]}
    if not found:
        raise ValueError(
            "the file contains no 3-D elements or supported structural elements. Expected "
            "Line2, Quad4 shell, or a supported 3-D solid element."
        )
    # More than one type is a MESH, not a mistake.  Gmsh caps every recombined
    # quadrilateral face that borders a tetrahedral volume with a pyramid, so a
    # single `Recombine Surface` makes the file mixed -- and someone deliberately
    # meshing the regular part of a structure with hexahedra and the awkward
    # part with tetrahedra gets the same file.  Both are now solved directly:
    # each type becomes a block, and the solver sums over blocks.
    if len(found) > 1 and any(t in _STRUCTURAL for t in found):
        kinds = ", ".join(f"{(_VOLUME | _STRUCTURAL)[t][0]} ({n})"
                          for t, n in found.items())
        raise UserError(
            "MIXED_ELEMENT_FAMILIES",
            f"the mesh mixes solid and structural elements: {kinds}. "
            "A solid model and a beam/shell model are solved by different "
            "solvers, so they cannot share one block list.",
            "split the model, or use the coupled reader for solid + Line2 "
            "reinforcement. `vfife mesh <file>` lists what is in the file.",
        )

    perms = {t: (_VOLUME | _STRUCTURAL)[t] for t in found}
    conn_rows = {t: [] for t in found}
    row_entity = {t: [] for t in found}
    group_nodes = {}
    # region name -> block name -> the block's row indices the group holds.
    # Which ELEMENTS a physical group names is what separates a hanger from
    # the deck it is tied to: every hanger node is a deck or cable node, so
    # the every-node-inside rule hands the deck's elements to the hangers
    # too (the coupled reader had this since v0.1).
    group_rows: dict = {}
    counts = {t: 0 for t in found}
    for dim, ent, et, rows in elems:
        mapped = lut(rows) if rows.size else rows
        kept = None
        if et in conn_rows:
            conn_rows[et].append(mapped[:, perms[et][1]])
            row_entity[et].append(np.full(rows.shape[0], ent, dtype=np.int64))
            kept = (perms[et][0], np.arange(counts[et], counts[et] + rows.shape[0], dtype=np.int64))
            counts[et] += rows.shape[0]
        for phys in ent_phys.get((dim, ent), []):
            name = names.get((dim, phys), f"physical_{phys}")
            group_nodes.setdefault(name, []).append(mapped.ravel())
            if kept is not None:
                group_rows.setdefault(name, {}).setdefault(kept[0], []).append(kept[1])

    # Sorted by name so the block order -- and therefore the summation order of
    # the assembly, and the global element numbering in every output -- is a
    # property of the model, not of the order Gmsh happened to write.
    blocks = {perms[t][0]: np.vstack(conn_rows[t]) for t in found}
    ents = {perms[t][0]: np.concatenate(row_entity[t]) for t in found}
    blocks = {k: blocks[k] for k in sorted(blocks)}
    blocks, notes, kept_rows = _merge_repeats(blocks, ents)
    groups = {k: np.unique(np.concatenate(v)) for k, v in group_nodes.items()}
    element_regions = {}
    for name, per_block in group_rows.items():
        for b, v in per_block.items():
            rows_ = np.unique(np.concatenate(v))
            row_map = kept_rows.get(b)
            if row_map is not None:
                # every original row, at the place of the cell it merged into
                rows_ = np.unique(row_map[rows_])
            if rows_.size:
                element_regions.setdefault(name, {})[b] = rows_
    return blocks, groups, notes, element_regions


def _merge_repeats(blocks, entities):
    """Collapse cells written more than once, and say why they were.

    An element is its node SET: four nodes determine one tetrahedron, and no
    valid mesh contains two distinct cells on the same nodes.  So a repeated
    node set is always one cell written twice, and keeping both would double
    that region's stiffness and mass -- an error that produces a plausible
    answer rather than a crash.

    Two things produce them, and they deserve different words:

    * The cell sits on a boundary between two entities and Gmsh emitted it for
      both.  Then it belongs to both physical groups, which is exactly what the
      user asked for by naming both -- merging is the whole answer.
    * The cell is repeated INSIDE one entity.  That is Gmsh output the user did
      not ask for and cannot fix by renaming a group; it comes from a meshing
      failure upstream (`Recombine Surface {:}` across an internal interface
      produces hundreds, with `non-manifold quad boundaries not supported yet`
      on stderr).  Merging is still right -- it is still one cell -- but the
      user has to be told, because the mesh they are holding is not the mesh
      they think they asked for.

    Returns the merged blocks, a list of notes for the preflight to report,
    and per repaired block the original rows that were kept (in order).
    """
    out, notes, kept = {}, [], {}
    for name, conn in blocks.items():
        canonical = np.sort(conn, axis=1)
        _uniq, first, inverse, counts = np.unique(
            canonical, axis=0, return_index=True,
            return_inverse=True, return_counts=True)
        if len(first) == conn.shape[0]:
            out[name] = conn
            continue
        # keep the first occurrence of each cell, in the original order, so the
        # element numbering of a clean mesh is untouched and that of a repaired
        # one is still the file's own order
        keep = np.sort(first)
        # EVERY row goes to the place of the row it merged into, so a cell a
        # 2.2 file lists once per physical group is in all of those groups --
        # it kept the first one only
        place = np.empty(len(first), dtype=np.int64)
        place[np.argsort(first)] = np.arange(len(first))
        row_map = place[np.asarray(inverse).reshape(-1)]
        ent = entities.get(name)
        repeated = np.flatnonzero(counts > 1)
        within = 0
        if ent is not None:
            inverse = inverse.reshape(-1)
            for g in repeated:
                rows = np.flatnonzero(inverse == g)
                # one entity claiming the same cell twice, rather than two
                # entities each claiming it once
                within += len(rows) - len(np.unique(ent[rows]))
        across = int((counts - 1).sum()) - within
        notes.append({"element": name, "merged": int((counts - 1).sum()),
                      "across_entities": int(across), "within_entity": int(within),
                      "kept": int(len(keep))})
        out[name] = np.ascontiguousarray(conn[keep]); kept[name] = row_map
    return out, notes, kept


def _tag_lookup(tags: np.ndarray):
    """Gmsh node tag -> 0-based index, vectorised.

    Tags are usually 1..N but the format allows them sparse, and a dict lookup
    per element node is far too slow at this project's target size.  A direct
    lookup table is O(1) and costs 8 bytes per tag value; if the tags are so
    sparse that the table would dwarf the mesh, fall back to a binary search on
    the sorted tags, which is slower but bounded.
    """
    if len(np.unique(tags)) != tags.size:
        u, c = np.unique(tags, return_counts=True)
        raise UserError("MESH_DUPLICATE_NODE_TAG",
                        f"node tag(s) {', '.join(str(int(t)) for t in u[c > 1][:5])} appear more "
                        "than once in the mesh file; the last coordinates would silently win",
                        "re-export the mesh from Gmsh")

    def missing(rows, bad):
        raise UserError("MESH_UNKNOWN_NODE_TAG",
                        f"an element names node tag(s) {', '.join(str(int(t)) for t in np.unique(rows[bad])[:5])}, "
                        "which the $Nodes section does not define",
                        "the file is incomplete or was edited by hand; re-export it")

    hi = int(tags.max())
    if hi < 8 * tags.size + 1024:
        table = np.full(hi + 1, -1, dtype=np.int64)
        table[tags] = np.arange(tags.size)

        def direct(rows):
            # a missing tag was -1, i.e. silently the LAST node
            bad = (rows < 0) | (rows > hi)
            out = table[np.clip(rows, 0, hi)]
            bad |= out < 0
            if bad.any():
                missing(rows, bad)
            return out
        return direct

    order = np.argsort(tags)
    sorted_tags = tags[order]

    def lookup(rows):
        pos = np.clip(np.searchsorted(sorted_tags, rows), 0, len(sorted_tags) - 1)
        bad = sorted_tags[pos] != rows
        if bad.any():
            missing(rows, bad)
        return order[pos]

    return lookup


def _used_tags(tags, blocks):
    """The tags of the nodes `_prune` keeps, in its order."""
    used = np.zeros(tags.shape[0], dtype=bool)
    for conn in blocks.values():
        used[np.unique(conn)] = True
    return tags[used].copy()


def apply_overlay(names, ent_phys, nodes, tags, elems, overlay):
    """The parsed file with `overlay` added: `overlay["nodes"]`, an (k, 3)
    array in the case's length unit (after `scale`), get the tags after the
    file's largest; `overlay["lines"]`, [(group name, (m, 2) node refs)],
    each a physical Line2 group of its own after the file's, a ref being a
    tag of the file (> 0) or -(j + 1) for the overlay's node j. The order is
    the overlay's, so what reads it is the same every time."""
    new_xyz = np.asarray(overlay.get("nodes", np.zeros((0, 3))), float).reshape(-1, 3)
    base = int(tags.max()) if tags.size else 0
    new_tags = base + 1 + np.arange(new_xyz.shape[0], dtype=np.int64)
    nodes = np.vstack([nodes, new_xyz]) if new_xyz.size else nodes
    tags = np.concatenate([tags, new_tags]) if new_xyz.size else tags
    names = dict(names)
    ent_phys = dict(ent_phys)
    elems = list(elems)
    taken = {n for n in names.values()}
    phys = max([p for (d, p) in names if d == 1] + [p for (d, _e), ps in ent_phys.items() if d == 1 for p in ps] + [0])
    ent = max([e for (d, e) in ent_phys if d == 1] + [e for d, e, _t, _r in elems if d == 1] + [0])
    for name, refs in overlay.get("lines", []):
        if name in taken:
            raise ValueError(f"the mesh already has a group named {name!r}")
        taken.add(name)
        refs = np.asarray(refs, dtype=np.int64).reshape(-1, 2)
        rows = np.where(refs > 0, refs, new_tags[np.clip(-refs - 1, 0, max(len(new_tags) - 1, 0))] if len(new_tags) else refs)
        phys += 1
        ent += 1
        names[(1, phys)] = name
        ent_phys[(1, ent)] = [phys]
        elems.append((1, ent, 1, rows))
    return names, ent_phys, nodes, tags, elems


def _prune(nodes, blocks, groups):
    """Drop nodes no volume element uses, and renumber.

    Gmsh files carry nodes belonging to points and curves that no volume
    element references.  Left in, each becomes a node with no stiffness and no
    mass -- exactly the free particle the erosion work was about, arriving for
    free at import time.
    """
    used = np.zeros(nodes.shape[0], dtype=bool)
    for conn in blocks.values():
        used[np.unique(conn)] = True
    if used.all():
        return nodes, blocks, groups
    new = np.full(nodes.shape[0], -1, dtype=np.int64)
    new[used] = np.arange(int(used.sum()))
    groups = {k: new[v][new[v] >= 0] for k, v in groups.items()}
    blocks = {k: new[v] for k, v in blocks.items()}
    return nodes[used], blocks, {k: v for k, v in groups.items() if v.size}


def _fix_orientation(nodes, conn, element):
    """Flip inverted elements at import, not with a Jacobian error later.

    Gmsh is consistent about orientation, but a mesh that has been through a
    format conversion or a hand edit may not be, and `precompute` rejects a
    negative Jacobian with a message about determinants -- true, and no help at
    all when the fix is a node swap.
    """
    conn = np.array(conn, dtype=np.int64, copy=True)
    x = nodes[conn]
    if element in ("tet4", "tet10"):
        vol = np.einsum("ei,ei->e",
                        np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]),
                        x[:, 3] - x[:, 0])
        bad = vol < 0
        if bad.any():
            # swapping corners 1 and 2 relabels the mid-edge nodes too
            order = ([0, 2, 1, 3, 6, 5, 4, 7, 9, 8] if element == "tet10"
                     else [0, 2, 1, 3])
            conn[bad] = conn[bad][:, order]
    elif element == "wedge6":
        n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        bad = np.einsum("ei,ei->e", n, x[:, 3] - x[:, 0]) < 0
        if bad.any():
            conn[bad] = conn[bad][:, [0, 2, 1, 3, 5, 4]]
    elif element == "hexa8":               # Jacobian at the centre
        a = x[:, 1] - x[:, 0]
        b = x[:, 3] - x[:, 0]
        c = x[:, 4] - x[:, 0]
        bad = np.einsum("ei,ei->e", np.cross(a, b), c) < 0
        if bad.any():
            conn[bad] = conn[bad][:, [4, 5, 6, 7, 0, 1, 2, 3]]
    elif element == "hexa20":
        a = x[:, 1] - x[:, 0]
        b = x[:, 3] - x[:, 0]
        c = x[:, 4] - x[:, 0]
        bad = np.einsum("ei,ei->e", np.cross(a, b), c) < 0
        if bad.any():
            conn[bad] = conn[bad][:, [4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11,16,17,18,19]]
    elif element == "pyramid5":            # base normal against the apex
        n = np.cross(x[:, 1] - x[:, 0], x[:, 3] - x[:, 0])
        bad = np.einsum("ei,ei->e", n, x[:, 4] - x[:, 0]) < 0
        if bad.any():
            conn[bad] = conn[bad][:, [0, 3, 2, 1, 4]]
    elif element == "wedge15":
        n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        bad = np.einsum("ei,ei->e", n, x[:, 3] - x[:, 0]) < 0
        if bad.any():
            conn[bad] = conn[bad][:, [3,4,5,0,1,2,9,10,11,6,7,8,12,13,14]]
    return conn
