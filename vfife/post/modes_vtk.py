"""Mode shapes as ParaView time series.

A modal result is frequencies and vectors. What anyone does with it first is
look at it, and until this stage the only file `vfife modal` wrote was an
HDF5 of arrays with no mesh in it -- ParaView could not open it, and
`vfife sweep` refuses anything that is not a fibre beam.

What is written, into one directory:

  <stem>_mode01.pvd ...   one collection per mode: `frames` snapshots over
                          one period, u(t) = scale * phi * sin(2 pi t). The
                          points ARE the displaced positions, so the file
                          animates on Play with no filter in front of it;
                          `displacement` and `|displacement|` come along for
                          colouring
  <stem>_all.pvd          every mode one after another on one time axis:
                          mode k occupies [k-1, k) -- scrub through them all
  <stem>/                 the .vtu frames both collections point at (written
                          once, shared)
  <stem>.txt              the table: mode, f, T, participation, label

`scale` is the largest translation any node reaches in any mode, metres;
the default is 3 % of the model's largest extent (80 m on a 2.7 km bridge),
enough to read a lateral mode at the whole-bridge zoom. Every shape is
normalised to a unit maximum translation first (modal.analyse does that),
so the scale is the same for every mode and their sizes do not compare --
a mode shape has no size.

The mesh is the solver's own: Line2 members for the line solvers, and the
continuum or shell blocks where there are any, with the member kind as cell
data where the model has several (so a Threshold separates cables from
girder from tower).
"""
from __future__ import annotations

import pathlib

import numpy as np

_MESHIO_TYPE = {"truss2": "line", "shell4": "quad", "shell3": "triangle",
                "hexa8": "hexahedron", "hexa20": "hexahedron20", "tetra4": "tetra",
                "tetra10": "tetra10", "wedge6": "wedge", "wedge15": "wedge15",
                "pyramid5": "pyramid"}


def _cells(solver):
    """[(meshio type, connectivity)], and a per-block cell-data list of
    member-kind codes (or None) with its legend."""
    mesh = solver.mesh
    blocks = getattr(mesh, "cell_blocks", None) or {}
    if blocks:
        cells = [(_MESHIO_TYPE.get(k, k), np.asarray(v)) for k, v in blocks.items()]
    else:
        conn = np.asarray(mesh.elements)
        kind = "line" if conn.shape[1] == 2 else _MESHIO_TYPE.get(
            getattr(solver, "element", ""), "line")
        cells = [(kind, conn)]
    data, legends = {}, {}
    mk = getattr(solver, "member_kind", None)
    if mk is not None and len(cells) == 1 and len(mk) == len(cells[0][1]):
        names = sorted(set(str(k) for k in mk))
        legends["member_kind"] = {n: i for i, n in enumerate(names)}
        data["member_kind"] = [np.array([legends["member_kind"][str(k)] for k in mk],
                                        dtype=np.int32)]
    # the mesh's own physical groups -- girder, tower, main cable, stays --
    # which is what one wants to Threshold on; the first group an element is
    # in wins, and an element in none is -1
    regions = getattr(mesh, "element_regions", None) or {}
    if regions and len(cells) == 1:
        block = "truss2" if cells[0][0] == "line" else next(
            (k for k, v in _MESHIO_TYPE.items() if v == cells[0][0]), cells[0][0])
        grp = np.full(len(cells[0][1]), -1, dtype=np.int32)
        legend = {}
        for name, per_block in regions.items():
            ids = per_block.get(block) if isinstance(per_block, dict) else None
            if ids is None or not len(ids):
                continue
            ids = np.asarray(ids)
            free = ids[grp[ids] < 0]
            if len(free):
                legend[name] = len(legend)
                grp[free] = legend[name]
        if legend:
            data["group"] = [grp]
            legends["group"] = legend
    return cells, data, legends


def _translations(solver, shapes):
    """(n_modes, nn, 3) translational part of the shapes."""
    nn = solver.mesh.nodes.shape[0]
    dofs = getattr(solver, "dofs", None)
    if dofs is not None:
        from ..core.dof import TRANSLATION
        T = dofs.component_indices(TRANSLATION)
        return shapes.reshape(len(shapes), nn, dofs.stride)[:, :, T]
    return shapes.reshape(len(shapes), nn, -1)[:, :, :3]


def write(solver, result, out_dir, stem="modes", frames=24, scale=None,
          labels=None, log=print, rows=None):
    """Write `result` (a modal.ModalResult) of `solver` as ParaView
    collections in `out_dir`. Returns the path of the all-modes .pvd."""
    import meshio
    out = pathlib.Path(out_dir)
    frame_dir = out / stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    x0 = np.asarray(solver.mesh.nodes, dtype=float)
    extent = float(np.ptp(x0, axis=0).max()) or 1.0
    scale = 0.03 * extent if scale is None else float(scale)
    frames = max(int(frames), 2)
    cells, cell_data, legends = _cells(solver)
    U = _translations(solver, np.asarray(result.shapes))
    n_modes = len(result.shapes)
    labels = list(labels) if labels is not None else [""] * n_modes

    def tag(k):
        lab = labels[k - 1]
        safe = "".join(c if c.isalnum() else "-" for c in lab).strip("-")
        return f"mode{k:02d}" + (f"_{safe}" if safe else "")

    everything = []
    for k in range(1, n_modes + 1):
        entries = []
        for j in range(frames):
            phase = j / frames
            u = scale * np.sin(2.0 * np.pi * phase) * U[k - 1]
            fn = f"{tag(k)}_{j:03d}.vtu"
            meshio.write(frame_dir / fn, meshio.Mesh(
                points=x0 + u, cells=cells,
                point_data={"displacement": u,
                            "displacement_magnitude": np.linalg.norm(u, axis=1),
                            "mode_shape": U[k - 1]},
                cell_data=cell_data), binary=True)
            entries.append((phase, fn))
            everything.append((k - 1 + phase, fn))
        _collection(out / f"{stem}_{tag(k)}.pvd", stem, entries)
    all_pvd = _collection(out / f"{stem}_all.pvd", stem, everything)

    if rows is None:
        # the modal table; a caller with other shapes (buckling) gives its own
        rows = ["mode  f [Hz]     T [s]      eff.mass x   y      z      label"]
        part = np.asarray(result.participation)
        for k in range(n_modes):
            rows.append(f"{k+1:4d}  {result.frequencies[k]:.5f}  {result.periods[k]:9.4f}  "
                        f"{part[k, 0]:6.3f} {part[k, 1]:6.3f} {part[k, 2]:6.3f}  {labels[k]}")
    rows = list(rows)
    rows.append("")
    rows.append(f"scale: the largest translation in every mode is drawn as {scale:.4g} (case length unit)")
    for key, legend in legends.items():
        rows.append(f"{key}: " + ", ".join(f"{i} = {n}" for n, i in legend.items()))
    (out / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    log(f"  paraview      {all_pvd} (every mode on one time axis; mode k is t in [k-1, k))")
    log(f"                {out / (stem + '_' + tag(1) + '.pvd')} ... one collection per mode, "
        f"{frames} frames each; largest translation drawn as {scale:.3g}")
    return all_pvd


def _collection(pvd, dirname, entries):
    lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="0.1">',
             '  <Collection>']
    lines += [f'    <DataSet timestep="{t:.10g}" file="{dirname}/{fn}"/>' for t, fn in entries]
    lines += ['  </Collection>', '</VTKFile>', '']
    pathlib.Path(pvd).write_text("\n".join(lines), encoding="utf-8")
    return str(pvd)
