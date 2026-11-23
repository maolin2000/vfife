"""Structured hexahedral mesh of the Chalmers PRC/RC T-beam (cases/case_15).

    python benchmarks/generate_prc_tbeam.py

WHY A GENERATOR AND NOT A HAND-DRAWN .geo
    The section is a T, and the box mesher cannot make one. A T is also not a
    four-sided region, so it cannot be one transfinite surface: it is built
    here as FOUR conformal rectangles -- the web, the flange directly above
    the web, and the two wings -- each transfinite, sharing their edges. That
    is what keeps every element a hexahedron and the mesh structured, which
    matters because the crack band (materials/cdp.py) is regularised on the
    element size and a graded or tetrahedral mesh would put `l_e/l_ref`
    outside the band the solver accepts.

    Fernandez et al. (2025), doi 10.1002/suco.70224:
        beam 8500 long, flange 320 x 130, web 150 x 515, total depth 645 mm.

THE ELEMENT SIZE IS CHOSEN, NOT CONVENIENT
    50 mm along the span and 37.5-51.5 mm across it. What matters is the
    SPREAD, 1.13 as built: `l_ref: auto` takes the smallest element, so every
    element's band factor `l_e/l_ref` lands inside the [0.5, 2.0] the solver
    accepts and the compression band is regularised.
    64 elements per slice, 170 slices, 10880 elements.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gmsh

# geometry, metres
L = 8.500
BF, TF = 0.320, 0.130          # flange width, thickness
BW, HW = 0.150, 0.515          # web width, height (soffit to flange soffit)
H = HW + TF                    # 0.645
Y0 = (BF - BW) / 2.0           # 0.085, the wing width

# divisions
NX = 170                       # 50.0 mm along the span
NY_WEB, NY_WING = 4, 2         # 37.5 and 42.5 mm across; the extra column puts
                               # a NODE PLANE on the beam axis, so the model can
                               # carry a real symmetry condition in y instead of
                               # restraining a soffit edge that wants to expand
NZ_WEB, NZ_FLANGE = 10, 3      # 51.5 and 43.3 mm through the depth


def _rect(y0, z0, y1, z1, ny, nz):
    """A transfinite quad in the y-z plane at x = 0, returned as its surface."""
    g = gmsh.model.geo
    p = [g.addPoint(0.0, y0, z0), g.addPoint(0.0, y1, z0),
         g.addPoint(0.0, y1, z1), g.addPoint(0.0, y0, z1)]
    l = [g.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    s = g.addPlaneSurface([g.addCurveLoop(l)])
    for line, n in zip(l, (ny, nz, ny, nz)):
        g.mesh.setTransfiniteCurve(line, n + 1)
    g.mesh.setTransfiniteSurface(s)
    g.mesh.setRecombine(2, s)
    return s


def build(path: Path, coarse: bool = False) -> None:
    """`coarse` halves every division: 1224 elements instead of 10880.

    The crack band is regularised by `Gf` and `Gc`, so the two meshes are
    SUPPOSED to give the same peak load -- that they do is the evidence, and
    it is the check that v0.1 exists to make possible. It is also the
    only version that runs in an hour without a GPU: 10880 CDP elements are
    about 20 relaxation steps per second on two cores.
    """
    nx, ny_web, ny_wing, nz_web, nz_flange = (
        # 68 along the span, not 85: 125 mm divides the 250 mm the supports
        # and load points sit at, and a support box that falls between node
        # planes selects nothing
        (68, 2, 1, 5, 2) if coarse else (NX, NY_WEB, NY_WING, NZ_WEB, NZ_FLANGE))
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("prc_tbeam")

    faces = [
        _rect(Y0, 0.0, Y0 + BW, HW, ny_web, nz_web),              # web
        _rect(Y0, HW, Y0 + BW, H, ny_web, nz_flange),             # flange, over the web
        _rect(0.0, HW, Y0, H, ny_wing, nz_flange),                # left wing
        _rect(Y0 + BW, HW, BF, H, ny_wing, nz_flange),            # right wing
    ]
    gmsh.model.geo.synchronize()
    # duplicate points on the shared edges would leave the four faces
    # touching but not joined -- the same defect that a bar route drawn as
    # separate lines has (io/schema.BarLineSpec), and here it would put a
    # crack straight up the web/flange junction
    gmsh.model.geo.removeAllDuplicates()
    gmsh.model.geo.synchronize()

    vols = gmsh.model.geo.extrude([(2, s) for s in faces], L, 0.0, 0.0,
                                  numElements=[nx], recombine=True)
    gmsh.model.geo.synchronize()
    tags = [t for dim, t in vols if dim == 3]
    gmsh.model.addPhysicalGroup(3, tags, name="body")

    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.model.mesh.generate(3)
    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(path))

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    types, tags_, _ = gmsh.model.mesh.getElements(3)
    counts = {int(t): len(v) for t, v in zip(types, tags_)}
    gmsh.finalize()
    print(f"{path}: {n_nodes} nodes, "
          + ", ".join(f"{v} of gmsh type {k}" for k, v in counts.items())
          + "  (type 5 is hexa8)")


if __name__ == "__main__":
    coarse = "--coarse" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    out = Path(args[0]) if args else (
        Path(__file__).resolve().parent / "assets" /
        ("prc_tbeam_645_coarse.msh" if coarse else "prc_tbeam_645.msh"))
    build(out, coarse)
