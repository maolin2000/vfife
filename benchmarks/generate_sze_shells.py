"""Structured Quad4 meshes for the Sze, Liu & Lo (2004) shell benchmarks.

    hinged cylindrical roof   (their 3.8, cases/case_17)   quarter model
    hemisphere with 18° hole  (their 3.4, cases/case_18)   quarter model

Both are written as Gmsh 2.2 ASCII with physical POINTS for the loaded and
monitored nodes, physical LINES for the boundary edges the case restrains,
and one physical SURFACE. No Gmsh needed: the surfaces are parametrised
rectangles, and writing the file is forty lines.

    python benchmarks/generate_sze_shells.py roof [--n 16]
    python benchmarks/generate_sze_shells.py hemisphere [--n 16]
"""
import argparse
import pathlib

import numpy as np


def _write(path, nodes, quads, points, lines, surface_name):
    """nodes (n,3); quads (m,4) 0-based; points {name: node}; lines {name: [(a,b),...]}."""
    names = list(points) + list(lines) + [surface_name]
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    tag = {}
    for k, nm in enumerate(names, 1):
        dim = 0 if nm in points else (1 if nm in lines else 2)
        out.append(f'{dim} {k} "{nm}"'); tag[nm] = k
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += [f"{i + 1} {x:.12g} {y:.12g} {z:.12g}" for i, (x, y, z) in enumerate(nodes)]
    n_el = len(points) + sum(len(v) for v in lines.values()) + len(quads)
    out += ["$EndNodes", "$Elements", str(n_el)]
    e = 1
    for nm, node in points.items():
        out.append(f"{e} 15 2 {tag[nm]} {tag[nm]} {node + 1}"); e += 1
    for nm, segs in lines.items():
        for a, b in segs:
            out.append(f"{e} 1 2 {tag[nm]} {tag[nm]} {a + 1} {b + 1}"); e += 1
    for a, b, c, d in quads:
        out.append(f"{e} 3 2 {tag[surface_name]} {tag[surface_name]} {a + 1} {b + 1} {c + 1} {d + 1}"); e += 1
    out += ["$EndElements", ""]
    pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
    return path


def _grid(n, m):
    """Node index of (i, j) on an (n+1) x (m+1) grid, and the quads."""
    idx = lambda i, j: i * (m + 1) + j
    quads = [(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1))
             for i in range(n) for j in range(m)]
    return idx, quads


def roof(path, n=16, R=2540.0, L=254.0, theta=0.1):
    """Quarter of the hinged cylindrical roof. Cylinder axis along y, crown
    at the origin, the roof falling away in -z. i runs along the arc (x),
    j along the axis (y). Sze et al. Fig. 9a."""
    idx, quads = _grid(n, n)
    nodes = []
    for i in range(n + 1):
        phi = theta * i / n
        for j in range(n + 1):
            nodes.append((R * np.sin(phi), L * j / n, R * (np.cos(phi) - 1.0)))
    points = {"crown": idx(0, 0)}
    lines = {
        "sym_x": [(idx(0, j), idx(0, j + 1)) for j in range(n)],          # x = 0 plane
        "sym_y": [(idx(i, 0), idx(i + 1, 0)) for i in range(n)],          # y = 0 plane
        "hinged": [(idx(n, j), idx(n, j + 1)) for j in range(n)],         # the straight edge at phi = theta
        "free": [(idx(i, n), idx(i + 1, n)) for i in range(n)],           # the curved edge at y = L
    }
    return _write(path, nodes, quads, points, lines, "roof")


def hemisphere(path, n=16, R=10.0, hole_deg=18.0):
    """Quarter of the hemisphere with an 18-degree hole at the pole. i runs
    in longitude from the x axis (phi = 0) to the y axis (phi = 90 deg), j in
    colatitude from the hole (18 deg) down to the equator (90 deg). Point A
    is on the x axis at the equator, B on the y axis. Sze et al. Fig. 5a."""
    idx, quads = _grid(n, n)
    nodes = []
    t0 = np.radians(hole_deg)
    for i in range(n + 1):
        phi = 0.5 * np.pi * i / n
        for j in range(n + 1):
            th = t0 + (0.5 * np.pi - t0) * j / n
            nodes.append((R * np.sin(th) * np.cos(phi), R * np.sin(th) * np.sin(phi), R * np.cos(th)))
    points = {"A": idx(0, n), "B": idx(n, n)}
    lines = {
        "sym_y": [(idx(0, j), idx(0, j + 1)) for j in range(n)],          # y = 0 plane (the x-z meridian)
        "sym_x": [(idx(n, j), idx(n, j + 1)) for j in range(n)],          # x = 0 plane (the y-z meridian)
        "hole": [(idx(i, 0), idx(i + 1, 0)) for i in range(n)],
        "equator": [(idx(i, n), idx(i + 1, n)) for i in range(n)],
    }
    return _write(path, nodes, quads, points, lines, "hemisphere")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["roof", "hemisphere"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"benchmarks/assets/sze_{a.which}_{a.n}_v22.msh"
    print("wrote", (roof if a.which == "roof" else hemisphere)(out, a.n))
