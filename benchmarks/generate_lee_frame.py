"""Lee's frame as a Gmsh 2.2 line mesh (cases/case_16).

Two members of length L at a right angle in the x-z plane: a column from the
origin up to the corner, a beam from the corner out along +x. Point A, where
the load acts, sits 0.2 L from the corner on the beam -- the position every
version of this benchmark uses (Lee, Manuel & Rossow 1968; NAFEMS NL7).

    pin_l    (0, 0, 0)         bottom of the column
    corner   (0, 0, L)
    load     (0.2 L, 0, L)     point A
    pin_r    (L, 0, L)         far end of the beam

`n` elements per member. A is a node whenever 0.2 n is an integer, so n is a
multiple of 5; the generator refuses anything else rather than put the load
between nodes.
"""
import argparse
import pathlib

L = 1.2


def build(path, n=20):
    if n % 5:
        raise ValueError(f"n = {n}: 0.2 L must land on a node, so n is a multiple of 5")
    nodes = [(0.0, 0.0, 0.0)]
    for i in range(1, n + 1):                       # the column, bottom to corner
        nodes.append((0.0, 0.0, L * i / n))
    for i in range(1, n + 1):                       # the beam, corner to the far pin
        nodes.append((L * i / n, 0.0, L))
    i_corner, i_load, i_end = n, n + n // 5, 2 * n
    lines = [(i, i + 1) for i in range(2 * n)]      # 0-based node indices

    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
           "$PhysicalNames", "5",
           '0 1 "pin_l"', '0 2 "pin_r"', '0 3 "load"', '0 4 "corner"',
           '1 5 "members"', "$EndPhysicalNames",
           "$Nodes", str(len(nodes))]
    out += [f"{k + 1} {x:.10g} {y:.10g} {z:.10g}" for k, (x, y, z) in enumerate(nodes)]
    out += ["$EndNodes", "$Elements", str(4 + len(lines))]
    e = 1
    for tag, node in ((1, 0), (2, i_end), (3, i_load), (4, i_corner)):
        out.append(f"{e} 15 2 {tag} {tag} {node + 1}"); e += 1
    for a, b in lines:
        out.append(f"{e} 1 2 5 5 {a + 1} {b + 1}"); e += 1
    out += ["$EndElements", ""]
    pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="benchmarks/assets/lee_frame_v22.msh")
    ap.add_argument("--n", type=int, default=20, help="elements per member")
    a = ap.parse_args()
    print("wrote", build(a.out, a.n))
