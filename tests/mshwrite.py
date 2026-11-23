"""Tiny hand-written .msh writers (MSH 2.2 and 4.1) for tests of the reader."""
import numpy as np

NN = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5, 15: 1, 11: 10, 17: 20, 18: 15}


def write22(path, nodes, elems, names, tags=None):
    """nodes: list of xyz; elems: list of (etype, phys, [1-based node tags]);
    names: list of (dim, tag, name)."""
    tags = tags or list(range(1, len(nodes) + 1))
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat"]
    if names:
        out += ["$PhysicalNames", str(len(names))]
        out += [f'{d} {t} "{n}"' for d, t, n in names]
        out += ["$EndPhysicalNames"]
    out += ["$Nodes", str(len(nodes))]
    out += [f"{t} {x!r} {y!r} {z!r}" for t, (x, y, z) in zip(tags, nodes)]
    out += ["$EndNodes", "$Elements", str(len(elems))]
    for k, (et, ph, c) in enumerate(elems):
        out.append(f"{k+1} {et} 2 {ph} {ph} " + " ".join(map(str, c)))
    out += ["$EndElements", ""]
    open(path, "w", encoding="utf-8").write("\n".join(out))


def write41(path, nodes, blocks, names, ent_phys, tags=None, param=False):
    """blocks: list of (dim, entity, etype, [[node tags]]);
    ent_phys: {(dim, ent): [phys tags]}; names [(dim, tag, name)].
    All nodes put in one node block on entity (3,1) (or dim of the max)."""
    tags = tags or list(range(1, len(nodes) + 1))
    ents = {d: sorted({e for (dd, e) in ent_phys if dd == d} |
                      {b[1] for b in blocks if b[0] == d}) for d in range(4)}
    out = ["$MeshFormat", "4.1 0 8", "$EndMeshFormat"]
    if names:
        out += ["$PhysicalNames", str(len(names))]
        out += [f'{d} {t} "{n}"' for d, t, n in names]
        out += ["$EndPhysicalNames"]
    out += ["$Entities", " ".join(str(len(ents[d])) for d in range(4))]
    for d in range(4):
        for e in ents[d]:
            ph = ent_phys.get((d, e), [])
            if d == 0:
                out.append(f"{e} 0 0 0 {len(ph)} " + " ".join(map(str, ph)))
            else:
                out.append(f"{e} 0 0 0 1 1 1 {len(ph)} " + " ".join(map(str, ph)) + " 0")
    out += ["$EndEntities"]
    out += ["$Nodes", f"1 {len(nodes)} {min(tags)} {max(tags)}",
            f"3 1 {1 if param else 0} {len(nodes)}"]
    out += [str(t) for t in tags]
    for (x, y, z) in nodes:
        out.append(f"{x!r} {y!r} {z!r}" + (" 0.5 0.5 0.5" if param else ""))
    out += ["$EndNodes"]
    ne = sum(len(b[3]) for b in blocks)
    out += ["$Elements", f"{len(blocks)} {ne} 1 {ne}"]
    k = 1
    for d, e, et, rows in blocks:
        out.append(f"{d} {e} {et} {len(rows)}")
        for r in rows:
            out.append(f"{k} " + " ".join(map(str, r))); k += 1
    out += ["$EndElements", ""]
    open(path, "w", encoding="utf-8").write("\n".join(out))


def hex_bar(n=2, L=1.0):
    """n hexa8 along z: nodes, hex rows (1-based)."""
    nodes = []
    for k in range(n + 1):
        z = L * k / n
        nodes += [(0, 0, z), (1, 0, z), (1, 1, z), (0, 1, z)]
    hexes = [[4*k+1, 4*k+2, 4*k+3, 4*k+4, 4*k+5, 4*k+6, 4*k+7, 4*k+8] for k in range(n)]
    return nodes, hexes
