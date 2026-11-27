"""The model, prepared for the translation (see `prepare`)."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple


from .. import midas


def prepare(model, leave, notes, f, length, link_rigid, cases):
    """The model as the translation takes it: point springs as links to ground
    nodes, supports in a node's own axes, the load cases of the finished
    bridge as steps, master-slave rigid links and rigid elastic links as
    arms or stiff springs, and two nodes at one place set a millimetre apart.
    Returns (model, all_links, tied, drawn, six, rigid)."""
    import dataclasses
    # A POINT SPRING (`*SPRING`: soil on a pile) IS A LINK TO THE GROUND: a
    # node of its own at the same place, held in every direction, and an
    # elastic link from the model's node to it -- with the spring's boundary
    # group, so built and removed with it. A direction the row fixes takes
    # `link_rigid`. The ground nodes are numbered after the model's.
    # A NODE WITH AXES OF ITS OWN (`*LOCALAXIS`: a skew or a curved bridge's
    # bearings) has its supports and springs in them. Turned about Z only:
    # its spring's ground node is set a millimetre along its own x, so the
    # link's axes are the node's; and a support that holds one of its x and
    # y and not the other -- a guided bearing: held across the bridge, free
    # along it -- is a link `link_rigid` stiff in that direction to a
    # ground node, the rest of it staying a support. (Held in both or in
    # neither it is the same in any axes.) Other turns are counted.
    turned_z: Dict[int, float] = {}
    to_ground = set()                                # the pairs (a node, its ground node)
    for n, (ax_, ay_, az_) in getattr(model, "local_axes", {}).items():
        if ax_ or ay_:
            leave("node axes turned about x or y (taken as global)")
        elif az_:
            turned_z[n] = math.radians(az_)
    if model.other.get("LOCALAXIS (not by angles)"):
        leave("node axes given by points or vectors (taken as global)", model.other["LOCALAXIS (not by angles)"])
    if turned_z:
        # what each turned node is held in, all its rows of one boundary group together: two rows that hold
        # x and y between them are a plain support
        merged: Dict[Tuple[int, str], str] = {}
        rows_ = []
        for at, fl, g in model.constraints:
            fl = fl[:6].ljust(6, "0")
            plain = [n for n in at if n not in turned_z]
            if plain:
                rows_.append((plain, fl, g))
            for n in at:
                if n in turned_z and n in model.nodes:
                    was = merged.get((n, g), "000000")
                    merged[(n, g)] = "".join("1" if "1" in (x, y) else "0" for x, y in zip(was, fl))
        by_flags: Dict[Tuple[str, str], List[int]] = {}
        for (n, g), fl in merged.items():
            by_flags.setdefault((fl, g), []).append(n)
        rows_ += [(at, fl, g) for (fl, g), at in by_flags.items()]
        model = dataclasses.replace(model, constraints=rows_)
    skew = [(i, [n for n in at if n in turned_z]) for i, (at, fl, _g) in enumerate(model.constraints)
            if fl[:6].ljust(6, "0")[0] != fl[:6].ljust(6, "0")[1] or fl[:6].ljust(6, "0")[3] != fl[:6].ljust(6, "0")[4]]
    skew = [(i, at) for i, at in skew if at]
    if model.springs or skew:
        nodes, links, held = dict(model.nodes), list(model.links), list(model.constraints)
        nxt = max(nodes) + 1
        grounds: Dict[int, int] = {}

        def ground_of(n):
            """A node for the ground under n: at n's place (set a millimetre off below, like any link at
            one place), or a millimetre along n's own x."""
            nonlocal nxt
            if n in turned_z:                        # (each further one of a node another millimetre out)
                grounds[n] = grounds.get(n, 0) + 1
                step = grounds[n] * 1.0e-3 / length
                nodes[nxt] = (nodes[n][0] + step * math.cos(turned_z[n]), nodes[n][1] + step * math.sin(turned_z[n]), nodes[n][2])
            else:
                nodes[nxt] = nodes[n]
            to_ground.add(frozenset((n, nxt)))
            nxt += 1
            return nxt - 1
        for at, fixed, stiff, g in model.springs:
            fixed = tuple(fixed) + (False,) * (6 - len(fixed))
            if not any(stiff) and not any(fixed):
                leave("point spring of no stiffness", len(at))
                continue
            ground = []
            for n in at:
                if n not in nodes:
                    leave("point spring on a node the file does not define")
                    continue
                c = ground_of(n)
                links.append(midas.Link(kind="GEN", nodes=(n, c), stiffness=tuple(stiff), rigid=fixed, group=g))
                ground.append(c)
            if ground:
                held.append((ground, "111111", g))
        for i, at in skew:
            every_, fl, g = model.constraints[i]
            fl = fl[:6].ljust(6, "0")
            both = lambda j: "11" if fl[j] == fl[j + 1] == "1" else "00"
            rest = both(0) + fl[2] + both(3) + fl[5]
            alone = (fl[0] != fl[1] and fl[0] == "1", fl[0] != fl[1] and fl[1] == "1", False,
                     fl[3] != fl[4] and fl[3] == "1", fl[3] != fl[4] and fl[4] == "1", False)
            held[i] = ([n for n in every_ if n not in at], fl, g)
            if "1" in rest:
                held.append((list(at), rest, g))
            ground = []
            for n in at:
                c = ground_of(n)
                links.append(midas.Link(kind="GEN", nodes=(n, c), stiffness=(0.0,) * 6, rigid=alone, group=g))
                ground.append(c)
            held.append((ground, "111111", g))
            notes.append("supports in a node's own axes: the direction held alone is a link to the ground, "
                         f"{link_rigid:g} N/m (N m/rad)")
            if any(g == g_ and pos == "ORIGINAL" for st_ in model.stages for g_, pos in st_.activate_boundaries):
                leave("support in a node's own axes, to be brought to its original position: held where the node stands", len(at))
        model = dataclasses.replace(model, nodes=nodes, links=links, constraints=[h for h in held if h[0]])

    # A LOAD CASE OF THE FINISHED BRIDGE (`cases`: wind, say) is one more step
    # after the last stage, with that case's nodal and member loads, kept:
    # what the step adds is what Midas shows for the case alone, as far as
    # the bridge answers in proportion.
    if cases:
        if isinstance(cases, str) or len(set(cases)) != len(cases):
            raise ValueError("cases: a list of load case names, each once")
        if not model.stages:
            raise ValueError("cases: the model has no construction stages -- its load cases are all applied together")
        nl, bl, st = list(model.nodal_loads), list(model.beam_loads), list(model.stages)
        in_stage = {g for t in model.stages for g, _d in t.activate_loads}
        for c in cases:
            # A SETTLEMENT GROUP (`*SM-GROUP`) is a case of its own here: its
            # nodes go down by its displacement (Midas' settlement load cases
            # are envelopes over combinations of groups, which this does not
            # do; one group at a time is what a step can be)
            if c in model.settlements and c not in model.load_cases:
                dz, nodes_ = model.settlements[c]
                st.append(midas.Stage(name=c, displacements=[(nodes_, "001000", (0.0, 0.0, dz, 0.0, 0.0, 0.0))]))
                continue
            if c not in model.load_cases:
                raise ValueError(f"cases: the file has no load case {c!r} (it has {', '.join(model.load_cases)}"
                                 + (f"; settlement groups: {', '.join(model.settlements)}" if model.settlements else "")
                                 + ")")
            disp = [(nodes_, fl, vals) for cc, nodes_, fl, vals, _g in model.displacements if cc == c]
            # its temperatures (1.23): the whole model's (`*SYSTEMPER`), the
            # elements' (`*ELTEMPER`) and over the sections' depth (`*BSTEMPER`)
            temps = ([("system", None, rise) for cc, rise, _g in model.temperatures if cc == c]
                     + [("element", els, t_) for cc, els, t_, _g in model.element_temperatures if cc == c]
                     + [("section", row[1], (row[2], row[3], row[4], row[6]))
                        for row in model.section_temperatures if row[0] == c])
            mine = [q[3] for q in nl if q[0] == c] + [q[5] for q in bl if q[0] == c]
            heat = [g_ for cc, _r, g_ in model.temperatures if cc == c] \
                + [q[3] for q in model.element_temperatures if q[0] == c] \
                + [row[5] for row in model.section_temperatures if row[0] == c]
            used = sorted({g_ for g_ in mine + heat + [q[4] for q in model.displacements if q[0] == c]
                           if g_ in in_stage})
            if used:
                raise ValueError(f"cases: the load case {c!r} is already applied by a stage (load group {used[0]})")
            if not mine and (disp or temps):
                # a case of imposed support displacements or temperatures alone
                st.append(midas.Stage(name=c, displacements=disp, temperatures=temps))
                continue
            if not mine:
                raise ValueError(f"cases: the load case {c!r} has no nodal or member load this translation reads")
            g = f"(case {c})"
            nl = [(q[0], q[1], q[2], g) if q[0] == c else q for q in nl]
            bl = [q[:5] + (g,) + q[6:] if q[0] == c else q for q in bl]
            if any(w[0] == c for w in model.self_weight):
                leave(f"self weight of the load case {c} (its step has the case's other loads)")
            st.append(midas.Stage(name=c, activate_loads=[(g, "FIRST")], displacements=disp, temperatures=temps))
        model = dataclasses.replace(model, nodal_loads=nl, beam_loads=bl, stages=st,
                                    load_groups=list(model.load_groups) + [f"(case {c})" for c in cases])

    if model.settlements and not any(c in model.settlements for c in (cases or ())):
        notes.append(f"settlement groups ({', '.join(model.settlements)}) not taken: Midas' settlement cases are "
                     "envelopes over combinations of them; one at a time is convert(..., cases=(group,))")

    # A MASTER AND ITS SLAVES (`*RIGIDLINK`) ARE THE SOLVER'S RIGID LINKS
    # (1.21), the flags as Midas writes them -- a saddle that slides on its
    # tower (011111) too: `rigid` here, made into `rigid_links` by the
    # converter with their boundary group. They were rigid arms (stiff beams,
    # their stiffness a multiple of the model's) for the links tying all six
    # directions, and the others were left out.
    # A GEN LINK MAY SAY "RIGID" DIRECTION BY DIRECTION, and its number there
    # is then nothing (zero, often). Rigid in all six, it is a RIGID link;
    # in some, those directions take `link_rigid` (N/m and N m/rad).
    six = tuple(link_rigid / (f / length if i < 3 else f * length) for i in range(6))
    all_links = []
    tied: List[Tuple[int, int]] = []               # two nodes at one place made one by six stiff springs
    for link in model.links:
        here = model.nodes[link.nodes[0]] == model.nodes[link.nodes[1]] or frozenset(link.nodes) in to_ground
        if link.kind == "RIGID" and here and link.nodes[0] != link.nodes[1]:
            link = dataclasses.replace(link, kind="GEN", stiffness=six)          # no arm between two points at one place
            tied.append(link.nodes)
        elif link.kind == "GEN" and any(link.rigid):
            if all(link.rigid) and not here:
                link = dataclasses.replace(link, kind="RIGID")
            else:
                link = dataclasses.replace(link, stiffness=tuple(
                    r_ if r else c for c, r, r_ in zip(link.stiffness, link.rigid, six)))
                if all(link.rigid):
                    tied.append(link.nodes)
                notes.append(f"elastic links rigid in some directions: {link_rigid:g} N/m (N m/rad) there")
        all_links.append(link)
    rigid = []
    for master, dofs, slaves, g in model.rigid_links:
        flags = dofs.strip().ljust(6, "0")[:6]
        if set(flags) - {"0", "1"} or "1" not in flags:
            leave(f"*RIGIDLINK with flags {dofs.strip()!r} (six 0/1, at least one 1)", len(slaves))
            continue
        rigid.append((master, flags, [sl for sl in slaves if sl != master], g))
    # TWO NODES AT ONE PLACE, JOINED BY A LINK (a bearing drawn at a point),
    # ARE SET A MILLIMETRE APART: the mesh has no two nodes at one place and
    # no member without a length. The one fewer things hold moves, along X,
    # to the first free place from the other. THE LINKS' AXES STAY THOSE OF
    # THE MODEL AS IT WAS WRITTEN (`drawn`): a link at one place has the
    # global axes for its own in Midas, and a bearing under a node that has
    # moved a millimetre is still vertical (found in review: it leaned, and
    # a sliding bearing took 2% of its reaction sideways). Counted.
    drawn = {n: tuple(c * length for c in v) for n, v in model.nodes.items()}
    place = dict(model.nodes)
    taken = set(place.values())
    uses: Dict[int, int] = {}
    for el in model.elements.values():
        for n in el.nodes:
            uses[n] = uses.get(n, 0) + 2
    for link in all_links:
        for n in link.nodes:
            uses[n] = uses.get(n, 0) + 1
    apart = 0
    crowd: Dict[tuple, int] = {}                     # how many nodes stand at a place
    for v in place.values():
        crowd[v] = crowd.get(v, 0) + 1
    for link in all_links:
        a_, b_ = link.nodes
        if link.kind != "GEN" or a_ == b_ or model.nodes[a_] != model.nodes[b_]:
            continue
        # whichever of the two still shares its place moves (an earlier link may have parted them and left one
        # of them on a third node); of two that do, the one fewer things hold
        shared = [n for n in (b_, a_) if crowd[place[n]] > 1]
        if not shared:
            continue
        mv = min(shared, key=lambda n: uses.get(n, 0))
        at = place[b_ if mv == a_ else a_]
        while at in taken:
            at = (at[0] + 1.0e-3 / length,) + tuple(at[1:])
        crowd[place[mv]] -= 1
        place[mv] = at
        crowd[at] = 1
        taken.add(at)
        apart += 1
    if apart:
        model = dataclasses.replace(model, nodes=place)
        notes.append(f"{apart} link(s) between two nodes at one place: one of the two is set 1 mm along X from the other")
    return model, all_links, tied, drawn, six, rigid
