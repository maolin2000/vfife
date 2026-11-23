"""Writing the case, the mesh and the map (see the package's docstring)."""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

import yaml


from .. import midas
from .geometry import _free_axis, _key, _local_y, _plain, _to_global, _up_of
from .prepare import prepare
from .shapes import G, _poisson, _weight, section_properties
from .tendons import tendon_specs


#: the clock a timed translation starts at, days: a relaxation reads only
#: the time since each tendon's lock-off, so its value does not matter yet
AGE0 = 28.0

class _FlowDumper(yaml.SafeDumper):
    """SafeDumper with the lists of scalars, and the mappings of scalars and
    such lists (a section, an assignment, a tendon's line), in flow style:
    one line each. Every float is still written as it is (repr)."""


def _scalar(v):
    return isinstance(v, (int, float, str, bool)) or v is None


def _flat(v):
    return _scalar(v) or (isinstance(v, list) and all(_scalar(q) for q in v))


def _flow_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=all(_scalar(v) for v in data))


def _flow_dict(dumper, data):
    flow = bool(data) and all(_flat(v) for v in data.values())
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items(), flow_style=flow)


_FlowDumper.add_representer(list, _flow_list)
_FlowDumper.add_representer(dict, _flow_dict)


def _shorten_tendons(entries: List[dict], folder: str, name: str) -> Tuple[dict, List[dict]]:
    """(tendon_defaults, the entries without them), and the profiles in CSV
    files beside the case. A key (or a key of a mapping:
    `stress: {ends}`) is a default only when EVERY tendon has it, at the
    value most of them have: a tendon that lacked it would take it. A
    profile goes to `{name}_tendon_z.csv` (`_y`) when every tendon has one:
    the default `{table: ...}` reads each tendon's rows by its name."""
    import collections
    import csv
    entries = [dict(e) for e in entries]
    defaults: dict = {}
    if len(entries) < 2:
        return defaults, entries
    # (a name the table cannot give back as it is -- spaces round it are
    # stripped -- keeps its profiles in the case; the name column is last, so
    # a name starting with `#` is not a comment line)
    plain = all(isinstance(e["name"], str) and e["name"] == e["name"].strip() for e in entries)
    for key, col in (("profile_z", "z"), ("profile_y", "y")):
        if not plain or not all(key in e for e in entries):
            continue
        fname = f"{name}_tendon_{col}.csv"
        with open(os.path.join(folder, fname), "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["x", col, "radius", "name"])
            for e in entries:
                for x, v, r in e.pop(key):
                    w.writerow([repr(float(x)), repr(float(v)), repr(float(r)), e["name"]])
        defaults[key] = {"table": fname}

    def mode(values):
        reprs = [yaml.safe_dump(v) for v in values]
        best, n = collections.Counter(reprs).most_common(1)[0]
        return (values[reprs.index(best)], best) if n >= 2 else (None, None)

    keys = [k for k in entries[0] if k not in ("name", "profile_y", "profile_z")]
    for k in keys:
        if not all(k in e for e in entries):
            continue
        if all(isinstance(e[k], dict) for e in entries):
            subs = [q for q in entries[0][k] if all(q in e[k] for e in entries)]
            for q in subs:
                v, key_ = mode([e[k][q] for e in entries])
                if key_ is None:
                    continue
                defaults.setdefault(k, {})[q] = v
                for e in entries:
                    if yaml.safe_dump(e[k][q]) == key_:
                        e[k] = {a: b for a, b in e[k].items() if a != q}
            for e in entries:
                if k in e and not e[k] and k in defaults:
                    del e[k]
        else:
            v, key_ = mode([e[k] for e in entries])
            if key_ is None:
                continue
            defaults[k] = v
            for e in entries:
                if yaml.safe_dump(e[k]) == key_:
                    del e[k]
    return defaults, entries


def _rule_offset_arms(case: dict, mesh, factor: float, limit: float) -> Tuple[dict, int, int]:
    """(the case with its offset arms as one `offset_arms` rule, arms in the
    rule, arms left as their own sections) -- v1.20. The rule works
    each arm out from the mesh as written (coordinates to 12 digits); an arm
    whose own section it does not give to 1e-9, or the orientation of -- a
    member set off that is a stub, or one end of it not set off -- keeps its
    section (renamed `offset_arm_own_K`: the rule's names are its own), and
    the rule is worked out again without it."""
    from ...prep import offsets as prep_offsets
    own = {a["region"]: a for a in case["assign"] if str(a.get("section", "")).startswith("offset_arm_")}
    if not own:
        return case, 0, 0
    ruled = list(own)
    while ruled:
        block = {"material": "rigid", "factor": factor, "limit": limit, "regions": ruled}
        probe = dict(case, offset_arms=block, assign=[a for a in case["assign"] if a["region"] not in set(ruled)])
        off: List[str] = []
        try:
            made = prep_offsets.arm_sections(probe, mesh, bad=off)
        except ValueError:
            return case, 0, len(own)
        for region, sec, orient in made:
            mine = case["sections"][own[region]["section"]]
            if list(own[region].get("orientation") or []) != orient or any(
                    abs(sec[q] - mine[q]) > 1.0e-9 * abs(mine[q]) for q in ("area", "iy", "iz", "j")):
                off.append(region)
        if not off:
            break
        ruled = [r for r in ruled if r not in set(off)]
    if not ruled:
        return case, 0, len(own)
    gone = set(ruled)
    kept = {a["section"] for a in case["assign"] if a["region"] in own and a["region"] not in gone}
    rename = {q: f"offset_arm_own_{i + 1}" for i, q in enumerate(sorted(kept, key=lambda q: int(q.rsplit("_", 1)[1])))}
    out = {}
    for k, v in case.items():
        if k == "sections":
            v = {rename.get(q, q): s_ for q, s_ in v.items()
                 if not (q.startswith("offset_arm_") and q not in kept)}
        elif k == "assign":
            v = [dict(a, section=rename.get(a["section"], a["section"])) for a in v if a["region"] not in gone]
        out[k] = v
        if k == "assign":
            out["offset_arms"] = {"material": "rigid", "factor": factor, "limit": limit, "regions": ruled}
    return out, len(ruled), len(own) - len(ruled)


def convert(model: midas.MidasModel, folder: str, name: str = "model", procedure: str = "static",
            rigid: float = 100.0, stiffest: float = 1.0e13, stub: float = 0.05, link_rotation: float = 1.0e9,
            link_rigid: float = 1.0e10, offset_rigid: float = 30.0, shear: bool = True,
            birth: str = "auto", jack_which: str = "one", stages: int = 0, cases: Tuple[str, ...] = (),
            expanded: bool = False) -> dict:
    """Write `<name>.msh`, `<name>.yaml` and `<name>.map.json` into `folder`;
    returns the report (also the map file's `report`). The tendons are the
    case's `tendons:` entries; `expanded` writes them as the solver reads
    them instead (what `vfife expand` gives)."""
    import dataclasses
    import yaml
    if stages:                       # the first stages only: what a later one builds is left out
        model = dataclasses.replace(model, stages=model.stages[:stages])
    f, length = model.si()
    os.makedirs(folder, exist_ok=True)
    left: Dict[str, int] = {}
    notes: List[str] = []

    def leave(what, n=1):
        left[what] = left.get(what, 0) + n

    model, all_links, tied, drawn, six = prepare(model, leave, notes, f, length, link_rigid, cases)

    # ---------------------------------------------------------------- nodes
    node_ids = sorted(model.nodes)
    index = {n: i + 1 for i, n in enumerate(node_ids)}                    # Midas id -> msh tag
    xyz = {n: tuple(c * length for c in model.nodes[n]) for n in node_ids}

    # ------------------------------------------- when things come and go
    # Each thing comes once and goes once: a member, a boundary group or a
    # load group that a later stage brings BACK is counted, not followed.
    born, gone = {}, {}
    for k, st in enumerate(model.stages):
        for g, _age in st.activate_elements:
            for e in model.groups[g][1]:
                if e in born and e in gone:
                    leave("member built again after its removal (built once, removed once)")
                born.setdefault(e, (k, g))
        for g, _r in st.deactivate_elements:
            for e in model.groups[g][1]:
                gone.setdefault(e, k)
    # A boundary group may come back: an arch foot is fixed for its first
    # segment, hinged while the rib is erected, and fixed again when the
    # hinge is closed. Its SUPPORTS follow every time it is on (`b_iv`, the
    # stages [on, off) of each); a link or a rigid arm of such a group is
    # built once and removed once (`b_on`, `b_off`), and counted.
    b_on, b_off = {}, {}
    b_iv: Dict[str, List[List[int]]] = {}
    # WHERE A SUPPORT THAT COMES LATER HOLDS ITS NODE is the stage's own word
    # (1.14): DEFORMED, where the node stands -- a plain
    # constraint, which holds a node that was free where it finds it;
    # ORIGINAL, at the node's coordinates -- the constraint is given
    # `value: 0`, which brings it there.
    b_pos: Dict[Tuple[str, int], str] = {}
    n_original = [0]
    never = 10 ** 9
    for k, st in enumerate(model.stages):
        for g in st.deactivate_boundaries:
            b_off.setdefault(g, k)
            if b_iv.get(g) and b_iv[g][-1][1] == never:
                b_iv[g][-1][1] = k
        for g, _pos in st.activate_boundaries:
            b_on.setdefault(g, k)
            if not b_iv.get(g) or b_iv[g][-1][1] != never:
                b_iv.setdefault(g, []).append([k, never])
                b_pos[(g, k)] = str(_pos).strip().upper()
    again = {g for g, iv in b_iv.items() if len(iv) > 1}
    l_on, l_off = {}, {}
    for k, st in enumerate(model.stages):
        for g, _d in st.activate_loads:
            if g in l_off:
                leave("load group activated again after its removal (not followed)")
            l_on.setdefault(g, k)
        for g, _d in st.deactivate_loads:
            l_off.setdefault(g, k)
    staged = bool(model.stages)
    tangent_stage: Dict[int, bool] = {}
    if model.tangent_group and model.tangent_group != "ALL" and model.tangent_group not in model.groups:
        notes.append(f"the tangent-assembly group {model.tangent_group!r} is no structure group of the model: "
                     "every stage builds at the design position")
    if model.tangent_group:
        tg = None if model.tangent_group == "ALL" else set(model.groups.get(model.tangent_group, ([], []))[1])
        for e, (k, _g) in born.items():
            if k > 0 and (tg is None or e in tg) and model.elements[e].kind == "BEAM":
                tangent_stage[k] = True

    # ------------------------------------------------------------ materials
    materials, sections = {}, {}
    for i, mat in model.materials.items():
        if mat.E is None:
            continue
        w, note = _weight(mat, f, length)
        if note:
            notes.append(f"material {i} ({mat.name.strip()}): {note}")
        materials[f"m{i}"] = {"type": "linear_elastic", "E": mat.E * f / length ** 2,
                              "nu": mat.nu if mat.nu is not None else _poisson(mat), "rho": w / G}
        if w <= 0.0:
            # weightless in the file (a dummy member); a density must be positive
            materials[f"m{i}"]["rho"] = 1.0
            notes.append(f"material {i} ({mat.name.strip()}): weightless in the file, taken as 1 kg/m3")
    # "Rigid" in a direct-stiffness program is a modulus 1e5 times steel's; here
    # the tangent is taken by differences of the forces, and the round-off of
    # such a member's force is larger than a cable's whole stiffness. A rigid
    # member is `rigid` times the model's ordinary modulus -- the model's own
    # stiffer materials are brought down to it, and said so.
    moduli = sorted(materials[f"m{el.material}"]["E"] for el in model.elements.values()
                    if f"m{el.material}" in materials) or [2.06e11]       # the ordinary one: the median member's
    e_rigid = rigid * moduli[len(moduli) // 2]
    for key, m in materials.items():
        if m["E"] > e_rigid:
            notes.append(f"material {key[1:]}: modulus {m['E']:.3g} Pa taken as {e_rigid:.3g} Pa (rigid = {rigid:g} x ordinary)")
            m["E"] = e_rigid
    materials["rigid"] = {"type": "linear_elastic", "E": e_rigid, "nu": 0.3, "rho": 1.0e-6}

    # -------------------------------------------------------------- members
    pretension = {}
    for case, e, t, g in model.pretension:
        if staged and g not in l_on:
            leave("pretension whose load group no stage activates")
            continue
        if staged and e in born and l_on[g] != born[e][0]:
            # stressed (or stressed again) in a stage other than the one the
            # cable is built in: here a cable has one designed length, from birth
            leave("pretension applied in another stage than the cable is built in (taken at its birth)")
        pretension[e] = pretension.get(e, 0.0) + t * f
    # WHICH OF A CABLE'S ELEMENTS THE JACK IS ON. Midas takes every pretension
    # as a force held in its stage, and a model gives the same one to each
    # element of a cable that runs over saddles -- the anchor cable, the
    # short piece between two saddles, the buckle cable. Their horizontal
    # parts do not balance at a sliding saddle (a tenth of the force is
    # ordinary), and in a geometrically linear solution nothing follows from
    # that. Here the saddle would slide until the geometry balanced it:
    # metres, on a spring of 0.1 kN/mm. So ONE element of such a run is
    # jacked to its force -- the one that reaches the newest part of the
    # structure, the rib segment being hung -- and the others are cut to the
    # length of theirs (`prestress`), and take what equilibrium gives them.
    # A run ends at a node a beam or a rigid link reaches, at one a support
    # holds in position (two stays from one anchor are two cables), and at
    # one where more than two line elements meet. Only an element with a
    # tension, built by some stage, is jacked; a compression (a negative
    # pretension) is a length, never a jack.
    line_kinds = ("TRUSS", "TENSTR", "COMPTR")
    beam_stages: Dict[int, List[int]] = {}
    for eid, el in model.elements.items():
        if el.kind == "BEAM" and (not staged or eid in born):
            for n in el.nodes:
                beam_stages.setdefault(n, []).append(born.get(eid, (0, ""))[0])
    for link in all_links:                    # a rigid link's arm is a beam too: a cable is often hung on one
        if link.kind == "RIGID" and (not staged or link.group in b_on):
            for n in link.nodes:
                beam_stages.setdefault(n, []).append(b_on.get(link.group, 0))
    pinned = {n for nodes, flags, g in model.constraints if "1" in flags[:3] and (not staged or g in b_on)
              for n in nodes}
    # ...and at one an elastic link holds in every direction (an anchorage:
    # a saddle's link has a direction it lets go)
    for link in all_links:
        if link.kind == "GEN" and len(link.stiffness) >= 3 and max(link.stiffness[:3]) > 0.0 \
                and min(link.stiffness[:3]) > 1.0e-3 * max(link.stiffness[:3]):
            pinned.update(link.nodes)
    lines_at: Dict[int, int] = {}
    for eid, el in model.elements.items():
        if el.kind in line_kinds:
            for n in el.nodes:
                lines_at[n] = lines_at.get(n, 0) + 1
    stressed = [eid for eid, t in pretension.items() if t > 0.0 and eid in model.elements
                and model.elements[eid].kind in line_kinds and (not staged or eid in born)]
    at_node: Dict[int, List[int]] = {}
    for eid in stressed:
        for n in model.elements[eid].nodes:
            at_node.setdefault(n, []).append(eid)
    run_of = {eid: eid for eid in stressed}

    def find(e):
        while run_of[e] != e:
            run_of[e] = run_of[run_of[e]]
            e = run_of[e]
        return e

    def stage_of(eid):
        return born.get(eid, (0, ""))[0]

    for n, es in at_node.items():
        if n not in beam_stages and n not in pinned and lines_at[n] == 2 and len(es) == 2 \
                and stage_of(es[0]) == stage_of(es[1]):
            run_of[find(es[0])] = find(es[1])
    runs: Dict[int, List[int]] = {}
    for eid in run_of:
        runs.setdefault(find(eid), []).append(eid)
    jacked = set()
    for es in runs.values():
        def reach(eid):
            a, b = model.elements[eid].nodes
            newest = max([q for n in (a, b) for q in beam_stages.get(n, ()) if q <= stage_of(eid)], default=-1)
            return (newest, math.dist(xyz[a], xyz[b]), -eid)
        if jack_which == "ends":
            # `jack_which="ends"`: every element of the run that reaches out
            # of it, the buckle cable AND the anchor cable, as Midas has them;
            # only what lies between two saddles is cut to length. Tried on
            # the real arch: the two jacks' horizontal parts then meet on the
            # saddles' soft springs, which slide a metre for a 400 kN cable
            # (and would five for a 2000 kN one). Not the default.
            joints = [n for n in at_node if sum(1 for e in at_node[n] if e in es) == 2 and len(es) > 1]
            outer = [e for e in es if not all(n in joints for n in model.elements[e].nodes)]
            jacked.update(outer or [max(es, key=reach)])
        else:
            jacked.add(max(es, key=reach))

    # A TAPERED SECTION'S PLACE ALONG ITS RUN (1.15). One tapered section
    # may run over a group of elements (`*TS-GROUP`): each element is then a
    # prismatic member with the properties at ITS mid-length along the run,
    # in the order the group lists them. On its own an element takes the
    # section's mid-length. A section key is the id, or (id, per-mille).
    taper_at: Dict[int, float] = {}
    taper_span: Dict[int, Tuple[float, float]] = {}
    for _name, run in getattr(model, "taper_groups", ()):
        run = [e for e in run if e in model.elements and len(model.elements[e].nodes) >= 2]
        lens = [math.dist(xyz[model.elements[e].nodes[0]], xyz[model.elements[e].nodes[1]]) for e in run]
        whole = sum(lens)
        if whole <= 0.0:
            continue
        done = 0.0
        for i_, (e, le) in enumerate(zip(run, lens)):
            taper_at[e] = (done + 0.5 * le) / whole
            # its I end is the run's near end unless it is drawn against the run (by the node it shares with its neighbour)
            ends_ = model.elements[e].nodes
            back = (i_ > 0 and ends_[1] in model.elements[run[i_ - 1]].nodes and ends_[0] not in model.elements[run[i_ - 1]].nodes) \
                or (i_ == 0 and len(run) > 1 and ends_[0] in model.elements[run[1]].nodes and ends_[1] not in model.elements[run[1]].nodes)
            taper_span[e] = ((done + le) / whole, done / whole) if back else (done / whole, (done + le) / whole)
            done += le

    def section_key(eid, sid):
        sec = model.sections.get(sid)
        if sec is not None and sec.kind == "TAPERED" and eid in taper_at:
            return (sid, int(round(1000 * taper_at[eid])))
        return sid

    props_cache: Dict[object, dict] = {}

    def props(skey):
        if skey not in props_cache:
            sid, t = (skey[0], skey[1] / 1000.0) if isinstance(skey, tuple) else (skey, 0.5)
            props_cache[skey] = section_properties(model.sections[sid], length, t)
            if "note" in props_cache[skey]:
                notes.append(f"section {sid} ({model.sections[sid].name.strip()}): {props_cache[skey]['note']}")
        return props_cache[skey]

    # A STUB IS A JOINT, NOT A BEAM. A member stiffer than `stiffest` (N/m, the
    # larger of EA/L and 12EI/L^3) AND shorter than `stub` times its
    # section's radius of gyration -- 2 mm of a girder between the ends of
    # two spans, 5 mm of a pipe -- is a gap between two points. The tangent
    # is taken by differences of the forces, whose round-off is some 4e-11
    # of the largest stiffness in the model, and such a member's 1e18 N/m
    # is more than a soft link's whole stiffness. It becomes a connector:
    # `stiffest` in each direction, and in turning what the section would
    # have over a length of `stub` radii.
    #   It used to be a beam with its section divided by a power of two.
    # Divided until 12EI/L^3 is under the limit, EI/L is the limit times
    # L^2/12 -- 3 N m/rad for 2 mm: A HINGE, and its EA/L next to nothing.
    # The two spans of a real curved bridge met in such a stub and came
    # apart: the far span turned in plan, 50 mm where Midas has 0.2. (And a
    # softened member's weight went with its area; and most members over
    # the limit were lengths of girder, not stubs -- a metre of a steel box
    # has 4e13 N/m sideways and softened by 8 the bridge deflected 8 times
    # too much. Those are left as they are, and said.)
    softened = [0]
    stiff_left = [0, 0.0]
    stubs: List[tuple] = []                          # (a, b, stage on, group, stage off, turning stiffness)
    stub_weight = [0.0]

    def is_stub(E, p, a, b):
        span = math.dist(xyz[a], xyz[b])
        k = max(E * p["area"] / span, 12.0 * E * max(p["iy"], p["iz"]) / span ** 3)
        if k <= stiffest:
            return 0.0
        gyration = math.sqrt(max(p["iy"], p["iz"]) / p["area"]) if p["area"] > 0.0 else 0.0
        if span >= stub * gyration:
            stiff_left[0] += 1
            stiff_left[1] = max(stiff_left[1], k)
            return 0.0
        softened[0] += 1
        return E * max(p["iy"], p["iz"]) / (stub * gyration)

    groups: Dict[tuple, List[Tuple[int, int, int]]] = {}                   # key -> [(a, b, midas id)]
    sprung = {frozenset(link.nodes) for link in all_links if link.kind != "RIGID"}    # node pairs an elastic link joins
    set_off = [0]
    offset_arms: Dict[Tuple[int, int], List] = {}                          # (model node, centroid node) -> [on, off, stiffnesses carried]
    centroids_at: Dict[int, List[Tuple[tuple, float, int]]] = {}           # model node -> [(place, depth, centroid node)]
    place_of = {tuple(round(c, 6) for c in v): n for n, v in xyz.items()}
    next_node = [max(node_ids) + 1]

    def centroid_node(n, vec, k_on, k_off, carried):
        """The node `vec` from model node n, and its arm's life. ONE for the
        members that meet at n with the same depth: on a grade or a vertical
        curve two members' own up differ by a hair, and two nodes a hair
        apart are two arms in series at every joint (found in review). A
        model node that is already there -- a pier's top drawn at the
        girder's centroid -- is the centroid node."""
        depth = math.sqrt(sum(c * c for c in vec))
        if depth < 1.0e-6:
            return n
        at = tuple(xyz[n][i] + vec[i] for i in range(3))
        c = next((c_ for p_, d_, c_ in centroids_at.get(n, ()) if math.dist(p_, at) <= 0.02 * max(depth, d_)), None)
        if c is None:
            c = place_of.get(tuple(round(q, 6) for q in at))
            if c is None:
                c = next_node[0]
                next_node[0] += 1
                node_ids.append(c)
                index[c] = len(node_ids)
                xyz[c] = at
                place_of[tuple(round(q, 6) for q in at)] = c
            centroids_at.setdefault(n, []).append((xyz[c], depth, c))
        life = offset_arms.setdefault((n, c), [k_on, k_off, 0.0, 0.0])
        life[0] = min(life[0], k_on)
        life[1] = -1 if life[1] < 0 or k_off < 0 else max(life[1], k_off)
        life[2], life[3] = max(life[2], carried[0]), max(life[3], carried[1])
        return c

    pairs = set()
    plates: List[int] = []
    for eid in sorted(model.elements):
        el = model.elements[eid]
        if el.kind == "PLATE" and el.section in model.thicknesses and f"m{el.material}" in materials \
                and (not staged or eid in born):
            plates.append(eid)                       # its weight goes on its nodes; see the loads
            continue
        if el.kind not in ("BEAM", "TRUSS", "TENSTR", "COMPTR"):
            leave(f"{el.kind} element")
            continue
        if staged and eid not in born:
            leave("member no stage activates")
            continue
        a, b = el.nodes
        if a == b or xyz[a] == xyz[b]:
            leave("zero-length member")
            continue
        if el.section not in model.sections:
            raise ValueError(f"element {eid} has section {el.section}, which the file does not define")
        if f"m{el.material}" not in materials:
            raise ValueError(f"element {eid} has material {el.material}, which the file does not define "
                             "with an elastic modulus")
        if el.kind in ("TENSTR", "COMPTR") and (min(a, b), max(a, b)) not in pairs:
            leave(f"{el.kind} taken as a bar that carries both tension and compression (kept)")
        k, g = born.get(eid, (0, ""))
        axial = el.kind != "BEAM"
        sec = model.sections[el.section]
        skey = section_key(eid, el.section)
        # A SECTION SET OFF ITS NODES (offset CT: the nodes are at the top of
        # the section, as a girder is drawn). The member is on its centroid's
        # line: a node of its own under each end, by the centroid's depth
        # there (a tapered girder's centroid line bends), joined to the
        # model's node by a rigid arm built and removed with the member.
        # Midas gives the displacement at the model's node, the top; that
        # is where the result's row for the node still is. Up and down
        # (T, B) is followed; a section set off sideways (L, R) is not.
        orient_set = None                            # a member set off keeps the axes of the line it was drawn on
        if not axial and sec.offset not in ("", "CC"):
            up = sec.offset[1:2]
            t0, t1 = taper_span.get(eid, (0.0, 1.0)) if sec.kind == "TAPERED" else (0.5, 0.5)
            try:
                c0, c1 = section_properties(sec, length, t0), section_properties(sec, length, t1)
            except ValueError:
                c0 = c1 = {}
            if (min(a, b), max(a, b)) in pairs and not (up in ("T", "B") and "czp" in c0):
                pass                                 # (left out below, as a second member on the two nodes)
            elif up in ("T", "B") and "czp" in c0 and "czp" in c1 and min(c0["czp"] + c0["czm"], c1["czp"] + c1["czm"]) > 0.0 \
                    and any(frozenset((n, place_of.get(tuple(round(xyz[n][i] + d_ * z0[i], 6) for i in range(3)), -1))) in sprung
                            for z0 in [_up_of(xyz[a], xyz[b], el.angle)]
                            for n, d_ in ((a, -c0["czp"] if up == "T" else c0["czm"]), (b, -c1["czp"] if up == "T" else c1["czm"]))):
                # the centroid is where a node of the model is, and an elastic link joins the two: on that node
                # the member would be on the far side of its own bearing
                leave(f"section offset from the centroid ({sec.offset}) not followed: a bearing's other node is at the centroid")
            elif up in ("T", "B") and "czp" in c0 and "czp" in c1 and min(c0["czp"] + c0["czm"], c1["czp"] + c1["czm"]) > 0.0:
                x_ = [xyz[b][i] - xyz[a][i] for i in range(3)]
                n_ = math.sqrt(sum(c * c for c in x_))
                z_ = _up_of(xyz[a], xyz[b], el.angle)
                # what the member carries at an end: along (the larger of EA/L and 12EI/L^3) and in turning (4EI/L)
                e_m, p_m = materials[f"m{el.material}"]["E"], props(skey)
                # (in the plane of its depth, where the offset is: sideways a wide box is ten times stiffer,
                # and an arm sized by that is past what the arithmetic resolves)
                carried = (max(e_m * p_m["area"] / n_, 12.0 * e_m * p_m["iy"] / n_ ** 3), 4.0 * e_m * p_m["iy"] / n_)
                carried = (min(carried[0], stiffest), min(carried[1], stiffest * n_ * n_ / 3.0))   # (a stub is a joint of that)
                ends = []
                for n, c_ in ((a, c0), (b, c1)):
                    d_ = -c_["czp"] if up == "T" else c_["czm"]
                    ends.append(centroid_node(n, [d_ * c for c in z_], k, gone.get(eid, -1), carried))
                round_ = sec.shape in ("P", "SR")
                orient_set = _free_axis(xyz[a], xyz[b]) if round_ else _key(_local_y(xyz[a], xyz[b], el.angle))
                a, b = ends
                set_off[0] += 1
                if sec.offset[:1] in ("L", "R"):
                    leave(f"section set off sideways ({sec.offset}): only its height is followed")
            else:
                leave(f"section offset from the centroid ({sec.offset}) not followed: the member is on its nodes' line")
        pair = (min(a, b), max(a, b))
        if pair in pairs or a == b:
            # the Gmsh reader would merge the two, and one would be lost
            # without a word: a pipe and its concrete core as two elements
            leave("member on the same two nodes as another (the first is kept)")
            continue
        pairs.add(pair)
        round_ = sec.shape in ("P", "SR") or (sec.kind == "SRC" and sec.shape == "EPC")
        orient = None if axial else orient_set if orient_set is not None else _free_axis(xyz[a], xyz[b]) if round_ \
            else _key(_local_y(xyz[a], xyz[b], el.angle))
        soft = 0
        if not axial:
            turning = is_stub(materials[f"m{el.material}"]["E"], props(skey), a, b)
            if turning > 0.0:
                stubs.append((a, b, k, g, gone.get(eid, -1), turning))
                stub_weight[0] += _weight(model.materials[el.material], f, length)[0] * props(skey)["area"] * math.dist(xyz[a], xyz[b])
                continue
        key = ("bar" if axial else "beam", k, g, skey, el.material, orient,
               (("jack" if eid in jacked else "cut", round(pretension[eid], 3))
                if axial and pretension.get(eid) else 0.0), gone.get(eid, -1), soft)
        groups.setdefault(key, []).append((a, b, eid))

    # the tendons, as the case's own entries: expanded where it is read (vfife/prep)
    specs = tendon_specs(model, materials, staged, l_on, l_off, born, xyz, length, f, leave, notes)

    # ---------------------------------------------------------------- links
    # A NODE NO BEAM HOLDS TAKES ITS ROTATION FROM ITS LINK. A Midas elastic
    # link is a set of springs between two points; the modeller's 0.01 of
    # rotational stiffness means "none", and nothing depends on it there. A
    # connector here has a length and follows its ends' rotations: a saddle
    # node that only cables and links reach would turn freely, and a link in
    # compression then leans over like a strut on two pins. Such a link's
    # rotational stiffnesses are raised to `link_rotation` (N m/rad): the
    # saddle turns with the tower it sits on. Counted. Only where the node's
    # rotation is really free: a node on a beam, on a rigid link's arm, or
    # with its three rotations held by a support has one of its own, and a
    # bearing between a girder and such a node stays the hinge it was
    # written as (found in review: ordinary bearings became moment joints).
    beam_nodes = {n for key, members in groups.items() if key[0] == "beam" for a, b, _e in members for n in (a, b)}
    beam_nodes |= {n for link in all_links if link.kind == "RIGID" for n in link.nodes}
    beam_nodes |= {n for nodes, flags, _g in model.constraints if flags[3:6] == "111" for n in nodes}
    for _pass in range(len(tied)):                 # ...and one made one with such a node by six stiff springs
        more = {n for pair in tied if (pair[0] in beam_nodes) != (pair[1] in beam_nodes) for n in pair}
        if not more:
            break
        beam_nodes |= more
    beam_nodes |= {n for pair in offset_arms for n in pair}
    beam_nodes |= {n for a_, b_, *_rest in stubs for n in (a_, b_)}       # a stub turns its nodes as a beam does
    stiffened = [0]
    link_groups: Dict[tuple, List[Tuple[int, int, int]]] = {}
    # AN OFFSET'S ARM IS `offset_rigid` TIMES WHAT IT CARRIES, along and in
    # turning: the girder's moment passes through it wherever the girder is
    # held or joined at its drawn node (a pier fixed to it, a clamp), and an
    # arm of the common stiffness is no stiffer in turning than a few metres
    # of a 7 m box (found in review: a cantilever clamped at its top node
    # deflected 12% too much). Its section is its own (`soft` holds its name).
    # WHAT EACH NODE'S MEMBERS CARRY (along, in turning), for the arms: an arm
    # is to be stiffer than what it joins, and a girder's root is stiffer in
    # turning than an arm of the common stiffness a metre or two long.
    carry: Dict[int, List[float]] = {}
    for key_, members_ in groups.items():
        if key_[0] != "beam":
            continue
        e_m, p_m = materials[f"m{key_[4]}"]["E"], props(key_[3])
        for a_, b_, _e in members_:
            span_ = math.dist(xyz[a_], xyz[b_])
            i_m = max(p_m["iy"], p_m["iz"])
            got = (min(max(e_m * p_m["area"] / span_, 12.0 * e_m * i_m / span_ ** 3), stiffest),
                   min(4.0 * e_m * i_m / span_, stiffest * span_ * span_ / 3.0))
            for n_ in (a_, b_):
                c_ = carry.setdefault(n_, [0.0, 0.0])
                c_[0], c_[1] = max(c_[0], got[0]), max(c_[1], got[1])
    arms_ = list(offset_arms) + [link.nodes for link in all_links if link.kind == "RIGID"]
    for _pass in range(len(arms_) + 1):              # ...and through an arm, what is at its other end
        changed = False
        for a_, b_ in arms_:
            both = [max(carry.get(a_, [0.0, 0.0])[i], carry.get(b_, [0.0, 0.0])[i]) for i in range(2)]
            if any(both) and (carry.get(a_) != both or carry.get(b_) != both):
                carry[a_], carry[b_] = list(both), list(both)
                changed = True
        if not changed:
            break
    for (n, c), (k_on, k_off, along, turning) in offset_arms.items():
        if (min(n, c), max(n, c)) in pairs:
            leave("section offset whose centroid is a node already joined to the member's node (no arm added)")
            continue
        reach_ = math.dist(xyz[n], xyz[c])
        i_arm = offset_rigid * max(turning * reach_, along * reach_ ** 3 / 12.0) / e_rigid
        sname = f"offset_arm_{len([q for q in sections if q.startswith('offset_arm_')]) + 1}"
        sections[sname] = {"type": "beam", "material": "rigid", "shape": "general",
                           "area": offset_rigid * along * reach_ / e_rigid, "iy": i_arm, "iz": i_arm, "j": 2.0 * i_arm}
        key = ("rigid", k_on, "(section offset)", 0, 0, _free_axis(xyz[n], xyz[c]), 0.0, k_off, sname)
        pairs.add((min(n, c), max(n, c)))
        link_groups.setdefault(key, []).append((n, c, -1))
    for a, b, k_on, g_, k_off, turning in stubs:
        # (its own kind: it does not wait for what it joins as an elastic link does -- it carries a moment, and
        # what hangs on its far node hangs on it from the stage it is built)
        key = ("stub", k_on, g_ or "(stub)", (stiffest,) * 3 + (turning,) * 3, "global", None, 0.0, k_off, 0)
        if k_on > 0 and (birth == "tangent" or (birth == "auto" and tangent_stage.get(k_on))):
            leave("stub built with a part placed along the tangent: the tangent does not pass through the joint")
        link_groups.setdefault(key, []).append((a, b, -1))
    link_at: Dict[Tuple[int, int], tuple] = {}                           # the link on two nodes: its key
    for n, link in enumerate(all_links):
        a, b = link.nodes
        k = b_on.get(link.group, 0)
        if link.group in again:
            leave("link of a boundary group that is activated again after its removal (built once, removed once)")
        if staged and link.group not in b_on:
            leave("link whose boundary group no stage activates")
            continue
        if link.kind == "RIGID":
            if xyz[a] == xyz[b]:
                leave("rigid link between coincident nodes")
                continue
            reach_ = math.dist(xyz[a], xyz[b])
            soft_ = int(round(2.0 * math.log2(reach_)))                    # its length, to half a power of two
            along_, turning_ = (max(carry.get(a, [0.0, 0.0])[i], carry.get(b, [0.0, 0.0])[i]) for i in range(2))
            if offset_rigid * turning_ * reach_ > (stiffest / 2.0) * (2.0 ** (soft_ / 2.0)) ** 3 / 12.0:
                # a section of its own: `offset_rigid` times what its ends' members carry in turning (a pier
                # fixed to a girder: the common arm was softer than the girder's root, and a real rigid
                # frame's cantilevers 6% too soft for it)
                # (never more than 10 times the limit across, though: past that the Newton iteration
                # stalls on the round-off -- found in review, on arms of a few centimetres)
                k_al = min(max(stiffest / 2.0, offset_rigid * along_), 10.0 * stiffest)
                i_arm = min(max(offset_rigid * turning_ * reach_, k_al * reach_ ** 3 / 12.0),
                            10.0 * stiffest * reach_ ** 3 / 12.0) / e_rigid
                own = {"type": "beam", "material": "rigid", "shape": "general",
                       "area": float(f"{k_al * reach_ / e_rigid:.4g}"), "iy": float(f"{i_arm:.4g}"),
                       "iz": float(f"{i_arm:.4g}"), "j": float(f"{2.0 * i_arm:.4g}")}
                soft_ = next((q for q, v in sections.items() if q.startswith("rigid_arm_own_") and v == own), None)
                if soft_ is None:
                    soft_ = f"rigid_arm_own_{len([q for q in sections if q.startswith('rigid_arm_own_')]) + 1}"
                    sections[soft_] = own
            key = ("rigid", k, link.group, 0, 0, _free_axis(xyz[a], xyz[b]), 0.0, b_off.get(link.group, -1), soft_)
        elif link.kind == "GEN":
            stiff = tuple(link.stiffness[i] * f / length for i in range(3)) + \
                tuple(link.stiffness[i] * f * length for i in range(3, 6))
            if (a not in beam_nodes or b not in beam_nodes) and min(stiff[3:]) < link_rotation:
                stiff = stiff[:3] + tuple(max(c, link_rotation) for c in stiff[3:])
                stiffened[0] += 1
            # ITS SPRINGS STAY WHERE MIDAS PUT THEM: in the link's axes as they
            # are in the model, not turning with its ends. Springs that turn
            # make the product of a slide and a rotation an elongation -- 3 cm
            # along a saddle's soft direction and a thousandth of a radian
            # were 2e7 N in its 1e12 N/m one, and the Newton iteration never
            # recovered. So the stiffness is written in GLOBAL axes: a
            # diagonal when the link lies along one, else the full matrix.
            if drawn[a] != drawn[b]:
                stiff = _to_global(stiff, drawn[a], drawn[b], link.angle)
            else:                                    # at one place: the global axes, turned by its angle about X
                stiff = _to_global(stiff, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), link.angle)
            key = ("link", k, link.group, stiff, "global", None, 0.0, b_off.get(link.group, -1), 0)
        else:
            leave(f"{link.kind} link")
            continue
        pair = (min(a, b), max(a, b))
        if pair in link_at:
            # THE SAME LINK IN A LATER BOUNDARY GROUP IS THE SAME LINK, KEPT: a
            # closure's bearings are written in the group of the side span's
            # closure and again in the finished bridge's, which comes on in
            # the stage the first goes off. One link, from the first's on to
            # the last's off. A DIFFERENT link on the two nodes (a bearing
            # that is fixed, then slides) cannot be followed -- the mesh has
            # one line between two nodes -- and the first is kept, counted.
            old = link_at[pair]
            same = old[0] == key[0] and old[3:7] == key[3:7]
            # hand-over only: one comes on in the stage the other goes off
            # (in either order of writing). Two that are on together are two
            # springs side by side in Midas, and one here: counted.
            after, before = old[7] >= 0 and key[1] == old[7], key[7] >= 0 and old[1] == key[7]
            if same and (after or before):
                off = (key[7] if after else old[7])
                merged = old[:1] + (min(old[1], key[1]),) + old[2:7] + (off,) + old[8:]
                link_groups[old] = [q for q in link_groups[old] if (min(q[0], q[1]), max(q[0], q[1])) != pair]
                link_groups.setdefault(merged, []).append((a, b, n))
                link_at[pair] = merged
                continue
            if not same:
                leave("a different link on the two nodes of an earlier one (a bearing that changes; the first is kept)")
            elif (old[7] < 0 or key[1] < old[7]) and (key[7] < 0 or old[1] < key[7]):
                leave("identical link on the two nodes of another while that one is on (one kept; Midas adds the two)")
            else:
                leave("link on the same two nodes as another, built again after a gap (the first is kept)")
            continue
        if pair in pairs:
            leave("link on the same two nodes as a member (the member is kept)")
            continue
        pairs.add(pair)
        link_at[pair] = key
        link_groups.setdefault(key, []).append((a, b, n))
    link_groups = {k_: v for k_, v in link_groups.items() if v}
    # A LINK WAITS FOR WHAT IT JOINS. A saddle's links are in the first
    # stage's boundary group, the cable they will carry is built stages
    # later; until then their far node would hang on them with the 0.01 of
    # rotational stiffness the modeller gave it, and swing. So a link is
    # built in the first stage, from its group's on, in which each of its
    # ends is held by something -- a member, a rigid arm, a support, or a
    # second link that leads to one -- and born free of force there.
    # One that never is, is left out and counted.
    n_stages = max(len(model.stages), 1)
    every = [(key, i) for key, members in link_groups.items() if key[0] == "link" for i in range(len(members))]
    ends = {ki: link_groups[ki[0]][ki[1]][:2] for ki in every}
    born_at: Dict[tuple, int] = {}
    alive = set()
    for stage in range(n_stages):
        for key, members in list(groups.items()) + [kv for kv in link_groups.items() if kv[0][0] in ("rigid", "stub")]:
            if key[1] == stage:
                alive.update(n for a, b, _e in members for n in (a, b))
        for nodes, _flags, g in model.constraints:
            if b_on.get(g, 0 if not staged else -1) == stage:
                alive.update(nodes)
        kept = set(born_at) | {ki for ki in every if ki not in born_at and ki[0][1] <= stage
                                and not (0 <= ki[0][7] <= stage)}
        while True:
            degree: Dict[int, int] = {}
            for ki in kept:
                for n in ends[ki]:
                    degree[n] = degree.get(n, 0) + 1
            drop = {ki for ki in kept if ki not in born_at
                    and any(n not in alive and degree[n] < 2 for n in ends[ki])}
            if not drop:
                break
            kept -= drop
        for ki in kept:
            born_at.setdefault(ki, stage)
    regrouped: Dict[tuple, List[Tuple[int, int, int]]] = {}
    for ki in every:
        key, i = ki
        if ki not in born_at:
            leave("elastic link whose two ends are never both held while its boundary group is on (left out)")
            continue
        regrouped.setdefault(key[:1] + (born_at[ki],) + key[2:], []).append(link_groups[key][i])
    link_groups = {k_: v for k_, v in link_groups.items() if k_[0] != "link"}
    link_groups.update(regrouped)

    # ---------------------------------------------- regions, sections, assign
    names: List[Tuple[int, str]] = []                                     # (dim, name), msh physical tags in order
    lines_out: List[Tuple[int, int, int]] = []                            # (physical tag, a, b)
    region_info, assign, element_map = {}, [], {}
    activate: Dict[int, List[str]] = {}
    deactivate: Dict[int, List[str]] = {}
    total = dict(groups)
    total.update(link_groups)
    for key in sorted(total, key=lambda q: (q[1], str(q))):
        kind, k, g, sid, mid, orient, jack, off, soft = key
        members = total[key]
        region = f"E{len(names) + 1}"
        names.append((1, region))
        tag = len(names)
        for row, (a, b, eid) in enumerate(members):
            lines_out.append((tag, index[a], index[b]))
            if kind in ("beam", "bar") and eid > 0:
                element_map[eid] = [region, row]
        info = {"kind": kind, "stage": k, "midas_group": g, "members": len(members)}
        sid_of = sid[0] if isinstance(sid, tuple) else sid
        if kind == "beam":
            sname = (f"s{sid[0]}_t{sid[1]}" if isinstance(sid, tuple) else f"s{sid}") + f"_m{mid}"
            if sname not in sections:
                p = props(sid)
                sections[sname] = {"type": "beam", "material": f"m{mid}", "shape": "general"}
                sections[sname].update({q: p[q] for q in ("area", "iy", "iz", "j")})
                if shear and p.get("ay", 0.0) > 0.0 and p.get("az", 0.0) > 0.0:
                    # Midas' beam gives way in shear too (its ASy, ASz); so does this one where they are known
                    sections[sname].update(theory="timoshenko", ay=p["ay"], az=p["az"])
            assign.append({"region": region, "section": sname, "orientation": list(orient)})
            info.update(section=sid_of, material=mid)
        elif kind == "bar":
            tendon_ = isinstance(sid, tuple) and sid[0] == "tdn"
            area_ = sid[1] if tendon_ else props(sid)["area"]
            sname = (f"tendon_{area_:.6g}" if tendon_ else f"c{sid[0]}_t{sid[1]}" if isinstance(sid, tuple) else f"c{sid}") + f"_m{mid}"
            sections.setdefault(sname, {"type": "bar", "material": f"m{mid}", "shape": "general", "area": area_})
            entry = {"region": region, "section": sname}
            if jack and jack[0] == "jack":
                entry["jack"] = jack[1]                    # the force, held in the step that builds it
            elif jack:
                entry["prestress"] = jack[1] / area_  # the length that force gives in the model's geometry
            assign.append(entry)
            info.update(section=sid_of, material=mid, jack=jack[1] if jack and jack[0] == "jack" else 0.0,
                        prestress_force=jack[1] if jack and jack[0] == "cut" else 0.0)
        elif kind == "rigid":
            # AN ARM IS AS STIFF AS `stiffest` / 2 WHATEVER ITS LENGTH, along and
            # across: its area goes with its length and its second moment with
            # the cube (to half a power of two of the length, `soft` here). It
            # was one section for all, a half-metre round: 4 m of that from a
            # pile cap's node to its piles bends at 5e8 N/m, the pile under it
            # carries 2e9, and the cap's weight went down the nearest pile.
            sname = soft if isinstance(soft, str) else f"rigid_arm_{soft}".replace("-", "m")
            if sname not in sections:
                reach_ = 2.0 ** (soft / 2.0)
                k_arm = stiffest / 2.0                # (to its bin's length: 0.42 to 0.84 of the limit at its own)
                i_arm = k_arm * reach_ ** 3 / (12.0 * e_rigid)
                sections[sname] = {"type": "beam", "material": "rigid", "shape": "general",
                                   "area": k_arm * reach_ / e_rigid, "iy": i_arm, "iz": i_arm, "j": 2.0 * i_arm}
            assign.append({"region": region, "section": sname, "orientation": list(orient)})
        else:
            sname = next((q for q, v in sections.items() if v.get("type") == "connector"
                          and v["stiffness"] == _plain(sid)), None)
            if sname is None:
                sname = f"k{len([q for q in sections if q.startswith('k')]) + 1}"
                sections[sname] = {"type": "connector", "stiffness": _plain(sid)}
            entry = {"region": region, "section": sname, "coordinate_system": mid}
            if orient is not None:
                entry["orientation"] = list(orient)
            assign.append(entry)
        region_info[region] = info
        if k > 0:
            activate.setdefault(k, []).append(region)
        if off >= 0:
            deactivate.setdefault(off, []).append(region)

    # ------------------------------------------------------------- supports
    points_out: List[Tuple[int, int]] = []

    def node_region(nodes, what):
        region = f"N{len(names) + 1}"
        names.append((0, region))
        tag = len(names)
        points_out.extend((tag, index[n]) for n in nodes)
        region_info[region] = {"kind": what, "nodes": len(nodes)}
        return region

    used_nodes = {index_ for _t, a, b in lines_out for index_ in (a, b)}
    constraints, step_constraints, release = [], {}, {}
    held_by: Dict[int, List[Tuple[str, str]]] = {}                        # node -> [(boundary group, flags)]
    for nodes, flags, g in model.constraints:
        if not staged or g in b_on:
            for n in nodes:
                held_by.setdefault(n, []).append((g, flags[:6].ljust(6, "0")))

    def on_through(h, k):
        """Is boundary group h on before stage k and still on in it?"""
        if not staged:
            return True
        return any(a < k < b for a, b in b_iv.get(h, ()))

    for nodes, flags, g in model.constraints:
        if staged and g not in b_on:
            leave("support whose boundary group no stage activates")
            continue
        lost = [n for n in nodes if index[n] not in used_nodes]
        if lost:
            leave("support on a node no member reaches", len(lost))
        nodes = [n for n in nodes if index[n] in used_nodes]
        if not nodes:
            continue
        region = node_region(nodes, f"support {g}")
        flags = flags[:6].ljust(6, "0")
        c = {"region": region, "dofs": [int(ch) for ch in flags]}
        for k, k_off in (b_iv.get(g) or [[0, never]]):
            if k == 0:
                constraints.append(c)
            elif b_pos.get((g, k)) == "ORIGINAL":
                step_constraints.setdefault(k, []).append(dict(c, keep=True, value=0.0))
                n_original[0] += len(nodes)
            else:
                step_constraints.setdefault(k, []).append(dict(c, keep=True))
            if k_off == never:
                continue
            # A RELEASE FREES THE NODE, whatever else holds it (a step's
            # `release` takes every inherited constraint off). What another
            # boundary group, activated earlier and still there, holds on the
            # node is given again as that step's own constraint, kept: a
            # fixed foot whose temporary fixity goes stays pinned. (A group
            # that comes on in the same stage writes its own.)
            release.setdefault(k_off, []).append(region)
            again_held: Dict[str, List[int]] = {}
            for n in nodes:
                others = [fl for h, fl in held_by[n] if h != g and on_through(h, k_off)]
                if others:
                    union = "".join("1" if any(fl[i] == "1" for fl in others) else "0" for i in range(6))
                    again_held.setdefault(union, []).append(n)
            for union, held in again_held.items():
                step_constraints.setdefault(k_off, []).append(
                    {"region": node_region(held, f"still held after the release of {g}"),
                     "dofs": [int(ch) for ch in union], "keep": True})

    # A PART ON PINS IN ONE LINE SPINS ABOUT THAT LINE. A side span cast on
    # falsework is a few beams in a line, held at its nodes in the three
    # translations: nothing holds its twist. Midas solves it; the Newton
    # iteration here has a direction with no stiffness and stops. In every
    # stage, each part -- what is joined by the members, arms and links
    # standing in that stage -- that has no rotation held anywhere and
    # whose supported nodes are all on one line (within a tenth of a global
    # axis) gets the rotation about that axis held at one of them, AS THAT
    # STAGE'S OWN CONSTRAINT: not kept, so it is gone the stage the part is
    # joined to something or a rotation is held (found in review: kept with
    # its support's group it outlived the reason for it, and a girder with
    # a cross beam added later carried 3% too much on it). Counted, by part.
    spin_nodes: Dict[int, str] = {}
    spin_left = set()
    for k in range(max(len(model.stages), 1)):
        reach: Dict[int, List[int]] = {}
        bent = set()                                 # nodes of beams: a line of bars alone has no twist to hold
        for key, members in total.items():
            if key[1] <= k and not (0 <= key[7] <= k):
                if key[0] in ("link", "stub"):       # a link that carries no moment does not hold a twist: not walked
                    turn_ = [r_[3:] for r_ in key[3][3:]] if isinstance(key[3][0], tuple) else [key[3][3:]]
                    if not any(c for r_ in turn_ for c in r_):
                        continue
                for a, b, _e in members:
                    reach.setdefault(a, []).append(b)
                    reach.setdefault(b, []).append(a)
                    if key[0] == "beam":
                        bent.update((a, b))
        pinned_k, turned = set(), set()
        for n, hs in held_by.items():
            for h, fl in hs:
                if not staged or any(a <= k < b for a, b in b_iv.get(h, ())):
                    if "1" in fl[:3]:
                        pinned_k.add(n)
                    if "1" in fl[3:6]:
                        turned.add(n)
        seen = set()
        for n0 in sorted(pinned_k):
            if n0 in seen or n0 not in reach:
                continue
            part, stack = {n0}, [n0]
            while stack:
                for q in reach.get(stack.pop(), ()):
                    if q not in part:
                        part.add(q)
                        stack.append(q)
            seen |= part
            pins = sorted(part & pinned_k)
            if part & turned or not part & bent or len(pins) < 2:
                continue
            far = max(pins, key=lambda q: math.dist(xyz[q], xyz[n0]))
            line = [xyz[far][i] - xyz[n0][i] for i in range(3)]
            span = math.sqrt(sum(c * c for c in line))
            if span == 0.0:
                continue
            line = [c / span for c in line]

            def off_line(q):
                d = [xyz[q][i] - xyz[n0][i] for i in range(3)]
                t = sum(d[i] * line[i] for i in range(3))
                return math.sqrt(max(sum(c * c for c in d) - t * t, 0.0))
            if max(off_line(q) for q in pins) > 1.0e-6 * span:
                continue                             # supports off the line hold the twist
            axis = max(range(3), key=lambda i: abs(line[i]))
            if abs(line[axis]) < 0.995:              # more than a tenth off the axis
                spin_left.add(n0)
                continue
            if n0 not in spin_nodes:
                spin_nodes[n0] = node_region([n0], "twist of a part on pins in one line, held")
            c = {"region": spin_nodes[n0], "dofs": [0, 0, 0] + [int(i == axis) for i in range(3)]}
            if staged:
                step_constraints.setdefault(k, []).append(c)
            else:
                constraints.append(c)
    if spin_left:
        leave("a part on pins in one line, free to spin about it (the line is not along a global axis: not held)", len(spin_left))
    if spin_nodes:
        notes.append(f"{len(spin_nodes)} part(s) on pins in one line: the rotation about that line is held at one of "
                     "the pins while nothing else holds it (Midas solves such a part as it is)")

    # ---------------------------------------------------------------- loads
    # A load comes on in the stage that activates its group -- or, on a
    # member or a node that is built later than that, when it is built.
    node_born: Dict[int, int] = {}
    for key, members in total.items():
        for a, b, _e in members:
            for n in (a, b):
                node_born[n] = min(node_born.get(n, key[1]), key[1])
    per_node: Dict[Tuple[str, int, int], List[float]] = {}                # (load group, stage on, node) -> force, moment
    gravity_dir = [0.0, 0.0, 0.0]                     # the self weight's factors: which way a weight acts
    for _case, factors, g in model.self_weight:
        if not (staged and g and g not in l_on):
            gravity_dir = [gravity_dir[i] + factors[i] for i in range(3)]
    plate_weight = [0.0]
    plates_loaded = [0]

    def stage_on(g, built):
        """The stage a load of group g comes on, on something built in
        stage `built`; None when it never does."""
        if staged and g not in l_on:
            leave("load whose load group no stage activates")
            return None
        k = max(l_on.get(g, 0), built)
        if g in l_off and l_off[g] <= k:
            leave("load whose group is removed before what it loads is built")
            return None
        return k

    # A PLATE'S WEIGHT, on its nodes from the stage that builds it -- its
    # stiffness is left out (this is a model of lines), its weight is not: the
    # gusset plates at an arch foot were 4% of the first segment's moment
    # about its hinge, and the buckle cable's force is designed to balance
    # exactly that.
    for eid in plates:
        el = model.elements[eid]
        pts = [np_xyz for np_xyz in (xyz[n] for n in el.nodes)]
        area = 0.0
        for i in range(1, len(pts) - 1):
            u = [pts[i][q] - pts[0][q] for q in range(3)]
            v = [pts[i + 1][q] - pts[0][q] for q in range(3)]
            c = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            area += 0.5 * math.sqrt(sum(q * q for q in c))
        wd, _note = _weight(model.materials[el.material], f, length)
        W = wd * area * model.thicknesses[el.section] * length
        held = [n for n in el.nodes if n in node_born]
        if not any(gravity_dir):
            leave("PLATE element in a model with no self weight (no weight, no stiffness)")
            continue
        if not held or W <= 0.0:
            leave("PLATE element of no area, or on nodes no member reaches (no weight, no stiffness)")
            continue
        # each node's share from the stage the node stands in; off again where the plate is removed
        k0 = born.get(eid, (0, ""))[0]
        k_off = gone.get(eid, -1)
        group = "(plate weight)" if k_off < 0 else f"(plate weight until stage {k_off})"
        if k_off >= 0:
            l_off[group] = k_off
        put = 0
        for n in held:
            k = max(k0, node_born[n])
            if 0 <= k_off <= k:
                continue
            acc = per_node.setdefault((group, k, n), [0.0] * 6)
            for q in range(3):
                acc[q] += gravity_dir[q] * W / len(held)
            put += 1
        if put:
            plate_weight[0] += W * put / len(held)
            plates_loaded[0] += 1
        else:
            leave("PLATE element removed before its nodes are built")
    if plates_loaded[0]:
        leave("PLATE element's stiffness (its weight is on its nodes)", plates_loaded[0])
    for _case, n, v, g in model.nodal_loads:
        if n not in node_born:
            leave("load on a node no member reaches")
            continue
        k = stage_on(g, node_born[n])
        if k is None:
            continue
        acc = per_node.setdefault((g, k, n), [0.0] * 6)
        for i in range(3):
            acc[i] += v[i] * f
            acc[3 + i] += v[3 + i] * f * length
    for _case, e, direction, w1, w2, g in model.beam_loads:
        if e not in model.elements or model.elements[e].kind not in ("BEAM", "TRUSS", "TENSTR", "COMPTR"):
            leave("member load on something that is not a line member")
            continue
        if e not in element_map:
            leave("member load on a member that was left out")
            continue
        if direction not in ("GX", "GY", "GZ"):
            leave(f"member load in direction {direction}")
            continue
        k = stage_on(g, born.get(e, (0, ""))[0])
        if k is None:
            continue
        if e in gone and gone[e] <= k:
            leave("member load on a member removed before the load comes")
            continue
        a, b = model.elements[e].nodes
        span = math.dist(xyz[a], xyz[b])
        axis = "XYZ".index(direction[1])
        # its two end forces, no end moments; over the member's whole length
        # and per unit of that length (a projected or a partial load is read so too)
        half = 0.5 * (w1 + w2) * f / length * span / 2.0
        for n in (a, b):
            per_node.setdefault((g, k, n), [0.0] * 6)[axis] += half
    step_loads: Dict[int, List[dict]] = {}
    base_loads = []
    by_group: Dict[Tuple[str, int], Dict[tuple, List[int]]] = {}
    for (g, k, n), v in per_node.items():
        by_group.setdefault((g, k), {}).setdefault(tuple(round(c, 6) for c in v), []).append(n)
    for (g, k), sets in by_group.items():
        for v, nodes in sets.items():
            if not any(v):
                continue
            # a force and a moment on one node are two loads here (a load is
            # one kind of thing): on the same nodes, on and off together
            region = node_region(nodes, f"load {g}")
            for key, part in (("vector", v[:3]), ("moment", v[3:])):
                if not any(part):
                    continue
                load = {"region": region, "per_node": True, key: list(part)}
                if k == 0 and g not in l_off:
                    base_loads.append(load)
                else:
                    step_loads.setdefault(k, []).append(dict(load, keep=True))
                    if g in l_off:
                        step_loads.setdefault(l_off[g], []).append(dict(load, keep=True, **{key: [0.0 - c for c in part]}))
    # Self weight is the model's gravity, on everything from the moment it is
    # built. A self-weight group that comes on late or goes off is not followed.
    gravity = [0.0, 0.0, 0.0]
    for _case, factors, g in model.self_weight:
        if staged and g and g not in l_on:
            leave("self weight whose load group no stage activates")
            continue
        if staged and g and (l_on[g] > 0 or g in l_off):
            leave("self weight that comes on after the first stage or goes off (taken as on throughout)")
        gravity = [gravity[i] + factors[i] * G for i in range(3)]
    if any(gravity):
        base_loads.insert(0, {"region": "all", "gravity": gravity})
    for case, rise, g in model.temperatures:
        if not staged or g in l_on:
            leave("temperature load")
    if model.other.get("PRESSURE"):
        leave("*PRESSURE", model.other["PRESSURE"])
    if model.other.get("SPRING (not read)"):
        leave("point spring that is not linear, or whose row is not in the form this reads", model.other["SPRING (not read)"])
    if model.other.get("BEAMLOAD (not uniform)"):
        leave("member load that is not uniform", model.other["BEAMLOAD (not uniform)"])

    # each tendon's nodes: the girder's, where its arms go
    for entry, own, _k_t, (mat_t, area_one, type_) in specs:
        entry["nodes"] = node_region([n for n in own if index[n] in used_nodes], f"tendon {entry['name']}")
        sname = f"strand_{type_}"
        sections.setdefault(sname, {"type": "bar", "material": mat_t, "shape": "general", "area": area_one})
        entry["section"] = sname

    # ---------------------------------------------------------------- write
    msh = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    msh += [f'{dim} {i + 1} "{n}"' for i, (dim, n) in enumerate(names)]
    msh += ["$EndPhysicalNames", "$Nodes", str(len(node_ids))]
    msh += [f"{index[n]} {xyz[n][0]:.12g} {xyz[n][1]:.12g} {xyz[n][2]:.12g}" for n in node_ids]
    rows = [f"15 2 {t} {t} {n}" for t, n in points_out] + [f"1 2 {t} {t} {a} {b}" for t, a, b in lines_out]
    msh += ["$EndNodes", "$Elements", str(len(rows))]
    msh += [f"{i + 1} {r}" for i, r in enumerate(rows)]
    msh += ["$EndElements", ""]
    with open(os.path.join(folder, f"{name}.msh"), "w", encoding="utf-8") as out:
        out.write("\n".join(msh))

    case = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": f"{name}.msh"},
            "materials": {k: v for k, v in materials.items()
                          if any(s.get("material") == k for s in sections.values())},
            "sections": sections, "assign": assign, "constraints": constraints, "loads": base_loads}
    # THE OFFSET ARMS AS ONE RULE: each arm's section worked
    # out from the mesh where the case is read, as `vfife.prep.offsets` does
    from .. import gmsh as gmsh_io
    from ...prep import offsets as prep_offsets
    msh_path = os.path.join(folder, f"{name}.msh")
    n_ruled = n_own = 0
    arm_mesh = None
    if specs or any(q.startswith("offset_arm_") for q in sections):
        arm_mesh, _el = gmsh_io.read(msh_path, verify=False)
        case, n_ruled, n_own = _rule_offset_arms(case, arm_mesh, offset_rigid, stiffest)
    if n_ruled:
        notes.append(f"{n_ruled} offset arm(s) written as one rule (`offset_arms`), each worked out on reading"
                     + (f"; {n_own} keep a section of their own" if n_own else ""))
    # THE CLOCK: a model that has something time-dependent
    # (for now its tendons' relaxation) runs each stage's duration after the
    # stage, as a step of time with the stage's loads held; Midas' results of
    # a stage are those at its end
    # THE TENDONS ARE THE CASE'S OWN ENTRIES, expanded onto
    # the mesh just written -- as the case's reading will do -- one at a
    # time first: to leave out the ones that cannot be, and to see whether
    # any relaxes (a bar whose loss rounds to nothing does not)
    relaxing = False
    if specs:
        from ...prep import tendons as prep_tendons
        mesh_ = arm_mesh
        kept_specs = []
        for spec_ in specs:
            probe = {"materials": case["materials"], "sections": case["sections"], "tendons": [spec_[0]]}
            try:
                low_ = prep_tendons.expand(probe, mesh_)[0]
            except ValueError as exc:
                leave(f"tendon: {str(exc).split(': ', 1)[-1]} (left out)")
                continue
            relaxing |= any(v.get("relaxation") for v in low_["materials"].values())
            kept_specs.append(spec_)
        specs = kept_specs
    timed = staged and relaxing
    age = AGE0
    stage_steps = []
    # A STAGE BUILDS MIDAS' GROUPS: the regions of one
    # structure group that a stage builds all of are that group's name in
    # its `activate`, and the group is written once in `groups`. A name the
    # mesh or the case has already, or one of the converter's own (in
    # brackets), stays as its regions.
    by_group: Dict[str, List[str]] = {}
    for region_, info_ in region_info.items():
        g_ = info_.get("midas_group")
        if isinstance(g_, str) and g_ and not g_.startswith("(") and info_.get("kind") in ("beam", "bar"):
            by_group.setdefault(g_, []).append(region_)
    taken_names = {n for _dim, n in names} | set(sections) | set(materials)
    used_groups: Dict[str, List[str]] = {}

    def grouped(regions_):
        have = set(regions_)
        out_ = []
        for g_, members_ in by_group.items():
            if g_ not in taken_names and len(members_) > 1 and set(members_) <= have:
                out_.append(g_)
                used_groups[g_] = members_
                have -= set(members_)
        return out_ + [r_ for r_ in regions_ if r_ in have]

    if staged:
        steps = []
        for k, st in enumerate(model.stages):
            step = {"name": st.name, "procedure": procedure}
            if activate.get(k):
                step["activate"] = grouped(activate[k])
                # along the tangent of what stands where the model says so for
                # a member the stage builds (its tangent-assembly group), else
                # where the model has them
                how = birth
                if birth == "auto":
                    how = "tangent" if tangent_stage.get(k) else "design"
                if how != "tangent":
                    step["birth"] = how
            if deactivate.get(k):
                step["deactivate"] = grouped(deactivate[k])
            if release.get(k):
                step["release"] = release[k]
            if step_constraints.get(k):
                step["constraints"] = step_constraints[k]
            if step_loads.get(k):
                step["loads"] = step_loads[k]
            steps.append(step)
            if timed and st.duration > 0.0:
                age += float(st.duration)
                tstep = {"name": f"{st.name} ({st.duration:g} d)", "procedure": procedure,
                         "to_age": round(age, 9), "load_steps": 3 if st.duration <= 60.0 else 6}
                # the stage's own constraints (the twist hold of a part on pins in
                # one line) hold through its time too; the kept ones carry over
                # (review, v1.16: the part was free to spin)
                own = [c for c in step_constraints.get(k, []) if not c.get("keep")]
                if own:
                    tstep["constraints"] = own
                steps.append(tstep)
            stage_steps.append(len(steps) - 1)
        case["steps"] = steps
        if used_groups:
            case["groups"] = used_groups
    elif specs:
        # a jack holds its force in the step that builds the bar: a model with no stages is one step
        case["steps"] = [{"name": "the model", "procedure": procedure}]
    case["solver"] = {"mode": procedure, "backend": "numba"}
    if timed:
        case["solver"]["age"] = AGE0
        notes.append(f"each stage's duration runs after it as a step of time (from {AGE0:g} days): "
                     "the tendons' relaxation; creep and shrinkage not followed")
    if procedure == "static":
        # the tangent's differences: a step of 1e-5 of a member's length keeps
        # their round-off under a soft link's stiffness (see `stiffest`)
        case["solver"]["static"] = {"perturbation": 1.0e-5}
    case["output"] = {"path": f"results/{name}.h5", "fields": ["displacement"] + (["axial_force"] if specs else [])}
    tendon_out: Dict[str, dict] = {}
    if specs:
        # THE TENDONS ARE THE CASE'S OWN ENTRIES, expanded
        # onto the mesh just written -- as the case's reading will do -- to
        # leave out the ones that cannot be and to count what they become
        stage_of = {entry["name"]: k_t for entry, _own, k_t, _m in specs}
        for entry, _own, k_t, _m in specs:
            if staged:
                entry["stress"] = {"step": model.stages[k_t].name, **entry["stress"]}
        case["tendons"] = [entry for entry, *_rest in specs]
        from ...prep import blocks as prep_blocks
        # as the case's reading does: its groups first, then the offset arms, then its tendons
        low, overlay, rep_ = prep_tendons.expand(prep_offsets.expand(prep_blocks.groups(case), mesh_), mesh_,
                                                 warn=lambda _who, t: leave(f"tendon {t}"))
        for nm, t_ in rep_.items():
            tendon_out[nm] = {"stage": stage_of[nm], "count": t_["count"], "area": t_["area"],
                              "jacking": t_["jacking"], "segments": [[r, 0, p] for r, p in t_["segments"]]}
        if tendon_out:
            notes.append(f"{len(tendon_out)} tendon(s) as bars bonded to the girder on rigid arms, each bar jacked to "
                         "the force after friction and draw-in in the stage that stresses it; creep and shrinkage "
                         "not followed" + ("" if timed else ", nor relaxation"))
        if expanded:
            # the long form, as `vfife expand` writes it: to compare the two
            tmp = msh_path + ".part"
            gmsh_io.write_expanded(msh_path, tmp, overlay=overlay)
            os.replace(tmp, msh_path)
            case = low
        else:
            # the short form: what every tendon has said once, the profiles in CSV
            defaults_, short_ = _shorten_tendons(case["tendons"], folder, name)
            if defaults_:
                at = list(case).index("tendons")
                items = list(case.items())
                items[at:at + 1] = [("tendon_defaults", defaults_), ("tendons", short_)]
                case = dict(items)
    if expanded and not specs and case.get("offset_arms"):
        case = prep_offsets.expand(case, arm_mesh)
    with open(os.path.join(folder, f"{name}.yaml"), "w", encoding="utf-8") as out:
        # a list of numbers on one line -- a tendon's profile point, an
        # orientation -- every float still written as it is (repr)
        yaml.dump(case, out, Dumper=_FlowDumper, sort_keys=False, allow_unicode=True, width=120)

    n_members = sum(len(v) for k, v in total.items() if k[0] in ("beam", "bar"))
    # THE MAP GIVES A NODE'S ROW IN THE RESULTS. The mesh reader leaves out a
    # node no element uses and numbers the rest in the order of the file, so
    # a node's row is its rank among the nodes that are used -- not its
    # place in the file. With every node used the two are the same, which is
    # why a whole model read its results right; a model cut at `stages`, or
    # one with nodes only left-out elements reach, read another node's
    # (found on the real arch, v1.14: a control point "moved" -14 mm that
    # had moved +1 -- the row was a tower node's).
    in_use = [n for n in node_ids if index[n] in used_nodes]
    row = {n: i for i, n in enumerate(in_use)}
    beams_ = [v for v in sections.values() if v.get("type") == "beam" and v.get("material") != "rigid"]
    sheared = sum(1 for v in beams_ if v.get("theory") == "timoshenko")
    if shear and beams_:
        notes.append(f"{sheared} of {len(beams_)} beam sections give way in shear too (Timoshenko, with the file's or the "
                     "shape's shear areas); the others have none stated and are Euler's")
    if stiff_left[0]:
        notes.append(f"{stiff_left[0]} member(s) stiffer than {stiffest:g} N/m (up to {stiff_left[1]:.3g}) are left as they "
                     f"are: lengths of beam, not stubs. If the static solution does not converge, raise `stub`")
    report = {"nodes": len(in_use), "nodes_no_member_reaches": len(node_ids) - len(in_use), "members": n_members,
              "rigid_arms": sum(len(v) for k, v in total.items() if k[0] == "rigid"),
              "offset_arms_ruled": n_ruled,
              "connectors": sum(len(v) for k, v in total.items() if k[0] in ("link", "stub")),
              "regions": len(names), "sections": len(case["sections"]), "steps": len(model.stages),
              "jacked_cables": sum(len(v) for k, v in total.items() if k[0] == "bar" and k[6] and k[6][0] == "jack" and not str(k[2]).startswith("(tendon")),
              "tendons": len(tendon_out), "tendon_bars": sum(len(t_["segments"]) for t_ in tendon_out.values()),
              "cables_cut_to_length": sum(len(v) for k, v in total.items() if k[0] == "bar" and k[6] and k[6][0] == "cut"),
              "members_set_off_their_nodes": set_off[0], "stubs_made_joints": softened[0], "stub_weight_left_out_N": round(stub_weight[0], 1),
              "members_over_the_limit_left_as_they_are": stiff_left[0], "links_made_to_turn_with_their_ends": stiffened[0],
              "plate_weight_on_nodes_N": round(plate_weight[0], 1),
              "supports_brought_to_their_original_position": n_original[0],
              "parts_held_from_spinning": len(spin_nodes),
              "left_out": dict(sorted(left.items())), "notes": sorted(set(notes))}
    with open(os.path.join(folder, f"{name}.map.json"), "w", encoding="utf-8") as out:
        json.dump({"report": report, "nodes": {str(n): row[n] for n in in_use},
                   "elements": {str(e): v for e, v in element_map.items()}, "regions": region_info,
                   "tendons": tendon_out,
                   "stages": [s.name for s in model.stages],
                   # the step each stage ends with (its step of time, if it has one)
                   "stage_steps": stage_steps or list(range(len(model.stages)))}, out, ensure_ascii=False)
    return report
