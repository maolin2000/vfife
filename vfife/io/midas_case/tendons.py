"""Midas tendons as bars bonded to the girder."""
from __future__ import annotations

import math

# the line and the force along it, shared with the case's own `tendons` (vfife/prep)
from ...prep.tendons import _rounded, _drawn_in, _drawn_in_both  # noqa: F401


def tendon_specs(model, materials, staged, l_on, l_off, born, xyz, length, f, leave, notes):
    """The tendons as the case's `tendons:` entries (vfife/prep/tendons.py,
    v1.20): [(entry, its model nodes, the stage that stresses it)].
    The entry's `nodes` and `stress.step` are filled in by the caller, which
    knows the mesh's groups and the steps."""
    # ------------------------------------------------------------- tendons
    # A TENDON IS A BAR BONDED TO THE GIRDER (1.15): a node of
    # its own on its line at every node of the members it runs through (and
    # at its two anchors), each joined to the girder's node by a rigid arm,
    # and a bar between each two. Each bar is jacked to the force the
    # tendon has there in the stage that stresses it -- the jacking force
    # less friction (mu per radian of the line's turning, k per length) and
    # the anchorage's draw-in -- and is an ordinary bar after: it shortens
    # with the girder (the elastic losses of the stages that follow) and it
    # adds its stiffness to the girder's section, as a bonded tendon does.
    # Since v1.20 this writes each tendon as a designer would, and the
    # case's reading expands it (`vfife.prep.tendons.expand`).
    out = []
    stress_of = {}
    for case_, name_, how_, ends_, at_b, at_e, g_ in getattr(model, "tendon_stress", ()):
        if name_ in stress_of:
            leave("second prestress of a tendon (the first is followed)")
        stress_of.setdefault(name_, (how_, ends_, at_b, at_e, g_))
    seen = set()
    for tp in getattr(model, "tendons", ()):
        tt = model.tendon_types.get(tp.type)
        if tp.name in seen:
            leave("tendon of a name an earlier one has (left out)")
            continue
        if tp.name not in stress_of:
            leave("tendon that no prestress load stresses")
            continue
        how_, ends_, at_b, at_e, g_ = stress_of[tp.name]
        if tt is None or tp.reference not in ("STRAIGHT", "ELEMENT") or tp.input != "2D" or not tp.z:
            leave("tendon whose type or profile is not in a form this reads (a 2-D profile on a straight or an element line)")
            continue
        if tp.curve not in ("ROUND", ""):
            leave(f"tendon profile drawn as {tp.curve} (taken as straight between its points, corners unrounded)")
        if getattr(tp, "fixed_angle", False):
            leave("tendon profile with a fixed tangent at a point (not followed: the corners are rounded as drawn)")
        if tt.kind != "INTERNAL" or not tt.post or not tt.bonded:
            leave(f"tendon not internal, post-tensioned and bonded ({tt.kind})")
            continue
        if f"m{tt.material}" not in materials:
            leave("tendon whose material has no elastic modulus")
            continue
        k_t = 0
        if staged:
            if g_ not in l_on:
                leave("tendon whose prestress load group no stage activates")
                continue
            # stressed when its load group is on AND its members are built, as in Midas: one load group
            # often holds every tendon of a cantilever bridge, each stressed with its own segment
            if any(e not in born for e in tp.elements if e in model.elements):
                leave("tendon on a member no stage builds")
                continue
            k_t = max([l_on[g_]] + [born[e][0] for e in tp.elements if e in model.elements])
            if g_ in l_off and l_off[g_] <= k_t:
                leave("tendon whose prestress load group is removed before its members are built")
                continue
            if g_ in l_off:
                leave("prestress load group removed after its tendons are stressed (the tendons stay)")
        # its reference line
        if tp.reference == "STRAIGHT":
            ex = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0)}.get(tp.axis)
            if ex is None:
                leave(f"tendon on a straight line along {tp.axis} (not followed)")
                continue
            origin = tuple(c * length for c in tp.origin)
        else:
            el_r = model.elements.get(tp.ref_element)
            if el_r is None:
                leave("tendon on a reference element the file does not define")
                continue
            ia, ja = el_r.nodes[:2]
            if tp.axis.startswith("J"):
                ia, ja = ja, ia
            d_ = [xyz[ja][i] - xyz[ia][i] for i in range(3)]
            n_d = math.sqrt(sum(c * c for c in d_))
            ex = tuple(c / n_d for c in d_)
            origin = xyz[el_r.nodes[0] if tp.ref_end == "I" else el_r.nodes[1]]
        if math.hypot(ex[0], ex[1]) < 1.0e-9:
            leave("tendon on a vertical reference line (not followed)")
            continue
        if any(b[0] <= a[0] for a, b in zip(tp.z, tp.z[1:])) or any(b[0] <= a[0] for a, b in zip(tp.y, tp.y[1:])):
            leave("tendon profile whose points do not run forward along its line (left out)")
            continue
        own = sorted({n for e in tp.elements if e in model.elements for n in model.elements[e].nodes[:2]})
        if not own:
            leave("tendon on no member of the model")
            continue
        if ends_ not in ("BEGIN", "END", "BOTH"):
            leave(f"tendon jacked at {ends_} (not followed)")
            continue
        area_one = tt.area * length ** 2
        area_t = area_one * tp.count
        f_len = f / length ** 2
        stress = {"ends": ends_.lower()}
        if how_ == "STRESS":
            stress["stress"] = [at_b * f_len, at_e * f_len]
        else:
            stress["force"] = [at_b * f * tp.count, at_e * f * tp.count]   # (a force is one tendon's)
        # ITS RELAXATION, as the code the model was built to says it
        relaxes = tt.relax and getattr(tt, "relax_method", 0) == 2 and tt.relax_zeta > 0.0 and tt.fpk > 0.0
        if tt.relax and getattr(tt, "relax_method", 0) != 2:
            leave(f"tendon relaxation of a kind not followed (method {getattr(tt, 'relax_method', 0)})")
        elif tt.relax and not relaxes:
            leave("tendon relaxation with no fpk or relaxation coefficient (not followed)")
        elif relaxes and not staged:
            leave("tendon relaxation in a model with no stages (no time passes)")
        entry = {"name": tp.name, "nodes": None, "section": None, "count": int(tp.count),
                 "line": {"origin": [float(c) for c in origin], "direction": [float(c) for c in ex]}}
        if tp.turn:
            entry["line"]["turn"] = float(tp.turn)
        if tp.reference == "ELEMENT" and any(tp.offset):
            entry["offset"] = [tp.offset[0] * length, tp.offset[1] * length]
        if tp.y:
            entry["profile_y"] = [[x * length, v * length, r * length] for x, v, r in tp.y]
        entry["profile_z"] = [[x * length, v * length, r * length] for x, v, r in tp.z]
        entry["stress"] = stress
        if tt.friction or tt.wobble:
            entry["friction"] = {"mu": tt.friction, "k": tt.wobble / length}
        if tt.slip_begin or tt.slip_end:
            entry["draw_in"] = {"begin": tt.slip_begin * length, "end": tt.slip_end * length}
        if relaxes:
            entry["relaxation"] = {"code": "jtg3362", "fpk": tt.fpk * f_len, "zeta": tt.relax_zeta,
                                   "overstressed": bool(tt.overstressed)}
        seen.add(tp.name)
        out.append((entry, own, k_t, (f"m{tt.material}", area_one, tp.type)))
    return out
