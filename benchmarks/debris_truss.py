"""A generic steel through truss losing an end post: the debris test of
v1.17 (own geometry, no real bridge).

60 m in 10 panels of 6 m, 8 m deep, the two trusses 10 m apart; box chords
and end posts, H-section webs, floor beams, three stringers carrying the
deck's weight, top struts, X bracing of bars top and bottom. Every member
breakable: plastic hinges (Q345, rotation capacity 0.04 rad) that split off
when they part, bars that snap. The dead load comes on over 0.6 s, the end
post at one corner is removed at 2 s, and what falls lands on a rigid ground
6 m below the deck (damping 0.5, friction 0.3).

    python benchmarks/debris_truss.py debris.yaml      # writes the case
    vfife run debris.yaml
"""
import sys

L, N, H, W = 60.0, 10, 8.0, 10.0


def _box(B, Hh, t):
    A = B * Hh - (B - 2 * t) * (Hh - 2 * t)
    iy = (B * Hh**3 - (B - 2 * t) * (Hh - 2 * t)**3) / 12
    iz = (Hh * B**3 - (Hh - 2 * t) * (B - 2 * t)**3) / 12
    j = 2 * t * ((B - t) * (Hh - t))**2 / ((B - t) + (Hh - t))
    return dict(area=A, iy=iy, iz=iz, j=j), [B / 2, Hh / 2], 1.12


def _h(Hh, B, tw, tf):
    A = 2 * B * tf + (Hh - 2 * tf) * tw
    iy = (B * Hh**3 - (B - tw) * (Hh - 2 * tf)**3) / 12
    iz = 2 * tf * B**3 / 12 + (Hh - 2 * tf) * tw**3 / 12
    j = (2 * B * tf**3 + (Hh - 2 * tf) * tw**3) / 3
    return dict(area=A, iy=iy, iz=iz, j=j), [B / 2, Hh / 2], 1.14


def case(total_time=6.0, output="debris.h5", damping=1.0, every=200):
    p = L / N
    pts, lines = {}, {}

    def P(name, x, y, z):
        pts[name] = [round(x, 6), round(y, 6), round(z, 6)]

    for j, y in enumerate((0.0, W)):
        for i in range(N + 1):
            P(f"b{j}_{i}", i * p, y, 0.0)
        for i in range(1, N):
            P(f"t{j}_{i}", i * p, y, H)
    for k, y in enumerate((W / 4, W / 2, 3 * W / 4)):
        for i in range(N + 1):
            P(f"s{k}_{i}", i * p, y, 0.0)

    def line(name, through):
        lines[name] = {"through": through}

    for j in range(2):
        line(f"chord_bot{j}", [f"b{j}_{i}" for i in range(N + 1)])
        line(f"chord_top{j}", [f"t{j}_{i}" for i in range(1, N)])
        line(f"endpost_a{j}", [f"b{j}_0", f"t{j}_1"])
        line(f"endpost_b{j}", [f"b{j}_{N}", f"t{j}_{N - 1}"])
        for i in range(1, N):
            line(f"vert{j}_{i}", [f"b{j}_{i}", f"t{j}_{i}"])
        for i in range(1, N - 1):
            line(f"diag{j}_{i}", [f"t{j}_{i}", f"b{j}_{i + 1}"] if i < N // 2 else [f"b{j}_{i}", f"t{j}_{i + 1}"])
    for i in range(N + 1):
        line(f"floor_{i}", [f"b0_{i}"] + [f"s{k}_{i}" for k in range(3)] + [f"b1_{i}"])
    for k in range(3):
        line(f"stringer{k}", [f"s{k}_{i}" for i in range(N + 1)])
    for i in range(1, N):
        line(f"strut_{i}", [f"t0_{i}", f"t1_{i}"])
    for i in range(N):
        line(f"latbot_{i}", [f"b0_{i}", f"b1_{i + 1}"] if i % 2 == 0 else [f"b1_{i}", f"b0_{i + 1}"])
    for i in range(1, N - 1):
        line(f"lattop_{i}", [f"t0_{i}", f"t1_{i + 1}"] if i % 2 == 0 else [f"t1_{i}", f"t0_{i + 1}"])
    groups = {
        "chords": [k for k in lines if k.startswith("chord")],
        "endposts_rest": ["endpost_b0", "endpost_a1", "endpost_b1"],
        "webs": [k for k in lines if k.startswith(("vert", "diag"))],
        "floors": [k for k in lines if k.startswith("floor")],
        "stringers": [k for k in lines if k.startswith("stringer")],
        "struts": [k for k in lines if k.startswith("strut")],
        "bracing": [k for k in lines if k.startswith(("latbot", "lattop"))],
    }
    S = {"chord": _box(0.5, 0.6, 0.02), "endpost": _box(0.5, 0.6, 0.02), "web": _h(0.45, 0.4, 0.016, 0.02),
         "floor": _h(1.2, 0.4, 0.016, 0.025), "stringer": _h(0.6, 0.3, 0.012, 0.016), "strut": _h(0.4, 0.3, 0.012, 0.016)}
    sections = {nm: dict(type="beam", material="deck" if nm == "stringer" else "steel", shape="general",
                         **{k: float(v) for k, v in pr.items()}) for nm, (pr, _ef, _sf) in S.items()}
    sections["brace"] = dict(type="bar", material="steel", shape="general", area=0.004)

    def fail(nm):
        _pr, ef, sf = S[nm]
        return dict(yield_strength=345e6, extreme_fibre=[float(x) for x in ef], shape_factor=sf,
                    rotation_capacity=0.04, split=True)

    assign = [dict(region="chords", section="chord", orientation=[0, 0, 1], failure=fail("chord")),
              dict(region="endpost_a0", section="endpost", orientation=[0, 1, 0], failure=fail("endpost")),
              dict(region="endposts_rest", section="endpost", orientation=[0, 1, 0], failure=fail("endpost")),
              dict(region="webs", section="web", orientation=[0, 1, 0], failure=fail("web")),
              dict(region="floors", section="floor", orientation=[0, 0, 1], failure=fail("floor")),
              dict(region="stringers", section="stringer", orientation=[0, 0, 1], failure=fail("stringer")),
              dict(region="struts", section="strut", orientation=[0, 0, 1], failure=fail("strut")),
              dict(region="bracing", section="brace", failure=dict(tensile_strength=470e6))]
    # the stringers carry the deck (8 kN/m^2 over a third of the width each) in their density
    rho_deck = 7850.0 + 8e3 * W / 3 / 9.81 / S["stringer"][0]["area"]
    return {
        "version": 1, "name": "debris_truss",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "lines", "points": pts, "lines": lines, "groups": groups},
        "materials": {"steel": {"type": "linear_elastic", "E": 2.06e11, "nu": 0.3, "rho": 7850.0},
                      "deck": {"type": "linear_elastic", "E": 2.06e11, "nu": 0.3, "rho": rho_deck}},
        "sections": sections, "assign": assign,
        "regions": {"bearing_fix": {"where": {"x": 0.0, "y": 0.0, "z": 0.0}},
                    "bearing_x0y": {"where": {"x": 0.0, "y": W, "z": 0.0}},
                    "bearing_l0": {"where": {"x": L, "y": 0.0, "z": 0.0}},
                    "bearing_lw": {"where": {"x": L, "y": W, "z": 0.0}}},
        "constraints": [{"region": "bearing_fix", "dofs": [1, 1, 1, 0, 0, 0]},
                        {"region": "bearing_x0y", "dofs": [1, 0, 1, 0, 0, 0]},
                        {"region": "bearing_l0", "dofs": [0, 1, 1, 0, 0, 0]},
                        {"region": "bearing_lw", "dofs": [0, 0, 1, 0, 0, 0]}],
        "amplitudes": {"up": [[0.0, 0.0], [0.6 / total_time, 1.0], [1.0, 1.0]]},
        "loads": [{"region": "all", "gravity": [0.0, 0.0, -9.81], "amplitude": "up"}],
        "removals": [{"region": "endpost_a0", "time": 2.0, "split": True}],
        "contact": {"rigid": [{"name": "ground", "plane": {"point": [0, 0, -6.0], "normal": [0, 0, 1]},
                               "damping": 0.5, "friction": 0.3}]},
        "solver": {"mode": "dynamic", "backend": "numba", "total_time": float(total_time),
                   "damping": {"type": "viscous", "coeff": float(damping)}},
        "output": {"path": output, "every": every,
                   "fields": ["displacement", "rotation", "member_failed", "capacity_ratio"]},
    }


if __name__ == "__main__":
    import yaml
    out = sys.argv[1] if len(sys.argv) > 1 else "debris.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(case(), f, sort_keys=False)
    print("written", out)
