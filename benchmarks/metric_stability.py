"""Which measurements of this model can be compared, and which cannot?

WHY
---
An entire round of work on the solid RC path was spent comparing configurations
by "the drift the run reached before relaxation stopped converging". That number
moved from 33.0 mm to 22.0 mm when l_ref changed by 3e-16 -- on warp, while
numba did not move at all. Four hypotheses were declared refuted on the strength
of it. None of them had actually been tested.

The fix is not a better hypothesis, it is a rule: BEFORE a quantity is used to
compare anything, perturb it with something that must not change it, and check
that it does not. This script does that for every candidate metric at once.

THE PERTURBATIONS, AND WHY EACH ONE MUST BE A NO-OP
  l_ref last bit   auto resolves to 0.0999999999999997 on this mesh; passing
                   0.1 is the same length to fifteen digits.
  load steps       the loading path is the same path, walked in smaller
                   increments. A quasi-static rate-independent answer that
                   changes here was not converged in the first place.
  tolerance        a tighter convergence bar on the same equilibrium.
  backend          numpy, numba and warp are three implementations of one
                   set of equations; the repo has equivalence tests asserting
                   they agree to 1e-11.

A metric that survives all of these can carry an argument. One that does not is
noise wearing a unit.

THE CONTROL
  `reached_mm` -- the metric already known to be broken -- is measured too. If
  this harness does not flag it, the harness is not working. Read that row
  first.

    python benchmarks\\metric_stability.py
    python benchmarks\\metric_stability.py --add-backend warp --device cuda
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
#  The model. Inlined rather than imported: the diagnostic scripts this once
#  shared a builder with were one-offs and have been deleted, and a qualification
#  harness should not depend on a script that might go with them.
# --------------------------------------------------------------------------- #

H = 1.6                       # to the point of inflection
FC = 25.6e6                   # unconfined cylinder strength, PEER sheet
EC = float(5000.0 * np.sqrt(25.6) * 1e6)   # Mander's own 5000 sqrt(f'c) = 25.3 GPa
# float(), not np.float64: yaml.safe_dump refuses numpy scalars, and this
# value is written straight into the case as the material's E.
FCC, EPS_CC, EPS_CU = 42.06e6, 0.0084, 0.0345      # Mander for the Tanaka cage


def mander_table(n=60):
    """Mander's confined curve as an Abaqus-style CDP compression table.

    sig = fcc x r / (r - 1 + x^r),  x = eps/eps_cc,  r = Ec/(Ec - Esec)

    This is Popovics 1973 as Mander 1988 uses it, and it is the curve fcc and
    eps_cc actually belong to -- the parabola concrete02 uses cannot hold both
    of them together, which is a separate finding recorded in
    verification/case_11_fiber_rc_column_cyclic.yaml.

    The table starts at 0.4 fcc, the elastic limit CDP's sigma_c0_ratio uses.
    Damage is taken as 1 - sig/fcc past the peak and zero before it: the usual
    convention that the unloading modulus is (1-d) E0.
    """
    esec = FCC / EPS_CC
    r = EC / (EC - esec)
    eps = np.linspace(0.4 * FCC / EC, EPS_CU, n)
    x = eps / EPS_CC
    sig = FCC * x * r / (r - 1.0 + x ** r)
    inelastic = np.maximum(eps - sig / EC, 0.0)
    damage = np.where(eps > EPS_CC, 1.0 - sig / FCC, 0.0)
    keep = np.concatenate(([True], np.diff(inelastic) > 1e-9))
    return (sig[keep].tolist(), inelastic[keep].tolist(),
            np.clip(damage[keep], 0.0, 0.95).tolist())


def compression_spec(curve):
    if curve == "unconfined":
        return {"source": "gb50010", "fc": FC,
                "beta_c": 0.7, "sigma_c0_ratio": 0.4}
    stress, inelastic, damage = mander_table()
    return {"source": "table", "stress": stress,
            "inelastic_strain": inelastic, "damage": damage}


def build_case(nxy, curve, drift, load_steps, backend, device, out,
               erosion=True, l_ref="auto"):
    nz = nxy * 4
    n_hoop = int(round(H / 0.08)) - 1
    c, far, mid = 0.062, 0.338, 0.2
    longitudinal = [
        {"start": [y, z, 0.0], "end": [y, z, H], "divisions": nz}
        for y, z in ((c, c), (far, c), (c, far), (far, far),
                     (mid, c), (mid, far), (c, mid), (far, mid))]
    corners = [(c, c), (far, c), (far, far), (c, far)]
    hoops = [{"start": [*corners[i], 0.08],
              "end": [*corners[(i + 1) % 4], 0.08], "divisions": 4,
              "repeat": {"count": n_hoop, "offset": [0.0, 0.0, 0.08]}}
             for i in range(4)]

    solver = {"backend": backend, "mode": "relaxation", "mass": "underwood",
              "n_steps": 400000, "tolerance": 1.0e-3,
              "relaxation_safety": 0.3, "damping": {"type": "kinetic"},
              # Erosion thresholds from the material, not from habit: the
              # confined core is still carrying load at 0.0345 strain, and
              # case_06's strain: 0.01 deletes it long before that.
              }
    if erosion:
        # Thresholds from the material, not from habit: the confined core is
        # still carrying load at 0.0345 strain, and case_06's strain: 0.01
        # deletes it long before that.
        solver["erosion"] = {"damage": 0.99, "strain": 0.05,
                             "points": 4, "every": 100}
    if backend == "warp":
        solver["device"] = device
        solver["precision"] = "fp64"

    return {
        "name": f"drift_limit_{nxy}_{curve}",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [0.4, 0.4, H], "divisions": [nxy, nxy, nz]},
        "element": {"formulation": "b-bar"},
        "material": {
            "type": "cdp", "E": EC, "nu": 0.2, "rho": 2400.0,
            "fb0_fc0": 1.16, "Kc": 0.6667, "dilation_angle": 36.0,
            "eccentricity": 0.1, "wt": 0.0, "wc": 1.0,
            "viscosity": 1.0, "n_return": 5, "crack_band": "both",
            "l_ref": l_ref,
            "compression": compression_spec(curve),
            "tension": {"source": "fracture_energy", "ft": 3.1e6,
                        "Gf": 120.0, "beta_t": 0.9},
        },
        "reinforcement": [
            {"name": "longitudinal", "area": 3.1416e-4,
             "host_deduction": EC,
             "material": {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
                          "yield_strength": 474.0e6, "hardening": 1.0e9,
                          "kinematic_fraction": 0.5},
             "lines": longitudinal},
            {"name": "hoops", "area": 1.131e-4, "host_deduction": EC,
             "material": {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
                          "yield_strength": 333.0e6, "hardening": 1.0e9,
                          "kinematic_fraction": 0.5},
             "lines": hoops},
        ],
        "regions": {
            "base": {"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}},
            "head": {"plane": {"point": [0, 0, H], "normal": [0, 0, 1]}},
        },
        "constraints": [{"region": "base", "dofs": [1, 1, 1]}],
        "loads": [{"region": "head", "vector": [0.0, 0.0, -819.0e3]}],
        "steps": [
            {"name": "axial", "n_steps": 40000, "tolerance": 1.0e-4},
            {"name": "push",
             "constraints": [{"region": "head", "dofs": [1, 0, 0],
                              "value": drift * H}],
             "load_steps": load_steps, "max_steps_per_load": 6000,
             "tolerance": 1.0e-3},
        ],
        "solver": solver,
        "output": {"path": out, "every": 200,
                   "monitors": {"shear": {"kind": "reaction", "region": "base",
                                          "dof": "x"},
                                "tip": {"kind": "displacement",
                                        "region": "head", "dof": "x"}}},
    }




def run(tag, nxy, drift, load_steps, backend, device, l_ref, tol,
        curve="unconfined"):
    import h5py
    import yaml
    from vfife.api import run_case

    out = os.path.join(ROOT, "results", f"_ms_{tag}.h5")
    case = build_case(nxy, curve, drift, load_steps, backend,
                          device, out, l_ref=l_ref)
    for s in case["steps"]:
        if s["name"] == "push":
            s["tolerance"] = tol
    case["output"]["every"] = 100000
    path = os.path.join(ROOT, "cases", f"_ms_{tag}.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(case, fh, allow_unicode=True)
    try:
        run_case(path, output=out)
    except Exception as e:
        print(f"    {tag}: {type(e).__name__}: {str(e).splitlines()[0][:60]}")
    finally:
        if os.path.exists(path):
            os.remove(path)
    with h5py.File(out) as f:
        tip = np.abs(np.asarray(f["monitors/tip"]))
        shear = np.abs(np.asarray(f["monitors/shear"]))
    return tip, shear


def metrics(tip, shear, probes=(4.0, 8.0, 12.0)):
    """Candidate measurements, including one known to be bad, as a control."""
    order = np.argsort(tip)
    t, s = tip[order], shear[order]
    m = {
        # THE CONTROL: where relaxation gave up. A threshold crossing.
        "reached_mm": float(tip.max() * 1e3),
        # Curve quantities, read at points the curve actually passes through.
        "peak_shear_kN": float(shear.max() / 1e3),
        "drift_at_peak_mm": float(tip[int(np.argmax(shear))] * 1e3),
    }
    for p in probes:
        m[f"shear@{p:.0f}mm_kN"] = float(np.interp(p / 1e3, t, s) / 1e3)
    m["secant@8mm_kN/mm"] = m["shear@8mm_kN"] / 8.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--divisions", type=int, default=4)
    ap.add_argument("--drift", type=float, default=0.08,
                    help="push PAST failure so every run stops at its own "
                         "limit -- otherwise the control metric never "
                         "crosses its threshold and the test proves nothing")
    ap.add_argument("--load-steps", type=int, default=128,
                    help="baseline increment count; the harness also runs 2x "
                         "this. Keep the increment near 1 mm -- a coarse "
                         "baseline makes every metric look unstable when what "
                         "is unstable is the baseline")
    ap.add_argument("--curve", default="unconfined",
                    choices=["unconfined", "confined"],
                    help="WHICH CONFIGURATION TO QUALIFY THE METRICS IN. "
                         "This matters more than it looks. With 'unconfined' "
                         "(the GB50010 curve) numba and warp agree to 0.02% "
                         "and even the control metric is steady, so the run "
                         "qualifies nothing -- there is no instability present "
                         "to detect. The 33.0 vs 22.0 mm backend split was seen "
                         "with 'confined', which reaches CDP through "
                         "source: table. Qualify the metrics in the "
                         "configuration that actually misbehaves.")
    ap.add_argument("--backend", default="numba")
    ap.add_argument("--add-backend", default=None,
                    help="a second backend to include as a perturbation")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    # 128 load steps over an 8% target is 1 mm per step, which is what the
    # drift-limit runs used. An earlier version of this harness kept the 12
    # steps that suited a 12 mm push and paired them with a 128 mm one -- 10.7
    # mm jumps. Doubling them then moved shear@4mm by 63%, which said nothing
    # about the metrics and everything about an under-resolved baseline.
    base = dict(nxy=a.divisions, drift=a.drift, load_steps=a.load_steps,
                backend=a.backend, device=a.device, l_ref="auto", tol=1.0e-3,
                curve=a.curve)
    # (slug, label, override). The slug is ASCII because it becomes a FILE
    # NAME and part of a YAML path: a Chinese tag written through Windows'
    # default ANSI codepage and read back as UTF-8 fails on the first non-ASCII
    # byte, which is exactly how this script first broke on the user's machine.
    variants = [
        ("base", "基准", dict()),
        ("lref", "l_ref 末位", dict(l_ref=0.1)),
        ("steps", "荷载级 x2", dict(load_steps=a.load_steps * 2)),
        ("tol", "容差收紧", dict(tol=5.0e-4)),
    ]
    if a.add_backend:
        variants.append((f"be_{a.add_backend}", f"后端 {a.add_backend}",
                         dict(backend=a.add_backend)))

    print(f"截面 {a.divisions}，{a.curve} 受压曲线，推至固定 {a.drift*100:.2f}% "
          f"（{a.drift*H*1e3:.0f} mm）\n每个扰动都不应改变任何一个量\n")
    results = {}
    for slug, label, override in variants:
        kw = dict(base); kw.update(override)
        tip, shear = run(slug, **kw)
        results[label] = metrics(tip, shear)
        print(f"  跑完 {label}", flush=True)

    names = list(next(iter(results.values())).keys())
    print(f"\n  {'量':>18} " + "".join(f"{t:>13}" for t in results)
          + f"{'相对散布':>10}  判定")
    for n in names:
        vals = np.array([results[t][n] for t in results])
        med = np.median(np.abs(vals))
        spread = (vals.max() - vals.min()) / max(med, 1e-12)
        verdict = ("可用" if spread < 0.01 else
                   "勉强" if spread < 0.05 else "不可用")
        star = "  <- 对照" if n == "reached_mm" else ""
        print(f"  {n:>18} " + "".join(f"{v:>13.3f}" for v in vals)
              + f"{spread*100:>9.2f}%  {verdict}{star}")

    print("\n  相对散布 = (最大-最小)/中位。<1% 可用，1-5% 勉强，>5% 不可用。")
    reached = np.array([results[t]["reached_mm"] for t in results])
    target = a.drift * H * 1e3
    if np.all(reached > target * 0.999):
        print("\n  !! 本次结论无效：所有运行都跑满了 "
              f"{target:.0f} mm 目标，对照量 reached_mm 从未穿越它的阈值。")
        print("  !! 扰动没有进入会失效的工况，其余各行的『可用』不构成证据。")
        print(f"  !! 把 --drift 调到失效点以上再跑（当前 {a.drift*100:.2f}%）。")
    elif (reached.max() - reached.min()) / max(np.median(reached), 1e-12) < 0.05:
        print("\n  注意：对照量这次没有散开，所以本次没有资格判定任何指标——")
        print("  扰动没有进入不稳定的工况，各行的『可用』只在本配置内成立。")
        if a.curve == "unconfined":
            print("  33.0 -> 22.0 mm 的后端分歧是在 confined 曲线上看到的，")
            print("  它经 source: table 进入 CDP。用 --curve confined 再跑一次。")
    else:
        print("\n  对照量如期散开，说明扰动确实进入了会失效的工况，")
        print("  因此其余各行的判定是有意义的。")


if __name__ == "__main__":
    main()
