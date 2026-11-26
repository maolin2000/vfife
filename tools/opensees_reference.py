"""Reference curves for the three uniaxial laws that share their names and
parameters with OpenSees materials: steel02, concrete01, concrete02.

vfife's laws are written from the papers (Menegotto-Pinto with Filippou et
al.'s isotropic hardening; Kent-Scott-Park with Karsan-Jirsa unloading;
Yassin's concrete with tension). Where a paper leaves a detail open, the
choice was fixed against the OpenSees program, used as a black box: a strain
history in, the stresses out. This script runs those histories through
OpenSees and stores the stresses; tests/test_opensees_reference.py checks the
vfife laws against them. OpenSees' source is not used.

    pip install openseespy numpy
    python tools/opensees_reference.py        # -> tests/data/opensees_uniaxial.npz
    python tools/opensees_reference.py --composite   # -> tests/data/opensees_composite.npz (1.21)

The stored file is what the tests read, so the suite does not need OpenSees.
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tests", "data",
                   "opensees_uniaxial.npz")


def _history(seed, n, lo, hi, kind="walk"):
    """A strain history in [lo, hi] from zero strain: a random walk, uniform
    random jumps, or a walk that holds each value for a few steps."""
    rng = np.random.default_rng(seed)
    if kind == "jumps":
        x = rng.uniform(0.0, 1.0, n)
    else:
        x = np.cumsum(rng.normal(0.0, 1.0, n))
        x = (x - x.min()) / (x.max() - x.min())
        if kind == "held":
            x = np.repeat(x[: n // 3], 3)
    return np.concatenate([[0.0], lo + (hi - lo) * x])


# name -> (OpenSees material, OpenSees arguments, history)
CASES = {
    "steel02": ("Steel02", [400e6, 200e9, 0.01, 20.0, 0.925, 0.15], _history(7, 3000, -0.02, 0.025)),
    "steel02_iso": ("Steel02", [400e6, 200e9, 0.01, 20.0, 0.925, 0.15, 0.04, 1.0, 0.04, 1.0],
                    _history(8, 3000, -0.02, 0.025)),
    "steel02_strand": ("Steel02", [1.58e9, 1.95e11, 0.063, 18.0, 0.9, 0.2, 0.02, 2.0, 0.01, 3.0],
                       _history(9, 3000, -0.01, 0.03)),
    "concrete01": ("Concrete01", [-30e6, -0.002, -6e6, -0.006], _history(10, 3000, -0.012, 0.001)),
    "concrete01_deep": ("Concrete01", [-45e6, -0.0025, -36e6, -0.02], _history(11, 3000, -0.03, 0.002)),
    "concrete02": ("Concrete02", [-30e6, -0.002, -6e6, -0.006, 0.1, 3e6, 1.5e9],
                   _history(12, 3000, -0.01, 0.002)),
    "steel02_small": ("Steel02", [400e6, 200e9, 0.02, 15.0, 0.925, 0.15], _history(14, 2000, -0.003, 0.004)),
    "concrete01_small": ("Concrete01", [-30e6, -0.002, -6e6, -0.006], _history(15, 2000, -0.0015, 0.0002)),
    "concrete02_small": ("Concrete02", [-30e6, -0.002, -6e6, -0.006, 0.1, 3e6, 1.5e9],
                         _history(16, 2000, -0.0012, 0.0006)),
    "steel02_b0_jumps": ("Steel02", [400e6, 200e9, 0.0, 20.0, 0.925, 0.15], _history(17, 1500, -0.02, 0.02, "jumps")),
    "steel02_softening_held": ("Steel02", [400e6, 200e9, 0.01, 20.0, 0.925, 0.15, -0.02, 1.0, -0.02, 1.0],
                               _history(18, 1500, -0.02, 0.02, "held")),
    "concrete01_jumps": ("Concrete01", [-30e6, -0.002, -30e6, -0.004], _history(19, 1500, -0.01, 0.002, "jumps")),
    "concrete02_rat0_crushed": ("Concrete02", [-30e6, -0.002, -6e6, -0.006, 0.0, 3e6, 1.5e9],
                                _history(20, 1500, -0.012, 0.002)),
    "concrete02_held": ("Concrete02", [-40e6, -0.0022, -8e6, -0.005, 0.3, 4e6, 2e9],
                        _history(21, 1500, -0.006, 0.003, "held")),
    "concrete02_tension": ("Concrete02", [-52e6, -0.0024, -39e6, -0.0063, 0.055, 4.3e6, 1.2e10],
                           _history(13, 3000, -0.004, 0.004)),
}


def run(material, args, history):
    import openseespy.opensees as ops
    ops.wipe()
    ops.uniaxialMaterial(material, 1, *args)
    ops.testUniaxialMaterial(1)
    out = np.empty(len(history))
    for i, e in enumerate(history):
        ops.setStrain(float(e))          # trial strain, committed
        out[i] = ops.getStress()
    return out


#: the composite laws (1.21) against OpenSees' Parallel and Series wrappers,
#: and `gap` against ElasticPPGap with a yield force out of reach: name ->
#: (wrapper, [(part, part arguments), ...], wrapper arguments, history).
#: Series is run with its own iteration to 1e-14 (its default is a single
#: pass); vfife's series iterates to its tolerance at every point.
COMPOSITE = {
    "parallel_steel02_elastic": ("Parallel", [("Steel02", [400e6, 200e9, 0.01, 20.0, 0.925, 0.15]),
                                              ("Elastic", [20e9])], [], _history(30, 2000, -0.02, 0.025)),
    "series_elastic_steel02": ("Series", [("Elastic", [400e9]),
                                          ("Steel02", [400e6, 200e9, 0.01, 20.0, 0.925, 0.15])],
                               ["-iter", 200, 1.0e-14],
                               # a growing cycle in small steps: OpenSees' iteration from the
                               # last split does not survive a jump of 0.013 in one step (its
                               # stress went to -1.75 GPa there and never came back)
                               0.02 * np.linspace(0.0, 1.0, 4001) * np.sin(np.linspace(0.0, 10.0 * np.pi, 4001))),
    "gap_elastic_pp_gap": ("ElasticPPGap", [], [2.0e9, 1.0e15, 0.002], _history(32, 1000, -0.004, 0.01)),
}

OUT_COMPOSITE = os.path.join(os.path.dirname(OUT), "opensees_composite.npz")


def run_composite(wrapper, parts, wargs, history):
    import openseespy.opensees as ops
    ops.wipe()
    for i, (material, args) in enumerate(parts):
        ops.uniaxialMaterial(material, i + 1, *args)
    tag = len(parts) + 1
    if parts:
        ops.uniaxialMaterial(wrapper, tag, *range(1, len(parts) + 1), *wargs)
    else:
        ops.uniaxialMaterial(wrapper, tag, *wargs)
    ops.testUniaxialMaterial(tag)
    out = np.empty(len(history))
    for i, e in enumerate(history):
        ops.setStrain(float(e))
        out[i] = ops.getStress()
    return out


def main():
    import sys
    from importlib.metadata import version
    data = {"openseespy_version": np.array(version("openseespy"))}
    if "--composite" in sys.argv:
        for name, (wrapper, parts, wargs, history) in COMPOSITE.items():
            for i, (material, args) in enumerate(parts):
                data[f"{name}/part{i}"] = np.array(material)
                data[f"{name}/part{i}_args"] = np.asarray(args, float)
            data[name + "/args"] = np.asarray([a for a in wargs if not isinstance(a, str)], float)
            data[name + "/strain"] = history
            data[name + "/stress"] = run_composite(wrapper, parts, wargs, history)
        np.savez_compressed(OUT_COMPOSITE, **data)
        print("wrote", os.path.normpath(OUT_COMPOSITE))
        return
    for name, (material, args, history) in CASES.items():
        data[name + "/args"] = np.asarray(args, float)
        data[name + "/strain"] = history
        data[name + "/stress"] = run(material, args, history)
    np.savez_compressed(OUT, **data)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
