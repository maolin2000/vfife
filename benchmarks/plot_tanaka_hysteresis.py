"""Measured against computed hysteresis for Tanaka & Park (1990) specimen 1.

    python benchmarks/plot_tanaka_hysteresis.py results/case_11_....h5

The measured record is verification/Tanaka_90_specimen_1.txt: a header line, a
count, then displacement (mm) and lateral force (kN) pairs. The solved result
carries the same two quantities as monitors, in metres and newtons.

WHAT TO LOOK AT, IN THE ORDER THAT MATTERS
------------------------------------------
1. Peak force. If the envelope is off by more than ~15% the section, the
   materials or the axial load are wrong -- not the hysteresis rule. The case
   file carries a hand check putting the capacity near 155-167 kN.
2. Initial stiffness. A fiber model with no bond slip and no shear deformation
   is stiffer than the specimen; a factor near 1.5 is normal and is worth
   stating rather than tuning away.
3. Loop shape near zero drift. Pinching comes from crack closure. The model
   has concrete02's own closure rule, and one concrete for the whole section,
   so it cannot show cover spalling -- expect the computed loops to stay
   fatter than measured beyond about 40 mm.
4. The final push past 120 mm. The bars buckled there in the test and the
   measured force drops away; the model has no buckling and will not follow.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import h5py
import numpy as np


def read_measured(path):
    lines = [l for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue          # the count line, and the title
    d = np.array(rows)
    return d[:, 0], d[:, 1]       # mm, kN


def read_computed(path):
    with h5py.File(path, "r") as f:
        status = f.attrs.get("analysis_status", b"")
        converged = bool(f.attrs.get("analysis_converged", False))
        u = np.asarray(f["monitors/tip_disp"]) * 1e3        # m  -> mm
        # base_shear monitors the REACTION at the fixed end, which is equal and
        # opposite to the lateral force pushing the head. The measured record is
        # the applied force, so one of them has to be flipped or the computed
        # loops come out mirrored -- corr(u, reaction) is about -0.8 here, which
        # is what that mistake looks like before you plot it.
        v = -np.asarray(f["monitors/base_shear"]) / 1e3     # N  -> kN, applied
    # Drop the axial step. Its samples sit at u = 0 to within solver noise
    # while the 819 kN ramps on, and that noise is enough to look like a load
    # reversal to anything that detects reversals by sign change -- which threw
    # the secant stiffness to -2276 kN/mm before this line existed.
    start = int(np.argmax(np.abs(u) > 0.05))
    if start:
        start -= 1
    return u[start:], v[start:], status, converged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="solved case_11 .h5")
    ap.add_argument("--measured", default=None)
    ap.add_argument("-o", "--out", default="tanaka_hysteresis.png")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    measured = args.measured or os.path.join(
        root, "verification", "Tanaka_90_specimen_1.txt")

    um, vm = read_measured(measured)
    uc, vc, status, converged = read_computed(args.result)

    if not converged:
        print(f"WARNING: {args.result} did not converge (status {status!r}).")
        print("The curve below is whatever the solver reached, not an answer.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))

    ax.plot(um, vm, color="#444444", lw=1.1, label="measured (Tanaka & Park 1990, No. 1)")
    ax.plot(uc, vc, color="#c1440e", lw=1.3, label="VFIFE fiber beam")
    ax.axhline(0, color="#999", lw=.6); ax.axvline(0, color="#999", lw=.6)
    ax.set_xlabel("tip displacement (mm)")
    ax.set_ylabel("lateral force (kN)")
    ax.set_title("Hysteresis")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.25)

    # The envelope is the part a design check would actually use, and the loops
    # overlay into an unreadable tangle -- so it gets its own panel.
    #
    # Take it at the load REVERSALS, not by binning on displacement. Binning
    # mixes points from different cycles and from unloading branches into the
    # same bin and produces a sawtooth that looks like a numerical defect but
    # is only an artefact of the reduction.
    def envelope(u, v, floor=1.0):
        turns = [0]
        for i in range(1, len(u) - 1):
            if (u[i] - u[i - 1]) * (u[i + 1] - u[i]) < 0 and abs(u[i]) > floor:
                turns.append(i)
        # Only include the final sample if the record is still moving AWAY from
        # zero there, i.e. it really is an envelope point. The measured record
        # ends at 110 mm on the unloading branch after peaking at 129.5, and
        # appending it unconditionally drew the envelope down to zero force --
        # a V-shaped dip that looks like a collapse and is only bookkeeping.
        if abs(u[-1]) >= abs(u[-2]):
            turns.append(len(u) - 1)
        cu, cv = u[turns], v[turns]
        order = np.argsort(cu)
        return cu[order], cv[order]

    eum, evm = envelope(um, vm)
    euc, evc = envelope(uc, vc)
    ax2.plot(eum, evm, "o-", color="#444444", ms=3, lw=1.2, label="measured")
    ax2.plot(euc, evc, "s-", color="#c1440e", ms=3, lw=1.2, label="computed")
    ax2.axhline(0, color="#999", lw=.6); ax2.axvline(0, color="#999", lw=.6)
    ax2.set_xlabel("tip displacement (mm)")
    ax2.set_ylabel("lateral force (kN)")
    ax2.set_title("Envelope")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=.25)

    fig.suptitle("Tanaka & Park (1990) specimen 1 — 400x400 RC column, "
                 "N = 819 kN (0.2 f'c Ag)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")

    # Numbers, so the picture is not the only evidence.
    print(f"\n{'':22s} {'measured':>10s} {'computed':>10s} {'ratio':>7s}")
    for label, f in (("peak force + (kN)", lambda v: v.max()),
                     ("peak force - (kN)", lambda v: v.min()),
                     ("peak disp + (mm)", None),
                     ("peak disp - (mm)", None)):
        if f is None:
            continue
        a, b = f(vm), f(vc)
        print(f"{label:22s} {a:10.1f} {b:10.1f} {b / a:7.2f}")
    print(f"{'peak disp + (mm)':22s} {um.max():10.1f} {uc.max():10.1f}")
    print(f"{'peak disp - (mm)':22s} {um.min():10.1f} {uc.min():10.1f}")

    # Secant stiffness on the FIRST loading excursion only. Searching the whole
    # record for the point nearest 8 mm finds one on an unloading branch of a
    # later cycle and reports a stiffness of the wrong sign.
    def secant(u, v, target):
        end = len(u)
        for i in range(1, len(u) - 1):
            if (u[i] - u[i - 1]) * (u[i + 1] - u[i]) < 0 and abs(u[i]) > 1.0:
                end = i + 1
                break
        i = np.argmin(np.abs(u[:end] - target))
        return v[i] / u[i] if u[i] else np.nan
    a, b = secant(um, vm, 8.0), secant(uc, vc, 8.0)
    print(f"{'secant @8mm (kN/mm)':22s} {a:10.2f} {b:10.2f} {b / a:7.2f}")
    # Energy. The loops' AREA is what a fiber model most often gets wrong, and
    # it is invisible in a peak-force comparison: two curves can share an
    # envelope and dissipate very differently.
    def absorbed(u, v):
        return float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(u)))   # kN.mm = J
    am, ac = absorbed(um, vm), absorbed(uc, vc)
    print(f"{'absorbed energy (kJ)':22s} {am / 1e3:10.1f} {ac / 1e3:10.1f} "
          f"{ac / am:7.2f}")
    print("\n(a fiber model with no bond slip and no shear deformation is stiffer\n"
          " than the specimen; ~1.5x on first loading is the expected penalty.\n"
          " Absorbed energy far below 1.0 is NOT a material property to explain\n"
          " away: it was once 0.32 here because the solver committed fibre state\n"
          " on every relaxation iteration, so the thousands of wiggles inside one\n"
          " load stage each registered as a load reversal and erased the steel's\n"
          " memory of the real excursion. Suspect the solver before the material.)")


if __name__ == "__main__":
    main()
