# Verification contracts and status

Contracts are independent acceptance criteria for numbered cases. They are not
solver inputs. Four classifications are used:

- `analytical`: closed-form mechanics;
- `public_benchmark`: published target external to this project;
- `independent_reference`: separately assembled/integrated solution;
- `regression`: project baseline or physical invariant, never presented as a
  public benchmark.

Run a case and verify the exact result:

```powershell
conda activate vfife
python -m vfife run cases/case_01_beam_cantilever.yaml
python -m vfife verify cases/case_01_beam_cantilever.yaml
```

Case 07 exercises conforming shared-node solid–truss assembly. Its contract
compares the dynamic-relaxation result with an independent direct linear
stiffness solution and checks global reaction equilibrium.

A metric is a `source` (`monitor`, `diagnostic`, `field`), a `name`, a
`reduction` and an `expect`. Reductions over one record: `final`, `first`,
`delta` (final minus first: what the last step added over the state the
record starts in -- a live load's increment over the dead load), `peak_abs`,
`max`, `min`, `mean`. Three read a second monitor named in `against`, so a
contract can hold the shape of a load-deflection curve and not only its peak:

```yaml
- {id: load_at_6p5mm, source: monitor, name: P_L, reduction: at,
   against: mid, value: -0.0065, expect: {target: 146022.0, rtol: 0.03}}
- {id: deflection_at_peak, source: monitor, name: mid, reduction: at_peak_of,
   against: P_L, expect: {target: -0.010070, rtol: 0.05}}
- {id: work, source: monitor, name: P_L, reduction: energy, against: mid,
   from_origin: true, expect: {target: 1614.4, rtol: 0.03}}
```

`at` interpolates this monitor where `against` first reaches `value` (signed,
as the monitor records it; a level never reached is a FAIL that names the
record's range). `energy` is the trapezoidal area under this monitor against
the other, reported as a magnitude (`signed: true` keeps the sign, for a cyclic
record's net dissipation). `at_peak_of` reads this monitor at the frame where
`|against|` is largest. `from_origin: true` prepends (0, 0) to both records,
since monitors hold only the converged frames; it is opt-in because a record
that starts under a previous step's load does not start at zero.

The command returns `0` on PASS and `2` on FAIL. It writes backend-specific
JSON and Markdown under `results/verification`; GPU and CPU runs therefore have
separate report names and execution metadata.

## Status on 2026-08-27

| Case | Class | NumPy | Numba | Warp/CUDA |
|---|---|---|---|---|
| 01 beam cantilever | analytical | PASS | not run | not applicable |
| 02 truss bar | analytical | PASS | not run | not supported |
| 03 mixed frame | analytical | PASS | not run | not supported |
| 04 NAFEMS LE10 | public benchmark | not run this audit | historical result within 15% contract | not run |
| 05 space-truss seismic | independent reference | PASS (1.90% from Newmark) | not run | not supported |
| 06 RC column cyclic | regression | not configured | **PASS (7 checks)** | one-line change from the numba case |
| 07 coupled solid–truss | independent_reference | PASS | not supported | not supported |
| 12 Alberta SF-0B four-point | public_benchmark | not run this audit | PASS (7 checks, 0.90 of measured at the measured peak deflection) | not run |
| 12b SF-0B with bond-slip | public_benchmark | not run this audit | PASS (7 checks, 0.90 of measured) | not supported |
| 12c SF-0B with confined core | public_benchmark | not run | PASS (7 checks, 0.90 of measured) | not run (two blocks on the resident loop are supported) |

Case 06 has a baseline for the first time. It was unverified
for the life of the project because it asked for `backend: warp` — a
256-element model that nobody without an NVIDIA card could run — and because
two of its numbers were taken with `host_deduction` in place. It now runs on
numba in two minutes, goes to ±1% drift (the note saying that diverged
outlived its cause by two stages), and its contract asserts the shape of the
hysteresis loop: the peak shear, the shear at +1% drift as well as the
envelope, and the energy the loop dissipates.

“Not run” is deliberately distinct from PASS. A status changes only when the
corresponding result is generated and accepted by `vfife verify`.

## Cases 12 and 12b: what a PASS here does and does not mean

Both contracts pass over the range they assert and both say, at length and in
their own files, what they stop short of. Read them before quoting either as
a validation of this solver for reinforced concrete beams:

- Case 12 is driven to 19.16 mm at midspan — the deflection at which SF-0B
  was measured to peak — and carries 360.1 kN there against the measured 399,
  ratio 0.90, still rising slowly. It does not crush, which is the reported
  failure mode ("flexural (tension) failure").
- Until v0.1 it peaked at 330 kN and 10.1 mm and came down, because
  the compression curve had no length scale: a code curve is a strain curve,
  so the crushing band's energy per unit area scaled with the element size
  (peak 334 / 330 / 321 / 300 kN over a 4:1 depth refinement). With the
  crushing energy stated (`Gc: 35.0e3`, a literature value, not fitted) the
  spread is 2.4% and the beam reaches a flexural plateau. Its contract has
  the table and the history, including three earlier decompositions of the
  gap that this measurement overturned.
- Case 12b reaches the same end with bond-slip, 360.4 against 360.1 kN,
  1–3% softer in the middle of the path. Its bars are deliberately left
  without `anchorage`, so the A/B against case 12 stays one line.
- Case 12c gives the 192 core elements a Mander-confined curve and lands
  within 0.1% of case 12: with the band regularised this beam never crushes,
  so confinement of the core is never exercised. That is the result the case
  now exists to hold — the 12.5% once attributed to confinement was the
  missing length scale.
- All three hold the SHAPE of the curve, not only its end: the load at 6.5 mm,
  the load at the measured peak deflection, and the work done over the path.

The measured deflection axis of this test is not trustworthy and the contract
for case 12 explains why: the reported 0.50 mm at first cracking is 1.8× softer
than gross-section elastic theory, which no constitutive model accounts for.
Comparisons are made on load at a given deflection, and `Ec` was deliberately
not tuned to absorb the difference.

## Where the numbers come from

A contract that cites a published figure reads an EXTRACT: a two-column text
file beside the contracts, `Chen2005_hollow_slab_1.txt`,
`Chalmers2025_PRC_four_point.txt`, `UAlberta_SF-0B_four_point.txt` and the
rest. Each carries its source in the file header -- author, year, figure
number, DOI where there is one -- and that header is the only provenance the
repository needs.

THE RAW MATERIAL IS NOT HERE. The datasets and papers those extracts were
taken from live outside the repository, in `../data/`, one directory per
source with a `SOURCE.md` saying what was taken from it and into which
extract. They are large (2.3 GB for the Chalmers dataset alone), they are
somebody else's copyright, and they never belong in a git working tree.

So the link is written at both ends: the extract's header points back at the
paper, and the paper's `SOURCE.md` points forward at the extract. Without the
raw data the repository still verifies -- the contracts read the extracts.
With it, you can check that the extraction was done right.

Adding a new measured case therefore has three steps and not two: put the
download in `../data/<source>/` with a `SOURCE.md`, digitise what the
contract needs into `verification/<Source><Year>_<what>.txt` with a header
citing it, and write the contract against that.
