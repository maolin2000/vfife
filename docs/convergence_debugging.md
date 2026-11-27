# Debugging convergence

The order to look at a case that does not converge, how the criteria of the two solvers differ, and a few normalisation traps that make a correct model report failure. Status codes and what a failed run writes are in `diagnostics.md`; this page is about using them to find the cause. It applies to staged loading with `mode: relaxation`.

## The order

The first three steps change no parameter.

**1. Read the whole failure message.** A fibre beam needs the residual **and** the kinetic-energy ratio to pass together; both are printed with their tolerances:

```
MAX_STEPS: fiber beam load stage 1/24 did not converge in 3000 iterations:
  residual=4.253e-03 (tol 5.0e-04), kinetic_ratio=0.000e+00 (tol 1.0e-10)
```

See which one failed first -- their causes and remedies are entirely different. A residual far below its tolerance that is still reported as not converged is trap B below.

**2. Read the `resets` column to tell "slow" from "stalled".** It counts the kinetic-energy resets and tells the outcome earlier than the residual does. Tens to hundreds per stage is normal; a reset every 2-3 iterations is oscillation in place, and **a larger budget does not help**. There is one way to tell: raise `max_steps_per_load` five times and run one stage to see whether the residual falls. One measured case: 1.8e-02 at 600 iterations, 1.5e-02 at 3000 -- stalled.

**3. Take the location from the failed result.**

```python
import h5py
f = h5py.File('results/xxx.h5')
print(dict(f['diagnostics'].attrs)['location_json'])
# {"array":"residual","index":[26],"node":4,"dof":2,"value":9740.6}
```

`dof` numbers `ux uy uz rx ry rz` from 0 to 5. Here a transverse degree of freedom of a middle node is out by 9.7 kN: a problem of internal transverse equilibrium, unrelated to the axial force or the material.

**4. Only now change parameters**, the mesh and the load stages first, the tolerances last.

## The two solvers' criteria differ

The same `tolerance` does not mean the same on both.

| | continuum `solver.py` | fibre beam `fiber_beam_solver.py` |
|---|---|---|
| residual normalised by | `max(‖f_int‖, ‖f_ext‖)` | `max(‖F_ext‖, 1.0)` |
| kinetic criterion | none | `KE/(‖F‖·‖q‖) ≤ kinetic_tolerance`, **together** with the residual |
| before convergence may be declared | `stage_ramp_iterations`, `max_steps_per_load/5` by default | none |
| defaults | `tolerance 1e-4` | `tolerance 1e-4`, `kinetic_tolerance 1e-10` |

The fibre beam normalises by the external force alone, which grows into orders of magnitude in two cases:

**Trap A -- under pure displacement control the tolerance becomes absolute newtons.** With no external load `‖F_ext‖≈0` and the denominator falls back to `1.0`: `tolerance: 5e-4` means a residual below 0.0005 N. The symptom is thousands of iterations per stage to converge at all; a measured 1.5 m cantilever fibre-beam cyclic case took about 16 s a stage, 2.3 hours for 512 stages. To recognise it: the case has only `constraints` with a `value` and no `loads:`. Remedy: set `tolerance` for the size of the force in absolute terms, or add the load that physically exists anyway (self-weight, axial compression) to give the denominator a meaning again.

**Trap B -- under a large load with small displacements the kinetic criterion cannot be reached.** With `‖q‖` near zero the denominator collapses. Measured: 819 kN axial compression, the column top shortened 0.30 mm, residual 2.06e-05 (tolerance 5e-4, 20 times inside) and kinetic ratio 7.77e-07. Remedy: set `kinetic_tolerance` for that case (e.g. `1e-6`) and write down why; the main criterion still guards it.

## Where a parameter takes effect

A parameter written in the wrong place is not refused; its default is used silently, and it looks like non-convergence.

| parameter | `solver:` | in a `steps:` entry | default |
|---|---|---|---|
| `load_steps` | yes | yes (the step's wins) | 20 |
| `max_steps_per_load` | yes | yes (the step's wins) | 5000 |
| `tolerance` | yes | continuum | 1e-4 |
| `kinetic_tolerance` | yes | — | 1e-10 |
| `relaxation_safety` | yes | — | 0.6 |

Set cheap and expensive steps apart: an axial-load step only applies a constant load, and 12 stages are plenty; a cyclic step needs hundreds. Sharing one `load_steps` makes the axial step cost tens of times more for nothing.

## Common causes

**Elements too short, sections too stiff.** The fictitious mass is `row_sum(K)/relaxation_safety²`; the stiffer the section and the shorter the element, the faster the row sums grow, the transverse modes get heavy and stop advancing. The symptom is stalling at the first stage, with the diagnostics pointing at a transverse degree of freedom of an interior node. Measured (400×400 section, 1.6 m column): 8×200 mm stalls, 6×267 mm converges at about 9 s a stage, 4×400 mm converges at about 3 s a stage. Try longer elements first; the cost is the resolution of the spreading plastic hinge.

**Load stages too large.** The path's length divided by `load_steps`: when one stage's increment approaches the characteristic displacement of cracking or yield, refine.

**A larger `max_steps_per_load` makes it slower -- the easiest one to fall for.** The ramp of staged relaxation is `max_steps_per_load // 5` by default, and **a stage may not be declared converged before its ramp ends**. Raising the budget from 60 000 to 200 000 therefore also raised the shortest time to convergence from 12 000 to 40 000 steps: the case sits idling on a residual far below its tolerance, looking "slow to converge" while it converged long ago. `case_24e` idled for forty thousand steps at a residual of 1e-9 against a tolerance of 1e-5 for this reason.

* Symptom: the residual is far below the tolerance and it does not stop; `resets` keeps rising, the stage number does not advance.
* Remedy: write `stage_ramp_iterations` explicitly. The ramp exists so as not to shock the explicit integrator and path-dependent materials; a **linear elastic** model (or any without path-dependent state) needs a few hundred steps.
* The converse holds too: **do not raise the budget casually**. It is also the length of the ramp.

## Slow on a GPU: find out which kind of slow first

`benchmarks/warp_loop_cost.py --n 16 32 64 128 --per-kernel` measures four costs of one iteration separately: the per-step synchronisation, the measures kernel, the Python-side launches and each kernel itself. How to read them is printed at the end. Experience says **the measured cause is never the guessed one**, so do not guess.

One that concerns convergence: the `A/E` ratio is how much faster changing the loop plumbing would make it, independent of the iteration count. **If your case takes too many iterations (hundreds of thousands), iterations seven times faster will not save it**: look at `relaxation_safety`, the density of load stages and the stiffest mode `vfife probe` reports, not at the GPU.

## Run `vfife probe` first

Since 0.1, before suspecting the solver, ask the model four questions that do not depend on the element type:

```powershell
python -m vfife probe case.yaml                       # reference configuration + a 0.8 rad arc state
python -m vfife probe case.yaml --result result.h5    # the state it stalled in
```

It reports:

| question | failing means |
|---|---|
| is the internal force a gradient (tangent symmetry) | asymmetric **near equilibrium**: the force field is not conservative, kinetic damping is a descent method, and no budget will help. Far from equilibrium the co-rotational tangent is asymmetric anyway (in proportion to the out-of-balance force); the report marks it SKIP and says so |
| is the load a gradient | the commonest trap of follower loads -- 0.1's face moment was one. A load depends on the state alone, there is no "far from equilibrium" excuse: it is a verdict in any state |
| are rigid translation and rotation free | the one direct symptom of a wrong co-rotational frame, polar decomposition or orientation state |
| does the fictitious mass still cover the stiffest mode | the ratio of ω² to its limit 4s². Near or above 1 is a step-size problem (the commonest cause of stalling at the first stage); well below 1, leave `relaxation_safety` alone |
| is this state itself stable | the eigenvalue of the softest mode and its x/y/z content. Negative, and out of plane: the solver is not failing to find an equilibrium, the state has no stable equilibrium to find (as with 0.1's rolling-up without a symmetry plane) |

For models up to a few thousand degrees of freedom: the tangent is a finite difference over every degree of freedom, and above `--max-dofs` it is skipped with a note. The library's own 17 elements are asked the same questions at every commit (`tests/test_invariants.py`).

**A load that is not a gradient.** Kinetic damping is a descent method: it assumes the force field has a potential. A non-conservative force (a tangent with an antisymmetric part) keeps pumping energy into the system; the symptom is **a surge of resets** (normally a few to a few tens a stage, here one every one or two hundred steps), a residual swinging between 1e-4 and 1e-3 without coming down, and worse the further it is loaded. 0.1's solid cantilever rolling up was this: the face moment was first written as "integrate the textbook stress again over the current face", a force that is no potential's gradient, with an almost purely antisymmetric tangent -- a thousandth of the element stiffness, but a thousand times the softest mode of the rolled beam. The test is the finite-difference tangent: `python -m vfife probe case.yaml [--result result.h5]` reports |K−Kᵀ|/|K| of the element part and the load part separately, the generalised eigenvalues against the current fictitious mass (the largest must be below 4s², a negative one is an unstable state) and the x/y/z content of the softest modes. For small models (up to a few thousand degrees of freedom). The same probe tells "the solver stalls" from "this state is unstable": the rolling-up without a symmetry plane has, at 306°, two negative eigenvalues both out of plane -- lateral-torsional instability of the discrete model, not a fault of the solver.

**Mass that does not follow rotation** (`mass: eigenvalue`). Underwood's row sum is a quantity of the reference configuration; after large rotations the row sums change and the mass does not, which may in principle become unstable. `mass: eigenvalue` bounds it by each element's largest stiffness eigenvalue (Hughes' element eigenvalue inequality), valid in any configuration, at the cost of a step 1.5-2 times smaller. It was written for the rolling-up problem, which turned out not to be its problem (the stiffest mode reached only half the limit); it stays as an option with a guarantee -- **measure with the probe before switching**.

**Do not tune `relaxation_safety` on a hunch. Either direction may be worse.**

- **Lighter** (a larger value): measured, 0.6 ran through 12 axial stages where 1.2 reached stage 3 and 1.6 stage 5, with resets rising from 136 to 228. A lighter mass oscillates more, and every reset sets the velocity to zero.
- **Heavier** (a smaller value): worse as well on **softening** problems. Measured on a solid RC column under cycles: 0.3 failed at load stage 37 of 40, 0.15 (mass ×4) earlier at stage 20, 0.075 (mass ×16) at stage 8. More mass cannot handle a negative tangent stiffness; it only makes each step advance less and run into the stage budget sooner.

This parameter has no safer direction. **Measure before changing it.**

**A slow first run is not a convergence problem.** numba and warp compile their kernels the first time. In one measured run of 3693 s about 94% was one-off compilation; with a warm cache the same run took 207 s. If it "hangs" before iterating, check whether it is compiling.

## Two changes that make old cases behave differently

Not misuse. A case that used to run and meets one of these: compare with this first, before looking for a cause in the model.

**The iteration budgets of beams, shells and fibre beams grew.** These paths lacked the central difference's **half-step start**: a restart used `v = a·dt` instead of `v = a·dt/2`. The solid and truss paths were always right, and their comments record the cost -- each restart injects a little energy, and kinetic damping restarts every few steps. That energy made the model advance faster, and it was wrong. Fixed, it is **more correct and slower**:

| | old budget | stopped at | needs |
|---|---|---|---|
| `case_03_mixed_frame` | 8000 | residual 6.6e-6 against 1e-6 | 16000 (set to 20000) |
| a fibre-beam cantilever case | 600 | residual 9.5e-4 against 1e-4 | 3000 |

To recognise it: only `beam2` / `beam2_timoshenko` / `beam2_fiber` / `shell3` / `shell4` / `mixed_line` / `coupled`; the residual **falls slowly** rather than sticking. Remedy: more `n_steps` or `max_steps_per_load`. **The converged answer did not change** -- both were checked against their contracts.

**The default damping of `mode: dynamic` changed.** All six solvers said "kinetic for relaxation, viscous for dynamics", but the schema's default set `damping.type` to `kinetic`, and the intention was never carried out. Every dynamic case without `damping` therefore got **kinetic damping**, which sets the velocity to zero whenever the kinetic energy turns down -- a device for finding the **static solution**, which wipes out the dynamics. The default now resolves to viscous (`coeff` 0, i.e. undamped).

**Write the damping of a dynamic case explicitly**: `damping: {type: none}`, or a Rayleigh `alpha`. `alpha` now **acts under any `type`** (before, only under `type: rayleigh`); `{type: kinetic, alpha: 0.002}` is why: kinetic damping resets only at a **peak** of the kinetic energy, a mode whose kinetic energy rises monotonically (a softening element's) never reaches one, while `alpha·M·v` opposes velocity whatever the energy is doing. It is not free: on a path that converges with nothing added, `alpha=0.002` moved the RMS base shear by 1.18%, `alpha=0.005` by 6.57% -- **in an ill-posed softening problem this chooses the branch of the solution**. Use the smallest value that runs, and report it.

## Where relaxing a tolerance stops

**Acceptable**: when a criterion has lost its physical meaning in a step (such as the axial step of trap B), relax it and write the reason in the case's comments; the main criterion still guards it.

**Not acceptable**: to make a curve look right, turn a contract green or close a task. That turns a problem waiting to be found into one that never will be. If it must be relaxed for the time being to stop a nightly run from being permanently red, say in the contract that "this does not mean the ratio is acceptable" and record it as open.

Every number on this page comes from one investigation of "the model looks wrong". The first explanation offered sounded reasonable -- a material unloading too steeply -- and went into the documentation, delaying the real cause (the solver committing path-dependent material state at every iteration) by a whole round. **Path-dependent materials with an iterative solver: suspect the state commit and the normalisation of the criteria before the constitutive law.**

## Known inconsistencies

Not misuse, but the current state of the implementation; do not look for an explanation in the model.

- The two solvers normalise the residual differently; remember it when comparing how hard convergence is across them.
- The element fields `axial_force` / `bending_y` / `bending_z` and so on are **signed** since 1.0 (tension positive, local axes right-handed, the larger magnitude of the two ends with its sign); the old unsigned envelopes are `*_max_abs`. The sign convention is in docs/result_format.md.
- These fields are **peak magnitudes over the integration points**, not the value at one section; mind that in a static check against the base shear.
