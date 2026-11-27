# Steel02 reinforcing steel (Giuffrè–Menegotto–Pinto)

The default for cyclic loading. `rebar_gmp` was another name for the same law and was removed in 0.1: over 18 parameter sets and a 2100-step cyclic path the two differed by 0.42% (of the peak stress), and it was 30 times slower with no scalar twin. A case that writes `type: rebar_gmp` is refused with a pointer here.

## Choose the law first, then its parameters

The four uniaxial steel laws, measured on the same ±1% cyclic path (the `steel02` row was measured again after 1.0 changed it to OpenSees' Steel02; before, 0.58 εy and ×1.08):

| | rounding at reversal | peak after 6 cycles | energy per cycle | fracture | fused kernel | ns/point |
|---|---|---|---|---|---|---|
| `rebar` | none (drops to 0.014E in one step) | ×1.36 | grows 17% | `rupture_strain` (accumulated plastic strain) | no | 105 |
| `steel02` | 4.61 εy | ×0.95 | steady | `rupture_strain` (peak tensile strain) | **yes** | 147 |
| `steel_cyclic` | 4.87 εy | ×1.16 | steady to the bit | degradation + ULCF | **yes** | 82 |
| `piecewise_uniaxial` | none | ×1.57 | grows 23% | none | no | 3700 |

- **Monotonic loading, pushover**: `rebar`. Bilinear, the cheapest, and on the solid and truss paths it also has numba and warp backends; give `rupture_strain` for it to break.
- **Cyclic, seismic**: `steel02`. Rounded reversals, stable hysteresis. Give `rupture_strain` for it to break too -- but **it does not measure what `rebar`'s does**, see below. Since 1.0 it is the same model as OpenSees' Steel02 (on random histories it agrees with the OpenSees program's output to round-off); the 3.45% by which the envelope used to exceed its asymptote is gone, see the last two sections.
- **Collapse, fracture, ultra-low-cycle fatigue**: `steel_cyclic` (`docs/steel_cyclic.md`). The only one with a degradation curve and a fracture strain, and the fastest.
- **A measured curve**: `piecewise_uniaxial`. Any piecewise-linear table, but 3.7 µs a point (a table-driven law cannot be inlined into the kernel); for small models or single-point studies.

Two traps: **under constant-amplitude cycles the peaks of `rebar` and `piecewise` keep rising** (+36% / +57% over 6 cycles, as the mixed hardening accumulates), which overstates hysteretic dissipation in a time history; and **a fibre section runs the fused kernel only with `steel02` or `steel_cyclic`** -- any other law sends the whole section back to the numpy path, four times slower. When choosing, this often decides whether a case runs to the end more than the constitutive detail does.

## Input

```yaml
# reinforcement in a fibre beam
materials:
  steel: {type: steel02, E: 200e9, Fy: 350e6, b: 0.01, R0: 20, cR1: 0.925, cR2: 0.15, rho: 7850}
sections:
  bar20: {type: bar, material: steel, shape: general, area: 3.14e-4}
  rc: {type: fiber, material: concrete, shape: rect, width: 0.25, height: 0.35, ny: 8, nz: 8,
       bars: [{y: 0.09, z: 0.14, section: bar20}]}
# a stand-alone material point
#   {type: steel02, E: 200e9, Fy: 350e6, b: 0.01, R0: 20, a1: 0, a2: 1, a3: 0, a4: 1}
```

`b` is the hardening ratio `E_h/E0`; `R0/cR1/cR2` control the curvature of the transition (`R = R0(1 − cR1·ξ/(cR2 + ξ))`, ξ the plastic excursion of the previous branch in units of εy; `0 ≤ cR1 < 1` and `cR2 > 0` are required); `a1/a2` (compression) and `a3/a4` (tension) control isotropic hardening (`a2` and `a4` must be positive). As in OpenSees, first loading also follows the rounded curve: `Fy` is the intersection of the two asymptotes and the curve passes below it -- 0.45% below `E·ε` at 0.89 Fy (R0 = 20). With `b = 1` it is Hooke's law.

State `n_state=13`: `eps_prev/sig_prev/eps_rev/sig_rev/direction/eps_target/hysteretic_energy/eps_max/sig_target/eps_max_os/eps_min_os/eps_pl/work`. `hysteretic_energy` is `∫σdε − σ²/2E`, the dissipated part. Rate-independent and safe to checkpoint; calling it again at the same strain does not change the state.

Qualified `verified` (`qualification/steel02.yaml`), `numpy` and `numba` backends, used in the mixed section of `beam2_fiber` (`case_11`, Tanaka & Park 1990 specimen 1, fy = 474 MPa and b = 0.012 from the measured fu/fy = 1.52).

`update()` runs through a compiled point loop (module-level `@njit(cache=True)`, falling back to `_update_reference` without numba). Measured on 480 fibre points: 1915 → 36 µs (53 times), about 68 ns a point compiled; `tests/test_uniaxial_numba.py` checks the equivalence step by step over a full cyclic history (stress and state differ at the 1e-14 level). Batching matters: for one point it is only twice as fast, as the call overhead dominates.

## `strain_scale`: the share of strain in a bar's anchorage

A bar does not stop carrying force at the member's face; it continues into the anchorage for a length Lsp. Spread over an element of length Le, the bar takes only `Le/(Le + Lsp)` of the section strain. `strain_scale` is that factor, 1.0 by default (full bond, as in every case before the parameter existed).

It is a **parameter** of `steel02`, not a material of its own, because of the fused kernel: the kernel inlines one steel law, and a second law type sends the whole model back to the NumPy path. Different values per element are carried by `spar[e, f]`, and the reference path reads the same parameter rows through `update(..., params=)`.

`tests/test_strain_penetration.py` checks the defining property over a full cyclic history: a material with `strain_scale=k` at strain eps gives the stress of the unscaled material at `k·eps` (difference < 1e-12). Its use in cases is in `docs/fiber_beam.md`.

## `rupture_strain`: one name, two different measures

```yaml
strand: {type: steel02, E: 1.9e+11, Fy: 1.58e+9, b: 0.065, rupture_strain: 0.035}
```

| | criterion | where the number comes from |
|---|---|---|
| `rebar` | **accumulated plastic strain** `alpha`, which grows at every reversal | fitted, or estimated from low-cycle fatigue |
| `steel02` | **peak tensile strain**, the largest the point has seen | A_gt of a tension test (≥ 3.5% for 1×7 strand by GB/T 5224) |

**The shared name is deliberate** (a case remembers one key), **and so is the different criterion**: `steel02` is Menegotto–Pinto, its corner is rounded, and its state has no plastic strain to accumulate. For a strand **pulled monotonically to rupture** the two agree; for a column bar **under cycles** they are far apart, and that case wants `rebar` or `steel_cyclic`. `tests/test_steel02_rupture.py` pins the difference with a twenty-cycle path: the total strain never exceeds 1%, `steel02` (limit 2%) survives and `rebar` (also given 2%) breaks.

**Compression never triggers it**: `eps_max` starts at 0 and only takes the maximum, so it is monotonic, the latch is irreversible, and no extra "broken" state column is needed.

**It is a finite jump in the constitutive function** -- a drop to zero in one step. In 0.1 that shape burnt 8000 kinetic-energy resets at one sticking node, and `steel_cyclic` exists because of it: that law **first falls along its degradation curve to f_uf**, and only a residue is cut at the end. So: when **the fall itself matters**, use `steel_cyclic`; when **the fibre should simply stop carrying force**, use this.

The case behind it is `case_25` (Chen Hengzhi's 13 m hollow-core slab): with 3.5% given, the strand of this beam peaks at **1.91%**, a margin of 1.83, and **none of its nine contracts moved by a digit**. That is the reason to write it: the slab's capacity is now computed with a strand that can break, not one that cannot.

### When fracture is decided: on the **accepted** state, not within an iteration

`rupture_strain` reads the peak strain of the **last accepted state**: a call decides first and updates `eps_max` after. The call that crosses the limit still carries force; the fibre is cut from the next one on.

Why: every iteration of staged dynamic relaxation drives the material points again from the last converged state (a trial state). A load stage puts its whole displacement increment on the loaded point at once, and for the first few dozen iterations the element next to it takes that increment alone, at a strain of about "increment ÷ h", far above the converged value. The old code cut the fibre in that **trial**; the section lost its strand, and the stage converged on **the broken equilibrium branch**, while the intact one was an equilibrium too.

It looks like mesh dependence, and 0.4 read it so:

| Li 2024 A7 (ρ = 8.31%) | h = 40 mm | h = 80 mm | section analysis | measured |
|---|---|---|---|---|
| 0.4 (cut in the trial) | 19.73 kN·m | 29.49 kN·m | 40.14 | 44.54 |
| 0.4 (cut once accepted) | **39.48** | **39.51** | 40.14 | 44.54 |

A4 (ρ = 4.52%, the strand yields first): 31.14 → **52.65 / 52.71** (measured 53.72). The two meshes differ by 0.1%, and the strand strain is uniform in the constant-moment zone -- **there is no localisation**. 0.4's "tightening the residual tolerance 100 times gives the same result to the bit" was true, but it could not rule this out: the tolerance governs how accurately a branch is reached, not which branch.

The matching solver change: when a stage has converged and been committed and new fracture points have appeared, **the same stage is solved again** (repeatedly for a cascade, up to 50 times) before it is recorded. Otherwise the record would be "the intact positions + the cut internal forces", a reaction of −169 kN on A7. An explicit time history has no trial state; there the delay is one time step.

### The half that still holds: it acts on a **smeared strain**

A fibre's strain is the section's strain. With full bond the **capacity** is not affected -- the section's moment when the strand reaches ε_u is set by section equilibrium, independent of h, and the table above is the evidence. What is affected is the **deformation at rupture**: a real strand is stressed more at a crack than on average between cracks, so it breaks at a smaller average strain.

How much depends on the ratio of the bond transfer length to the crack spacing. Seven-wire strand in a grouted duct bonds weakly (EN 1992-1-1 Table 6.2, ξ = 0.5); on Li's beams the transfer length of an isolated crack is about 0.5 m, far more than the measured crack spacing (≈ the stirrup spacing, 100 mm): the transfer zones of neighbouring cracks overlap, the strand strain is nearly uniform, and full bond overstates the average strain at rupture by 3–12% (bond strength 0.5–2 times). At a crack spacing of 300 mm it is 14–43%. (A first integration of one-dimensional bond-slip; it depends on the particular bond and corrosion assumptions, so it is not in the library.)

## 1.0: changed to OpenSees' Steel02

"The envelope exceeds its own asymptote by 3.45%", recorded in 0.4 and never fixed, came up in the second review with three other findings, all from one source: the law was not OpenSees' Steel02 but a variant.

| | the variant (before 1.0) | OpenSees' Steel02 (from 1.0) |
|---|---|---|
| each branch's target point | ±Fy_eff, the yield strain placed so that the **curve** passes through Fy | the intersection of the elastic line from the reversal point with **the other side's hardening asymptote** |
| degradation of R | `R0 − cR1·ξ/(cR2+ξ)`: with cR1 = 0.925, R hardly moves (≈19) | `R0(1 − cR1·ξ/(cR2+ξ))`: about 2.3 after ±1% cycles |
| `a3/a4` | read, not used | isotropic hardening in tension |
| monotonic envelope | 3.45% above the asymptote | below the asymptote |
| a small reversal of 2e-4 at 2%, then pushed to 3% | 469.8 → **440.3** MPa (the hardened envelope forgotten) | 456.0, with or without the reversal |
| 50 small reversals at 2% | the stress walks back to Fy | pushed back onto the envelope |
| small closed loops | can **create** energy | dissipate |

The implementation is one Python function, and that is what the kernel compiles (in 1.0 it was transcribed from the OpenSees source; it was later rewritten from the original paper, see the last section).

**The cost (in the direction measured in 0.4)**: the capacity of sections whose bars are on the plateau drops about 2.6% (`case_31` midspan section 58.98 → 57.50 kN·m); the rounding at reversal changes from 0.58 εy to 4.61 εy, the loops are "thinner", and each cycle dissipates less than with the variant. The dozen or so cases with `steel02` were all run again; the differences item by item are in the 1.0.0 section of the CHANGELOG.

In `tests/test_material_invariants.py` this item moved out of `_OPEN` into a positive assertion: the envelope must stay below the asymptote.

## Rewritten from the original paper (from 1.20.1)

The 1.0 implementation was ported line by line from the OpenSees source, and OpenSees' licence does not allow its code in an MIT-licensed repository. The present `_point_py` is rewritten from Eqs. (3.4)–(3.8) of Section 3.1 of Filippou, Popov and Bertero (1983, UCB/EERC-83/19), without reference to the OpenSees source:

* the normalised curve of Eqs. (3.4)–(3.6) and R of Eq. (3.7); the paper's a1 and a2 are R0·cR1 and cR2 here;
* isotropic hardening by the documented meaning of `a1..a4` (the yield asymptotes of the compression and tension sides rise by a1·Fy and a3·Fy after plastic excursions of a2·εy and a4·εy), with the excursion taken as (ε_max − ε_min)/2 and the exponent 0.8;
* what the paper leaves open -- that exponent, normalising ξ by the yield strain, the strain the excursion is measured from -- was settled by treating the OpenSees program as a black box and comparing outputs on random strain histories. `tools/opensees_reference.py` generates the reference curves and `tests/test_opensees_reference.py` checks them; the differences are round-off.

The parameters and the state's columns and their meaning are unchanged; old cases and old checkpoints work as before.
