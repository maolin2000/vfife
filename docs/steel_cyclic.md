# steel_cyclic: cyclic steel (Cai et al. 2023, §3)

A uniaxial steel law with a **degradation curve** and an **ultra-low-cycle-fatigue fracture strain**. The fracture framework of 0.1 needs it: with a latch alone, the stress of a breaking fibre drops to zero in one step, a finite jump in the constitutive function, and 0.1 showed that this shape traps an explicit solver in a limit cycle. With this law the stress first falls continuously along the degradation curve to `f_uf`, and the latch removes only a residue.

## Input

```yaml
materials:
  steel: {type: steel_cyclic, grade: Q345B, E: 2.06e11, rho: 7850.0}
```

`grade` is a row of the paper's Table 1 (`Q345B` / `Q235B` / `HPS485` / `HT440`) and gives all eleven parameters at once. They can also be given one by one (without `grade` they **must** all be given):

| parameter | meaning | source |
|---|---|---|
| `fy`, `fu` | yield and ultimate strength | Table 1 |
| `k1`, `k2` | end of the yield plateau, end of hardening (in units of εy) | Eq. (10) |
| `k3` | `k3σy` of Fig. 4, i.e. `fu/fy` for a self-consistent curve; used to compute `fu` when it is not given | Fig. 4 |
| `a`, `b0`, `b1`, `b2` | the shape of the hysteretic skeleton (Ramberg–Osgood) | Eq. (11) |
| `eps_cri` | monotonic fracture strain (coupon value) | Eq. (13) |
| `lambda_dsps` | ULCF fatigue parameter | Eq. (13) |
| `f_uf` | nominal failure stress, **not in Table 1**, `0.5 fu` by default | Eq. (12) |
| `skeleton_fit` | `shift` (default) \| `raw`, see below | Eq. (11) |

Output fields: `eps_max`, `eps_acc_c` (accumulated compressive plastic strain), `eps_uf` (the current fracture strain), `failed`. `failure()` connects to the framework of 0.1; `damage_measure()` reports `1 − exp(−λ·ε_acc)` (fatigue life used, never decreasing; not "current strain / current fracture strain", which falls on unloading).

## Four curves and five rules

| branch | when | equation |
|---|---|---|
| monotonic curve | first loading (tension or compression) | (10) |
| hysteretic skeleton | loading after any reversal | (11) |
| degradation curve | after the curve in use has reached `fu` | (12) |
| reloading curve | after unloading to zero stress, before the peak point | (14)–(18) |

Unloading is always elastic (slope E); that is all of rule (2). The peak point is **isotropic**: its strain is the largest **absolute** strain the point has reached, of either sign -- the paper's reading of isotropic hardening for steel, and why in Fig. 7 A is recorded at +ε₁ and E lands at −ε₁.

There are two reloading curves, chosen by the secant stiffness `E_k` (Eq. 14): Légeron for `E/5 ≤ E_k < E`, Shi–Wang for `0 < E_k < E/5`. **The wider the loop, the smaller `E_k`**, so the amplitude decides which applies.

**Two things the paper leaves unsaid, added here:**

- **A small reversal on the monotonic curve is not a cycle.** Rule (5) says loading after a reversal follows the skeleton. Taken literally, stepping back 1e-5 on the plastic plateau and pushing forward again makes the stress jump from the monotonic curve onto the skeleton: Q345B at 2% strain jumps 461 → 541 MPa in one step, 561 → 584 in the hardening range. A point that reverses on the monotonic curve and is pushed back before its stress changes sign now returns to the monotonic curve it left, changing to the skeleton only where the skeleton (or its degradation curve) is lower. A point that really unloads past zero stress and reloads follows the skeleton, as the paper says.
- **Reloading with `E_k ≥ E` does not jump onto the skeleton at the peak strain.** When the secant is steeper than E, Eq. (14) can only follow the elastic line, and at the peak strain the stress is still below the peak point; it used to jump onto the skeleton there: pushed to 2%, unloaded into compression and reloaded, a strain step of 1e-4 jumped 92 MPa. It now continues with slope E until it meets the skeleton.

There are two `eps_u` (necking strains), which is the point of Fig. 5: the monotonic curve reaches `fu` at `k2·εy`, the skeleton earlier. For Q345B they are **9.20%** and **3.61%**: cycling brings necking forward, and degradation starts earlier.

## Three printed equations that do not close

Each is pinned in `tests/test_steel_cyclic.py`; the changes are on the record, not hidden.

**The hardening coefficient of Eq. (10)** is printed as `E(1−k1)/(εy(k2−k1)²)`. With the numbers of Table 1 the curve jumps by `fy(1−k1)` at `ε = k1·εy` -- **−1.5 GPa** for Q345B, and in the wrong direction. The coefficient that takes the parabola through `(k1εy, fy)` and `(k2εy, fu)` with zero slope at the latter is `(fy−fu)/(εy²(k2−k1)²)`; when `fu = k3·fy` (the curve Fig. 4 draws) it is **exactly the printed one with k1 replaced by k3**. The latter is used.

**The degradation curve of Eq. (12)** is printed as `fu + (fu−fuf)·((ε−ε_uf)/(ε_u−ε_uf))²`, which gives `2fu − fuf` at necking and `fu` at fracture -- wrong at both ends. The form that agrees with Fig. 5 (flat at necking, steep at fracture) is `fu + (fuf−fu)·((ε−ε_u)/(ε_uf−ε_u))²`. The latter is used.

**The skeleton of Eq. (11) does not meet the elastic line**: at ε̄ = 1 it gives 1.030 fy, a 3% step at first yield. `skeleton_fit: shift` (the default) subtracts this constant, leaving the slope unchanged everywhere; `raw` reproduces the paper faithfully.

When is the step reached? **Only when a reversal's peak strain lies near yield** -- below the peak the reloading curve applies, and the skeleton is not evaluated there at all. A path built to reach it (reverse a little below yield, then push past yield) measures: `raw` jumps **13 MPa in one step, 35 times the elastic slope**; `shift` gives 1.00 (purely elastic). That is the shape of the sticking defect of 0.1, so the default removes it. (Since 1.0 `raw` does not jump on this path either: that step was the `E_k ≥ E` reloading meeting the skeleton, described above, which now follows E to the intersection. The 3% of Eq. (11) at yield is still there, and `skeleton()` still measures it.)

## A row of Table 1 that does not fit

**HT440 is refused by the constructor**: `k3 = 1.2` but the measured `fu/fy = 1.60`, while its four Ramberg–Osgood parameters are exactly those of the other three rows. With the measured `fu = 800 MPa` the skeleton reaches it only at **14.8%** strain, the monotonic curve at **11.2%** -- **cycling would delay necking**, the opposite of the premise of Fig. 5. Such a parameter set would run through a whole analysis quietly, so it is refused on construction, with the ways out: lower `fu` to `k3·fy` (601 MPa), or fit `a, b0, b1, b2` again for this steel. Raising `k3` to 1.60 **does not help**: the skeleton itself is in the way.

## What can be checked against the paper

| | paper | here |
|---|---|---|
| cyclic over monotonic hardening (ε̄ = 10) | "about 20%" | 17.3% (shift) / 21.2% (raw) |
| peak stress of ±2.4% cycles (Q345B) | Fig. 12(b), about 550 MPa | 555 MPa |
| monotonic tension to fracture (Q345B) | `ε_cri = 0.36` | peak 586 MPa at 9.20%, fracture at 36.0% |
| compressive cycles shorten the fracture strain | Eq. (13) | after six −5% excursions `ε_acc = 0.269`, fracture strain 36.0% → 33.4% |

## Performance

The scalar law is compiled with numba, **82 ns/point** (39 µs for 480 fibre points), the order of steel02's 68 ns. The same source is compiled: `_point.py_func` is the Python version, the two paths are **bit-identical** over a 3700-step cyclic history, and the fibre beam's fused kernel inlines the same `_point`.

The law therefore **fills a gap left in 0.1**: the three laws the kernel could inline (concrete01/02, steel02) have no fracture criterion, so a section on the kernel path could not break by itself; now it can, and the fracture events of the two paths agree bit for bit (`tests/test_steel_cyclic.py::test_the_fused_kernel_now_has_a_law_that_can_fracture`).
