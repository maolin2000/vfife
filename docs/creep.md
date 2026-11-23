# Creep, shrinkage and ageing

Concrete's strain that arrives with time. Added in v0.4;
`vfife/core/creep.py`.

```yaml
materials:
  concrete:
    type: linear_elastic
    E: 3.87e+10                   # Ec at the reference age
    nu: 0.2
    rho: 2450.0
    creep:     {model: mc90, notional_size: 480.0, rh: 75.0, fcm: 63.0e+6}
    shrinkage: {model: mc90, notional_size: 480.0, rh: 75.0, drying_start: 7.0}
    ageing:    {s: 0.25, of: modulus, reference_age: 28.0}

steps:
- {name: build,  age: 28.0,     load_steps: 1}
- {name: 3_years, to_age: 1123.0, load_steps: 12}   # log-spaced stations

solver:
  age: 28.0                       # the concrete age the analysis starts at
```

All three are optional and independent. A material with none of them behaves
exactly as before. **Each member uses its own material's laws**:
a steel beam beside a creeping concrete one does not creep, and a member whose
material declares none of the three takes no part. Every key of every spec is
checked when the material is made — a key the model does not read (`RH` for
`rh`, `S` for `s`) is refused by name.

## The one idea

Creep, shrinkage and temperature are the **same mechanism**: a free strain,
subtracted from the deformation before the member law is applied,
`f = k (d - d_imposed)`. They differ only in what the imposed strain depends
on.

| | depends on |
|---|---|
| temperature | the field you impose, and nothing else |
| shrinkage | the **clock** |
| creep | the member's whole **stress history** |

And in all three cases what the structure *does* about it is decided by what
is in the way, not by the strain:

* a **determinate** member creeps and its member actions do not move. A
  simply supported beam under a held load sags by `(1 + phi)` times its
  elastic deflection and the reactions do not change by a newton.
* a **restrained** one relaxes. Hold a bar at a fixed strain and its force
  decays; restrain a shrinking member and it develops far less than
  `EA eps_sh`, because it creeps out of most of it.

A creep implementation that reports a changed force on the determinate member
has put the creep strain in as a load. That is the first thing `case_29`
checks.

## The models

**Creep**, `phi(t, t0)` — the creep strain as a multiple of the elastic one:

* `mc90` — CEB-FIP Model Code 1990. Needs `notional_size` (h0 = 2Ac/u in mm),
  and takes `rh` (%, 40–100), `fcm` (Pa, 20–88 MPa: MC90's C12–C80, fcm = fck + 8; outside it is refused —
  above about 110 MPa the shrinkage expression's base strain changes sign), `cement` (4 slow, 5
  normal, 8 rapid high-strength — MC90's beta_sc; it adjusts the loading age,
  `t0,adj = t0 [9/(2 + t0^1.2) + 1]^alpha >= 0.5 d`, alpha = -1 / 0 / 1).
  **MC90's phi is referenced to the 28-day modulus**, `J(t, t0) = 1/Ec(t0) +
  phi/Ec28`, so the solver needs Ec28: from the material's `ageing` law, or
  from `E28` (Pa) in the creep spec when `E` is declared at the loading age.
  One of the two is required and both is refused. The creep strain is then
  `phi Ec(t0)/Ec28` times the elastic one. Until v1.0 phi was taken
  against Ec(t0), 17–22% low on concrete loaded late.
  **Known limit:** with `ageing` instead of `E28` the result is still about
  2% short (a bar loaded at 400 days and held to 1030: growth 1.603 against
  MC90's 1.641), because the ageing scheme scales the whole stiffness by one
  factor and the elastic part drifts from `1/Ec(t0)` towards `1/Ec(t)`.
  `E28` gives the MC90 value exactly.
* `aci209` — the ACI 209 hyperbola, times the loading-age factor
  `gamma_la = 1.25 t0^-0.118` (`curing: moist`, the default) or
  `1.13 t0^-0.094` (`curing: steam`), evaluated for every increment at its
  own loading age (0.994 at 7 days moist). Takes `ultimate`, `curing`.
  `ultimate` is therefore phi_u WITHOUT the loading-age factor. **`ultimate`
  defaults to 2.35, and 2.35 is the UNCORRECTED value**: ACI 209R-92 multiplies
  it by up to six correction factors, and for concrete loaded at a year in a
  marine climate their product is near 0.45. Left at the default on the
  Confederation Bridge the model gives `phi(630 d) = 1.94` where MC90 gives
  0.53, a ratio of 3.6 — against the 1.4–1.6 between the paper's own two
  curves. Supply `ultimate` or the model is not ACI 209, it is a hyperbola.
* `curve` — measured data, `[[t - t0, phi], ...]`, read **linearly in
  log(t - t0)**; or `{t0: [[...]], ...}` for several loading ages, blended
  linearly in `log(t0)`. This is how a project's own creep tests get in.
  The origin `(0, 0)` is implicit: from zero to the first row the value
  rises linearly in `t - t0` (a log axis has no zero). A `[0, 0]` row says
  the same; a row at duration 0 with any other value is refused.

**Shrinkage**, `eps_sh(t)` — negative, and zero before `drying_start`:
the same three model names, with `mc90` the default. `mc90` follows MC90's
`beta_RH`: `-1.55 [1 - (RH/100)^3]` for 40 <= RH < 99, and **+0.25 — swelling,
the member lengthens — at 99% and above**. The jump at 99% is the code's own.

**Ageing**, `Ec(t)/Ec(ref) = exp[s (1 - sqrt(28/t))]`: `of: strength` (the
default) applies `s/2`, because MC90's `s` is a strength coefficient and E
follows the square root of strength; `of: modulus` applies `s` directly,
which is what an analyst who fitted `s` to measured moduli means.

## The time axis

A step declares where in time it goes:

* `age:` — the concrete age this step sits at (no time passes).
* `to_age:` — the age this step runs to, in `load_steps` stations. A
  relaxation step **without** `load_steps` is one station, at `to_age`.
* `time_spacing: log` (default) or `linear`.

The clock only runs forward: a step whose `age` is earlier than the last
recorded frame, or whose `to_age` is earlier than its own `age`, is refused.
Time stations are the levels of a staged relaxation, and since 1.16 the
converged increments of a `mode: static` step with `to_age` on beam and
mixed-line models (fibre models and `static.arc_length` are
refused): the two give the same answers. Creep needs time stations, and a dynamic run has none: **a `mode: dynamic`
step that moves the clock** (a `to_age`, or an `age` other than where the
clock is) **is refused when a material declares `creep`**. Run the creep in
relaxation steps and the dynamics from its result (`solver.initial_state`),
with no `age` / `to_age` in the dynamic steps: the creep and shrinkage the
state reached are then held as a frozen imposed strain, and the crept member
stays at its crept length (v1.0 refused every dynamic run
with `creep`, and dropping `creep` to get past it let the member spring back
to its elastic length).

**Log spacing is the default because that is the axis creep happens on.**
Twenty-eight days to ten thousand in ten linear stations puts nine of them
after the first thousand, where nothing is left to happen, and one across the
decade where most of it does.

Each station is relaxed to equilibrium, its elastic deformation is recorded,
and the next station's imposed strain is the superposition over everything
recorded so far:

    eps_cr(t) = sum_j phi(t, t_j) * (elastic increment that arrived at t_j)

recomputed from the whole history at every station rather than accumulated,
so refining the axis converges to the integral instead of drifting.

The concrete age of every frame is written to the result file beside the
monitors, as `age`. `vfife plot --x age` draws a creep curve, and a
verification contract can say `reduction: at, against: age, value: 365`.

## Two things that are easy to get wrong

**R is not 1/(1 + phi).** The relaxation modulus — what is left of a held
stress — is the solution of the superposition integral, not the reciprocal of
the creep coefficient. `1/(1 + phi)` is the effective-modulus guess and it is
6% high in the middle of the decay; the age-adjusted `1 - phi/(1 + chi phi)`
is the same guess with `chi` fitted to the integral afterwards.
`core.creep.relaxation` solves the integral, which is what lets `case_29b` be
checked against algebra rather than against another approximation.

**The station must carry its own increment's creep.** Each station solves

    e (1 + phi_cur) = d - (imposed by the recorded history) + phi_cur e_prev

with `phi_cur` the creep of the interval just elapsed — implemented as a
`1/(1 + phi_cur)` factor on the local stiffness plus a `phi_cur e_prev` term
in the imposed deformation. Leaving it out makes the scheme explicit, and on
a held strain that is not merely first order but marginally **unstable**:
twelve stations wobbled and landed 5% low, twenty-four were worse. The
history is dated at the geometric middle of each interval to match. Both
halves are needed; either alone diverges.

## On a fibre section

The same mechanism, one level down. A `type: fiber` section
has no member law to divide, so:

| beam path | fibre path |
|---|---|
| free deformation subtracted from the member's | free **strain** into the fibre's imposed-strain slot, the one a temperature and a tendon's prestrain already use |
| `1/(1 + phi_cur)` on the member stiffness | `1/(1 + phi_cur)` as a **scale on the law's own input** (`concrete02.strain_scale`) |
| the member's local elastic deformation recorded | the material's own committed strain (`epsP`) recorded — already divided, in the one place that could not forget it |

The scale is what a nonlinear law needs: it has no stiffness to scale, only a
stress at a strain, and what the station must hand it is the **elastic**
strain, at which it then answers with its own nonlinear stress. The free
strain goes in with a minus sign, exactly as a temperature does, because what
the slot carries is what the material sees.

**Creep is evaluated once per element, at the middle integration station.**
The imposed-strain slot is `(nel, n_fibres)` and has no station axis, while a
member's three stations carry different moments and so different creep. This
is the same resolution a draped tendon's prestrain already uses (it varies
along the element for the same reason and is evaluated at the element
midpoint) and it uses the same slot. Refining it means giving
the slot a station axis and changing the fused kernel's signature.

Two fibre cases check it against the same algebra the beam ones use, and
`tests/test_fiber_creep.py` asserts the two paths agree to 1e-6 at every
station of the same problem.

## What it does not do

* **No code relaxation model.** Relaxation itself is here (see below), but
  only as shapes a case states its own coefficients for: `mc90` and `ec2`
  are deliberately absent, because their constants could not be sourced
  verbatim and a recalled version failed its own definition when checked.
  Declaring `creep` on a reinforcement material is still
  refused — that is the other law, and accepting it would produce a
  plausible-looking prestress loss computed by the wrong physics.
* **Linear superposition only** (McHenry). That is the assumption every code
  model makes, and it holds to about 0.4 f'c. Nothing checks that the stress
  stayed inside that range.
* **No drying-shrinkage gradient through a section**, and no coupling to
  temperature or humidity fields.
* Ageing scales each member's local stiffness by one factor. With no `creep`
  declared, that is a total-strain model of a stiffening material, which is
  not a model of concrete — ageing is meant to be used *with* creep. **On a
  FIBRE section `ageing` is refused**: it changes Ec, and on a nonlinear law
  that is not one number (the peak strain and the strength age with it), so
  it cannot ride the strain scale creep uses. Declare the aged modulus
  directly instead and say in the case which age it is —
  `case_28d_confederation_sdl` does exactly that on the beam path.
* **Only the frame and fibre paths run the clock.** `creep`, `shrinkage`,
  `ageing` or `relaxation` on a solid, a shell, a coupled model, a
  standalone `truss2` model (a bar section assigned to `all`) or embedded
  reinforcement is refused by name. Bars assigned to named regions run on
  the mixed-line path and creep, shrink and age like beams.
* **One creep law per model on a fibre section.** Two concrete materials with
  different `creep` specs are refused. The scale and the imposed strain are
  already per fibre, so several laws are buildable; the history and the
  station's own `phi_cur` are kept once for the model, and no case needs
  more.

## Relaxation of prestressing steel

The third free strain, and the only one that is not concrete's
(`vfife/core/relaxation.py`).

```yaml
materials:
  strand:
    type: steel02
    E: 1.95e+11
    Fy: 1.674e+9
    relaxation: {model: power, rho_1000: 0.031, k: 0.21, tensioned_at: 28.0}
```

**It is a loss of the tendon's own prestrain, not a strain added to it.**
`sigma_p = E_p (eps_section + eps_pre)`, so a stress loss at held section
strain *is* `eps_pre` falling by `d_sigma_pr / E_p`. It therefore rides the
imposed-strain slot a tendon already uses, and nothing else in the fibre
solver had to learn a new kind of quantity. On a fibre whose prestrain is zero
it does nothing at all, so a material used for both tendons and ordinary bars
needs no separate declaration — but a spec that would touch **no** prestrained
fibre is refused, because that is where it would silently do nothing.

Models:

* `power` — `r(t) = rho_1000 (t / 1000 h)^k`. **Both coefficients are
  required**: there is no default, on purpose. `rho_1000` depends on the steel
  class *and* on the stress ratio it is held at, and a default here would be a
  number nobody chose that comes out as a plausible prestress loss.
* `curve` — measured data `[[t_hours, loss], ...]`, read linearly in log t;
  `interpolation: linear` reads it linearly in t (1.16), as a table of ratios
  against days is meant. Before its first row the loss grows linearly from
  zero, after its last it stays.

`rho_1000` is a **fraction** (0.031 for 3.1%), checked to lie in [0, 0.2];
`k` in (0, 1); a `curve`'s losses are fractions too. A loss reaching 100% of
the prestress stops the run.

`tensioned_at` is the concrete age in days when the tendon was stressed
(default: `solver.age`, the age the analysis starts at, or the first step's
`age` when that is earlier); relaxation's own clock is hours from there. A
continuation from `solver.initial_state` keeps the start of the run it
continues (it rides in the checkpoint), so its relaxation goes on rather than
beginning again. It may not be later than the analysis starts — the tendon carries its
prestress from the first station; it is refused by name. A tendon stressed
mid-run is modelled by the bar's own clock instead (next paragraph).

Besides fibre tendons, a **prestressed bar** (a `bar` section assigned to a
named region with `prestress` or `unstressed_length`) relaxes the same way:
its prestrain `L0/Lu - 1` falls by `r(t)`. A material that is also used by
beams or ordinary bars relaxes only its prestressed bars; a declaration on a
material none of whose members carries a prestrain is refused.
**Each bar has its own clock (1.16):** a bar prestressed
from the start relaxes from `tensioned_at`; one born in a later step
(`activate`) from the age it is born at; one `jack`ed in a step from the age
its step locks it off, from the strain the jack left (`T / EA`, on the length
it is locked at). Until then it has nothing to lose. A checkpoint keeps each
bar's clock.
`reduction` is a plain multiplier for the code factor that accounts for a
tendon in a member that is itself shortening relaxing less than one held at
constant strain — this library does **not** compute it; state it or leave it
at 1.0 and know the answer is the intrinsic loss.

**Code formulas live outside the solver (1.16).** `vfife.codes` holds them,
one module per code, as functions that return the `curve` (or `power`) spec
the solver reads; the solver never imports it, and a case file carries the
curve itself. `vfife.codes.jtg3362.relaxation_curve(sigma_pe, fpk, zeta,
psi)` is JTG 3362's (and JTG D62's) relaxation of strand -- psi zeta (0.52
sigma_pe / fpk - 0.26) of the stress after anchoring, developing by the
code's ratio table -- checked against Midas Civil's tendon loss table to four
figures; the table's rows after two days are not yet checked against an
independent source. The Midas translation uses it bar by bar.

**There is no `mc90` and no `ec2` in the solver.** Those expressions are short and several
plausible-looking forms of them are in circulation; while this was written the
coefficients could not be sourced verbatim, and a recalled version was checked
against its own definition and failed it — at the stress ratio where it must
reduce to `rho_1000` it returned 0.96% for a declared 2.5%. A relaxation law
written from memory would be a number with the shape of a code answer and none
of its authority, and it would be invisible. So the case file has to say where
its coefficients came from; `case_29f`'s come from a published MC90 curve for
a real strand (Ghali 2000 Fig. 7, digitised in
`verification/Ghali2000_confederation_longterm.txt`).

## Verification

* `cases/case_29_creep_bar.yaml` — a bar loaded at 28 days and held. The
  elongation triples, to six figures, and the axial force sits at 300.0 kN
  from the first station to the last.
* `cases/case_29b_relaxation.yaml` — the same bar stretched 10 mm and held.
  300 kN decays to 100.000 kN, which is `300 x R(1028, 28)` exactly, on six,
  twelve or twenty-four stations. The creep function is a standard linear
  solid, whose relaxation modulus is closed form, so this is a check against
  algebra.
* `cases/case_29c_shrinkage_bars.yaml` — two bars, one free and one held at
  both ends, drying under MC90. The free one shortens by `eps_sh L` and
  carries 2e-8 N against a scale of 82.7 kN; the held one carries
  `EA eps_sh` at 28, 100, 365 and 1000 days.
* `cases/case_28d_confederation_sdl.yaml` — the same three statements on a
  real frame that is three times redundant, which a bar cannot be. The
  Confederation Bridge's 54 kN/m superimposed dead load, held for 630 days:
  every displacement monitor grows by 1.632–1.642 against an exact
  `1 + phi Ec(400)/Ec28 = 1 + 0.53305 x 46.51/38.7 = 1.64063` (MC90's phi is
  referenced to Ec28; the case gives `E28: 3.87e+10`; before v1.0
  it was `1 + phi = 1.53305`), and the largest bending moment in the model sits
  at 397.8 MN·m from the instant of loading to the last station. **A bar
  cannot redistribute even when the implementation is wrong**; a frame can, so
  this is where "creep is a free strain and not a load" is actually at risk.
  The instantaneous half is checked against Ghali's own published analysis.
* `cases/case_29d_fiber_creep_bar.yaml` — case_29's bar built out of sixteen
  `concrete02` fibres. Same three numbers (10.000 mm, 30.000 mm, 300.0 kN)
  through a mechanism that shares nothing with the beam path but this module,
  plus one metric that has no counterpart there: the SECTION's plane strain
  reads 3.000e-3 while the material is being driven at 1.000e-3, which is the
  whole claim of the imposed-strain formulation as one number.
* `cases/case_29e_fiber_relaxation.yaml` — case_29b's held bar, likewise.
  100.000 kN at the end, and the middle of the decay 2.8% / 1.0% / 0.4% from
  the integral on 6 / 12 / 24 stations — monotone towards the algebra, which
  is the implicit step working on a mechanism v0.4 did not have.
* `cases/case_29f_tendon_relaxation.yaml` — a tendon with both ends held, so
  the section strain is zero and the support reaction IS the tendon force:
  208.320 kN falls to 197.846 = `A_p sigma_pi [1 - r(t)]`, the law on its own
  definition with the structure removed -- less the 0.45% by which steel02's
  rounded first branch (OpenSees' Steel02) sits under E eps at 0.89
  of its yield stress: 207.388 kN falling to 196.914.
* `cases/case_29g_prestress_loss.yaml` — the same tendon with one end free and
  all three laws on. The three losses stop being independent: separately they
  are 200.7 + 53.6 + 82.4 = 336.8 MPa, together 320.8. The answer is bracketed
  by the age-adjusted effective modulus method at `chi = 0` and `chi = 1`, and
  backing `chi` out station by station gives **0.22 at 38 days rising to 0.99
  at 1028** — it is not the textbook constant 0.8, and the long-term answer
  here is the plain effective-modulus method.
* `tests/test_fiber_creep.py` — the two paths against each other station by
  station, the decay against `core.creep.relaxation`, the fused kernel against
  the reference path with the scale on, the shared clock, and the four things
  a fibre section refuses (`ageing`, creep on reinforcement, creep on a law
  with no `strain_scale`, two different laws in one model).
* `tests/test_creep.py` — the model functions against the code documents,
  the superposition against a two-increment hand sum, `relaxation` against
  the Zener modulus and for stability in the number of stations, and the
  solver's own scheme including the regression that a relaxing bar never
  carries more than it started with.

```
python benchmarks/generate_creep_bar.py --check
python -m vfife run cases/case_29b_relaxation.yaml
```
