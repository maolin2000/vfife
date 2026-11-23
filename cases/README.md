# Numbered reference cases

These are references to read, not a place to start. To build your own analysis
from scratch, follow [`../docs/tutorial.md`](../docs/tutorial.md).

The public case library is intentionally small. Names use `case_NN_topic.yaml`;
the case name and generated result prefix match the file stem.

| ID | Case | Purpose |
|---|---|---|
| 01 | `case_01_beam_cantilever.yaml` | Euler--Bernoulli 3-D beam against the closed-form tip displacement |
| 02 | `case_02_truss_bar.yaml` | Axial bar against `PL/(EA)` |
| 03 | `case_03_mixed_frame.yaml` | Unified truss--beam assembly |
| 04 | `case_04_nafems_le10.yaml` | Public NAFEMS LE10 3-D solid benchmark |
| 05 | `case_05_space_truss_seismic.yaml` | Spatial truss under the El Centro record |
| 06 | `case_06_rc_column_cyclic.yaml` | Nonlinear reinforced-concrete column, 800 kN axial then +-1% drift cycles |
| 07 | `case_07_coupled_solid_truss.yaml` | Conforming shared-node solid–truss assembly against an independent stiffness solution |
| 08 | `case_08_timoshenko_short_beam.yaml` | 3-D Timoshenko short cantilever against bending-plus-shear theory |
| 09 | `case_09_connector_support.yaml` | Reversible bilinear support connector against its analytical inverse curve |
| 10 | `case_10_clamped_square_shell.yaml` | Mindlin--Reissner Shell4 clamped square plate against classical thin-plate theory |
| 11 | `case_11_fiber_rc_column_cyclic.yaml` | Fibre-section RC column against the Tanaka & Park (1990) specimen 1 measured record |
| 12 | `case_12_ualberta_sf0b.yaml` | Solid-element RC beam in four-point bending against the Alberta SF-0B test, through the peak |
| 12b | `case_12b_ualberta_sf0b_bond.yaml` | The same beam with fib MC2010 bond-slip on the tension layer -- the A/B partner of case 12 |
| 12c | `case_12c_ualberta_sf0b_confined.yaml` | The same beam with a second solid section: a Mander-confined curve on the concrete inside the stirrups |
| 13 | `case_13_pretensioned_beam.yaml` | Pretensioned concrete beam: the prestress primitive (a bar strained before the model is) against the camber and the transfer length |
| 14 | `case_14_draped_tendon_balance.yaml` | Draped `path` tendon: load balancing against the parabolic equivalent load |
| 15 | `case_15_chalmers_prc.yaml` | Post-tensioned RC beam against the Chalmers/Fernandez (2025) measured record |
| 16 | `case_16_lee_frame.yaml` | Lee's frame: the snap-through limit point of a co-rotational frame to 0.1% |
| 16b | `case_16b_lee_frame_arc_length.yaml` | Lee's frame (NAFEMS NL7) loaded instead of driven, 31772 N as a pattern (PL^2/EI = 31.887), traced by arc length in 61 increments, 6 s: limit 18.577 EI/L^2 at v/L = 0.402 (NAFEMS 18.552, 0.407), the fold where v/L turns back from 0.509 to 0.424, the negative-load minimum -9.43, and v/L = 0.781 at the full load (NAFEMS 0.784, Abaqus 0.781) |
| 17 | `case_17_hinged_cylindrical_roof.yaml` | Sze's hinged cylindrical shell roof, the shell4 snap-through on numpy, numba and warp |
| 18 | `case_18_bend45_cantilever.yaml` | Bathe & Bolourchi's 45-degree bend: finite rotations of a spatial beam against the 1979 and Abaqus tips |
| 19 | `case_19_floating_beam.yaml` | Wu & Ting's unconstrained solid beam in free flight: energy, momentum and rigid-body kinematics under dynamics |
| 20 | `case_20_rolled_cantilever.yaml` | A solid cantilever rolled into a full circle by an end couple, against the closed-form elastica at every quarter turn |
| 21 | `case_21_vecchio_emara_frame.yaml` | Vecchio & Emara's one-bay two-storey RC frame pushed to 155 mm: the plateau to 2%, the peak to 4%, and the rising branch held as a band that says what the element lacks |
| 21b | `case_21b_vecchio_emara_cracked_shear.yaml` | case_21 with each member's shear on the cracking law: uncracked kappa G A until the member cracks in flexure, then Park & Paulay's truss stiffness of its own No.10 @ 125 stirrups (8% of uncracked), nothing fitted. The rising branch at 30 mm goes from 1.15 to 1.08 of the measurement (277 kN against 257); plateau 329 / 301 kN at 100 / 155 mm (0.99 / 0.97), peak 343.5 kN (1.03), energy 45.6 kN m (44.8) |
| 21c | `case_21c_vecchio_emara_static.yaml` | case_21 solved by Newton instead of dynamic relaxation (`mode: static`): 8 s against 22 minutes on the same machine, 35 increments, 109 Newton iterations, no cutback; case_21's contract target for target, the two solvers within 0.3% at every level it reads |
| 22 | `case_22_cable_catenary.yaml` | A hanging cable found from the straight chord by relaxation, against Irvine's elastic catenary: H and sag to 0.03%, the first cable primitive |
| 23 | `case_23_suspension_bridge.yaml` | A 300 m suspension bridge: the dead-load state form-found by `vfife formfind` (the closed-form lengths to 4e-7), then a live load against the linearised deflection theory: h to 0.1%, the deflections to 0.5% |
| 24 | `case_24_hzb_girder.yaml`, `case_24b_hzb_girder_modal.yaml`, `case_24c_hzb_girder_collapse.yaml`, `case_24d_hzb_girder_piers.yaml`, `case_24e_hzb_girder_bearings.yaml`, `case_24f_hzb_girder_bearings.yaml`, `case_24g_hzb_static_pushover.yaml` | The Hangzhou Bay Bridge 70 m box girder: the elastic simply supported girder and its modes against Xu's ANSYS (24b), the 3 x 70 m continuous unit with Chen's 29 design tendons under dead load and the JTG lane load to k = 10 (24), and the same pushed to collapse by a dynamic ramp against Chen's k_u = 16.8 at 727 mm (24c: pier yield and a snap at k = 14.8, his 727 mm at 17.3, mechanism at 18.1). And the unit on real PIERS (24d: vertical equilibrium closes to 0.03% and no horizontal load gives no horizontal reaction, but the monolithic pier-girder joint DOUBLES the middle-span dead-load deflection), and its elastic twin on pot BEARINGS (24e: the same shortening then costs 0.11 mm instead of 9.88, against the classical three-span reactions 1.1 qL and 0.4 qL and an independent direct-stiffness frame). 24f is the fix itself: the FIBRE model on those bearings, which is what the fibre solver learned to host in v0.4 -- middle span -7.68 mm against 24d's -18.99 and the twin's -7.59, every bearing reproducing its own declared stiffness (R/k to five figures, and the sliding pot slides 14.08x the fixed one against a declared ratio of 14.08). 24g is 24d pushed by the static solver: dead load by one Newton level, live load by arc length, about 30 s against 24d's 50 minutes, through the peak k = 14.69 onto the softening branch (down to 12.76, back up to 14.08 at the -0.38 m stop) that 24d's ramp jumps; its dead state is -23.47 mm (-23.18 before v1.0's anchor-set fix), 24d's -27.77 being its t = 0 load's overshoot |
| 25 | `case_25_chen_hollow_slab.yaml`, `case_25b_chen_hollow_slab.yaml` | Chen's two 13 m pretensioned hollow-core slabs loaded to failure by two jacks (the first measured destructive test here): a `patches` section with circular voids, nine strands with four bonded lengths, against the digitised load-deflection points -- 1# within 1-5% over the curve and its peak 8% low, 2# within 2% and its peak 5% low |
| 26 | `case_26_moving_load_beam.yaml`, `case_26b_moving_load_influence.yaml` | A moving load: one axle crossing a 20 m simply supported beam at 40 m/s against Fryba's closed-form series (inside 0.3%, peak 1.70x static and a third of the crossing late), and the same axle walked across quasi-statically against the analytic influence line (worst error 1.3e-4, half the stations between nodes) |
| 27 | `case_27_thermal_two_span.yaml`, `case_27b_thermal_eigenstress.yaml` | Temperature as a free strain rather than a load: a two-span continuous beam that grows 19.2 mm and carries nothing when nothing is in the way, and makes 55.0 kN m of secondary moment out of a 15 K/m gradient when the middle support is; and a nonlinear code-shaped profile on a determinate FIBRE beam, which is stressed anyway (-2.20 MPa at the heated top, +1.84 MPa under it) with zero net force and moment. Both within 0.05% of closed form |
| 28 | `case_28_confederation_trucks.yaml`, `case_28b_confederation_thermal.yaml`, `case_28c_confederation_uniform.yaml`, `case_28d_confederation_sdl.yaml` | The Confederation Bridge's P31-P32 portal frame (250 m span, two 95 m cantilevers, 14.0 m deep at the pier and 4.5 at midspan), reconstructed from four papers and a thesis with nothing calibrated on a deflection. Two trucks walked the 440 m girder in 22 stations: 17.0 mm at midspan against 17.1 measured, and the sign reversal beyond the pier (3.4 mm UP at x = -180 m) to 5%. The published summer temperature profile: midspan rises 58.2 mm against 60.6 published, x = -180 m drops 43.9 against 43.7, the tips 22% too far. And the same field with the gradient removed, which is what says the gradient was worth 9 mm at midspan and 31 at every tip. And the 54 kN/m superimposed dead load of April 1997, the one increment of the paper's two-year record that is fully specified: 66 mm at midspan and 89 at the tips against the 70 and 80 of Ghali's own analysis, then held for 630 days of MC90 creep, where every displacement grows by exactly 1 + phi and no internal force moves at all |
| 29 | `case_29_creep_bar.yaml`, `case_29b_relaxation.yaml`, `case_29c_shrinkage_bars.yaml`, `case_29d_fiber_creep_bar.yaml`, `case_29e_fiber_relaxation.yaml`, `case_29f_tendon_relaxation.yaml`, `case_29g_prestress_loss.yaml` | Creep, relaxation and shrinkage as one mechanism -- a strain that arrives for reasons that are not force. A determinate bar under a held load creeps to exactly three times its elastic elongation and carries the same 300.0 kN throughout; the same bar stretched 10 mm and HELD decays to 100.000 kN, which is `300 x R(1028, 28)` from the standard linear solid's closed-form relaxation modulus, unchanged on six, twelve or twenty-four time stations; and two drying bars, the free one shortening by `eps_sh L = -2.755 mm` while carrying 2e-8 N against an 82.7 kN scale, the held one carrying `EA eps_sh` at 28, 100, 365 and 1000 days. And then the first two of those built again out of FIBRES, where the free strain rides a different slot and the implicit step is a scale on the law's own input rather than on a stiffness: the two paths land on the same thirteen numbers to a millionth. And the third law, the steel's: a tendon held at its strain loses exactly `rho_1000 (t/1000 h)^k`, and the same tendon in a member free to shorten loses creep, shrinkage and relaxation together -- 320.8 MPa against 336.8 for the three taken separately, with an effective ageing coefficient that runs from 0.22 to 0.99 rather than sitting at the textbook 0.8 |
| 30 | `case_30_kuang_control_beam.yaml` | Kuang et al. (2023)'s uncorroded scaled PC T-beam L1 in four-point bending: the cracking load (72.0 kN) and the stiffness before and after it are checked; the published 232.5 kN ultimate is not, because the beam's one strand and two bars cannot develop it (the arithmetic is in the case header and the contract) |
| 31 | `case_31_li_control_beam.yaml` | Li et al. (2024)'s uncorroded post-tensioned scaled T-beam A0: the campaign whose capacities close, held through cracking, yield and the 145.0 kN peak against its digitised load-deflection record |
| 32 | `case_32_bearing_laws.yaml` | A girder on a FIXED and a SLIDING bearing, warming against an abutment stop: Coulomb friction that presses on the bearing's own vertical spring, and a 6 mm gap that carries nothing until it closes. Dead load puts W/2 on each; +30 K spends the 0.09 mm of stick, slides at mu W/2 = 9.19 kN, closes the gap at 6 mm and pushes 1341.7 kN into the stop, sharing the last 3 mm with the girder's EA -- every number within 0.3% of the closed form, the friction as the connector's own action to 0.2% |
| 33 | `case_33_falling_beam.yaml` | A 4 m beam hung 1 m above the ground from nine strings prestressed to its weight, all cut at t = 0.1 s: free fall to the rigid ground plane at 0.1 + sqrt(2 x 0.99 / g) = 0.5493 s (0.5494), never through it (1e-7 m), and -- the contact damped -- at rest on it by 0.7 s carrying its 3.079 kN. Member removal and a rigid surface together: the last act of a collapse |
| 34 | `case_34_brittle_cantilever.yaml` | A 2 m cantilever, one beam2 section on the whole mesh with `failure` on its assignment, loaded at the tip until the root reaches M_cap = f_y I / c at P = M_cap / L = 15 kN, t = 2 s: static before it (16.0 mm at 12 kN, 0.2%), the root reaction peaks at 14.97 kN at t = 2.0005 s and is zero after, one member of four marked failed. The elastic-brittle criterion outside mixed_line |
| 35 | `case_35_spectrum_cantilever.yaml` | A 10 m uniform steel cantilever under a flat 2 m/s^2 spectrum along x (`response_spectrum`): the four lowest bending modes against the closed-form continuum (periods 0.2% and, lumping, 2.4%; effective masses 61.3% and 18.8% of m L to 0.2%) and the SRSS peaks -- tip 28.44 mm (28.41), base shear from the modal sums of the support reactions 4045 N (4053), foot moment from the modal end actions 28081 N m (28085) |
| 36 | `case_36_multi_support_wave.yaml` | A 100 t deck mass between two piers 400 m apart on axial members (2 Hz), the ground u_g = 0.1 sin(2 pi 1.3 t) arriving as a wave at 1000 m/s (a constraint `record` with `wave`): the east pier 0.4 s behind the west, nearly in anti-phase. Against the closed form: deck peak 0.10190 m (0.101899), pier forces 1.5034 / 1.5076 MN (1.5036 / 1.5075), both piers on their records to 1e-9. Uniform input would move the deck 0.285 m with about the same member force |
| 37 | `case_37_pier_in_water.yaml` | A 20 m circular RC pier (D = 2 m) with 500 t of deck on top, standing in 20 m of water (`hydrodynamic_mass`: 3220 kg/m across the pier, weightless), under a flat 2 m/s^2 spectrum: the cantilever-with-tip-mass closed form gives periods 1.4649 / 0.1016 s (model +0.02 / +0.11%), deck 114.80 mm (+0.01%), base shear 1.2376 MN (-0.01%), base moment 23.706 MN m; the axial period is the dry pier's (0.06349 s) because the water acts across the pier only. Dry, the first period is 1.4% shorter and the base shear 6% lower |
| 38 | `case_38_sprung_mass_crossing.yaml` | Yang & Yau's (1997) sprung mass -- 5750 kg on a 1595 kN/m spring -- crossing a 25 m simple beam (EI = 2.87e9 x 2.9, 2303 kg/m) at 27.78 m/s (a `vehicle` with one tyre-less axle), against the beam's first 20 sine modes coupled to the mass and integrated by DOP853 to 1e-11: peak midspan 2.4072 mm (2.4066), 2.0137 mm with the mass at midspan (2.0127), the mass's peak acceleration 0.1477 m/s^2 (0.1479), contact force 55571-57238 N (55571-57239) |

Case 12, 12b and 12c are the same specimen and each differs from case 12 in
exactly one thing: 12b adds the bond interface, 12c gives the core inside the
stirrups its own section (two solid sections on one mesh, `assign` in order).
All three are driven to 19.2 mm at midspan, the deflection at which the
specimen was measured to peak, and carry 360 kN there against the measured
399 — ratio 0.90, with no crushing, which is the reported failure mode. The
bond interface costs 0.1% at that point and the confined core nothing at all:
case 12's contract carries the mesh study that says why (the compression band
had no length scale until a crushing energy was stated), and the history of
three earlier explanations of the gap that it overturned.

Case 28 is the first case whose SECTION is itself a piece of verification. No
source publishes the Confederation Bridge's cross section as dimensions; the
outline comes off a shear-design paper's figure, and the two thicknesses it
does not dimension are fitted on a thesis appendix's gross-section table.
Fitting two numbers there and then predicting area, centroid and I at two
depths within 4% -- and the midspan centroid within 1% with nothing free at
all -- is what makes the deflection comparisons afterwards mean anything. The
one number with no source is the pier wall: case_28 moves 25% over a fourfold
change in it, and case_28c, whose load is a uniform temperature rise, moves 4%,
because E and the pier stiffness both very nearly cancel out of that one.

Every element the library registers also has a minimal model in
`vfife/testing/models.py` and is asked the element-independent questions --
is the force a gradient, is a rigid motion of the deformed body free, does
the fictitious mass bound the step -- by `tests/test_invariants.py` on every
commit. Those are not cases and are not run from here; `python -m vfife probe
<case>` asks the same of yours.

`vfife sweep` draws fibre-beam models only, so case 28 -- whose sections are
`general`, area and second moments with no fibres -- has nothing the library
can sweep. `benchmarks/sweep_confederation.py` walks the outline THOSE numbers
came from along the deformed axis instead and writes a ParaView collection
(`results/<case>_box.pvd`, one `.vtu` a frame). The moment and axial force
ride on it as SIGNED point data: the `bending_z` a run writes is
`max(|M_i|, |M_j|)`, a magnitude on purpose so that averaging cannot halve a
cantilever's support moment, and colouring by it puts a jump wherever the
moment changes sign inside an element. The diagram wants the signed end
actions, so those cases declare `bending_z_i`/`bending_z_j` (and the axial
pair) as well, and each element gets its own two rings -- which leaves the
moment discontinuous at exactly one place, the pier, where the shaft really
does deliver its own moment into the joint.

Run from the repository root after `conda activate vfife`, for example:

```powershell
python -m vfife info cases/case_01_beam_cantilever.yaml
python -m vfife run cases/case_01_beam_cantilever.yaml
```

Mesh-format fixtures belong in `tests/data`; generated refinement cases belong
under `results/` and are not part of this reference library.
