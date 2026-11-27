# Fibre beam beam2_fiber

`beam2_fiber` is a co-rotational Euler–Bernoulli fibre beam with 6 dof a node: the section is divided into fibres and `B^T·s` is integrated along the axis; mixed sections of `concrete01/02 + steel02/steel_cyclic/rebar/piecewise_uniaxial/linear_elastic` are supported.

## Input

```yaml
materials:
  concrete: {type: concrete02, fpc: 35e6, epsc0: 0.002, fcu: 7e6, epscu: 0.0035, ft: 3.2e6, Ets: 1.2e9}
  steel:    {type: steel02, E: 200e9, Fy: 350e6, b: 0.01}
sections:
  bar20: {type: bar, material: steel, shape: general, area: 3.14e-4}
  rc:
    type: fiber
    material: concrete
    shape: rect
    width: 0.25
    height: 0.35
    ny: 8
    nz: 8
    integration: lobatto3       # gauss2 (default) | lobatto3 | lobatto4 | lobatto5 (end points include the end sections' yielding; five points add one at the quarter points)
    bars:                       # OpenSees' layer: one bar a row, referring to a bar section
      - {y: 0.09, z: 0.14, section: bar20}
assign:
  - {region: all, section: rc}
```

- `rect` divides into `ny×nz` equal fibres of area `dy·dz`, `y∈[-w/2,w/2] z∈[-h/2,h/2]`;
- **Hollow sections** (for steel): `shape: tube` (a round tube: `diameter` outside, `t` wall, `nt` fibres around, optionally `nr` layers through the wall) and `shape: box` (square or rectangular tube: outer `width`/`height`, `t`, `ny`/`nz` fibres along the walls). `bars:` in a tube is refused -- a steel tube has no reinforcement, and a concrete-filled tube needs a confinement model this library does not have.

  A round tube's fibres sit at each wedge's **RMS radius** `sqrt((r0²+r1²)/2)` rather than its centroid: one point cannot carry both the first and the second moment of its cell; at the centroid the second moment is short (a point has no inertia of its own), at the RMS radius the second moment is exact while the first is still zero by symmetry, the wedges being congruent. So **A, Iy and Iz are exact for any `nt ≥ 3`** (at the centroid: 5.1% off with 8 fibres, 0.84% with 20). A box's cells are not congruent and cannot do this; it uses the centroid grid and converges from below with refinement. `rect` is unchanged.
- **Any shape: `shape: patches`**: `patches: [{corners: [[y, z]×4], ny, nz}, ...]`; each quadrilateral is divided by a bilinear map into `ny×nz` cells, the fibre at the cell's centroid with the cell's own area (the shoelace formula; a triangle is a quadrilateral with two corners coincident) -- OpenSees' `patch quad`. The Hangzhou Bay 70 m box girder (inclined webs, fillets, tapered cantilevers) is ten patches and 166 fibres (`benchmarks/generate_hzb_girder.py`); refined four times, A and I are within 0.3% of exact. The sweep export draws the patches' cells as solids. The origin of the coordinates is the node line (the member axis); to put it at the centroid, compute that first (the generator does).
- **Holes**: a patch may carry `holes: [[y, z, r], ...]` (circular). A cell whose centre is in a hole is dropped, and one a hole cuts is weighted by 4×4 sampling -- a circle of r = 0.25 cut from a unit square has its net area to 0.3% (hollow-core slabs).
- `j` (old spelling `torsion_constant`, still read): the fibre section's Saint-Venant torsion constant J (m⁴). Without it the fibres' polar moment Iy+Iz is used -- right for a circle, five times too large for a box girder (Hangzhou Bay: Bredt's J = 29.5 against 150 m⁴), putting the torsional modes at more than twice their frequency.
- A row of `bars` may carry `count: n`: n identical bars at one point (a hundred-odd Φ16 in a top slab as ten rows).
- `gauss2` (`s=0.21/0.79`), `lobatto3` (`0/0.5/1`), `lobatto4` (`0/0.276/0.724/1`), `lobatto5` (`0/0.173/0.5/0.827/1`); Lobatto includes the end points and is more accurate for cyclic end sections, and five points put one integration point 0.17L from the end -- a member 2h long resolves a 0.5h hinge without being divided (case_11: peak 161 → 165 kN, hysteretic energy unchanged); the `checkpoint` shape is `(nel, n_stations, n_in_group, n_state)`, and the kernel is compiled per number of points;
- a mixed section is grouped by `fiber_group`, each group with its own `material.update` and `state`; the totals `N=ΣσA, My=ΣσzA, Mz=Σ-σyA` are then assembled with `B^T`.

The first `public_benchmark` is `cases/case_11_fiber_rc_column_cyclic.yaml`: Tanaka & Park (1990) specimen 1 (a 400×400 column, 1.6 m to the point of contraflexure, 8Φ20, f'c = 25.6 MPa, axial load 819 kN = 0.2f'c·Ag), 4×Line2, `lobatto3`, 480 cyclic stages, `concrete02+steel02`; the measured hysteresis is in `verification/Tanaka_90_specimen_1.txt` (593 points).

The measured deviations from the test (`benchmarks/plot_tanaka_hysteresis.py` prints them on every run):

| quantity | test | analysis | ratio |
|---|---|---|---|
| peak force | 166.8 / −167.1 kN | 148.4 / −161.1 kN | 0.89 / 0.96 |
| secant stiffness at 8 mm | 14.75 kN/mm | 17.02 kN/mm | 1.15 |
| hysteretic energy | 116.3 kJ | 127.6 kJ | 1.10 |

The high strength agrees with an independent section check: the same fibres at the right N = 819 kN reach an M-φ limit of 291.5 kN·m (V = 182 kN, 9% above the test, i.e. with no cover spalling), and the beam gives 193 kN, 6% above its own section capacity, covered by the cyclic hardening of `steel02`. The high stiffness comes from having no bond slip and no shear deformation.

**These three numbers mean something only after three solver bugs were fixed** (the energy went 0.32 → 1.77 → 1.69 → 1.30): the material state committed at every iteration, the fibre subdivision on an axis that does not bend, and the axial internal force missing a factor 1/L0. The steel hardening ratio is 0.005 rather than the 0.012 implied by fu/fy -- the latter takes the bars to 709 MPa at ±3.5% drift (measured ultimate 721); a monotonic M-φ does not show this at all, only the cyclic history does.

The softening concrete is not regularised; the mesh sensitivity is recorded in `test_fiber_benchmarked` as linear elastic mesh convergence plus the post-cracking 5-60 kN range. Note also that the case uses four 400 mm elements rather than a finer mesh, because 8×200 mm makes staged relaxation stall at the first cyclic stage (diagnostics pointing at a transverse dof of a middle node, residual 9.7 kN); the reason is in the case's comments.

## The fused kernel of the internal forces

`beam2_fiber` takes `backend: numpy` and `backend: numba` (since 0.1). `numpy` is the readable reference path and dispatches **automatically** to the fused kernel of `kernels/numba_fiber_beam.py` when its conditions hold; `numba` **requires** the kernel and, when it cannot be built, refuses with the kernel's own reason (`solver.kernel_status`) rather than quietly running a hundred times slower. The kernel does a whole element in one pass (gather, co-rotation, fibre strains, constitutive laws, section resultants, Hermite virtual work, rotation back), with the two uniaxial laws inlined as scalar device functions; the shape derivatives `Bky/Bkz` depend only on the station and the element length and are computed once at assembly.

**Condition: one constitutive law for all the section's fibres, another for all its bars (if any).** A homogeneous section without bars (a steel tube or box) also takes the kernel, with empty bar arrays (before 0.1 a line `if not is_hybrid: return` kept them out: a 456-member lattice shell ran 12 steps/s, 88 with that line gone). When the bars' material types or parameters differ, the kernel does not apply and the NumPy reference path is used instead (`solver.kernel_status` gives the reason). That is a real limit of the kernel; `tests/test_fiber_beam_kernel.py` asserts both the fallback and that the fallback still solves.

Measured (case_11, 4 elements × 3 stations × 40 fibres):

| | one evaluation of the internal forces | the whole case |
|---|---|---|
| reference path | 2.54 ms | about 47 minutes |
| fused kernel | **0.025 ms** | **58 seconds** |

101 times for the internal forces, 18.7 times end to end (the Python left in the relaxation loop becomes the new dominant term); the whole case agrees with the reference path to round-off (the element fields differ by exactly 0).

## Several fibre sections in one model

```yaml
sections:
  col:  {type: fiber, material: s, shape: box,  width: 0.2, height: 0.2, t: 0.008, ny: 6, nz: 6}
  beam: {type: fiber, material: s, shape: tube, diameter: 0.15, t: 0.005, nt: 16}
assign:
  - {region: columns, section: col,  orientation: [0, 0, 1]}
  - {region: beams,   section: beam, orientation: [0, 1, 0]}
```

Each section has its own region, its own `orientation`, and **its own materials, `cover`/`core_material`, `bars` and `anchorage`** (since 0.2 each section owns its material groups -- cover, core, one per bar -- and `solver.materials` is the sections' groups concatenated; the reference path drives each element's own section's groups; the kernel still inlines one family of laws once and reads parameters fibre by fibre, with the bar slots padded to the widest section). Sections with different numbers of fibres are **padded with zeros to the widest** (the padding fibres have zero area and do not count in the fracture fraction), so the kernel's fibre count is still a compile-time constant and a single-section model's arithmetic is unchanged to the bit (0.1 checked the geometry, 0.2 the grouping, both bit for bit against the old code). The limits that remain are in the input checks and name the element: `integration`/`fracture`/`formulation` are element-level settings and must agree across sections; `all` may not stand beside other regions; overlapping regions or an element no section covers are refused; the kernel requires one concrete law type for all sections and one steel law type (the parameters may differ), or it falls back to the reference path.

```yaml
sections:
  beam:   {type: fiber, material: c30, shape: rect, width: 0.3, height: 0.4, ny: 4, nz: 16,
           cover: 0.03, core_material: core, bars: [{y: 0.089, z: 0.150, section: no20}, ...]}
  column: {type: fiber, material: c30, shape: rect, width: 0.3, height: 0.4, ny: 4, nz: 16,
           cover: 0.02, core_material: core, bars: [{y: 0.089, z: 0.160, section: no20}, ...]}
assign:
  - {region: columns, section: column, orientation: [0, 1, 0], anchorage: {region: joints, bar_diameter: 0.02}}
  - {region: beams,   section: beam,   orientation: [0, 1, 0], anchorage: {region: joints, bar_diameter: 0.02}}
```

This is how `cases/case_21_vecchio_emara_frame.yaml` (the pushover of Vecchio & Emara's 1992 one-bay two-storey RC frame) is written: 36 s, peak 345 kN against 333 measured, the plateau within 3%, the rising branch 1.15 times too stiff -- the shear deformation this element lacks (the authors measured it at 20% of the total), which the contract records as a range rather than tuning it away.

## Member shear flexibility

```yaml
sections:
  col: {type: fiber, material: c30, shape: rect, width: 0.3, height: 0.4, ny: 4, nz: 16,
        shear_stiffness: 1.25e9}     # kappa·G·A_s [N], given by the analyst
```

A fibre section may declare an effective shear stiffness κGA_s. With it, the member's bending interpolation changes from Hermite to Reddy's interdependent interpolation (Timoshenko), with one φ = 12EI/(GA_s L²) per member per bending plane: the curvature rows are multiplied by μ = 1/(1+φ) and carry φ in their rotation terms, and a shear strain γ constant along the member is added, with V = GA_s·γ entering the internal forces through its own row. **The fibres do not see shear** -- this is a member-level elastic shear flexibility (or a cracked effective one the analyst gives). Without it φ = 0 and the curvature rows are Hermite's numbers, unchanged to the bit. Both paths read the same precomputed rows; `actions[:, 1]` and `[:, 2]` report the shear forces along local y and z.

What guards it: a shear-flexible cantilever, PL³/3EI + PL/GA_s, is exact to 2e-4 with one element (the interdependent interpolation is nodally exact for prismatic members); on an RC section with shear the kernel agrees with the reference to < 1e-10; without a shear stiffness the curvature rows are Hermite's to the bit. On Vecchio & Emara's frame (case_21) the parameter-free uncracked κGA is worth 2% (at 30 mm 1.17 → 1.15); most of the authors' measured "20% of the deformation in shear" is shear **after cracking**, and a quarter of the uncracked stiffness reaches 1.10, but that is a parameter not derived here, and the case does not use it.

## Prestressing tendons

```yaml
sections:
  strand: {type: bar, material: strand_steel, shape: general, area: 1.4e-3}
  girder:
    type: fiber
    material: c50
    shape: box
    width: 4.0
    height: 15.8
    t: 0.3
    tendons:
      - section: strand
        profile: [[0, 1.6, 0], [35, -1.5, 0], [70, 1.6, 0]]   # [s, y, z], s along the member from its start
        interpolation: quadratic                               # a parabola through each three points
        jack: 1.395e9                                          # jacking stress
        friction: {mu: 0.17, k: 0.0015, jack: both, anchor_set: 0.006}
        elastic_shortening: compensate
      - section: strand
        profile: [[0, -1.7, 0], [70, -1.7, 0]]                 # a straight tendon
        prestress: 1.2e9
```

A tendon is **a steel fibre with its own position in each element**: the `profile` is evaluated at each element's midpoint; (y, z) are section coordinates (as for `bars`), `s` is the distance along the member from its start (the end with the smaller global x, then y, z), a fraction of the member length when `normalised: true`. `linear` is the polyline through the control points; `quadratic` is **one parabola through each three consecutive points** (points 0-1-2, 2-3-4, ..., each through its own three; so the number of points must be odd and at least three, and an even number is refused by name). Before 1.0 it was **one least-squares parabola** through all the points, and a five-point continuous tendon was laid as a straight tendon at z = −0.08. The curve as laid (at the element midpoints, and sampled densely along the tendon) must fall inside the section's outline, or it is refused -- a parabola's vertex may lie between two control points and leave the section, and only the listed points used to be checked. The outline (the positions of `bars` are checked the same way): the outer rectangle of `rect`/`box`, the outer circle of `tube`, the union of the patches for `patches` (tolerance 1% of the section size, as patches often draw fillets as right angles: two corner bars of Hangzhou Bay are 25 mm outside the patches); box cells, tube bores and patch `holes` count as inside (a duct is a hole itself, and external tendons run in the box cell). The bounding rectangle used before accepted the empty corners beside a T beam's web and the corners of a tube's circumscribed square. The s in a refusal is in the profile's own units (a fraction when `normalised`).

A tendon carries an **initial strain**: `prestress` is the tendon's stress with the section unstrained, divided by the tendon's modulus; or `jack` is the stress at the jack and `friction` the post-tensioning loss profile the solid embedded reinforcement has long used (`elements/truss.tendon_profile`: σ(s) = σ_con·exp(−(μθ + ks)), reflected for anchor set), evaluated element by element along the tendon's own polyline.

**Elastic shortening comes out of the relaxation**: a tendon with an initial strain pulls on the section, the concrete shortens at the tendon, and the tendon stress drops -- the physics of releasing a pretensioned beam. A post-tensioning jack reading already includes this shortening, and `elastic_shortening: compensate` adds back the linear elastic strain of the section (gross, uncracked, all the section's tendons) at the tendon, so that the stress given is what the tendon **finally** carries (8% on 0.3's case). The section strain is solved with the gross section's 3×3 stiffness about the **origin** (with the first moments and the product of inertia), so T and box girders whose origin is not at the centroid are right too. Before 1.0 it was N/A + M·z/I (z and I about the origin), which on a T beam with the origin at the flange-web junction, 0.1 m above the centroid, added 20% too much (1200 MPa given as 1227). A section with no stiffness in some direction (`rect` with `ny: 1`, a plane model with all fibres and tendons at y = 0) is solved for the part that has stiffness by a diagonally scaled pseudo-inverse (1.0 reported "Singular matrix", earlier NaN).

The reflection length of anchor set is interpolated **within** an element, so the set is exactly the value given; before 1.0 it stopped at the midpoint of the first element passed and always set too much: a 20 m tendon in 20 elements with 6 mm set actually set 6.66 mm, with 1161 MPa at the anchorage (converged value about 1167, now 1168). A set larger than the tendon can absorb is refused by name, with the most that end can absorb (`jack: both` up to the tendon's midpoint; before 1.0 that number came out negative for jacking at both ends).

**Why curved tendons are right**: the tendon moment P·e(x) is taken element by element, and the jump in moment between neighbouring elements lands on the node -- the concentrated form of the "equivalent load". The camber of a simply supported beam with a parabolic tendon, 5Pe_mL²/48EI: −2.4% / −0.5% / −0.04% / +0.08% with 4/8/16/32 elements. A straight tendon: the section resultant is zero (the prestress is self-equilibrated, with no external force and no reaction), the concrete alone carries −P·e, the camber PeL²/8EI and end rotation PeL/2EI to 2e-3, and the roller end draws in by PL/EA plus the chord shortening of the bent axis κ²L³/24 (seen by the co-rotational element, 3% here). Kernel and reference path agree to < 1e-10 on models with tendons.

**The scale of the convergence test**: a prestressed member with no external load is self-equilibrated -- the external force is zero, and after shortening so are the reactions and the section resultants; what is left in `f_int` is the residual itself, and a relative residual would be it divided by itself. The tendon force is the load this analysis carries, so it enters the scale (`_tendon_force_scale`); without tendons it is zero and other models are unchanged to the bit. This is the scale of the relaxation path (`mode: relaxation`); the residual scale of implicit statics (`mode: static`) is in implicit.py.

**A tendon anchored within a member**: `extent: [s0, s1]` (in the units of `profile`'s s) -- the tendon exists only in the elements whose midpoint falls between its anchorages; in the others its row has zero area: no force, no weight, skipped by the sweep. The top-slab closure tendons over the piers of a continuous girder, and each span's own web tendons when three spans share a section, rely on it. The friction profile is computed along the tendon's own polyline; a short tendon (an 18 m closure tendon) cannot absorb a 6 mm anchor set, which `tendon_profile` refuses -- give `prestress` (the calculation is in the comments of `cases/case_24_hzb_girder.yaml`).

**A stress profile given along the tendon**: `prestress: [[s, σ], ...]` -- interpolated linearly in `s` and taken at the element midpoints; a table that does not cover every element the tendon is in is refused (no extrapolation). Two uses: first, **one post-tensioned tendon split into segments** (a corroded stretch in another steel, another bar section, say) -- giving each segment `jack` + `friction` would jack each from both ends, a different tendon; compute the whole tendon's friction profile once (`s.tendons[i]["sigma"]` is it) and give it segment by segment as a table, and it stays the original tendon (`tests/test_fiber_tendons.py`: split in two, initial strains to the bit, midspan deflection to 1e-6). Second, a **measured effective prestress** along the tendon.

**`s` with several fibre-section blocks**: `s` is along **each section block's own** member axis, from that block's start. When a beam is given several sections by stretch (`assign` to several regions), the later block's `s = 0` is at its own start, and a profile written in the whole beam's coordinates is misplaced; friction is also computed over that block's stretch alone. To change materials along a stretch, **keep one section and split the affected tendons** (the previous paragraph), rather than dividing the beam into section blocks.

**Drawing bars thick in ParaView**: the bar and tendon series are `.vtu` (unstructured grids), and the Tube filter takes vtkPolyData only, reporting "Input must be vtkPolyData" -- tick **Render Lines As Tubes** in Display and set the Line Width, or add Extract Surface before Tube.

**One entry per duct**: merging a group of ducts by strand count into one on their centroid line is equivalent bit for bit in vertical bending (the transverse coordinates enter only I_z and torsion), but the picture is wrong -- 22 ducts over a pier drawn as 4 coincident lines, web tendons running down the middle of the box. The Hangzhou Bay case now has 110 ducts, each with its own (y, z) profile: web tendons at mid-thickness of the web, moving outwards with height (|z| = 3.03 at the low point, 3.52 at the anchorage); **the plan alignment comes in by itself**, and friction accumulates the angle along the three-dimensional polyline (1% more loss than drawn straight in plan). The cost: fibre columns 243 → 324, the case 1.5 times slower; dead-load strains change by < 0.2%.

**How the sweep draws a tendon**: the bar and tendon series has one segment per element. The section integration samples a tendon **once per element** (at the midpoint), but drawing that value at both ends of the element turned a curved tendon into level steps -- on a 210 m beam it looked straight at a glance, which is what a user saw in ParaView. Each segment is now drawn between **the profile's values at the element's two ends**: continuous at the nodes, on the design profile, with the integration-point slices in between interpolated along the chord (< 3e-3 m off the parabola on 1 m elements). What is drawn is the tendon's geometry; what is computed is still the midpoint-sampled staircase -- they converge together as the mesh is refined.

**Reading one point of a section**: the `fiber` monitor (`{kind: fiber, region: node, field: strain|stress, at: [y, z]}`) -- at the integration point nearest the node (Lobatto: on the node; Gauss: at 0.21L), `strain` is the plane-section strain ε0 − y·κz + z·κy (the concrete's, without the tendon's initial strain), `stress` the stress of the fibre nearest the point. What a strain gauge on the deck or the soffit reads, the top- and bottom-fibre values of a thesis. `FiberBeamSolver.section_strains()` gives [ε0, κy, κz] at every integration point.

**Not here**: unbonded and external tendons (a force uniform along the length, which needs a non-local fibre), a stress-field output for tendons (to come with the sweep export), and the transfer length of pretensioning (it can be written as a `prestress: [[s, σ], ...]` table). Tendon relaxation (a steel's `relaxation`) and concrete creep and shrinkage (`creep`/`shrinkage`) **are here**, see `docs/creep.md`, `cases/case_29f`, `case_29g`; before 1.0 this paragraph said they were not. A fibre section's creep takes each element's elastic strain **at its midpoint**: a Lobatto rule with an odd number of points takes the middle integration point, a rule with an even number (the default `gauss2`, `lobatto4`) the average of the two points symmetric about the midpoint. Before 1.0 Lobatto took an end point (a cantilever crept 20% more); the first fix in 1.0 had even rules take the first of the two, 0.21 L from the element's own node 0, and the same cantilever crept 6.7% more fixed at its left end and 5.9% less fixed at its right.

## The sweep export: sweeping the section along the deformed axis

```
vfife run   cases/case_21_vecchio_emara_frame.yaml      # with fiber_stress in output.fields
vfife sweep cases/case_21_vecchio_emara_frame.yaml      # -> results/..._sweep.pvd + one .vtu per frame
```

A fibre beam's results are points and lines, and a collapse drawn from them is a few lines falling. The model has everything else: each member has a section outline (rectangle, box, tube -- the fibres are laid out in it), and each particle has an exact orientation. The sweep puts the cell each fibre stands for (a cell of the rectangular grid, of a box wall, a sector of a tube) on the section axes of the member frame and rotates it with each end particle's triad -- the section point at node n is x_n + R_n(y·e_y + z·e_z) -- joining the two ends into hexahedra: **the member is solid, one hexahedron per concrete fibre per integration point** -- the member is cut into slices at the midpoints between integration points (two slices for gauss2, three for lobatto3: [0, 0.25, 0.75, 1]), each slice carrying its own point's values without averaging. In ParaView it can be cut (Clip/Slice) to look inside; the cut face is the fibre-by-fibre damage map, and which end of the element the plastic hinge is at can be seen. `--surface` gives the outer surface alone. The surfaces bend and twist, and members that split (`fracture.split`) fall with their own particles: each frame writes `connectivity` (after a split the member hangs on spare particles, and the connectivity in the mesh groups is no longer its own).

What is drawn is what the section knows. With `fiber_stress` and `fiber_damage` in `output.fields`, each frame writes the fibre stresses (nel, integration points, fibres) (both paths write them; the kernel has one more output array) and a **damage class** -- each constitutive law answers `damage_class(state)` itself: 0 intact, 1 cracked (concrete pulled past f_t; concrete01 has no tensile strength and cracks on opening), 2 crushed (compressed past the peak strain ε_c0), 3 yielded (the steel has a plastic residual strain), 4 failed (the fracture latched by `failure`). The outline's edges are subdivided by the fibre grid (`ny`, `nz`, `nt`), each face facing one boundary fibre and taking its stress and class (the more severe of two integration points) -- in solid mode each hexahedron is its own fibre, with no nearest one to look for; bars and tendons are written as a separate line series (`<stem>_sweep_bars.pvd`, one segment per element per bar row, tendons along their own profiles), coloured by their own stress and class and rendered as tubes in ParaView. The member's internal forces, `fac` and `fractured` are carried along as cell data. The output is a ParaView `.pvd` collection with one `.vtu` per frame (meshio). The `fields` of case_11 and case_21 already include both fields, so `vfife sweep` works on them directly.

**Colour steps**: cell data is one number per slice, a staircase along the member. The solid output also carries point data, `fiber_stress_nodal` -- each fibre's stresses at its integration points interpolated linearly to each slice and extrapolated linearly beyond the Gauss points to the end faces; in ParaView apply Clean to Grid first (merging the coincident points of neighbouring members' end faces), then colour by `fiber_stress_nodal` for a colour continuous along the member. `fiber_damage` is a class and should be in blocks.

What it cannot do: fragments below the section -- concrete spalling and fragments flying off are a matter for solids and debris.

## Cover and confined core

A `fiber` section takes `cover` (a thickness measured in from each face) and `core_material` (a name in `materials`). Fibres whose centroids fall outside the core rectangle use `material` (unconfined cover, which spalls); the rest use `core_material`. The reference path makes two material groups of them; the kernel still inlines one concrete law and takes its parameters fibre by fibre from `cpar[e, f]`, and the two paths agree fibre by fibre (`tests/test_fiber_beam_kernel.py`).

`nz` must resolve the cover. In case_11 (400×400, 52 mm cover) the cover is 25% of the section with `nz=2` and converges to 43.8% from `nz=8`, against 45.2% exactly -- **a fibre spanning half the section depth is both cover and core and cannot be told apart**.

**A `patches` section uses another rule: a patch names its `material`**.

```yaml
patches:
  - {corners: [...], ny: 4, nz: 2, material: cover_conc}   # the top layer of a flange
  - {corners: [...], ny: 4, nz: 8, material: core_conc}    # the core the stirrups confine
  - {corners: [...], ny: 12, nz: 4}                        # none: the section's `material`
```

`cover` + `core_material` is **the rectangle's** rule -- "within `cover` of any free face is cover". It **does not carry over to patches**: the seam between two patches is not a free face, and judging by distance would make interior fibres cover. So a patch section **says it outright**: which patch uses which law. The names are resolved in `materials` and reach the solver by the same road as `core_material` (several patches of the same name form one material group, with no law built twice).

**The two rules may not be used together on one section**; doing so reports "two ways to say the same thing". A patch section without `material` is **unchanged to the bit** (`tests/test_patch_material.py` asserts the same group map). Holes can swallow whole cells, so which patch each fibre belongs to is returned by `patch_fibers(..., owner=True)`, not computed afterwards from `ny*nz`.

**A limit to know: `concrete02`'s initial modulus is tied to `2·fpc/epsc0`.** Mander gives the core a high peak strain (0.0084 here), so the core's Ec = 1.0e10 while the cover's is 2.56e10 -- 2.6 times softer, over 56% of the section. Run again with `epsc0=0.0033`, which keeps Ec: peak force 166.1 → 183.1 kN, ratios at the reversal points 0.97/1.05/1.13/1.06 → 1.12/1.07/1.11/1.02. **The improvement at large deformation holds either way (the spalling is real); the near match of the peak does not -- the honest reading of this model's peak accuracy is close to 1.10, not 1.00.**

## Strain penetration (the first-order effect of bond slip)

```yaml
assign:
  - {region: all, section: rc, anchorage: {region: base, bar_diameter: 0.020}}
```

In the elements that meet `region`, the bars' strain is reduced by `Le/(Le + Lsp)`, `Lsp = 0.022·fy·db` (Priestley). case_11: Lsp = 209 mm, base element 400 mm, factor 0.657. **No free parameter** -- Lsp follows from fy and db, both in the PEER table.

It is a **parameter** of `steel02`, not a material of its own: the fused kernel inlines one steel law, and a second law type sends the whole model back to the NumPy path. That is the direct reason per-element parameters (`spar[e, f]`) exist.

It behaves as expected: the base moment around yield drops 6-15%, while the ultimate capacity is unaffected (ratio 0.997 at a 40 mm tip displacement, with the bars fully plastic).

## The three steps together

| | peak force | stiffness at 8 mm | energy |
|---|---|---|---|
| one concrete, full bond | 1.15 | 1.37 | 1.30 |
| + cover spalling, Mander core | 1.00 | 1.28 | 1.24 |
| + strain penetration at the base | **0.89** | **1.15** | **1.10** |

Stiffness and energy approach the test monotonically, the best of the three steps; **the peak force passes through 1.0 to 11% low**. The ratios at the successive reversal points are 1.19 / 0.95 / 1.03 / 1.11 / 1.04 -- the envelope is within 11% everywhere and within 5% mostly, which describes this model better than a single peak ratio.

**Do not tune Lsp to bring the peak back to 1.0.** The remaining error is of the order of the uncertainty of the modelling choices already in the case -- that `concrete02` cannot keep both Mander's initial modulus and his peak strain is worth 10% on its own. Covering the uncertainty of one derived quantity with another is the mistake that kept recurring along this road.

## Fracture

```yaml
sections:
  rc:
    type: fiber
    material: concrete
    # ...
    fracture: {threshold: 1.0, split: false}   # the fraction that counts as broken; whether to fall when broken
```

The criterion itself **is not here**: when a fibre is finished is its own material's to say (`rupture_strain` of `rebar` and `steel02` -- one name, different criteria, see `docs/steel02.md`; the degradation + ULCF of `steel_cyclic`; `concrete01/02` have no fracture criterion yet). How a section adds its fibres up is the element's business, not an option (Cai et al. 2023, Eqs. 19/20: a fibre broken at any integration section no longer counts at all, FAC = broken fibres / total fibres). The three layers are in `core/fracture.py`.

- An element that reaches the threshold is **deactivated**: it contributes no internal force and its fibres are no longer driven, **its mass stays on the nodes** (erosion on the solid path deletes, and loses mass);
- deactivation **is not separation**: the broken member's ends are still the same two particles, and the member does not fall out. For it to fall, write `split: true` (requires `solver.mode: dynamic`): the model keeps a **bond graph** (particles are vertices, "an element's end is tied to a particle" is an edge, `core/topology.py`); fracture cuts the member's two edges, an end whose particle still holds other members moves to a particle of its own, **taking the member's share of the mass with it**, and the connected components then tell what has come away. What is conserved is the total mass and the total momentum (the mass moved keeps its velocity -- any other choice is an impulse from nowhere); a fragment's centre of mass falls at 9.806 m/s²;
- the particles for splitting are **reserved before the run**: two per element, parked on the ends they may replace, so no array changes size at run time and no capacity runs out (an end can separate once). The reservation happens **after the regions are resolved**, or a spare particle parked on a fixed support would be caught by `where: {x: 0}` and a fragment would be welded back to the support the moment it is born;
- after a split **the bound of the time step is measured again**: splitting takes mass off a particle; with the dead element also taken out of the stiffness the axial ratio is restored exactly and the rotational one to 0.56 (the bending coupling of two members at a shared node cancels, see 0.1). dt is not adjusted midway; when the bound does not hold a warning is printed;
- **all three levels are optional, and their cost was measured** (40 members, 64 fibres + 4 bars):

  | per step | fused kernel (0.21 ms/step) | NumPy path (8.6 ms/step) |
  |---|---|---|
  | no `fracture:` | 1.00× | 1.00× |
  | `threshold` alone | 1.20× | 1.00× (below the noise) |
  | with `split: true` | 1.30× | 1.12× |

  Checking for fracture is **a fixed cost per step** (0.03-0.04 ms on this machine), so the faster your step, the more it shows; the extra for splitting does not depend on whether anything breaks: the reserved particles triple the model's point count (orientation updates, boundary masks, ndof vectors), and it is paid as long as it is on. A splitting **event** (rebuilding K, recomputing the bound of dt, rebuilding the step's loads) takes 3.6-3.8 ms, less than half a step of the NumPy model;
- there is **no contact yet**: fragments pass through the floor below;
- two more element fields are written: `fac` (0-1, the progress of partial fracture) and `fractured` (the latched 0/1);
- staged relaxation latches only at **the converged commit** (trial states within a stage are discarded and recomputed, and a latch cannot be undone); the dynamic path latches step by step;
- when no law in the whole section has a fracture criterion, `fracture:` is refused rather than running to the end and reporting a building that did not fall;
- the fused kernel reads the deactivation flags (the same `active` array), but the three laws it can inline have no fracture criterion yet, so a section on the kernel path today cannot break by itself.
