# Tutorial: from a piece of geometry to a ParaView picture

This page walks through one complete case: **mesh in Gmsh → write the YAML → run → look at the result in ParaView → check it**.

The example is a steel cantilever bracket with a point load at its tip. It is simple enough to check by beam theory by hand, and real enough to run into the trap everyone meets the first time (locking tetrahedra).

**Every command and every output on this page was actually run**; follow it and you should see the same numbers. It takes about 15 minutes, 3 of them solving.

> Convention: commands are written `python -m vfife ...`. After `pip install -e .` you can type `vfife ...`.

---

## 0. Setting up

```bash
conda activate vfife          # or your own environment
cd <project root>
python -m vfife list          # listing elements and materials means it is installed
```

You also need **Gmsh** (meshing) and **ParaView** (results), two separate programs, each downloaded and installed on its own.

---

## 1. Geometry and mesh in Gmsh

Create `bracket.geo`:

```c
SetFactory("OpenCASCADE");
L = 1.2; b = 0.15; h = 0.20;          // a 1.2 m cantilever, section 150 × 200 mm
Box(1) = {0, 0, 0, L, b, h};

Physical Volume("steel")   = {1};      // the solid itself
Physical Surface("fixed")  = {1};      // the fixed end at x = 0
Physical Surface("tip")    = {2};      // the loaded end at x = L

Mesh.MeshSizeMax = 0.05;               // target element size 50 mm
```

### The one thing that matters here: Physical groups

**Without Physical groups a mesh cannot be used.** Their names are how the solver knows where it is fixed and where it is loaded -- `region: fixed` in the YAML is this `Physical Surface("fixed")`.

The surface numbers (`{1}`, `{2}` above) **must not be guessed**. Click a surface in the Gmsh GUI to see its number in the status bar, or list each surface's centroid with a few lines of script:

```python
import gmsh
gmsh.initialize(); gmsh.open("bracket.geo"); gmsh.model.occ.synchronize()
for dim, tag in gmsh.model.getEntities(2):
    c = gmsh.model.occ.getCenterOfMass(2, tag)
    print(f"surface {tag}: centroid x={c[0]:.3f}")
gmsh.finalize()
```

```
surface 1: centroid x=0.000      ← fixed end
surface 2: centroid x=1.200      ← loaded end
```

A wrong number is not an error: the boundary condition goes onto another surface, and you get a reasonable-looking wrong answer. **Step 2 is there to catch this on the spot.**

### Making the mesh

```bash
gmsh -3 bracket.geo -o bracket.msh
```

`-3` makes three-dimensional elements. Forget it and there is only a surface mesh, which the solver will tell you.

---

## 2. Look at what is in the mesh first

**Do not skip this.** It takes two seconds and stops half the later problems:

```bash
python -m vfife mesh bracket.msh
```

```
mesh: /tmp/tut/bracket.msh
  element       tet4 (continuum), 1791 elements, 565 nodes
  formulations  full
  backends      numpy, numba, warp
  extent        x [0, 1.2]  y [0, 0.15]  z [0, 0.2]
  physical groups (use these names in `region:`)
    fixed                25 nodes   x=0, y [0, 0.15], z [0, 0.2]
    steel               565 nodes   x [0, 1.2], y [0, 0.15], z [0, 0.2]
    tip                  25 nodes   x=1.2, y [0, 0.15], z [0, 0.2]
  next          python -m vfife init <case>.yaml --mesh bracket.msh
```

What to read:

| what | why |
|---|---|
| `fixed` lies at `x=0` | the right surface was chosen. Had it said `x=1.2`, the numbers in the `.geo` are swapped -- cheaper to fix now than after a run |
| `element tet4` | Gmsh makes tetrahedra by default in 3-D. **Remember this; step 8 needs it** |
| `formulations full` | tet4 supports `full` **only**; anything else is refused |
| how many kinds of 3-D element | one is simplest. `element  mixed: tet4 (18253) + pyramid5 (2957)` runs as well -- the kinds share nodes and are solved as one. Pyramids are Gmsh's transition elements, made by a `Recombine Surface` line in the `.geo`. The cost is in section 8: a mixed mesh runs on the `numpy` backend only, and `formulations` is the intersection of the blocks' |

---

## 3. The YAML skeleton

```bash
python -m vfife init bracket.yaml --mesh bracket.msh
```

It asks three questions (the group names are those seen in step 2):

```
Imported Gmsh Physical groups: fixed, steel, tip
Fully fixed support region [fixed, steel, tip]: fixed
Loaded region [fixed, steel, tip]: tip
Total load on 'tip' as Fx Fy Fz in N [0 0 -1000]: 0 0 -10000
  created       bracket.yaml
  next          python -m vfife info bracket.yaml
```

The file it writes:

```yaml
version: 1
name: bracket
units: {length: m, force: N, stress: Pa}

mesh:
  type: gmsh
  path: bracket.msh
  scale: 1.0                 # 0.001 if the mesh was drawn in mm

materials:
  steel: {type: linear_elastic, E: 2.1e+11, nu: 0.3, rho: 7850.0}   # steel by default; change as needed

sections:
  body: {type: solid, material: steel, formulation: full}   # the formulation is chosen from the mesh's elements;
                                                            # the one solid section is given to the whole mesh

constraints:
  - region: fixed
    dofs: [1, 1, 1]          # 1 = fixed, 0 = free, in the order [ux, uy, uz]

loads:
  - region: tip
    vector: [0.0, 0.0, -10000.0]   # the **total force** on the region, shared out to its nodes

solver:
  backend: numpy
  mode: relaxation
  n_steps: 40000
  tolerance: 1.0e-06
  damping: {type: kinetic}

output:
  path: results/bracket.h5
  every: 100000
  fields: [displacement, von_mises]
  monitors:
    u: {kind: displacement, region: tip, dof: z}     # the loaded region's displacement along the load
    R: {kind: reaction, region: fixed, dof: z}       # the support's reaction in the same direction
```

### Adjusting the monitors

The skeleton already carries two monitors: the loaded region's displacement `u` and the support reaction `R` (along the load). Give them readable names and write a few more frames, and you will know whether the answer is right without opening ParaView:

```yaml
output:
  path: results/bracket.h5
  every: 2000                      # a frame every 2000 steps (for a ParaView animation)
  fields: [displacement, von_mises]
  monitors:
    tip_uz:   {kind: displacement, region: tip,   dof: z}
    reaction: {kind: reaction,     region: fixed, dof: z}
```

While at it, tighten `tolerance` to `1.0e-8` (a small linear elastic case converges fast).

> **`reaction` is the most valuable monitor.** It should be **equal and opposite** to the load you apply --
> the support holds up a downward load with an upward force. If the magnitudes do not match, a constraint or a
> load is in the wrong place; this is the cheapest self-check there is.

`reaction` and `displacement` read the region's **nodes**. Two other kinds read other things:

```yaml
  monitors:
    # an element field, restricted to a region. reduce: max | min | mean | absmax
    dc_span:   {kind: element, field: dc, reduce: max, region: span_block}
    # stress is a tensor: give the component (xx yy zz xy yz zx, this package's one Voigt order)
    sigma_bot: {kind: element, field: stress, component: xx, reduce: mean,
                region: span_soffit}
    # reinforcement and tendons. `of` is the **name on the reinforcement assignment**, not the geometry's key
    sigma_p:   {kind: bar, of: tendons, field: axial_stress, reduce: mean,
                region: tendon_flat}
```

- For `element`, `field` is any per-element quantity the material outputs (`d`/`dt`/`dc` for CDP), plus `stress` and `von_mises`; for `bar`, `axial_stress`, `axial_force`, `plastic_strain`. A misspelt name is refused **when the model is built**, with the available ones listed.
- An `element` region owns an element by the same rule as section assignment: **every node of the element** is in the region. A surface region one layer of nodes thick therefore owns no element, which is refused on the spot rather than quietly reducing to 0.
- `displacement` takes `relative_to: <region>`: it reports this region's mean displacement minus that region's. A body in free flight moves its centre of mass while its ends rotate about it -- `{kind: displacement, region: A, dof: z, relative_to: C}` reads that rotation (case_19 of 0.1).
- A `bar` region is a **geometric** test: reinforcement has its own nodes, unrelated to the node set the region resolves to, so the test is that both ends of a segment lie in the region's `box`. A region that is not a box (a Gmsh physical group, an explicit node list) is refused rather than approximated by its bounding box.
- Both may leave out `region`, meaning the whole model.
- An element deleted by erosion takes no part in an `element` reduction: its damage stops at the moment it leaves, and letting it decide `max` would make the monitor report history rather than state.

**Why these two exist.** A verification contract can read only monitors, whole-model extremes of fields and diagnostics. Before them, `cases/case_15` could not assert the tendon stress it is built on (written in `rebar_steps/`, out of the contract's reach) and used the camber as a stand-in; nor could it say "nothing crushed in the constant-moment zone", only "no element of the whole model crushed", whose global maximum the anchorage zone decides. Both have the shape of a monitor, so they are monitors now.

---

## 4. Checking before solving

```bash
python -m vfife info bracket.yaml
```

```
case: bracket
  preflight     PASS (0 errors, 0 warnings)
  mesh          565 nodes, 1791 elements
  element       tet4 / full (1 integration point)
  material      linear_elastic (0 state vars/point)
  mode          relaxation, mass = underwood, damping = kinetic
  backend       numpy
  time step     dt = 1.0000e+00, 40000 steps
  regions       fixed(25), tip(25), steel(565), all(565), surface(available)
```

`info` does not solve; it builds the model and prints it. **Preflight checks rigid-body motion, orphan nodes, element-material compatibility and more**; when it reports FAIL, do not force a run.

Besides the regions you named, two are always there: `all` (every node) and `surface` (the mesh's outer surface, computed when used).

---

## 5. Running

```bash
python -m vfife run bracket.yaml
```

The progress line while it runs:

```
  step     8000/40000   20.0%  resid 4.71e-06  KE 1.203e-12  resets 62    458 steps/s  eta   70s
```

- **`resid`** is the relative residual of the out-of-balance force -- below `tolerance` means converged. **The one number to watch.**
- `KE` is the kinetic energy, `resets` the restarts of kinetic damping (the solver dissipating oscillation; normal)

At the end:

```
  analysis      CONVERGED (residual …, analysis step 1 of 1)
  wall time     54.54 s (458 steps/s, 0.82 M element-steps/s)
  max |u|       2.351146e-04
  tip_uz        final -2.349903e-04
  reaction      final +1.000000e+04
  written       results/bracket.h5 (+ .xdmf time series, _last.xdmf fallback for ParaView; .ckpt.h5 checkpoint)
```

**Read the `analysis` line first**: only `CONVERGED` is an answer; `MAX_STEPS` and the other failures still write the result file (for diagnosis), but the command exits with code 3.

**Then `reaction`: +1.0000e+04 N, equal and opposite to the −10 kN applied.** Equilibrium holds; the boundary conditions are in the right places.

> To stop part way: **Ctrl-C**. The result file is closed properly and the checkpoint `results/bracket.ckpt.h5` is kept beside it; later
> `python -m vfife run bracket.yaml --restart results/bracket.h5 -o results/bracket_rest.h5` continues from there --
> into a **new file** (writing back to the original is refused: `RESTART_OVERWRITES`); the two files' frames together are one complete run.

---

## 6b. Curves: plotting and exporting

ParaView shows fields; **the monitors' histories** take two other commands -- no Python needed:

```bash
python -m vfife plot results/bracket.h5                       # every monitor against time, written to results/bracket.png
python -m vfife plot results/bracket.h5 --y tip_uz            # one of them
python -m vfife plot results/bracket.h5 --x tip_uz --y reaction   # one against another
python -m vfife export results/bracket.h5                     # every monitor -> results/bracket.csv
python -m vfife export results/bracket.h5 --monitor tip_uz --monitor reaction
```

**`--x` with `--y` is the important one**: reaction against displacement is the load-displacement curve, and with a cyclic amplitude it is the **hysteresis loop**.

The axis labels are automatic -- a monitor stores its `kind`/`region`/`dof` in the result file, so `--x tip_uz` is labelled `tip_uz  (displacement z on tip) [m]` (the unit from the case's declared `units`), and you need not remember what the name meant.

`plot` needs matplotlib (`pip install vfife[plot]`). Without it there is no traceback: it tells you what to install, and that `export` needs nothing extra.

---

## 6. Looking in ParaView

A run makes three files:

| file | use |
|---|---|
| `bracket.h5` | all the data |
| `bracket.xdmf` | **open this in ParaView** -- the full time series |
| `bracket_last.xdmf` | the last frame alone, a fallback for ParaView versions that have trouble with time series |

Steps:

1. **File → Open → `results/bracket.xdmf`**, choose the **XDMF Reader**, click **Apply**
2. In the drop-down at the top left change `Solid Color` to **`von_mises`**, and the one beside it to **Cell Data**
3. **Filters → Alphabetical → Warp By Vector**, Vectors `displacement`, Scale Factor **500** (the displacement is only 0.24 mm and invisible unmagnified), **Apply**
4. Press **▶** in the toolbar to play the loading

You should see the highest stress at the top and bottom faces of the root, decreasing towards the tip -- the classic bending distribution of a cantilever.

---

## 7. Checking -- not to be skipped

A number from a run is not a right number. This example can be checked by beam theory by hand:

```
section  b = 0.15 m, h = 0.20 m,  I = b·h³/12 = 1.0e-4 m⁴
bending  δ = P·L³/(3·E·I) = 1e4 × 1.728 / (3 × 2.1e11 × 1e-4) = 0.2743 mm
shear    δ = P·L/(κ·G·A),  κ = 5/6                            = 0.0059 mm
total                                                          = 0.2802 mm
```

The analysis gave **0.2350 mm** -- **83.9%** of theory, i.e. **16% too stiff**.

### Not a bug: tetrahedral locking

`tet4` is a constant-strain element: the strain is constant over the element, with too few degrees of freedom to express bending, so it behaves stiffer than the real structure. Refining the mesh does **not** cure it effectively (it converges very slowly).

The right thing is **a quadratic element**:

```bash
gmsh -3 -order 2 bracket.geo -o bracket_o2.msh
```

```bash
python -m vfife mesh bracket_o2.msh
```
```
  element       tet10 (continuum), 1791 elements, 3399 nodes
  formulations  full, b-bar
```

The same number of elements, nodes up from 565 to 3399 (the mid-edge nodes). And tet10 supports `b-bar`. Change two lines of the YAML:

```yaml
mesh:
  path: bracket_o2.msh       # ← the new mesh

sections:
  body: {type: solid, material: steel, formulation: b-bar}   # ← supported by tet10, better against locking
```

Run again:

```
  max |u|       2.811409e-04
  tip_uz        final -2.808897e-04
  reaction      final -9.999995e+03
```

**0.2809 mm, 100.2% of beam theory.**

| element | result | against beam theory |
|---|---|---|
| tet4 / full | 0.2350 mm | 83.9% (locking) |
| tet10 / b-bar | 0.2809 mm | **100.2%** |

### Remember this

> **For a solid member in bending use tet10, not tet4.**
> tet4 suits filling and mainly compressed regions. If first-order elements are a must, use hexahedra (recombine in `gmsh`); a hexahedron with `b-bar` does not have this problem.

This is also why **every case deserves one quantity checked by hand**. The 16% here was the choice of element; without the check it would have gone all the way into the conclusions.

---

## 8. What you see and what it means

| you see | it means | what to do |
|---|---|---|
| `error: unknown material 'linear_elasitc'` | a misspelt material name | the message suggests what you meant and lists every name |
| `element.Iz: Extra inputs are not permitted` | a wrong YAML field name | the string before the colon is the field's path |
| `FORMULATION_UNSUPPORTED: element 'tet4' does not support 'b-bar'` | element and formulation do not match | `vfife mesh` shows what the element supports |
| `UNKNOWN_REGION: region 'fixd' is referenced but unavailable` | a misspelt region name | check the names with `vfife mesh`. **It usually brings the `RIGID_BODY_MODE` below with it** -- a consequence, not a second fault; fix the name and both go |
| `RIGID_BODY_MODE: ... suppresses only N of 6 ...` | the structure is not held: it can translate or rotate as a whole | with an `UNKNOWN_REGION` as well, fix that first; otherwise check the constraints are sufficient |
| `MAX_STEPS: ... residual=... > tolerance=...` | the iterations ran out before convergence | see whether `resid` is falling slowly or stuck. Slowly: more `n_steps`; stuck: usually a mechanism in the model, or a softening nonlinear material. **A beam, shell or fibre-beam case that used to converge and now does not** is most likely the budget change from the half-step-start fix, see `docs/convergence_debugging.md` |
| `MIXED_MESH_BACKEND: ... runs on the numpy backend only` | more than one kind of 3-D element in the mesh, with `numba` / `warp` asked for | a mixed mesh **runs**, on the `numpy` backend only for now. For the fused kernels or the GPU make the mesh one kind: `Mesh.SubdivisionAlgorithm = 2;` in the `.geo` → all hexahedra (hexa8 is the better element anyway, with `b-bar` and no locking); or delete the `Recombine Surface` / `Transfinite` lines → all tetrahedra. `Mesh.RecombineAll = 0` **does not override** an explicit Recombine line; it must be deleted |
| `FORMULATION_UNSUPPORTED: element 'tet4' ...` (on a mixed mesh) | one `formulation` must suit every element block; `b-bar` is valid for hexa8 and not for tet4 | the message gives the intersection of the blocks, usually `full`; the `formulations` line of `vfife mesh` lists only the intersection |
| `[WARNING] REPEATED_ELEMENTS_IN_ONE_VOLUME: N tet4 cells were written more than once inside a single volume` | **Gmsh's meshing failed, and you missed its error.** The commonest cause is `Recombine Surface {:}` in the `.geo` recombining **internal interfaces** too -- a face with volume on both sides is a non-manifold quad boundary, which Gmsh says it does not support; it prints hundreds of lines of `Error : ... non-manifold quad boundaries not supported yet` **and writes the file anyway** | they are merged on reading (the same node set is the same element; kept twice they would double the stiffness and mass there), so it **can still run**. The root is in the `.geo`: recombine the outer surface only, not the internal interfaces. See "Recombining the outer surface only" below |
| `[INFO] MERGED_SHARED_ELEMENTS: N cells were listed under more than one entity` | one element belongs to two Physical groups (an element at an interface counted in both the skin and the core); MSH 2.2 even **writes it once per Physical group** | nothing to do: it is merged into one element that **belongs to every group naming it**, and region selection works as usual |
| `DUPLICATE_NODE` with a mixed mesh | two element blocks use **their own nodes** at an interface, coincident in position but numbered differently -- a non-conforming interface. Such a model reads and runs, and computes **two almost independent bodies** | look at the blocks' common face: quadrilaterals on the hexahedral side and triangles on the tetrahedral side are non-conforming by nature. Either recombine neither side of the interface (with pyramids to transition), or use `BooleanFragments` with shared faces to guarantee shared nodes |
| no convergence, and no telling where in the model | it may not be the model at all: a non-conservative load, a frame that is not objective, a fictitious mass that does not cover, or a state with no stable equilibrium | `python -m vfife probe case.yaml --result result.h5` asks these four element-independent questions **in the stalled state**, a few seconds for up to a few thousand dof. See `docs/convergence_debugging.md` |
| `NONFINITE` | inf/nan appeared: divergence | check for under-constrained parts. You may try a smaller `relaxation_safety` (default 0.6, try 0.3, i.e. a larger fictitious mass) -- **but it is no cure-all, and either direction may be worse**: on a softening material more mass was measured to fail **earlier**. See `docs/convergence_debugging.md` |
| a run finished, but a field is missing in ParaView | a misspelt name in `output.fields` is ignored silently | compare with `output_fields` in `vfife describe <element>` |

`--debug` on any command shows the full Python traceback, for problems with the tool itself.

---

## 9. One step further

This case is linear elastic with one load step. Real analyses usually need:

**A nonlinear material** -- change the `material` and leave the rest:

```yaml
materials:
  concrete:
    type: cdp                          # concrete damaged plasticity
    E: 3.0e10
    nu: 0.2
    rho: 2400.0
    compression: {source: gb50010, fc: 30.0e6}
    tension:     {source: fracture_energy, ft: 3.0e6, Gf: 120.0}
sections:
  body: {type: solid, material: concrete}
```

`python -m vfife list materials` lists them all, `python -m vfife describe cdp` gives one's parameters and scope.

**Staged loading** (needed for a load-displacement curve):

```yaml
solver:
  load_steps: 20                     # 20 equilibrium points, each converged to tolerance
  max_steps_per_load: 5000
```

**Several analysis steps + amplitudes** (self-weight first, then a pushover; or cyclic loading):

```yaml
amplitudes:
  drift: {cyclic: {peaks: [0.5, 1.0], repeats: 2}}

steps:
  - name: gravity                    # the model-level loads are applied in this step and kept
  - name: push
    constraints:
      - {region: top, dofs: [1,0,0], value: 0.02, amplitude: drift}
    load_steps: 16
```

**A moment on a solid face** -- a solid has no rotational degrees of freedom, and `moment` on a face region is a couple of fixed axis with a potential: −|M|·θ(x), θ the angle the face's best-fit rigid rotation turns about the axis of M. In the reference configuration its gradient is the textbook linear bending stress M z / I (the normal component, the linear shear stress of Saint-Venant torsion) turned into nodal forces by tributary area; `follower: true` makes the couple turn with the face with magnitude M throughout, smooth over a full turn and stateless; without `follower` it is a dead load with the reference distribution frozen, the couple decaying with the cosine of the rotation. Both are conservative. In three dimensions a couple of fixed orientation is not conservative in itself (a full turn about slightly different axes does different work); this potential pays only when the face turns parallel to the axis of M (which rolling up in bending never reaches):

```yaml
regions:
  tip: {plane: {point: [1, 0, 0], normal: [1, 0, 0]}}
steps:
  - loads:
      - {region: tip, moment: [0.0, 39.27, 0.0], follower: true}   # N m, about y
    load_steps: 40
```

**Regions of your own** (when the Gmsh physical groups are not enough):

```yaml
regions:
  base:      {plane: {point: [0,0,0], normal: [0,0,1]}}
  mid_span:  {where: {x: {min: 0.5, max: 0.7}}}
  corner:    {intersect: [base, mid_span]}
  interior:  {difference: [all, surface]}
```

**Recombining the outer surface only** -- for a regular outer surface with an unconstrained interior, do not write `Recombine Surface {:}`: it also recombines the internal interfaces `BooleanFragments` makes, and Gmsh reports errors and writes a bad mesh. Subtract the internal interfaces:

```gmsh
v() = BooleanFragments { Volume{1}; Delete; }{ Volume{2}; Delete; };
sh() = Abs(Boundary{ Volume{v(0)}; });
co() = Abs(Boundary{ Volume{v(1)}; });
sh() -= co();              // remove the internal interface the two share
Recombine Surface{ sh() };
```

Be clear too that **`Recombine Surface` does not turn tetrahedra into hexahedra**. It recombines *surfaces* only; the volume mesh is still tetrahedral, with a pyramid transition element tucked under each quadrilateral face. A real hexahedral boundary layer needs `Extrude { ... } Layers{n}; Recombine;`.

**Mixed meshes** -- hexahedra in the regular parts and tetrahedra in the irregular ones is the usual way with real members, and Gmsh almost always puts pyramids where they meet. The blocks share nodes and are solved as **one** body: masses and internal forces summed block by block, the time step the smallest of the blocks', one mesh with one set of fields in ParaView. Two costs: **(a)** the `numpy` backend only for now; **(b)** there is one `formulation`, which must suit every block, so hex + tet means `full`. `vfife mesh` says both directly.

**Which elements mix?** **Any combination** of the seven solid elements builds and runs -- the solver works block by block and does not care about element names. But for two blocks to be really "welded", they must present **the same kind of face** at the interface:

| | tet4 | hexa8 | wedge6 | pyramid5 | tet10 | hexa20 | wedge15 |
|---|---|---|---|---|---|---|---|
| **tet4** | · | ✗ | tri3 | tri3 | ✗ | ✗ | ✗ |
| **hexa8** | ✗ | · | quad4 | quad4 | ✗ | ✗ | ✗ |
| **wedge6** | tri3 | quad4 | · | tri3 | ✗ | ✗ | ✗ |
| **pyramid5** | tri3 | quad4 | tri3 | · | ✗ | ✗ | ✗ |
| **tet10** | ✗ | ✗ | ✗ | ✗ | · | ✗ | tri6 |
| **hexa20** | ✗ | ✗ | ✗ | ✗ | ✗ | · | quad8 |
| **wedge15** | ✗ | ✗ | ✗ | ✗ | tri6 | quad8 | · |

Two rules: **linear and quadratic elements never conform** (a linear element does not know the mid-edge nodes of a quadratic face); **hexa8 meeting tet4 directly does not conform either** -- one presents quadrilaterals, the other triangles. Pyramids and wedges connect on both sides, which makes them the standard transition elements -- which is what Gmsh's automatic pyramids are for.

hexa8 meeting tet4 directly (four corner nodes shared, the quadrilateral face covered by two triangles) was measured: it **passes the patch test** (residual 2.5e-12), the displacement jump at the interface is about **0.1%** of the deflection and vanishes linearly with refinement (h halved, the jump halved). It runs and converges, but the interface is not exactly continuous on any finite mesh. For exact conformity let Gmsh insert pyramids between them.

**Faster backends** -- a large model of one element type can use `backend: numba` (fused CPU kernels) or `backend: warp` (GPU). Only `hexa8` / `tet4` / `tet10` / `wedge6` have all three backends; `hexa20` / `wedge15` / `pyramid5` and all beam, shell and truss elements are `numpy` only for now. The `backends` line of `vfife mesh` tells you what the current mesh can use.

The reference cases are in `cases/`, with a table in `cases/README.md`.

---

## Appendix: all the commands

```bash
gmsh -3 -order 2 bracket.geo -o bracket.msh    # 1. mesh (quadratic elements)
python -m vfife mesh bracket.msh               # 2. what is in the mesh
python -m vfife init bracket.yaml --mesh bracket.msh   # 3. the skeleton
#    edit bracket.yaml: material, monitors
python -m vfife info bracket.yaml              # 4. check
python -m vfife run  bracket.yaml              # 5. solve
#    open results/bracket.xdmf in ParaView     # 6. fields
python -m vfife plot   results/bracket.h5      # 6b. curves
python -m vfife export results/bracket.h5      #     or a CSV
#    check the reaction and one quantity by hand   # 7. check
```
