# Structural line-element framework

> Written in the time of the old input model (`element:` / `material:` / `members:`) and a record of that state; the input model is now materials / sections / assign (see the user manual), and the case fragments on this page are no longer valid input.

2026-08-26: added the first shared structural-element gateway,
`vfife.core.structural.make_structural_solver`.  It selects `truss2`, `beam2`
or `beam2_timoshenko` after one common Gmsh `Line2` import and keeps YAML schema, named
regions, amplitudes, steps, nodal masses, ground motions, checkpoints and
XDMF output consistent.

`beam2` is a six-DOF/node, linear Euler--Bernoulli 3-D frame member.  It has
section `A, Iy, Iz, J`, local-axis orientation, rotational end releases,
nodal forces/moments, global or local uniform `line_load`, self weight through
`gravity`, direct linear static solution and explicit dynamics. Uniform line
loads use the consistent two-node beam load vector; released rotations are
statically condensed from that vector with the same member stiffness used by
the force law.
The verification case `cases/case_01_beam_cantilever.yaml` checks the Euler--
Bernoulli tip deflection and support reaction. A load that TRAVELS along a
line of members -- a vehicle on a bridge, an influence line -- is
`moving_load`; see `docs/moving_load.md` and cases 26/26b.

`beam2_timoshenko` uses the same six DOFs, loads, releases, output and VFIFE
co-rotational kinematics, while adding transverse shear flexibility. Its
section requires `ay`, `az`, `kappa_y` and `kappa_z`; local-y shear pairs with
`Iz`, and local-z shear pairs with `Iy`. The bending-plane parameter is
`phi=12 EI/(kappa G As L^2)`, so the matrix approaches `beam2` continuously as
shear rigidity increases and does not shear-lock. Numbered case 08 verifies
`u=P[L^3/(3EI)+L/(kappa G As)]` for a short cantilever. The implementation is
qualified at `verified` on NumPy; no GPU frame kernel or nonlinear section law
is claimed.

`connector2` adds a work-conjugate six-component spring/dashpot to the same
framework. It may use local or global component directions, diagonal or full
matrices, and reversible piecewise-linear skeletons. Physical connector
damping is evaluated only in dynamic physical time. See [connector2.md](connector2.md)
and numbered case 09.

Large-rotation mechanics remain in the co-rotational implementation and its
automated tests; the former convergence-tuning case is no longer part of the
small public case library. Beam relaxation requires both force residual
`solver.tolerance` and normalised kinetic energy `solver.kinetic_tolerance`.

The dynamic and quasi-static member-force paths are co-rotational: they remove the member rigid
translation and chord rotation, apply the classical section law in the pure
deformation frame, then rotates force and moment resultants back to global
coordinates.  This follows the VFIFE compatible-description principle.  The
quasi-static equilibrium is obtained with Underwood masses and kinetic dynamic
relaxation, rather than reusing a fixed global stiffness matrix.

Verification performed on 2026-08-26: `tests/test_beam2.py`,
`tests/test_truss2.py` and `tests/test_gmsh.py` passed (59 tests).  The
example writes `results/case_01_beam_cantilever.h5` with XDMF sidecars and
reports `u_z = -1.6666667e-5 m`, equal to `-PL^3/(3 E Iz)`, and a `+1000 N`
root vertical reaction.

Solid models retain their independent three-translational-DOF path. Line
structures may use a homogeneous `truss2`/`beam2`/`beam2_timoshenko` assignment or the mixed
six-DOF assembly described below.

## Mixed truss--frame models

Bar, beam or connector sections assigned to a line mesh's groups (the `mixed_line` path) activate one six-DOF structural particle system.
Each entry in `members` names a Gmsh Physical Curve region and assigns
`truss2`, `beam2`, `beam2_timoshenko` or `connector2` plus its properties. Every Line2 must belong to exactly one
entry; overlaps and omissions are setup errors. Trusses contribute current-axis
axial force and translational mass, while beams use the VFIFE co-rotational
reverse-motion/pure-deformation/forward-force path. See
`cases/case_03_mixed_frame.yaml` for the minimal two-member example.
Its guided free joint gives `u_z=-6.66667e-5 m`, the analytical sum of beam
bending (`-PL^3/(3EI)`) and truss shortening (`-PL/(EA)`). A single truss end
has no transverse or rotational stiffness, so those mechanism DOFs are
explicitly constrained in the example.

### Cables

A `bar` section on a line region is a cable when its assignment says so:

```yaml
assign:
  - {region: main_cable, section: strand, unstressed_length: 102.0, tension_only: true}
  - {region: stays,      section: stay,   prestress: 6.0e8,        tension_only: true}
```

Every truss member has an unstressed length `L_u`, its mesh length unless
stated. `prestress` is the effective stress the bar carries in the mesh
geometry, `L_u = L_0 / (1 + sigma/E)`; `unstressed_length` is a group's
total unstressed length, shared out over its elements in proportion to
their mesh lengths (one cable, stated by the length it was cut to). The two
are the same statement and only one may be given. The force is then
`E A (L - L_u) / L_u`, so a cable meshed on its design geometry carries its
design tension at `u = 0`: the dead-load state of a cable-stayed or
suspension bridge is a statement, not a search. `tension_only` clips the
force at zero, so a stay that goes slack carries nothing.

A cable's transverse stiffness is its tension over its length -- zero on a
slack straight chord -- and a fictitious mass sized on that row would be
the floor and the first transverse force would blow up. A cable member
therefore puts its axial stiffness on all three translations of its nodes
for the mass, an upper bound on anything it can develop; the sagging
motion is over-massed by 1/eps and converges slower for it. Plain trusses
keep their masses to the bit. `cases/case_22_cable_catenary.yaml` starts a
102 m cable on the straight 100 m chord and finds Irvine's elastic catenary
to 0.03% in 45 s; a run whose frame solver default ramp is a fifth of the
budget should say `stage_ramp_iterations` when nothing is prescribed.

### Form-finding

A cable bridge is drawn at its design geometry, and that geometry is where
it must sit under its dead load; what is not drawn is how long each cable
was cut. `vfife formfind` finds those lengths:

```
vfife formfind cases/case_23_suspension_bridge.yaml --groups main_cable,hangers
```

relaxes the dead-load step (the first step, or `--step k`) with the
current unstressed lengths, resets every element of the named cable
groups by `L_u <- L_u L_0 / L` (a member that ended too long is cut
shorter, one that ended too short is let out), and repeats until no node
moves more than `--tol` (1 mm). Convergence is geometric -- the bridge of
case_23 goes from 1.1 m to 0.2 mm in four relaxations -- and the found
case is written next to the source with per-element `unstressed_lengths`
on the matching `assign` entries (the third way of stating a cable's
length; element order is ascending element id within the group, which is
the mesh's order). The beams are not touched: a deck at zero displacement
under dead load carries no bending, the classic assumption that the girder
was built to profile. A system whose drawn geometry is not one the cables
alone can hold does not converge, and the command says so (exit status 3)
rather than pretending. `vfife.formfind.find` returns the same as a
`FormFound` (lengths, forces, history).

`cases/case_23_suspension_bridge.yaml` is the worked example: the
form-found lengths reproduce the closed form of a weightless parabolic
cable to 4e-7, and a live-load step then meets the linearised deflection
theory (Irvine ch. 6) to 0.1% on the cable's tension increment and 0.5% on
the deflection.

## Member actions: an envelope is not a diagram

Member results carry one value per member (`axial_force`, `shear_y`,
`shear_z`, `torsion`, `bending_y`, `bending_z`), their envelopes
(`*_max_abs`) and the local end actions with `_i` and `_j` suffixes. **They
are not three spellings of the same thing.**

Since v1.0 `bending_z` is SIGNED: the internal moment at the end with the
larger magnitude, with its sign (tension-positive axial force, right-hand rule
on the +x face; docs/result_format.md has the convention). `bending_z_max_abs`
is `max(|M_i|, |M_j|)`, the unsigned magnitude the plain name held before. A
maximum rather than an average on purpose: averaging a linearly varying moment
halves a cantilever's support moment. The cost of one value per member is the
moment's shape: where the moment passes through zero INSIDE a member, the
reported value jumps from one end to the other (and, signed, changes sign), and
a picture coloured by it is discontinuous everywhere the real diagram is
smooth.

The diagram is the signed pair. The internal moment is `-M_i` at the i end and
`+M_j` at the j end, linear between them, so ONE value per element cannot
carry it and a per-element ParaView array is the wrong container: the diagram
needs a value at each end, which means two points per member. Written that
way it is continuous along a run of members and discontinuous exactly where it
should be -- at a joint another member frames into, which delivers its own
moment. `benchmarks/sweep_confederation.py` is a worked example (a pier top
where the girder's diagram jumps by the shaft's moment).

So: `bending_z_max_abs` for "how hard is this member worked", `bending_z` for
which way (sagging or hogging), the `_i`/`_j` pair for a diagram, for
connection design and for equilibrium checks.

`vfife sweep`, which draws a member as the solid its section outline sweeps
out, is for fibre-beam models: it needs a section that HAS an outline. A
`general` section states area and second moments and says nothing about where
the material is, so there is nothing to draw and sweep refuses it rather than
inventing an equivalent rectangle (for a box girder that rectangle is 30% too
deep). Every beam section in `cases/` is `general` today, which is why the one
case that wanted a picture of its section brought its own script.

## Modal analysis

```
vfife modal cases/case_23_suspension_bridge.yaml --result results/case_23_suspension_bridge.h5 --frame 00001501 --modes 8
```

`vfife modal` solves `K phi = omega^2 M phi` at a state: K is the
finite-difference tangent of the internal force (`testing.invariants.tangent`,
2 n force evaluations -- a few thousand dofs at most, and it refuses past
`--max-dofs`), M the lumped physical mass the dynamic mode integrates with.
Held dofs are removed; massless dofs are condensed out when they carry
stiffness (Guyan) and dropped when they carry none (a truss node's free
rotation). The table gives frequency, period and the effective modal mass
fraction in x, y, z; the shapes go to `results/<case>_modes.h5` as
(n_modes, nn, stride) node fields.

`--paraview [DIR]` also writes the shapes for looking at:
one collection per mode, `--frames` snapshots over one period (24), each the
reference mesh displaced by `scale * phi * sin(2 pi t)` -- the points ARE the
displaced positions, so a mode animates on Play with no filter in front of
it -- and `modes_all.pvd` with every mode on one time axis, mode k on
[k-1, k). `--mode-scale` is the largest drawn translation (default 3 % of the
model's largest extent, which is right for global sway and too much for a
torsion mode of a wide deck). The cells carry the mesh's physical groups as
`group`, so a Threshold picks out the girder, the tower or the cables; the
legend is in `modes.txt` with the frequency table. Massless dofs that were
condensed out follow their neighbours in the shapes (u_s = -K_ss^-1 K_sr u_r);
until this stage they were left at zero.

The state matters. Asked at the drawing (`--result` absent) a cable is a
slack chord with no transverse stiffness and a prestressed girder has no
axial force; the geometric stiffness that makes a bridge's modes what they
are lives in the tangent at the LOADED state, so a cable-supported bridge
is asked at its dead-load frame. Held: the Euler beam's frequencies (0.1 /
0.4 / 0.9 % on 20 lumps), Irvine's cable modes at the catenary (the
antisymmetric string modes and the symmetric transcendental equation, 1 %
on a 4 % sag; the theory is a shallow-cable one and the 8.8 % cable of
case_22 sits 3 % below it), and case_23's vertical modes against the
linearised deflection theory's eigenproblem on a fine grid
(`benchmarks/generate_suspension_bridge.vertical_modes`): 0.8 / 0.3 / 0.7 /
1.1 % on the first four with 30 members.

## Verification audit trail

Numbered cases have independent contracts in `verification/`. Running
`python -m vfife verify cases/case_NN_topic.yaml` checks the generated HDF5
against analytical, public-benchmark, independent-reference, or explicitly
labelled regression targets and writes JSON plus Markdown reports. Reports
record backend, device and precision separately. Terminal beam/mixed residual
and kinetic-energy diagnostics are persisted even when convergence occurs
before the first progress-print interval, so a successful result remains
auditable after the run.
