# E03 Four-node co-rotational Mindlin–Reissner shell

> **Experimental, not covered by the 1.0 promises.** A review measured that the stabilisation of the
> default `hourglass_coeff: 0.1` makes bending too stiff, and refining the mesh does not remove it (a
> clamped plate converges to 0.980; in a strip under pure bending the rotation is 0.965, 0.828 and 0.706
> of the exact value at element aspect ratios 1, 5 and 10), with the output moments low in the same way;
> distorted quadrilaterals lock in thin plates and fail the bending patch test (the claim further down
> that the patch test and distorted meshes are covered does not hold). With `hourglass_coeff: 0` large
> rotations are exact and regular meshes converge to 1.00. Rewriting the element as MITC4 is what will
> fix it, planned after 1.0. The triangle `shell3` locks in shear: do not use it.

`shell4` is a homogeneous linear elastic quadrilateral shell with nodal degrees of freedom `[ux, uy, uz, rx, ry, rz]`. The element removes the rigid translation and the finite rigid rotation in its current local frame, then computes the membrane, bending and transverse shear deformation, keeping VFIFE's separation of motion.

## Input

```yaml
materials:
  steel: {type: linear_elastic, E: 2.0e11, nu: 0.3, rho: 7850}
sections:
  plate:
    type: shell
    material: steel
    formulation: selective
    thickness: 0.01
    shear_correction: 0.8333333333333333
    drilling_factor: 1.0
    hourglass_coeff: 0.1
    warpage_tolerance: 0.05
```

- Membrane and bending: 2×2 integration;
- transverse shear: selective one-point integration;
- the zero-energy shear modes are controlled by a full-minus-reduced energy scaled by `t²/A`, of the order of the bending stiffness `Et³`, so it does not become a residual shear spring in thin shells;
- the drilling constraint penalises the nodal normal rotation minus the in-plane continuum spin, which is exactly zero for a rigid in-plane rotation;
- a reference surface warped beyond `warpage_tolerance` of the element's in-plane size is refused: refine or rebuild the mesh.

Nodal forces and moments, uniform pressure, tractions, self-weight, added nodal masses, base excitation, dynamic and relaxation and checkpoints are supported. The only material is `linear_elastic`.

## Backends

`solver.backend` is `numpy | numba | warp`:

- `numpy`: all elements in one batch (einsum), the readable reference;
- `numba`: a serial fused kernel per element, about 2500 iterations/s on a 256-element roof (numpy about 900). On two cores `prange` was 20 times slower than serial -- each element is a few microseconds of work, less than the thread scheduling -- so it is serial;
- `warp`: the whole staged relaxation stays on the device. The six-dof state is treated as a (2·nodes, 3) array, so the solid backend's nodal integrator, boundary conditions and reduction kernels are reused unchanged; each iteration returns six scalars to the host (kinetic energy, the four norms of the convergence test, the number of faces refused for warpage). `precision: mixed` stores the 24×24 local stiffness in fp32 with all geometry in fp64; `fp64` is for checking. Only displacement constraints and loads under `mode: relaxation` are taken; base motion, velocity and acceleration constraints and `mode: dynamic` go through `numba`.

A node's local rotation is its orientation relative to the face frame, log(R·T_n·R₀ᵀ), with the orientation updated by a spin multiplication each step. The three are pinned to each other (`tests/test_shell_at_scale.py`): numba and warp-fp64 internal forces agree with numpy within 1e-9, warp-mixed within 1e-6; the staged relaxation on the device converges at the same iteration as on the host, with the same curve. `shell3` stays on the per-element reference loop.

Case 17 (the hinged cylindrical roof of Sze, Liu & Lo 2004, 256 elements, displacement control through the whole snap-through) is the first benchmark that needs this speed: limit load 0.7485 P_max against 0.7420 in the paper, final rising branch 1.0224 against 1.0208.

## Output and ParaView

XDMF uses the `Quadrilateral` topology. Nodal output: `displacement`, `rotation`, reactions and reaction moments; element output: membrane forces, moments, shear forces, top and bottom `Tensor6` stresses and top and bottom von Mises. The engineering Voigt order is `[xx, yy, zz, xy, yz, xz]`.

Open `results/case_10_clamped_square_shell.xdmf` for the history (`_last.xdmf` for old ParaView versions). Show the displacement with `Warp By Vector`; the top and bottom stresses are components in the shell's local axes, not in global directions.

## Shells with beams

`shell_line` reads the Quad4 shells and the Physical Curves named by `members` from one Gmsh file. They share all six nodal degrees of freedom, with no interpolation or penalty coupling. Boundary Line2 elements no member group names are not built as beams. The mixed path takes equivalent nodal loads on beams, bars and connectors; a beam's distributed `line_load` is not yet taken there and must be converted to nodal loads first.

## What is verified

The tests cover the membrane and bending patches, finite rigid motion, exactly six rigid zero-energy modes, distorted quadrilaterals, the balance of surface loads, thin-plate locking, 4×4 and 8×8 mesh convergence, the free vibration of a simply supported square plate and shell-beam shared assembly. Case 10's centre deflection is within 2.73% of the classical solution for a clamped square plate. The qualification level is `verified`; it is not a laminated, nonlinear, buckling-eigenvalue or contact shell.
