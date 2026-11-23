# Conforming solid–line assembly

The coupled path (a solid section on the volume, bar/beam sections on its line groups) imports one supported solid topology and Gmsh Line2
members from the same `.msh` file. Node tags are mapped once across both blocks.
A coincident tag is therefore one node—not two nodes tied approximately—and its
three translations receive both solid and member forces and masses.

```yaml
sections:
  body:  {type: solid, material: steel, formulation: full}
  frame: {type: beam, material: steel, shape: general, area: 0.01, iy: 1.0e-4, iz: 1.0e-4, j: 2.0e-4}
  brace: {type: bar, material: steel, shape: general, area: 5.0e-4}
assign:
  - {region: all, section: body}
  - {region: frame_curves, section: frame, orientation: [0, 0, 1]}
  - {region: brace_curves, section: brace}
```

A solid section on the volume plus bar/beam sections on the mesh's line
groups is the coupled path; every Line2 must be in exactly one assignment. A member network must
share a node with the solid. Rotational slots on solid-only and truss-only nodes
are automatically inactive, rather than being stabilized with artificial
springs or masses.

Before solving, preflight assembles the actual released beam rotational
stiffness and checks its free-DOF rank. The solver repeats this guard during
direct Python construction. A torsional or bending mechanism is therefore an
explicit error, not a seemingly stable response produced by minimum masses.

This connection transfers the three nodal force components. A continuum point
has no rotational degree of freedom, so a beam end moment cannot enter a solid
through one shared point. Constrain that beam-end rotation when the physical
joint is pinned or fixed externally. Moment-spreading rigid/MPC connections are
deliberately outside this implementation.

Current scope is one solid topology plus Line2, `linear_elastic` material and
the NumPy backend. Rayleigh mass damping (`alpha`) is supported; stiffness
damping (`beta`) is rejected until a combined tangent is available. Embedded
reinforcement cannot be mixed into the same case—represent it as Line2 members.

The result HDF5 has two ParaView views: the normal `.xdmf` contains the solid,
and `_members.xdmf` contains beam/truss cells. Open both; they reference the same
HDF5 node coordinates and displacement history. Case 07 is the executable
shared-node verification example.
