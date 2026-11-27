# Degrees of freedom and mixed assembly

`vfife/base/dof.py` (before 1.22, `vfife/core/dof.py`) is the one map from the physical components of a node to the global equation numbers. An element must not assume `6*node+dof`; it asks `DofLayout` for its map.

## Layouts

```python
solid = DofLayout.translations(n_nodes)  # ux, uy, uz
frame = DofLayout.frame(n_nodes)         # ux, uy, uz, rx, ry, rz

member_map = frame.element_dofs([node_i, node_j])
translation_map = frame.node_dofs(nodes, TRANSLATION)
rotation = frame.field(global_vector, ROTATION)
```

This lets:

- solids and stand-alone bars keep a compact three-dof nodal state;
- beams and mixed line models use the six-dof state;
- a coupled model assemble the solids' forces and masses into the shared translations only;
- the rotation slots of a node no beam reaches stay inactive, rather than being given a small false mass;
- new shells, springs, connectors and interface elements ask for the components they need by name.

`SparseAssembler` assembles matrices through each element's explicit equation map and checks the size of the local matrices. `DofLayout.add_nodal()` assembles nodal loads or coupling forces by named component.

## The VFIFE geometry

The layout unifies who owns a degree of freedom and how things are assembled; it changes no element's deformation. A beam still removes the rigid translation and rotation first and finds its internal forces from the local deformation; solids and bars keep their co-rotational / VFIFE paths. Mixed assembly therefore brings no small-displacement global stiffness into the update of internal forces.

The beam's 12×12 local-to-global transformation uses `repeated_rotation()`, a deterministic dense block rotation. It replaced SciPy's `sparse.block_diag`, whose return type was changing, and removed 36,303 deprecation warnings from the tests that used it.

## Where it is used

- the solid `Solver`: registers the three-translation layout;
- `TrussSolver`: registers the three-translation layout;
- `BeamSolver`: degrees of freedom, nodal loads, mass access, monitors and matrix assembly through the layout;
- `MixedStructuralSolver`: beams and bars share an equation map and the sparse assembler;
- `CoupledSolver`: the solids' masses and internal forces are assembled into the named translations only, and which rotations are active follows from the named rotation components.

## A new element

A new element must:

1. declare the names of its physical degrees of freedom in the capability registry;
2. make its equation map with `DofLayout`;
3. assemble through the common interface, without copying stride arithmetic;
4. leave the degrees of freedom it does not own inactive, and not hide a mechanism with a small mass;
5. pass the tests of local matrix size, nodal force balance, rigid-body motion, mixed assembly and a classical solution.
