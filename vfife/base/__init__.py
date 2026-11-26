"""The bottom layer (1.22): what every other layer uses and that uses
nothing of vfife's -- the errors a caller sees (`errors`), the degrees of
freedom (`dof`), finite rotations (`rotations`), the assembly of element
vectors (`scatter`), and the stress and strain notations (`voigt`,
`tensor`). Elements, materials and kernels import from here; until 1.22 these
lived in `core` and the top of the package, and the lower layers reached up
for them. The old module paths still work (each is this one).
"""
