"""A Midas Civil stage model as a vfife case.

`convert(model, folder)` writes three files:

  <name>.msh     the line mesh (Gmsh 2.2): one physical group per set of
                 members that are born together and share a section, a
                 material and an orientation; point groups for the supports
                 and the loaded nodes
  <name>.yaml    the case: materials, sections, assign, supports, links,
                 gravity, and one analysis step per construction stage --
                 what it activates, removes, holds, releases and loads
  <name>.map.json  which Midas node and element became which, what each
                 region is, and what was approximated or left out

It is a translation of what `io/midas.read` understands, and it SAYS what it
did not carry over (`report["left_out"]`, `report["notes"]`). Lengths become
metres and forces newtons whatever the file's units.

New members are put where the model has them (`birth: design`, Midas'
default: a new node starts with no displacement, and a member is free of
stress as it then lies between its nodes) -- except in a stage that builds
members of the model's tangent-assembly group, whose new nodes are carried
along the tangent of what stands (`birth: tangent`).

What it maps to:

* BEAM -> a `beam` section, `shape: general`, with the area, the second
  moments and the torsion constant of the Midas shape (`section_properties`:
  pipe, solid round, box, H -- by its sizes or its database name -- and
  double channel are computed; a concrete-filled pipe is its steel pipe, a
  built-up box a plain one; others are refused by name);
* TRUSS -> a `bar` section; its `*PRETENSION` becomes `jack`, the force it
  is stressed to in the step that builds it -- Midas' "external"
  pretension. Of a cable modelled as several elements over sliding
  saddles, one is jacked and the others are cut to the length of their
  force (`prestress`): see "WHICH OF A CABLE'S ELEMENTS" below;
* an elastic link GEN -> a `connector` with its six stiffnesses, built in
  the first stage (from its boundary group's) in which both its ends are
  held by something, and born free of force there: a saddle's links wait
  for the cable they carry;
* an elastic link RIGID -> a stiff, weightless beam (a connector has no arm);
* `*CONSTRAINT` -> constraints; those of a boundary group activated in a
  later stage are that step's, kept; a deactivated group is released;
* self weight -> gravity on everything; nodal loads and uniform member loads
  (as their two end forces, no end moments) -> the loads of the stage that
  activates their load group, kept, and taken off again by their opposite
  where the group is deactivated;
* plates, temperatures, pressures, creep and composite sections' later
  parts: left out, counted.

Two things are done to the stiffness, because the static solver takes its
tangent by differences of the forces and their round-off is in proportion to
the stiffest member: a material stiffer than `rigid` times the ordinary one
is brought down to that (Midas models write 1e5 times steel for "rigid"),
and a stub stiffer than `stiffest` -- 5 mm of a pipe; a member shorter than
`stub` times its section's radius of gyration -- is made a stiff connector
(a beam that short cannot be under the limit and still carry a moment).
Both are counted in the report, and so are the members over the limit that
are left as they are.
A rigid link's arm is `stiffest` / 2 along and across at its own length
(so `stiffest`, not `rigid`, sets it); the moment through an arm of length
L meets k L^2 / 12, which for an arm of a few centimetres is a spring.
"""
from .convert import convert
from .geometry import _free_axis, _key, _local_y, _plain, _to_global, _up_of
from .shapes import G, _GRADE_WEIGHT, _poisson, _weight, section_properties
from .tendons import _drawn_in, _drawn_in_both, _rounded

__all__ = ["convert", "section_properties"]
