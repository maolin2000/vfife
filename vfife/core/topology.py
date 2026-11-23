"""The bond graph: which particle each element end is tied to.

WHAT THIS FILE IS FOR. v0.1 gave fracture three layers and stopped at the
third one's first action: a fractured element stops carrying load and its mass
stays where it was lumped. Its own docstring names what that is not --

    "a beam cut in the middle of a frame does not fall out of it -- it hangs
     from whatever else those nodes are attached to"

-- and this file is the operation that makes it fall. Deactivation is a
statement about FORCES; falling out is a statement about TOPOLOGY, and the two
are different objects. A structure's topology in a particle method is a graph:
particles are vertices, and an element END tied to a particle is an edge. Cut
the two edges of a fractured member and the member, with its mass, becomes its
own connected component -- debris -- while the frame it hung in loses exactly
that member's weight.

WHY A BOND GRAPH AND NOT AN ELEMENT LIST. NVIDIA's Blast, which is the mature
engineering of this in graphics, keeps precisely
this: a support graph whose edges are the bonds between chunks, broken one at a
time, with connected components recomputed to find the islands that have come
free. The reason the edge is (element, end) rather than the element is that a
particle is the thing that gets duplicated, and it is duplicated per end.

WHAT IS DELIBERATELY NOT HERE: mass, velocity, degrees of freedom, elements'
internal forces, anything about beams. This is a graph and an allocator. The
solver decides what a new particle inherits (core/fiber_beam_solver.py:
`apply_pending_splits`), and it is testable on its own because of that.

THE ONE RULE THAT IS NOT OBVIOUS is `detach`'s: an end is given a NEW particle
only when the particle it sits on still carries other bonds. A cantilever tip
attached to nothing else is already the fragment's own particle, and copying it
would leave a massless orphan behind and inflate every count that follows. This
is the plan's "duplicate only the particles that straddle components", and it
is what keeps the particle count equal to the number of separations rather than
twice the number of fractures.
"""

from __future__ import annotations

import numpy as np

from .diagnostics import UserError


class BondGraph:
    """Particles as vertices, element ends as edges, with room to grow.

    `bonds[e, k]` is the particle that end k of element e is tied to. It starts
    as the mesh connectivity and is the only thing `detach` changes: the mesh's
    own array is left alone, and the solver copies from here.

    CAPACITY IS DECLARED UP FRONT and never exceeded. Not because growing a
    NumPy array mid-run is expensive -- a handful of fractures is nothing --
    but because the alternative is a model that quietly turns into a different
    model: a run that was going to produce three fragments and instead produces
    four hundred has found a defect, not a collapse, and it should say so
    while the answer is still small enough to look at.
    """

    def __init__(self, connectivity, n_particles: int, capacity: int = 0,
                 reserved=None):
        conn = np.asarray(connectivity, dtype=np.int64)
        if conn.ndim != 2 or conn.shape[1] != 2:
            raise ValueError("a bond graph needs (n_elements, 2) connectivity")
        if conn.size and (conn.min() < 0 or conn.max() >= n_particles):
            raise ValueError("connectivity refers to a particle that does not exist")
        if capacity < 0:
            raise ValueError("split capacity cannot be negative")
        self.bonds = conn.copy()
        self.n_original = int(n_particles)
        self.n_particles = int(n_particles)
        self.capacity = int(capacity)
        #: parent of every particle created by a split, in creation order
        self.parents: list[int] = []
        #: (element, end) each created particle was made for
        self.made_for: list[tuple[int, int]] = []
        #: RESERVED SLOTS: the particle each end is allowed to move onto,
        #: allocated before the run and already counted in `n_particles`.
        #: Two per element is the true upper bound on what fracture can
        #: need -- an end can separate once -- so a solver that reserves
        #: them cannot run out, and nothing is sized at run time. `capacity`
        #: and the allocating branch below are for the graph used on its own.
        self.reserved = None if reserved is None else np.asarray(reserved, dtype=np.int64)
        if self.reserved is not None and self.reserved.shape != self.bonds.shape:
            raise ValueError("reserved slots must be (n_elements, 2)")

    # ---------------------------------------------------------------- counts
    @property
    def n_elements(self) -> int:
        return len(self.bonds)

    @property
    def n_created(self) -> int:
        """Ends that have been moved onto a particle of their own.

        `len(self.parents)`, not `n_particles - n_original`: with reserved
        slots the particles existed before the run and the count that means
        anything is how many separations happened.
        """
        return len(self.parents)

    @property
    def room(self) -> int:
        return self.capacity - self.n_created

    def attachments(self) -> np.ndarray:
        """How many element ends sit on each particle."""
        counts = np.zeros(self.n_particles, dtype=np.int64)
        if self.bonds.size:
            np.add.at(counts, self.bonds.ravel(), 1)
        return counts

    # ------------------------------------------------------------ components
    def components(self) -> np.ndarray:
        """Component label per particle, over the bonds as they stand now.

        OVER THE BONDS, NOT OVER THE ACTIVE ELEMENTS, and the difference is the
        whole point: a fractured member carries no force but it still ties its
        own two particles together, so the fragment comes out as ONE component
        with the member's mass and length rather than two loose points. What
        separates the fragment from the frame is that `detach` cut its bonds to
        the frame's particles, not that the element stopped carrying load.

        Weighted union-find, near-linear; a frame is a few thousand vertices
        and this runs only when something fractures.
        """
        parent = np.arange(self.n_particles, dtype=np.int64)
        rank = np.zeros(self.n_particles, dtype=np.int64)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, j in self.bonds:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            if rank[ri] < rank[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]:
                rank[ri] += 1

        labels = np.array([find(a) for a in range(self.n_particles)], dtype=np.int64)
        # relabel to 0..k-1 in order of first appearance, so the number is
        # comparable between calls and readable in a log
        _, out = np.unique(labels, return_inverse=True)
        return out

    def n_components(self, attached_only: bool = True) -> int:
        """Number of connected components.

        `attached_only` skips particles that carry no bond at all. They are
        not fragments of anything -- they are what is left of a particle whose
        every element has detached from it -- and counting them would make the
        headline number grow for a reason that is not a separation.
        """
        labels = self.components()
        if not attached_only:
            return int(labels.max()) + 1 if len(labels) else 0
        attached = self.attachments() > 0
        return int(len(np.unique(labels[attached]))) if attached.any() else 0

    # ----------------------------------------------------------------- split
    def detach(self, element: int, held=None, ends=(0, 1)) -> list[tuple[int, int, int]]:
        """Cut the `ends` of `element` (both by default) loose from the structure.

        Returns one `(end, parent_particle, new_particle)` per end that was
        actually duplicated; an end whose particle carried no other bond keeps
        the particle it had and is not in the list (see the module docstring)
        -- UNLESS that particle is in `held` (a support: constrained, or
        otherwise tied to the world by something that is not a bond), in
        which case the end is duplicated anyway, because a fragment that keeps
        its root's particle stays clamped where it broke off.

        The element itself is untouched: it keeps its length, its section and
        its state, and whether it still carries force is the FRACTURE layer's
        business, not this one's. What changes is only who it is tied to.
        """
        e = int(element)
        if not 0 <= e < self.n_elements:
            raise IndexError(f"element {e} is outside the bond graph")
        counts = self.attachments()
        made = []
        for k in ends:
            p = int(self.bonds[e, k])
            if counts[p] <= 1 and not (held is not None and held[p]):
                continue                  # already this fragment's own particle
            if self.reserved is not None:
                new = int(self.reserved[e, k])
                if new == p:
                    continue              # the end is already on its own slot
            else:
                if self.room <= 0:
                    raise UserError(
                        "SPLIT_CAPACITY",
                        f"element {e} needs a new particle and the split capacity "
                        f"of {self.capacity} is used up ({self.n_created} created)",
                        "raise the split capacity, or ask why this run fractures "
                        "far more members than it was expected to")
                new = self.n_particles
                self.n_particles += 1
                counts = np.append(counts, 0)
            self.parents.append(p)
            self.made_for.append((e, k))
            self.bonds[e, k] = new
            counts[p] -= 1
            counts[new] += 1
            made.append((k, p, new))
        return made
