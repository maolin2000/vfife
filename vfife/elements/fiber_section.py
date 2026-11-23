"""Fiber section discretization for beam2_fiber.

Each fiber carries one uniaxial material point (piecewise_uniaxial / rebar /
steel02 / steel_cyclic / concrete0x / linear_elastic mapped to axial) and
contributes N = sum sigma*A, My = sum sigma*z*A, Mz = sum -sigma*y*A .

FOUR SHAPES, and the two hollow ones are not a convenience: a steel spatial
structure is built of tubes, and a tube's material sits in a thin ring far
from the centroid. Discretising it as a solid circle (or worse, as a
rectangle of the same area) puts fibres where there is no steel, which
changes the very thing a fibre section exists to compute -- when the extreme
fibre yields, and how much of the section is left after it does. Cai et al.
(2023) model their tube members with 20 fibres around the ring and their
Kiwitt-8 dome with a 114 x 3 mm tube; neither is expressible as a solid.
"""
from __future__ import annotations
import numpy as np

def rect_fibers(width: float, height: float, ny: int, nz: int):
    """Return (y, z, area) for a rectangular section centred at the origin.

    y in [-width/2, width/2] (local ey), z in [-height/2, height/2] (local ez).
    Uniform grid, equal area per fiber.
    """
    if width <= 0 or height <= 0:
        raise ValueError("rect fiber width/height must be > 0")
    if ny < 1 or nz < 1:
        raise ValueError("rect fiber ny/nz must be >= 1")
    dy = width / ny
    dz = height / nz
    ys = np.linspace(-width/2 + dy/2, width/2 - dy/2, ny)
    zs = np.linspace(-height/2 + dz/2, height/2 - dz/2, nz)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    y = yy.ravel()
    z = zz.ravel()
    area = np.full_like(y, dy * dz, dtype=float)
    return y, z, area

def tube_fibers(diameter: float, thickness: float, nt: int, nr: int = 1):
    """A circular hollow section (CHS): `nt` fibres round the ring, `nr` through.

    `diameter` is the OUTER diameter, the convention every steel section table
    uses (a 114 x 3 tube is 114 mm outside, 108 mm inside).

    THE FIBRES SIT AT THE RMS RADIUS OF THEIR OWN WEDGE, sqrt((r0^2+r1^2)/2),
    and not at its area centroid. That looks like the wrong choice and is the
    right one, for a reason worth writing down because it is easy to get
    backwards:

        a lumped point cannot carry both the first and the second moment of
        the cell it stands for. At the AREA CENTROID the first moment is
        right and the second is too small -- a wedge has a second moment
        about its own centroid, and a point has none. At the RMS radius the
        second moment is exactly right, and the first moment is still zero
        BY SYMMETRY, because every wedge of a ring is congruent and their
        centroids sum to zero whatever radius they are placed at.

    So for a ring, and only because every cell is congruent, both can be had.
    Measured on a 114 x 3 tube against A = pi/4 (D^2-d^2) and
    I = pi/64 (D^4-d^4):

        nt = 8, 20, 40 fibres    A and I exact to 1e-15 (this placement)
        the same at the centroid  I low by 5.1%, 0.84%, 0.23%

    `rect_fibers` above is left on the ordinary centroid grid: a solid section
    is refined in both directions anyway, and moving it would move every
    verification number in the library for a discretisation error the analyst
    controls. A thin wall cannot be refined the same way -- 20 fibres round a
    ring is what the paper uses and what the cost allows -- so it is worth the
    fifteen characters here.
    """
    if diameter <= 0 or thickness <= 0:
        raise ValueError("tube diameter and thickness must be > 0")
    if thickness >= diameter / 2:
        raise ValueError(
            f"tube thickness {thickness} is not less than the radius "
            f"{diameter / 2}; that is a solid bar, use type: circle")
    if nt < 3 or nr < 1:
        raise ValueError("tube needs nt >= 3 fibres round the ring and nr >= 1")
    r_out, r_in = diameter / 2.0, diameter / 2.0 - thickness
    dtheta = 2.0 * np.pi / nt
    ys, zs, areas = [], [], []
    for ir in range(nr):
        r0 = r_in + (r_out - r_in) * ir / nr
        r1 = r_in + (r_out - r_in) * (ir + 1) / nr
        r_rms = np.sqrt(0.5 * (r0**2 + r1**2))
        a = 0.5 * dtheta * (r1**2 - r0**2)
        for it in range(nt):
            theta = dtheta * (it + 0.5)
            ys.append(r_rms * np.cos(theta))
            zs.append(r_rms * np.sin(theta))
            areas.append(a)
    return np.array(ys), np.array(zs), np.array(areas)


def box_fibers(width: float, height: float, thickness: float,
               n_width: int, n_height: int, nr: int = 1):
    """A rectangular/square hollow section (RHS/SHS), walls only.

    `width` (y) and `height` (z) are OUTER dimensions. The four walls are
    discretised separately so that the corners are counted once: the top and
    bottom walls span the full width, and the two side walls span only what is
    left between them. `n_width` fibres along each horizontal wall, `n_height`
    along each vertical one, `nr` through the thickness.
    """
    if width <= 0 or height <= 0 or thickness <= 0:
        raise ValueError("box width, height and thickness must be > 0")
    if 2 * thickness >= min(width, height):
        raise ValueError(
            f"box thickness {thickness} closes a {width} x {height} section; "
            "use type: rect for a solid one")
    if n_width < 1 or n_height < 1 or nr < 1:
        raise ValueError("box needs n_width, n_height and nr >= 1")
    ys, zs, areas = [], [], []
    t = thickness / nr
    for ir in range(nr):
        # horizontal walls: full width, one strip per layer
        z_off = height / 2.0 - thickness + t * (ir + 0.5)
        dy = width / n_width
        for i in range(n_width):
            y = -width / 2.0 + dy * (i + 0.5)
            for sign in (+1.0, -1.0):
                ys.append(y); zs.append(sign * z_off); areas.append(dy * t)
        # vertical walls: only the height between the horizontal walls
        y_off = width / 2.0 - thickness + t * (ir + 0.5)
        inner_h = height - 2.0 * thickness
        dz = inner_h / n_height
        for i in range(n_height):
            z = -inner_h / 2.0 + dz * (i + 0.5)
            for sign in (+1.0, -1.0):
                ys.append(sign * y_off); zs.append(z); areas.append(dz * t)
    return np.array(ys), np.array(zs), np.array(areas)


def circle_fibers(diameter: float, nr: int, nt: int):
    """Simple polar discretization for a solid circle (optional, not required)."""
    if diameter <= 0:
        raise ValueError("circle diameter must be > 0")
    r_max = diameter / 2
    # approximate equal area rings
    ys, zs, areas = [], [], []
    for ir in range(nr):
        r0 = r_max * ir / nr
        r1 = r_max * (ir + 1) / nr
        r_mid = 0.5 * (r0 + r1)
        # number of circumferential fibers scales with radius
        n_t = max(1, int(round(nt * r_mid / r_max))) if r_mid > 0 else 1
        for it in range(n_t):
            theta = 2 * np.pi * (it + 0.5) / n_t
            ys.append(r_mid * np.cos(theta))
            zs.append(r_mid * np.sin(theta))
            areas.append(np.pi * (r1**2 - r0**2) / n_t)
    return np.array(ys), np.array(zs), np.array(areas)


def _hole_weight(quad, holes, n=4):
    """The fraction of `quad` outside every hole, by an n x n sample of the
    cell. A raster cut by a circle keeps whole cells and
    drops whole cells, which costs a percent or two of area on a 38 cm
    void; weighting the cut cells by their sampled fraction costs nothing
    and puts the area back. The fibre stays at the full cell's centroid:
    a cut cell is at most one cell wide, and its second moment about the
    section axis is what the weight is for."""
    u = (np.arange(n) + 0.5) / n
    uu, vv = np.meshgrid(u, u, indexing="ij")
    c = np.asarray(quad, dtype=float)
    pts = ((1 - uu)[..., None] * (1 - vv)[..., None] * c[0] + uu[..., None] * (1 - vv)[..., None] * c[1]
           + uu[..., None] * vv[..., None] * c[2] + (1 - uu)[..., None] * vv[..., None] * c[3]).reshape(-1, 2)
    inside = np.zeros(len(pts), bool)
    for h in holes:
        y0, z0, r = (float(v) for v in h)
        inside |= (pts[:, 0] - y0) ** 2 + (pts[:, 1] - z0) ** 2 < r * r
    return 1.0 - inside.mean()


def patch_fibers(patches, owner=False):
    """A section built of quadrilateral patches: each patch
    is {corners: [[y, z] x 4] counter-clockwise, ny, nz}, subdivided
    bilinearly into ny x nz cells; the fibre sits at each cell's centroid
    with the cell's own area.

    `ny` AND `nz` COUNT THE PATCH'S OWN EDGES, not the section's axes: `ny`
    divides corner0 -> corner1 and `nz` divides corner1 -> corner2. On a patch
    whose first edge runs along z -- which is the natural way to write a deck
    slab, left to right along the width -- `ny` is therefore the count ACROSS
    and `nz` the count through the depth. Getting them the wrong way round
    changes neither the area nor the fibre count, only which direction is
    resolved finely, so it is invisible until a through-depth quantity
    (a temperature profile, a strain gradient) needs the layers.
    A box girder with inclined webs, haunches and
    tapered cantilevers is a handful of these -- OpenSees' `patch quad`.

    A patch may carry `holes: [[y, z, r], ...]`: a cell whose sample points
    all fall inside a hole is dropped and a cut cell keeps the fraction of
    its area outside -- a hollow-core slab, a void former,
    a duct too big to ignore.

    Returns (y, z, area, cells) with `cells` the (n, 4, 2) corner arrays the
    sweep paints, in the same order. With `owner=True` a fifth array comes
    back: which patch each fibre came from. Holes make that array the only
    way to recover it -- a cell wholly inside a void produces no fibre, so
    the counts per patch are not ny*nz and cannot be reconstructed after the
    fact. It is what lets a patch carry its own material.
    """
    ys, zs, areas, cells, own = [], [], [], [], []
    for k, p in enumerate(patches):
        c = np.asarray(p["corners"], dtype=float)
        holes = [tuple(float(v) for v in h) for h in (p.get("holes") or [])]
        if c.shape != (4, 2):
            raise ValueError(f"patch {k}: `corners` is four [y, z] points")
        ny, nz = int(p.get("ny", 4)), int(p.get("nz", 4))
        if ny < 1 or nz < 1:
            raise ValueError(f"patch {k}: ny and nz must be >= 1")
        # bilinear map from the unit square: corner order 0-1 along "y", 1-2 along "z"
        def at(u, v):
            return ((1 - u) * (1 - v) * c[0] + u * (1 - v) * c[1] + u * v * c[2] + (1 - u) * v * c[3])
        for i in range(ny):
            for j in range(nz):
                q = np.array([at(i / ny, j / nz), at((i + 1) / ny, j / nz),
                              at((i + 1) / ny, (j + 1) / nz), at(i / ny, (j + 1) / nz)])
                # shoelace area, orientation-independent
                x, y = q[:, 0], q[:, 1]
                a = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
                if a <= 0.0:
                    raise ValueError(f"patch {k}: a degenerate cell (are the corners in order?)")
                if holes:
                    w = _hole_weight(q, holes)
                    if w <= 1e-9:            # wholly inside a void: no fibre at all
                        continue
                    a *= w
                cen = q.mean(axis=0)
                ys.append(cen[0]); zs.append(cen[1]); areas.append(a); cells.append(q)
                own.append(k)
    out = (np.array(ys), np.array(zs), np.array(areas), np.asarray(cells))
    return out + (np.array(own, dtype=int),) if owner else out


def _in_quad(quad, y, z, tol):
    """Points (y, z) inside the quadrilateral `quad` (4 x 2, either winding),
    or within `tol` of one of its edges."""
    q = np.asarray(quad, dtype=float)
    inside = np.zeros(y.shape, dtype=bool)
    near = np.zeros(y.shape, dtype=bool)
    for i in range(4):
        (y0, z0), (y1, z1) = q[i], q[(i + 1) % 4]
        crosses = (z0 > z) != (z1 > z)
        with np.errstate(divide="ignore", invalid="ignore"):
            y_at = y0 + (z - z0) * (y1 - y0) / (z1 - z0)
        inside ^= crosses & (y < y_at)
        dy, dz = y1 - y0, z1 - z0
        L2 = dy * dy + dz * dz
        t = np.clip(((y - y0) * dy + (z - z0) * dz) / L2, 0.0, 1.0) if L2 > 0 else np.zeros_like(y)
        near |= np.hypot(y - (y0 + t * dy), z - (z0 + t * dz)) <= tol
    return inside | near


#: how far outside its patches' union steel may sit, as a fraction of the
#: patch section's larger dimension (see outline_contains)
PATCH_OUTLINE_TOLERANCE = 0.01


def outline_contains(shape, y, z, width=None, height=None, diameter=None, patches=None, tol=1e-9,
                     cells=None, exact=False):
    """Which points (y, z) lie inside a fibre section's OUTER outline:
    the rectangle of a `rect` or a `box`, the circle of a
    `tube`, the union of the patches of a `patches` section. A void -- a box's
    or a tube's cell, a patch's `holes` -- counts as inside: a duct IS a hole
    and an external tendon runs in a box's cell. None when the shape has no
    outline to test (missing dimensions). `tol` is relative to the section's
    size. It replaces the bounding rectangle, which let a tendon sit in the
    empty corner beside a T's web or outside a circle in its square."""
    y = np.atleast_1d(np.asarray(y, dtype=float)); z = np.atleast_1d(np.asarray(z, dtype=float))
    shape = shape or "rect"
    if shape in ("rect", "box") and width and height:
        t = tol * max(float(width), float(height))
        return (np.abs(y) <= float(width) / 2 + t) & (np.abs(z) <= float(height) / 2 + t)
    if shape in ("tube", "circle") and diameter:
        r = float(diameter) / 2
        return np.hypot(y, z) <= r * (1.0 + tol)
    if shape == "patches" and patches:
        # Patches APPROXIMATE an outline (a chamfer drawn square, a haunch
        # left out), and steel is placed against the real one: the Hangzhou
        # Bay girder's two corner bars sit 25 mm outside its patches, 0.16% of
        # its 15.8 m depth. A patch section is therefore held to within 1% of
        # its size -- far tighter than the empty corner beside a web that the
        # bounding rectangle accepted.
        #
        # A box or T girder (1.2) is drawn from its dimensions, so its outline
        # is exact (`exact`), and a box's `cells` count as inside, as a fibre
        # `box`'s does: an external tendon runs in one.
        quads = [np.asarray(p["corners"], dtype=float) for p in patches]
        pts = np.concatenate(quads)
        quads += [np.asarray(c, dtype=float) for c in (cells or [])]
        span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-300)
        out = np.zeros(y.shape, dtype=bool)
        for q in quads:
            out |= _in_quad(q, y, z, (tol if exact else max(tol, PATCH_OUTLINE_TOLERANCE)) * span)
        return out
    return None


def outline_name(shape, exact=False):
    """How the outline `outline_contains` tests is named in a message."""
    if exact and shape == "patches":
        return "the girder's walls, a box's cells counting as inside"
    return {"box": "the outer rectangle, its cell counting as inside",
            "tube": "the outer circle, its bore counting as inside",
            "circle": "the circle",
            "patches": "the union of its patches to within 1% of the section's size, holes counting as inside"}.get(shape or "rect", "the rectangle")
