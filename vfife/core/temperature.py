"""A temperature change over a member's section.

TEMPERATURE IS NOT A FORCE. It is a free strain the section would take on if
nothing held it,

    eps_th(y, z) = alpha * T(y, z),

and what it produces depends entirely on what is in the way:

  * NOTHING IN THE WAY -- a determinate member, a linear T. The member simply
    changes shape: it elongates by alpha dT L, it bows to a curvature
    alpha dT/dy, and it carries no stress at all. A model that reports a force
    here has the sign of the thing backwards.
  * SOMETHING IN THE WAY -- a restrained end, a redundant support. The part of
    the free strain the structure refuses becomes force: N = EA alpha dT in a
    doubly restrained member, the secondary moments of a continuous beam under
    a gradient.
  * A NONLINEAR T THROUGH THE DEPTH, which is what a bridge code's design
    profile is. Now even a simply supported beam cannot take the shape,
    because plane sections must stay plane. The section keeps the linear part
    of the profile as elongation and curvature and turns the REST into
    self-equilibrated stresses: zero net force, zero net moment, and real
    stress -- tension at the soffit of a deck warmed on top. Only a fibre
    section can hold them.

WHY IT IS SUBTRACTED FROM THE DEFORMATION rather than added as a load. The
equivalent-nodal-force form of this (F = int B^T D eps_th dV) gives the right
displacements and the WRONG member actions: a free beam under uniform dT comes
out stretched, and the stretch read back through the section law reports
N = EA alpha dT on a member that carries nothing. The fix in that form is to
remember to subtract the thermal part again at stress recovery, everywhere,
forever. Taking it out of the deformation instead -- f = k (d - d_th) -- is the
same equilibrium and leaves nothing to remember: the actions a member reports
are the ones it carries.
"""
from typing import Optional

import numpy as np


def section_field(spec: dict, y, z):
    """T(y, z) in K over a section's own axes, from a `temperature` spec.

    `uniform` + `gradient` is the linear field; `profile` is [[y, T], ...]
    interpolated linearly and held constant beyond its ends, which is how a
    code's design profile is stated (it is defined over the depth it names and
    says nothing about what is outside it).
    """
    y = np.asarray(y, float); z = np.asarray(z, float)
    if spec.get("profile") is not None:
        table = np.asarray(spec["profile"], float)
        along = z if spec.get("axis", "y") == "z" else y
        return np.interp(along, table[:, 0], table[:, 1])
    g = spec.get("gradient") or [0.0, 0.0]
    g = [float(g), 0.0] if not hasattr(g, "__len__") else [float(g[0]), float(g[1])]
    return float(spec.get("uniform", 0.0)) + g[0]*y + g[1]*z


def linear_field(spec: dict):
    """(dT, dT/dy, dT/dz) for a section with no fibres to integrate over.

    A `profile` has no answer here: reducing it to a mean and a slope needs the
    section's area distribution, which a `general` beam section (area and
    second moments, no shape) does not have. Inventing a reduction would be
    inventing the section, so the caller refuses instead.
    """
    if spec.get("profile") is not None:
        return None
    g = spec.get("gradient") or [0.0, 0.0]
    g = [float(g), 0.0] if not hasattr(g, "__len__") else [float(g[0]), float(g[1])]
    return float(spec.get("uniform", 0.0)), g[0], g[1]


def local_deformation(alpha: float, spec: dict, lengths) -> Optional[np.ndarray]:
    """The local deformation vectors (n, 12) a set of members would take on if
    the temperature field were free to act -- what `f = k (d - d_th)` removes.

    The co-rotational local deformation of a frame member is only ever
    [0,0,0, tx1,ty1,tz1, dL, 0,0, tx2,ty2,tz2] (the rigid motion is gone), so
    the thermal one is the axial elongation and the end rotations of a member
    bent to a constant curvature. With the transverse ends held at zero, the
    Hermite curvature row is constant only when the two end rotations are
    equal and opposite, and then kappa = -2 theta1 / L -- so theta1 = -kappa
    L/2 and theta2 = +kappa L/2, in both bending planes (the rotation entries
    of Bky and Bkz are the same expressions).

    The sign of each curvature comes from the section's own strain convention,
    eps = eps0 - y kappa_z + z kappa_y: matching alpha (dT + gy y + gz z) term
    by term gives kappa_z = -alpha gy and kappa_y = +alpha gz.
    """
    linear = linear_field(spec)
    if linear is None:
        return None
    dT, gy, gz = linear
    L = np.asarray(lengths, float)
    out = np.zeros((len(L), 12))
    out[:, 6] = alpha * dT * L                      # axial elongation
    kz = -alpha * gy
    ky = +alpha * gz
    out[:, 5], out[:, 11] = -kz * L / 2, kz * L / 2      # bending about local z
    out[:, 4], out[:, 10] = -ky * L / 2, ky * L / 2      # and about local y
    return out
