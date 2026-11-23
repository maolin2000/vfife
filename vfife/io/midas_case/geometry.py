"""Midas' local axes and the converter's small geometric helpers."""
from __future__ import annotations

import math




def _local_y(a, b, beta_deg):
    """The local y axis of a Midas line element from a to b: local z is in
    the plane of the member and global Z, upward (a vertical member: local z
    along global X), turned about the member by the beta angle."""
    x = [b[i] - a[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in x)) or 1.0
    x = [c / n for c in x]
    if abs(x[0]) < 1e-9 and abs(x[1]) < 1e-9:
        z = [1.0, 0.0, 0.0]
    else:
        dot = x[2]
        z = [-dot * x[0], -dot * x[1], 1.0 - dot * x[2]]
        nz = math.sqrt(sum(c * c for c in z))
        z = [c / nz for c in z]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    if beta_deg:
        c, s = math.cos(math.radians(beta_deg)), math.sin(math.radians(beta_deg))
        y = [c * y[i] + s * z[i] for i in range(3)]
    return y


def _up_of(a, b, beta_deg):
    """The local z axis (the section's depth, upward) of a Midas line element from a to b."""
    x = [b[i] - a[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in x)) or 1.0
    x = [c / n for c in x]
    y = _local_y(a, b, beta_deg)
    return [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]


def _to_global(stiff, a, b, beta):
    """A link's six local stiffnesses in global axes: six numbers when the
    result is diagonal (the link along a global axis), else the 6x6 matrix
    as a tuple of rows."""
    x = [b[i] - a[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in x))
    x = [c / n for c in x]
    y = _local_y(a, b, beta)
    z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]
    E = (x, y, z)                                           # rows: the local axes
    blocks = []
    for k3 in (stiff[:3], stiff[3:]):
        blocks.append([[sum(E[m][i] * k3[m] * E[m][j] for m in range(3)) for j in range(3)] for i in range(3)])
    top = max(abs(v) for blk in blocks for row in blk for v in row) or 1.0
    if all(abs(blk[i][j]) <= 1e-12 * top for blk in blocks for i in range(3) for j in range(3) if i != j):
        return tuple(blk[i][i] for blk in blocks for i in range(3))
    K = [[0.0] * 6 for _ in range(6)]
    for q, blk in enumerate(blocks):
        for i in range(3):
            for j in range(3):
                K[3 * q + i][3 * q + j] = round(blk[i][j], 6)
    return tuple(tuple(row) for row in K)


def _plain(stiff):
    return [list(r) for r in stiff] if isinstance(stiff[0], tuple) else list(stiff)


def _key(v):
    return tuple(round(c, 3) + 0.0 for c in v)


def _free_axis(a, b):
    """The orientation of a member whose section has no axes of its own
    (a pipe, a round bar, a rigid arm): global Z, or global X for a member
    that is nearer the vertical than 45 degrees. Members of one birth and
    one section are split by it, so no group's vector is along a member."""
    x = [b[i] - a[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in x)) or 1.0
    return (1.0, 0.0, 0.0) if abs(x[2]) / n > 0.7071 else (0.0, 0.0, 1.0)
