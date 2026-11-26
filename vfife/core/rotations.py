"""Finite rotations moved to `vfife.base.rotations` (1.22); this name is that module."""
import sys

from ..base import rotations as _impl

sys.modules[__name__] = _impl
