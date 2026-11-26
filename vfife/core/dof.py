"""The degrees of freedom moved to `vfife.base.dof` (1.22); this name is that module."""
import sys

from ..base import dof as _impl

sys.modules[__name__] = _impl
