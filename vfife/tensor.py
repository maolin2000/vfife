"""Tensor helpers moved to `vfife.base.tensor` (1.22); this name is that module."""
import sys

from .base import tensor as _impl

sys.modules[__name__] = _impl
