"""The Voigt notation moved to `vfife.base.voigt` (1.22); this name is that module."""
import sys

from .base import voigt as _impl

sys.modules[__name__] = _impl
