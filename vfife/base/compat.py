"""What differs between the versions of the dependencies vfife supports.

The oldest versions in pyproject.toml are tested by the CI job that installs
.github/minimum-versions.txt; a function that exists in only some of them is
reached through here, once, rather than with a `hasattr` at each call."""
from __future__ import annotations

import numpy as np

#: the trapezoidal rule: `numpy.trapezoid` from NumPy 2.0, `numpy.trapz` before
trapezoid = getattr(np, "trapezoid", None) or np.trapz
