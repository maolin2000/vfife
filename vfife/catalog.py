"""Unified, machine-readable element and material catalogue.

This module is deliberately independent of CLI presentation.  The CLI, Python
API and a future GUI consume the same immutable descriptors, so adding a model
does not require another hard-coded list in every front end.
"""
from __future__ import annotations

from .elements import element_definition, element_definitions
from .materials import definition as material_definition
from .materials import definitions as material_definitions


def library() -> dict:
    """Return serialisable definitions for every registered model."""
    return {
        "schema_version": 1,
        "elements": {k: v.as_dict() for k, v in element_definitions().items()},
        "materials": {k: v.as_dict() for k, v in material_definitions().items()},
    }


def compatibility(element: str, material: str, *, backend: str | None = None,
                  mode: str | None = None) -> tuple[bool, tuple[str, ...]]:
    """Check a requested combination without constructing a solver/backend."""
    elem = element_definition(element)
    mat = material_definition(material)
    reasons: list[str] = []
    if material not in elem.compatible_materials:
        reasons.append(f"{element} does not support material {material}")
    if not set(elem.material_kinematics).intersection(mat.kinematics):
        reasons.append(
            f"kinematics do not match: element {elem.material_kinematics}, "
            f"material {mat.kinematics}"
        )
    if backend is not None:
        if backend not in elem.backends:
            reasons.append(f"{element} does not support backend {backend}")
        if backend not in mat.backends:
            reasons.append(f"{material} does not support backend {backend}")
        if element == "beam2_fiber" and backend == "numba":
            # THE FIBRE KERNEL INLINES A UNIAXIAL SCALAR TWIN, and a law
            # without one is refused when the solver is built. Saying
            # SUPPORTED here (the material's own `backends` speak of the solid
            # kernel) let preflight pass a run that then stopped with "no
            # scalar twin" (found in review)
            from .materials.base import _REGISTRY
            cls = _REGISTRY.get(material)
            # said only when it is what decides: a law the element or the
            # backend already refuses needs no second reason
            if not reasons and cls is not None and not hasattr(cls, "numba_uniaxial_fn"):
                reasons.append(f"{material} has no scalar uniaxial twin for the beam2_fiber numba "
                               "kernel; use backend numpy")
    if mode is not None:
        if mode not in elem.analysis_modes:
            reasons.append(f"{element} does not support analysis mode {mode}")
        if mode not in mat.analysis_modes:
            reasons.append(f"{material} does not support analysis mode {mode}")
    return not reasons, tuple(reasons)
