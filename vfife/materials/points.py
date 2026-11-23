"""Material-point lifecycle shared by continuum and uniaxial elements."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .base import definition


@dataclass(frozen=True)
class MaterialPointContract:
    material: str
    kinematics: str
    strain_components: tuple[str, ...]
    stress_components: tuple[str, ...]
    state_width: int
    state_fields: tuple[str, ...]
    backend: str
    characteristic_length: str
    wave_speed: float
    signature: str

    def as_dict(self):
        return asdict(self)

    def stable_time_step(self, length) -> np.ndarray:
        return np.asarray(length, dtype=float) / self.wave_speed


SOLID_STRAIN = ("exx", "eyy", "ezz", "gxy", "gyz", "gxz")
SOLID_STRESS = ("sxx", "syy", "szz", "sxy", "syz", "sxz")


def _validate_state_layout(material) -> None:
    occupied = np.zeros(material.n_state, dtype=bool)
    for name, part in material.state_names.items():
        if not isinstance(part, slice) or part.step not in (None, 1):
            raise ValueError(f"material {material.name} state field {name!r} needs a unit slice")
        start = 0 if part.start is None else part.start
        stop = material.n_state if part.stop is None else part.stop
        if start < 0 or stop < start or stop > material.n_state:
            raise ValueError(f"material {material.name} state field {name!r} is out of bounds")
        if occupied[start:stop].any():
            raise ValueError(f"material {material.name} state field {name!r} overlaps another field")
        occupied[start:stop] = True
    if material.n_state and not occupied.all():
        missing = np.flatnonzero(~occupied).tolist()
        raise ValueError(f"material {material.name} has unnamed state columns {missing}")


def initialise_points(material, n_points: int, *, kinematics: str,
                      backend: str, char_length=None, dtype=np.float64):
    """Bind geometry and create a validated, checkpoint-stable state array."""
    meta = definition(material.name)
    if kinematics not in meta.kinematics:
        raise ValueError(f"material {material.name} does not support {kinematics} kinematics")
    if backend not in meta.backends:
        raise ValueError(f"material {material.name} does not support backend {backend}")
    if n_points < 1:
        raise ValueError("a material-point set needs at least one point")
    _validate_state_layout(material)
    if char_length is None:
        lengths = np.ones(n_points)
        if meta.regularised:
            raise ValueError(f"regularised material {material.name} requires characteristic length")
        length_kind = "not_required"
    else:
        lengths = np.broadcast_to(np.asarray(char_length, dtype=float), (n_points,)).copy()
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("material-point characteristic lengths must be finite and positive")
        length_kind = "per_point"
        material.bind_points(lengths)
    state = np.ascontiguousarray(material.init_state(n_points, dtype=dtype))
    if state.shape != (n_points, material.n_state):
        raise ValueError(f"material {material.name} returned state shape {state.shape}; "
                         f"expected {(n_points, material.n_state)}")
    wave_speed = float(getattr(material, "wave_speed", 0.0))
    if not np.isfinite(wave_speed) or wave_speed <= 0:
        raise ValueError(f"material {material.name} must provide a positive wave_speed")
    components = (("axial_strain",), ("axial_stress",)) if kinematics == "uniaxial" else (SOLID_STRAIN, SOLID_STRESS)
    identity = {"material": material.name, "kinematics": kinematics,
                "state_width": material.n_state,
                "state_fields": [(k, v.start, v.stop) for k, v in material.state_names.items()]}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    contract = MaterialPointContract(
        material.name, kinematics, tuple(components[0]), tuple(components[1]),
        material.n_state, tuple(material.state_names), backend, length_kind,
        wave_speed, signature,
    )
    return state, contract


def validate_restored_state(state, contract: MaterialPointContract, n_points: int):
    state = np.ascontiguousarray(state)
    expected = (n_points, contract.state_width)
    if state.shape != expected:
        raise ValueError(f"checkpoint material state shape {state.shape} does not match {expected}")
    if not np.isfinite(state).all():
        raise ValueError("checkpoint material state contains non-finite values")
    return state
