"""Material protocol and registry.

The single most important design decision in this package lives here:
state variables are a FLAT float array of shape (n_points, n_state), not a
list of dicts / structs.  That is what makes the following possible:

  * batch vectorisation over all integration points at once
  * a bit-for-bit block copy to GPU memory (a Warp / CuPy array is just this
    same buffer)
  * checkpoint / restart as a single HDF5 dataset write
  * one indexing scheme shared by the NumPy, Numba and (later) GPU kernels

`state_names` maps slices of that array back to human names.  It is used only
at initialisation and at output time -- never inside the time loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
from typing import Callable, Dict, Optional

import numpy as np

_REGISTRY: Dict[str, Callable[..., "Material"]] = {}
_DEFINITIONS: Dict[str, "MaterialDef"] = {}


@dataclass(frozen=True)
class MaterialDef:
    """Machine-readable engineering contract for one material model."""

    name: str
    kinematics: tuple[str, ...]
    backends: tuple[str, ...]
    analysis_modes: tuple[str, ...]
    state_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    parameters: tuple[str, ...]
    nonlinear: bool
    regularised: bool
    verification: str
    caveat: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def register(name: str, *, kinematics=("solid_3d",),
             backends=("numpy", "numba", "warp"),
             analysis_modes=("dynamic", "relaxation"),
             output_fields=(), nonlinear=True, regularised=False,
             verification="unit", caveat=""):
    """Decorator: register a material class under `name` for use in YAML."""

    def _wrap(cls):
        cls.name = name
        _REGISTRY[name] = cls
        signature = inspect.signature(cls.__init__)
        parameters = tuple(
            p.name for p in signature.parameters.values()
            if p.name != "self" and p.kind not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        )
        state_fields = tuple(cls.state_names)
        _DEFINITIONS[name] = MaterialDef(
            name=name, kinematics=tuple(kinematics), backends=tuple(backends),
            analysis_modes=tuple(analysis_modes), state_fields=state_fields,
            output_fields=tuple(output_fields) or state_fields,
            parameters=parameters, nonlinear=bool(nonlinear),
            regularised=bool(regularised), verification=verification,
            caveat=caveat,
        )
        return cls

    return _wrap


#: Laws that were removed, and what to use instead. A retired name gets its
#: own message rather than "unknown material", because the two questions a
#: reader has are different: an unknown name is a typo, a retired one is a
#: model decision someone else made and has to be told to.
RETIRED = {
    "rebar_gmp": (
        "removed in v0.1: it was the same Giuffre-Menegotto-Pinto law as "
        "`steel02` then (0.42% of the peak stress over 18 parameter sets; "
        "since v1.0 `steel02` is Filippou et al.'s form, whose envelope stays "
        "under its hardening asymptote), runs 30x faster (147 ns a point against 4.5 us) "
        "and is the one with a compiled scalar twin -- without which a fibre "
        "section drops off the fused kernel. Change `type: rebar_gmp` to "
        "`type: steel02` and rename `yield_strength` to `Fy` and `R` to `R0`; "
        "b keeps its name. See docs/steel02.md."),
}


def accepted_parameters(name: str) -> set:
    """Every keyword the law's constructor (and those it passes on to) reads."""
    import inspect

    names = {"thermal_expansion", "creep", "shrinkage", "ageing", "relaxation"}
    for cls in _REGISTRY[name].__mro__:
        init = cls.__dict__.get("__init__")
        if init is None or cls is object:
            continue
        names |= {p.name for p in inspect.signature(init).parameters.values()
                  if p.name != "self" and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                                         inspect.Parameter.VAR_KEYWORD)}
    for aliases in ALIASES.get(name, {}).values():
        names |= set(aliases)
    return names


def _refuse_unknown_parameters(name, kwargs):
    """Every law took `**_ignored`, so `s1x` for `s1` or `R00` for `R0` was
    accepted and the default used in its place, without a word (found in
    review)."""
    import difflib

    known = accepted_parameters(name)
    unknown = sorted(k for k in kwargs if k not in known)
    if unknown:
        from ..core.diagnostics import UserError
        lines = []
        for k in unknown:
            near = difflib.get_close_matches(k, known, n=1, cutoff=0.6)
            lines.append(repr(k) + (f" (did you mean {near[0]!r}?)" if near else ""))
        raise UserError("UNKNOWN_MATERIAL_PARAMETER",
                        f"material {name!r} does not read " + ", ".join(lines),
                        f"{name} reads: " + ", ".join(sorted(known)))
    _refuse_alias_conflicts(name, kwargs)


#: Other names a law reads for one of its parameters: {law: {canonical: (aliases...)}}.
#: The law takes the canonical name when both are given, so an alias written
#: beside it was dropped without a word -- `fc: 5e7` next to `fpc: 4e7` ran at
#: 40 MPa, and a study over `fc` gave one answer for every value.
ALIASES = {
    "concrete01": {"fpc": ("fc",), "epsc0": ("eps_c0",), "fpcu": ("fcu",), "epscu": ("eps_cu",)},
    "concrete02": {"fpc": ("fc",), "epsc0": ("eps_c0",), "fcu": ("fpcu", "fcu2"),
                   "epscu": ("eps_cu",)},
    "steel02": {"E": ("E0",), "Fy": ("yield_strength", "fy")},
    # `fy` is the yield strength in every law that has one (steel_cyclic's own name)
    "rebar": {"yield_strength": ("fy",)},
    "j2": {"yield_strength": ("fy",)},
    "chaboche": {"yield_strength": ("fy",)},
}


def _fold_aliases(name, kwargs):
    """An alias the law's constructor does not read itself is passed on
    under the canonical name (a conflict between the two is refused before)."""
    import inspect
    own = set(inspect.signature(_REGISTRY[name].__init__).parameters)
    for canonical, aliases in ALIASES.get(name, {}).items():
        for alias in aliases:
            if alias in kwargs and alias not in own:
                value = kwargs.pop(alias)
                if kwargs.get(canonical) is None:
                    kwargs[canonical] = value


def _refuse_alias_conflicts(name, kwargs):
    """A parameter written under two of its names with different values is
    refused by name, as a section's old and new spellings are."""
    from ..core.diagnostics import UserError
    for canonical, aliases in ALIASES.get(name, {}).items():
        given = [(k, kwargs[k]) for k in (canonical,) + aliases if kwargs.get(k) is not None]
        values = {repr(float(v)) if isinstance(v, (int, float)) else repr(v) for _, v in given}
        if len(values) > 1:
            names = " and ".join(f"`{k}` = {v!r}" for k, v in given)
            raise UserError("MATERIAL_ALIAS_CONFLICT",
                            f"material {name!r}: {names} name the same parameter and differ",
                            f"keep `{canonical}` (the law's own name) and remove the other")


def _check_slow(name, slow):
    """The time-dependent specs' own keys and ranges, when the material is
    made: `RH` for `rh`, `S` for `s` and `rho_1000: 3.1` all ran on defaults
    or on nonsense."""
    from ..core import creep, relaxation
    checks = {"creep": creep.validate_creep, "shrinkage": creep.validate_shrinkage,
              "ageing": creep.validate_ageing, "relaxation": relaxation.validate}
    for key, check in checks.items():
        spec = slow.get(key)
        if spec is None:
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"material {name!r}: `{key}` is a mapping of its model's "
                             f"parameters, not {spec!r}")
        check(dict(spec), f"material {name!r} `{key}`")
    creep.check_reference(slow.get("creep"), slow.get("ageing"), f"material {name!r}")


def create(name: str, **kwargs) -> "Material":
    if name not in _REGISTRY:
        from ..core.diagnostics import UserError
        import difflib

        if name in RETIRED:
            raise UserError("RETIRED_MATERIAL", f"material {name!r} no longer exists",
                            RETIRED[name])
        near = difflib.get_close_matches(name, _REGISTRY, n=1, cutoff=0.6)
        raise UserError(
            "UNKNOWN_MATERIAL",
            f"unknown material {name!r}",
            (f"did you mean {near[0]!r}? " if near else "")
            + "available: " + ", ".join(sorted(_REGISTRY)),
        )
    thermal = kwargs.pop("thermal_expansion", None)
    slow = {k: kwargs.pop(k, None)
            for k in ("creep", "shrinkage", "ageing", "relaxation")}
    _refuse_unknown_parameters(name, kwargs)
    _fold_aliases(name, kwargs)
    _check_slow(name, slow)
    material = _REGISTRY[name](**kwargs)
    if thermal is not None:
        material.thermal_expansion = float(thermal)
    for k, v in slow.items():
        if v is not None:
            setattr(material, k, dict(v))
    return material


def available() -> list[str]:
    return sorted(_REGISTRY)


def definition(name: str) -> MaterialDef:
    if name not in _DEFINITIONS:
        raise KeyError(f"unknown material {name!r}; available: {sorted(_DEFINITIONS)}")
    return _DEFINITIONS[name]


def definitions() -> dict[str, MaterialDef]:
    return {name: _DEFINITIONS[name] for name in sorted(_DEFINITIONS)}


class Material:
    """Base class. Subclasses must set n_state / state_names and implement update()."""

    name: str = "base"
    n_state: int = 0
    #: name -> slice into the per-point state row
    state_names: Dict[str, slice] = {}

    E: float
    nu: float
    rho: float

    #: Coefficient of thermal expansion, 1/K. Zero unless the case says
    #: otherwise, so every model without a temperature load is unchanged and a
    #: model WITH one but without a coefficient does nothing rather than
    #: something made up. It is set in `create()` rather than
    #: in each law's __init__ because it is a property of the substance, not
    #: of the constitutive model: concrete is ~1.0e-5 whether it is elastic,
    #: concrete01 or concrete02.
    #:
    #: NOT named `alpha`, which was the obvious choice and is already taken:
    #: `bond_ceb`'s CEB curve exponent is an `alpha`, and popping the key here
    #: stopped its validator ever seeing it (tests/test_bond.py caught it).
    #: One key, two meanings, decided by which material you happened to write
    #: -- so it is spelled out instead.
    thermal_expansion: float = 0.0

    #: Time-dependent concrete, all optional and all specs
    #: rather than numbers because each names a MODEL: `creep` and `shrinkage`
    #: are read by `core.creep`, `ageing` is `{s, of}` for Ec(t)/Ec(28). They
    #: sit on the material for the same reason `thermal_expansion` does --
    #: they are properties of the substance, not of the load.
    creep: Optional[dict] = None
    shrinkage: Optional[dict] = None
    ageing: Optional[dict] = None

    #: Relaxation of PRESTRESSING steel, read by
    #: `core.relaxation`. The same kind of thing one level along: creep and
    #: shrinkage belong to the concrete and arrive as a strain, this belongs
    #: to the steel and arrives as a loss of the tendon's own prestrain. It
    #: does nothing at all on a fibre that carries no prestrain, which is what
    #: makes it safe to put on a material a section uses for both tendons and
    #: ordinary bars.
    relaxation: Optional[dict] = None

    def init_state(self, n_points: int, dtype=np.float64) -> np.ndarray:
        return np.zeros((n_points, self.n_state), dtype=dtype)

    def damage_measure(self, state: np.ndarray):
        """Scalar degradation per integration point, for the erosion criterion.

        The kinematic half of the criterion (max principal strain, volume
        collapse) is the solver's business, not the material's.

        Return None if the model has no notion of failure, in which case
        configuring erosion on it is an error rather than a no-op -- silently
        never eroding is exactly the kind of thing that looks like a converged
        answer.
        """
        return None

    #: the damage classes `damage_class` may return
    DAMAGE_INTACT, DAMAGE_CRACKED, DAMAGE_CRUSHED, DAMAGE_YIELDED, DAMAGE_FAILED = 0, 1, 2, 3, 4

    def damage_class(self, state: np.ndarray):
        """What state each material point is in, as one small integer per
        point, for a picture to be painted with:

            0  intact (elastic so far)
            1  cracked   -- concrete that has been past its tensile strength
            2  crushed   -- concrete past its peak compressive strain
            3  yielded   -- steel with plastic strain
            4  failed    -- what `failure` latches: ruptured, exhausted

        Coarser than `failure` and not monotone (a cracked fibre reads
        crushed once it is driven past the peak the other way); it is what a
        crack map or a hinge picture wants, not what a fracture criterion
        wants. None from a law that has no such notion (elastic).
        """
        return None

    def failure(self, state: np.ndarray):
        """How dead each material point is: 0 intact, 1 gone.

        THE ONE THING A MATERIAL HAS TO SAY ABOUT FRACTURE, and the reason a
        fracture criterion does not have to be written once per material times
        once per element. A material knows when its own point is finished; an
        element knows how its points add up; the topology update knows what to
        do with a finished element. Three interfaces, M + E implementations,
        not M x E.

        Two properties this must have, and `damage_measure` deliberately does
        not:

        * MONOTONE. It may never fall. `cdp`'s `d` legitimately falls when a
          crack closes under compression (`wc`), which is why the erosion
          criterion in `core/solver.py` has to pair it with a strain that only
          ever grows. A failure fraction is a latch: what is dead stays dead.
        * PER POINT, not per element. Aggregation is the element's business,
          because only the element knows whether its points are eight Gauss
          points of a hexahedron or twenty fibres of a section.

        Return None if the law has no notion of failure. That is not the same
        as returning zeros: `None` means "asking me about fracture is a
        modelling error", and the solver says so rather than never fracturing.
        """
        return None

    def bind_points(self, char_length: np.ndarray) -> None:
        """Optional hook, called once by the solver with the per-integration-point
        element characteristic length V/A_max.

        Only regularised models need it (crack-band scaling).  It stays out of
        `update()` deliberately: a per-point array set once at setup is one more
        buffer on the GPU, whereas an extra argument would change the kernel
        signature for every material.
        """
        return None

    def update(
        self, strain: np.ndarray, state: np.ndarray, dt: float
    ) -> np.ndarray:
        """Advance the material one step.

        Parameters
        ----------
        strain : (n, 6) engineering-Voigt total strain at the end of the step. Read only.
        state  : (n, n_state) state variables, updated IN PLACE.
        dt     : time increment (needed for rate-dependent / viscous terms).

        Returns
        -------
        stress : (n, 6) Cauchy stress, engineering-Voigt convention.
        """
        raise NotImplementedError

    # -- scalar outputs derived from state, for postprocessing ---------------

    def output_fields(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        return {}

    def describe(self) -> str:
        """Extra lines for the run summary. Override to report derived parameters."""
        return ""
