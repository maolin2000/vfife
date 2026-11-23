"""Machine-readable admission gates for element and material libraries."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .catalog import library


LEVELS = ("registered", "verified", "benchmarked", "production")


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str
    reference: str
    backends: list[Literal["numpy", "numba", "warp"]] = Field(default_factory=list)
    note: str = ""


class Qualification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    kind: Literal["element", "material"]
    name: str
    target_level: Literal["registered", "verified", "benchmarked", "production"]
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _known_model(self):
        group = self.kind + "s"
        if self.name not in library()[group]:
            raise ValueError(f"unknown {self.kind} {self.name!r}")
        return self


def required_gates(record: Qualification, level: str | None = None) -> set[str]:
    """Required evidence categories, including conditional model properties."""
    target = level or record.target_level
    if target not in LEVELS:
        raise ValueError(f"unknown qualification level {target!r}")
    if target == "registered":
        return set()
    item = library()[record.kind + "s"][record.name]
    if record.kind == "element":
        gates = {"rigid_body", "fundamental_response", "equilibrium"}
        if target in {"benchmarked", "production"}:
            gates |= {"mesh_convergence", "dynamic", "restart"}
        if item["family"] in {"line", "mixed_dimensional"}:
            gates.add("mixed_assembly")
    else:
        gates = {"monotonic", "step_convergence", "energy"}
        if item["nonlinear"]:
            gates.add("unload_reverse")
        if "solid_3d" in item["kinematics"]:
            gates.add("multiaxial")
        if item["regularised"]:
            gates.add("mesh_objectivity")
        if target in {"benchmarked", "production"}:
            gates.add("restart")
    if target in {"benchmarked", "production"}:
        gates.add("external_reference")
    if target == "production":
        gates |= {"backend_parity", "failure_path"}
    return gates



def _names_a_real_test(path: Path, node: str) -> bool:
    """Does `path` actually define the test this node id names?

    Existence only -- this does NOT run the test, so a PASS still means "the
    evidence points at something real", not "the evidence passes". Running the
    suite from here would make `qualify check` a test runner, which is a
    different tool; `pytest <node id>` is the one that answers that.
    """
    import re

    # strip a parametrisation suffix and any class path: Cls::test_x[case]
    name = node.split("::")[-1].split("[")[0].strip()
    if not name:
        return False
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(name)}\s*\(",
                     source, re.MULTILINE) is not None


def assess(record: Qualification, *, root: str | Path | None = None) -> dict:
    supplied = {item.category for item in record.evidence}
    required = required_gates(record)
    missing = sorted(required - supplied)
    invalid = []
    if root is not None:
        base = Path(root)
        for item in record.evidence:
            # pytest node ids use path::test; references may also be URLs or
            # named verification contracts, which are intentionally external.
            path = item.reference.split("::", 1)[0]
            if path.startswith(("http://", "https://")):
                continue
            if not (base / path).exists():
                invalid.append(item.reference)
            elif "::" in item.reference and not _names_a_real_test(
                    base / path, item.reference.split("::", 1)[1]):
                # The file existing was the whole check, so a manifest could
                # cite a test that had been renamed or deleted and still be
                # reported PASS. Evidence that names nothing is not evidence.
                invalid.append(item.reference)
    backend_missing = []
    if record.target_level == "production":
        declared = set(library()[record.kind + "s"][record.name]["backends"])
        covered = {b for item in record.evidence if item.category == "backend_parity"
                   for b in item.backends}
        backend_missing = sorted(declared - covered)
    passed = not missing and not invalid and not backend_missing
    return {
        "schema_version": 1, "kind": record.kind, "name": record.name,
        "target_level": record.target_level,
        "status": "PASS" if passed else "INCOMPLETE",
        "required": sorted(required), "supplied": sorted(supplied),
        "missing": missing, "invalid_references": invalid,
        "missing_backend_coverage": backend_missing,
    }


def load(path: str | Path) -> Qualification:
    from .io.schema import safe_load_named
    return Qualification.model_validate(safe_load_named(Path(path).read_text(encoding="utf-8"), path))


def template(kind: Literal["element", "material"], name: str,
             target_level: str = "verified") -> dict:
    record = Qualification(version=1, kind=kind, name=name,
                           target_level=target_level, evidence=[])
    return {"version": 1, "kind": kind, "name": name,
            "target_level": target_level,
            "evidence": [{"category": gate, "reference": "TODO",
                          "backends": [], "note": ""}
                         for gate in sorted(required_gates(record))]}
