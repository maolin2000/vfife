import json
from types import SimpleNamespace

from vfife.catalog import compatibility, library
from vfife.elements import element_definition, element_definitions
from vfife.materials import available as available_materials, definition
from vfife.api import prepare_case
from vfife import cli
from vfife.io import schema


def test_every_existing_element_has_an_engineering_contract():
    definitions = element_definitions()
    assert set(definitions) == {
        "hexa8", "hexa20", "tet4", "tet10", "wedge6", "wedge15", "pyramid5",
        "truss2", "beam2", "beam2_timoshenko", "beam2_fiber", "shell3", "shell4", "shell_line", "connector2", "mixed_line", "coupled",
    }
    for item in definitions.values():
        assert item.dofs and item.backends and item.analysis_modes
        assert item.compatible_materials and item.output_fields
        assert item.geometric_nonlinearity == "co_rotational_vfife"


def test_every_registered_material_has_an_engineering_contract():
    assert set(available_materials()) == {
        "linear_elastic", "orthotropic", "viscoelastic", "j2", "chaboche", "cdp", "rebar", "steel02", "piecewise_uniaxial", "concrete01", "concrete02",
        # a traction-separation law, not a stress-strain one: its "strain" is a
        # slip in metres and its "stress" is a bond stress. Registered under
        # kinematics ("bond",) so no element can pick it up by accident.
        "bond_ceb",
        # the same law with Eligehausen-type degradation and a pinched
        # reversal; separate, because bond_ceb must keep reproducing
        "bond_cyclic",
        # Cai et al. 2023: cyclic steel with a degradation curve and an
        # ultra-low-cycle-fatigue fracture strain
        "steel_cyclic",
    }
    for name in available_materials():
        item = definition(name)
        assert item.parameters and item.backends and item.analysis_modes
        assert item.kinematics and item.output_fields and item.verification
    assert definition("linear_elastic").nonlinear is False
    assert definition("cdp").regularised is True
    assert definition("rebar").kinematics == ("uniaxial",)


def test_catalogue_is_serialisable_and_checks_combinations():
    data = library()
    assert data["schema_version"] == 1
    json.dumps(data)

    assert compatibility("hexa8", "j2", backend="warp", mode="dynamic")[0]
    ok, reasons = compatibility("beam2", "j2", backend="warp")
    assert not ok
    assert any("material j2" in reason for reason in reasons)
    assert any("backend warp" in reason for reason in reasons)


def test_element_contract_exposes_real_formulation_limits():
    assert element_definition("hexa8").formulations == ("full", "reduced", "b-bar")
    tet = element_definition("tet4")
    assert tet.formulations == ("full",)
    assert "locks" in tet.caveat


def test_preflight_uses_the_catalogue_for_invalid_combinations():
    raw = {
        "name": "bad_beam_material",
        "mesh": {"type": "gmsh", "path": "benchmarks/assets/beam_cantilever_v22.msh"},
        "element": {"type": "beam2", "formulation": "full", "area": 1e-3,
                    "iy": 1e-6, "iz": 1e-6, "j": 1e-6},
        "material": {"type": "j2", "E": 2e11, "nu": 0.3, "rho": 7800,
                     "yield_strength": 2.5e8},
        "constraints": [{"region": "fixed", "dofs": [1, 1, 1, 1, 1, 1]}],
        "solver": {"mode": "relaxation", "n_steps": 10},
    }
    # Build is intentionally allowed to reach preflight: capability rejection
    # belongs to one central registry, not to a front-end-specific schema list.
    case = schema.Case.model_validate(raw)
    prepared = prepare_case(case)
    issues = prepared.preflight["issues"]
    assert any(i["code"] == "CAPABILITY_MISMATCH" for i in issues)


def test_cli_list_and_describe_are_catalogue_driven(capsys):
    assert cli.main(["list", "elements", "--format", "json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert "beam2" in listed["elements"]
    assert cli.main(["describe", "cdp", "--format", "json"]) == 0
    described = json.loads(capsys.readouterr().out)
    assert described["kind"] == "material"
    assert described["regularised"] is True
    assert "compression" in described["parameters"]


def test_cli_capabilities_explains_supported_and_rejected_combinations(capsys):
    assert cli.main(["capabilities", "--element", "beam2", "--backend", "warp",
                     "--show-unsupported", "--format", "json"]) == 0
    result = json.loads(capsys.readouterr().out)
    rows = {row["material"]: row for row in result["combinations"]}
    assert not rows["linear_elastic"]["supported"]
    assert any("backend warp" in text for text in rows["linear_elastic"]["reasons"])
    assert any("material j2" in text for text in rows["j2"]["reasons"])

    assert cli.main(["capabilities", "--element", "hexa8", "--material",
                     "j2", "--backend", "warp"]) == 0
    assert capsys.readouterr().out.startswith("SUPPORTED")
