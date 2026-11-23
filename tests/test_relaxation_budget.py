"""A relaxation run's iteration budget may be left out when every step has one.

The schema has said so since the staged steps were given their own budgets
(`load_steps * max_steps_per_load`), and the README repeats it. It was true of
the continuum solver, which added the steps up for itself, and false of the
five line and shell solvers, which read `solver.n_steps` and raised
`TypeError: int() ... not 'NoneType'` from inside their setup -- for a file
the schema had just accepted. Every shipped line-element case carried a
placeholder `n_steps: 100` around it. The sum is now written into the Case by
the validator that allows the omission, so every solver reads the same number.
"""
import os

import pytest
import yaml

from vfife.api import _with_element, prepare_case
from vfife.core.structural import make_structural_solver
from vfife.io import schema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _without_n_steps(stem, tmp_path):
    src = os.path.join(ROOT, "cases", stem + ".yaml")
    with open(src, encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    d["solver"].pop("n_steps", None)
    # the mesh is named relative to the case file
    d["mesh"]["path"] = os.path.normpath(os.path.join(ROOT, "cases", d["mesh"]["path"]))
    out = tmp_path / (stem + ".yaml")
    out.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return d, str(out)


def _expected(d):
    default = schema.SolverSpec.model_fields["max_steps_per_load"].default
    per_load = d["solver"].get("max_steps_per_load", default)
    return sum(int(st["n_steps"]) if st.get("n_steps") is not None
               else int(st["load_steps"]) * int(st.get("max_steps_per_load") or per_load)
               for st in d["steps"])


# one case per solver family that reads the budget in its setup
@pytest.mark.parametrize("stem", [
    "case_16_lee_frame",                   # beam2
    "case_17_hinged_cylindrical_roof",     # shell4
    "case_21b_vecchio_emara_cracked_shear",  # beam2_fiber (case_21 is static since 1.5)
    "case_24e_hzb_girder_bearings",        # mixed_line (beam + connector)
    "case_24f_hzb_girder_bearings",        # beam2_fiber + connector
])
def test_a_staged_case_without_n_steps_sets_up_with_the_summed_budget(stem, tmp_path):
    d, path = _without_n_steps(stem, tmp_path)
    prep = prepare_case(path)
    solver = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                                    prep.mesh, prep.material)
    assert solver.n_steps == _expected(d)


def test_an_explicit_n_steps_is_left_alone():
    case, _raw = schema.load(os.path.join(ROOT, "cases", "case_16_lee_frame.yaml"))
    assert case.solver.n_steps == 3000000


def test_a_step_with_no_budget_is_still_refused(tmp_path):
    d, path = _without_n_steps("case_16_lee_frame", tmp_path)
    for st in d["steps"]:
        st.pop("load_steps", None)
        st.pop("n_steps", None)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(d, fh, sort_keys=False)
    with pytest.raises(Exception, match="no iteration budget"):
        schema.load(path)
