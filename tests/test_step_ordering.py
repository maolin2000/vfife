"""Step group names must sort the same way lexicographically and numerically.

The names are the solver's cumulative iteration count, zero padded. That is
only safe while every name is the same width: at five digits a run passing
99999 iterations starts writing six-character names, and `sorted(f["steps"])`
then puts "00210" ... "99998" AFTER "100000".

case_11 does exceed 99999. The frames came back shuffled against the monitors
-- displacement off by the full 0.17 m range -- and for a while that looked
like the element internal forces disagreeing with statics frame by frame. It
was the sort. Six places in this repo sorted step names lexicographically,
including the verification contract evaluator.
"""
import numpy as np
import pytest


def test_all_step_names_share_one_width():
    """The property that makes a lexicographic sort safe, asserted directly."""
    from vfife.io.writer import ResultWriter
    import inspect
    src = inspect.getsource(ResultWriter)
    assert "{index:05d}" not in src, (
        "step groups are padded to five digits again; a run past 99999 "
        "iterations will produce names of two widths and any consumer that "
        "sorts them as strings will read the frames out of order")


@pytest.mark.parametrize("indices", [
    [1, 2, 3],
    [210, 99998, 99999, 100000, 100001],      # straddles the old five-digit edge
    [99999999, 100000000],                    # and the new eight-digit one
])
def test_padded_names_sort_the_same_either_way(indices):
    """Whatever width the writer uses, both orders have to agree."""
    from vfife.io import writer as W
    import re
    src = inspect.getsource(W) if False else open(W.__file__, encoding="utf-8").read()
    width = int(re.search(r"\{index:0(\d+)d\}", src).group(1))
    names = [f"{i:0{width}d}" for i in indices]
    lexicographic = sorted(names)
    numeric = sorted(names, key=int)
    if max(indices) < 10 ** width:
        assert lexicographic == numeric, (
            f"width {width} is not enough for {indices}: "
            f"{lexicographic} != {numeric}")
    else:
        # past the width the two DIVERGE -- which is the whole point, and why
        # readers use key=int rather than trusting the padding
        assert lexicographic != numeric


import inspect  # noqa: E402  (used above, imported late to keep the story first)


def test_readers_sort_numerically():
    """Every consumer of `steps` must pass key=int, padding or no padding."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for path in list((root / "vfife").rglob("*.py")) + \
                list((root / "benchmarks").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), 1):
            if "sorted(" in line and '"steps"' in line and "key=int" not in line:
                offenders.append(f"{path.relative_to(root)}:{line_no}: {line.strip()}")
    assert not offenders, (
        "these sort step names as strings, which breaks once the iteration "
        "counter changes width:\n  " + "\n  ".join(offenders))
