"""io/yamlfmt.py: case files in the shape a person writes.

WHY
    A generator that used `yaml.safe_dump` produced a 297-line file for a
    case a person writes in 77 lines: every number of a 3-vector on its own
    line, 0.21000000000000002 for 0.21, 192000000000.0 for 1.92e+11. Same
    content, hostile shape -- and a case file is read far more than written.
"""
import yaml

from vfife.io.yamlfmt import dump_compact


def _case():
    return {
        "name": "t", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [2.0, 0.2, 0.3], "divisions": [24, 4, 8]},
        "material": {"type": "cdp", "E": 29849623113.198597, "nu": 0.2,
                     "compression": {"source": "gb50010", "fc": 44e6}},
        "reinforcement": [{"name": "bar", "area": 200e-6,
                           "material": {"type": "rebar", "E": 1.92e11},
                           "lines": [{"start": [0, 0.0443, 0.0443], "end": [2.0, 0.0443, 0.0443],
                                      "segment_length": 0.025}]}],
        "regions": {"sup": {"where": {"x": 0.1, "z": 0}}},
        "output": {"monitors": {"P": {"kind": "reaction", "region": "sup", "dof": "z"}}},
    }


def test_round_trips_to_the_same_values():
    text = dump_compact(_case())
    back = yaml.safe_load(text)

    def eq(a, b):
        if isinstance(a, float) and isinstance(b, (int, float)):
            return abs(a - b) <= 1e-9 * max(1.0, abs(b))
        if isinstance(a, dict):
            return set(a) == set(b) and all(eq(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
        return a == b
    assert eq(back, _case())


def test_the_shape_is_the_one_a_person_writes():
    text = dump_compact(_case())
    assert "dimensions: [2.0, 0.2, 0.3]" in text
    assert "{kind: reaction, region: sup, dof: z}" in text
    assert "E: 1.92e+11" in text and "E: 2.984962311e+10" in text
    assert "fc: 4.4e+7" in text
    assert "0.21000000000000002" not in text
    # the same case through the plain dumper is more than twice as long
    plain = yaml.safe_dump(_case(), sort_keys=False)
    assert len(text.splitlines()) * 2 < len(plain.splitlines())


def test_exponent_floats_are_floats_when_read_back():
    """PyYAML reads `1e+11` and `1.92e11` as strings -- the YAML 1.1 float
    needs a dot in the mantissa AND a signed exponent. Both always written."""
    for v in (1e11, 449e6, 2.5e-5, 1e6, 999999.0, 0.0, -0.02):
        back = yaml.safe_load(dump_compact([v]))[0]
        assert isinstance(back, float), (v, dump_compact([v]))
        assert abs(back - v) <= 1e-9 * max(1.0, abs(v))
