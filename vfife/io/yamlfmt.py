"""Compact YAML for case files: the style a person writes, not the style a
dumper emits.

WHY
    `yaml.safe_dump` puts every number of a 3-vector on its own line, writes
    0.21 as 0.21000000000000002, and the generator that used it produced a
    297-line file for a case a person writes in 77. The
    content was identical; only the shape was hostile. A case file is read
    far more often than it is written, and it is read by people.

RULES
    * a list of scalars goes on one line: `[2.0, 0.2, 0.3]`
    * a mapping whose values are all scalars or short scalar lists goes on
      one line when it fits: `{kind: reaction, region: sup_L, dof: z}`
    * floats are written to 10 significant digits, so 0.21000000000000002
      is 0.21 and 29849623113.198597 is 2.984962311e10
    * everything else stays in block style, keys in the order given
"""
from __future__ import annotations

from typing import Any

import yaml

_FLOW_WIDTH = 100


def _short(x: float) -> float:
    return float(f"{x:.10g}")


def _prepare(obj: Any) -> Any:
    if isinstance(obj, float):
        return _short(obj)
    if isinstance(obj, dict):
        return {k: _prepare(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_prepare(v) for v in obj]
    return obj


class _Flow(list):
    pass


class _FlowMap(dict):
    pass


def _is_scalar(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _mark(obj: Any) -> Any:
    """Wrap the lists and mappings that should be written inline."""
    if isinstance(obj, list):
        items = [_mark(v) for v in obj]
        if all(_is_scalar(v) for v in items):
            return _Flow(items)
        return items
    if isinstance(obj, dict):
        items = {k: _mark(v) for k, v in obj.items()}
        inline = all(_is_scalar(v) or isinstance(v, _Flow) for v in items.values())
        if inline and len(repr(items)) <= _FLOW_WIDTH and items:
            return _FlowMap(items)
        return items
    return obj


class _Dumper(yaml.SafeDumper):
    pass


def _repr_flow(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", list(data), flow_style=True)


def _repr_flow_map(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", dict(data), flow_style=True)


def _repr_float(dumper, x):
    """192000000000.0 is unreadable and 1.92e+11 is what the analyst typed.
    Exponent form outside [1e-4, 1e6). PyYAML reads `1.92e11` and `1e+11` as
    STRINGS: the YAML 1.1 float needs a dot in the mantissa and a signed
    exponent, so both are always written."""
    import math
    if x != x or x in (float("inf"), float("-inf")):
        return dumper.represent_float(x)
    a = abs(x)
    if a != 0.0 and (a >= 1e6 or a < 1e-4):
        exp = int(math.floor(math.log10(a)))
        mant = x / 10.0 ** exp
        m = f"{mant:.10g}"
        if "." not in m:
            m += ".0"
        text = f"{m}e{exp:+d}"
    else:
        text = f"{x:.10g}"
        if "." not in text and "e" not in text:
            text += ".0"
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


_Dumper.add_representer(_Flow, _repr_flow)
_Dumper.add_representer(_FlowMap, _repr_flow_map)
_Dumper.add_representer(float, _repr_float)


def dump_compact(obj: Any) -> str:
    """YAML text for `obj` in the compact style. Round-trips through
    `yaml.safe_load` to the same values (floats to 10 significant digits)."""
    return yaml.dump(_mark(_prepare(obj)), Dumper=_Dumper, sort_keys=False,
                     allow_unicode=True, width=120)
