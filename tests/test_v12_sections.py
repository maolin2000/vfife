"""1.2, item 8: box and T girders from the drawing's dimensions, and bars in
layers.

The checks are closed forms the shapes must reduce to -- a box with vertical
webs and no cantilevers is the hollow rectangle; a T is two rectangles; a
symmetric two-cell box has the torsion of its one-cell outline, because the
middle web carries no shear flow -- and the one statement the feature makes:
a beam and a fibre model of the same girder agree.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vfife import section_shapes as ss
from vfife.api import run_case
from vfife.elements.fiber_section import patch_fibers
from vfife.io.model import SectionSpec

ROOT = Path(__file__).resolve().parents[1]
E = 3.45e10

BOX = dict(shape="box_girder", height=2.5, top_width=12.0, bottom_width=6.0, cantilever=2.5,
           top_thickness=0.25, bottom_thickness=0.22, web_thickness=0.45,
           cantilever_thickness=[0.5, 0.2], top_haunch=[0.6, 0.2], bottom_haunch=[0.3, 0.15])


def _case(tmp_path, kind, sec=None, loads=None, orientation=(0, 1, 0), name=None, **extra):
    name = name or kind
    s = dict(type=kind, material="c", **(sec or BOX), **extra)
    return {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "B": [20, 0, 0]},
                     "lines": {"g": {"through": ["A", "B"], "size": 2.0}}},
            "materials": {"c": {"type": "linear_elastic", "E": E, "nu": 0.2, "rho": 2600.0},
                          "s": {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}},
            "sections": {"s": s, "d16": {"type": "bar", "material": "s", "shape": "circle",
                                         "diameter": 0.016}},
            "assign": [{"region": "g", "section": "s", "orientation": list(orientation)}],
            "constraints": [{"region": "A", "dofs": [1] * 6}],
            "loads": loads if loads is not None else [{"region": "B", "vector": [0, 0, -1.0e6]}],
            "solver": {"mode": "static", "backend": "numpy"},
            "output": {"path": str(tmp_path / f"{name}.h5"),
                       "fields": ["displacement", "rotation"],
                       "monitors": {"w": {"kind": "displacement", "region": "B", "dof": "z"}}}}


def _tip(tmp_path, *a, **k):
    """(w, rx) at the free end, at the last frame."""
    import h5py
    out = run_case(_case(tmp_path, *a, **k))
    assert out.converged, out.message
    with h5py.File(out.result_path, "r") as f:
        x = np.asarray(f["mesh/nodes"])[:, 0]
        last = f["steps"][sorted(f["steps"], key=lambda n: int(n.split("_")[-1]))[-1]]
        rx = float(np.asarray(last["rotation"])[int(np.argmax(x)), 0])
    return out.monitors["w"][-1], rx


def test_a_box_without_cantilevers_or_lean_is_the_hollow_rectangle():
    sp = dict(shape="box_girder", height=2.0, top_width=4.0, bottom_width=4.0, cantilever=0.0,
              top_thickness=0.2, bottom_thickness=0.2, web_thickness=0.2)
    g = ss.girder(sp)
    assert g["area"] == pytest.approx(4 * 2 - 3.6 * 1.6, rel=1e-12)
    assert g["iy"] == pytest.approx((4 * 2 ** 3 - 3.6 * 1.6 ** 3) / 12, rel=1e-12)
    assert g["iz"] == pytest.approx((2 * 4 ** 3 - 1.6 * 3.6 ** 3) / 12, rel=1e-12)
    # Bredt on the centre line, as sections.py's hollow rect
    am = 3.8 * 1.8
    assert g["j"] == pytest.approx(4 * am * am * 0.2 / (2 * (3.8 + 1.8)), rel=1e-12)
    from vfife.sections import properties
    rect = properties({"shape": "rect", "b": 4.0, "h": 2.0, "t": 0.2})
    for k in ("area", "iy", "iz", "j"):
        assert g[k] == pytest.approx(rect[k], rel=1e-12)


def test_a_t_is_two_rectangles_and_an_i_is_three():
    g = ss.girder(dict(shape="t_girder", height=1.5, top_width=2.0, top_thickness=0.2,
                       web_thickness=0.3))
    parts = [(2.0, 0.2, -0.1), (0.3, 1.3, -0.85)]        # (width, depth, centre z)
    A = sum(b * h for b, h, _ in parts)
    zc = sum(b * h * z for b, h, z in parts) / A
    assert g["area"] == pytest.approx(A, rel=1e-12)
    assert g["centroid"][1] == pytest.approx(zc, rel=1e-12)
    assert g["iy"] == pytest.approx(sum(b * h ** 3 / 12 + b * h * (z - zc) ** 2
                                        for b, h, z in parts), rel=1e-12)
    g = ss.girder(dict(shape="t_girder", height=1.5, top_width=2.0, top_thickness=0.2,
                       web_thickness=0.3, bottom_width=0.8, bottom_thickness=0.25))
    parts = [(2.0, 0.2, -0.1), (0.3, 1.05, -0.725), (0.8, 0.25, -1.375)]
    A = sum(b * h for b, h, _ in parts)
    zc = sum(b * h * z for b, h, z in parts) / A
    assert g["iy"] == pytest.approx(sum(b * h ** 3 / 12 + b * h * (z - zc) ** 2
                                        for b, h, z in parts), rel=1e-12)
    assert g["iz"] == pytest.approx(sum(h * b ** 3 / 12 for b, h, _ in parts), rel=1e-12)


def test_the_middle_web_of_a_symmetric_two_cell_box_carries_no_twist():
    sp = dict(shape="box_girder", height=2.0, top_width=8.0, bottom_width=8.0, cantilever=0.0,
              top_thickness=0.2, bottom_thickness=0.2, web_thickness=0.3)
    one = ss.girder(sp)["j"]
    two = ss.girder(dict(sp, cells=2))["j"]
    three = ss.girder(dict(sp, cells=3))["j"]
    assert two == pytest.approx(one, rel=1e-12)
    assert three != pytest.approx(one, rel=1e-3)        # unequal flows: the webs do work


def test_the_fibres_tile_the_outline_and_converge_on_its_second_moment():
    g = ss.girder(BOX)
    errs = []
    for k, (layers, size) in enumerate(((2, 0.6), (4, 0.3), (8, 0.15))):
        P = ss.patches(dict(BOX, layers=layers, fiber_size=size))
        y, z, a, _ = patch_fibers(P["patches"])
        assert a.sum() == pytest.approx(g["area"], rel=1e-12)       # bilinear cells tile a quad
        assert (a * z).sum() == pytest.approx(0.0, abs=1e-3 * g["area"])   # about the centroid
        errs.append(abs((a * z * z).sum() - g["iy"]) / g["iy"])
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-3
    # the default is near two hundred fibres (case_24d ran its girder with 166)
    P = ss.patches(BOX)
    assert 150 < sum(p["ny"] * p["nz"] for p in P["patches"]) < 350


def test_a_beam_and_a_fibre_girder_bend_and_twist_alike(tmp_path):
    g = ss.girder(BOX)
    w_beam, _ = _tip(tmp_path, "beam")
    assert w_beam == pytest.approx(-1.0e6 * 20 ** 3 / (3 * E * g["iy"]), rel=2e-6)   # corotational, 1e-4 rad
    w_fib, _ = _tip(tmp_path, "fiber")
    assert w_fib == pytest.approx(w_beam, rel=5e-3)
    # torsion: the fibre section gets the box's Bredt constant, not the polar moment
    T = [{"region": "B", "moment": [1.0e6, 0, 0]}]
    G = E / 2.4
    for kind in ("beam", "fiber"):
        _w, rx = _tip(tmp_path, kind, loads=T, name=f"{kind}_t")
        assert rx == pytest.approx(1.0e6 * 20 / (G * g["j"]), rel=1e-6)


def test_the_origin_moves_the_member_line_not_the_stiffness(tmp_path):
    w_c, _ = _tip(tmp_path, "fiber", name="c")
    w_t, _ = _tip(tmp_path, "fiber", name="t", origin="top")
    # an element whose axis is off the centroid couples its linear axial
    # and cubic bending fields: 0.11% stiff at 2 m members, 0.024% at 1 m,
    # 0.001% at 0.5 m (measured)
    assert w_t == pytest.approx(w_c, rel=2e-3)
    # pulled along the deck's top, the girder bends: P e L^2 / (2 E I)
    g = ss.girder(BOX)
    P = [{"region": "B", "vector": [1.0e6, 0, 0]}]
    w, _ = _tip(tmp_path, "fiber", loads=P, name="tp", origin="top")
    e = g["centroid"][1]                                 # the centroid below the top (negative)
    assert w == pytest.approx(1.0e6 * e * 20 ** 2 / (2 * E * g["iy"]), rel=1e-2)
    with pytest.raises(ValueError, match="does not read `origin`"):
        SectionSpec(type="beam", material="c", **BOX, origin="top")


def test_bars_in_layers_by_count_and_by_spacing():
    s = SectionSpec(type="fiber", material="c", **BOX, bar_layers=[
        {"section": "d16", "from": [-5.9, 0.05], "to": [5.9, 0.05], "spacing": 0.15, "z_from": "top"},
        {"section": "d16", "from": [-2.9, 0.06], "to": [2.9, 0.06], "count": 30, "z_from": "bottom"},
        {"section": "d16", "from": [0.0, 0.0], "to": [0.0, 0.0], "count": 1}])
    assert s.shape == "patches" and not s.bar_layers
    top = [b for b in s.bars if b["z"] > 0.5]
    bot = [b for b in s.bars if b["z"] < -1.0]
    # 11.8 m at 150 mm or closer: 79 gaps, 80 bars, both ends on the line
    assert len(top) == 80
    assert np.diff(sorted(b["y"] for b in top)).max() <= 0.15 + 1e-12
    g = ss.girder(BOX)
    assert top[0]["z"] == pytest.approx(g["top"] - 0.05, rel=1e-12)
    assert len(bot) == 30 and bot[0]["z"] == pytest.approx(g["bottom"] + 0.06, rel=1e-12)
    assert [b["y"] for b in bot][::29] == pytest.approx([-2.9, 2.9])
    assert sum(1 for b in s.bars if b["y"] == 0.0 and b["z"] == 0.0) == 1
    # a rectangle measures from its own faces
    r = SectionSpec(type="fiber", material="c", width=0.4, height=0.6,
                    bar_layers=[{"section": "d16", "from": [-0.15, 0.04], "to": [0.15, 0.04],
                                 "count": 4, "z_from": "bottom"}])
    assert [b["z"] for b in r.bars] == pytest.approx([-0.26] * 4)


def test_bars_in_layers_run_and_carry_their_steel(tmp_path):
    lay = [{"section": "d16", "from": [-5.9, 0.05], "to": [5.9, 0.05], "count": 60, "z_from": "top"}]
    d = _case(tmp_path, "fiber", bar_layers=lay, name="rc")
    out = run_case(d)
    assert out.converged, out.message
    w_plain, _ = _tip(tmp_path, "fiber", name="plain")
    assert abs(out.monitors["w"][-1]) < abs(w_plain)        # the steel stiffens it


@pytest.mark.parametrize("sec, words", [
    (dict(type="fiber", material="c", width=0.4, height=0.6, top_width=2.0), "for shape: box_girder"),
    (dict(type="fiber", material="c", **BOX, width=1.0), "does not read `width`"),
    (dict(type="beam", material="c", **BOX, layers=4), "does not read `layers`"),
    (dict(type="fiber", material="c", **dict(BOX, top_thickness=1.4, bottom_thickness=1.2)), "fill the height"),
    (dict(type="fiber", material="c", **dict(BOX, top_haunch=[3.1, 0.2])), "close the cell"),
    (dict(type="fiber", material="c", **dict(BOX, cantilever=5.8)), "no room for the webs"),
    (dict(type="fiber", material="c", shape="t_girder", height=1.5, top_width=2.0, top_thickness=0.2,
          web_thickness=0.3, bottom_width=0.8), "needs both bottom_width and bottom_thickness"),
    (dict(type="shell", material="c", thickness=0.2, shape="box_girder"), "beam or fiber section"),
    (dict(type="beam", material="c", shape="rect", width=0.4, height=0.6,
          bar_layers=[{"section": "d16", "from": [0, 0], "to": [0, 0], "count": 1}]), "fibre section"),
    (dict(type="fiber", material="c", width=0.4, height=0.6,
          bar_layers=[{"section": "d16", "from": [0, 0], "to": [0.1, 0]}]), "one of `count` or `spacing`"),
    (dict(type="fiber", material="c", width=0.4, height=0.6,
          bar_layers=[{"section": "d16", "from": [0, 0], "to": [0.1, 0], "count": 2, "n": 3}]),
     "does not read `n`"),
])
def test_what_a_girder_and_a_layer_refuse(sec, words):
    with pytest.raises(ValueError, match=words):
        SectionSpec(**sec)


def test_a_bar_layer_outside_the_girder_is_refused(tmp_path):
    lay = [{"section": "d16", "from": [-5.9, -0.3], "to": [5.9, -0.3], "count": 10, "z_from": "top"}]
    with pytest.raises(ValueError, match="outside the section"):
        run_case(_case(tmp_path, "fiber", bar_layers=lay, name="out"))


def test_info_says_when_a_girder_lies_on_its_side(tmp_path):
    import yaml
    for kind in ("beam", "fiber"):
        d = _case(tmp_path, kind, orientation=(0, 0, 1), name=f"side_{kind}")
        p = tmp_path / f"side_{kind}.yaml"
        p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
        r = subprocess.run([sys.executable, "-m", "vfife", "info", str(p)], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr
        assert "section height 2.5 m lies along local z" in r.stdout, r.stdout


# --------------------------------------------------------------------------- #
#  from the review (1.2)
# --------------------------------------------------------------------------- #

def _rect_j(a, b):
    a, b = max(a, b), min(a, b)
    s = sum(np.tanh(n * np.pi * a / (2 * b)) / n ** 5 for n in range(1, 200, 2))
    return a * b ** 3 / 3 * (1 - 192 / np.pi ** 5 * b / a * s)


def test_a_t_s_torsion_is_saint_venant_s_on_its_outline():
    """The sum of the rectangles was 11-26% low: it leaves out the junctions."""
    web = ss.Wall([(-0.15, -1.3), (0.15, -1.3), (0.15, 0), (-0.15, 0)], "web", 0.3)
    assert ss.saint_venant_j([web]) == pytest.approx(_rect_j(0.3, 1.3), rel=1e-2)
    for sp in (dict(shape="t_girder", height=1.5, top_width=2.0, top_thickness=0.2, web_thickness=0.3),
               dict(shape="t_girder", height=1.5, top_width=2.0, top_thickness=0.3, web_thickness=0.3,
                    cantilever_thickness=[0.3, 0.15], bottom_width=0.7, bottom_thickness=0.2)):
        g = ss.girder(sp)
        fine = ss.saint_venant_j(g["walls"], cells_per_thickness=72)
        assert g["j"] == pytest.approx(fine, rel=1e-2)
    # the plain T against its two rectangles (flange 2.0 x 0.2, web 0.3 x 1.3)
    plain = ss.girder(dict(shape="t_girder", height=1.5, top_width=2.0, top_thickness=0.2,
                           web_thickness=0.3))["j"]
    assert plain > 1.1 * (_rect_j(2.0, 0.2) + _rect_j(0.3, 1.3))


def test_bars_must_be_in_the_walls_and_a_tendon_may_run_in_the_cell(tmp_path):
    above = [{"section": "d16", "from": [-5.9, -0.1], "to": [5.9, -0.1], "count": 40, "z_from": "top"}]
    with pytest.raises(ValueError, match="outside the section"):         # 1% of 12 m let it in
        run_case(_case(tmp_path, "fiber", bar_layers=above, name="above"))
    cell = [{"section": "d16", "from": [-1.0, 0.35], "to": [1.0, 0.35], "count": 5, "z_from": "top"}]
    with pytest.raises(ValueError, match="outside the section"):
        run_case(_case(tmp_path, "fiber", bar_layers=cell, name="cell"))
    d = _case(tmp_path, "fiber", name="ext")
    d["sections"]["s"]["tendons"] = [{"section": "d16", "profile": [[0, 0, -0.5], [20, 0, -0.5]],
                                      "prestress": 1.0e9}]
    out = run_case(d)
    assert out.converged, out.message


@pytest.mark.parametrize("sec, words", [
    # an interior web through the outer one near the bottom slab
    (dict(BOX, bottom_width=2.0, cells=3, top_haunch=None, bottom_haunch=None), "no room between the webs"),
    (dict(BOX, bottom_width=40.0, top_haunch=None, bottom_haunch=None), "past the cantilever tips"),
    (dict(BOX, cantilever=0.0, top_width=6.0, cantilever_thickness=[5, 5]), "without cantilevers"),
    (dict(shape="t_girder", height=1.5, top_width=2.0, top_thickness=0.2, web_thickness=0.3,
          cantilever_thickness=[0.3, 0.15]), r"cantilever_thickness as \[0.2, <tip>\]"),
])
def test_what_the_review_found_accepted(sec, words):
    sec = {k: v for k, v in sec.items() if v is not None}
    with pytest.raises(ValueError, match=words):
        SectionSpec(type="fiber", material="c", **sec)


def test_a_deep_top_haunch_alone_fits():
    sp = {k: v for k, v in BOX.items() if k != "bottom_haunch"}
    SectionSpec(type="fiber", material="c", **dict(sp, top_haunch=[0.6, 1.2]))   # > half the cell


def test_layer_counts_and_places_are_numbers_not_booleans():
    for bad in ({"count": True}, {"count": 2, "from": [True, 0]}):
        layer = {"section": "d16", "from": [0, 0], "to": [0.1, 0], **bad}
        with pytest.raises(ValueError):
            SectionSpec(type="fiber", material="c", width=0.4, height=0.6, bar_layers=[layer])
    with pytest.raises(ValueError, match="take no bars"):
        SectionSpec(type="fiber", material="c", shape="box", width=0.4, height=0.6, t=0.02,
                    bar_layers=[{"section": "d16", "from": [0, 0], "to": [0.1, 0], "count": 2,
                                 "z_from": "top"}])
