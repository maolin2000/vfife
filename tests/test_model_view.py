"""`vfife view case.yaml`: the model page -- the case as the
solver reads it, step by step, before it is run."""
import os

import numpy as np
import pytest

from vfife.post import model_view

from test_tendon_block import P, _staged, _tendon, _write


def _case(tmp_path):
    d = _staged(tmp_path, _tendon(stress={"ends": "both", "force": [P, P]}))
    d["loads"] = [{"region": "all", "gravity": [0, 0, -9.81]}]
    d["steps"][1]["loads"] = [{"region": "girder", "line_load": [0, 0, -2.0e4], "keep": True},
                              {"region": "right", "vector": [0, 0, -1.0e5]}]
    d["steps"].append({"name": "after", "release": ["right"]})
    return _write(tmp_path, d, "staged.yaml")


def test_the_model_step_by_step_as_the_solver_reads_it(tmp_path):
    data = model_view.load(str(_case(tmp_path)))
    a, m = data["arrays"], data["meta"]
    kinds = np.bincount(a["kind"], minlength=len(model_view.KINDS))
    assert kinds.tolist() == [15, 0, 15, 16, 0]                      # girder, tendon bars, tendon arms
    # the cantilevers in the first step, the closure and the tendon in the second
    assert [s["built"] for s in m["steps"]] == [10, 24, 0]
    assert [s["standing"] for s in m["steps"]] == [22, 46, 46]
    reg = {r: i for i, r in enumerate(m["regions"])}
    assert np.all(a["born"][a["reg"] == reg["C"]] == 1) and np.all(a["born"][a["reg"] == reg["L"]] == 0)
    assert a["built"][a["reg"] == reg["C"]].tolist()[0] == [0, 1, 1]
    # supports: the left end's ux uy uz rx (15), the right's uy uz (6); released in the third step
    masks = dict(zip(a["sup_nodes"].tolist(), a["sup_masks"].tolist()))
    assert masks[0] == [15, 15, 15] and masks[15] == [6, 6, 0]
    # loads: nothing drawn in the first step (gravity is listed); the line load on the girder from the
    # second step on, kept into the third (one arrow for both); the point load the second step's own
    arrows = a["arrows"]                                              # [from, to, kind, x, y, z, vx, vy, vz]
    lines = arrows[arrows[:, 2] == 2]
    assert len(lines) == 15 and set(lines[:, 0]) == {1.0} and set(lines[:, 1]) == {2.0}
    point = arrows[arrows[:, 2] == 0]
    assert len(point) == 1 and point[0, 0] == point[0, 1] == 1 and point[0, 8] == pytest.approx(-1.0e5)
    assert any("gravity" in t for t in m["steps"][0]["loads"])
    assert any("kept from an earlier step" in t for t in m["steps"][2]["loads"])
    assert m["tendons"][0]["name"] == "T1" and len(m["tendons"][0]["bars"]) == 15


def test_the_page_is_one_file_and_the_cli_writes_it(tmp_path):
    from vfife import cli
    path = _case(tmp_path)
    assert cli.main(["view", str(path)]) in (0, None)
    html = open(tmp_path / "staged.model.html", encoding="utf-8").read()
    assert "__META__" not in html and "__BLOB__" not in html and '"n_members":46' in html
    assert cli.main(["view", str(path), str(path)]) not in (0, None)       # one case only


def test_the_model_page_runs_in_a_browser(tmp_path):
    # its script executed: no error, a member picked by clicking at it.
    # Needs Playwright with Chromium and three.js r128 in VFIFE_THREE_DIR
    three = os.environ.get("VFIFE_THREE_DIR")
    pw = pytest.importorskip("playwright.sync_api")
    if not three:
        pytest.skip("VFIFE_THREE_DIR not set")
    page = model_view.write(str(_case(tmp_path)), scripts=three)
    errors = []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:                      # no browser installed
            pytest.skip(f"no Chromium: {e}")
        pg = b.new_page(viewport={"width": 1280, "height": 760})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.route("**/*", lambda r: r.abort() if r.request.url.startswith("http") else r.continue_())
        pg.goto("file://" + os.path.abspath(page))
        pg.wait_for_timeout(1000)
        assert pg.evaluate("window.__vfife_model.steps") == 3
        pg.click("#next"); pg.wait_for_timeout(200)
        x, y = pg.evaluate("window.__vfife_model.at(7)")     # a member of the closure, built in step 2
        pg.mouse.click(x, y); pg.wait_for_timeout(200)
        picked = pg.inner_text("#pick")
        pg.select_option("#colour", "section"); pg.wait_for_timeout(200)
        b.close()
    assert not errors, errors
    assert "closure" in picked


def test_the_solvers_rules_for_stages_supports_and_loads(tmp_path):
    """What the review of v1.20 found: the case's regions and `all` hold
    members; a member removed and built again; a support kept after a
    release; a line load only on members built, with both nodes in the region."""
    d = _staged(tmp_path, _tendon(stress={"ends": "both", "force": [P, P]}))
    d.pop("tendons")                                  # (with tendons a step builds mesh groups only)
    d["regions"] = {"near": {"box": [-1, 10.01, -1, 1, -1, 1]}}
    d["assign"] = [{"region": "all", "section": "box", "orientation": [0, 0, 1]}]
    d["steps"] = [{"name": "a", "activate": ["L", "R"], "loads": [{"region": "girder", "line_load": [0, 0, -1.0]}]},
                  {"name": "b", "activate": ["C"], "release": ["right"],
                   "constraints": [{"region": "right", "dofs": [0, 0, 1, 0, 0, 0], "keep": True}]},
                  {"name": "c", "deactivate": ["near"]},
                  {"name": "d", "activate": ["near"]}]
    a = model_view.load(str(_write(tmp_path, d, "rules.yaml")))["arrays"]
    girder = a["kind"] == 0
    assert np.all(a["sec"][girder] >= 0)                                # `all` holds every member
    near = np.flatnonzero(girder & (a["built"][:, 2] == 0))
    assert len(near) == 5 and np.all(a["built"][near, 3] == 1)          # removed in c, built again in d
    masks = dict(zip(a["sup_nodes"].tolist(), a["sup_masks"].tolist()))
    assert masks[15] == [6, 4, 4, 4]                                    # kept after the release: still held
    lines = a["arrows"][a["arrows"][:, 2] == 2]
    assert len(lines) == 10                                             # not the closure, unbuilt in a


def test_a_name_is_text_in_the_page(tmp_path):
    d = _staged(tmp_path, _tendon(stress={"ends": "both", "force": [P, P]}))
    d["name"] = '<img src=x onerror="alert(1)">'
    page = model_view.write(str(_write(tmp_path, d, "x.yaml")))
    html = open(page, encoding="utf-8").read()
    assert "<img src=x" not in html and "&lt;img" in html


def _bearings(tmp_path):
    """A girder on two bearings and a pier on a pile in the ground: every
    kind of component a connector has, supports of every kind, soil springs."""
    import yaml
    pts = {"a": [0, 0, 0], "b": [10, 0, 0], "c": [20, 0, 0], "at": [0, 0, -0.5], "ct": [20, 0, -0.5],
           "bt": [10, 0, -0.6], "bf": [10, 0, -12.6], "ag": [0, 0, -1.3], "cg": [20, 0, -1.3]}
    lines = {"deck": {"through": ["a", "b", "c"], "divisions": 4},
             "arm_a": {"through": ["a", "at"], "divisions": 1}, "arm_c": {"through": ["c", "ct"], "divisions": 1},
             "brg_a": {"through": ["at", "ag"], "divisions": 1}, "brg_c": {"through": ["ct", "cg"], "divisions": 1},
             "brg_b": {"through": ["b", "bt"], "divisions": 1}, "pile": {"through": ["bt", "bf"], "divisions": 6}}
    d = {"version": 1, "name": "bearings", "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": pts, "lines": lines,
                  "groups": {"arms": ["arm_a", "arm_c"]}},
         "materials": {"c": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2500.0}},
         "sections": {"g": {"type": "beam", "material": "c", "shape": "general", "area": 5.0, "iy": 3.0, "iz": 20.0, "j": 4.0},
                      "offset_arm_x": {"type": "beam", "material": "c", "shape": "general", "area": 50.0, "iy": 99.0, "iz": 99.0, "j": 99.0},
                      "slide": {"type": "connector", "stiffness": [2e9, 0, 0, 0, 0, 0], "connector_damping": [0, 1e6, 0, 0, 0, 0],
                                "laws": {"uz": {"type": "gap", "lo": -0.01, "hi": 0.01, "stiffness": 1e9},
                                         "rx": {"type": "bilinear", "k1": 1e8, "k2": 1e6, "fy": 1e5}}},
                      "fix": {"type": "connector", "stiffness": [1e9] * 6}},
         "assign": [{"region": "deck", "section": "g", "orientation": [0, 0, 1]},
                    {"region": "arms", "section": "offset_arm_x", "orientation": [1, 0, 0]},
                    {"region": "pile", "section": "g", "orientation": [1, 0, 0]},
                    {"region": "brg_a", "section": "slide"},
                    {"region": "brg_c", "section": "slide", "orientation": [0, 1, 0]},
                    {"region": "brg_b", "section": "fix", "coordinate_system": "global"}],
         "constraints": [{"region": "ag", "dofs": [1, 1, 1, 1, 1, 1]},
                         {"region": "cg", "dofs": [1, 1, 1, 0, 0, 0], "value": [0.01, 0, 0, 0, 0, 0]},
                         {"region": "bf", "dofs": [0, 0, 1, 0, 0, 0]}],
         "soil_springs": [{"region": "pile", "ground": -4.6, "curves": [{"depth": 0.0, "p_y": [[0, 0], [0.01, 1e5]]}]}],
         "solver": {"mode": "static"}}
    p = tmp_path / "bearings.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return p


def test_connectors_and_soil_springs_as_the_solver_reads_them(tmp_path):
    data = model_view.load(str(_bearings(tmp_path)))
    a, m = data["arrays"], data["meta"]
    S, CU, GAP, DAMP = model_view.SPRING, model_view.CURVE, model_view.GAP, model_view.DAMPER
    codes = a["cn_codes"].reshape(-1, 6)
    axes = a["cn_axes"].reshape(-1, 3, 3)
    nodes = a["nodes"] + np.asarray(m["centre"], np.float32)
    rows = {}
    for c, e in enumerate(a["cn_ids"].tolist()):
        i, j = a["conn"][e]
        rows[(round(float(nodes[i][0])), round(float(nodes[i][2]), 1))] = (codes[c].tolist(), axes[c])
    # ux a stiffness, uy a damping only, uz a gap law, rx a bilinear law
    (sa, ax_a), (sc, ax_c), (sb, ax_b) = rows[(0, -0.5)], rows[(20, -0.5)], rows[(10, 0.0)]
    assert sa == sc == [S, DAMP, GAP, CU, 0, 0]
    assert sb == [S] * 6
    # the axes, as the solver builds them: x along the member (down), y the global axis most nearly
    # normal to it or the orientation; the global axes with coordinate_system: global
    from vfife.cli import _with_element, build, make_solver
    from vfife.io import schema
    path = str(tmp_path / "bearings.yaml")
    case, _raw = schema.load(path)
    mesh, mat, el = build(case, path)
    solver = make_solver(_with_element(case, el, path), mesh, mat)
    R0 = np.asarray(solver.R0)
    for c, e in enumerate(a["cn_ids"].tolist()):
        glob = solver.member_props[e].get("coordinate_system") == "global"
        assert np.allclose(axes[c], np.eye(3) if glob else R0[e], atol=1e-6)
    assert np.allclose(ax_a[0], [0, 0, -1]) and np.allclose(ax_a[1], [1, 0, 0]) and np.allclose(ax_c[1], [0, 1, 0])
    assert np.allclose(ax_b, np.eye(3))
    assert sorted(solver.soil.node_ids.tolist()) == sorted(a["soil_nodes"].tolist())
    words = m["connector_words"]
    assert any("gap lo -0.01" in w for ws in words.values() for w in ws)
    assert any(w.startswith("bilinear") for ws in words.values() for w in ws)
    # a damper law with its own spring beside it (elements/connector.py): both
    pot = model_view._connector({"stiffness": [2e9, 0, 3e6, 0, 0, 1e7],
                                 "laws": {"uy": {"type": "damper", "c": 1e6, "exponent": 1.0, "stiffness": 3e6}}})
    assert pot[0] == [S, S | DAMP, S, 0, 0, S] and "stiffness 3000000.0" in pot[1][1]
    # one friction law on uy and uz: a law in the y-z plane, marked on both
    iso = model_view._connector({"stiffness": [2e9, 0, 0, 0, 0, 0],
                                 "laws": {c: {"type": "friction", "mu": 0.05, "stiffness": 1e8, "normal": "ux"} for c in ("uy", "uz")}})
    assert iso[0][1] == iso[0][2] == model_view.SLIDE | model_view.PLANAR and "y-z plane" in iso[1][1]
    # supports, by the dof; the one moving dof of a prescribed motion
    masks = dict(zip(a["sup_nodes"].tolist(), a["sup_masks"][:, 0].tolist()))
    moving = dict(zip(a["sup_nodes"].tolist(), a["sup_moving"][:, 0].tolist()))
    assert sorted(masks.values()) == [4, 63, 7 | 64]
    assert sorted(moving.values()) == [0, 0, 1]
    # soil springs: the pile's nodes below the ground (and the one its member crosses at), radial
    assert len(a["soil_nodes"]) == 5 and set(a["soil_axes"].tolist()) == {1 | 2 | 8}
    assert nodes[a["soil_nodes"]][:, 2].max() < -4.0
    assert "radial" in m["soil"][0]


def test_one_symbol_each_with_its_code_and_the_one_clicked_in_full(tmp_path):
    """Version 1.20: a symbol for each support, connector and node with soil
    springs, at any distance; the codes that would overlap left out; the one
    clicked drawn dof by dof, component by component."""
    three = os.environ.get("VFIFE_THREE_DIR")
    pw = pytest.importorskip("playwright.sync_api")
    if not three:
        pytest.skip("VFIFE_THREE_DIR not set")
    page = model_view.write(str(_bearings(tmp_path)), scripts=three)
    data = model_view.load(str(_bearings(tmp_path)))
    nodes = data["arrays"]["nodes"] + np.asarray(data["meta"]["centre"], np.float32)
    ground = int(np.argmin(np.linalg.norm(nodes - np.array([0, 0, -1.3]), axis=1)))
    other = int(np.argmin(np.linalg.norm(nodes - np.array([20, 0, -1.3]), axis=1)))
    n_symbols = (len(data["arrays"]["sup_nodes"]) + len(data["arrays"]["cn_ids"])
                 + len(np.unique(data["arrays"]["soil_nodes"])))
    errors = []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:                      # no browser installed
            pytest.skip(f"no Chromium: {e}")
        pg = b.new_page(viewport={"width": 1280, "height": 760})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.route("**/*", lambda r: r.abort() if r.request.url.startswith("http") else r.continue_())
        pg.goto("file://" + os.path.abspath(page))
        pg.wait_for_timeout(1000)
        pg.evaluate(f"window.__vfife_model.look({ground}, 300)"); pg.wait_for_timeout(200)
        far = pg.evaluate("window.__vfife_model.glyphs()")
        pg.evaluate(f"window.__vfife_model.look({ground}, 30)"); pg.wait_for_timeout(200)
        mid = pg.evaluate("window.__vfife_model.glyphs()")
        pg.evaluate(f"window.__vfife_model.look({other}, 6)"); pg.wait_for_timeout(200)
        mid["codes"] += pg.evaluate("window.__vfife_model.glyphs()")["codes"]
        pg.select_option("#region", label="pile"); pg.wait_for_timeout(200)
        pile_only = pg.evaluate("window.__vfife_model.glyphs()")
        pg.select_option("#region", value="-1"); pg.wait_for_timeout(100)
        pg.click("#codes"); pg.evaluate(f"window.__vfife_model.look({ground}, 300)"); pg.wait_for_timeout(200)
        no_codes = pg.evaluate("window.__vfife_model.glyphs()")
        pg.evaluate(f"window.__vfife_model.look({ground}, 4)"); pg.wait_for_timeout(200)
        x, y = pg.evaluate(f"window.__vfife_model.node_at({ground})")
        pg.mouse.click(x, y); pg.wait_for_timeout(200)
        picked = pg.inner_text("#pick")
        near = pg.evaluate("window.__vfife_model.glyphs()")
        b.close()
    assert not errors, errors
    assert far["symbols"] == n_symbols and far["hidden"] > 0            # all drawn; crowded codes left out
    assert no_codes["labels"] == 0 and no_codes["symbols"] == n_symbols
    assert near["detail"] == 1 and "ux uy uz rx ry rz" in picked
    # the codes say what each dof and component is: ux a spring, uy a damper alone, uz a gap, rx a
    # bilinear law; the global connector's six springs; the supports, held and free
    assert {"KDGN00", "KKKKKK", "111111", "111000", "001000"} <= set(mid["codes"]), mid["codes"]
    assert pile_only["symbols"] < mid["symbols"]                       # the pile's region only
