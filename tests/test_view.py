"""`vfife view` (post/view.py): a result file into one page."""
import h5py
import numpy as np
import pytest

from vfife.post import view


def _h5(path, family, nodes, elements, frames, fields_per_frame, monitors=None, failures=None):
    with h5py.File(path, "w") as f:
        f.attrs["element_family"] = family
        f.attrs["case_yaml"] = "name: synthetic_case\n"
        f.create_dataset("mesh/nodes", data=np.asarray(nodes, float))
        f.create_dataset("mesh/elements", data=np.asarray(elements, np.int32))
        for i, (t, fl) in enumerate(zip(frames, fields_per_frame)):
            g = f.create_group(f"steps/{i:08d}")
            g.attrs["time"] = t
            for k, v in fl.items():
                g.create_dataset(k, data=np.asarray(v, np.float32))
        if monitors:
            for k, v in monitors.items():
                f.create_dataset(f"monitors/{k}", data=np.asarray(v, float))
        if failures is not None:
            f.create_dataset("diagnostics/member_failures", data=np.asarray(failures, float))


def test_a_solid_shows_its_exterior_faces_only(tmp_path):
    # two hexa8 side by side: 12 faces, 2 shared -> 10 drawn
    x = [0, 1, 2]
    nodes = [(xi, y, z) for z in (0, 1) for y in (0, 1) for xi in x]
    n = lambda i, j, k: i + 3 * j + 6 * k
    els = [[n(i, 0, 0), n(i + 1, 0, 0), n(i + 1, 1, 0), n(i, 1, 0),
            n(i, 0, 1), n(i + 1, 0, 1), n(i + 1, 1, 1), n(i, 1, 1)] for i in (0, 1)]
    u = np.zeros((12, 3)); u[:, 2] = 0.01 * np.arange(12)
    _h5(tmp_path / "s.h5", "hexa8", nodes, els, [0.0, 1.0],
        [{"displacement": 0 * u, "von_mises": np.zeros(2)}, {"displacement": u, "von_mises": [1.0, 2.0]}])
    d = view.load(str(tmp_path / "s.h5"))
    m = d["meta"]
    assert [p["kind"] for p in m["parts"]] == ["face"]
    assert len(m["parts"][0]["conn"]) == 10
    assert sorted(set(m["parts"][0]["owner"])) == [0, 1]
    act = m["acts"][0]
    names = {(f["on"], f["name"]) for f in act["fields"]}
    assert ("elements", "von_mises") in names and ("nodes", "|u|") in names and ("nodes", "u_z") in names
    vm = next(f for f in act["fields"] if f["name"] == "von_mises")
    assert (vm["lo"], vm["hi"]) == (0.0, 2.0)
    # the displacement survives its 16-bit quantisation to the grid's resolution
    q = np.frombuffer(d["blob"][: 2 * 12 * 3 * 2], dtype=np.int16).reshape(2, 12, 3) * act["scale"]
    assert np.abs(q[1] - u).max() <= act["scale"] / 2 + 1e-12


def test_a_line_model_keeps_its_breaks_and_monitors_and_the_page_is_self_contained(tmp_path):
    nodes = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    els = [[0, 1], [1, 2], [2, 3]]
    frames = [0.0, 0.5, 1.0]
    fl = [{"displacement": np.zeros((4, 3)), "member_failed": [0, 0, 0], "capacity_ratio": [0.2, 0.1, 0.05]},
          {"displacement": np.zeros((4, 3)), "member_failed": [1, 0, 0], "capacity_ratio": [1.0, 0.3, 0.1]},
          {"displacement": np.r_[np.zeros((1, 3)), np.full((3, 3), -0.5)], "member_failed": [1, 0, 0], "capacity_ratio": [1.0, 0.0, 0.0]}]
    _h5(tmp_path / "l.h5", "beam2", nodes, els, frames, fl,
        monitors={"time": [0, 0.5, 1.0], "w": [0, 0, -0.5]}, failures=[[0.4, 0, 1.0, 0]])
    out = view.write(str(tmp_path / "l.h5"), every=1)
    assert out.endswith("l.html")
    html = open(out, encoding="utf-8").read()
    assert "__META__" not in html and "__BLOB__" not in html and "__TITLE__" not in html
    assert "synthetic_case" in html and '"failures":[[0.4,0.0,1.0,0.0]]' in html
    assert '"w":[0.0,0.0,-0.5]' in html
    d = view.load(str(tmp_path / "l.h5"))
    m = d["meta"]
    assert m["parts"][0]["kind"] == "line" and m["acts"][0]["frames"] == 3
    # member_failed is rebuilt in the page from the event list, not stored
    assert m["acts"][0]["failed_from_events"] is True
    assert not any(f["name"] == "member_failed" for f in m["acts"][0]["fields"])
    # and --fields keeps only what is asked for (the displacement components always)
    # (a field the result does not hold is refused since v1.0: tests/test_review4_cli.py)
    names = {f["name"] for f in view.load(str(tmp_path / "l.h5"), fields=["member_failed"])["meta"]["acts"][0]["fields"]}
    assert "capacity_ratio" not in names and "u_z" in names
    # two results of the one model are two acts of one page, each with its own data
    _h5(tmp_path / "l2.h5", "beam2", nodes, els, [0.0, 1.0], fl[:2], failures=[[0.9, 2, 1.0, 0]])
    d2 = view.load([str(tmp_path / "l.h5"), str(tmp_path / "l2.h5")], names=["first", "second"])
    a, b = d2["meta"]["acts"]
    assert (a["name"], b["name"]) == ("first", "second") and (a["frames"], b["frames"]) == (3, 2)
    assert b["offset"] == len(d["blob"]) and len(d2["blob"]) == b["offset"] + 2 * 4 * 3 * 2 + 2 * 3
    stored = [f for f in b["fields"] if "offset" in f]
    assert stored[0]["offset"] >= b["offset"] and b["failures"] == [[0.9, 2.0, 1.0, 0.0]]
    # the displacement's components are not stored: ranges only, derived in the page
    u = next(f for f in a["fields"] if f["name"] == "u_z")
    assert "offset" not in u and u["derived"] == 2 and (u["lo"], u["hi"]) == (-0.5, 0.0)
    nodes2 = nodes + [(4, 0, 0)]
    _h5(tmp_path / "other.h5", "beam2", nodes2, els + [[3, 4]], [0.0], [{"displacement": np.zeros((5, 3))}])
    import pytest
    with pytest.raises(ValueError, match="different mesh"):
        view.load([str(tmp_path / "l.h5"), str(tmp_path / "other.h5")])


def test_a_split_off_member_carries_its_own_connectivity_per_frame(tmp_path):
    # four particles + two spares; the member moves onto the spares in frame 2
    nodes = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 0, 0), (1, 0, 0)]
    els = [[0, 1], [1, 2], [2, 3]]
    conn = [[[0, 1], [1, 2], [2, 3]], [[0, 1], [1, 2], [2, 3]], [[4, 5], [1, 2], [2, 3]]]
    fl = [{"displacement": np.zeros((6, 3)), "connectivity": np.asarray(c, float)} for c in conn]
    _h5(tmp_path / "s.h5", "beam2", nodes, els, [0.0, 0.5, 1.0], fl)
    d = view.load(str(tmp_path / "s.h5"))
    a = d["meta"]["acts"][0]
    # stored as its changes: element 0 moved onto particles 4, 5 at frame 2
    assert a["conn"] is not None and a["conn"]["base"] is None
    assert a["conn"]["changes"] == [[2, 0, 4, 5]]
    # a connectivity that never changes is not stored
    fl2 = [{"displacement": np.zeros((6, 3)), "connectivity": np.asarray(conn[0], float)} for _ in range(3)]
    _h5(tmp_path / "s2.h5", "beam2", nodes, els, [0.0, 0.5, 1.0], fl2)
    assert view.load(str(tmp_path / "s2.h5"))["meta"]["acts"][0]["conn"] is None


def test_only_the_particles_a_run_uses_are_stored(tmp_path):
    # two members on four particles, and four spares nothing ever moved onto
    nodes = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)] + [(0, 0, 0)] * 4
    els = [[0, 1], [2, 3]]
    u = np.zeros((8, 3)); u[:4, 2] = -0.1
    _h5(tmp_path / "p.h5", "beam2", nodes, els, [0.0, 1.0], [{"displacement": 0 * u}, {"displacement": u}])
    d = view.load(str(tmp_path / "p.h5"))
    a = d["meta"]["acts"][0]
    assert a["nodes"]["n"] == 4
    ids = np.frombuffer(d["blob"][a["nodes"]["offset"]:a["nodes"]["offset"] + 16], dtype=np.int32)
    assert ids.tolist() == [0, 1, 2, 3]
    assert a["nodes"]["offset"] == 2 * 4 * 3 * 2                # displacement of 4 particles, 2 frames


def test_frames_are_thinned_but_the_last_is_kept(tmp_path):
    nodes = [(0, 0, 0), (1, 0, 0)]
    frames = [0.1 * i for i in range(30)]
    fl = [{"displacement": np.zeros((2, 3))} for _ in frames]
    _h5(tmp_path / "t.h5", "truss2", nodes, [[0, 1]], frames, fl)
    a = view.load(str(tmp_path / "t.h5"), every=4)["meta"]["acts"][0]
    assert a["frames"] == 9 and a["times"][-1] == frames[-1]
    a = view.load(str(tmp_path / "t.h5"), max_frames=5)["meta"]["acts"][0]
    assert a["frames"] <= 6 and a["times"][-1] == frames[-1]


def test_member_sizes_come_from_the_sections():
    E_steel, E_conc = 2.0e11, 3.0e10
    assert view._section_size({"type": "beam", "shape": "rect", "b": 0.3, "h": 0.6}, E_conc) == (0.6, 0.3, 0.0)
    assert view._section_size({"type": "bar", "shape": "circle", "d": 0.05}, E_steel) == (0.05, 0.05, 1.0)
    assert view._section_size({"type": "connector"}, E_steel) is None
    # general: thin-walled for steel, solid for concrete (told apart by E, not
    # by a density a model smears at will) -- a 5 m deep steel box (I = A h^2 / 4)
    # and a 2 m square concrete pier (I = A h^2 / 12) both read back
    h, A = 5.0, 0.9
    d = view._section_size({"type": "beam", "shape": "general", "area": A, "iy": A * h * h / 4, "iz": A * 13 * 13 / 4}, E_steel)
    assert d[0] == pytest.approx(5.0) and d[1] == pytest.approx(13.0) and d[2] == 0.0
    d = view._section_size({"type": "beam", "shape": "general", "area": 4.0, "iy": 4.0 * 4 / 12, "iz": 4.0 * 4 / 12}, E_conc)
    assert d == pytest.approx((2.0, 2.0, 0.0))
    d = view._section_size({"type": "bar", "shape": "general", "area": np.pi * 0.01 ** 2}, E_steel)
    assert d == pytest.approx((0.02, 0.02, 1.0))
    # no value is not the bottom of the scale: NaN quantises to 255
    q, lo, hi = view._quantise8(np.array([[0.0, np.nan, 2.0]]))
    assert q.tolist() == [[0, 255, 254]] and (lo, hi) == (0.0, 2.0)


def test_real_section_outlines_are_given_per_region(tmp_path):
    """`profiles` (the D1 demo's streamlined boxes and tower legs): the page
    gets each outline once and, per element, which one it draws; a region the
    case does not have, or an outline that is not a polygon, is an error."""
    import os
    case = os.path.join(os.path.dirname(__file__), "..", "cases", "case_34_brittle_cantilever.yaml")
    nodes = [(0.25 * i, 0.0, 0.0) for i in range(5)]
    elements = [(i, i + 1) for i in range(4)]
    h5 = tmp_path / "c.h5"
    _h5(h5, "beam2", nodes, elements, [0.0, 0.1], [{"displacement": np.zeros((5, 3))}] * 2)
    box = {"outline": [[-0.1, -0.2], [0.1, -0.2], [0.1, 0.2], [-0.1, 0.2]]}
    data = view.load(str(h5), case=case, profiles={"beam": dict(box, outboard=True)})
    meta = data["meta"]
    assert meta["profiles"] == [{"outline": box["outline"], "outboard": True}]
    assert meta["profile_of"] == [0, 0, 0, 0]
    assert view.load(str(h5), case=case)["meta"]["profiles"] is None
    with pytest.raises(ValueError, match="no such region"):
        view.load(str(h5), case=case, profiles={"nowhere": box})
    with pytest.raises(ValueError, match="three or more"):
        view.load(str(h5), case=case, profiles={"beam": {"outline": [[0, 0], [1, 1]]}})
    with pytest.raises(ValueError, match="give the case"):
        view.load(str(h5), profiles={"beam": box})
    html = view.render(data)
    assert '"profile_of":[0,0,0,0]' in html and "ExtrudeGeometry" in html


def test_the_page_offers_continuous_colour_and_the_global_axes(tmp_path):
    """Continuous colouring blends two colours per member along its axis
    (instanceColorB, the shader); the axis box and triad label the model's
    own axes in the case's length unit."""
    h5 = tmp_path / "u.h5"
    _h5(h5, "beam2", [(0, 0, 0), (1, 0, 0)], [(0, 1)], [0.0, 0.1], [{"displacement": np.zeros((2, 3))}] * 2)
    with h5py.File(h5, "a") as f:
        f.attrs["case_yaml"] = "name: u\nunits: {length: mm, force: N, stress: MPa}\n"
    data = view.load(str(h5))
    assert data["meta"]["units"] == {"length": "mm"}
    html = view.render(data)
    assert "instanceColorB" in html and 'id="smooth" checked' in html and 'id="axes" checked' in html


def _line_page(tmp_path, **kw):
    nodes = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    fl = [{"displacement": np.zeros((3, 3))}, {"displacement": np.full((3, 3), -0.1)}]
    _h5(tmp_path / "p.h5", "beam2", nodes, [[0, 1], [1, 2]], [0.0, 1.0], fl)
    return view.write(str(tmp_path / "p.h5"), output=str(tmp_path / "p.html"), **kw)


def test_scripts_writes_three_js_into_the_page(tmp_path):
    # v1.16: where the CDNs are not reachable the page was black; with
    # --scripts it needs no network, the two scripts are in it
    lib = tmp_path / "three"
    (lib / "build").mkdir(parents=True)
    (lib / "build" / "three.min.js").write_text('const e="128"; var THREE_MARK = 1, REVISION = e; // </script> inside\n', encoding="utf-8")
    (lib / "OrbitControls.js").write_text("var ORBIT_MARK = 2; THREE.OrbitControls = 0;\n", encoding="utf-8")
    html = open(_line_page(tmp_path, scripts=str(lib)), encoding="utf-8").read()
    assert "THREE_MARK" in html and "ORBIT_MARK" in html
    assert "cdnjs.cloudflare.com/ajax" not in html.split("$(\"note\")")[0]
    assert '<script src="http' not in html
    assert "</script> inside" not in html          # a script's own "</script" cannot end the tag
    # without it, the CDN tags as before
    plain = open(_line_page(tmp_path), encoding="utf-8").read()
    assert all(f'<script src="{u}"></script>' in plain for _n, u in view.SCRIPTS)


def test_scripts_names_what_is_missing_or_the_wrong_three(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="three.min.js"):
        _line_page(tmp_path, scripts=str(tmp_path / "empty"))
    # another three.js: refused by name rather than inlined to fail in the browser
    lib = tmp_path / "other"
    lib.mkdir()
    (lib / "three.min.js").write_text('const e="160"; var REVISION = e;\n', encoding="utf-8")
    (lib / "OrbitControls.js").write_text("import { EventDispatcher } from 'three';\n", encoding="utf-8")
    with pytest.raises(ValueError, match="r128"):
        _line_page(tmp_path, scripts=str(lib))
    assert not (tmp_path / "p.html").exists()      # nothing written


def test_a_page_without_section_outlines_does_not_stop_on_its_first_frame():
    # v1.16: every member part has a `pm` array, empty when the page has
    # no real sections; profileAt read profOf[...] of a null profOf and the
    # page stayed black. The guard must test profOf too.
    from vfife.post.view_page import PAGE
    body = PAGE.split("const profileAt", 1)[1].split("};", 1)[0]
    assert "!profOf" in body


def test_the_page_runs_in_a_browser(tmp_path):
    # the page's script, executed: no JavaScript error and the frame counter
    # set. Needs Playwright with Chromium and three.js r128 in a folder named
    # by VFIFE_THREE_DIR (the CDNs are not used); skipped otherwise
    import os
    three = os.environ.get("VFIFE_THREE_DIR")
    pw = pytest.importorskip("playwright.sync_api")
    if not three:
        pytest.skip("VFIFE_THREE_DIR not set")
    page_path = _line_page(tmp_path, scripts=three)
    errors = []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:                      # no browser installed
            pytest.skip(f"no Chromium: {e}")
        pg = b.new_page()
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.route("**/*", lambda r: r.abort() if r.request.url.startswith("http") else r.continue_())
        pg.goto("file://" + os.path.abspath(page_path))
        pg.wait_for_timeout(1500)
        frame = pg.inner_text("#r-f")
        b.close()
    assert not errors, errors
    assert "/" in frame
