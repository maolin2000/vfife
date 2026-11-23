"""Final review, scenario-based: a whole-bridge seismic workflow.

One test per item the review found, each on a model a few elements large:

  M1  a fibre member weighs and vibrates with each fibre's own density
  M2  `modal` / `probe --result` restore the full checkpoint state
  M3  modes and spectrum after a creep run use the instantaneous modulus
  M4  a displacement record is read C1 (continuous support velocity)
  minors: connector-kink warning, residual sampling, reference, damping line,
  info (connectors, mass), box-bars message, describe, sweep help, mesh hint,
  progress wording, plot axes.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _msh(path, nodes, lines, points=None):
    """A Gmsh 2.2 line mesh: `nodes` [(x, y, z)], `lines` {group: [(a, b), ...]}
    (1-based node numbers), `points` {group: [n, ...]}."""
    groups = list(lines) + list(points or {})
    tag = {g: i + 1 for i, g in enumerate(groups)}
    els = []
    for g, pairs in lines.items():
        els += [f"1 2 {tag[g]} {tag[g]} {a} {b}" for a, b in pairs]
    for g, ns in (points or {}).items():
        els += [f"15 2 {tag[g]} {tag[g]} {n}" for n in ns]
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n" % len(groups))
        for g in groups:
            f.write(f'{1 if g in lines else 0} {tag[g]} "{g}"\n')
        f.write("$EndPhysicalNames\n$Nodes\n%d\n" % len(nodes))
        for i, (x, y, z) in enumerate(nodes, 1):
            f.write(f"{i} {x} {y} {z}\n")
        f.write("$EndNodes\n$Elements\n%d\n" % len(els))
        for i, e in enumerate(els, 1):
            f.write(f"{i} {e}\n")
        f.write("$EndElements\n")


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return str(p)


def _solver(case_path):
    from vfife import cli
    from vfife.io import schema
    case, _ = schema.load(case_path)
    mesh, material, element = cli.build(case, case_path)
    return cli.make_solver(cli._with_element(case, element, case_path), mesh, material), case, element


def _cli(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, "-m", "vfife", *args], cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


# ------------------------------------------------------------ the bearing beam
# a 10 m steel fibre beam on a friction slider (left) and a spring (right),
# pushed sideways past the slider's mu N: the slider SLIDES at the end state
BEAM_W = 0.4 * 0.8 * 7850 * 10.0 * 9.80665


def _bearing_beam(tmp_path, solver_block=None, extra="", output_fields="[displacement, rotation, reaction_force]"):
    _msh(tmp_path / "bb.msh",
         [(0, 0, 0), (5, 0, 0), (10, 0, 0), (0, 0, -0.05), (10, 0, -0.05)],
         {"beam": [(1, 2), (2, 3)], "brg_l": [(4, 1)], "brg_r": [(5, 3)]},
         {"base": [4, 5], "mid": [2]})
    solver_block = solver_block or """
        solver: {mode: relaxation, backend: numpy, damping: {type: kinetic}, tolerance: 1.0e-6}
        steps:
          - {name: push, load_steps: 4, max_steps_per_load: 20000}
        response_spectrum:
          spectra: [{direction: [1, 0, 0], table: [[0.0, 5.0], [20.0, 5.0]]}]
          modes: 2
    """
    return _write(tmp_path, "bb.yaml", f"""
        version: 1
        name: bb
        units: {{length: m, force: N, stress: Pa}}
        mesh: {{type: gmsh, path: bb.msh}}
        materials:
          steel: {{type: linear_elastic, E: 2.0e+11, nu: 0.3, rho: 7850}}
        sections:
          girder: {{type: fiber, material: steel, shape: rect, width: 0.4, height: 0.8, ny: 2, nz: 8}}
          slider:
            type: connector
            stiffness: [1.0e+10, 0, 1.0e+9, 0, 0, 0]
            laws:
              uy: {{type: friction, mu: 0.1, normal: ux, stiffness: 1.0e+8}}
          spring: {{type: connector, stiffness: [1.0e+10, 2.0e+6, 1.0e+9, 0, 0, 0]}}
        assign:
          - {{region: beam, section: girder, orientation: [0, 1, 0]}}
          - {{region: brg_l, section: slider, orientation: [1, 0, 0], coordinate_system: local}}
          - {{region: brg_r, section: spring, orientation: [1, 0, 0], coordinate_system: local}}
        constraints:
          - {{region: all, dofs: [0, 1, 0, 1, 0, 1]}}
          - {{region: base, dofs: [1, 1, 1, 1, 1, 1]}}
        loads:
          - {{region: all, gravity: [0, 0, -9.80665]}}
          - {{region: mid, vector: [{0.3 * BEAM_W:.6g}, 0, 0]}}
        {textwrap.dedent(solver_block).strip().replace(chr(10), chr(10) + '        ')}
        {extra}
        output:
          path: {tmp_path / 'bb.h5'}
          every: 1
          fields: {output_fields}
          monitors:
            u: {{kind: displacement, region: mid, dof: x}}
    """)


@pytest.fixture(scope="module")
def bearing_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bb")
    path = _bearing_beam(tmp)
    from vfife import api
    out = api.run_case(path, output=str(tmp / "bb.h5"), progress_every=50)
    assert out.converged, out.message
    return tmp, path, out


# ------------------------------------------------------------------- M1
def test_m1_each_fibre_weighs_with_its_own_material(tmp_path):
    """Two fibre sections and a patch with its own material, and bars: the
    lumped mass is sum(rho_i A_i) L over every fibre, not the first section's
    cover density times the whole area (a pier's own concrete, every bar and
    every tendon used to weigh as the girder's concrete)."""
    _msh(tmp_path / "m.msh", [(0, 0, 0), (5, 0, 0), (10, 0, 0)],
         {"a": [(1, 2)], "b": [(2, 3)]}, {"ends": [1, 3]})
    path = _write(tmp_path, "m.yaml", f"""
        version: 1
        name: m
        mesh: {{type: gmsh, path: m.msh}}
        materials:
          conc: {{type: linear_elastic, E: 3.0e+10, nu: 0.2, rho: 2500}}
          steel: {{type: linear_elastic, E: 2.0e+11, nu: 0.3, rho: 7850}}
        sections:
          bar: {{type: bar, material: steel, area: 1.0e-3}}
          rc: {{type: fiber, material: conc, shape: rect, width: 0.5, height: 1.0, ny: 2, nz: 4,
               bars: [{{y: 0.0, z: -0.4, section: bar, count: 4}}]}}
          comp:
            type: fiber
            material: steel
            shape: patches
            patches:
              - {{corners: [[-1.0, 0.0], [1.0, 0.0], [1.0, 0.25], [-1.0, 0.25]], ny: 4, nz: 2, material: conc}}
              - {{corners: [[-0.01, -1.5], [0.01, -1.5], [0.01, 0.0], [-0.01, 0.0]], ny: 1, nz: 6}}
        assign:
          - {{region: a, section: rc, orientation: [0, 1, 0]}}
          - {{region: b, section: comp, orientation: [0, 1, 0]}}
        constraints:
          - {{region: all, dofs: [0, 1, 0, 1, 0, 1]}}
          - {{region: ends, dofs: [1, 1, 1, 0, 0, 0]}}
        loads:
          - {{region: all, gravity: [0, 0, -9.80665]}}
        solver: {{mode: relaxation, backend: numpy, n_steps: 10}}
        output: {{path: {tmp_path / 'm.h5'}}}
    """)
    s, _, _ = _solver(path)
    m_rc = (0.5 * 1.0 * 2500 + 4e-3 * 7850) * 5.0
    m_comp = (2.0 * 0.25 * 2500 + 0.02 * 1.5 * 7850) * 5.0
    from vfife.core.dof import TRANSLATION
    m = s.dofs.field(s.physical_mass, TRANSLATION)
    assert float(m[:, 0].sum()) == pytest.approx(m_rc + m_comp, rel=1e-12)
    # and before: the first section's 2500 on all of it
    assert float(m[:, 0].sum()) > (0.5 + 4e-3 + 0.5 + 0.03) * 2500 * 5.0 * 1.05
    # the rotary inertia is the density-weighted second moment
    e = 1
    ra = s.fiber_area[e] * s._fiber_rho[e]
    assert s.rho_Iy[e] == pytest.approx(float((ra * s.fiber_z[e] ** 2).sum()), rel=1e-12)


# ------------------------------------------------------------------- M2 + minor 1
def test_m2_modal_at_a_result_restores_the_full_state(bearing_run):
    """`vfife modal --result` read only positions: the friction slider came
    back virgin (and sliding by its whole travel), T1 was not the spectrum's
    at the same state. With the checkpoint restored they are equal, and a
    probe at the converged frame finds it in equilibrium."""
    import h5py
    from vfife import modal
    from vfife.testing.state import probe as make_probe
    tmp, path, out = bearing_run
    with h5py.File(out.result_path) as f:
        T_rs = float(f["response_spectrum/period"][0])
        kinks = f["response_spectrum"].attrs.get("connector_kinks", "")
    s, case, element = _solver(path)
    x, frame, k, full, _missing, note = modal.restore_state(s, case, element, path, out.result_path)
    assert full and note is None
    res = modal.analyse(s, n_modes=2, x=x)
    assert res.periods[0] == pytest.approx(T_rs, rel=1e-6)
    p = make_probe(s)
    r = np.abs(p.residual(x)).max() / np.abs(p.external(x)).max()
    assert r < 1e-3, r
    # minor 1: the sliding slider is named, in the result and by `connector_kinks`
    assert "friction" in kinks and "sliding" in kinks
    assert any("friction" in line for line in modal.connector_kinks(s))


def test_m2_probe_and_modal_cli_say_full_state(bearing_run):
    tmp, path, out = bearing_run
    code, text = _cli("modal", path, "--result", out.result_path, "--modes", "2",
                      "-o", str(tmp / "modes.h5"), cwd=str(tmp))
    assert code == 0, text
    assert "full state from its checkpoint" in text
    assert "warning       connector" in text and "friction" in text


# ------------------------------------------------------------------- M3
def _creep_beam(tmp_path, stations, creep_step=True):
    _msh(tmp_path / "c.msh", [(2.5 * i, 0, 0) for i in range(5)],
         {"beam": [(i, i + 1) for i in range(1, 5)]}, {"left": [1], "right": [5]})
    steps = "  - {name: dead, age: 28.0, load_steps: 1, max_steps_per_load: 40000}\n"
    if creep_step:
        steps += f"  - {{name: creep, to_age: 1028.0, load_steps: {stations}, max_steps_per_load: 40000}}\n"
    return _write(tmp_path, f"c{stations}{int(creep_step)}.yaml", f"""
        version: 1
        name: c{stations}
        units: {{length: m, force: N, stress: Pa}}
        mesh: {{type: gmsh, path: c.msh}}
        materials:
          concrete:
            type: concrete02
            fpc: 3.0e+10
            epsc0: 2.0
            fcu: 6.0e+9
            epscu: 3.0
            ft: 1.0e+8
            Ets: 1.5e+9
            rat: 0.1
            rho: 2500.0
            creep:
              model: curve
              curve: [[0, 0.0], [1, 0.1], [10, 0.5], [100, 1.2], [1000, 1.9], [3000, 2.0]]
        sections:
          girder: {{type: fiber, material: concrete, shape: rect, width: 0.4, height: 0.8, ny: 2, nz: 8}}
        assign:
          - {{region: beam, section: girder, orientation: [0, 1, 0]}}
        constraints:
          - {{region: all, dofs: [0, 1, 0, 1, 0, 1]}}
          - {{region: left, dofs: [1, 1, 1, 0, 0, 0]}}
          - {{region: right, dofs: [0, 1, 1, 0, 0, 0]}}
        loads:
          - {{region: all, gravity: [0, 0, -9.80665]}}
        steps:
        {steps.rstrip().replace(chr(10), chr(10) + '        ')}
        response_spectrum:
          spectra: [{{direction: [0, 0, 1], table: [[0.0, 3.0], [10.0, 3.0]]}}]
          modes: 1
        solver: {{mode: relaxation, backend: numpy, age: 28.0, damping: {{type: kinetic}}, tolerance: 1.0e-7}}
        output: {{path: {tmp_path / f'c{stations}{int(creep_step)}.h5'}, every: 1}}
    """)


def test_m3_spectrum_after_creep_uses_the_instantaneous_modulus(tmp_path):
    """A simply supported creeping beam: its stresses do not change, so its
    vibration does not either. The spectrum at the end of the creep run used
    the last station's 1/(1 + phi_cur) and its period moved with the number
    of stations; now it equals the 28-day period, and the base force agrees
    with the modal mass times Sa."""
    import h5py
    from vfife import api
    T = {}
    for n, creep_step in ((2, True), (6, True), (0, False)):
        path = _creep_beam(tmp_path, n, creep_step)
        out = api.run_case(path)
        assert out.converged, out.message
        with h5py.File(out.result_path) as f:
            g = f["response_spectrum"]
            T[n] = float(g["period"][0])
            base = float(g["base_force"][2])
            mm_sa = float(g["modal_mass"][0, 0] * g["spectral_acceleration"][0, 0])
        assert base == pytest.approx(mm_sa, rel=5e-3), (n, base, mm_sa)
    assert T[2] == pytest.approx(T[0], rel=2e-3)
    assert T[6] == pytest.approx(T[0], rel=2e-3)


# ------------------------------------------------------------------- M4
def test_m4_a_displacement_record_is_read_c1(tmp_path):
    """The support's velocity is continuous at the samples (it jumped at
    every one, an impulse on the support every record step); the samples are
    still interpolated, and the velocity is the derivative of the position."""
    from vfife.core.ground import RecordMotion
    t = np.array([0.0, 0.1, 0.25, 0.3, 0.5, 0.8])
    d = np.array([0.0, 0.01, -0.02, 0.005, 0.03, 0.0])
    p = tmp_path / "d.dat"
    np.savetxt(p, np.c_[t, d])
    r = RecordMotion({"path": str(p), "quantity": "displacement"}, duration=1.0)
    for tk, dk in zip(t[1:], d[1:]):
        assert r(tk) == pytest.approx(dk, abs=1e-15)
    for tk in t[1:-1]:
        # abs: at a sample that is a local extremum the (PCHIP) slope is 0 and
        # the rate passes through it with the curvature, ~1e-8 at +-1e-9 s
        assert r.rate(tk - 1e-9) == pytest.approx(r.rate(tk + 1e-9), rel=1e-5, abs=1e-6)
    for tm in (0.05, 0.2, 0.42, 0.7):
        h = 1e-6
        assert (r(tm + h) - r(tm - h)) / (2 * h) == pytest.approx(r.rate(tm), rel=1e-5)
    # before and after the record as before: at rest, then held
    assert r(-0.1) == 0.0 and r.rate(-0.1) == 0.0
    assert r(0.9) == pytest.approx(d[-1]) and r.rate(0.9) == 0.0
    # an acceleration record is still the stage-146 exact integration
    a = RecordMotion({"path": str(p), "quantity": "acceleration"}, duration=1.0)
    assert a.knot_slope is None


# ------------------------------------------------------------------- minor 3
def test_minor3_relaxation_residual_is_sampled_at_progress_checks(bearing_run):
    import h5py
    tmp, path, out = bearing_run
    with h5py.File(out.result_path) as f:
        n_frames = len(f["monitors/time"])
        n_checks = len(f["monitors/residual"])
    assert n_checks > n_frames


# ------------------------------------------------------------------- minor 4
def test_minor4_reference_documents_response_spectrum():
    from vfife import reference
    text = reference.generate()
    assert "### `response_spectrum` — ResponseSpectrumSpec" in text
    assert "`beyond_table`" in text


# ------------------------------------------------------------------- minor 7, 8
def test_minor7_8_info_prints_damping_connectors_and_weight(tmp_path):
    path = _bearing_beam(tmp_path, solver_block="""
        solver:
          mode: dynamic
          backend: numpy
          total_time: 0.01
          damping: {ratio: 0.05, frequencies: [1.0, 10.0]}
    """)
    code, text = _cli("info", path, cwd=str(tmp_path))
    assert code == 0, text
    assert "damping       rayleigh alpha=" in text and "beta=" in text and "step x" in text
    assert "connector 0" in text and "brg_l" in text and "uy friction" in text
    assert "(2 connectors)" in text
    line = next(l for l in text.splitlines() if l.strip().startswith("mass"))
    weight = float(line.split("weight")[1].split("N")[0])
    assert weight == pytest.approx(BEAM_W, rel=1e-5)        # printed to 6 digits


# ------------------------------------------------------------------- minor 9
def test_minor9_box_bars_message_points_at_patches(tmp_path):
    from vfife.io import model as im
    with pytest.raises(ValueError, match="patches"):
        im.SectionSpec(type="fiber", material="c", shape="box", width=2.0, height=1.0, t=0.2,
                       ny=4, nz=4, bars=[{"y": 0.0, "z": 0.4, "section": "b"}])


# ------------------------------------------------------------------- minor 10
def test_minor10_describe_and_connector_slip_group(tmp_path):
    code, text = _cli("describe", "beam2_fiber")
    assert code == 0, text
    modes = next(l for l in text.splitlines() if "analysis_modes" in l)
    assert "static" in modes
    fields = next(l for l in text.splitlines() if "output_fields" in l)
    for f in ("fiber_stress", "fiber_damage", "connector_force", "connector_slip"):
        assert f in fields
    from vfife import api
    path = _bearing_beam(tmp_path, output_fields="[displacement, connector_slip]")
    prepared = api.prepare_case(path)
    assert prepared.preflight["status"] == "PASS", prepared.preflight["issues"]


# ------------------------------------------------------------------- minor 11, 12
def test_minor11_12_sweep_help_and_mesh_hint(tmp_path):
    code, text = _cli("sweep", "--help")
    assert "output.path" in text and "results/<case-stem>.h5" not in text
    _msh(tmp_path / "l.msh", [(0, 0, 0), (1, 0, 0)], {"beam": [(1, 2)]}, {"left": [1]})
    code, text = _cli("mesh", str(tmp_path / "l.msh"))
    assert code == 0, text
    assert "vfife init" not in text and "write the case by hand" in text


# ------------------------------------------------------------------- minor 13
def test_minor13_progress_says_increment_and_analysis_step(tmp_path):
    path = _bearing_beam(tmp_path)
    code, text = _cli("run", path, "--progress-every", "100", cwd=str(tmp_path))
    assert code == 0, text
    assert "increment" in text and "iterations/s" in text and " 0 steps/s" not in text
    assert "analysis step 1 of 1" in text and "load step(s)" not in text


# ------------------------------------------------------------------- minor 15
def test_minor15_plot_time_axis_says_what_it_measures(bearing_run):
    from vfife.post import Series, _time_axis, plot
    assert "load factor" in _time_axis(Series("time", [0, 1], kind="load_factor"))
    assert _time_axis(Series("time", [0, 1], kind="time")) == "time [s]"
    tmp, path, out = bearing_run
    png = plot(out.result_path, str(tmp / "p.png"))
    assert os.path.getsize(png) > 0
