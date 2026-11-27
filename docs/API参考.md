# vfife Python API 参考（自动生成，API 版本 1，1.23.0）

> 由 `python -m vfife reference --api` 从 `vfife/api.py` 的签名和文档字符串生成；`tests/test_api_v1.py` 保证它与代码一致。**不要手改**——改文档字符串，再重新生成。

`vfife.api.__all__` 里的名字就是 API：1.x 之内签名和含义不变，只会加带默认值的参数；包里其余的一切都是内部实现，可能改变。概念和完整例子见 `docs/用户手册.md` 12.6。

## 模块说明

```text
The Python API, version 1: what a script, a notebook or a
front end uses to run vfife.

The names in `__all__` are the API. Within 1.x they keep their signatures
and meaning -- arguments may be added, with defaults, and nothing is taken
away; everything else in the package is internal and may change. The
reference is generated from these docstrings (`python -m vfife reference
--api`, docs/API参考.md), and a test keeps it in step.

What the functions raise. A mistake in the input -- a name
that is not there, a refused value, a file that is not a case -- is a
`UserError` (a `ValueError` with `code`, `message`, `hint`; the codes are in
docs/error_codes.md). A case file that does not fit the schema raises
pydantic's `ValidationError` (also a `ValueError`); a missing file
`FileNotFoundError`; a law refusing its own numbers a plain `ValueError`.
Catching `ValueError` and `OSError` catches every input mistake. A run that
fails numerically is not raised: `run_case` returns it as an outcome; a run
stopped on purpose raises `AnalysisInterrupted`.

    from vfife import api
    case = api.load_case("cases/case_24g_hzb_static_pushover.yaml")
    rows = api.study(case, {"sections.girder.j": [20.0, 29.5, 40.0]},
                     output_dir="results/j_study",
                     extract=lambda o: {"k_peak": 20 * (max(o.times) - 1)})
```

## 名字一览

| 名字 | 种类 | 一句话 |
|---|---|---|
| `API_VERSION` | 常量 | `1` |
| `load_case` | 函数 | A case file as a `CaseDraft`, to change and run. |
| `CaseDraft` | 类 | A case as data: read from a file, changed by dotted keys, run again. |
| `prepare_case` | 函数 | Validate, import the mesh and run preflight, without building a solver. |
| `run_case` | 函数 | Run one analysis; returns an `AnalysisOutcome`, never exits. |
| `study` | 函数 | Run a case once per set of parameter values. |
| `PreparedAnalysis` | 类 | A case validated, its mesh imported and preflighted, no solver built yet (`prepare_case`). |
| `AnalysisOutcome` | 类 | What `run_case` returns: never an exit, whatever happened. |
| `case_schema` | 函数 | JSON Schema of a case file as it is written -- materials, sections, assign and the rest -- for GUI property editors, external case generators and text editors (`vfife schema` writes it to a file); every shipped case validates against it. |
| `read_monitors` | 函数 | Every monitor of a result file, {name: Series}; a `Series` has `name`, `values` (a float array), `kind`, `regions`, `dof`, `unit` (the dataset's `units` attribute; None when the case declared no units) and `units` (the same, "" when none). |
| `verify` | 函数 | A result checked against a verification contract (a YAML file): {"status": PASS/FAIL, "checks": [...]} -- see docs/用户手册.md 13.3. |
| `AnalysisInterrupted` | 类 | A run that was stopped on purpose, after its output was preserved. |
| `UserError` | 类 | A mistake in the input, reported as a sentence rather than a traceback. |

## `load_case(path: Union[str, Path]) -> CaseDraft`

```text
A case file as a `CaseDraft`, to change and run.
```

## `CaseDraft`

`CaseDraft(data: Mapping[str, Any], base: Union[str, Path, None] = None, name: Optional[str] = None)`

```text
A case as data: read from a file, changed by dotted keys, run again.

It remembers the folder the case came from, so a changed copy still
finds its mesh, records and initial state by the paths the file wrote
-- which a plain dict handed to `run_case` does not (its relative paths
resolve against the working directory). `output.path` is the one path
that is not the case's: as for `vfife run`, it is relative to the
working directory.

    case = load_case("pc_beam.yaml")             # the manual's chapter 3 case
    case.get("materials.concrete.fpc")           # 4.0e7
    stiffer = case.with_changes({"materials.concrete.fpc": 5.0e7,
                                 "loads[0].line_load[2]": -3.0e4})
    run_case(stiffer, output="results/pc_beam_fpc50.h5")

Keys are dotted paths through the YAML; `[i]` (or `.i`) indexes a list.
Only the LAST part of a key may be new -- `solver.threads` on a solver
block that does not have one -- so a misspelt section or material name
is an error here, naming what exists, and a misspelt field is refused
by the schema when the draft is prepared.
```

### `CaseDraft.get(self, key: str) -> Any`

```text
The value at a dotted key.
```

### `CaseDraft.path`（属性）

```text
Where the case would be, for resolving its relative paths: in
`base`, under the case's own name. The file need not exist. The
name is only a label: a `/` in it no longer moves the folder.
```

### `CaseDraft.set(self, key: str, value: Any) -> CaseDraft`

```text
Change one value in place; returns the draft, so calls chain.
```

### `CaseDraft.to_yaml(self) -> str`

```text
The draft as YAML text (comments are not kept: it is data).
```

### `CaseDraft.with_changes(self, changes: Mapping[str, Any], name: Optional[str] = None) -> CaseDraft`

```text
A copy with `changes` ({dotted key: value}) applied; this draft is
left as it was.
```

### `CaseDraft.write(self, path: Union[str, Path]) -> Path`

```text
The draft as a case file. Relative paths in it are written as
they are, so write it into `base` (or a folder beside the mesh) if
it is to be run from the file.
```

## `prepare_case(source: str | Path | dict | schema.Case | CaseDraft) -> PreparedAnalysis`

```text
Validate, import the mesh and run preflight, without building a
solver. `source` is a case file, a `CaseDraft`, a dict (its relative
paths then resolve against the working directory) or a `schema.Case`.
```

## `run_case(source, *, output: str | None = None, restart: str | None = None, on_ready: Callable | None = None, on_progress: Callable | None = None, on_frame: Callable | None = None, progress_every: int = 500) -> AnalysisOutcome`

```text
Run one analysis; returns an `AnalysisOutcome`, never exits.

`source` is anything `prepare_case` takes, or its `PreparedAnalysis`.
`output` overrides the case's `output.path`; `restart` continues from a
result's checkpoint. `on_ready(solver, prepared)` is called once the
solver is built, `on_progress(step, n, residual, kinetic, extra)` every
`progress_every` steps (return False to stop, keeping the result), and
`on_frame(step, t, fields)` at every output frame. A numerical failure
comes back as the outcome's status and message, with its diagnostic
result written. A stop -- `on_progress` returning False, Ctrl-C or
SIGTERM, honoured at the next progress check -- raises
`AnalysisInterrupted` after the result and checkpoint are saved;
its `result_path`, `sidecars` and `report` (the diagnostics) say what
was kept, and `--restart` / `restart=` resumes from it. The checkpoint
is its own file beside the result, `<result>.ckpt.h5`,
and a restart or an initial_state from another model is refused
(`CHECKPOINT_MISMATCH`).
```

## `study(case: Union[str, Path, CaseDraft], changes, *, output_dir: Union[str, Path], combine: str = 'product', extract: Optional[Callable[[AnalysisOutcome], Any]] = None, on_run: Optional[Callable[[int, Dict[str, Any]], None]] = None, quiet: bool = True) -> List[Dict[str, Any]]`

```text
Run a case once per set of parameter values.

`changes` is {dotted key: [values]} -- every combination is run
(`combine="product"`), or the lists are taken side by side
(`combine="zip"`, equal lengths) -- or an explicit list of
{dotted key: value} dicts, one per run. Each run writes
`output_dir/<name>_<nnn>.h5`; its row records the index, the values,
the status, whether it converged, the result path, the wall time and,
with `extract(outcome)`, whatever that returns (a dict is merged into
the row, anything else goes under "value"). A run that raises is
recorded with its message and the study goes on; an `extract` that
raises leaves the run's own columns as they were and puts its message
under "extract_error". A misspelt key or field name is refused before
the first run. The rows are returned
and written to `output_dir/study.csv`; `pandas.DataFrame(rows)` makes
them a table. `on_run(index, values)` is called before each run; with
`quiet` (the default) the solver's progress lines are not printed.
```

## `PreparedAnalysis`

| 字段 | 类型 | 默认 |
|---|---|---|
| `case` | `schema.Case` | — |
| `raw_case` | `str` | — |
| `case_path` | `str \| None` | — |
| `mesh` | `Any` | — |
| `material` | `Any` | — |
| `element` | `str` | — |
| `preflight` | `dict` | — |

```text
A case validated, its mesh imported and preflighted, no solver built
yet (`prepare_case`). `preflight["status"]` is PASS or FAIL (warnings
leave it PASS) and `preflight["issues"]` lists every error and warning; hand the object to `run_case` so the
mesh is not read twice.
```

## `AnalysisOutcome`

| 字段 | 类型 | 默认 |
|---|---|---|
| `status` | `str` | — |
| `converged` | `bool` | — |
| `result_path` | `str \| None` | — |
| `sidecars` | `list[str]` | — |
| `preflight` | `dict` | — |
| `diagnostics` | `dict \| None` | `None` |
| `monitors` | `dict[str, list[float]] \| None` | `None` |
| `wall_time` | `float \| None` | `None` |
| `message` | `str \| None` | `None` |
| `times` | `list[float] \| None` | `None` |

```text
What `run_case` returns: never an exit, whatever happened.

`status` is the diagnostics' word (CONVERGED, COMPLETED_DYNAMIC,
MAX_STEPS, NONFINITE, PREFLIGHT_FAILED, ...), `converged` whether the
result is an answer; `result_path` the HDF5 file (written even for a
failed run, with the diagnostics and the last state), `monitors` every
monitor's record by name and `times` the instant (or load factor) of
each record; `message` the failure, if any.
```

### `AnalysisOutcome.as_dict(self)`

```text
Every field as plain data, for JSON.
```

## `case_schema() -> dict`

```text
JSON Schema of a case file as it is written -- materials, sections,
assign and the rest -- for GUI property editors, external case
generators and text editors (`vfife schema` writes it to a file);
every shipped case validates against it. A number written
`2e11` is a string to YAML 1.1 and a number to vfife: validate what
`yaml.safe_load` returns after turning such strings into numbers.
```

## `read_monitors(path)`

```text
Every monitor of a result file, {name: Series}; a `Series` has
`name`, `values` (a float array), `kind`, `regions`, `dof`, `unit` (the
dataset's `units` attribute; None when the case declared no units) and
`units` (the same, "" when none). `time`, `residual` and `kinetic_energy` (and `age`)
are Series too. The result's format is checked first.
```

## `verify(contract, result) -> dict`

```text
A result checked against a verification contract (a YAML file):
{"status": PASS|FAIL, "checks": [...]} -- see docs/用户手册.md 13.3.
The contract comes first; the two given the other way round are refused
as `VERIFY_ARGUMENTS` (it was a UnicodeDecodeError).
```

## `AnalysisInterrupted`

`AnalysisInterrupted(result_path, sidecars, report, signal=None)`

```text
A run that was stopped on purpose, after its output was preserved.

`signal` is the signal that stopped it (`signal.SIGINT`, `signal.SIGTERM`),
or None when `on_progress` returned False. A batch script that catches
`Exception` around `run_case` should look at it: a SIGTERM is the
machine's time limit or a shutdown asking the PROCESS to end, and the
library does not end it for the script; `vfife run`
exits 143 on it, 130 on Ctrl-C.
```

## `UserError`

`UserError(code: str, message: str, hint: str = '')`

```text
A mistake in the input, reported as a sentence rather than a traceback.

A `ValueError` subclass on purpose: the input paths already raised plain
ValueErrors and several callers catch them, so inheriting keeps every
existing handler working while adding the code/hint a caller can act on.

Carries the three things a caller needs to act, whether that caller is a
terminal or a dialog box: a stable `code` to branch on, a `message` saying
what is wrong, and a `hint` saying what to do about it.  A GUI can render
the same object; before this, `api.prepare_case` raised a bare `KeyError`
and the CLI printed twenty lines of stack for a typo.
```

### `UserError.as_dict(self) -> dict`

（无说明）
