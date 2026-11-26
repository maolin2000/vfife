"""The fibre beam's time-dependent strains (1.22, out of
`fiber_beam_solver`): creep, shrinkage, ageing and relaxation of the
sections' fibres and bars, and the time stations. A mixin of
`FiberBeamSolver`, which is where its state lives."""
from __future__ import annotations
import numpy as np

from . import creep
from . import relaxation


class FiberTimeDependence:
    # ---------------------------------------------------------------- time
    #
    #  Creep and shrinkage on a FIBRE section. The mechanism is
    #  v0.4's, one level down: the beam path subtracts a free deformation
    #  from the member's and divides the member stiffness by (1 + phi_cur);
    #  here the free strain goes into the fibre's own imposed-strain slot and
    #  the (1 + phi_cur) is a SCALE ON THE LAW'S INPUT, which is what
    #  `concrete02.strain_scale` exists for. The material is then
    #  evaluated at the elastic strain itself -- not at an effective modulus --
    #  so it answers with its own nonlinear stress there.
    #
    #  ONE JUDGEMENT, and the plan records it before it was made: the imposed
    #  slot is (nel, n_fibres) and has NO INTEGRATION-STATION INDEX, while a
    #  member's three stations carry different moments and so different creep.
    #  Creep is therefore evaluated ONCE PER ELEMENT, at the middle station.
    #  Not "for now": a draped tendon's prestrain, which varies along the
    #  element for the same reason, is already evaluated at the element
    #  midpoint and uses the same slot. Refining this means giving
    #  `_cpre` a station axis and changing the kernel signature, which is a
    #  different piece of work.
    def _setup_time_dependence(self):
        """The fibre materials' `creep` / `shrinkage` specs, read once.

        A model whose materials declare neither is untouched: `time_dependent`
        is False and nothing below ever runs.
        """
        self._creep_spec = self._shrink_spec = None
        self._creep_who = None
        #: [(fibre mask, spec, concrete age in days when it was stressed), ...]
        #: -- a LIST, not one law, because relaxation has no history: it is a
        #: pure function of the time since stressing, so several tendon steels
        #: in one model cost nothing. Creep cannot say that.
        self._relax = []
        self._prestrain_now = self.fiber_prestrain
        self._history = None
        self._prev_station = None
        self._phi_cur = 0.0
        self._scale_cur = None
        self._scale_el = None
        self._base_scale = {}
        self._creep_cols = None
        self._age0 = float(self.p.solver.age or 28.0)
        self._current_age = self._age0
        self._step_ages = (self._age0, None, "log")
        self.ages = []
        conc = {g for lay in self._layouts for g in lay.conc_groups}
        groups = []
        for g, mat in enumerate(self.materials):
            c, s = getattr(mat, "creep", None), getattr(mat, "shrinkage", None)
            if getattr(mat, "ageing", None) is not None:
                raise ValueError(
                    f"material {mat.name!r} declares `ageing` on a FIBRE "
                    "section, and this solver has none. Ageing changes Ec, "
                    "which on a nonlinear law is not one number -- the peak "
                    "strain and the strength age with it -- so it cannot ride "
                    "the strain scale creep uses. Declare the aged modulus "
                    "directly instead, as cases/case_28d_confederation_sdl.yaml "
                    "does on the beam path, and say in the case which age it is")
            if c is None and s is None:
                continue
            if g not in conc:
                raise ValueError(
                    f"material {mat.name!r} declares `creep` or `shrinkage` "
                    "but its fibres are REINFORCEMENT. Steel does not creep; "
                    "what it does is relax, which is a different law and is "
                    "not in this library (docs/creep.md)")
            if getattr(mat, "strain_scale", None) is None:
                raise ValueError(
                    f"material {mat.name!r} declares `creep` or `shrinkage` "
                    "but has no `strain_scale`, which is how a fibre station "
                    "carries the creep of the interval it is arriving in. "
                    "`concrete02` has one; `linear_elastic` "
                    "does not, and for a linear fibre the same thing is said "
                    "by declaring the modulus you mean")
            groups.append((g, c, s))
        self._read_relaxation(conc)
        if not groups:
            return
        specs = {(repr(sorted((c or {}).items())), repr(sorted((s or {}).items())))
                 for _, c, s in groups}
        if len(specs) > 1:
            raise ValueError(
                "two concrete fibre materials declare DIFFERENT `creep` or "
                "`shrinkage` specs. One law per model in this version -- the "
                "history and the station's own phi are kept once for the "
                "whole model, not per group. Several laws are buildable (the "
                "scale and the imposed strain are already per fibre); no case "
                "needs them yet, so the refusal is deliberate rather than an "
                "oversight")
        self._creep_spec, self._shrink_spec = groups[0][1], groups[0][2]
        if self._creep_spec is not None:
            mat = self.materials[groups[0][0]]
            # a dynamic run carries creep only on a clock that stands still:
            # the history an initial_state restored is then a frozen imposed
            # strain; a step that moves the clock is refused
            # (v1.0: `to_age` here, a moved `age` in `step_data`)
            self._creep_who = f"material {mat.name!r}"
            if self.p.solver.mode == "dynamic":
                for st in self.p.get("steps") or []:
                    if st.get("to_age") is not None:
                        creep.refuse_moving_clock(st, (float(st.get("age") or self._age0),
                                                       float(st["to_age"]), "log"), None,
                                                  self._creep_who)
            # mc90's phi is referenced to Ec28; `ageing` is
            # refused above, so it is the spec's own `E28`
            self._creep_spec = creep.bind_reference(
                self._creep_spec, self._modulus(mat), None, f"material {mat.name!r}")
        self._base_scale = {g: float(self.materials[g].strain_scale)
                            for g, _, _ in groups}
        # which fibre COLUMNS creep, in the section-row order of `fiber_y`
        self._creep_cols = np.zeros(self.fiber_y.shape, dtype=bool)
        for k, lay in enumerate(self._layouts):
            cols = np.concatenate([self.group_indices[g] for g, _, _ in groups
                                   if g in lay.conc_groups] or [np.zeros(0, int)])
            if len(cols):
                self._creep_cols[np.ix_(self._rows[k], cols)] = True
        if self._creep_spec is not None:
            self._history = creep.CreepHistory(self._creep_spec)
        #: the integration station(s) creep is read at -- the element's
        #: middle. `gauss_s` is s/L in [0, 1]: the middle is 0.5. It read
        #: argmin(|s|), which is the END station s = 0 of every Lobatto rule --
        #: a cantilever's creep came out 20% high (found in review,
        #: v1.0). An EVEN rule (gauss2, the default, and lobatto4)
        #: has no station there: its two middle stations straddle it
        #: symmetrically and their mean is the strain at the middle. Reading
        #: the first of them alone read 0.21 L from the element's
        #: own node 0, so the same cantilever crept +6.7% fixed at one end
        #: and -5.9% fixed at the other (found in review)
        d = np.abs(np.asarray(self.gauss_s, dtype=float) - 0.5)
        self._creep_stations = tuple(int(i) for i in np.flatnonzero(d <= d.min() + 1e-12))
        self._creep_station = self._creep_stations[0]

    def _read_relaxation(self, conc):
        """`relaxation` on the steels, one entry per tendon fibre group.

        It does nothing at all on a fibre whose prestrain is zero, so a
        material a section uses for BOTH tendons and ordinary bars is fine and
        needs no separate declaration. A spec that would touch no prestrained
        fibre at all is refused instead, because that is the case where it
        would silently do nothing.
        """
        for g, mat in enumerate(self.materials):
            spec = getattr(mat, "relaxation", None)
            if spec is None:
                continue
            if g in conc:
                raise ValueError(
                    f"material {mat.name!r} declares `relaxation` but its "
                    "fibres are CONCRETE. Concrete's version of this is "
                    "`creep`, and the two are different laws -- relaxation "
                    "here is the prestressing steel losing the strain it was "
                    "stretched to")
            relaxation.validate(dict(spec))
            mask = np.zeros(self.fiber_y.shape, dtype=bool)
            for k, lay in enumerate(self._layouts):
                if g in lay.rebar_groups:
                    mask[np.ix_(self._rows[k], self.group_indices[g])] = True
            mask &= self.fiber_prestrain != 0.0
            if not mask.any():
                raise ValueError(
                    f"material {mat.name!r} declares `relaxation` but no fibre "
                    "of it carries a PRESTRAIN, so there is nothing to relax "
                    "and the declaration would do nothing. Relaxation is the "
                    "loss of a tendon's initial strain; an ordinary bar's "
                    "stress comes from the structure and is not held")
            # default solver.age, and not after the analysis starts
            # (v1.0: relaxation.stressed_at)
            t0 = relaxation.stressed_at(spec, self._age0, self.p.get("steps") or [],
                                        f"material {mat.name!r}")
            self._relax.append((mask, dict(spec), t0))

    def _shifted(self) -> bool:
        """Whether a member is on a clock of its own (1.8): one born in an
        analysis step at its `activate_age` (core/analysis_steps)."""
        shift = self.staging.age_shift
        if shift is None:
            return False
        return bool(np.any(shift) or np.isfinite(self.staging.age_floor).any())

    @property
    def time_dependent(self):
        return getattr(self, "_creep_spec", None) is not None \
            or getattr(self, "_shrink_spec", None) is not None \
            or bool(getattr(self, "_relax", ()))

    def _time_strain(self, age):
        """(nel, n_fibres) of MINUS the free strain creep and shrinkage have
        imposed by `age`, zero on every fibre that does not creep.

        Minus, because what goes in the imposed slot is what the MATERIAL
        sees added to the section's plane strain, and a material sees the
        mechanical strain -- the same sign a temperature uses.

        Only the increments ALREADY RECORDED appear here. The one this station
        is about to produce is carried by `_age_scale` instead, because it is
        not known until the station is solved.
        """
        out = np.zeros(self.fiber_y.shape)
        shifted = self._shifted()
        if self._shrink_spec is not None:
            if shifted:
                # a segment born at `floor` shrank in its formwork before
                # then, which is no load: its shrinkage counts from its birth
                # (the frames subtract it from the birth offset, 1.2)
                for rows, sh, floor in self._age_groups(np.arange(self.nel)):
                    eps = creep.shrinkage_strain(self._shrink_spec, max(age + sh, floor))
                    if np.isfinite(floor):
                        eps -= creep.shrinkage_strain(self._shrink_spec, floor)
                    if eps:
                        out[rows] -= eps
            else:
                eps = creep.shrinkage_strain(self._shrink_spec, age)
                if eps:
                    out -= eps
        if self._history is not None:
            c = (creep.shifted_strain(self._history, age, self._age_groups) if shifted
                 else self._history.strain(age))
            if c is not None:
                out -= c
            # the current increment creeps from where the last station left
            # off, so its own creep is `phi_cur * (e - e_prev)`: the `e` half
            # is the scale in `_age_scale` and the `e_prev` half is an imposed
            # strain the other way
            phi = np.asarray(self._phi_cur, dtype=float)
            if np.any(phi) and self._history._last is not None:
                out += (phi[:, None] if phi.ndim else phi) * self._history._last
        return np.where(self._creep_cols, out, 0.0)

    def _age_scale(self):
        """`1 / (1 + phi_cur)` on the creeping fibres' own input.

        With the imposed strain of `_time_strain`, a fibre then satisfies

            eps_el (1 + phi_cur) = eps_section - eps_free_past + phi_cur eps_prev

        which is the increment's own creep solved for TOGETHER with it -- the
        implicit step. Leaving it out makes the scheme explicit, and on a held
        strain that is not merely first order but marginally UNSTABLE
        (v0.4 measured it; case_30d checks it here).

        Both paths have to be written: the reference path reads
        `mat.strain_scale` on every call (`numba_params` is rebuilt each
        time), the fused kernel reads column 7 of its own `_cpar`. A scale
        that reached only one of them would be invisible until a case ran, and
        would then look like a creep result.
        """
        phi = np.asarray(self._phi_cur, dtype=float)
        if phi.ndim:
            # one factor per member (1.8): each segment on its own clock. The
            # reference path sets the law's scale member by member
            # (`_scale_el`, read in `_internal_force_true`), the kernel reads
            # its per-member `_cpar`
            f_el = 1.0 / (1.0 + phi)
            if isinstance(self._scale_cur, np.ndarray) and self._scale_cur.shape == f_el.shape \
                    and np.all(np.abs(f_el - self._scale_cur) < 1e-14):
                return
            self._scale_cur = f_el
            self._scale_el = f_el
            for g, base in self._base_scale.items():
                self.materials[g].strain_scale = base
            if getattr(self, "_cpar", None) is not None:
                for k, lay in enumerate(self._layouts):
                    c_idx = self._c_order[k]
                    keep = self._creep_cols[self._rows[k][0], c_idx]
                    cols = np.flatnonzero(keep)
                    if len(cols):
                        base = np.array([self._base_scale[g]
                                         for g in self._c_group[k][cols]])
                        rows = self._rows[k]
                        self._cpar[np.ix_(rows, cols, [7])] = \
                            (base[None, :] * f_el[rows][:, None])[:, :, None]
            return
        f = 1.0 / (1.0 + float(phi))
        self._scale_el = None
        if self._scale_cur is not None and not isinstance(self._scale_cur, np.ndarray) \
                and abs(f - self._scale_cur) < 1e-14:
            return
        self._scale_cur = f
        for g, base in self._base_scale.items():
            self.materials[g].strain_scale = base * f
        if getattr(self, "_cpar", None) is not None:
            for k, lay in enumerate(self._layouts):
                c_idx = self._c_order[k]
                keep = self._creep_cols[self._rows[k][0], c_idx]
                cols = np.flatnonzero(keep)
                if len(cols):
                    base = np.array([self._base_scale[g]
                                     for g in self._c_group[k][cols]])
                    self._cpar[np.ix_(self._rows[k], cols, [7])] = \
                        (base * f)[None, :, None]

    def hold_time_strain(self) -> bool:
        """Hold the creep and shrinkage reached as an imposed strain at the
        instantaneous modulus -- the state a dynamic run from here starts in
        -- for a tangent that is asked for vibration (modes, a response
        spectrum).

        The last time station leaves `1 / (1 + phi_cur)` on the law's input:
        that is the IMPLICIT creep step of the interval just solved, a
        sustained-load stiffness, and the modes and spectrum at the end of a
        creep run were taken from it -- periods 20-35% long, and longer the
        fewer stations the run used. With the station recorded, phi_cur = 0
        and the history's own strain give the same elastic strain in every
        fibre (the equilibrium does not move) at the true stiffness.

        Returns True when the solver is time-dependent and the state is a
        recorded station (or has no creep history to hold), False when there
        is nothing to hold or the state is between stations.
        """
        if not self.time_dependent:
            return False
        age = float(self._current_age)
        prev = self._prev_station
        if self._creep_spec is not None and prev is not None and age > prev * (1 + 1e-12) + 1e-12:
            return False
        keep = self._step_ages
        self._step_ages = (age, None, "log")
        try:
            self._refresh_imposed(1.0)
        finally:
            self._step_ages = keep
        return True

    def _record_time_station(self, age):
        """Called once a time station has reached equilibrium: the ELASTIC
        strain every creeping fibre settled at goes into the history, and
        everything after it creeps from there.

        `epsP` -- state column 2 of `concrete02` -- is exactly that and needs
        no correction: it is the strain the MATERIAL saw, which is already
        `(eps_section - eps_free) / (1 + phi_cur)` because the scale was
        applied to the law's input. The beam path has to divide by
        `(1 + phi_cur)` by hand at this point; here the division has already
        happened, in the one place that could not forget it.
        """
        if self._history is None:
            return
        elastic = np.zeros(self.fiber_y.shape)
        gi = list(getattr(self, "_creep_stations", (self._creep_station,)))
        for k, lay in enumerate(self._layouts):
            rows = self._rows[k]
            for g in lay.conc_groups:
                if g not in self._base_scale:
                    continue
                st = self.fiber_committed[g]
                if st.size:
                    # the middle of the element: one station, or the mean of
                    # the two that straddle it
                    elastic[np.ix_(rows, self.group_indices[g])] = \
                        st[rows][:, gi][..., 2].mean(axis=1)
        built = getattr(self, "member_built", None) if self.plan is not None else None
        if built is not None:
            elastic[~np.asarray(built, bool)] = 0.0      # a member not built has no history (1.8)
        self._history.add(creep.station_date(age, self._prev_station),
                          np.where(self._creep_cols, elastic, 0.0))
        if self.staging.age_shift is not None:
            # the interval itself, so that a member on its own clock files the
            # increment where its station's phi dated it (1.2, the frames')
            if not hasattr(self._history, "windows") or len(self._history.windows) != len(self._history.ages) - 1:
                self._history.windows = [(None, a) for a in self._history.ages[:-1]]
            self._history.windows.append((self._prev_station, float(age)))
        self._prev_station = age
