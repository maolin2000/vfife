"""The two campaigns of corroded post-tensioned scaled beams, and their
uncorroded control beams (case_30, case_31).

    python benchmarks/generate_corroded_pc_beam.py            # both meshes
    python benchmarks/generate_corroded_pc_beam.py --check    # the audits

TWO CAMPAIGNS, AND ONLY ONE OF THEM CAN CARRY A CAPACITY CHECK.

    A  Kuang et al., Appl. Sci. 13 (2023) 8396      six beams L1..L6
    B  Li et al.,   Buildings 14 (2024) 3322        eight beams A0..A7

Both scale the same kind of prototype box girder down to a T, both post-
tension ONE 1x7-15.2 strand in a grouted duct to a jacking force fixed by a
12 mm elongation, both then corrode the strand electrically and break the beam
in four-point bending. They are independent: different authors, different
scale (1/15.68 against 1/16.155), different spans, different corrosion levels.

A'S REPORTED ULTIMATE LOADS VIOLATE SECTION EQUILIBRIUM. The tension the
section can develop is bounded by what the steel is:

    strand   140 mm2 x 1872 MPa                 = 262.1 kN
    2 phi 8 HRB335 at the soffit, 430.4 MPa     =  43.3 kN
                                          T_max = 305.4 kN

The beam is 250 mm deep, so no internal lever arm can exceed 0.250 m and

    M_u <= 305.4 kN x 0.250 m = 76.34 kN m

is an absolute ceiling -- it puts the compression resultant on the top surface
and the tension resultant on the soffit, with every bar at its ULTIMATE
strength at once. A's Table 4 reports 76.14 kN m for the uncorroded L1, which
is 99.7% of that ceiling and needs a lever arm of 249.4 mm on a 250 mm beam.
It is not attainable. Three things corroborate:

  * the authors' OWN section analysis in the same table says 49.43 kN m, 35%
    below their own measurement, and they do not remark on it;
  * this library's fibre model says 54.6 kN m -- two independent calculations
    agreeing with each other and not with the number;
  * L1's reported failure mode is concrete crushing, which means the strand
    did NOT reach f_pu, so the true capacity is below the ceiling, not at it.

A's CRACKING loads are a different matter and are used: measured 23.58 kN m
against the authors' predicted 23.06, a 2.3% agreement that also confirms the
0.655 m shear span (23.58/72.0 = a/2). The same lever arm that gives a
consistent cracking moment gives an impossible ultimate one, which is why the
problem is read as a load-scale error and not as a geometry error. A's
DEFLECTIONS are consistent too -- 11.36 mm at yield, 36.5 mm at failure, both
of which the model reproduces at its own (lower) load.

B is self-consistent throughout: 145.0 kN = 2 x 72.5 kN of measured support
reaction, and 72.5 x 0.850 = 61.63 kN m, which needs a lever arm of 0.200 m on
a 240 mm beam. So case_31 (B) carries the flexural-capacity check and case_30
(A) is reduced to a cracking check.

WHAT IS ASSUMED, in both cases: the EFFECTIVE prestress. Both papers give the
jacking force and neither publishes a loss breakdown, and for a 2 m tendon the
anchorage draw-in alone dominates (2 mm of set is 9.6e-4 of strain, 187 MPa,
17%). Each case therefore back-calculates the effective force from its own
measured CRACKING load and says so in its header. That is one calibrated
number per beam; the flexural capacity of a bonded tendon does not depend on
it, so the capacity check stays a free prediction.
"""
from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np

#: 1x7-15.2 seven-wire strand, and the jacking force both papers state
AREA_STRAND = 1.40e-4                            #: m2
P_JACK = 1.57e+5                                 #: N -- 12 mm elongation over ~2 m


class Beam:
    """One campaign's scaled T-beam. All lengths in m, y from the MIDSPAN
    CENTROID, z from the vertical plane of symmetry.

    THE SOFFIT AND THE TOP FACE ARE STRAIGHT, so both sections share one
    y-origin and the SUPPORT section's own centroid sits above it. That is the
    real geometry, not an idealisation: the beam's centroid genuinely rises
    where the web widens, and putting both patch sets on a common datum is
    what reproduces it. (The element axis is then centroidal only at midspan,
    which is where the beam is being bent.)
    """

    def __init__(self, tag, total, span, shear_span, depth, flange_t, flange_b,
                 web_mid, web_sup, taper, tendon_mid, tendon_sup, source):
        self.tag, self.source = tag, source
        self.total, self.span, self.shear_span = total, span, shear_span
        self.overhang = 0.5 * (total - span)
        self.depth, self.flange_t, self.flange_b = depth, flange_t, flange_b
        self.web_mid, self.web_sup, self.taper = web_mid, web_sup, taper
        self.tendon_mid, self.tendon_sup = tendon_mid, tendon_sup
        self.y_bot = -self.section(web_mid)[1]        #: soffit, from the datum
        self.y_top = depth + self.y_bot
        self.y_web_top = self.y_top - flange_t

    # ---------------------------------------------------------------- geometry
    def section(self, web):
        """(area, centroid above the soffit, I about it) of the T."""
        a1, y1 = self.flange_b * self.flange_t, self.depth - 0.5 * self.flange_t
        hw = self.depth - self.flange_t
        a2, y2 = web * hw, 0.5 * hw
        a = a1 + a2
        yc = (a1 * y1 + a2 * y2) / a
        i = (self.flange_b * self.flange_t ** 3 / 12 + a1 * (y1 - yc) ** 2
             + web * hw ** 3 / 12 + a2 * (y2 - yc) ** 2)
        return a, yc, i

    def patches(self, web, ny_web=12, nz_web=4, ny_f=4, nz_f=10):
        """The two quads of the T, in the shared (y, z) frame."""
        hz_f, hz_w = 0.5 * self.flange_b, 0.5 * web
        return [{"corners": [[self.y_web_top, -hz_f], [self.y_web_top, hz_f],
                             [self.y_top, hz_f], [self.y_top, -hz_f]],
                 "ny": ny_f, "nz": nz_f},
                {"corners": [[self.y_bot, -hz_w], [self.y_bot, hz_w],
                             [self.y_web_top, hz_w], [self.y_web_top, -hz_w]],
                 "ny": ny_web, "nz": nz_web}]

    def tendon_profile(self):
        """Draped: flat between the tapers, rising to the supports."""
        y_mid = self.tendon_mid + self.y_bot
        y_sup = self.tendon_sup + self.y_bot
        return [[0.0, y_sup, 0.0], [self.taper, y_mid, 0.0],
                [self.total - self.taper, y_mid, 0.0], [self.total, y_sup, 0.0]]

    def keys(self):
        """The stations the mesh must land on, exactly."""
        return sorted({0.0, self.overhang, self.taper,
                       self.overhang + self.shear_span, 0.5 * self.total,
                       self.total - self.overhang - self.shear_span,
                       self.total - self.taper, self.total - self.overhang,
                       self.total})

    # -------------------------------------------------------------------- mesh
    def write_mesh(self, path, h=0.040):
        """Nodes on the supports, the load points, midspan and both tapers.

        THE KEY POINTS DIVIDE THE MESH; they are not inserted into it. Laying a
        uniform grid down and then adding the key points is the obvious thing
        to do and it is wrong: on beam A, 1.795 lands 5 mm from the grid's
        1.800, and a 5 mm element between 40 mm ones sets the relaxation's
        critical time step for the WHOLE model. The first version of this mesh
        did that at 0.285 and 1.795. What it looked like was not a step-size
        problem: the prestress transfer crawled (2.7e-3 after 40 000
        iterations), raising `relaxation_safety` to 0.9 appeared to fix it, and
        the run then died in the loading stage with its worst residual at node
        49 -- x = 1.7950, to the millimetre. Dividing instead of inserting
        leaves 38..47 mm elements, and the transfer converges in seconds at the
        default 0.6. (v0.4 hit the same thing on the Confederation
        piers; v0.4 is where it cost a day.)
        """
        k = self.keys()
        x = [k[0]]
        for a, b in zip(k[:-1], k[1:]):
            n = max(1, int(round((b - a) / h)))
            x.extend(a + (b - a) * (j + 1) / n for j in range(n))
        x = np.array(x).round(6)
        def at(v):
            return int(np.argmin(np.abs(x - v))) + 1
        pts = {"left": at(self.overhang), "right": at(self.total - self.overhang),
               "load_l": at(self.overhang + self.shear_span),
               "load_r": at(self.total - self.overhang - self.shear_span),
               "mid": at(0.5 * self.total)}
        names = list(pts) + ["web_wide_l", "web_narrow", "web_wide_r"]
        out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
               "$PhysicalNames", str(len(names))]
        out += [f'0 {i + 1} "{nm}"' for i, nm in enumerate(pts)]
        out += [f'1 {len(pts) + i + 1} "{nm}"'
                for i, nm in enumerate(names[len(pts):])]
        out += ["$EndPhysicalNames", "$Nodes", str(len(x))]
        out += [f"{i + 1} {v:.6f} 0 0" for i, v in enumerate(x)]
        out += ["$EndNodes", "$Elements", str(len(pts) + len(x) - 1)]
        eid = 1
        for i, nm in enumerate(pts):
            out.append(f"{eid} 15 2 {i + 1} {i + 1} {pts[nm]}")
            eid += 1
        for i in range(len(x) - 1):
            xm = 0.5 * (x[i] + x[i + 1])
            tag = len(pts) + (1 if xm < self.taper
                              else (3 if xm > self.total - self.taper else 2))
            out.append(f"{eid} 1 2 {tag} {tag} {i + 1} {i + 2}")
            eid += 1
        out += ["$EndElements", ""]
        pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
        d = 1e+3 * np.diff(x)
        return len(x) - 1, d.min(), d.max()


#: A -- Kuang et al. (2023), Figure 2. Total 2080, supports 1890 apart, load
#: points 655 from each support (recovered from M_cr/P_cr = 23.58/72.0). T of
#: 270 x 80 on a web 90 wide at midspan and 160 at the supports, 250 deep; the
#: 192 mm taper is idealised as a step at 285 from each end. Tendon 53 mm above
#: the soffit at midspan and 90 at the supports, read from Figure 2(c,d) at
#: 300 dpi against the 20 mm cover.
KUANG = Beam("A", 2.080, 1.890, 0.655, 0.250, 0.080, 0.270, 0.090, 0.160,
             0.285, 0.053, 0.090, "Kuang et al., Appl. Sci. 13 (2023) 8396")

#: B -- Li et al. (2024), Figures 1-3. Total 2020, supports 1952 apart (34 from
#: each end), load points 850 from each support. T of 265 x 75 on a web 89 wide
#: at midspan and 155 at the supports, 240 deep; the elevation chain is
#: 34|59|186|1462|186|59|34, so the web is full width for 93 mm, tapers over
#: 186, and the step idealisation is placed at 93 + 186 = 279 from each end.
#: Tendon 33 mm above the soffit at midspan and 74 at the supports -- the
#: Figure 3 labels, confirmed by measuring the duct circle at 600 dpi against
#: the section depth (33.3 and 71.8) and, at midspan, by the construction
#: itself: 20 mm bottom cover plus half of the 25 mm duct is 32.5.
LI = Beam("B", 2.020, 1.952, 0.850, 0.240, 0.075, 0.265, 0.089, 0.155,
          0.279, 0.033, 0.074, "Li et al., Buildings 14 (2024) 3322")

BEAMS = {"A": KUANG, "B": LI}

#: Non-prestressed bars, (y above the soffit, |z|, count at that |z|). Both
#: campaigns put 8 phi 8 in the flange and 2 at the soffit flanking the duct.
#: A's flange bars are one layer, B's are two (Figure 3). The flange bars are
#: in compression throughout and their z-spread is drawn, not dimensioned; an
#: even spread over the 225 mm stirrup width is used.
REBAR = {
    "A": [(0.226, 0.1125), (0.226, 0.0804), (0.226, 0.0482), (0.226, 0.0161),
          (0.226, -0.0161), (0.226, -0.0482), (0.226, -0.0804), (0.226, -0.1125),
          (0.024, 0.0220), (0.024, -0.0220)],
    "B": [(0.216, 0.1125), (0.216, 0.0375), (0.216, -0.0375), (0.216, -0.1125),
          (0.181, 0.1125), (0.181, 0.0375), (0.181, -0.0375), (0.181, -0.1125),
          (0.024, 0.0220), (0.024, -0.0220)],
}
AREA_BAR = 5.027e-5                              #: phi 8


# --------------------------------------------------------------------------- #
#  CORROSION, AS PREPROCESSING (decision D2)
# --------------------------------------------------------------------------- #
#  Four laws take one number -- rho, the strand's mass loss -- and produce a
#  DIFFERENT STRAND. None of them is a constitutive law: three change numbers
#  the case file already has (area, E, Fy) and the fourth feeds `steel02`'s
#  `rupture_strain`, which is the one thing the library gained for this.
#  That is the whole of decision D2: corrosion is a
#  preprocessor, not a material, and nothing in `vfife/` knows the word.
#
#  The laws are Kuang et al.'s own (their section 3.1), so the campaign that
#  cannot carry a capacity check still contributes the model that is tested
#  against the campaign that can.
#
#  1. AREA -- a hemispherical pit, their eqs (2)-(6).
#     The average penetration of a wire of diameter d losing a fraction rho of
#     its mass is pa = d(1 - sqrt(1-rho))/2, which is just the uniform answer.
#     Pitting concentrates it: the maximum depth is pm = R * pa with R the
#     pitting factor. Vecchi et al. measure R between 2.93 and 15.05 on
#     corroded strand; Kuang take the mean, R = 9, and so does this. THE CORE
#     WIRE IS NOT ATTACKED and the six outer wires share one pit depth -- their
#     simplification, kept, because a stochastic model of where the pits land
#     is exactly what nobody has.
#
#  2. MODULUS      E(rho)    = (1 - 0.848 rho) E
#  3. STRENGTH     f(rho)    = (1 - 2.683 rho) f          [Yu et al.]
#  4. DUCTILITY    eps_u(rho) = (0.1 + 0.9 exp(-26 rho)) eps_u   [Li]
#
#  TWO DELIBERATE DEPARTURES FROM THEIR PAPER, both so that rho = 0 reproduces
#  the uncorroded beam this library already validated:
#
#  (a) Their eq (10) defines the corroded yield point as eps_py = 0.85 f_ptk/E,
#      a proof-stress convention. At rho = 0 that would move the strand's yield
#      from the MEASURED 1834 MPa (their own Table 3) to 1591, a 13% change
#      with no corrosion in sight. So the measured f_py/f_pu = 0.98 is kept and
#      BOTH strengths are scaled by the same (1 - 2.683 rho). Their law is used
#      for how much strength is lost; their convention for where yield sits is
#      not.
#
#  (b) The hardening ratio `b` is not a free number here: it is whatever makes
#      the strand reach its (corroded) ultimate strength at its (corroded)
#      ultimate strain,
#
#          b = (f_u - f_y) / (E (eps_u - f_y/E))
#
#      At rho = 0 this returns 0.0079 -- which is the 0.008 case_31 was written
#      with, before any of this existed. That agreement is the check that the
#      four laws are wired to the right numbers.
#
#  WHERE THE LAW BREAKS ITS OWN MODEL, AND WHY THAT IS THE INTERESTING PART.
#  Law 4 is savage: at 8.31% of mass loss it leaves 20% of the ultimate strain.
#  Past rho = 7.4% that is BELOW the elastic limit -- the strand can no longer
#  reach its own proof stress before it breaks. The model then has no plastic
#  range at all and `b` is meaningless, so it is set to zero and the strand is
#  elastic to rupture. This is not a numerical guard dressed up as physics: it
#  is the law saying the strand has gone brittle, and Li's beams switch from
#  concrete crushing to wire fracture at about the same place. Whether that
#  switch lands where they measured it is what case_32 asks.
#
#  NOT APPLIED: their eqs (11)-(13), Coronelli's reduction of concrete strength
#  in the TENSION zone from corrosion-induced splitting. It is a smeared-crack
#  device for models that do not crack explicitly -- it lowers the compressive
#  strength of concrete that a fibre model has already cracked and softened
#  fibre by fibre, so applying it here would count the same damage twice. Its
#  size is printed by --check so the decision is visible rather than silent.

D_CORE = 5.15e-3          #: middle wire of a 1x7-15.2 strand, Kuang's eq (6)
D_OUTER = 5.00e-3         #: the six outer wires
R_PIT = 9.0               #: max/mean corrosion depth (Vecchi et al.: 2.93-15.05)
E_STRAND = 1.95e+11       #: Kuang Table 3, measured
FY_STRAND = 1.834e+9      #: ditto
FU_STRAND = 1.872e+9      #: ditto
EPS_U_STRAND = 0.035      #: A_gt, the minimum GB/T 5224 requires of 1x7 strand
MU_RS = 2.78              #: volume ratio of oxide to steel, for eq (13) only


def pit_area(rho, d=D_OUTER, r_pit=R_PIT):
    """Area one hemispherical pit removes from a wire (Kuang eqs 2-5)."""
    if rho <= 0.0:
        return 0.0
    pa = 0.5 * d * (1.0 - np.sqrt(1.0 - rho))
    pm = r_pit * pa
    if pm > d / np.sqrt(2.0):
        raise ValueError(
            "pit depth %.3f mm is past d/sqrt(2) on a %.2f mm wire: that is "
            "Kuang's Figure 5(d) branch, which is not implemented because no "
            "beam in either campaign reaches it (rho would have to exceed 30%%)"
            % (pm * 1e+3, d * 1e+3))
    a = 2.0 * pm * np.sqrt(1.0 - (pm / d) ** 2)
    th1 = 2.0 * np.arcsin(a / d)
    th2 = 2.0 * np.arcsin(min(1.0, a / (2.0 * pm)))
    a1 = 0.5 * (th1 * (0.5 * d) ** 2 - a * abs(0.5 * d - pm ** 2 / d))
    a2 = 0.5 * (th2 * pm ** 2 - a * pm ** 2 / d)
    return a1 + a2


def corroded_strand(rho, prestress_0=None):
    """Everything a case file needs to write ONE corroded strand.

    `prestress_0` is the UNCORRODED effective prestress (Pa). The tendon is
    anchored, so corrosion does not change its length: the strain stays put and
    the stress follows the modulus down. That propagates the one calibrated
    number of the uncorroded beam into every corroded one WITHOUT a second
    calibration -- which is the point.
    """
    ac = pit_area(rho)
    a_full = np.pi / 4.0 * (D_CORE ** 2 + 6.0 * D_OUTER ** 2)
    a_left = np.pi / 4.0 * D_CORE ** 2 + 6.0 * (np.pi / 4.0 * D_OUTER ** 2 - ac)
    out = {
        "rho": rho,
        "pit_depth": R_PIT * 0.5 * D_OUTER * (1.0 - np.sqrt(1.0 - rho)),
        "area_ratio": a_left / a_full,
        "area": AREA_STRAND * a_left / a_full,
        "E": (1.0 - 0.848 * rho) * E_STRAND,
        "Fy": (1.0 - 2.683 * rho) * FY_STRAND,
        "Fu": (1.0 - 2.683 * rho) * FU_STRAND,
        "eps_u": (0.1 + 0.9 * np.exp(-26.0 * rho)) * EPS_U_STRAND,
    }
    eps_y = out["Fy"] / out["E"]
    out["eps_y"] = eps_y
    out["brittle"] = out["eps_u"] <= eps_y
    out["b"] = (0.0 if out["brittle"] else
                (out["Fu"] - out["Fy"]) / (out["E"] * (out["eps_u"] - eps_y)))
    if prestress_0 is not None:
        out["prestress"] = prestress_0 / E_STRAND * out["E"]
    return out


def crack_width(rho, d=D_OUTER):
    """Kuang eq (13): the splitting crack one wire's oxide opens. NOT APPLIED
    (see the block above); --check prints it so the size of what is left out
    is on the record."""
    pa = 0.5 * d * (1.0 - np.sqrt(1.0 - rho))
    return 2.0 / 3.0 * (2.0 * np.pi * (MU_RS - 1.0) * pa)


#: Li et al. (2024) Table 2 -- the eight beams, their measured mass loss and
#: their measured flexural capacity. THE TARGET OF THE DEGRADATION CURVE.
LI_BEAMS = [("A0", 0.0001, 61.63), ("A1", 0.0033, 61.71), ("A2", 0.0124, 59.93),
            ("A3", 0.0237, 56.70), ("A4", 0.0452, 53.72), ("A5", 0.0615, 49.39),
            ("A6", 0.0747, 47.86), ("A7", 0.0831, 44.54)]


def _audit(b, table):
    a_mid, yc_mid, i_mid = b.section(b.web_mid)
    a_sup, yc_sup, i_sup = b.section(b.web_sup)
    print("SECTION of beam %s (%s)" % (b.tag, b.source))
    print("   %-9s %12s %12s %12s %12s" % ("", "area/mm2", "paper", "I/mm4", "paper"))
    for nm, (a, yc, i), (pa, pi) in (("midspan", (a_mid, yc_mid, i_mid), table[0]),
                                     ("support", (a_sup, yc_sup, i_sup), table[1])):
        pa_s = "%12.3e" % pa if pa else "%12s" % "(not given)"
        pi_s = "%12.3e" % pi if pi else "%12s" % "(not given)"
        print("   %-9s %12.3e %s %12.3e %s   centroid %.1f mm above soffit"
              % (nm, a * 1e+6, pa_s, i * 1e+12, pi_s, yc * 1e+3))
    print("   the two centroids differ by %.1f mm; both patch sets share the"
          " midspan datum," % (1e+3 * (yc_mid - yc_sup)))
    print("   so the support section's own centroid sits that far ABOVE the"
          " element axis, as it does in the beam.")


def _ceiling(name, area_p, fpu, area_s, fus, depth, m_test, m_pred, m_model):
    t = area_p * fpu + area_s * fus
    print("\nWHAT THE SECTION CAN DEVELOP  (%s)" % name)
    print("   strand %6.0f mm2 x %6.0f MPa = %6.1f kN"
          % (area_p * 1e+6, fpu / 1e+6, area_p * fpu / 1e+3))
    print("   soffit %6.0f mm2 x %6.0f MPa = %6.1f kN"
          % (area_s * 1e+6, fus / 1e+6, area_s * fus / 1e+3))
    print("                            T_max = %6.1f kN" % (t / 1e+3))
    print("   x the full depth %.3f m (an unreachable lever arm) = %.2f kN m"
          % (depth, t * depth / 1e+3))
    print("   reported  %6.2f kN m   -> lever arm %6.1f mm  (%.0f%% of the ceiling)"
          % (m_test, 1e+3 * m_test * 1e+3 / t, 100 * m_test / (t * depth / 1e+3)))
    if m_pred is not None:
        print("   authors'  %6.2f kN m" % m_pred)
    if m_model is not None:
        print("   this model%6.2f kN m" % m_model)


def _check():
    _audit(KUANG, [(3.69e+4, 1.88e+8), (4.88e+4, 2.65e+8)])
    _ceiling("A, uncorroded L1", AREA_STRAND, 1.872e+9, 2 * AREA_BAR, 4.304e+8,
             KUANG.depth, 76.14, 49.43, 54.59)
    print("   -> A's ultimate loads are NOT usable as a target. Its cracking"
          " loads are:")
    a, yc, i = KUANG.section(KUANG.web_mid)
    print("      measured M_cr 23.58 kN m, authors' prediction 23.06, and"
          " 23.58 / 72.0 = %.4f m" % (KUANG.shear_span / 2))
    print("      recovers the shear span, so the lever arm itself is confirmed.")

    print()
    _audit(LI, [(None, None), (None, None)])
    _ceiling("B, uncorroded A0", AREA_STRAND, 1.872e+9, 2 * AREA_BAR, 4.55e+8,
             LI.depth, 61.63, None, None)
    print("   -> 0.200 m of lever arm on a 240 mm beam: attainable, and B's own")
    print("      table closes (145.0 kN = 2 x 72.5 kN measured reaction,"
          " 72.5 x 0.850 = 61.63).")

    print("\nPRESTRESS: the jacking force is pinned, the effective force is not")
    print("   %.0f kN over %.0f mm2 = %.0f MPa = %.2f f_py (f_py = 1834 MPa)"
          % (P_JACK / 1e+3, AREA_STRAND * 1e+6, P_JACK / AREA_STRAND / 1e+6,
             P_JACK / AREA_STRAND / 1.834e+9))
    print("   -- NOT the 0.72 f_py A's text also states, which would be %.0f kN."
          % (0.72 * 1.834e+9 * AREA_STRAND / 1e+3))
    for b in (KUANG, LI):
        e_mm = P_JACK * b.total / (1.95e+11 * AREA_STRAND) * 1e+3
        print("   beam %s: P L / (E A) = %.2f mm against the stated 12 mm"
              % (b.tag, e_mm))
    print("   so 157 kN and 140 mm2 pin each other and 0.72 f_py is the loose"
          " sentence.")
    print("   Neither paper publishes the LOSSES. Each case back-calculates the")
    print("   effective force from its own measured cracking load and labels it.")


def _corrosion(prestress_0=7.786e+08):
    print("\nCORROSION: one number in, a different strand out")
    print("   R = %.0f, core wire %.2f mm intact, six outer wires %.2f mm"
          % (R_PIT, D_CORE * 1e+3, D_OUTER * 1e+3))
    print("   %-4s %6s %7s %7s %7s %6s %6s %8s %8s %7s %8s"
          % ("beam", "rho%", "pit/mm", "A/A0", "P_e/kN", "E/GPa", "fu", "eps_u",
             "eps_y", "b", "state"))
    for nm, rho, _m in LI_BEAMS:
        c = corroded_strand(rho, prestress_0)
        print("   %-4s %6.2f %7.3f %7.4f %7.1f %6.1f %6.0f %8.5f %8.5f %7.4f %8s"
              % (nm, 100 * rho, c["pit_depth"] * 1e+3, c["area_ratio"],
                 c["prestress"] * c["area"] / 1e+3, c["E"] / 1e+9,
                 c["Fu"] / 1e+6, c["eps_u"], c["eps_y"], c["b"],
                 "BRITTLE" if c["brittle"] else "ductile"))
    c0 = corroded_strand(0.0)
    print("   rho = 0 returns b = %.4f, which is the 0.008 case_31 was written"
          " with before" % c0["b"])
    print("   any of this existed -- the four laws are wired to the right numbers.")
    print("\n   THE LAW GOES BRITTLE BETWEEN A6 AND A7. Solving eps_u(rho) ="
          " eps_y(rho):")
    lo, hi = 0.0, 0.15
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        c = corroded_strand(mid)
        lo, hi = (mid, hi) if not c["brittle"] else (lo, mid)
    print("      rho = %.2f%% is where the strand can no longer reach its own"
          " proof stress" % (100 * hi))
    print("      before it breaks. Li's beams switch from concrete crushing to"
          " wire fracture")
    print("      at about 8%. That coincidence is what case_32 is for.")
    print("\n   NOT APPLIED -- Coronelli's tension-zone concrete (their eqs"
          " 11-13), for size:")
    for nm, rho, _m in (LI_BEAMS[3], LI_BEAMS[-1]):
        w = crack_width(rho)
        eps1 = 6.0 * w / LI.web_mid
        print("      %s  rho %.2f%%  w_cr %.3f mm  eps_1 %.4f  ->  f_ck x %.2f"
              % (nm, 100 * rho, w * 1e+3, eps1, 1.0 / (1.0 + 0.1 * eps1 / 0.002)))
    print("      It lowers the COMPRESSIVE strength of concrete in the TENSION"
          " zone, which a")
    print("      fibre model has already cracked fibre by fibre. Applying it"
          " would count the")
    print("      same damage twice, and the tension zone carries no compression"
          " to lose.")


CASE_31 = pathlib.Path(__file__).resolve().parents[1] / "cases" / \
    "case_31_li_control_beam.yaml"


def write_corroded_case(rho, name, out_dir, template=CASE_31,
                        prestress_0=7.786e+08):
    """One corroded beam, as case_31 with a different strand in it.

    Deriving from the control beam's own file rather than emitting a fresh one
    is the point: every geometry, mesh, load step and monitor is IDENTICAL, so
    the only thing that can move the answer is the four corrosion laws. A
    hand-written corroded case would have a dozen quiet opportunities to differ.
    """
    c = corroded_strand(rho, prestress_0)
    text = template.read_text(encoding="utf-8")
    subs = [
        ("  strand: {type: steel02, E: 1.95e+11, Fy: 1.834e+9, b: 0.008, "
         "R0: 20.0, rho: 7850}",
         "  strand: {type: steel02, E: %.4e, Fy: %.4e, b: %.5f, R0: 20.0,\n"
         "           rho: 7850, rupture_strain: %.5f}   # rho_corr = %.4f"
         % (c["E"], c["Fy"], c["b"], c["eps_u"], rho)),
        ("  strand15: {type: bar, material: strand, area: 1.40e-4}",
         "  strand15: {type: bar, material: strand, area: %.4e}   # %.2f%% of"
         " the pristine area" % (c["area"], 100 * c["area_ratio"])),
        ("        prestress: 7.786e+08",
         "        prestress: %.4e" % c["prestress"]),
        ("name: case_31_li_control_beam", "name: %s" % name),
        ("  path: results/case_31_li_control_beam.h5",
         "  path: results/%s.h5" % name),
    ]
    for old, rep in subs:
        if old not in text:
            raise ValueError("template no longer contains:\n  %s" % old)
        text = text.replace(old, rep)
    # the mesh path is relative TO THE CASE FILE, and the copy may not sit in
    # the same directory as the template
    mesh = (template.parent / "../benchmarks/assets").resolve()
    try:
        rel = pathlib.Path(os.path.relpath(mesh, pathlib.Path(out_dir).resolve()))
    except ValueError:
        # on Windows a copy on another drive (the temp folder on C:, the
        # repository on D:) has no relative path to the mesh: name it whole
        rel = mesh
    text = text.replace("path: ../benchmarks/assets/", "path: %s/" % rel.as_posix())

    head = ("# GENERATED by benchmarks/generate_corroded_pc_beam.py from\n"
            "# cases/case_31_li_control_beam.yaml -- do not edit by hand.\n"
            "# Mass loss rho = %.4f. The strand is the ONLY difference from the\n"
            "# control beam: area x %.4f, E x %.4f, strengths x %.4f, ultimate\n"
            "# strain x %.4f%s.\n"
            % (rho, c["area_ratio"], c["E"] / E_STRAND, c["Fy"] / FY_STRAND,
               c["eps_u"] / EPS_U_STRAND,
               ", and it is BRITTLE (breaks below its own proof stress)"
               if c["brittle"] else ""))
    out = pathlib.Path(out_dir) / ("%s.yaml" % name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + text, encoding="utf-8")
    return out, c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "assets"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--corroded-cases", metavar="DIR",
                    help="write one case per corroded Li beam, derived from "
                         "cases/case_31_li_control_beam.yaml")
    a = ap.parse_args(argv)
    if a.check:
        _check()
        _corrosion()
        return 0
    if a.corroded_cases:
        for nm, rho, m_meas in LI_BEAMS[1:]:
            path, c = write_corroded_case(
                rho, "c31_%s" % nm.lower(), a.corroded_cases)
            print("%s  rho %.2f%%  A %.1f mm2  fu %.0f MPa  eps_u %.5f%s"
                  " -- measured M_u %.2f kN m"
                  % (path, 100 * rho, c["area"] * 1e+6, c["Fu"] / 1e+6,
                     c["eps_u"], "  BRITTLE" if c["brittle"] else "         ",
                     m_meas))
        return 0
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for tag, b in BEAMS.items():
        p = out / ("corroded_pc_beam_%s.msh" % tag)
        n, dmin, dmax = b.write_mesh(p)
        print("%s  %d elements, %.1f..%.1f mm" % (p, n, dmin, dmax))
        print("   midspan patches:", b.patches(b.web_mid))
        print("   support patches:", b.patches(b.web_sup))
        print("   tendon profile:", [[round(v, 5) for v in q]
                                     for q in b.tendon_profile()])
        print("   rebars (y, z) from the datum:",
              [(round(y + b.y_bot, 5), round(z, 4)) for y, z in REBAR[tag]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
