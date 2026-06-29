"""Two-U(1) gauge anomalies: SU(3)xSU(2)xU(1)_Y x U(1)_X.

When a NEW U(1)' (X) is gauged alongside hypercharge, the full anomaly set
includes the pure-X coefficients AND the MIXED Y-X ones. A repair that ignores
the mixed anomalies could declare a still-anomalous model 'fixed', so we check
all of them.
"""
from fractions import Fraction as F

from nsm.reps import SU3, SU2, Chirality, Fermion
from nsm.anomalies import gauge_anomaly_coefficients

# One SM generation: (name, color, weak, hypercharge Y, chirality).
_SM = [
    ("Q_L", SU3.TRIPLET, SU2.DOUBLET, F(1, 3), Chirality.LEFT),
    ("u_R", SU3.TRIPLET, SU2.SINGLET, F(4, 3), Chirality.RIGHT),
    ("d_R", SU3.TRIPLET, SU2.SINGLET, F(-2, 3), Chirality.RIGHT),
    ("L_L", SU3.SINGLET, SU2.DOUBLET, F(-1), Chirality.LEFT),
    ("e_R", SU3.SINGLET, SU2.SINGLET, F(-2), Chirality.RIGHT),
]
_BL = {"Q_L": F(1, 3), "u_R": F(1, 3), "d_R": F(1, 3), "L_L": F(-1), "e_R": F(-1)}


def _sm_with_x(xmap, nu_R_x=None):
    ferms = [Fermion(n, c, w, y, ch, x_charge=xmap.get(n, F(0)))
             for (n, c, w, y, ch) in _SM]
    if nu_R_x is not None:
        ferms.append(Fermion("nu_R", SU3.SINGLET, SU2.SINGLET, F(0),
                             Chirality.RIGHT, x_charge=nu_R_x))
    return ferms


def test_gauging_hypercharge_itself_is_anomaly_free():
    # X = Y for every field: gauging a copy of hypercharge -> all coefficients
    # vanish (a non-trivial check that the X machinery is right).
    ferms = [Fermion(n, c, w, y, ch, x_charge=y) for (n, c, w, y, ch) in _SM]
    assert all(v == 0 for v in gauge_anomaly_coefficients(ferms).values())


def test_b_minus_l_without_nu_R_fails_only_grav_and_cubic():
    coeffs = gauge_anomaly_coefficients(_sm_with_x(_BL))
    assert coeffs["grav^2-X"] == F(-1)
    assert coeffs["X^3"] == F(-1)
    # the mixed and the SU(N)^2-X anomalies already cancel for B-L
    assert coeffs["Y^2-X"] == 0
    assert coeffs["Y-X^2"] == 0
    assert coeffs["SU3^2-X"] == 0
    assert coeffs["SU2^2-X"] == 0


def test_b_minus_l_with_nu_R_cancels_everything_including_mixed():
    coeffs = gauge_anomaly_coefficients(_sm_with_x(_BL, nu_R_x=F(-1)))
    assert all(v == 0 for v in coeffs.values()), coeffs


def test_a_weird_charge_fails_a_mixed_anomaly():
    # X only on e_R: the mixed Y^2-X anomaly no longer cancels -- exactly the
    # failure mode a single-U(1) check would miss.
    coeffs = gauge_anomaly_coefficients(_sm_with_x({"e_R": F(1)}))
    assert coeffs["Y^2-X"] != 0
