"""Gell-Mann-Nishijima: electric charge is DERIVED from weak isospin + hypercharge.

Convention: Q = T3 + Y/2  (so the SM quark doublet has Y = 1/3).
Charge is never stored as a primitive; it falls out of (T3, Y) by arithmetic.
"""
from fractions import Fraction as F

from nsm.charges import gell_mann_nishijima as Q


def test_up_quark_charge_from_t3_and_hypercharge():
    # up-type member of Q_L doublet: T3 = +1/2, Y = 1/3
    assert Q(t3=F(1, 2), y=F(1, 3)) == F(2, 3)


def test_down_quark_charge():
    # down-type member of Q_L doublet: T3 = -1/2, Y = 1/3
    assert Q(t3=F(-1, 2), y=F(1, 3)) == F(-1, 3)


def test_neutrino_is_neutral():
    # neutrino member of L_L doublet: T3 = +1/2, Y = -1
    assert Q(t3=F(1, 2), y=F(-1)) == F(0)


def test_charged_lepton_is_minus_one():
    # charged-lepton member of L_L doublet: T3 = -1/2, Y = -1
    assert Q(t3=F(-1, 2), y=F(-1)) == F(-1)


def test_right_handed_electron_from_singlet_hypercharge():
    # e_R is a weak singlet (T3 = 0) with Y = -2
    assert Q(t3=F(0), y=F(-2)) == F(-1)


def test_result_is_exact_rational_not_float():
    assert isinstance(Q(t3=F(1, 2), y=F(1, 3)), F)
