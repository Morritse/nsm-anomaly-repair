"""The payoff: derive forced facts.

We hand the solver the Standard Model *skeleton* -- which reps each field sits
in, the gauge-invariance of the Yukawa couplings, and one normalization choice
(Y_H = 1) -- with every hypercharge left UNKNOWN. Anomaly cancellation + Yukawa
gauge invariance then force the hypercharges, and electric charges fall out of
Gell-Mann-Nishijima. No charge is ever stored as an input.
"""
from fractions import Fraction as F

from nsm.sm import standard_model_skeleton
from nsm.derive import derive_hypercharges


def test_hypercharges_are_forced_by_consistency():
    result = derive_hypercharges(standard_model_skeleton())
    assert result.satisfiable
    assert result.hypercharges == {
        "Q_L": F(1, 3),
        "u_R": F(4, 3),
        "d_R": F(-2, 3),
        "L_L": F(-1),
        "e_R": F(-2),
        "H": F(1),
    }


def test_electric_charges_fall_out():
    charges = derive_hypercharges(standard_model_skeleton()).charges
    assert charges == {
        "Q_L:up": F(2, 3),
        "Q_L:down": F(-1, 3),
        "u_R": F(2, 3),
        "d_R": F(-1, 3),
        "L_L:nu": F(0),
        "L_L:e": F(-1),
        "e_R": F(-1),
    }


def test_the_full_assignment_is_unique():
    result = derive_hypercharges(standard_model_skeleton())
    assert result.is_unique is True
