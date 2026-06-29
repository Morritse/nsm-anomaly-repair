"""Gell-Mann-Nishijima relation.

Electric charge is not a fundamental input of the Standard Model; it is
derived from weak isospin (T3) and weak hypercharge (Y):

    Q = T3 + Y/2

We use the convention in which the left-handed quark doublet Q_L has Y = 1/3.
"""
from fractions import Fraction


def gell_mann_nishijima(t3: Fraction, y: Fraction) -> Fraction:
    """Electric charge of a field component with weak isospin ``t3`` and
    hypercharge ``y``, as an exact rational."""
    return Fraction(t3) + Fraction(y) / 2
