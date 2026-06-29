"""Generic local gauge + gravitational anomaly coefficients.

A chiral gauge theory is quantum-mechanically consistent only if every anomaly
coefficient in the perturbative local anomaly set vanishes. We compute them
from the field content alone, so the same code judges the Standard Model and any
non-standard spectrum. Global checks are separate; see ``witten_anomaly_free``
for the modeled SU(2) parity condition.

Each contribution is weighted by the chirality sign s = +1 (left) / -1 (right),
so a field and its opposite-chirality twin cancel -- exactly the bookkeeping
that is error-prone to do by hand.

The arithmetic is polymorphic: pass Fraction hypercharges and you get exact
Fraction coefficients back; pass z3 expressions and you get z3 expressions,
ready to feed to a solver.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable

import z3

from nsm.reps import Fermion


def _is_z3(x) -> bool:
    return isinstance(x, z3.ExprRef)


def _to_z3(x):
    if _is_z3(x):
        return x
    fr = Fraction(x)
    return z3.RealVal(fr.numerator) / z3.RealVal(fr.denominator)


def _scale(coeff: Fraction, y):
    """coeff * y, where coeff is rational and y may be Fraction or z3."""
    if _is_z3(y):
        return _to_z3(coeff) * y
    return coeff * y


def _sum(terms):
    """Sum terms that may mix Fraction and z3 expressions."""
    if any(_is_z3(t) for t in terms):
        total = z3.RealVal(0)
        for t in terms:
            total = total + _to_z3(t)
        return total
    total = Fraction(0)
    for t in terms:
        total += t
    return total


# Coefficient names, used as keys throughout.
SU3_SQ_U1 = "SU3^2-U1"
SU2_SQ_U1 = "SU2^2-U1"
GRAV_SQ_U1 = "grav^2-U1"
U1_CUBE = "U1^3"
SU3_CUBE = "SU3^3"


def anomaly_coefficients(fermions: Iterable[Fermion]) -> dict:
    """The five local perturbative anomaly coefficients for a list of fermions.
    All must be zero for this local anomaly check to pass."""
    fermions = list(fermions)

    su3_sq_u1, su2_sq_u1, grav_u1, u1_cube, su3_cube = [], [], [], [], []

    # Note on normalization: we use the hypercharge Y directly as the U(1)
    # charge. In the Q = T3 + Y/2 convention the gauged charge is really Y/2, so
    # every U(1) coefficient here differs from the textbook one by an overall
    # constant (1/2 for the linear/mixed coefficients, 1/8 for the cubic). That
    # rescaling does not affect whether a coefficient vanishes, which is all the
    # anomaly conditions require.
    for f in fermions:
        s = f.chirality.sign
        d3, Tc, Ac = f.color.dim, f.color.dynkin, f.color.cubic  # color rep data
        d2, Tw = f.weak.dim, f.weak.dynkin                       # weak rep data
        Y = f.hypercharge
        Y3 = Y * Y * Y  # works for Fraction and z3 alike

        su3_sq_u1.append(_scale(Fraction(s) * d2 * Tc, Y))
        su2_sq_u1.append(_scale(Fraction(s) * d3 * Tw, Y))
        grav_u1.append(_scale(Fraction(s) * d3 * d2, Y))
        u1_cube.append(_scale(Fraction(s) * d3 * d2, Y3))
        su3_cube.append(Fraction(s) * d2 * Ac)

    return {
        SU3_SQ_U1: _sum(su3_sq_u1),
        SU2_SQ_U1: _sum(su2_sq_u1),
        GRAV_SQ_U1: _sum(grav_u1),
        U1_CUBE: _sum(u1_cube),
        SU3_CUBE: _sum(su3_cube),
    }


def gauge_anomaly_coefficients(fermions: Iterable[Fermion]) -> dict:
    """Local gauge + gravitational anomaly coefficients for the gauge group
    SU(3) x SU(2) x U(1)_Y x U(1)_X, using each fermion's hypercharge Y and its
    second-U(1) charge X (``x_charge``). Every coefficient must vanish for this
    perturbative local check to pass -- including the MIXED Y-X coefficients,
    which a single-U(1) check misses. Global anomaly checks are separate.
    """
    fermions = list(fermions)
    acc = {k: [] for k in (
        "SU3^3", "SU3^2-Y", "SU3^2-X", "SU2^2-Y", "SU2^2-X",
        "grav^2-Y", "grav^2-X", "Y^3", "Y^2-X", "Y-X^2", "X^3",
    )}
    for f in fermions:
        s = Fraction(f.chirality.sign)
        d3, Tc, Ac = f.color.dim, f.color.dynkin, f.color.cubic
        d2, Tw = f.weak.dim, f.weak.dynkin
        Y, X = f.hypercharge, f.x_charge

        acc["SU3^3"].append(s * d2 * Ac)
        acc["SU3^2-Y"].append(_scale(s * d2 * Tc, Y))
        acc["SU3^2-X"].append(_scale(s * d2 * Tc, X))
        acc["SU2^2-Y"].append(_scale(s * d3 * Tw, Y))
        acc["SU2^2-X"].append(_scale(s * d3 * Tw, X))
        acc["grav^2-Y"].append(_scale(s * d3 * d2, Y))
        acc["grav^2-X"].append(_scale(s * d3 * d2, X))
        acc["Y^3"].append(_scale(s * d3 * d2, Y * Y * Y))
        acc["Y^2-X"].append(_scale(s * d3 * d2, Y * Y * X))
        acc["Y-X^2"].append(_scale(s * d3 * d2, Y * X * X))
        acc["X^3"].append(_scale(s * d3 * d2, X * X * X))

    return {k: _sum(v) for k, v in acc.items()}


def witten_doublet_count(fermions: Iterable[Fermion]) -> int:
    """Number of SU(2) doublets (each color component counts once). Witten's
    global anomaly requires this to be even."""
    return sum(f.color.dim for f in fermions if f.is_weak_doublet)


def witten_anomaly_free(fermions: Iterable[Fermion]) -> bool:
    """Witten's SU(2) global anomaly is absent iff the number of weak doublets
    is even."""
    return witten_doublet_count(fermions) % 2 == 0
