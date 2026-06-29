"""Assemble the consistency constraints and solve for the forced hypercharges.

The constraints are:
  1. anomaly cancellation   -- every anomaly coefficient = 0
  2. Declared U(1) Yukawa charge sums vanish. Non-abelian singlet contractions
     are assumed by the predefined representation skeleton.
  3. normalization           -- one declared value, fixing the U(1) scale

z3 then either forces a unique assignment, reports a family of solutions, or
declares the spectrum inconsistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import z3

from nsm.anomalies import anomaly_coefficients, witten_anomaly_free
from nsm.charges import gell_mann_nishijima
from nsm.reps import weak_components


def _q(x):
    """A z3 rational numeral from a Fraction/int."""
    fr = Fraction(x)
    return z3.RealVal(fr.numerator) / z3.RealVal(fr.denominator)


@dataclass
class DerivationResult:
    # None means z3 could not decide within the timeout (undecided), as
    # distinct from a definite True/False.
    satisfiable: bool | None
    hypercharges: dict       # name -> Fraction
    charges: dict            # component label -> Fraction
    is_unique: bool | None


@dataclass
class Enumeration:
    solutions: list  # list of {name: Fraction}
    complete: bool   # True iff the search proved no further solutions exist
    reason: str      # "exhausted" | "limit" | "undecided"


def build_constraints(model, omit=()) -> list:
    """The z3 constraints encoding consistency of ``model``.

    ``omit`` is a set of anomaly-coefficient names (e.g. ``{"grav^2-U1"}``) to
    leave out -- useful for probing which constraints are load-bearing vs.
    redundant."""
    cons = []
    if not witten_anomaly_free(model.fermions):
        cons.append(z3.BoolVal(False))
    for name, val in anomaly_coefficients(model.fermions).items():
        if name in omit:
            continue
        # The SU(3)^3 coefficient carries no hypercharge dependence, so it may
        # be a plain Fraction; coerce to a z3 Bool so add() can never silently
        # drop it.
        cons.append(val == 0 if z3.is_expr(val) else z3.BoolVal(val == 0))
    for yuk in model.yukawas:
        total = z3.RealVal(0)
        for name, coeff in yuk.terms:
            total = total + _q(coeff) * model.symbols[name]
        cons.append(total == 0)
    for name, value in model.normalization:
        cons.append(model.symbols[name] == _q(value))
    return cons


def _as_fraction(value) -> Fraction:
    # z3 rational numerals expose .as_fraction(). An algebraic (irrational)
    # value cannot be an SM-style hypercharge; flag it loudly rather than
    # silently mangling it through a string round-trip.
    if z3.is_algebraic_value(value):
        raise ValueError(f"non-rational hypercharge encountered: {value}")
    return value.as_fraction()


def derive_hypercharges(model, timeout_ms: int = 10000, omit=()) -> DerivationResult:
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(*build_constraints(model, omit))

    status = solver.check()
    if status == z3.unsat:
        return DerivationResult(False, {}, {}, None)
    if status == z3.unknown:  # z3 gave up within the timeout
        return DerivationResult(None, {}, {}, None)

    m = solver.model()
    hypercharges = {
        name: _as_fraction(m.eval(sym, model_completion=True))
        for name, sym in model.symbols.items()
    }

    charges = {}
    for f in model.fermions:
        y = hypercharges[f.name]
        for label, t3 in weak_components(f):
            charges[label] = gell_mann_nishijima(t3, y)

    # Uniqueness: does any *different* assignment also satisfy the constraints?
    solver.push()
    solver.add(z3.Or([sym != _q(hypercharges[name])
                      for name, sym in model.symbols.items()]))
    again = solver.check()
    solver.pop()
    if again == z3.unsat:
        is_unique = True       # no other assignment exists
    elif again == z3.sat:
        is_unique = False      # a different assignment also works
    else:
        is_unique = None       # undecided within the timeout

    return DerivationResult(True, hypercharges, charges, is_unique)


def enumerate_solutions(model, limit: int = 16, timeout_ms: int = 10000) -> Enumeration:
    """Distinct hypercharge assignments satisfying ``model``.

    Returns an :class:`Enumeration` whose ``complete`` flag distinguishes "these
    are ALL the solutions" (search exhausted) from "these are SOME" (cut off by
    ``limit``, or z3 gave up -- ``reason`` says which). Never reports a
    truncated family as if it were complete.
    """
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(*build_constraints(model))

    solutions = []
    while len(solutions) < limit:
        status = solver.check()
        if status == z3.unsat:
            return Enumeration(solutions, complete=True, reason="exhausted")
        if status == z3.unknown:
            return Enumeration(solutions, complete=False, reason="undecided")
        m = solver.model()
        sol = {
            name: _as_fraction(m.eval(sym, model_completion=True))
            for name, sym in model.symbols.items()
        }
        solutions.append(sol)
        solver.add(z3.Or([sym != _q(sol[name])
                          for name, sym in model.symbols.items()]))
    return Enumeration(solutions, complete=False, reason="limit")
