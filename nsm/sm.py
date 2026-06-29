"""The Standard Model as a *skeleton*: representation assignments and coupling
structure, with all hypercharges left unknown.

This is the entire empirical input -- which boxes each field sits in, and the
fact that the Yukawa couplings exist (so fermions can get mass). Everything
numeric is derived downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import z3

from nsm.reps import SU3, SU2, Chirality, Fermion


@dataclass(frozen=True)
class Scalar:
    """A scalar field (the Higgs). Carries hypercharge but, being a boson, makes
    no contribution to the chiral anomalies."""

    name: str
    color: object
    weak: object
    hypercharge: object


@dataclass(frozen=True)
class Yukawa:
    """A Yukawa coupling, recorded structurally. Gauge invariance requires the
    signed hypercharges of the participating fields to sum to zero. Terms are
    ``(field_name, coefficient)`` pairs; a conjugated (barred) field or H-tilde
    enters with a negative coefficient."""

    label: str
    terms: tuple  # ((name, Fraction), ...)


@dataclass
class Model:
    """A gauge model: fermions, scalars, the Yukawa structure, and a
    normalization choice. Hypercharges are shared z3 symbols held in
    ``symbols`` and referenced by every field and constraint."""

    fermions: list
    scalars: list
    yukawas: list
    symbols: dict           # name -> z3 Real hypercharge symbol
    normalization: list = field(default_factory=list)  # [(name, Fraction value)]

    def add_fermion(self, name, color, weak, chirality, doublet_labels=None):
        """Extend the model with a new Weyl fermion whose hypercharge is a fresh
        unknown. This is the entry point for exploring non-standard spectra."""
        sym = z3.Real(f"Y_{name}")
        self.symbols[name] = sym
        self.fermions.append(
            Fermion(name, color, weak, sym, chirality, doublet_labels))
        return sym


def _one_generation_fermions(Y: dict) -> list:
    """The five chiral multiplets of one SM generation, given a dict of
    hypercharge symbols (or values) keyed by field name."""
    return [
        Fermion("Q_L", SU3.TRIPLET, SU2.DOUBLET, Y["Q_L"], Chirality.LEFT,
                doublet_labels=("up", "down")),
        Fermion("u_R", SU3.TRIPLET, SU2.SINGLET, Y["u_R"], Chirality.RIGHT),
        Fermion("d_R", SU3.TRIPLET, SU2.SINGLET, Y["d_R"], Chirality.RIGHT),
        Fermion("L_L", SU3.SINGLET, SU2.DOUBLET, Y["L_L"], Chirality.LEFT,
                doublet_labels=("nu", "e")),
        Fermion("e_R", SU3.SINGLET, SU2.SINGLET, Y["e_R"], Chirality.RIGHT),
    ]


def standard_model_skeleton() -> Model:
    """One generation of the Standard Model with unknown hypercharges.

    Convention: Q = T3 + Y/2, normalized by Y_H = 1.
    """
    names = ["Q_L", "u_R", "d_R", "L_L", "e_R", "H"]
    Y = {n: z3.Real(f"Y_{n}") for n in names}

    fermions = _one_generation_fermions(Y)
    higgs = Scalar("H", SU3.SINGLET, SU2.DOUBLET, Y["H"])

    # Gauge-invariant Yukawa couplings (the source of fermion mass):
    #   down-type:  Qbar_L  H        d_R    ->  -Y_Q + Y_H + Y_d = 0
    #   up-type:    Qbar_L  Htilde   u_R    ->  -Y_Q - Y_H + Y_u = 0
    #   charged-lepton: Lbar_L  H    e_R    ->  -Y_L + Y_H + Y_e = 0
    yukawas = [
        Yukawa("down", (("Q_L", Fraction(-1)), ("H", Fraction(1)), ("d_R", Fraction(1)))),
        Yukawa("up", (("Q_L", Fraction(-1)), ("H", Fraction(-1)), ("u_R", Fraction(1)))),
        Yukawa("lepton", (("L_L", Fraction(-1)), ("H", Fraction(1)), ("e_R", Fraction(1)))),
    ]

    return Model(
        fermions=fermions,
        scalars=[higgs],
        yukawas=yukawas,
        symbols=Y,
        normalization=[("H", Fraction(1))],
    )


def anomaly_only_skeleton() -> Model:
    """The fermion content with ONLY anomaly cancellation imposed -- no Yukawa
    couplings and no Higgs. Normalized by fixing Y_Q = 1/3. Used to show what
    anomaly cancellation alone does and does not determine."""
    names = ["Q_L", "u_R", "d_R", "L_L", "e_R"]
    Y = {n: z3.Real(f"Y_{n}") for n in names}
    return Model(
        fermions=_one_generation_fermions(Y),
        scalars=[],
        yukawas=[],
        symbols=Y,
        normalization=[("Q_L", Fraction(1, 3))],
    )
