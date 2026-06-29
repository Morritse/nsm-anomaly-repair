"""Representation skeleton: the structural "facts" a particle is built from.

A fermion is a Weyl field carrying:
  - a representation of SU(3)_color
  - a representation of SU(2)_weak
  - a weak hypercharge Y (a number, or an unknown to be solved for)
  - a chirality (left/right-handed)

Electric charge, anomaly contributions, and allowed couplings are all DERIVED
from these. Nothing about "electron" or "muon" is primitive here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction


@dataclass(frozen=True)
class Rep:
    """A simple-group irrep, carrying the invariants the anomaly engine needs.

    - ``dim``:    dimension (multiplicity of components)
    - ``dynkin``: Dynkin index T(R), entering the [G^2 U(1)] anomaly
    - ``cubic``:  cubic anomaly coefficient A(R), entering the [G^3] anomaly
                  (0 for real/pseudo-real reps such as SU(2) and SU(3) singlets)
    """

    name: str
    dim: int
    dynkin: Fraction
    cubic: Fraction = Fraction(0)


class SU3:
    SINGLET = Rep("1", 1, Fraction(0), Fraction(0))
    TRIPLET = Rep("3", 3, Fraction(1, 2), Fraction(1))
    ANTITRIPLET = Rep("3bar", 3, Fraction(1, 2), Fraction(-1))


class SU2:
    SINGLET = Rep("1", 1, Fraction(0))
    DOUBLET = Rep("2", 2, Fraction(1, 2))


class Chirality(Enum):
    LEFT = 1
    RIGHT = -1

    @property
    def sign(self) -> int:
        """Anomalies are summed over left-handed Weyl fermions; a right-handed
        field contributes with the opposite sign (it is a left-handed
        antiparticle)."""
        return self.value


@dataclass(frozen=True)
class Fermion:
    name: str
    color: Rep
    weak: Rep
    hypercharge: object  # Fraction (known) or a z3 expression (unknown)
    chirality: Chirality
    # Optional human labels for the two members of a weak doublet
    # (upper T3=+1/2, lower T3=-1/2), e.g. ("up", "down") or ("nu", "e").
    doublet_labels: tuple = None
    # Charge under a second gauged U(1)' (X), for two-U(1) anomaly checks.
    # Defaults to 0, so single-U(1) code is unaffected.
    x_charge: object = Fraction(0)

    @property
    def is_weak_doublet(self) -> bool:
        return self.weak is SU2.DOUBLET


def weak_components(fermion: Fermion):
    """Expand a fermion into its weak-isospin components, each with a definite
    T3. A doublet splits into T3 = +1/2 and -1/2; a singlet has T3 = 0."""
    if fermion.is_weak_doublet:
        upper, lower = fermion.doublet_labels or ("T+", "T-")
        return [
            (f"{fermion.name}:{upper}", Fraction(1, 2)),
            (f"{fermion.name}:{lower}", Fraction(-1, 2)),
        ]
    return [(fermion.name, Fraction(0))]
