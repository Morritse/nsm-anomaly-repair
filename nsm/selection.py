"""Conservation-law (selection-rule) analysis for a process.

Distinguishes:
  - EXACT gauge symmetries: electric charge, color triality (Z3 center).
  - ACCIDENTAL global symmetries of the renormalizable SM: baryon number B,
    total lepton number L.
  - APPROXIMATE symmetries: individual lepton flavor (broken by neutrino mixing).
  - B - L, the combination that survives the leading B/L-violating operators.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from nsm.particles import FLAVORS, particle, lepton_number, baryon_number

_TRIALITY = {"singlet": 0, "octet": 0, "triplet": 1, "antitriplet": 2}


def _sum(names, fn) -> Fraction:
    return sum((fn(particle(n)) for n in names), Fraction(0))


def _triality(names) -> int:
    return sum(_TRIALITY[particle(n).color] for n in names) % 3


@dataclass
class ConservationReport:
    charge_conserved: bool
    color_conserved: bool
    baryon_conserved: bool
    lepton_conserved: bool
    flavor_conserved: bool
    bminusl_conserved: bool
    delta_baryon: Fraction
    delta_lepton: Fraction
    delta_bminusl: Fraction


def analyze(initial, final) -> ConservationReport:
    b_initial, b_final = _sum(initial, baryon_number), _sum(final, baryon_number)
    l_initial, l_final = _sum(initial, lepton_number), _sum(final, lepton_number)
    delta_b = b_final - b_initial
    delta_l = l_final - l_initial
    delta_bminusl = delta_b - delta_l
    charge = _sum(final, lambda p: p.charge) == _sum(initial, lambda p: p.charge)
    color = _triality(final) == _triality(initial)
    baryon = delta_b == 0
    lepton = delta_l == 0
    flavor = all(
        _sum(final, lambda p, fl=fl: p.lepton[fl]) == _sum(initial, lambda p, fl=fl: p.lepton[fl])
        for fl in FLAVORS
    )
    b_minus_l = delta_bminusl == 0
    return ConservationReport(charge, color, baryon, lepton, flavor, b_minus_l,
                              delta_b, delta_l, delta_bminusl)
