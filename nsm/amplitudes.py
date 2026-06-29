"""Symbolic soft-emission checks for candidate mediators.

This module does not compute cross sections. It encodes the leading soft
factor structure that a massless mediator must obey:

  spin 1:  S ~ sum_i eta_i q_i (p_i.eps) / (p_i.k)
  spin 2:  S ~ sum_i eta_i     (p_i.eps.p_i) / (p_i.k)

The payoff is a Ward-identity sanity check. Gauge-boson emission is consistent
when the signed charge flow vanishes; graviton emission is consistent when the
coupling is universal, so replacing a polarization by the soft momentum reduces
the factor to total momentum conservation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from nsm.extensions import (
    Candidate, REP_TO_PARTICLES, validate_candidate, _vector_charge_for_fermion,
)
from nsm.particles import antiparticle, particle

# The rep -> physical-fermion map is shared with the extension layer (one source
# of truth). If a physical field receives different left/right charges, that is a
# chiral/axial coupling and this symbolic vector soft factor refuses to collapse
# it to one number.


@dataclass(frozen=True)
class ExternalLeg:
    particle: str
    flow: int          # -1 incoming, +1 outgoing

    @property
    def crossed_name(self) -> str:
        return self.particle if self.flow == 1 else antiparticle(self.particle).name


@dataclass(frozen=True)
class SoftTerm:
    leg: ExternalLeg
    coefficient: Fraction
    numerator: str
    denominator: str


@dataclass(frozen=True)
class WardCheck:
    passed: bool
    reason: str


@dataclass(frozen=True)
class SoftFactor:
    candidate: str
    spin: Fraction
    current_rank: int
    terms: tuple[SoftTerm, ...]
    ward_identity: WardCheck
    note: str


def external_legs(initial, final) -> tuple[ExternalLeg, ...]:
    """External process legs with the soft-theorem incoming/outgoing sign."""
    return tuple([ExternalLeg(n, -1) for n in initial]
                 + [ExternalLeg(n, 1) for n in final])


def _canonical_particle_name(name: str) -> str:
    """Return the particle-side name whose U(1)' charge should be looked up."""
    p = particle(name)
    if p.name in _all_physical_fermions():
        return p.name
    anti = antiparticle(p.name).name
    return anti if anti in _all_physical_fermions() else p.name


def _all_physical_fermions() -> set[str]:
    out = set()
    for names in REP_TO_PARTICLES.values():
        out.update(names)
    return out


def gauge_charge_for_particle(c: Candidate, name: str) -> Fraction:
    """Effective vector charge of a physical particle under a U(1)' candidate.

    The per-rep vector charge -- and the chiral-coupling rejection -- is shared
    with the extension layer via ``_vector_charge_for_fermion``; only the
    antiparticle sign flip is local here.
    """
    if c.coupling != "gauge_charge":
        raise ValueError(f"{c.name} does not couple to a gauge charge")
    p = particle(name)
    base = _canonical_particle_name(p.name)
    q = _vector_charge_for_fermion(c, base)
    return -q if p.name != base else q


def leading_soft_factor(c: Candidate, initial, final) -> SoftFactor:
    """Symbolic leading soft factor for adding ``candidate`` as a soft emission.

    ``initial`` and ``final`` are the hard process legs, excluding the emitted
    soft candidate. The returned terms are intentionally symbolic: they expose
    the coefficient pattern and the Ward-identity condition without pretending
    to evaluate momenta.
    """
    validate_candidate(c)
    legs = external_legs(initial, final)
    if c.mass_mev != 0:
        raise ValueError("soft theorem layer only models massless mediators")

    if c.coupling == "stress_energy":
        if c.spin != 2:
            raise ValueError("stress-energy soft factor requires a spin-2 candidate")
        terms = tuple(
            SoftTerm(
                leg=leg,
                coefficient=Fraction(leg.flow),
                numerator=f"p({leg.particle})^mu eps_mu_nu p({leg.particle})^nu",
                denominator=f"p({leg.particle}).k",
            )
            for leg in legs
        )
        return SoftFactor(
            candidate=c.name,
            spin=c.spin,
            current_rank=2,
            terms=terms,
            ward_identity=WardCheck(
                True,
                "polarization replacement reduces to total momentum conservation; coupling is universal",
            ),
            note="universal Weinberg soft-graviton factor",
        )

    if c.coupling == "gauge_charge":
        if c.spin != 1:
            raise ValueError("gauge-charge soft factor requires a spin-1 candidate")
        charged = [(leg, gauge_charge_for_particle(c, leg.particle)) for leg in legs]
        terms = tuple(
            SoftTerm(
                leg=leg,
                coefficient=Fraction(leg.flow) * q,
                numerator=f"p({leg.particle}).eps",
                denominator=f"p({leg.particle}).k",
            )
            for leg, q in charged
            if q != 0
        )
        signed = sum((Fraction(leg.flow) * q for leg, q in charged), Fraction(0))
        return SoftFactor(
            candidate=c.name,
            spin=c.spin,
            current_rank=1,
            terms=terms,
            ward_identity=WardCheck(
                signed == 0,
                "signed U(1)' charge flow vanishes"
                if signed == 0
                else f"signed U(1)' charge flow is {signed}",
            ),
            note="charge-weighted Weinberg soft-vector factor",
        )

    raise ValueError(f"unsupported soft coupling: {c.coupling}")


def soft_theorem_report(c: Candidate, initial, final) -> WardCheck:
    """A compact pass/fail check for the leading soft theorem (its Ward identity).
    Named to distinguish it from nsm.factorization, which locates poles."""
    sf = leading_soft_factor(c, initial, final)
    if not sf.terms:
        return WardCheck(False, f"{c.name} has no soft-coupled external legs")
    return sf.ward_identity
