"""Candidate-extension search.

Accept a hypothesized new mediator and ask: what kind of field is it, is it
consistent, what may it couple to, and what processes would expose it?

The classification rests on implemented coupling templates rather than a full
mediator theorem: scalar source, conserved rank-1 gauge current, and rank-2
stress-energy tensor T_mu_nu, which every particle sources -- hence universal.
Massless higher-spin (>= 3) long-range couplings are treated as OUTSIDE this
modeled consistency class (cf. Coleman-Mandula / Weinberg-Witten), not as a
general no-go theorem proved here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from math import isfinite

from nsm.reps import SU3, SU2, Chirality, Fermion
from nsm.anomalies import gauge_anomaly_coefficients
from nsm.particles import Particle, register_particle, antiparticle, particle, _lep, PARTICLES
from nsm.vertices import conjugate_closure
from nsm.factorization import factorization_mediators
from nsm.verdict import classify, Verdict


# coupling structure -> rank of the conserved current it attaches to
_CURRENT_RANK = {
    "trace": 0,          # scalar source (trace of stress-energy / mass)
    "gauge_charge": 1,   # a conserved U(1) charge (vector current)
    "stress_energy": 2,  # the universal rank-2 tensor T_mu_nu
}
# Conserved currents available in an interacting QFT, by rank.
_MAX_CONSERVED_RANK = 2


@dataclass(frozen=True)
class Candidate:
    name: str
    spin: Fraction
    mass_mev: float
    charge: Fraction = Fraction(0)
    color: str = "singlet"
    weak: str = "singlet"
    self_conjugate: bool = True
    coupling: str = "stress_energy"
    # for coupling == "gauge_charge": ((field_name, U(1)' charge), ...)
    charge_map: tuple = ()
    coupling_scale: str | None = None


class FieldType(Enum):
    SCALAR = "spin-0 scalar field"
    GAUGE_BOSON = "massless spin-1 gauge boson (couples to a conserved charge)"
    GRAVITON_LIKE = "massless spin-2 field coupling universally to stress-energy"
    MASSIVE_MEDIATOR = "massive mediator (short-range force)"
    OUTSIDE_MODELED_CLASS = "massless spin >= 3: outside the modeled consistency class"


def classify_field_type(c: Candidate):
    """Return (FieldType, reasoning)."""
    if c.spin == 0:
        return FieldType.SCALAR, "spin 0 -> couples to a scalar source"
    if c.mass_mev and c.mass_mev > 0:
        return (FieldType.MASSIVE_MEDIATOR,
                f"massive (m = {c.mass_mev} MeV) -> finite-range, not a long-range force")
    if c.spin == 1:
        return FieldType.GAUGE_BOSON, "massless spin 1 -> couples to a rank-1 conserved charge"
    if c.spin == 2:
        return (FieldType.GRAVITON_LIKE,
                "massless spin 2 -> couples to the rank-2 stress-energy tensor (universal)")
    return (FieldType.OUTSIDE_MODELED_CLASS,
            f"massless spin {c.spin} needs a rank-{c.spin} conserved current, which "
            "is outside the modeled consistency class (cf. Coleman-Mandula / "
            "Weinberg-Witten)")


def required_spin_for_universal_massless_force() -> Fraction:
    """The spin a massless mediator must have to couple universally to energy-
    momentum. Universality => it couples to the stress-energy tensor, a rank-2
    conserved current => the mediator is rank-2 => spin 2."""
    rank = _CURRENT_RANK["stress_energy"]
    assert rank <= _MAX_CONSERVED_RANK
    return Fraction(rank)


# Standard-Model fermion skeleton (rep + forced hypercharge), for checking a
# hypothesized new U(1)' gauge charge against the FULL two-U(1) anomaly set --
# including the mixed Y-X anomalies a single-U(1) check would miss.
_FERMION_REPS = {
    "Q_L": (SU3.TRIPLET, SU2.DOUBLET, Chirality.LEFT),
    "u_R": (SU3.TRIPLET, SU2.SINGLET, Chirality.RIGHT),
    "d_R": (SU3.TRIPLET, SU2.SINGLET, Chirality.RIGHT),
    "L_L": (SU3.SINGLET, SU2.DOUBLET, Chirality.LEFT),
    "e_R": (SU3.SINGLET, SU2.SINGLET, Chirality.RIGHT),
    "nu_R": (SU3.SINGLET, SU2.SINGLET, Chirality.RIGHT),
}
_SM_HYPERCHARGE = {
    "Q_L": Fraction(1, 3), "u_R": Fraction(4, 3), "d_R": Fraction(-2, 3),
    "L_L": Fraction(-1), "e_R": Fraction(-2), "nu_R": Fraction(0),
}


@dataclass
class ConsistencyReport:
    is_consistent: bool
    checks: list  # list of (name, passed: bool, detail)

    @property
    def failures(self):
        return [(n, d) for (n, p, d) in self.checks if not p]


_SM_GENERATION = ("Q_L", "u_R", "d_R", "L_L", "e_R")  # the chiral SM fields


def normalize_charge_map(charge_map):
    """The U(1)' charge map over a full SM generation: the five chiral SM fields
    default to X=0, overwritten by supplied charges; ``nu_R`` is included only if
    supplied (it is sterile/optional). Omitted fields mean X=0 -- they are NOT
    removed from the spectrum. Unknown or duplicate field names raise ValueError."""
    supplied = {}
    for name, q in charge_map:
        if name not in _FERMION_REPS:
            raise ValueError(f"unknown fermion field '{name}'")
        if name in supplied:
            raise ValueError(f"duplicate fermion field '{name}'")
        supplied[name] = Fraction(q)
    fields = list(_SM_GENERATION)
    if "nu_R" in supplied:
        fields.append("nu_R")
    return tuple((name, supplied.get(name, Fraction(0))) for name in fields)


def gauge_charge_fermions(charge_map):
    """The SM fermions (with forced hypercharge Y) carrying the candidate's
    U(1)' charge X, for the two-U(1) anomaly check. Partial maps are normalized
    against the full SM generation (omitted fields -> X=0)."""
    return [
        Fermion(name, _FERMION_REPS[name][0], _FERMION_REPS[name][1],
                _SM_HYPERCHARGE[name], _FERMION_REPS[name][2], x_charge=q)
        for name, q in normalize_charge_map(charge_map)
    ]


def _anomaly_checks(charge_map):
    """Require every anomaly of SU(3)xSU(2)xU(1)_Y x U(1)_X to cancel -- the full
    set, including the mixed Y-X coefficients."""
    coeffs = gauge_anomaly_coefficients(gauge_charge_fermions(charge_map))
    return [(f"anomaly[{coeff}]", value == 0, f"{coeff} = {value}")
            for coeff, value in coeffs.items()]


def _has_valid_mass(c: Candidate) -> bool:
    return isinstance(c.mass_mev, (int, float)) and isfinite(float(c.mass_mev))


def _has_valid_spin(c: Candidate) -> bool:
    return isinstance(c.spin, Fraction) and c.spin >= 0 and c.spin.denominator == 1


def _candidate_domain_checks(c: Candidate) -> list:
    checks = []
    finite_mass = _has_valid_mass(c)
    checks.append(("finite nonnegative mass", finite_mass and c.mass_mev >= 0,
                   f"mass_mev = {c.mass_mev}"))
    checks.append(("bosonic nonnegative spin",
                   _has_valid_spin(c),
                   f"spin = {c.spin}"))

    if c.coupling in ("stress_energy", "gauge_charge"):
        checks.append(("neutral self-conjugate mediator",
                       c.charge == 0 and c.self_conjugate,
                       f"charge={c.charge}, self_conjugate={c.self_conjugate}"))
        checks.append(("gauge-singlet mediator",
                       c.color == "singlet" and c.weak == "singlet",
                       f"color={c.color}, weak={c.weak}"))

    if c.coupling == "trace":
        checks.append(("implemented process-exposure template", False,
                       "scalar trace coupling is classified, but scalar vertices "
                       "are not implemented in the exposure layer"))
    return checks


def consistency_report(c: Candidate) -> ConsistencyReport:
    checks = []
    valid_mass = _has_valid_mass(c) and c.mass_mev >= 0
    valid_spin = _has_valid_spin(c)
    massless = valid_mass and c.mass_mev == 0
    rank = _CURRENT_RANK.get(c.coupling)

    # 0. The coupling must be a recognized structure -- otherwise there is
    #    nothing to judge and we must NOT report a silent "consistent".
    checks.append(("recognized coupling", rank is not None, f"coupling '{c.coupling}'"))
    checks.extend(_candidate_domain_checks(c))

    # 1. For a MASSLESS mediator, spin must match the rank of the current it
    #    couples to (Weinberg's soft theorems: universal->spin2, charge->spin1,
    #    scalar source->spin0). Massive mediators are short-range and may couple
    #    to non-conserved currents, so this and the higher-spin class check are gated
    #    on masslessness.
    if massless and valid_spin and rank is not None:
        checks.append((
            "spin matches current rank",
            c.spin == rank,
            f"spin {c.spin} vs rank-{rank} '{c.coupling}' current",
        ))
    if massless and valid_spin and c.spin >= 3:
        checks.append((
            "higher-spin within modeled class",
            False,
            f"massless spin {c.spin}: a rank-{c.spin} current is outside the modeled class",
        ))

    # 2. A gauge boson gauges a symmetry, so its charge must be anomaly-free --
    #    required even if the symmetry is later spontaneously broken (massive). An
    #    all-zero (or empty) charge map is the trivial zero generator: no gauged
    #    U(1)', not a meaningful consistent extension.
    if c.coupling == "gauge_charge":
        try:
            cmap = normalize_charge_map(c.charge_map)
        except ValueError as exc:
            checks.append(("valid charge map", False, str(exc)))
        else:
            if all(q == 0 for _, q in cmap):
                checks.append(("nontrivial gauge charge", False,
                               "all U(1)' charges are zero: trivial (no gauged U(1)')"))
            else:
                checks.extend(_anomaly_checks(c.charge_map))

    is_consistent = all(passed for _, passed, _ in checks)
    return ConsistencyReport(is_consistent, checks)


def validate_candidate(c: Candidate) -> Candidate:
    """Return ``c`` if it is usable by the configured extension/exposure layer.

    This is the gate downstream operations use so an anomalous or out-of-domain
    candidate cannot still generate vertices or process-exposure claims.
    """
    report = consistency_report(c)
    if not report.is_consistent:
        failures = "; ".join(detail for _, detail in report.failures)
        raise ValueError(f"candidate {c.name!r} is not valid in the configured model: {failures}")
    return c


def _coupling_scale(c: Candidate) -> str:
    if c.coupling_scale:
        return c.coupling_scale
    return "planck" if c.coupling == "stress_energy" else "weak"


# --- registration and process exposure --------------------------------------

_ALL_FERMIONS = ["e-", "mu-", "tau-", "nu_e", "nu_mu", "nu_tau",
                 "u", "d", "c", "s", "t", "b"]
_BOSON_PAIRS = [("gamma", "gamma"), ("gluon", "gluon"), ("Z", "Z"),
                ("W+", "W-"), ("h", "h")]


def register_candidate(c: Candidate) -> None:
    """Make the candidate a known particle so processes can reference it. (Its
    vertices are supplied per-query via ``candidate_vertices`` rather than
    mutating the global SM vertex set.) Re-registering replaces the prior
    definition of the same name."""
    register_particle(Particle(
        name=c.name, charge=c.charge, baryon=Fraction(0), lepton=_lep(),
        color=c.color, spin=c.spin, is_boson=True,
        anti=c.name if c.self_conjugate else f"{c.name}_bar",
        mass_mev=c.mass_mev))


@contextmanager
def _scoped_candidate_registration(c: Candidate):
    """Temporarily register a candidate while classifying one exposure query."""
    snapshot = dict(PARTICLES)
    register_candidate(c)
    try:
        yield
    finally:
        PARTICLES.clear()
        PARTICLES.update(snapshot)


# Physical fermions sitting in each rep-field of the kernel's charge_map. A
# gauged U(1)' charge is family-universal, so a charge on a rep applies to every
# generation. (nu_R is sterile: it matters for anomaly consistency but has no
# observable vertex here, so it is omitted.)
REP_TO_PARTICLES = {
    "Q_L": ("u", "d", "c", "s", "t", "b"),
    "u_R": ("u", "c", "t"),
    "d_R": ("d", "s", "b"),
    "L_L": ("e-", "mu-", "tau-", "nu_e", "nu_mu", "nu_tau"),
    "e_R": ("e-", "mu-", "tau-"),
}


def coupled_fermions(c: Candidate) -> set:
    """The physical fermions the candidate couples to. Universal (stress-energy)
    => all of them; gauge_charge => those carrying nonzero vector-like U(1)'
    charge. Chirality-specific gauge charges require explicit helicity/chirality
    labels, which this process-exposure layer does not model."""
    if c.coupling == "stress_energy":
        return set(_ALL_FERMIONS)
    if c.coupling == "gauge_charge":
        return {
            f for f in _ALL_FERMIONS
            if _vector_charge_for_fermion(c, f) != 0
        }
    return set()


def _vector_charge_for_fermion(c: Candidate, fermion: str) -> Fraction:
    charges = {
        Fraction(q)
        for rep, q in normalize_charge_map(c.charge_map)
        if fermion in REP_TO_PARTICLES.get(rep, ())
    }
    if not charges:
        return Fraction(0)
    if len(charges) > 1:
        raise ValueError(
            f"{c.name} couples chirally to {fermion} (charges {sorted(charges)}): "
            "chirality-specific coupling not modeled; supply explicit "
            "chirality/helicity labels")
    return next(iter(charges))


def candidate_vertices(c: Candidate) -> set:
    """The effective vertices the candidate introduces, conjugation-closed.

    - stress_energy (graviton): attaches to every fermion line and boson pair.
    - gauge_charge (Z'): attaches to each fermion carrying its U(1)' charge,
      flavor-diagonally.

    Precondition: the candidate must be registered (register_candidate) so its
    particle data is available."""
    validate_candidate(c)
    raw = []
    if c.coupling == "stress_energy":
        raw += [(c.name, f, antiparticle(f).name) for f in _ALL_FERMIONS]
        raw += [(c.name, *pair) for pair in _BOSON_PAIRS]
    elif c.coupling == "gauge_charge":
        # A neutral U(1)' gauge boson: Z'-f-fbar for each charged fermion.
        raw += [(c.name, f, antiparticle(f).name)
                for f in sorted(coupled_fermions(c))]
    out = set()
    for v in raw:
        out |= conjugate_closure(v)
    return out


def _mediates(initial, final, candidate: Candidate) -> bool:
    """Does the candidate sit on an internal line of some tree diagram for
    initial -> final? It mediates iff it appears as a factorization channel --
    the same vertex-level tree search, so it never claims a coupling the
    candidate lacks (no flavor-/charge-violating false positives)."""
    if candidate.name in (list(initial) + list(final)):
        return False  # produced/external, not an internal mediator
    extra = candidate_vertices(candidate)
    if not extra:
        return False
    return candidate.name in factorization_mediators(initial, final, extra_vertices=extra)


@dataclass
class ExposureResult:
    verdict: object
    detail: str
    mediators: object
    notes: list


# Verdicts that the n->1 phase-space argument may REFINE. More fundamental
# verdicts (exact-symmetry / kinematic / higher-dim-operator) must not be
# overwritten by it.
_REFINABLE = (Verdict.ALLOWED_AT_TREE_LEVEL,
              Verdict.ALLOWED_BY_CONSERVATION_BUT_NO_SM_VERTEX)


def exposing_verdict(initial, final, candidate: Candidate) -> ExposureResult:
    """Classify a process that may involve the candidate, annotating with the
    effects that the bare verdict cannot express (suppression scale, and the
    vanishing phase space of an all-massless n->1)."""
    initial, final = list(initial), list(final)  # materialize: classify consumes them
    validate_candidate(candidate)
    with _scoped_candidate_registration(candidate):
        res = classify(initial, final, extra_vertices=candidate_vertices(candidate))
        verdict, detail, notes = res.verdict, res.detail, []
        names = initial + final

        # An n->1 process needs the single product's mass to exceed the sum of the
        # initial masses (the minimum invariant mass); otherwise the required
        # invariant mass is unreachable -- vanishing phase space. This covers the
        # all-massless 2->1 (e.g. gamma gamma -> G) and a too-light product.
        if verdict in _REFINABLE and len(final) == 1 and len(initial) >= 2:
            m_prod = particle(final[0]).mass_mev
            m_init = [particle(n).mass_mev for n in initial]
            if m_prod is not None and all(m is not None for m in m_init):
                s_min = sum(m_init)
                if m_prod <= s_min:
                    verdict = Verdict.KINEMATICALLY_FORBIDDEN
                    detail = (f"{len(initial)}->1 needs product mass > {s_min:.3f} MeV "
                              f"(have {m_prod:.3f}): vanishing phase space")
                    notes.append(f"{len(initial)}->1 with insufficient/degenerate "
                                 "phase space (kinematically special)")

        flavor_diagonal = candidate.coupling in ("stress_energy", "gauge_charge")
        if candidate.name in names:
            notes.append(f"involves {candidate.name}; amplitude is "
                         f"{_coupling_scale(candidate)}-suppressed")
            # A flavor-diagonal mediator cannot itself drive a flavor change.
            if flavor_diagonal and verdict is Verdict.ALLOWED_ONLY_WITH_MIXING_OR_LOOPS:
                kind = "gravity" if candidate.coupling == "stress_energy" else candidate.name
                notes.append(f"{candidate.name} couples flavor-diagonally, so this "
                             f"flavor-changing process is not a generic {kind} signal")
        elif _mediates(initial, final, candidate):
            if candidate.coupling == "stress_energy":
                how = f"{candidate.name} exchange"
            else:
                how = "s-/t-channel; appears as a resonance / interference"
            notes.append(f"{candidate.name} also mediates this process "
                         f"({how}, {_coupling_scale(candidate)}-suppressed)")

        return ExposureResult(verdict, detail, res.mediators, notes)
