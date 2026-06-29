"""Standard Model interaction vertices.

A vertex is recorded as an unordered multiset of particles in the ALL-OUTGOING
convention (every leg treated as an outgoing particle), so a vertex is valid
only if its legs' additive charges sum to zero. We store the SM coupling types
explicitly -- this is the content the user insisted on: real vertices, not a
generic "charges balance" rule. (A flavor-changing {gamma, mu-, e+} balances
charge but is deliberately absent.)

The set is closed under conjugation (every vertex's antiparticle image is also a
vertex), so the search may attach any leg in either orientation.
"""
from __future__ import annotations

from collections import Counter

from nsm.particles import antiparticle, particle

_CHARGED_FERMIONS = ["e-", "mu-", "tau-", "u", "d", "c", "s", "t", "b"]
_ALL_FERMIONS = _CHARGED_FERMIONS + ["nu_e", "nu_mu", "nu_tau"]
_LEPTON_CC = [("e-", "nu_e_bar"), ("mu-", "nu_mu_bar"), ("tau-", "nu_tau_bar")]
_QUARK_CC = [("u", "d_bar"), ("c", "s_bar"), ("t", "b_bar")]


def _canon(names) -> tuple:
    return tuple(sorted(names))


def _build() -> set:
    raw = []
    # QED: photon to charged fermion-antifermion
    raw += [("gamma", f, antiparticle(f).name) for f in _CHARGED_FERMIONS]
    # Neutral current: Z to any fermion-antifermion
    raw += [("Z", f, antiparticle(f).name) for f in _ALL_FERMIONS]
    # Higgs Yukawa: h to massive charged fermion-antifermion
    raw += [("h", f, antiparticle(f).name) for f in _CHARGED_FERMIONS]
    # Charged current: W to a lepton and its (anti)neutrino  {W+, l-, nu_l_bar}
    raw += [("W+", l, nu) for (l, nu) in _LEPTON_CC]
    # Charged current: W to up- and anti-down-type quarks  {W-, u, d_bar}
    raw += [("W-", uq, dq) for (uq, dq) in _QUARK_CC]
    # Gluon to quark-antiquark
    raw += [("gluon", q, antiparticle(q).name) for q in ["u", "d", "c", "s", "t", "b"]]
    # Triple gauge couplings
    raw += [("W+", "W-", "gamma"), ("W+", "W-", "Z")]
    # Higgs to weak bosons
    raw += [("h", "W+", "W-"), ("h", "Z", "Z")]
    # Three-gluon (non-abelian) and Higgs self-coupling: both 3-point and neutral,
    # so gg -> gg and hh -> hh are tree-level via internal exchange. Genuine
    # quartic (4-point) vertices remain out of the modeled subset.
    raw += [("gluon", "gluon", "gluon"), ("h", "h", "h")]

    vertices = set()
    for v in raw:
        # sanity: all-outgoing electric charge must vanish
        assert sum(particle(n).charge for n in v) == 0, v
        vertices.add(_canon(v))
        vertices.add(_canon(antiparticle(n).name for n in v))  # conjugate image
    return vertices


VERTICES = _build()


def _leg_priority(name: str) -> tuple:
    """Stable, human-useful ordering for alternative mediator choices.

    When several tree diagrams exist, prefer gauge exchange over Higgs/Yukawa
    exchange in the first explanation. This keeps common processes like
    e+e- -> mu+mu- from depending on Python set iteration order.
    """
    order = {
        "gamma": 0,
        "Z": 1,
        "W+": 2,
        "W-": 2,
        "gluon": 3,
        "h": 4,
    }
    return (order.get(name, 10), name)


_VERTEX_COUNTERS = [Counter(v) for v in sorted(VERTICES)]


def is_vertex(names) -> bool:
    return _canon(names) in VERTICES


def third_legs(a: str, b: str):
    """All particles c such that {a, b, c} is an SM vertex."""
    pair = Counter([a, b])
    out = []
    for vc in _VERTEX_COUNTERS:
        if pair <= vc:
            out.extend((vc - pair).elements())
    return sorted(out, key=_leg_priority)


def conjugate_closure(legs):
    """The canonical vertex and its conjugate image, for a charge-neutral,
    all-outgoing leg multiset. Used to build candidate (extension) vertices
    without mutating the global SM set."""
    if sum(particle(n).charge for n in legs) != 0:
        raise ValueError(f"vertex is not charge-neutral: {tuple(legs)}")
    return {_canon(legs), _canon(antiparticle(n).name for n in legs)}
