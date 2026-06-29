"""Factorization-channel enumeration.

Locality says a tree amplitude's only poles are where an internal line goes
on-shell. For a process initial -> final, this finds every internal particle X
such that the external legs split into two groups that each form a valid tree
once X is attached -- the channels the amplitude factorizes through.

It works at the level of vertices and quantum numbers: it reports WHICH channels
exist and, by construction, that each uses only allowed vertices (so no spurious
poles). It does NOT evaluate residues -- the statement that the residue equals
the product of the two on-shell sub-amplitudes requires actual kinematics
(spinor-helicity / BCFW), a separate layer that is deliberately out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from nsm.particles import antiparticle
from nsm.processes import legs_form_tree
from nsm.vertices import VERTICES


@dataclass(frozen=True)
class Channel:
    mediator: str   # internal particle that can go on-shell
    left: tuple     # all-outgoing external legs on one side of the cut
    right: tuple    # ... and the other side


def internal_line_candidates(extra_vertices) -> set:
    """Particle names that appear on some vertex -- the only ones that can be an
    internal line."""
    names = set()
    for v in VERTICES:
        names.update(v)
    for v in extra_vertices:
        names.update(v)   # only names matter; ordering is irrelevant
    return names


def factorization_channels(initial, final, extra_vertices=(), max_external: int = 7):
    """Every single-internal-line factorization channel of initial -> final.

    Each channel cuts one internal line X: the all-outgoing external legs split
    into ``left`` (forming a tree with X) and ``right`` (forming a tree with the
    conjugate of X)."""
    out_legs = [antiparticle(n).name for n in initial] + list(final)
    if len(out_legs) > max_external:
        raise ValueError(
            f"factorization search needs {len(out_legs)} external legs, "
            f"exceeding max_external={max_external}")
    if len(out_legs) < 3:
        return []

    candidates = internal_line_candidates(extra_vertices)
    n = len(out_legs)
    seen = set()
    channels = []
    for r in range(1, n):
        for idx in combinations(range(n), r):
            left = [out_legs[i] for i in idx]
            right = [out_legs[i] for i in range(n) if i not in idx]
            for x in candidates:
                xbar = antiparticle(x).name
                if (legs_form_tree(left + [x], extra_vertices, max_external)
                        and legs_form_tree(right + [xbar], extra_vertices, max_external)):
                    L, R = tuple(sorted(left)), tuple(sorted(right))
                    # the cut (L,X) | (R,Xbar) is an unordered pair -- dedupe it
                    key = tuple(sorted([(L, x), (R, xbar)]))
                    if key in seen:
                        continue
                    seen.add(key)
                    channels.append(Channel(x, L, R))
    return channels


def factorization_mediators(initial, final, extra_vertices=(),
                            max_external: int = 7) -> set:
    """The set of internal particles the process factorizes through."""
    return {
        ch.mediator
        for ch in factorization_channels(initial, final, extra_vertices, max_external)
    }
