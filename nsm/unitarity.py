"""Structural unitarity cuts (kinematics-free).

Optical-theorem structure: the imaginary part of A(i -> f) receives a
contribution from each on-shell intermediate state |n>, via
A(i -> n) * A(f -> n)^*. At the topological level a CUT STATE is a set of
intermediate particles {X1..Xk} that is tree-reachable as a final state from
BOTH the initial and the final side.

This enumerates candidate intermediate-state topologies and orders them by
nominal mass threshold. It is NOT a loop calculation: no cross sections, no
rates, and no claim about which intermediate state is physically open or
dominates. It generalizes factorization (a single-particle cut is a pole) to
k >= 2 particles.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

from nsm.particles import particle
from nsm.processes import tree_level_reachable
from nsm.factorization import internal_line_candidates


@dataclass(frozen=True)
class CutState:
    particles: tuple              # the intermediate on-shell particles (sorted)
    left_reachable: bool          # initial -> particles is a tree
    right_reachable: bool         # final -> particles is a tree
    threshold_mev: float | None   # sum of the cut particles' masses (None if unknown)


def _threshold(names):
    masses = [particle(n).mass_mev for n in names]
    if any(m is None for m in masses):
        return None
    return sum(masses)


def cut_states(initial, final, max_cut_size: int = 2, extra_vertices=(),
               max_external: int = 7):
    """Every multi-particle cut state (size 2 .. max_cut_size) of initial -> final.

    A cut state is an intermediate set the process can pass through: tree-
    reachable as a final state from both sides. (Single-particle cuts are
    factorization poles -- see nsm.factorization.) Returns a list of CutState.

    ``max_cut_size`` should stay small (2-3): the search is
    combinations-with-replacement over every internal-line particle, with a tree
    search per combination, so cost grows fast. A cut of size k is only checkable
    when ``max(|initial|, |final|) + k <= max_external``; requesting more raises
    rather than silently dropping the larger cuts."""
    initial, final = list(initial), list(final)
    reachable_size = max_external - max(len(initial), len(final))
    if max_cut_size > reachable_size:
        raise ValueError(
            f"max_cut_size={max_cut_size} exceeds the reachable cut size "
            f"{reachable_size} for this process (max_external={max_external} minus "
            f"the larger of |initial|={len(initial)}, |final|={len(final)}); "
            f"raise max_external or lower max_cut_size")
    candidates = sorted(internal_line_candidates(extra_vertices))
    cuts = []
    for k in range(2, max_cut_size + 1):
        for combo in combinations_with_replacement(candidates, k):
            cut = list(combo)
            if tree_level_reachable(initial, cut, max_external=max_external,
                                    extra_vertices=extra_vertices) is None:
                continue
            if tree_level_reachable(final, cut, max_external=max_external,
                                    extra_vertices=extra_vertices) is None:
                continue
            # left_reachable / right_reachable are True by construction (a cut is
            # recorded only after both checks pass); kept for an explicit record.
            cuts.append(CutState(tuple(sorted(combo)), True, True, _threshold(combo)))
    return cuts


def threshold_order(cuts):
    """Cut states ordered by ascending mass threshold (lightest intermediate
    state first). Unknown thresholds sort last. This is the threshold structure,
    not a rate ordering."""
    inf = float("inf")
    return sorted(
        cuts,
        key=lambda c: (c.threshold_mev if c.threshold_mev is not None else inf, c.particles),
    )
