"""Tree-level reachability: does a connected tree of vertices connect the
initial and final states?

Method: cross all external particles to the all-outgoing convention, then
repeatedly fuse two legs through a vertex (replacing them with the conjugate of
the vertex's third leg -- an internal propagator) until either three legs remain
that themselves form a vertex (success) or no fusion applies (failure). A tree
with E external legs uses E-2 cubic vertices, so the recursion depth is bounded.

``extra_vertices`` lets a caller add effective vertices (e.g. a hypothesized new
mediator's couplings) for one query without mutating the global SM vertex set.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from nsm.particles import antiparticle
from nsm.vertices import VERTICES, _VERTEX_COUNTERS, _canon, _leg_priority


def _vset_counters(extra_vertices):
    extra = {_canon(v) for v in extra_vertices}
    return VERTICES | extra, _VERTEX_COUNTERS + [Counter(v) for v in extra]


class TreeSearchStatus(Enum):
    FOUND = "found"
    EXHAUSTIVE_NONE = "exhaustive-none"
    BUDGET_EXCEEDED = "budget-exceeded"


@dataclass(frozen=True)
class TreeSearchResult:
    status: TreeSearchStatus
    mediators: list | None
    detail: str = ""


def tree_level_search(initial, final, max_external: int = 7, extra_vertices=()) -> TreeSearchResult:
    """Search for a tree diagram, distinguishing no-tree from search-budget cutoff."""
    legs = [antiparticle(n).name for n in initial] + list(final)
    if len(legs) > max_external:
        return TreeSearchResult(
            TreeSearchStatus.BUDGET_EXCEEDED,
            None,
            f"{len(legs)} external legs exceeds max_external={max_external}",
        )
    vset, counters = _vset_counters(extra_vertices)
    mediators = _reduce(Counter(legs), set(), vset, counters)
    if mediators is None:
        return TreeSearchResult(TreeSearchStatus.EXHAUSTIVE_NONE, None)
    return TreeSearchResult(TreeSearchStatus.FOUND, mediators)


def tree_level_reachable(initial, final, max_external: int = 7, extra_vertices=()):
    """Return a list of internal mediator particles if a tree diagram exists,
    else None. An empty list means the external legs form a vertex directly.

    Backward-compatible wrapper around :func:`tree_level_search`; callers that
    need to distinguish no-tree from a budget cutoff should use that function.
    """
    result = tree_level_search(initial, final, max_external=max_external,
                               extra_vertices=extra_vertices)
    return result.mediators if result.status is TreeSearchStatus.FOUND else None


def legs_form_tree(out_legs, extra_vertices=(), max_external: int = 7) -> bool:
    """True if the given ALL-OUTGOING legs assemble into one connected tree of
    vertices. (tree_level_reachable crosses the initial state first; this takes
    legs already in the all-outgoing convention.)"""
    if not (3 <= len(out_legs) <= max_external):
        return False
    vset, counters = _vset_counters(extra_vertices)
    return _reduce(Counter(out_legs), set(), vset, counters) is not None


def _third_legs(a, b, counters):
    pair = Counter([a, b])
    out = []
    for vc in counters:
        if pair <= vc:
            out.extend((vc - pair).elements())
    return sorted(out, key=_leg_priority)


def _reduce(legs: Counter, seen: set, vset, counters):
    total = sum(legs.values())
    if total == 3:
        return [] if _canon(legs.elements()) in vset else None
    if total < 3:
        return None

    key = tuple(sorted(legs.elements()))
    if key in seen:
        return None
    seen.add(key)

    items = list(legs.elements())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            for c in _third_legs(a, b, counters):
                propagator = antiparticle(c).name
                nxt = legs.copy()
                nxt[a] -= 1
                nxt[b] -= 1
                nxt[propagator] += 1
                nxt += Counter()  # drop zero/negative entries
                rest = _reduce(nxt, seen, vset, counters)
                if rest is not None:
                    return [propagator] + rest
    return None
