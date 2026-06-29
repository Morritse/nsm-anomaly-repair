"""Minimal repair search for inconsistent U(1)' gauge extensions.

Given a U(1)' charge map over one SM generation, this module searches for added
fermions whose U(1)' charges cancel the full two-U(1) anomaly set. The narrow
``minimal_repair`` API can reproduce the classic sterile-singlet result: gauged
B-L is repaired by one right-handed neutrino. The broader ``tiered_repair`` API
searches a declared menu of sterile singlets plus SM-vector-like, U(1)'-chiral
exotic pairs (N/E/L/D/Q) and reports the first tier that yields a rational
repair:
already-consistent, sterile-repairable, colorless-exotic-repairable,
colored-exotic-required, or still-blocked.

Searches emit a minimality evidence record: for each lower-cost menu
combination, the support-prune or solver verdict is recorded. Support-pruned and
``unsat`` rungs certify that the combination is impossible; ``unknown`` and
``sat-algebraic-model`` rungs are reported honestly and do not certify
minimality (an algebraic model does not prove that no rational solution exists).

Repairs are minimal WITHIN THE DECLARED MENU and cost function only when the
certificate is marked minimal; they are not unique, not a naturalness claim, and
not a claim that the added fields exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import z3

from nsm.reps import SU3, SU2, Chirality, Fermion
from nsm.anomalies import gauge_anomaly_coefficients, _to_z3
from nsm.extensions import gauge_charge_fermions

_SM_SECTOR = ("SU3^3", "SU3^2-Y", "SU2^2-Y", "grav^2-Y", "Y^3")
_X_ANOMALIES = ("SU3^2-X", "SU2^2-X", "grav^2-X", "Y^2-X", "Y-X^2", "X^3")
_DEFAULT_MENU = ("N", "E", "L", "D", "Q")
_CERTIFYING_VERDICTS = {"support-prune", "unsat"}


@dataclass
class RepairCertificate:
    minimal: bool   # True iff no lower-cost rung is unresolved
    rungs: list     # [(combination description, verdict)] for prior attempts
    note: str = ""


@dataclass
class Repair:
    repairable: bool
    added: list
    verified: bool
    blocked_by: list = field(default_factory=list)
    note: str = ""
    certificate: object = None  # RepairCertificate


def _as_fraction(v):
    if z3.is_algebraic_value(v):
        raise ValueError(f"non-rational charge: {v}")
    return v.as_fraction()


def _search_sterile(g, c, max_added, timeout_ms):
    """Search for the smallest list of nonzero rational charges X_i with
    sum X_i = g and sum X_i^3 = c. Returns (charges, n, rungs):
      - charges: the solution (or None), n: its size (or None);
      - rungs: [(size, verdict)] for every size tried strictly below the
        solution (or all sizes if none found), where verdict is "unsat"
        (no real solution -> certifies no repair of that size), "unknown"
        (solver undecided), or "sat-algebraic-model" (the solver returned an
        algebraic model; rational solubility is left undetermined).
    """
    rungs = []
    for n in range(1, max_added + 1):
        xs = [z3.Real(f"X_{i}") for i in range(n)]
        solver = z3.Solver()
        solver.set("timeout", timeout_ms)
        solver.add(z3.Sum(xs) == _to_z3(g))
        solver.add(z3.Sum([x * x * x for x in xs]) == _to_z3(c))
        for x in xs:
            solver.add(x != 0)
        status = solver.check()
        if status == z3.unsat:
            rungs.append((n, "unsat"))
            continue
        if status == z3.unknown:
            rungs.append((n, "unknown"))
            continue
        try:
            charges = [_as_fraction(solver.model().eval(x, model_completion=True))
                       for x in xs]
        except ValueError:
            rungs.append((n, "sat-algebraic-model"))
            continue
        return charges, n, rungs
    return None, None, rungs


def _parse_fermion_from_name(name, x):
    if name.startswith("nu_R") or name.startswith("N_"):
        return Fermion(name, SU3.SINGLET, SU2.SINGLET, Fraction(0), Chirality.RIGHT, x_charge=x)
    elif name.startswith("E_L"):
        return Fermion(name, SU3.SINGLET, SU2.SINGLET, Fraction(-2), Chirality.LEFT, x_charge=x)
    elif name.startswith("E_R"):
        return Fermion(name, SU3.SINGLET, SU2.SINGLET, Fraction(-2), Chirality.RIGHT, x_charge=x)
    elif name.startswith("L_L"):
        return Fermion(name, SU3.SINGLET, SU2.DOUBLET, Fraction(-1), Chirality.LEFT, x_charge=x)
    elif name.startswith("L_R"):
        return Fermion(name, SU3.SINGLET, SU2.DOUBLET, Fraction(-1), Chirality.RIGHT, x_charge=x)
    elif name.startswith("D_L"):
        return Fermion(name, SU3.TRIPLET, SU2.SINGLET, Fraction(-2, 3), Chirality.LEFT, x_charge=x)
    elif name.startswith("D_R"):
        return Fermion(name, SU3.TRIPLET, SU2.SINGLET, Fraction(-2, 3), Chirality.RIGHT, x_charge=x)
    elif name.startswith("Q_L"):
        return Fermion(name, SU3.TRIPLET, SU2.DOUBLET, Fraction(1, 3), Chirality.LEFT, x_charge=x)
    elif name.startswith("Q_R"):
        return Fermion(name, SU3.TRIPLET, SU2.DOUBLET, Fraction(1, 3), Chirality.RIGHT, x_charge=x)
    else:
        raise ValueError(f"Unknown added fermion name pattern: {name}")


def _verify(charge_map, added):
    fermions = gauge_charge_fermions(charge_map)
    for name, x in added:
        fermions.append(_parse_fermion_from_name(name, x))
    return all(v == 0 for v in gauge_anomaly_coefficients(fermions).values())


def _make_added_fermions(comb, charges):
    n_N, n_E, n_L, n_D, n_Q = comb
    added = []
    idx = 0
    # N: singlet, singlet, Y=0
    for i in range(n_N):
        added.append(Fermion(f"N_{i+1}", SU3.SINGLET, SU2.SINGLET, Fraction(0), Chirality.RIGHT, x_charge=charges[idx]))
        idx += 1
    # E: singlet, singlet, Y=-2
    for i in range(n_E):
        added.append(Fermion(f"E_L_{i+1}", SU3.SINGLET, SU2.SINGLET, Fraction(-2), Chirality.LEFT, x_charge=charges[idx]))
        added.append(Fermion(f"E_R_{i+1}", SU3.SINGLET, SU2.SINGLET, Fraction(-2), Chirality.RIGHT, x_charge=charges[idx+1]))
        idx += 2
    # L: singlet, doublet, Y=-1
    for i in range(n_L):
        added.append(Fermion(f"L_L_{i+1}", SU3.SINGLET, SU2.DOUBLET, Fraction(-1), Chirality.LEFT, x_charge=charges[idx]))
        added.append(Fermion(f"L_R_{i+1}", SU3.SINGLET, SU2.DOUBLET, Fraction(-1), Chirality.RIGHT, x_charge=charges[idx+1]))
        idx += 2
    # D: triplet, singlet, Y=-2/3
    for i in range(n_D):
        added.append(Fermion(f"D_L_{i+1}", SU3.TRIPLET, SU2.SINGLET, Fraction(-2, 3), Chirality.LEFT, x_charge=charges[idx]))
        added.append(Fermion(f"D_R_{i+1}", SU3.TRIPLET, SU2.SINGLET, Fraction(-2, 3), Chirality.RIGHT, x_charge=charges[idx+1]))
        idx += 2
    # Q: triplet, doublet, Y=1/3
    for i in range(n_Q):
        added.append(Fermion(f"Q_L_{i+1}", SU3.TRIPLET, SU2.DOUBLET, Fraction(1, 3), Chirality.LEFT, x_charge=charges[idx]))
        added.append(Fermion(f"Q_R_{i+1}", SU3.TRIPLET, SU2.DOUBLET, Fraction(1, 3), Chirality.RIGHT, x_charge=charges[idx+1]))
        idx += 2
    return added


def _make_added_tuples(comb, charges):
    n_N, n_E, n_L, n_D, n_Q = comb
    added = []
    idx = 0
    # N
    for i in range(n_N):
        if n_N == 1 and (n_E + n_L + n_D + n_Q) == 0:
            name = "nu_R"
        else:
            name = f"nu_R_{i+1}"
        added.append((name, charges[idx]))
        idx += 1
    # E
    for i in range(n_E):
        suffix = "" if n_E == 1 else f"_{i+1}"
        added.append((f"E_L{suffix}", charges[idx]))
        added.append((f"E_R{suffix}", charges[idx+1]))
        idx += 2
    # L
    for i in range(n_L):
        suffix = "" if n_L == 1 else f"_{i+1}"
        added.append((f"L_L{suffix}", charges[idx]))
        added.append((f"L_R{suffix}", charges[idx+1]))
        idx += 2
    # D
    for i in range(n_D):
        suffix = "" if n_D == 1 else f"_{i+1}"
        added.append((f"D_L{suffix}", charges[idx]))
        added.append((f"D_R{suffix}", charges[idx+1]))
        idx += 2
    # Q
    for i in range(n_Q):
        suffix = "" if n_Q == 1 else f"_{i+1}"
        added.append((f"Q_L{suffix}", charges[idx]))
        added.append((f"Q_R{suffix}", charges[idx+1]))
        idx += 2
    return added


def _solve_combination(charge_map, comb, timeout_ms):
    n_N, n_E, n_L, n_D, n_Q = comb
    num_vars = n_N + 2 * (n_E + n_L + n_D + n_Q)
    xs = [z3.Real(f"X_{i}") for i in range(num_vars)]
    added_fermions = _make_added_fermions(comb, xs)
    all_fermions = list(gauge_charge_fermions(charge_map)) + added_fermions
    coeffs = gauge_anomaly_coefficients(all_fermions)

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    for key in ["SU3^2-X", "SU2^2-X", "grav^2-X", "Y^2-X", "Y-X^2", "X^3"]:
        solver.add(coeffs[key] == 0)
    # An N singlet must carry a nonzero U(1)' charge; each exotic PAIR must be
    # U(1)'-CHIRAL (X_L != X_R), else it is vector-like under U(1)' too and
    # contributes nothing -- inert padding that must not count as a real field.
    for i in range(n_N):
        solver.add(xs[i] != 0)
    p = n_N
    for _ in range(n_E + n_L + n_D + n_Q):
        solver.add(xs[p] != xs[p + 1])
        p += 2

    status = solver.check()
    if status == z3.unsat:
        return "unsat", None
    if status == z3.unknown:
        return "unknown", None
    try:
        model = solver.model()
        charges = []
        for x in xs:
            v = model.eval(x, model_completion=True)
            charges.append(_as_fraction(v))
        return "sat", charges
    except ValueError:
        return "sat-algebraic-model", None


def _comb_description(comb):
    n_N, n_E, n_L, n_D, n_Q = comb
    parts = []
    if n_N > 0:
        parts.append(f"{n_N}N")
    if n_E > 0:
        parts.append(f"{n_E}E")
    if n_L > 0:
        parts.append(f"{n_L}L")
    if n_D > 0:
        parts.append(f"{n_D}D")
    if n_Q > 0:
        parts.append(f"{n_Q}Q")
    desc = " + ".join(parts)
    fields = n_N + 2 * (n_E + n_L + n_D + n_Q)
    return f"{fields} field{'s' if fields > 1 else ''} ({desc})"


def get_combinations(max_added, menu=_DEFAULT_MENU):
    """Field-count combinations of the menu items, smallest first. Items not in
    ``menu`` are held at count zero, so the search uses only the active menu."""
    def rng(item, hi):
        return range(hi + 1) if item in menu else range(1)  # range(1) == {0}
    combs = []
    for n_N in rng("N", max_added):
        for n_E in rng("E", max_added // 2):
            for n_L in rng("L", max_added // 2):
                for n_D in rng("D", max_added // 2):
                    for n_Q in rng("Q", max_added // 2):
                        fields = n_N + 2 * (n_E + n_L + n_D + n_Q)
                        if 0 < fields <= max_added:
                            comb = (n_N, n_E, n_L, n_D, n_Q)
                            weights = ((n_N, 1), (n_E, 2), (n_L, 4), (n_D, 6), (n_Q, 12))
                            # cost: field count, then the MAX single-rep complexity
                            # used, then total rep complexity.
                            max_c = max((w for c, w in weights if c > 0), default=0)
                            total_c = sum(c * w for c, w in weights)
                            combs.append((fields, (max_c, total_c), comb))
    combs.sort(key=lambda x: (x[0], x[1], x[2]))
    return combs


_ITEM_COMB = {"N": (1, 0, 0, 0, 0), "E": (0, 1, 0, 0, 0), "L": (0, 0, 1, 0, 0),
              "D": (0, 0, 0, 1, 0), "Q": (0, 0, 0, 0, 1)}
_ITEM_KEYS = ("N", "E", "L", "D", "Q")
_ITEM_REACHABLE_CACHE = None


def _item_reachable():
    """X-sector anomalies each single menu item can contribute to, computed once
    by probing the item with a chiral charge split (X_L=1, X_R=0)."""
    global _ITEM_REACHABLE_CACHE
    if _ITEM_REACHABLE_CACHE is None:
        cache = {}
        for item, comb in _ITEM_COMB.items():
            nvars = comb[0] + 2 * sum(comb[1:])
            probe = [Fraction(1)] + [Fraction(0)] * (nvars - 1)
            coeffs = gauge_anomaly_coefficients(_make_added_fermions(comb, probe))
            cache[item] = frozenset(k for k in _X_ANOMALIES if coeffs[k] != 0)
        _ITEM_REACHABLE_CACHE = cache
    return _ITEM_REACHABLE_CACHE


def _comb_reachable(comb):
    """The X-sector anomalies a field-count combination can contribute to: the
    union of its items' reachable sets. A nonzero anomaly outside this set can
    never be cancelled by ``comb``, so the combination is provably impossible
    without invoking the solver -- a sound support-prune certificate rung."""
    reach = set()
    items = _item_reachable()
    for count, key in zip(comb, _ITEM_KEYS):
        if count:
            reach |= items[key]
    return reach


def _menu_touchable(menu):
    """The X-sector anomalies at least one item in ``menu`` can contribute to.
    Each item is probed with a chiral charge split (X_L=1, X_R=0), which lights
    up exactly the anomalies its representation can reach. Nonzero anomalies
    OUTSIDE this set are structural obstructions no menu addition can cancel
    (for the sterile-only menu this reproduces grav^2-X / X^3 reachability)."""
    touchable = set()
    for item in menu:
        comb = _ITEM_COMB[item]
        nvars = comb[0] + 2 * sum(comb[1:])
        probe = [Fraction(1)] + [Fraction(0)] * (nvars - 1)
        coeffs = gauge_anomaly_coefficients(_make_added_fermions(comb, probe))
        touchable |= {k for k in _X_ANOMALIES if coeffs[k] != 0}
    return touchable


def _validate_search_args(menu, max_added: int, timeout_ms: int) -> tuple:
    menu = tuple(menu)
    unknown = sorted(set(menu) - set(_ITEM_KEYS))
    if unknown:
        raise ValueError(f"unknown repair menu item(s): {', '.join(unknown)}")
    if max_added < 0:
        raise ValueError("max_added must be nonnegative")
    if timeout_ms < 0:
        raise ValueError("timeout_ms must be nonnegative")
    return menu


def classify_repair(repair_added):
    if not repair_added:
        return "sterile-repairable"
    has_N = any(name.startswith("nu_R") or name.startswith("N_") for name, _ in repair_added)
    has_E = any(name.startswith("E_") for name, _ in repair_added)
    has_L = any(name.startswith("L_") for name, _ in repair_added)
    has_D = any(name.startswith("D_") for name, _ in repair_added)
    has_Q = any(name.startswith("Q_") for name, _ in repair_added)
    if has_D or has_Q:
        return "colored-exotic-required"
    elif has_E or has_L:
        return "colorless-exotic-repairable"
    elif has_N:
        return "sterile-repairable"
    return "unknown"


def minimal_repair(charge_map, max_added: int = 4, timeout_ms: int = 10000,
                   menu=_DEFAULT_MENU, _solve_cache=None) -> Repair:
    """Search ``menu`` for a rational anomaly-cancelling repair.

    The menu contains sterile singlets N and SM-vector-like / U(1)'-chiral
    exotic pairs E, L, D, Q. Results are certified minimal only when every
    lower-cost rung is support-pruned or solver-unsat. If no repair is found,
    distinguish a structural block (a nonzero anomaly no menu item can cancel)
    from a budget limit (no repair found within max_added, although every
    failing anomaly is individually reachable by the menu).
    """
    menu = _validate_search_args(menu, max_added, timeout_ms)
    coeffs = gauge_anomaly_coefficients(gauge_charge_fermions(charge_map))

    sm_bad = sorted(n for n in _SM_SECTOR if coeffs[n] != 0)
    if sm_bad:
        return Repair(False, [], False, blocked_by=sm_bad,
                      note="input is not an anomaly-free SM generation (Y-sector "
                           "does not cancel); supply a full generation")

    if all(coeffs[k] == 0 for k in _X_ANOMALIES):
        return Repair(True, [], True, note="already anomaly-free; no repair needed",
                      certificate=RepairCertificate(True, [], "no repair needed"))

    combs = get_combinations(max_added, menu)
    nonzero = {k for k in _X_ANOMALIES if coeffs[k] != 0}
    cache = {} if _solve_cache is None else _solve_cache
    rungs = []
    winning_comb = None
    winning_charges = None

    for fields, complexity, comb in combs:
        if comb in cache:
            verdict, charges = cache[comb]
        elif not nonzero <= _comb_reachable(comb):
            # A nonzero anomaly is unreachable by this combination: provably
            # impossible, no solver call needed (and a genuine certificate rung).
            verdict, charges = "support-prune", None
            cache[comb] = (verdict, charges)
        else:
            verdict, charges = _solve_combination(charge_map, comb, timeout_ms)
            cache[comb] = (verdict, charges)
        if verdict == "sat":
            winning_comb = comb
            winning_charges = charges
            break
        else:
            rungs.append((_comb_description(comb), verdict))

    if winning_comb is None:
        # Distinguish a structural block from a budget limit: a nonzero anomaly
        # that NO menu item can contribute to can never be repaired in this menu;
        # otherwise no support-level obstruction was detected, but no joint repair
        # was found within max_added fields.
        touchable = _menu_touchable(menu)
        blocked = sorted(k for k in _X_ANOMALIES if coeffs[k] != 0 and k not in touchable)
        if blocked:
            return Repair(False, [], False, blocked_by=blocked,
                          note="blocked: nonzero anomalies no item in the menu can cancel ("
                               + ", ".join(blocked) + ")",
                          certificate=RepairCertificate(False, rungs,
                                                        "structurally blocked in this menu"))
        return Repair(False, [], False, blocked_by=[],
                      note=f"no repair within {max_added} fields, but every failing anomaly "
                           "is reachable by the menu -- budget-limited, raise max_added",
                      certificate=RepairCertificate(False, rungs,
                                                    "budget-limited; raise max_added"))

    added = _make_added_tuples(winning_comb, winning_charges)
    if not _verify(charge_map, added):
        return Repair(False, [], False, note="candidate charges failed re-verification",
                      certificate=RepairCertificate(False, rungs, "re-verification failed"))

    certified = all(verdict in _CERTIFYING_VERDICTS for _, verdict in rungs)
    if certified and rungs:
        cnote = ("no lower-cost repair exists in the menu "
                 "(prior rungs support-pruned or solver-unsat)")
    elif certified:
        cnote = "minimal: a smaller (zero-field) repair would leave the input inconsistent"
    else:
        undecided = [r for r in rungs if r[1] not in _CERTIFYING_VERDICTS]
        cnote = ("first rational repair found, but minimality not certified at "
                 "prior rungs "
                 + ", ".join(f"{s} ({v})" for s, v in undecided))
    cert = RepairCertificate(certified, rungs, cnote)

    rep_type = classify_repair(added)
    if rep_type == "sterile-repairable":
        note = f"add {winning_comb[0]} sterile RH singlet(s) per generation"
    elif rep_type == "colorless-exotic-repairable":
        note = "add SM-vector-like but U(1)'-chiral colorless exotics"
    elif rep_type == "colored-exotic-required":
        note = "add SM-vector-like but U(1)'-chiral colored exotics"
    else:
        note = "repaired successfully"

    return Repair(True, added, True, certificate=cert, note=note)


_TIERS = (
    ("sterile-repairable", ("N",)),
    ("colorless-exotic-repairable", ("N", "E", "L")),
    ("colored-exotic-required", ("N", "E", "L", "D", "Q")),
)


@dataclass
class TieredRepair:
    tier: str       # first tier with a repair; or "already-consistent" / "still-blocked"
    repair: Repair  # the Repair from that tier (the full-menu attempt if still blocked)


def tiered_repair(charge_map, max_added: int = 6, timeout_ms: int = 10000) -> TieredRepair:
    """Report the first menu tier that yields a rational repair for ``charge_map``:

        sterile (N)  <  colorless exotics (N,E,L)  <  colored exotics (N,E,L,D,Q)

    so the result classifies the charge as sterile-repairable,
    colorless-exotic-repairable, colored-exotic-required, or still-blocked (within
    the budget). The repair's certificate says whether minimum cost is proved
    within the declared menu/cost ordering; no result is a uniqueness,
    naturalness, or existence claim."""
    last = None
    cache = {}  # share solver verdicts across tiers (lower-tier combos recur)
    for label, menu in _TIERS:
        r = minimal_repair(charge_map, max_added=max_added, timeout_ms=timeout_ms,
                           menu=menu, _solve_cache=cache)
        if r.repairable:
            return TieredRepair("already-consistent" if not r.added else label, r)
        last = r
    return TieredRepair("still-blocked", last)
