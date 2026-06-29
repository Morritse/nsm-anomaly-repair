"""Minimal and tiered repair search for inconsistent gauge extensions."""
from fractions import Fraction as F

from nsm.repair import minimal_repair, tiered_repair, Repair

_BL = (("Q_L", F(1, 3)), ("u_R", F(1, 3)), ("d_R", F(1, 3)),
       ("L_L", F(-1)), ("e_R", F(-1)))


def test_gauged_b_minus_l_is_repaired_by_one_sterile_neutrino():
    r = minimal_repair(_BL)
    assert r.repairable
    assert len(r.added) == 1
    name, x = r.added[0]
    assert x == F(-1)        # nu_R with X = -1, rediscovered
    assert r.verified        # every anomaly re-checked to cancel


def test_an_already_consistent_charge_needs_no_repair():
    r = minimal_repair(_BL + (("nu_R", F(-1)),))
    assert r.repairable
    assert r.added == []
    assert r.verified


def test_a_colored_anomaly_blocks_a_colorless_menu_but_not_the_full_menu():
    # X only on Q_L breaks the colored SU(3)^2-X anomaly.
    cm = (("Q_L", F(1)), ("u_R", F(0)), ("d_R", F(0)), ("L_L", F(0)), ("e_R", F(0)))
    # Colorless menu (N, E, L): no item carries color, so SU(3)^2-X can never be
    # cancelled -- a genuine structural block, named in blocked_by.
    colorless = minimal_repair(cm, menu=("N", "E", "L"))
    assert not colorless.repairable
    assert "SU3^2-X" in colorless.blocked_by
    # Full menu (with the colored D/Q): SU(3)^2-X is reachable, so it is NOT a
    # structural block -- the result is repaired or merely budget-limited.
    full = minimal_repair(cm, menu=("N", "E", "L", "D", "Q"))
    assert "SU3^2-X" not in full.blocked_by


def test_a_partial_charge_map_is_normalized_to_a_full_generation():
    # Omitted SM fields mean X=0 (not "removed from the spectrum"): a partial map
    # like u_R-only is the SAME well-posed problem as the explicit full generation
    # with zeros, NOT flagged as an ill-posed / incomplete input.
    partial = minimal_repair((("u_R", F(1)),))
    full = minimal_repair((("Q_L", F(0)), ("u_R", F(1)), ("d_R", F(0)),
                           ("L_L", F(0)), ("e_R", F(0))))
    assert "SM generation" not in partial.note
    assert partial.note == full.note
    assert partial.repairable == full.repairable


def test_certificate_for_b_minus_l_is_certified_minimal():
    r = minimal_repair(_BL)
    assert r.repairable
    assert r.certificate.minimal              # one field; nothing smaller to beat it
    assert r.certificate.rungs == []          # no positive smaller size


def test_search_ladder_certifies_a_size_two_repair():
    # g=3, c=9 has no size-1 solution (3^3 != 9) but a rational size-2 one (1,2);
    # the size-1 rung is a genuine 'unsat' certificate of minimality.
    from nsm.repair import _search_sterile
    charges, n, rungs = _search_sterile(F(3), F(9), 4, 5000)
    assert n == 2 and sorted(charges) == [F(1), F(2)]
    assert rungs == [(1, "unsat")]


def test_search_ladder_flags_unsolved_sizes_honestly():
    # g=-2, c=4: sizes 1 and 2 are provably impossible (unsat); at size 3 the
    # solver may return an algebraic model, which leaves rational solubility
    # unresolved and must never be mistaken for a certificate.
    from nsm.repair import _search_sterile
    charges, n, rungs = _search_sterile(F(-2), F(4), 3, 5000)
    rung = dict(rungs)
    assert rung.get(1) == "unsat" and rung.get(2) == "unsat"
    if charges is None:
        assert rung.get(3) in ("sat-algebraic-model", "unknown")
    else:
        assert n == 3 and all(isinstance(x, F) for x in charges)


# --- tiered repair: minimum menu tier that fixes a charge --------------------

_eR = (("Q_L", F(0)), ("u_R", F(0)), ("d_R", F(0)), ("L_L", F(0)), ("e_R", F(1)))
_QL = (("Q_L", F(1)), ("u_R", F(0)), ("d_R", F(0)), ("L_L", F(0)), ("e_R", F(0)))


def test_tiered_b_minus_l_is_sterile_repairable():
    t = tiered_repair(_BL)
    assert t.tier == "sterile-repairable"
    assert t.repair.added == [("nu_R", F(-1))]


def test_tiered_already_consistent_charge():
    t = tiered_repair(_BL + (("nu_R", F(-1)),))
    assert t.tier == "already-consistent"
    assert t.repair.added == []


def test_tiered_e_R_charge_needs_a_colorless_exotic():
    # X only on e_R breaks Y^2-X, which no sterile singlet can touch; a colorless
    # hypercharged exotic (E) fixes it -- one tier above sterile.
    assert tiered_repair(_eR).tier == "colorless-exotic-repairable"


def test_support_pruned_rungs_are_recorded_distinctly_and_certify():
    r = minimal_repair(_eR, max_added=2, menu=("N", "E"), timeout_ms=4000)
    assert r.repairable
    assert r.certificate.minimal
    assert ("1 field (1N)", "support-prune") in r.certificate.rungs
    assert "support-pruned" in r.certificate.note


def test_minimal_repair_validates_search_arguments():
    import pytest
    with pytest.raises(ValueError, match="unknown repair menu"):
        minimal_repair(_eR, menu=("N", "X"))
    with pytest.raises(ValueError, match="max_added"):
        minimal_repair(_eR, max_added=-1)
    with pytest.raises(ValueError, match="timeout_ms"):
        minimal_repair(_eR, timeout_ms=-1)


def test_tiered_Q_L_charge_requires_a_colored_exotic():
    # X only on Q_L breaks the colored SU(3)^2-X; only D/Q can reach it.
    assert tiered_repair(_QL).tier == "colored-exotic-required"


def test_repair_pairs_are_u1_chiral_not_inert():
    # Every exotic pair in a repair must be U(1)'-chiral (X_L != X_R); an equal
    # pair would be vector-like under U(1)' too and contribute nothing.
    added = dict(tiered_repair(_eR).repair.added)
    assert added["E_L"] != added["E_R"]


# --- reachability pruning of combinations (a sound support certificate) -------

def test_comb_reachable_reports_the_anomalies_a_combination_can_touch():
    from nsm.repair import _comb_reachable, _X_ANOMALIES
    assert _comb_reachable((0, 0, 0, 0, 0)) == set()                 # no fields
    assert _comb_reachable((1, 0, 0, 0, 0)) == {"grav^2-X", "X^3"}   # sterile N
    # one E pair (hypercharged colorless): mixed + grav + cubic, no color/weak
    assert _comb_reachable((0, 1, 0, 0, 0)) == {"grav^2-X", "Y^2-X", "Y-X^2", "X^3"}
    assert _comb_reachable((0, 0, 0, 0, 1)) == set(_X_ANOMALIES)     # a Q reaches all six


def test_unreachable_combination_is_genuinely_unsat_so_pruning_is_sound():
    # The prune records a support-prune certificate WITHOUT a solver call
    # whenever a combination cannot reach a nonzero anomaly; that must agree
    # with the solver, or the certificate would be unsound.
    from nsm.repair import _comb_reachable, _solve_combination
    cm = (("Q_L", F(1)), ("u_R", F(0)), ("d_R", F(0)), ("L_L", F(0)), ("e_R", F(0)))
    e_pair = (0, 1, 0, 0, 0)            # colorless: cannot reach the colored SU(3)^2-X
    assert "SU3^2-X" not in _comb_reachable(e_pair)
    assert _solve_combination(cm, e_pair, 4000)[0] == "unsat"   # solver agrees
    assert "SU3^2-X" in _comb_reachable((0, 0, 0, 0, 1))        # a Q pair can reach it


def test_repair_map_classifies_representative_charges():
    # The reproducible scan's classifier primitive, on a small fixed set:
    # 3*(B-L) -> sterile, X on e_R -> colorless exotic, X on Q_L -> colored exotic.
    from nsm.repair_map import classify_grid
    counts, _ = classify_grid([(1, 1, 1, -3, -3), (0, 0, 0, 0, 1), (1, 0, 0, 0, 0)],
                              max_added=4, timeout_ms=4000)
    assert counts["sterile-repairable"] == 1
    assert counts["colorless-exotic-repairable"] == 1
    assert counts["colored-exotic-required"] == 1
