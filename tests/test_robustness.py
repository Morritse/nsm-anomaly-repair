"""Solver robustness: no hangs, no silent truncation, honest "undecided".

These guard the soundness fixes: enumerate_solutions must always terminate and
must never report a truncated family as if it were complete.
"""
from nsm.reps import SU3, SU2, Chirality
from nsm.sm import standard_model_skeleton
from nsm.derive import derive_hypercharges, enumerate_solutions


def test_enumerate_terminates_on_multiple_free_directions():
    # Two free hypercharge directions (sterile nu_R + chi_L) make z3's
    # nonlinear search solver-undetermined within the timeout. It must RETURN
    # (this once hung forever) and must NOT claim completeness.
    model = standard_model_skeleton()
    model.add_fermion("nu_R", SU3.SINGLET, SU2.SINGLET, Chirality.RIGHT)
    model.add_fermion("chi_L", SU3.SINGLET, SU2.SINGLET, Chirality.LEFT)
    enum = enumerate_solutions(model, limit=16, timeout_ms=2000)
    assert enum.complete is False
    assert enum.reason in ("undecided", "limit")


def test_finite_solution_set_is_reported_complete():
    # The genuinely finite SM case must still come back proven-complete.
    enum = enumerate_solutions(standard_model_skeleton(), timeout_ms=5000)
    assert enum.complete is True
    assert enum.reason == "exhausted"


def test_su3_squared_u1_constraint_is_redundant():
    # SU(3)^2-U(1) is auto-satisfied by the Yukawa relations: dropping it
    # changes nothing, the SM stays uniquely determined.
    result = derive_hypercharges(standard_model_skeleton(), omit={"SU3^2-U1"})
    assert result.is_unique is True


def test_dropping_the_cubic_keeps_uniqueness_via_the_linear_grav_anomaly():
    # U(1)^3 is redundant once the linear grav^2-U(1) anomaly pins Y_Q -- and
    # the linear constraint keeps the uniqueness check easy for z3.
    assert derive_hypercharges(standard_model_skeleton(), omit={"U1^3"}).is_unique is True


def test_dropping_grav_alone_is_unique_or_undecided_but_never_false():
    # With only the cubic (3Y_Q-1)^3 = 0 left to pin Y_Q the solution is still
    # genuinely unique, but proving it requires nonlinear reasoning z3 may not
    # finish in the timeout. The sound invariant: it is True or honestly None,
    # NEVER False (there is no second solution to find).
    r = derive_hypercharges(standard_model_skeleton(), omit={"grav^2-U1"}, timeout_ms=2000)
    assert r.satisfiable is True
    assert r.is_unique in (True, None)


def test_dropping_both_grav_and_cubic_loses_uniqueness():
    # grav^2-U(1) and U(1)^3 are mutually redundant but jointly load-bearing:
    # remove both and Y_Q is left free.
    both = derive_hypercharges(standard_model_skeleton(), omit={"grav^2-U1", "U1^3"})
    assert both.satisfiable is True
    assert both.is_unique is False
