"""Forced vs. free: isolating what each principle buys you.

With the full Standard Model skeleton the hypercharges are unique. But strip the
Yukawa couplings and keep only anomaly cancellation (normalized by fixing Y_Q),
and a *discrete* ambiguity survives: the solver cannot tell which colored
singlet is "u_R" and which is "d_R" -- their hypercharges (4/3, -2/3) may be
swapped. It is Yukawa gauge invariance (Y_d = Y_Q - Y_H) that picks the
assignment. The engine reports "not enough information" as a real result.
"""
from fractions import Fraction as F

from nsm.sm import standard_model_skeleton, anomaly_only_skeleton
from nsm.derive import derive_hypercharges, enumerate_solutions


def test_full_skeleton_is_unique():
    assert derive_hypercharges(standard_model_skeleton()).is_unique is True


def test_anomalies_alone_leave_an_up_down_swap_ambiguity():
    model = anomaly_only_skeleton()
    result = derive_hypercharges(model)
    assert result.satisfiable
    assert result.is_unique is False


def test_the_ambiguity_is_exactly_the_two_up_down_assignments():
    enum = enumerate_solutions(anomaly_only_skeleton())
    # complete=True means the solver PROVED there are no further solutions.
    assert enum.complete is True
    assert enum.reason == "exhausted"
    assert len(enum.solutions) == 2
    up_down = sorted((s["u_R"], s["d_R"]) for s in enum.solutions)
    assert up_down == [(F(-2, 3), F(4, 3)), (F(4, 3), F(-2, 3))]
    # Everything else is pinned regardless of the swap.
    for s in enum.solutions:
        assert s["Q_L"] == F(1, 3)
        assert s["L_L"] == F(-1)
        assert s["e_R"] == F(-2)
