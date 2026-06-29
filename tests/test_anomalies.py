"""The generic anomaly engine.

Anomaly coefficients are computed from each fermion's (color rep, weak rep,
hypercharge, chirality) -- NOT hand-written per theory. A consistent chiral
gauge theory must have every coefficient vanish. We check that the known
Standard Model spectrum cancels, and that a deliberately broken spectrum does
not -- proving the engine actually computes, rather than asserting, consistency.
"""
from fractions import Fraction as F

from nsm.reps import SU3, SU2, Chirality, Fermion
from nsm.anomalies import anomaly_coefficients, witten_doublet_count


def sm_one_generation():
    """One generation, with the *known* hypercharges plugged in (convention
    Q = T3 + Y/2). Right-handed fields are stored as right-handed Weyl fields."""
    return [
        Fermion("Q_L", SU3.TRIPLET, SU2.DOUBLET, F(1, 3), Chirality.LEFT),
        Fermion("u_R", SU3.TRIPLET, SU2.SINGLET, F(4, 3), Chirality.RIGHT),
        Fermion("d_R", SU3.TRIPLET, SU2.SINGLET, F(-2, 3), Chirality.RIGHT),
        Fermion("L_L", SU3.SINGLET, SU2.DOUBLET, F(-1), Chirality.LEFT),
        Fermion("e_R", SU3.SINGLET, SU2.SINGLET, F(-2), Chirality.RIGHT),
    ]


def test_standard_model_cancels_every_gauge_anomaly():
    coeffs = anomaly_coefficients(sm_one_generation())
    # SU(3)^2-U(1), SU(2)^2-U(1), grav^2-U(1), U(1)^3, SU(3)^3
    for name, value in coeffs.items():
        assert value == 0, f"{name} should cancel but is {value}"


def test_all_five_coefficients_are_reported():
    coeffs = anomaly_coefficients(sm_one_generation())
    assert set(coeffs) == {"SU3^2-U1", "SU2^2-U1", "grav^2-U1", "U1^3", "SU3^3"}


def test_witten_doublet_count_is_even_for_standard_model():
    # Q_L is a color triplet weak doublet (3 doublets) + L_L (1 doublet) = 4
    assert witten_doublet_count(sm_one_generation()) == 4


def test_lone_charged_weyl_fermion_has_gravitational_anomaly():
    # A single left-handed fermion charged only under U(1) cannot cancel the
    # mixed gravitational anomaly: Tr(Y) != 0.
    rogue = [Fermion("X_L", SU3.SINGLET, SU2.SINGLET, F(1), Chirality.LEFT)]
    coeffs = anomaly_coefficients(rogue)
    assert coeffs["grav^2-U1"] == F(1)
    assert coeffs["U1^3"] == F(1)


def test_right_handed_field_enters_with_opposite_sign():
    # Same quantum numbers, opposite chirality => opposite contribution.
    left = [Fermion("a", SU3.SINGLET, SU2.SINGLET, F(1), Chirality.LEFT)]
    right = [Fermion("b", SU3.SINGLET, SU2.SINGLET, F(1), Chirality.RIGHT)]
    assert anomaly_coefficients(left)["grav^2-U1"] == -anomaly_coefficients(right)["grav^2-U1"]


def test_nonzero_coefficients_match_hand_computation():
    # A spectrum that does NOT cancel, pinned to hand-computed values. This
    # catches a wrong Dynkin/dimension/multiplicity factor that would still
    # happen to vanish for the (highly symmetric) Standard Model.
    #   colored doublet Q (3,2,Y=2,L) and lepton doublet L (1,2,Y=5,L)
    spectrum = [
        Fermion("Q", SU3.TRIPLET, SU2.DOUBLET, F(2), Chirality.LEFT),
        Fermion("L", SU3.SINGLET, SU2.DOUBLET, F(5), Chirality.LEFT),
    ]
    coeffs = anomaly_coefficients(spectrum)
    assert coeffs["SU2^2-U1"] == F(11, 2)   # 3*(1/2)*2 + 1*(1/2)*5
    assert coeffs["SU3^2-U1"] == F(2)        # 2*(1/2)*2
    assert coeffs["grav^2-U1"] == F(22)      # 3*2*2 + 1*2*5
    assert coeffs["U1^3"] == F(298)          # 6*8 + 2*125
    assert coeffs["SU3^3"] == F(2)           # 2*A(3) = 2*1


def test_antitriplet_cancels_the_su3_cubic_of_a_triplet():
    # A conjugate (3bar) pair makes the color sector vector-like: SU(3)^3 = 0.
    triplet_only = [Fermion("q", SU3.TRIPLET, SU2.SINGLET, F(0), Chirality.LEFT)]
    assert anomaly_coefficients(triplet_only)["SU3^3"] == F(1)
    vector_like = triplet_only + [
        Fermion("qbar", SU3.ANTITRIPLET, SU2.SINGLET, F(0), Chirality.LEFT)
    ]
    assert anomaly_coefficients(vector_like)["SU3^3"] == F(0)


def test_empty_spectrum_is_anomaly_free():
    coeffs = anomaly_coefficients([])
    assert all(v == 0 for v in coeffs.values())
    assert witten_doublet_count([]) == 0
