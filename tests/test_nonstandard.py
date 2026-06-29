"""Non-standard spectra: the same engine, asked questions not in the SM table.

These cases have no "muon row" to look up -- the answers can only come from
computing gauge consistency. They also surface a genuinely subtle result that a
lookup table would get wrong.
"""
from fractions import Fraction as F

from nsm.reps import SU3, SU2, Chirality
from nsm.sm import standard_model_skeleton, Yukawa
from nsm.derive import derive_hypercharges
from nsm.anomalies import anomaly_coefficients, witten_anomaly_free


def test_adding_a_free_right_handed_neutrino_unfreezes_the_hypercharges():
    # A sterile nu_R (color/weak singlet) with no coupling contributes nothing
    # to SU(3)/SU(2) anomalies, but opens a flat direction (gauged B-L): the
    # hypercharges are no longer uniquely fixed. A lookup table would wrongly
    # just report "nu charge = 0"; consistency says the assignment is now free.
    model = standard_model_skeleton()
    model.add_fermion("nu_R", SU3.SINGLET, SU2.SINGLET, Chirality.RIGHT)
    result = derive_hypercharges(model)
    assert result.satisfiable
    assert result.is_unique is False


def test_a_majorana_mass_refreezes_them_and_makes_the_neutrino_neutral():
    # A Majorana mass nu_R nu_R requires 2*Y_nuR = 0, breaking B-L. That lifts
    # the flat direction: the Standard Model is recovered uniquely and the
    # neutrino is forced neutral.
    model = standard_model_skeleton()
    model.add_fermion("nu_R", SU3.SINGLET, SU2.SINGLET, Chirality.RIGHT)
    model.yukawas.append(Yukawa("majorana", (("nu_R", F(2)),)))
    result = derive_hypercharges(model)
    assert result.is_unique is True
    assert result.hypercharges["nu_R"] == F(0)
    assert result.charges["nu_R"] == F(0)        # derived electric charge: neutral
    assert result.hypercharges["Q_L"] == F(1, 3)  # full SM recovered


def test_an_unpaired_colored_fermion_is_rejected_by_the_su3_cubic_anomaly():
    # Adding a single colored Weyl fermion with no mirror partner makes the
    # SU(3)^3 anomaly non-zero: the spectrum is inconsistent, period.
    model = standard_model_skeleton()
    model.add_fermion("X_R", SU3.TRIPLET, SU2.SINGLET, Chirality.RIGHT)
    # The SU(3)^3 coefficient is the specific culprit...
    assert anomaly_coefficients(model.fermions)["SU3^3"] == F(-1)
    assert derive_hypercharges(model).satisfiable is False
    # ...and the spectrum becomes consistent again precisely when that one
    # constraint is lifted, confirming attribution rather than coincidence.
    assert derive_hypercharges(model, omit={"SU3^3"}).satisfiable is True


def test_witten_global_anomaly_rejects_an_odd_number_of_doublets():
    model = standard_model_skeleton()
    assert witten_anomaly_free(model.fermions)
    model.add_fermion("Lprime_L", SU3.SINGLET, SU2.DOUBLET, Chirality.LEFT)
    assert not witten_anomaly_free(model.fermions)
    assert derive_hypercharges(model).satisfiable is False
