"""Soft-theorem layer: symbolic leading factors and Ward checks.

This is not a cross-section calculator. It verifies the universal/charge-
weighted coefficient patterns that a massless candidate mediator must obey.
"""
from fractions import Fraction as F

import pytest

from nsm.amplitudes import (
    soft_theorem_report,
    gauge_charge_for_particle,
    leading_soft_factor,
)
from nsm.extensions import Candidate


_BL = (("Q_L", F(1, 3)), ("u_R", F(1, 3)), ("d_R", F(1, 3)),
       ("L_L", F(-1)), ("e_R", F(-1)), ("nu_R", F(-1)))


def graviton():
    return Candidate("G", spin=F(2), mass_mev=0.0, coupling="stress_energy",
                     coupling_scale="planck")


def zprime(charge_map=_BL):
    return Candidate("Zp", spin=F(1), mass_mev=0.0, coupling="gauge_charge",
                     charge_map=tuple(charge_map), coupling_scale="weak")


def test_graviton_soft_factor_is_rank_two_and_universal():
    sf = leading_soft_factor(graviton(), ["e-", "e+"], ["mu-", "mu+"])
    assert sf.current_rank == 2
    assert sf.ward_identity.passed
    assert "universal" in sf.note
    assert len(sf.terms) == 4
    assert {term.coefficient for term in sf.terms} == {F(-1), F(1)}
    assert all("eps_mu_nu" in term.numerator for term in sf.terms)


def test_zprime_soft_factor_is_charge_weighted():
    sf = leading_soft_factor(zprime(), ["e-", "e+"], ["mu-", "mu+"])
    assert sf.current_rank == 1
    assert sf.ward_identity.passed
    assert "charge-weighted" in sf.note
    assert [term.coefficient for term in sf.terms] == [F(1), F(-1), F(-1), F(1)]
    assert all("eps_mu_nu" not in term.numerator for term in sf.terms)


def test_gauge_ward_identity_fails_when_charge_flow_does_not_cancel():
    report = soft_theorem_report(zprime(), ["e-"], ["gamma"])
    assert not report.passed
    assert "charge flow" in report.reason


def test_graviton_factorization_is_not_charge_sensitive():
    report = soft_theorem_report(graviton(), ["e-"], ["nu_e"])
    assert report.passed
    assert "momentum conservation" in report.reason


def test_zprime_antiparticle_charge_is_opposite():
    zp = zprime()
    assert gauge_charge_for_particle(zp, "e-") == F(-1)
    assert gauge_charge_for_particle(zp, "e+") == F(1)
    assert gauge_charge_for_particle(zp, "u") == F(1, 3)
    assert gauge_charge_for_particle(zp, "u_bar") == F(-1, 3)


def test_leptonic_zprime_is_not_a_valid_exposed_mediator_without_anomaly_repair():
    leptonic = zprime((("L_L", F(1)), ("e_R", F(1))))
    with pytest.raises(ValueError, match="not valid"):
        leading_soft_factor(leptonic, ["u", "u_bar"], ["e-", "e+"])


def test_chiral_charge_assignment_is_rejected_as_not_vector_like():
    chiral = zprime((("L_L", F(1)), ("e_R", F(2))))
    with pytest.raises(ValueError, match="chiral"):
        gauge_charge_for_particle(chiral, "e-")


def test_anomalous_soft_candidate_is_rejected_before_factor_construction():
    anomalous = zprime((("Q_L", F(1, 3)), ("u_R", F(1, 3)), ("d_R", F(1, 3)),
                        ("L_L", F(-1)), ("e_R", F(-1))))
    with pytest.raises(ValueError, match="not valid"):
        leading_soft_factor(anomalous, ["e-", "e+"], ["mu-", "mu+"])


def test_purely_chiral_zero_vs_nonzero_coupling_is_rejected():
    # X=0 on L_L but X=1 on e_R is a purely right-handed (chiral) coupling to the
    # electron; the leading vector soft factor must not silently treat it as a
    # vector charge of 1. (Requires the omitted/zero L_L charge to be seen.)
    chiral = zprime((("e_R", F(1)),))
    with pytest.raises(ValueError, match="chiral"):
        gauge_charge_for_particle(chiral, "e-")


def test_massive_candidate_has_no_massless_soft_factor():
    heavy = Candidate("Zp", spin=F(1), mass_mev=1000.0,
                      coupling="gauge_charge", charge_map=_BL)
    with pytest.raises(ValueError, match="massless"):
        leading_soft_factor(heavy, ["e-", "e+"], ["mu-", "mu+"])
