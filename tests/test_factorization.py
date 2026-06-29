"""Factorization-channel enumeration.

Locality: a tree amplitude's poles sit only where an internal line goes
on-shell. These tests pin WHICH internal particles a process factorizes through
(s- and t-channel), at the level of vertices/quantum numbers. No residues are
evaluated -- that is the separate kinematics fork.
"""
from fractions import Fraction as F

from nsm.factorization import factorization_channels, factorization_mediators
from nsm.extensions import Candidate, register_candidate, candidate_vertices

_BL = [("Q_L", F(1, 3)), ("u_R", F(1, 3)), ("d_R", F(1, 3)),
       ("L_L", F(-1)), ("e_R", F(-1)), ("nu_R", F(-1))]


def test_dilepton_factorizes_through_the_neutral_bosons():
    meds = factorization_mediators(["e-", "e+"], ["mu-", "mu+"])
    assert {"gamma", "Z", "h"} <= meds


def test_a_candidate_zprime_opens_an_extra_channel():
    zp = Candidate("Zp", spin=F(1), mass_mev=2000.0, coupling="gauge_charge",
                   charge_map=tuple(_BL))
    register_candidate(zp)
    meds = factorization_mediators(["e-", "e+"], ["mu-", "mu+"],
                                   extra_vertices=candidate_vertices(zp))
    assert "Zp" in meds
    assert {"gamma", "Z", "h"} <= meds   # the SM channels are still present


def test_graviton_opens_a_channel_in_two_photon_pair_production():
    G = Candidate("G", spin=F(2), mass_mev=0.0, coupling="stress_energy")
    register_candidate(G)
    meds = factorization_mediators(["gamma", "gamma"], ["e-", "e+"],
                                   extra_vertices=candidate_vertices(G))
    assert "G" in meds


def test_one_to_one_has_no_factorization_channel():
    assert factorization_channels(["e-"], ["gamma"]) == []


def test_factorization_raises_when_external_budget_is_exceeded():
    import pytest
    with pytest.raises(ValueError, match="max_external"):
        factorization_channels(["e-", "e+"], ["e-", "e+", "e-", "e+", "e-", "e+"])
    with pytest.raises(ValueError, match="max_external"):
        factorization_mediators(["e-", "e+"], ["mu-", "mu+"], max_external=3)


def test_a_flavor_violating_process_factorizes_through_nothing():
    assert factorization_mediators(["e-", "e+"], ["mu-", "tau+"]) == set()


def test_a_channel_partitions_all_the_external_legs():
    chans = factorization_channels(["e-", "e+"], ["mu-", "mu+"])
    gammas = [c for c in chans if c.mediator == "gamma"]
    assert gammas
    c = gammas[0]
    assert sorted(c.left + c.right) == sorted(["e+", "e-", "mu-", "mu+"])
