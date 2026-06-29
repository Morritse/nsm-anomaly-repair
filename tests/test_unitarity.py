"""Structural unitarity cuts (kinematics-free).

A cut state is an intermediate set the process can pass through on BOTH sides
(optical-theorem topology). These tests pin which intermediate states are
candidate cut topologies and their nominal mass-threshold ordering -- not rates,
not "dominant" contributions, not a loop calculation.
"""
from fractions import Fraction as F

import pytest

from nsm.unitarity import cut_states, threshold_order, CutState
from nsm.extensions import Candidate, register_candidate, candidate_vertices


def _sets(cuts):
    return {c.particles for c in cuts}


def test_dilepton_cut_states_include_the_expected_thresholds():
    s = _sets(cut_states(["e-", "e+"], ["mu-", "mu+"]))
    for expected in [("e+", "e-"), ("mu+", "mu-"), ("gamma", "gamma"),
                     ("W+", "W-"), ("nu_e", "nu_e_bar")]:
        assert tuple(sorted(expected)) in s, expected


def test_two_photon_pair_production_cuts_include_charged_fermion_pairs():
    s = _sets(cut_states(["gamma", "gamma"], ["e-", "e+"]))
    assert tuple(sorted(("e+", "e-"))) in s
    assert tuple(sorted(("mu+", "mu-"))) in s


def test_muon_decay_has_a_charged_current_consistent_cut():
    cuts = cut_states(["mu-"], ["e-", "nu_e_bar", "nu_mu"])
    assert cuts  # non-empty
    assert any("W+" in c.particles or "W-" in c.particles for c in cuts)


def test_a_charge_violating_process_has_no_cuts():
    # e- e- -> mu- mu+ : charge -2 vs 0; no intermediate state suits both sides.
    assert cut_states(["e-", "e-"], ["mu-", "mu+"]) == []


def test_thresholds_order_by_total_mass():
    ordered = threshold_order(cut_states(["e-", "e+"], ["mu-", "mu+"]))
    thresholds = [c.threshold_mev for c in ordered]
    assert thresholds == sorted(thresholds)
    assert ordered[0].threshold_mev <= ordered[-1].threshold_mev


def test_cut_records_reachability_and_threshold():
    cuts = cut_states(["e-", "e+"], ["mu-", "mu+"])
    gg = next(c for c in cuts if c.particles == ("gamma", "gamma"))
    assert gg.left_reachable and gg.right_reachable
    assert gg.threshold_mev == 0   # two massless photons


def test_a_candidate_zprime_adds_a_cut_channel():
    BL = (("Q_L", F(1, 3)), ("u_R", F(1, 3)), ("d_R", F(1, 3)),
          ("L_L", F(-1)), ("e_R", F(-1)), ("nu_R", F(-1)))
    zp = Candidate("Zp", spin=F(1), mass_mev=2000.0, coupling="gauge_charge",
                   charge_map=BL)
    register_candidate(zp)
    s = _sets(cut_states(["e-", "e+"], ["mu-", "mu+"], extra_vertices=candidate_vertices(zp)))
    assert any("Zp" in p for p in s)


def test_oversized_cut_request_raises_rather_than_truncating():
    # Beyond the max_external reach, larger cuts would be silently dropped;
    # the engine raises instead of returning a quietly-truncated answer.
    with pytest.raises(ValueError, match="reachable cut size"):
        cut_states(["e-", "e+"], ["mu-", "mu+"], max_cut_size=6)
