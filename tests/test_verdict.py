"""The layered verdict -- the whole point of the dynamics layer.

A process is not just "allowed" or "forbidden": the engine says WHY, combining
exact symmetries, accidental symmetries, kinematics, and tree-level vertex
reachability. These are the three milestone queries plus boundary cases.
"""
from nsm.verdict import classify, Verdict


def test_muon_decay_is_allowed_at_tree_level_via_W():
    r = classify(["mu-"], ["e-", "nu_e_bar", "nu_mu"])
    assert r.verdict is Verdict.ALLOWED_AT_TREE_LEVEL
    assert any(m in ("W+", "W-") for m in r.mediators)


def test_mu_to_e_gamma_needs_mixing_or_loops():
    r = classify(["mu-"], ["e-", "gamma"])
    assert r.verdict is Verdict.ALLOWED_ONLY_WITH_MIXING_OR_LOOPS


def test_proton_decay_needs_a_higher_dimensional_operator():
    r = classify(["proton"], ["e+", "pi0"])
    assert r.verdict is Verdict.ALLOWED_BY_HIGHER_DIM_OPERATOR


def test_charge_violation_is_forbidden_outright():
    r = classify(["mu-"], ["nu_mu"])   # -1 -> 0
    assert r.verdict is Verdict.FORBIDDEN_BY_EXACT_SYMMETRY


def test_uphill_decay_is_kinematically_forbidden():
    # e- -> mu- nu_e nu_mu_bar conserves every charge and flavor, but the muon
    # is heavier than the electron.
    r = classify(["e-"], ["mu-", "nu_e", "nu_mu_bar"])
    assert r.verdict is Verdict.KINEMATICALLY_FORBIDDEN


def test_unequal_mass_one_to_one_transition_is_kinematically_forbidden():
    r = classify(["mu-"], ["e-"])
    assert r.verdict is Verdict.KINEMATICALLY_FORBIDDEN
    assert "1->1" in r.detail


def test_equal_threshold_decay_has_no_phase_space():
    r = classify(["gamma"], ["gamma", "gamma"])
    assert r.verdict is Verdict.KINEMATICALLY_FORBIDDEN
    assert "no decay phase space" in r.detail


def test_unknown_particle_is_undermodelled():
    r = classify(["mu-"], ["e-", "snark"])
    assert r.verdict is Verdict.UNDERMODELLED


def test_light_by_light_conserves_everything_but_has_no_tree_diagram():
    # gamma gamma -> gamma gamma is a genuine loop-only SM process: it conserves
    # every charge yet no tree of fundamental vertices connects four photons.
    r = classify(["gamma", "gamma"], ["gamma", "gamma"])
    assert r.verdict is Verdict.ALLOWED_BY_CONSERVATION_BUT_NO_SM_VERTEX
    assert "configured vertex subset" in r.detail


def test_gluon_self_scattering_is_tree_level_via_the_three_gluon_vertex():
    # gg -> gg is genuinely tree-level (non-abelian 3-gluon vertex); with that
    # vertex modeled, the engine must find a tree rather than report no-vertex.
    r = classify(["gluon", "gluon"], ["gluon", "gluon"])
    assert r.verdict is Verdict.ALLOWED_AT_TREE_LEVEL


def test_higgs_self_scattering_is_tree_level_via_the_three_higgs_vertex():
    # hh -> hh is tree-level via the Higgs self-coupling (h-exchange); modeled.
    r = classify(["h", "h"], ["h", "h"])
    assert r.verdict is Verdict.ALLOWED_AT_TREE_LEVEL


def test_beta_decay_is_flagged_as_composite_not_expanded():
    # n -> p e- nu_e_bar IS tree-level at the quark level, but the search does
    # not decompose hadrons, so it is honestly flagged rather than mislabeled.
    r = classify(["neutron"], ["proton", "e-", "nu_e_bar"])
    assert r.verdict is Verdict.ALLOWED_BY_CONSERVATION_BUT_NO_SM_VERTEX
    assert "composite" in r.detail


def test_proton_decay_detail_notes_b_minus_l_is_conserved():
    r = classify(["proton"], ["e+", "pi0"])
    assert r.verdict is Verdict.ALLOWED_BY_HIGHER_DIM_OPERATOR
    assert "B-L conserved" in r.detail
    assert "DeltaB" in r.detail


def test_neutron_antineutron_oscillation_also_violates_b_minus_l():
    # delta B = 2: distinguished from proton decay by the B-L flag.
    r = classify(["neutron"], ["neutron_bar"])
    assert r.verdict is Verdict.ALLOWED_BY_HIGHER_DIM_OPERATOR
    assert "B-L violated" in r.detail


def test_charge_violation_outranks_baryon_violation():
    # proton -> pi0 violates BOTH charge and baryon number; the exact gauge
    # symmetry is the reported (more fundamental) reason.
    r = classify(["proton"], ["pi0"])
    assert r.verdict is Verdict.FORBIDDEN_BY_EXACT_SYMMETRY


def test_kinematics_outranks_flavor_violation():
    # e- -> mu- gamma is flavor-violating AND kinematically impossible; the
    # kinematic gate fires first.
    r = classify(["e-"], ["mu-", "gamma"])
    assert r.verdict is Verdict.KINEMATICALLY_FORBIDDEN


def test_verdict_is_invariant_under_charge_conjugation():
    forward = classify(["mu-"], ["e-", "nu_e_bar", "nu_mu"]).verdict
    conjugate = classify(["mu+"], ["e+", "nu_e", "nu_mu_bar"]).verdict
    assert forward is conjugate is Verdict.ALLOWED_AT_TREE_LEVEL


def test_empty_state_is_undermodelled():
    assert classify([], ["gamma"]).verdict is Verdict.UNDERMODELLED
    assert classify(["gamma"], []).verdict is Verdict.UNDERMODELLED


def test_tree_search_budget_exceeded_is_undermodelled_not_no_vertex():
    r = classify(["e-", "e+"], ["e-", "e+", "e-", "e+", "e-", "e+"])
    assert r.verdict is Verdict.UNDERMODELLED
    assert "budget exceeded" in r.detail
