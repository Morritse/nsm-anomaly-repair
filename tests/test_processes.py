"""Tree-level reachability: can initial -> final be drawn as a tree of SM
vertices? This is what upgrades "conserves all charges" to "actually happens at
tree level via <mediator>", and what correctly says mu -> e gamma has no
tree-level diagram at all.
"""
from nsm.processes import TreeSearchStatus, tree_level_reachable, tree_level_search
from nsm.vertices import is_vertex


def test_qed_vertex_is_recognized():
    assert is_vertex(["gamma", "e-", "e+"])
    assert not is_vertex(["gamma", "mu-", "e+"])   # flavor-changing: not SM


def test_muon_decay_is_tree_level_via_W():
    mediators = tree_level_reachable(["mu-"], ["e-", "nu_e_bar", "nu_mu"])
    assert mediators is not None
    assert any(m in ("W+", "W-") for m in mediators)


def test_mu_to_e_gamma_has_no_tree_diagram():
    assert tree_level_reachable(["mu-"], ["e-", "gamma"]) is None


def test_ee_to_mumu_is_tree_level_via_neutral_boson():
    mediators = tree_level_reachable(["e-", "e+"], ["mu-", "mu+"])
    assert mediators == ["gamma"]


def test_lepton_flavor_change_with_two_legs_is_unreachable():
    # e- -> mu- conserves charge but there is no 2-point SM vertex.
    assert tree_level_reachable(["e-"], ["mu-"]) is None


def test_no_flavor_changing_neutral_vertex_exists():
    # Pin the deliberate absence of tree-level FCNC.
    assert not is_vertex(["gamma", "mu-", "e+"])
    assert not is_vertex(["Z", "mu-", "e+"])


def test_multibody_flavor_violation_has_no_tree_diagram():
    # Conserves charge but changes flavor; the recursion must not fabricate a
    # tree (false-positive guard exercising the multi-leg fusion).
    assert tree_level_reachable(["mu-"], ["e-", "nu_e", "nu_mu_bar"]) is None


def test_neutral_current_scattering_is_tree_level_via_Z():
    mediators = tree_level_reachable(["nu_mu", "e-"], ["nu_mu", "e-"])
    assert mediators is not None
    assert "Z" in mediators


def test_quark_level_charged_current_is_tree_level_via_W():
    # Beta decay AT THE QUARK LEVEL is reachable, even though the hadron-level
    # process is not (composites are not expanded).
    mediators = tree_level_reachable(["d"], ["u", "e-", "nu_e_bar"])
    assert mediators is not None
    assert any(m in ("W+", "W-") for m in mediators)


def test_tree_search_reports_budget_exceeded_separately_from_no_tree():
    process = (["e-", "e+"], ["e-", "e+", "e-", "e+", "e-", "e+"])
    limited = tree_level_search(*process)
    assert limited.status is TreeSearchStatus.BUDGET_EXCEEDED
    widened = tree_level_search(*process, max_external=10)
    assert widened.status is TreeSearchStatus.FOUND
