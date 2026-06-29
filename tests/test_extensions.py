"""Candidate-extension search, chunk 1: what KIND of field is a hypothesized
mediator, and what spin must a universal massless force carrier have?

The classification is not a lookup -- it follows from the rank of the current
the mediator couples to (a spin-s field couples to a rank-s conserved current),
and from which conserved currents actually exist (rank 0,1,2; nothing higher).
"""
from fractions import Fraction as F

import pytest

from nsm.extensions import (
    Candidate, FieldType, classify_field_type,
    required_spin_for_universal_massless_force, consistency_report,
    register_candidate, exposing_verdict, candidate_vertices, coupled_fermions,
    normalize_charge_map, validate_candidate,
)
from nsm.verdict import Verdict

_BL = [("Q_L", F(1, 3)), ("u_R", F(1, 3)), ("d_R", F(1, 3)),
       ("L_L", F(-1)), ("e_R", F(-1))]
_BL_REPAIRED = _BL + [("nu_R", F(-1))]


def graviton():
    return Candidate("G", spin=F(2), mass_mev=0.0, coupling="stress_energy",
                     coupling_scale="planck")


def test_graviton_candidate_is_classified_graviton_like():
    ft, _ = classify_field_type(graviton())
    assert ft is FieldType.GRAVITON_LIKE


def test_massless_vector_coupling_to_a_charge_is_a_gauge_boson():
    z_prime = Candidate("Zp", spin=F(1), mass_mev=0.0, coupling="gauge_charge")
    assert classify_field_type(z_prime)[0] is FieldType.GAUGE_BOSON


def test_massless_scalar_is_a_scalar():
    s = Candidate("phi", spin=F(0), mass_mev=0.0, coupling="trace")
    assert classify_field_type(s)[0] is FieldType.SCALAR


def test_scalar_trace_template_is_classified_but_not_process_exposed():
    s = Candidate("phi", spin=F(0), mass_mev=0.0, coupling="trace")
    assert not consistency_report(s).is_consistent
    assert "not implemented" in consistency_report(s).failures[0][1]


def test_massless_spin_three_is_outside_the_modeled_class():
    g3 = Candidate("X3", spin=F(3), mass_mev=0.0, coupling="stress_energy")
    assert classify_field_type(g3)[0] is FieldType.OUTSIDE_MODELED_CLASS


def test_massive_vector_is_a_short_range_mediator_not_a_gauge_force():
    massive = Candidate("Zp", spin=F(1), mass_mev=1000.0, coupling="gauge_charge")
    assert classify_field_type(massive)[0] is FieldType.MASSIVE_MEDIATOR


def test_a_universal_massless_force_must_be_spin_two():
    # The deep result: coupling universally to energy-momentum (a rank-2
    # conserved tensor) forces a rank-2 -> spin-2 mediator. Uniquely.
    assert required_spin_for_universal_massless_force() == F(2)


def test_graviton_candidate_is_consistent():
    assert consistency_report(graviton()).is_consistent


def test_candidate_consistency_checks_domain_before_exposure():
    charged = Candidate("Wp", spin=F(1), mass_mev=1000.0, charge=F(1),
                        self_conjugate=False, coupling="gauge_charge",
                        charge_map=tuple(_BL + [("nu_R", F(-1))]))
    colored = Candidate("X", spin=F(1), mass_mev=1000.0, color="triplet",
                        coupling="gauge_charge",
                        charge_map=tuple(_BL + [("nu_R", F(-1))]))
    octet_grav = Candidate("G8", spin=F(2), mass_mev=0.0, color="octet",
                           coupling="stress_energy")
    for candidate in (charged, colored, octet_grav):
        assert not consistency_report(candidate).is_consistent
        with pytest.raises(ValueError, match="not valid"):
            validate_candidate(candidate)


def test_consistency_report_handles_invalid_spin_and_mass_without_crashing():
    bad = Candidate("Bad", spin="not-a-spin", mass_mev=float("nan"),
                    coupling="stress_energy")
    report = consistency_report(bad)
    assert not report.is_consistent
    assert any(name == "finite nonnegative mass" for name, _ in report.failures)
    assert any(name == "bosonic nonnegative spin" for name, _ in report.failures)


def test_a_spin1_cannot_couple_universally_to_energy():
    # rank mismatch: a vector (spin 1) cannot source the rank-2 stress tensor.
    bad = Candidate("V", spin=F(1), mass_mev=0.0, coupling="stress_energy")
    assert not consistency_report(bad).is_consistent


def test_a_spin2_coupled_to_a_charge_is_inconsistent():
    # soft-graviton theorem: a massless spin-2 must couple universally, not to a
    # selective charge.
    bad = Candidate("g2", spin=F(2), mass_mev=0.0, coupling="gauge_charge")
    assert not consistency_report(bad).is_consistent


def test_gauging_b_minus_l_is_anomalous_without_right_handed_neutrinos():
    zp = Candidate("Zp", spin=F(1), mass_mev=0.0, coupling="gauge_charge",
                   charge_map=tuple(_BL))
    assert not consistency_report(zp).is_consistent   # gravitational & cubic anomalies


def test_gauging_b_minus_l_becomes_consistent_with_right_handed_neutrinos():
    zp = Candidate("Zp", spin=F(1), mass_mev=0.0, coupling="gauge_charge",
                   charge_map=tuple(_BL + [("nu_R", F(-1))]))
    assert consistency_report(zp).is_consistent       # anomaly-free with one nu_R per generation


# --- chunk 3: registration + process exposure -------------------------------

def test_emit_a_graviton_with_a_photon_is_tree_level_but_suppressed():
    register_candidate(graviton())  # idempotent
    r = exposing_verdict(["e-", "e+"], ["gamma", "G"], graviton())
    assert r.verdict is Verdict.ALLOWED_AT_TREE_LEVEL
    assert any("planck" in n.lower() for n in r.notes)


def test_two_photons_to_a_single_graviton_is_kinematically_special():
    register_candidate(graviton())
    r = exposing_verdict(["gamma", "gamma"], ["G"], graviton())
    assert r.verdict is Verdict.KINEMATICALLY_FORBIDDEN
    assert any("phase space" in n for n in r.notes)


def test_flavor_changing_graviton_emission_is_not_a_gravity_signal():
    register_candidate(graviton())
    r = exposing_verdict(["mu-"], ["e-", "G"], graviton())
    assert r.verdict is Verdict.ALLOWED_ONLY_WITH_MIXING_OR_LOOPS
    assert any("flavor" in n.lower() for n in r.notes)


def test_unrecognized_coupling_is_not_silently_consistent():
    # A typo'd coupling must NOT slip through as vacuously consistent.
    typo = Candidate("Q", spin=F(1), mass_mev=0.0, coupling="gauge")  # not 'gauge_charge'
    assert not consistency_report(typo).is_consistent


def test_massive_gauge_boson_with_anomaly_free_charge_is_consistent():
    # Massive => short-range => the soft-theorem spin/rank gate does not apply,
    # but a gauged charge still needs anomaly cancellation (it passes here).
    zp = Candidate("Zp", spin=F(1), mass_mev=1000.0, coupling="gauge_charge",
                   charge_map=tuple(_BL + [("nu_R", F(-1))]))
    assert consistency_report(zp).is_consistent


def test_massive_gauge_boson_still_requires_anomaly_cancellation():
    zp = Candidate("Zp", spin=F(1), mass_mev=1000.0, coupling="gauge_charge",
                   charge_map=tuple(_BL))  # no nu_R -> still anomalous
    assert not consistency_report(zp).is_consistent


def test_an_arbitrary_anomalous_charge_is_rejected():
    weird = Candidate("Zp", spin=F(1), mass_mev=0.0, coupling="gauge_charge",
                      charge_map=(("L_L", F(1)),))   # SU(2)^2 anomaly != 0
    assert not consistency_report(weird).is_consistent


def test_empty_gauge_charge_map_is_trivial_not_consistent():
    # A gauge_charge candidate with no nonzero U(1)' charge is the trivial zero
    # generator (no gauged symmetry), not a meaningful consistent extension.
    trivial = Candidate("Zp", spin=F(1), mass_mev=0.0, coupling="gauge_charge",
                        charge_map=())
    assert not consistency_report(trivial).is_consistent


def test_normalize_charge_map_fills_omitted_fields_with_zero():
    # Omitted SM fields mean X=0, not "removed from the spectrum."
    norm = dict(normalize_charge_map((("e_R", F(1)),)))
    assert norm == {"Q_L": F(0), "u_R": F(0), "d_R": F(0), "L_L": F(0), "e_R": F(1)}


def test_normalize_charge_map_rejects_unknown_and_duplicate_names():
    with pytest.raises(ValueError, match="unknown"):
        normalize_charge_map((("X_L", F(1)),))
    with pytest.raises(ValueError, match="duplicate"):
        normalize_charge_map((("e_R", F(1)), ("e_R", F(2))))


def test_consistency_report_records_invalid_charge_map_instead_of_raising():
    bad = Candidate("Zp", spin=F(1), mass_mev=0.0, coupling="gauge_charge",
                    charge_map=(("not_a_field", F(1)),))
    report = consistency_report(bad)
    assert not report.is_consistent
    assert any(name == "valid charge map" and "unknown" in detail
               for name, detail in report.failures)


def test_exposing_verdict_does_not_leak_candidate_registration():
    from nsm.particles import PARTICLES

    assert "ScopedZp" not in PARTICLES
    scoped = Candidate("ScopedZp", spin=F(1), mass_mev=2000.0,
                       coupling="gauge_charge", charge_map=tuple(_BL_REPAIRED))
    exposing_verdict(["e-", "e+"], ["ScopedZp"], scoped)
    assert "ScopedZp" not in PARTICLES


def test_kinematic_override_does_not_mask_lepton_number_violation():
    # nu_e nu_e -> gamma is all-massless 2->1 BUT violates lepton number; the
    # phase-space refinement must not overwrite the more fundamental verdict.
    r = exposing_verdict(["nu_e", "nu_e"], ["gamma"], graviton())
    assert r.verdict is Verdict.ALLOWED_BY_HIGHER_DIM_OPERATOR


def test_graviton_exposure_is_charge_conjugation_symmetric():
    a = exposing_verdict(["mu-"], ["e-", "G"], graviton()).verdict
    b = exposing_verdict(["mu+"], ["e+", "G"], graviton()).verdict
    assert a is b is Verdict.ALLOWED_ONLY_WITH_MIXING_OR_LOOPS


def test_a_charged_universal_candidate_is_rejected_cleanly():
    # A stress-energy (universal) coupling requires a neutral self-conjugate
    # mediator; a charged one must raise a clear domain error, not crash deep
    # inside vertex construction.
    from nsm.extensions import candidate_vertices
    charged = Candidate("Cp", spin=F(2), mass_mev=0.0, charge=F(1),
                        self_conjugate=False, coupling="stress_energy")
    with pytest.raises(ValueError, match="not valid"):
        candidate_vertices(charged)


def test_anomalous_candidate_cannot_generate_exposure_vertices():
    anomalous = Candidate("Zp", spin=F(1), mass_mev=2000.0, coupling="gauge_charge",
                          charge_map=tuple(_BL))
    assert not consistency_report(anomalous).is_consistent
    with pytest.raises(ValueError, match="not valid"):
        exposing_verdict(["e-", "e+"], ["mu-", "mu+"], anomalous)


# --- gauge-boson (Z') exposure: charge-weighted, flavor-diagonal coupling ----

def zprime(mass_mev=2000.0, charge_map=None):
    return Candidate("Zp", spin=F(1), mass_mev=mass_mev, coupling="gauge_charge",
                     coupling_scale="weak",
                     charge_map=tuple(charge_map if charge_map is not None else _BL_REPAIRED))


def test_b_minus_l_zprime_couples_to_quarks_leptons_and_neutrinos():
    cf = coupled_fermions(zprime())
    assert {"u", "d", "t"} <= cf          # quarks (B-L = 1/3)
    assert {"e-", "mu-", "tau-"} <= cf     # charged leptons (B-L = -1)
    assert "nu_e" in cf                    # neutrinos couple too


def test_a_leptonic_zprime_does_not_couple_to_quarks():
    # Charge-WEIGHTED, unlike the graviton's universality: a Z' that only
    # charges leptons must not produce quark vertices.
    lep = zprime(charge_map=[("L_L", F(1)), ("e_R", F(1))])
    cf = coupled_fermions(lep)
    assert {"e-", "nu_e"} <= cf
    assert not ({"u", "d", "c", "s", "t", "b"} & cf)


def test_zprime_is_produced_resonantly_in_electron_positron():
    zp = zprime(mass_mev=2000.0)  # massive -> resonant 2->1 is allowed
    r = exposing_verdict(["e-", "e+"], ["Zp"], zp)
    assert r.verdict is Verdict.ALLOWED_AT_TREE_LEVEL
    assert any("Zp" in n for n in r.notes)


def test_a_massless_zprime_cannot_be_made_in_a_2_to_1():
    zp = zprime(mass_mev=0.0)  # all-massless 2->1 has vanishing phase space
    r = exposing_verdict(["e-", "e+"], ["Zp"], zp)
    assert r.verdict is Verdict.KINEMATICALLY_FORBIDDEN


def test_zprime_contributes_to_dilepton_production_as_a_mediator():
    # e+ e- -> mu+ mu- proceeds via SM gamma/Z, but a B-L Z' coupling to both e
    # and mu also mediates it (s-channel) -- that is how it would be exposed.
    r = exposing_verdict(["e-", "e+"], ["mu-", "mu+"], zprime())
    assert r.verdict is Verdict.ALLOWED_AT_TREE_LEVEL
    assert any("Zp" in n and "mediates" in n for n in r.notes)


def test_flavor_changing_zprime_emission_is_not_a_zprime_signal():
    # A light, flavor-diagonal Z' cannot drive mu -> e Zp; that needs the weak
    # mixing sector, so it is not a generic Z' signal.
    zp = zprime(mass_mev=0.0)
    r = exposing_verdict(["mu-"], ["e-", "Zp"], zp)
    assert r.verdict is Verdict.ALLOWED_ONLY_WITH_MIXING_OR_LOOPS
    assert any("flavor" in n.lower() for n in r.notes)


def test_a_charged_gauge_candidate_is_rejected_cleanly():
    charged = Candidate("Wp", spin=F(1), mass_mev=1000.0, charge=F(1),
                        self_conjugate=False, coupling="gauge_charge",
                        charge_map=tuple(_BL))
    with pytest.raises(ValueError, match="not valid"):
        candidate_vertices(charged)


def _has_mediation_note(result):
    return any("mediates" in n for n in result.notes)


def test_coupled_fermions_respects_vector_like_rep_structure():
    # Equal L/R charges produce an ordinary vector-like coupling to charged
    # leptons plus the left-handed neutrinos in L_L.
    leptonic = Candidate("X", spin=F(1), mass_mev=0.0, coupling="gauge_charge",
                         charge_map=(("L_L", F(1)), ("e_R", F(1))))
    assert {"e-", "mu-", "tau-", "nu_e", "nu_mu", "nu_tau"} == coupled_fermions(leptonic)

    # Unequal L/R charges, including zero vs nonzero, are chirality-specific and
    # cannot be collapsed to one physical-particle vertex.
    up = Candidate("X", spin=F(1), mass_mev=0.0, coupling="gauge_charge",
                   charge_map=(("u_R", F(1)),))
    with pytest.raises(ValueError, match="chirally"):
        coupled_fermions(up)
    er = Candidate("X", spin=F(1), mass_mev=0.0, coupling="gauge_charge",
                   charge_map=(("e_R", F(1)),))
    with pytest.raises(ValueError, match="chirally"):
        coupled_fermions(er)


def test_zprime_exposure_reports_its_own_suppression_scale():
    r = exposing_verdict(["e-", "e+"], ["Zp"], zprime(mass_mev=2000.0))
    assert any("weak-suppressed" in n for n in r.notes)


def test_gauge_candidate_default_suppression_is_weak_not_planck():
    zp = Candidate("Zp", spin=F(1), mass_mev=2000.0, coupling="gauge_charge",
                   charge_map=tuple(_BL_REPAIRED))
    r = exposing_verdict(["e-", "e+"], ["Zp"], zp)
    assert any("weak-suppressed" in n for n in r.notes)
    assert not any("planck-suppressed" in n for n in r.notes)


def test_massless_zprime_2to1_is_forbidden_via_phase_space():
    r = exposing_verdict(["e-", "e+"], ["Zp"], zprime(mass_mev=0.0))
    assert r.verdict is Verdict.KINEMATICALLY_FORBIDDEN
    assert any("phase space" in n for n in r.notes)


def test_mediation_note_absent_for_a_flavor_violating_final():
    # A flavor-diagonal Z' cannot mediate e+e- -> mu- tau+.
    assert not _has_mediation_note(exposing_verdict(["e-", "e+"], ["mu-", "tau+"], zprime()))


def test_mediation_note_absent_when_candidate_couples_to_one_side_only():
    # Z' couples to the electrons but not to photons -> cannot mediate.
    assert not _has_mediation_note(exposing_verdict(["e-", "e+"], ["gamma", "gamma"], zprime()))


def test_leptonic_zprime_is_rejected_before_pure_quark_exposure():
    leptonic = zprime(charge_map=[("L_L", F(1)), ("e_R", F(1))])
    with pytest.raises(ValueError, match="not valid"):
        exposing_verdict(["u", "u_bar"], ["d", "d_bar"], leptonic)


def test_zprime_mediates_dilepton_scattering_via_t_channel():
    assert _has_mediation_note(exposing_verdict(["e-", "mu-"], ["e-", "mu-"], zprime()))


def test_graviton_mediates_a_two_photon_to_fermion_process():
    # Uses boson legs -> only a vertex-based (not fermion-membership) check catches it.
    assert _has_mediation_note(exposing_verdict(["gamma", "gamma"], ["e-", "e+"], graviton()))
