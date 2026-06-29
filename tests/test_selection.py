"""Selection rules: which symmetries a process respects or violates.

The point is to separate EXACT gauge symmetries (charge, color) from the
ACCIDENTAL global symmetries of the renormalizable SM (baryon #, lepton #) and
the APPROXIMATE ones (lepton flavor) -- because they imply different verdicts.
"""
from nsm.selection import analyze


def test_muon_decay_conserves_everything():
    r = analyze(["mu-"], ["e-", "nu_e_bar", "nu_mu"])
    assert r.charge_conserved
    assert r.color_conserved
    assert r.baryon_conserved
    assert r.lepton_conserved
    assert r.flavor_conserved
    assert r.bminusl_conserved


def test_mu_to_e_gamma_violates_only_lepton_flavor():
    r = analyze(["mu-"], ["e-", "gamma"])
    assert r.charge_conserved
    assert r.lepton_conserved        # total lepton number is fine
    assert not r.flavor_conserved    # but muon-ness -> electron-ness is not


def test_proton_decay_violates_baryon_and_lepton_but_not_b_minus_l():
    r = analyze(["proton"], ["e+", "pi0"])
    assert r.charge_conserved
    assert r.color_conserved
    assert not r.baryon_conserved
    assert not r.lepton_conserved
    assert r.bminusl_conserved       # B-L is the surviving symmetry
