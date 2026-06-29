"""Particles: components + antiparticles, carrying the quantum numbers the
dynamics layer needs (charge, baryon number, lepton flavor, color, spin).

Electric charges here must agree with the kernel's derived values -- the two
layers share one notion of charge.
"""
from fractions import Fraction as F

from nsm.particles import (
    Particle,
    antiparticle,
    baryon_number,
    lepton_number,
    particle,
    register_particle,
    _lep,
)
from nsm.sm import standard_model_skeleton
from nsm.derive import derive_hypercharges


def test_muon_quantum_numbers():
    mu = particle("mu-")
    assert mu.charge == F(-1)
    assert mu.lepton["mu"] == F(1)
    assert mu.lepton["e"] == F(0)
    assert baryon_number(mu) == F(0)


def test_antiparticle_flips_additive_charges():
    anti = antiparticle("mu-")
    assert anti.name == "mu+"
    assert anti.charge == F(1)
    assert anti.lepton["mu"] == F(-1)


def test_photon_is_self_conjugate():
    assert antiparticle("gamma").name == "gamma"


def test_neutrino_lepton_number():
    assert lepton_number(particle("nu_mu")) == F(1)
    assert antiparticle("nu_e").name == "nu_e_bar"
    assert lepton_number(particle("nu_e_bar")) == F(-1)


def test_proton_is_uud_composite():
    p = particle("proton")
    assert p.charge == F(1)
    assert baryon_number(p) == F(1)
    assert lepton_number(p) == F(0)


def test_pion_zero_is_neutral_meson():
    pi0 = particle("pi0")
    assert pi0.charge == F(0)
    assert baryon_number(pi0) == F(0)


def test_charges_agree_with_the_kernel():
    # Cross-layer consistency: the charge table is not independent of the
    # kernel's anomaly-derived charges.
    derived = derive_hypercharges(standard_model_skeleton()).charges
    assert particle("e-").charge == derived["L_L:e"]
    assert particle("u").charge == derived["u_R"]
    assert particle("d").charge == derived["d_R"]


def test_register_particle_replaces_antiparticle_atomically():
    register_particle(Particle(
        name="X", charge=F(1), baryon=F(0), lepton=_lep(), color="singlet",
        spin=F(0), is_boson=True, anti="Xbar", mass_mev=10.0))
    assert particle("Xbar").charge == F(-1)
    assert particle("Xbar").mass_mev == 10.0

    register_particle(Particle(
        name="X", charge=F(2), baryon=F(0), lepton=_lep(), color="singlet",
        spin=F(0), is_boson=True, anti="Xbar", mass_mev=20.0))
    assert particle("Xbar").charge == F(-2)
    assert particle("Xbar").mass_mev == 20.0

    register_particle(Particle(
        name="X", charge=F(0), baryon=F(0), lepton=_lep(), color="singlet",
        spin=F(0), is_boson=True, anti="X", mass_mev=30.0))
    assert particle("X").is_self_conjugate
    try:
        particle("Xbar")
        assert False, "stale antiparticle should have been removed"
    except KeyError:
        pass
