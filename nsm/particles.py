"""Particle registry for the dynamics layer.

Fundamental fields (leptons, quarks, gauge/Higgs bosons) plus a few composite
hadrons, each carrying the additive quantum numbers process queries need:
electric charge, baryon number, per-flavor lepton number, color, spin.

Antiparticles are generated automatically by conjugating additive charges. The
electric charges are consistent with the kernel's anomaly-derived values (see
tests/test_particles.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction


FLAVORS = ("e", "mu", "tau")
_CONJ_COLOR = {"singlet": "singlet", "octet": "octet",
               "triplet": "antitriplet", "antitriplet": "triplet"}


@dataclass(frozen=True)
class Particle:
    name: str
    charge: Fraction
    baryon: Fraction
    lepton: dict           # {"e": .., "mu": .., "tau": ..}
    color: str
    spin: Fraction
    is_boson: bool
    anti: str              # name of the antiparticle
    mass_mev: float | None = None
    constituents: tuple = ()   # for composites, informational

    @property
    def is_self_conjugate(self) -> bool:
        return self.anti == self.name


def _lep(**kw) -> dict:
    return {fl: Fraction(kw.get(fl, 0)) for fl in FLAVORS}


_BASE = {}


def _add(name, anti, **kw):
    kw.setdefault("baryon", Fraction(0))
    kw.setdefault("lepton", _lep())
    kw.setdefault("color", "singlet")
    kw.setdefault("spin", Fraction(1, 2))
    kw.setdefault("is_boson", False)
    kw["charge"] = Fraction(kw["charge"])
    _BASE[name] = Particle(name=name, anti=anti, **kw)


# --- charged leptons & neutrinos -------------------------------------------
for fl, m in [("e", 0.511), ("mu", 105.658), ("tau", 1776.86)]:
    _add(f"{fl}-", f"{fl}+", charge=-1, lepton=_lep(**{fl: 1}), mass_mev=m)
    _add(f"nu_{fl}", f"nu_{fl}_bar",
         charge=0, lepton=_lep(**{fl: 1}), mass_mev=0.0)

# --- quarks (B = 1/3, color triplet) ---------------------------------------
for q, ch, m in [("u", Fraction(2, 3), 2.2), ("d", Fraction(-1, 3), 4.7),
                 ("c", Fraction(2, 3), 1275.0), ("s", Fraction(-1, 3), 95.0),
                 ("t", Fraction(2, 3), 173000.0), ("b", Fraction(-1, 3), 4180.0)]:
    _add(q, f"{q}_bar", charge=ch, baryon=Fraction(1, 3), color="triplet", mass_mev=m)

# --- gauge & Higgs bosons ---------------------------------------------------
_add("gamma", "gamma", charge=0, spin=Fraction(1), is_boson=True, mass_mev=0.0)
_add("Z", "Z", charge=0, spin=Fraction(1), is_boson=True, mass_mev=91187.6)
_add("W+", "W-", charge=1, spin=Fraction(1), is_boson=True, mass_mev=80379.0)
_add("gluon", "gluon", charge=0, spin=Fraction(1), is_boson=True, color="octet", mass_mev=0.0)
_add("h", "h", charge=0, spin=Fraction(0), is_boson=True, mass_mev=125250.0)


def _conjugate(p: Particle) -> Particle:
    return Particle(
        name=p.anti,
        charge=-p.charge,
        baryon=-p.baryon,
        lepton={fl: -p.lepton[fl] for fl in FLAVORS},
        color=_CONJ_COLOR[p.color],
        spin=p.spin,
        is_boson=p.is_boson,
        anti=p.name,
        mass_mev=p.mass_mev,
        constituents=tuple(antiparticle(c).name for c in p.constituents) if p.constituents else (),
    )


# Build the full registry: base particles + their (non-self-conjugate) antis.
PARTICLES: dict = dict(_BASE)
for _p in list(_BASE.values()):
    if not _p.is_self_conjugate and _p.anti not in PARTICLES:
        PARTICLES[_p.anti] = _conjugate(_p)


def particle(name: str) -> Particle:
    if name not in PARTICLES:
        raise KeyError(f"unknown particle: {name!r}")
    return PARTICLES[name]


def register_particle(p: Particle) -> None:
    """Add a new particle (and its antiparticle, if distinct) to the registry.
    Used by the candidate-extension layer to inject hypothesized fields."""
    old = PARTICLES.get(p.name)
    # Drop a now-orphaned antiparticle from a prior registration (covers both a
    # changed anti and a become-self-conjugate transition, since a self-conjugate
    # p has p.anti == p.name != old.anti).
    if old is not None and not old.is_self_conjugate and old.anti != p.anti:
        PARTICLES.pop(old.anti, None)
    PARTICLES[p.name] = p
    if not p.is_self_conjugate:
        PARTICLES[p.anti] = _conjugate(p)


def antiparticle(name: str) -> Particle:
    return particle(particle(name).anti)


def lepton_number(p: Particle) -> Fraction:
    return sum(p.lepton.values(), Fraction(0))


def baryon_number(p: Particle) -> Fraction:
    return p.baryon


def _composite(name, anti, constituents, spin, is_boson, mass_mev, self_conj=False):
    parts = [particle(c) for c in constituents]
    PARTICLES[name] = Particle(
        name=name,
        charge=sum((q.charge for q in parts), Fraction(0)),
        baryon=sum((q.baryon for q in parts), Fraction(0)),
        lepton={fl: sum((q.lepton[fl] for q in parts), Fraction(0)) for fl in FLAVORS},
        color="singlet",  # physical hadrons are color singlets
        spin=Fraction(spin),
        is_boson=is_boson,
        anti=name if self_conj else anti,
        mass_mev=mass_mev,
        constituents=tuple(constituents),
    )


# --- composite hadrons ------------------------------------------------------
_composite("proton", "proton_bar", ("u", "u", "d"), Fraction(1, 2), False, 938.272)
_composite("neutron", "neutron_bar", ("u", "d", "d"), Fraction(1, 2), False, 939.565)
_composite("pi+", "pi-", ("u", "d_bar"), Fraction(0), True, 139.570)
_composite("pi0", "pi0", ("u", "u_bar"), Fraction(0), True, 134.977, self_conj=True)

for _name in ["proton", "neutron", "pi+"]:
    _src = PARTICLES[_name]
    if _src.anti not in PARTICLES:
        PARTICLES[_src.anti] = _conjugate(_src)
