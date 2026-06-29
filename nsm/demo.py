"""Run the derivations and print them. Thin presentation layer over the
tested engine; run with ``python -m nsm``."""
from fractions import Fraction

from nsm.reps import SU3, SU2, Chirality
from nsm.sm import standard_model_skeleton, anomaly_only_skeleton, Yukawa
from nsm.derive import derive_hypercharges, enumerate_solutions
from nsm.verdict import classify
from nsm.extensions import (
    Candidate, classify_field_type, consistency_report, exposing_verdict,
    required_spin_for_universal_massless_force, candidate_vertices,
)
from nsm.amplitudes import soft_theorem_report, leading_soft_factor
from nsm.factorization import factorization_mediators
from nsm.unitarity import cut_states, threshold_order
from nsm.repair import minimal_repair


def _fmt(fr: Fraction) -> str:
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


def _rule(title: str):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main():
    _rule("1. DERIVE: SM one generation, all hypercharges unknown")
    print("Input facts: gauge reps + Yukawa structure + normalization (Y_H=1).")
    print("Nothing numeric is supplied. Solving consistency constraints...\n")
    r = derive_hypercharges(standard_model_skeleton())
    print(f"  {'field':6} {'hypercharge Y':>14}")
    for name, y in r.hypercharges.items():
        print(f"  {name:6} {_fmt(y):>14}")
    print(f"\n  electric charges Q = T3 + Y/2 (derived, never stored):")
    for label, q in r.charges.items():
        print(f"  {label:10} Q = {_fmt(q)}")
    print(f"\n  unique solution: {r.is_unique}")

    _rule("2. CONTRAST: anomaly cancellation ALONE (no Yukawa)")
    print("Drop the Yukawa couplings; normalize by Y_Q = 1/3.")
    enum = enumerate_solutions(anomaly_only_skeleton())
    exact = "exactly " if enum.complete else "at least "
    print(f"Anomaly cancellation permits {exact}{len(enum.solutions)} assignments:\n")
    for i, s in enumerate(enum.solutions, 1):
        print(f"  solution {i}: u_R={_fmt(s['u_R']):>5}  d_R={_fmt(s['d_R']):>5}  "
              f"(L_L={_fmt(s['L_L'])}, e_R={_fmt(s['e_R'])} pinned)")
    print("\n=> Anomalies fix everything EXCEPT a u<->d swap. The Yukawa")
    print("   coupling Y_d = Y_Q - Y_H is what picks d_R = -2/3.")

    _rule("3. NON-STANDARD: add a right-handed neutrino")
    m = standard_model_skeleton()
    m.add_fermion("nu_R", SU3.SINGLET, SU2.SINGLET, Chirality.RIGHT)
    r2 = derive_hypercharges(m)
    print(f"SM + free nu_R: unique = {r2.is_unique}")
    print("  => a flat direction opens (gauged B-L); hypercharges no longer fixed.")
    m2 = standard_model_skeleton()
    m2.add_fermion("nu_R", SU3.SINGLET, SU2.SINGLET, Chirality.RIGHT)
    m2.yukawas.append(Yukawa("majorana", (("nu_R", Fraction(2)),)))
    r3 = derive_hypercharges(m2)
    print(f"SM + nu_R + Majorana mass (breaks B-L): unique = {r3.is_unique}, "
          f"Y_nuR = {_fmt(r3.hypercharges['nu_R'])}, Q_L = {_fmt(r3.hypercharges['Q_L'])}")
    print("  => only by demanding a Majorana mass is the neutrino forced neutral.")

    _rule("4. REJECTION: an unpaired colored fermion")
    m3 = standard_model_skeleton()
    m3.add_fermion("X_R", SU3.TRIPLET, SU2.SINGLET, Chirality.RIGHT)
    r4 = derive_hypercharges(m3)
    print(f"SM + lone colored Weyl fermion: consistent = {r4.satisfiable}")
    print("  => SU(3)^3 anomaly cannot cancel. The engine rejects the spectrum.")

    _rule("5. DYNAMICS: which processes are allowed, forbidden, or suppressed")
    print("Layered verdict from selection rules + tree-level vertex search.\n")
    queries = [
        (["mu-"], ["e-", "nu_e_bar", "nu_mu"]),
        (["mu-"], ["e-", "gamma"]),
        (["proton"], ["e+", "pi0"]),
        (["e-", "e+"], ["mu-", "mu+"]),
        (["mu-"], ["nu_mu"]),
    ]
    for initial, final in queries:
        res = classify(initial, final)
        proc = f"{' '.join(initial)} -> {' '.join(final)}"
        print(f"  {proc:34} {res.verdict.name}")
        print(f"  {'':34} {res.detail}")

    _rule("6. CANDIDATE EXTENSION: hypothesize a new mediator (a graviton)")
    G = Candidate("G", spin=Fraction(2), mass_mev=0.0, coupling="stress_energy",
                  coupling_scale="planck")
    ft, why = classify_field_type(G)
    print(f"Candidate G (spin 2, massless, universal): {ft.name}")
    print(f"  reason: {why}")
    print(f"  a universal massless force must be spin "
          f"{required_spin_for_universal_massless_force()} (couples to rank-2 T_mu_nu)")
    print(f"  consistent: {consistency_report(G).is_consistent}\n")

    BL = [("Q_L", Fraction(1, 3)), ("u_R", Fraction(1, 3)), ("d_R", Fraction(1, 3)),
          ("L_L", Fraction(-1)), ("e_R", Fraction(-1))]
    zp = Candidate("Zp", spin=Fraction(1), mass_mev=0.0, coupling="gauge_charge",
                   charge_map=tuple(BL))
    zp_nu = Candidate("Zp", spin=Fraction(1), mass_mev=0.0, coupling="gauge_charge",
                      charge_map=tuple(BL + [("nu_R", Fraction(-1))]))
    print("Gauge boson coupling to B-L (consistency = anomaly cancellation, via the kernel):")
    print(f"  without nu_R: consistent = {consistency_report(zp).is_consistent} "
          f"(fails {[d for _, d in consistency_report(zp).failures]})")
    print(f"  with    nu_R: consistent = {consistency_report(zp_nu).is_consistent}")

    print("\nWhich processes expose the graviton?")
    for initial, final in [(["e-", "e+"], ["gamma", "G"]),
                           (["gamma", "gamma"], ["G"]),
                           (["mu-"], ["e-", "G"])]:
        r = exposing_verdict(initial, final, G)
        proc = f"{' '.join(initial)} -> {' '.join(final)}"
        print(f"  {proc:22} {r.verdict.name}")
        for note in r.notes:
            print(f"  {'':22}   - {note}")

    print("\nWhich processes expose a B-L Z' (charge-weighted gauge boson)?")
    zp_heavy = Candidate("Zp", spin=Fraction(1), mass_mev=2000.0, coupling="gauge_charge",
                         coupling_scale="weak", charge_map=tuple(BL + [("nu_R", Fraction(-1))]))
    zp_light = Candidate("Zp", spin=Fraction(1), mass_mev=0.0, coupling="gauge_charge",
                         coupling_scale="weak", charge_map=tuple(BL + [("nu_R", Fraction(-1))]))
    for initial, final, cand in [(["e-", "e+"], ["Zp"], zp_heavy),
                                 (["e-", "e+"], ["mu-", "mu+"], zp_heavy),
                                 (["mu-"], ["e-", "Zp"], zp_light)]:
        r = exposing_verdict(initial, final, cand)
        proc = f"{' '.join(initial)} -> {' '.join(final)}"
        print(f"  {proc:22} {r.verdict.name}")
        for note in r.notes:
            print(f"  {'':22}   - {note}")

    _rule("7. SOFT THEOREM: symbolic leading emission factors")
    print("No cross sections here: just the universal/charge-weighted leading soft factors.\n")
    hard_initial, hard_final = ["e-", "e+"], ["mu-", "mu+"]
    grav_soft = leading_soft_factor(G, hard_initial, hard_final)
    zp_soft = leading_soft_factor(zp_light, hard_initial, hard_final)
    print("Soft graviton attached to e- e+ -> mu- mu+:")
    print(f"  current rank: {grav_soft.current_rank}; terms: {len(grav_soft.terms)}")
    print(f"  Ward check: {grav_soft.ward_identity.passed} "
          f"({grav_soft.ward_identity.reason})")
    print("Soft B-L Z' attached to e- e+ -> mu- mu+:")
    print(f"  current rank: {zp_soft.current_rank}; coefficients: "
          f"{[_fmt(t.coefficient) for t in zp_soft.terms]}")
    print(f"  Ward check: {zp_soft.ward_identity.passed} "
          f"({zp_soft.ward_identity.reason})")
    bad = soft_theorem_report(zp_light, ["e-"], ["gamma"])
    print("Bad soft B-L Z' attached to e- -> gamma:")
    print(f"  Ward check: {bad.passed} ({bad.reason})")

    _rule("8. FACTORIZATION: which internal lines a process factorizes through")
    print("Channels = internal particles that can go on-shell (locality). No residues.\n")
    rows = [
        ("e- e+ -> mu- mu+ (SM)", ["e-", "e+"], ["mu-", "mu+"], ()),
        ("e- e+ -> mu- mu+ (+Z')", ["e-", "e+"], ["mu-", "mu+"], candidate_vertices(zp_heavy)),
        ("gamma gamma -> e- e+ (+G)", ["gamma", "gamma"], ["e-", "e+"], candidate_vertices(G)),
        ("e- -> gamma", ["e-"], ["gamma"], ()),
    ]
    for label, ini, fin, extra in rows:
        meds = sorted(factorization_mediators(ini, fin, extra_vertices=extra))
        print(f"  {label:26} channels: {', '.join(meds) if meds else '(none)'}")

    _rule("9. UNITARITY CUTS: candidate intermediate-state topologies")
    print("Two-particle cut states, ordered by mass threshold. No rates, no loop")
    print("calculation -- just candidate cut topologies with nominal thresholds.\n")
    cuts = threshold_order(cut_states(["e-", "e+"], ["mu-", "mu+"]))
    print("e- e+ -> mu- mu+, lightest five thresholds:")
    for c in cuts[:5]:
        print(f"  {c.threshold_mev:>10.3f} MeV   {' '.join(c.particles)}")
    heavy = [c for c in cuts if c.threshold_mev and c.threshold_mev > 1e4]
    print(f"  ... up to {' '.join(heavy[-1].particles)} at {heavy[-1].threshold_mev:.0f} MeV")

    _rule("10. MINIMAL REPAIR: from inconsistent model to a fix (checker -> designer)")
    print("Smallest sterile-singlet addition that makes a gauge extension consistent.")
    print("Minimal within the menu; not unique, not naturalness, not a detection claim.\n")
    bl = Candidate("Zp", spin=Fraction(1), mass_mev=0.0, coupling="gauge_charge",
                   charge_map=tuple(BL))
    print("Input: gauge B-L with SM fermions only")
    print(f"  inconsistent: {[d for _, d in consistency_report(bl).failures]}")
    rep = minimal_repair(tuple(BL))
    print(f"  minimal repair: {[(n, str(x)) for n, x in rep.added]}  ({rep.note})")
    print(f"  re-verified anomaly-free: {rep.verified}")
    print(f"  certified minimal: {rep.certificate.minimal}  ({rep.certificate.note})")
    print("  => the engine rediscovers the right-handed neutrino, then a B-L Z'")
    print("     becomes a consistent, exposable mediator.")
    print()


if __name__ == "__main__":
    main()
