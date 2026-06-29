# Case study: gauging B−L, and rediscovering the right-handed neutrino

This is the shortest end-to-end story of what `nsm` does. It runs the full
pipeline on one question and shows the engine *designing*, not just classifying:

```
detect inconsistency  ->  explain the obstruction  ->  search a minimal repair
                      ->  re-verify  ->  expose the consequences
```

Every number below is real tool output. Reproduce it with the snippet at the end.

## The question

The Standard Model has an accidental global symmetry **B−L** (baryon minus
lepton number). What if we try to *gauge* it — promote it to a new force with
its own boson, a `Z'`? Assign each SM fermion its B−L charge:

| field | Q_L | u_R | d_R | L_L | e_R |
|---|---|---|---|---|---|
| **B−L (X)** | +1/3 | +1/3 | +1/3 | −1 | −1 |

## Step 1 — detect + explain the obstruction

Gauging a new U(1)′ alongside hypercharge requires **every** anomaly of
`SU(3) × SU(2) × U(1)_Y × U(1)_X` to vanish — including the *mixed* `Y²·X` and
`Y·X²` that a single-U(1) check would miss. The engine computes all eleven:

```
SU3^3    = 0      SU3^2-Y  = 0      SU3^2-X  = 0
SU2^2-Y  = 0      SU2^2-X  = 0      grav^2-Y = 0
grav^2-X = -1   <-- FAILS          Y^3      = 0
Y^2-X    = 0      Y-X^2    = 0      X^3      = -1   <-- FAILS
```

So gauged B−L is **inconsistent** — but the engine says *exactly why*: only the
gravitational `grav²·X` and the cubic `X³` fail. Notably the mixed anomalies
`Y²·X`, `Y·X²` already cancel; the obstruction is narrow and specific.

## Step 2 — search a minimal repair

Ask for the smallest addition (v1 menu: sterile right-handed singlets) that
cancels the residual anomalies:

```
minimal_repair(B−L)
  -> added:    [('nu_R', -1)]
     note:     add 1 sterile RH singlet(s) per generation
     verified: True
     certificate.minimal: True
       "a smaller (zero-field) repair would leave the input inconsistent"
```

A sterile singlet (color/weak singlet, `Y = 0`) touches **only** `grav²·X` and
`X³`, so cancellation needs `Σ Xᵢ = -1` and `Σ Xᵢ³ = -1` — uniquely solved by a
single field with `X = -1`. The engine has **rediscovered the right-handed
neutrino**: it was not told the answer; it searched for it.

The repair is **certified minimal**, not merely asserted: for each smaller field
count the solver returns `unsat` (a decision over the reals, so no charges of
that size work — rational or not), and that ladder of `unsat` rungs *is* the
certificate. Here the only smaller size is zero fields — the unrepaired,
inconsistent input — so one field is minimal. A size-two repair would carry the
rung `[(1, "unsat")]`; a size the solver cannot decide is recorded honestly and
minimality is *not* claimed.

## Step 3 — re-verify

Adding `ν_R` (X = −1) and re-running the *full* anomaly set:

```
consistency_report(B−L + nu_R).is_consistent  ->  True
```

All eleven coefficients now vanish, including the mixed ones. The repaired model
is genuinely anomaly-free.

## Step 4 — expose the consequences

The B−L `Z'` is now a consistent mediator. What processes would expose it?

```
e- e+ -> Zp           ALLOWED_AT_TREE_LEVEL
  - involves Zp; amplitude is weak-suppressed                  (resonant production)
e- e+ -> mu- mu+      ALLOWED_AT_TREE_LEVEL
  - Zp also mediates this process (s-/t-channel resonance / interference)
mu- -> e- Zp          ALLOWED_ONLY_WITH_MIXING_OR_LOOPS
  - Zp couples flavor-diagonally, so this is not a generic Zp signal
```

## Why this is the canonical demo

It is not cosmetic. One run threads together five things that are usually
separate facts in a textbook:

```
gauged B-L  ->  full (incl. mixed) anomaly check  ->  minimal sterile repair
            ->  right-handed-neutrino motivation  ->  Z' process exposure
```

That chain — *a broken model, the precise obstruction, the minimal fix, and what
the fix implies* — is the thing the tool does that calculators and simulators do
not.

## What this is NOT

A structural / quantum-number result. No cross sections, no rates, no detector
significance, no claim that this `Z'` or `ν_R` exists or is natural. The repair
is minimal *within the sterile-singlet menu*, and not unique. See
[limits.md](limits.md).

## Reproduce it

```python
from fractions import Fraction as F
from nsm.extensions import Candidate, consistency_report, exposing_verdict
from nsm.repair import minimal_repair

BL = (("Q_L", F(1,3)), ("u_R", F(1,3)), ("d_R", F(1,3)), ("L_L", F(-1)), ("e_R", F(-1)))

zp = Candidate("Zp", spin=F(1), mass_mev=0.0, coupling="gauge_charge", charge_map=BL)
print(consistency_report(zp).failures)        # grav^2-X = -1, X^3 = -1
print(minimal_repair(BL).added)               # [('nu_R', Fraction(-1))]

fixed = BL + (("nu_R", F(-1)),)
zp2 = Candidate("Zp", spin=F(1), mass_mev=2000.0, coupling="gauge_charge",
                coupling_scale="weak", charge_map=fixed)
print(consistency_report(zp2).is_consistent)  # True
print(exposing_verdict(["e-","e+"], ["mu-","mu+"], zp2).notes)  # Zp mediates ...
```

(Or just run `python -m nsm` and read section 10.)
