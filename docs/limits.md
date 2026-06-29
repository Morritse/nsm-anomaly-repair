# What nsm does NOT do

`nsm` is a **structural / quantum-number consistency engine**. It reasons about
representations, conserved charges, vertices, and tree topology. It does not
compute physical observables. Read this before quoting any result.

## Hard boundaries

- **No cross sections, decay rates, or branching ratios.** A verdict of
  "allowed at tree level" means a tree diagram exists, not that it is large — or
  even observable.
- **No kinematics.** There are no momenta, spinors, polarization vectors, or
  phase-space integrals. Mass appears only as a number used for crude
  threshold/`n→1` checks, never in an amplitude.
- **No amplitude values.** The soft-theorem layer checks the *Ward identity* of
  the leading soft factor symbolically; it never evaluates the factor. The
  factorization layer enumerates *which* channels exist; it never computes a
  residue. `Res A = Σ A_L × A_R` is **out of scope** (it requires the kinematics
  layer that does not exist yet).
- **No detector simulation, no statistical significance, no "discovery."**
  "Which processes expose a Z′/graviton" means which processes it can contribute
  to topologically — not whether any experiment could see it.
- **No proof of graviton (or any particle) detection.** The graviton work shows
  that a universal massless mediator must be spin-2 and is internally consistent,
  and which processes it could mediate. That is a consistency statement, not
  evidence of existence or detectability.

## What the kernel does NOT derive

Forced by the engine: hypercharges/electric charges, allowed/forbidden quantum
numbers, anomaly consistency of a spectrum. **Not** derived (these are empirical
inputs to the Standard Model): particle **masses**, **CKM/PMNS mixing**, **Yukawa
/ coupling values**, and the **number of generations**. If the engine ever
appears to "predict" one of these, treat it as a bug, not a discovery.

## Modeling simplifications currently in place

- Convention `Q = T3 + Y/2`; anomaly coefficients carry an overall normalization
  that does not affect whether they vanish.
- Vertex set is **CKM-diagonal**; Cabibbo-suppressed *tree* processes are
  reported as having no tree channel.
- Composite hadrons are **not expanded** into quark constituents for the tree
  search (e.g. β-decay is flagged, not resolved as a quark-level tree).
- Candidate gauge couplings are **family-universal and vector-like**; chiral /
  axial and flavor-non-universal couplings are out of scope (and explicitly
  rejected rather than silently mishandled).
- Three-gluon and Higgs-trilinear vertices are modeled, but quartic vertices and
  some multi-gauge/scalar interactions remain out of scope. "No tree found"
  therefore means no tree in the configured vertex subset.

## The one honest summary line

> nsm establishes structural and quantum-number consistency. It does not compute,
> measure, or predict any physical observable.
