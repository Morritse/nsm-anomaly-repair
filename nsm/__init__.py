"""nsm -- a Standard Model consistency engine.

A structural / quantum-number explanation pipeline (exact algebraic constraints
+ graph searches). It derives what gauge consistency forces, classifies
processes, evaluates hypothesized mediators, and checks amplitude-inspired
structural constraints. It computes no physical observables -- see docs/limits.md.

The names below are the per-layer entry points; submodules hold the rest.
"""
# kernel: what gauge consistency forces
from nsm.sm import standard_model_skeleton
from nsm.derive import (
    DerivationResult, Enumeration, derive_hypercharges, enumerate_solutions,
)
# dynamics: which processes are allowed / forbidden, and why
from nsm.verdict import classify, Verdict
from nsm.processes import TreeSearchResult, TreeSearchStatus
# extensions: hypothesized new mediators
from nsm.extensions import (
    Candidate, ConsistencyReport, ExposureResult, FieldType,
    classify_field_type, consistency_report, exposing_verdict, validate_candidate,
)
# amplitudes: soft theorems + Ward identities
from nsm.amplitudes import (
    ExternalLeg, SoftFactor, SoftTerm, WardCheck,
    leading_soft_factor, soft_theorem_report,
)
# factorization: locality / channel enumeration
from nsm.factorization import Channel, factorization_channels, factorization_mediators
# unitarity: candidate intermediate-state topologies
from nsm.unitarity import CutState, cut_states, threshold_order
# repair: rational witnesses and certified-minimality evidence
from nsm.repair import Repair, RepairCertificate, TieredRepair, minimal_repair, tiered_repair
from nsm.repair_map import (
    AtlasCell, RepairAtlas, build_atlas, classify_vector, primitive_representative,
)

__all__ = [
    "standard_model_skeleton", "DerivationResult", "Enumeration",
    "derive_hypercharges", "enumerate_solutions",
    "classify", "Verdict", "TreeSearchResult", "TreeSearchStatus",
    "Candidate", "ConsistencyReport", "ExposureResult", "FieldType",
    "classify_field_type", "consistency_report", "exposing_verdict",
    "validate_candidate",
    "ExternalLeg", "SoftFactor", "SoftTerm", "WardCheck",
    "leading_soft_factor", "soft_theorem_report",
    "Channel", "factorization_channels", "factorization_mediators",
    "CutState", "cut_states", "threshold_order",
    "Repair", "RepairCertificate", "TieredRepair", "minimal_repair",
    "tiered_repair",
    "AtlasCell", "RepairAtlas", "build_atlas", "classify_vector",
    "primitive_representative",
]
