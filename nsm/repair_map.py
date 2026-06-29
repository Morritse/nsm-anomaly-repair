"""Repair-census scans for one-generation U(1)' charge assignments.

This module is a reporting layer over :func:`nsm.repair.tiered_repair`. It scans
integer charge vectors, reduces rescaling/sign duplicates to primitive
representatives, reports the repair tier found for each representative, and
records minimality-certification metadata. The counts are a basis-dependent
integer slice, not a measure over all U(1)' extensions.
"""
from __future__ import annotations

import csv
import itertools
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from io import StringIO
from math import gcd
from typing import Iterable

from nsm.repair import tiered_repair

FIELDS = ("Q_L", "u_R", "d_R", "L_L", "e_R")
HYPERCHARGE_INTEGER = (1, 4, -2, -3, -6)  # 3Y in the field order above.
TIER_ORDER = (
    "already-consistent",
    "sterile-repairable",
    "colorless-exotic-repairable",
    "colored-exotic-required",
    "budget-limited",
    "structurally-blocked",
)
CERTIFYING_RUNG_VERDICTS = {"support-prune", "unsat"}


@dataclass(frozen=True)
class AtlasCell:
    charges: tuple[int, ...]
    tier: str
    added: tuple[tuple[str, str], ...]
    verified: bool
    certified: bool
    rungs: tuple[tuple[str, str], ...]
    blocked_by: tuple[str, ...]
    note: str

    @property
    def has_uncertified_rung(self) -> bool:
        return any(verdict not in CERTIFYING_RUNG_VERDICTS
                   for _, verdict in self.rungs)

    @property
    def minimality_certified(self) -> bool:
        return self.certified

    @property
    def solver_uncertified_rung(self) -> bool:
        return self.has_uncertified_rung


@dataclass(frozen=True)
class RepairAtlas:
    radius: int
    max_added: int
    timeout_ms: int
    raw_count: int
    cells: tuple[AtlasCell, ...]
    basis: str = "integer charges over (Q_L,u_R,d_R,L_L,e_R)"
    menu: str = "N/E/L/D/Q"
    dedup_rule: str = "primitive sign-fixed representative; no hypercharge quotient"

    @property
    def distinct_count(self) -> int:
        return len(self.cells)

    @property
    def dedup_ratio(self) -> float:
        return self.raw_count / self.distinct_count if self.distinct_count else 0.0

    @property
    def counts(self) -> Counter:
        return Counter(cell.tier for cell in self.cells)

    @property
    def uncertified_count(self) -> int:
        return sum(1 for cell in self.cells if cell.has_uncertified_rung)

    @property
    def uncertified_fraction(self) -> Fraction:
        if not self.cells:
            return Fraction(0)
        return Fraction(self.uncertified_count, len(self.cells))

    def parameters(self) -> str:
        return (
            f"N={self.radius}, basis={self.basis}, menu={self.menu}, "
            f"max_added={self.max_added}, timeout_ms={self.timeout_ms}, "
            f"dedup={self.dedup_rule}"
        )


def primitive_representative(vector: Iterable[int]) -> tuple[int, ...]:
    """Return the primitive sign-fixed representative of an integer vector."""
    vals = tuple(int(v) for v in vector)
    if not vals:
        raise ValueError("charge vector cannot be empty")
    g = 0
    for v in vals:
        g = gcd(g, abs(v))
    if g == 0:
        return vals
    vals = tuple(v // g for v in vals)
    first = next(v for v in vals if v != 0)
    if first < 0:
        vals = tuple(-v for v in vals)
    return vals


def integer_grid(radius: int) -> Iterable[tuple[int, ...]]:
    """All integer charge assignments with |q| <= radius, including zero once."""
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    yield from itertools.product(range(-radius, radius + 1), repeat=len(FIELDS))


def canonical_grid(radius: int) -> tuple[tuple[int, ...], ...]:
    """Primitive sign-fixed representatives in deterministic lexicographic order."""
    reps = {primitive_representative(v) for v in integer_grid(radius)}
    return tuple(sorted(reps))


def hypercharge_shift(vector: Iterable[int], k: int) -> tuple[int, ...]:
    """Integer hypercharge-basis shift X -> X + k*(3Y)."""
    vals = tuple(int(v) for v in vector)
    if len(vals) != len(HYPERCHARGE_INTEGER):
        raise ValueError(f"expected {len(HYPERCHARGE_INTEGER)} charges")
    return tuple(v + k * y for v, y in zip(vals, HYPERCHARGE_INTEGER))


def charge_map(vector: Iterable[int]) -> tuple[tuple[str, Fraction], ...]:
    vals = tuple(vector)
    if len(vals) != len(FIELDS):
        raise ValueError(f"expected {len(FIELDS)} charges")
    return tuple((field, Fraction(value)) for field, value in zip(FIELDS, vals))


def _display_tier(tier: str, blocked_by: tuple[str, ...]) -> str:
    if tier != "still-blocked":
        return tier
    return "structurally-blocked" if blocked_by else "budget-limited"


def classify_vector(vector: Iterable[int], max_added: int = 6,
                    timeout_ms: int = 4000) -> AtlasCell:
    charges = primitive_representative(vector)
    result = tiered_repair(charge_map(charges), max_added=max_added, timeout_ms=timeout_ms)
    repair = result.repair
    blocked_by = tuple(repair.blocked_by)
    cert = repair.certificate
    rungs = tuple((str(size), str(verdict)) for size, verdict in (cert.rungs if cert else ()))
    return AtlasCell(
        charges=charges,
        tier=_display_tier(result.tier, blocked_by),
        added=tuple((name, str(x)) for name, x in repair.added),
        verified=repair.verified,
        certified=bool(cert and cert.minimal),
        rungs=rungs,
        blocked_by=blocked_by,
        note=repair.note,
    )


def _classify_cell(args):
    """Top-level worker so cells can be classified in a process pool."""
    vector, max_added, timeout_ms = args
    return classify_vector(vector, max_added=max_added, timeout_ms=timeout_ms)


def _multiprocessing_context(method: str | None = None):
    import multiprocessing

    methods = [method] if method else ["spawn", "forkserver", "fork"]
    last_error = None
    for name in methods:
        try:
            return multiprocessing.get_context(name)
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"no supported multiprocessing context from {methods}") from last_error


def build_atlas(radius: int = 2, max_added: int = 6, timeout_ms: int = 4000,
                workers: int = 1, mp_context: str | None = None) -> RepairAtlas:
    """Scan the canonical grid and classify every representative. ``workers`` > 1
    classifies cells in parallel across processes (cells are independent); the
    result is identical to and ordered like the serial scan."""
    reps = canonical_grid(radius)
    work = [(rep, max_added, timeout_ms) for rep in reps]
    if workers and workers > 1 and len(reps) > 1:
        from concurrent.futures import ProcessPoolExecutor
        ctx = _multiprocessing_context(mp_context)
        chunk = max(1, len(reps) // (workers * 4))
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            cells = tuple(pool.map(_classify_cell, work, chunksize=chunk))
    else:
        cells = tuple(_classify_cell(w) for w in work)
    return RepairAtlas(
        radius=radius,
        max_added=max_added,
        timeout_ms=timeout_ms,
        raw_count=(2 * radius + 1) ** len(FIELDS),
        cells=cells,
    )


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0.0%"
    return f"{100 * part / whole:.1f}%"


def summary_markdown(atlas: RepairAtlas) -> str:
    counts = atlas.counts
    lines = [
        "# U(1)' Repair Census",
        "",
        f"Parameters: `{atlas.parameters()}`.",
        "",
        f"Raw grid cells: {atlas.raw_count}",
        f"Primitive sign-fixed representatives: {atlas.distinct_count}",
        f"Dedup ratio: {atlas.dedup_ratio:.2f}x",
        f"Uncertified cells: {atlas.uncertified_count} "
        f"({_pct(atlas.uncertified_count, atlas.distinct_count)})",
        "",
        "| Tier | Count | Share | Example repair |",
        "|---|---:|---:|---|",
    ]
    for tier in TIER_ORDER:
        count = counts.get(tier, 0)
        if count == 0:
            continue
        example = next(cell for cell in atlas.cells if cell.tier == tier)
        added = ", ".join(f"{name}={x}" for name, x in example.added) or "-"
        lines.append(f"| {tier} | {count} | {_pct(count, atlas.distinct_count)} | {added} |")
    lines.extend([
        "",
        "Budget-limited means no repair was found within `max_added`; it is not an "
        "unrepairability claim.",
    ])
    return "\n".join(lines) + "\n"


def latex_summary_table(atlas: RepairAtlas) -> str:
    counts = atlas.counts
    rows = []
    for tier in TIER_ORDER:
        count = counts.get(tier, 0)
        if count:
            share = _pct(count, atlas.distinct_count).replace("%", r"\%")
            rows.append(f"{tier} & {count} & {share} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{@{}lrr@{}}\n"
        "\\toprule\n"
        "tier & count & share\\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def csv_rows(atlas: RepairAtlas) -> str:
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Q_L", "u_R", "d_R", "L_L", "e_R", "tier", "added", "verified",
        "minimality_certified", "solver_uncertified_rung", "blocked_by",
        "note", "rungs",
    ])
    for cell in atlas.cells:
        writer.writerow([
            *cell.charges,
            cell.tier,
            ";".join(f"{name}={x}" for name, x in cell.added),
            cell.verified,
            cell.certified,
            cell.has_uncertified_rung,
            ";".join(cell.blocked_by),
            cell.note,
            ";".join(f"{size}:{verdict}" for size, verdict in cell.rungs),
        ])
    return buf.getvalue()


def classify_grid(assignments, max_added=4, timeout_ms=4000):
    """Compatibility helper: classify explicit vectors without canonical dedup."""
    counts = Counter()
    examples = {}
    for combo in assignments:
        cell = classify_vector(combo, max_added=max_added, timeout_ms=timeout_ms)
        counts[cell.tier] += 1
        examples.setdefault(cell.tier, (tuple(combo), list(cell.added)))
    return counts, examples
