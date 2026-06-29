from nsm.repair_map import (
    build_atlas,
    canonical_grid,
    classify_vector,
    csv_rows,
    hypercharge_shift,
    latex_summary_table,
    primitive_representative,
    summary_markdown,
)


def test_primitive_representative_divides_gcd_and_fixes_sign():
    assert primitive_representative((2, 4, -2, 0, 0)) == (1, 2, -1, 0, 0)
    assert primitive_representative((-2, -4, 2, 0, 0)) == (1, 2, -1, 0, 0)
    assert primitive_representative((0, 0, 0, 0, 0)) == (0, 0, 0, 0, 0)


def test_canonical_grid_deduplicates_scaling_and_sign():
    reps = canonical_grid(1)
    assert (0, 0, 0, 0, 0) in reps
    assert (1, 0, 0, 0, 0) in reps
    assert (-1, 0, 0, 0, 0) not in reps
    assert len(reps) < 3 ** 5


def test_classify_vector_reports_minimum_repair_tier_and_solver_honesty():
    cell = classify_vector((0, 0, 0, 0, 1), max_added=4, timeout_ms=4000)
    assert cell.tier == "colorless-exotic-repairable"
    assert cell.verified
    assert cell.certified
    assert cell.rungs
    assert not cell.has_uncertified_rung
    assert ("E_L", "1") in cell.added


def test_hypercharge_shift_preserves_the_physical_tier_for_b_minus_l():
    b_minus_l = (1, 1, 1, -3, -3)
    shifted = hypercharge_shift(b_minus_l, 1)
    a = classify_vector(b_minus_l, max_added=2, timeout_ms=4000)
    b = classify_vector(shifted, max_added=2, timeout_ms=4000)
    assert a.tier == b.tier == "sterile-repairable"


def test_atlas_n1_classification_matches_golden():
    # Regression guard for the solver-path optimization: the minimum repair tier
    # for every N=1 primitive representative must be unchanged by pruning/memoizing.
    import json
    import pathlib
    golden = json.loads((pathlib.Path(__file__).parent / "golden_atlas_n1.json").read_text())
    for key, tier in golden.items():
        vec = tuple(int(x) for x in key.split(","))
        assert classify_vector(vec, max_added=6, timeout_ms=4000).tier == tier, vec


def test_build_atlas_parallel_matches_serial():
    serial = build_atlas(radius=1, max_added=4, timeout_ms=4000, workers=1)
    parallel = build_atlas(radius=1, max_added=4, timeout_ms=4000, workers=4)
    assert parallel.cells == serial.cells


def test_latex_summary_table_escapes_percent():
    # A bare '%' is a LaTeX comment char (eats the row terminator); the LaTeX
    # exporter must emit \% so the generated table compiles.
    atlas = build_atlas(radius=0, max_added=2, timeout_ms=4000)
    latex = latex_summary_table(atlas)
    assert "\\%" in latex
    assert "%" not in latex.replace("\\%", "")


def test_radius_zero_atlas_and_exports_are_reproducible():
    atlas = build_atlas(radius=0, max_added=2, timeout_ms=4000)
    assert atlas.raw_count == 1
    assert atlas.distinct_count == 1
    assert atlas.counts["already-consistent"] == 1
    assert atlas.uncertified_count == 0
    assert "N=0" in atlas.parameters()
    assert "already-consistent" in summary_markdown(atlas)
    assert "\\begin{tabular}" in latex_summary_table(atlas)
    assert "Q_L,u_R,d_R,L_L,e_R,tier" in csv_rows(atlas)
