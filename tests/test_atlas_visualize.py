from nsm.atlas_visualize import build_visual_data, render_html, write_repair_atlas_visualization


def test_visual_data_projects_real_repair_atlas_points():
    data = build_visual_data(radius=1, max_added=4, timeout_ms=4000, workers=1)
    assert data.raw_count == 243
    assert data.distinct_count == 122
    assert data.counts["colorless-exotic-repairable"] == 8
    assert data.points
    point = next(p for p in data.points if p.tier == "sterile-repairable")
    assert point.charges == (0, 1, -1, 0, -1)
    assert -1 <= point.x <= 1
    assert -1 <= point.y <= 1
    assert set(point.anomalies) == {"SU3^2-X", "SU2^2-X", "grav^2-X", "Y^2-X", "Y-X^2", "X^3"}


def test_render_html_contains_interactive_atlas_payload():
    html = render_html(build_visual_data(radius=0, max_added=2, timeout_ms=4000, workers=1))
    assert "<!doctype html>" in html
    assert "Repair Census Landscape" in html
    assert "const DATA =" in html
    assert "2D PCA projection" in html
    assert "already-consistent" in html


def test_write_repair_atlas_visualization_outputs_html(tmp_path):
    path = write_repair_atlas_visualization(tmp_path, radius=0, max_added=2,
                                            timeout_ms=4000, workers=1)
    assert path.name == "repair-census.html"
    assert path.exists()
    assert "Repair Census Landscape" in path.read_text(encoding="utf-8")
