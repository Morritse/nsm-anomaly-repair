from fractions import Fraction as F

from nsm.visualize import build_diagram_data, render_svg, write_visualization


def test_visual_data_uses_solver_derived_charges():
    data = build_diagram_data()
    assert data.is_unique is True
    assert data.solver_summary["Q_L"] == F(1, 3)
    assert data.solver_summary["L_L"] == F(-1)

    electron = next(tile for tile in data.matter if tile.particle_name == "e-")
    assert electron.charge == F(-1)
    assert electron.y_left == F(-1)
    assert electron.y_right == F(-2)

    neutrino = next(tile for tile in data.matter if tile.particle_name == "nu_e")
    assert neutrino.charge == F(0)
    assert neutrino.y_right is None


def test_render_svg_contains_derivation_labels():
    svg = render_svg(build_diagram_data())
    assert "Derived Standard Model charge map" in svg
    assert "Y(Q_L)=+1/3" in svg
    assert "Q=-1" in svg
    assert "Masses and spins are inputs" in svg


def test_write_visualization_outputs_svg_and_html(tmp_path):
    svg_path, html_path = write_visualization(tmp_path)
    assert svg_path.exists()
    assert html_path.exists()
    assert svg_path.read_text(encoding="utf-8").startswith("<svg")
    assert "<!doctype html>" in html_path.read_text(encoding="utf-8")
