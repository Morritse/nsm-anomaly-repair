"""Generate an explanatory Standard Model diagram from solver results.

The visual borrows the familiar generation-grid shape, but the displayed
hypercharges and electric charges come from ``derive_hypercharges`` rather than
from a hand-entered particle table. Masses and spins remain registry inputs and
are labeled that way in the figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from html import escape
from pathlib import Path

from nsm.derive import derive_hypercharges
from nsm.particles import particle
from nsm.sm import standard_model_skeleton


@dataclass(frozen=True)
class MatterTile:
    symbol: str
    name: str
    particle_name: str
    family: str
    generation: str
    charge: Fraction
    y_left: Fraction
    y_right: Fraction | None
    mass_label: str


@dataclass(frozen=True)
class BosonTile:
    symbol: str
    name: str
    particle_name: str
    family: str
    charge_label: str
    spin_label: str
    mass_label: str


@dataclass(frozen=True)
class DiagramData:
    matter: list[MatterTile]
    bosons: list[BosonTile]
    solver_summary: dict[str, Fraction]
    is_unique: bool


def _fmt_fraction(value: Fraction, signed: bool = False) -> str:
    prefix = "+" if signed and value > 0 else ""
    if value.denominator == 1:
        return f"{prefix}{value.numerator}"
    return f"{prefix}{value.numerator}/{value.denominator}"


def _fmt_mass(name: str) -> str:
    p = particle(name)
    if p.mass_mev is None:
        return "mass input: none"
    if name.startswith("nu_"):
        return "mass input: not fixed"
    if p.mass_mev >= 1000:
        return f"mass input: {p.mass_mev / 1000:.3g} GeV"
    if p.mass_mev == 0:
        return "mass input: 0"
    return f"mass input: {p.mass_mev:.3g} MeV"


def _charge_from_pair(charges: dict, left_component: str, right_component: str | None) -> Fraction:
    left = charges[left_component]
    if right_component is not None:
        right = charges[right_component]
        if right != left:
            raise ValueError(f"left/right electric charges disagree: {left_component}, {right_component}")
    return left


def build_diagram_data() -> DiagramData:
    """Build display data from the actual one-generation derivation."""
    result = derive_hypercharges(standard_model_skeleton())
    if result.satisfiable is not True or result.is_unique is not True:
        raise RuntimeError("SM skeleton did not derive a unique charge assignment")

    y = result.hypercharges
    q = result.charges
    rows = [
        (
            "quark up-type",
            [
                ("u", "up", "u", "I"),
                ("c", "charm", "c", "II"),
                ("t", "top", "t", "III"),
            ],
            "Q_L:up",
            "u_R",
            y["Q_L"],
            y["u_R"],
        ),
        (
            "quark down-type",
            [
                ("d", "down", "d", "I"),
                ("s", "strange", "s", "II"),
                ("b", "bottom", "b", "III"),
            ],
            "Q_L:down",
            "d_R",
            y["Q_L"],
            y["d_R"],
        ),
        (
            "charged lepton",
            [
                ("e", "electron", "e-", "I"),
                ("mu", "muon", "mu-", "II"),
                ("tau", "tau", "tau-", "III"),
            ],
            "L_L:e",
            "e_R",
            y["L_L"],
            y["e_R"],
        ),
        (
            "neutrino",
            [
                ("nu_e", "electron neutrino", "nu_e", "I"),
                ("nu_mu", "muon neutrino", "nu_mu", "II"),
                ("nu_tau", "tau neutrino", "nu_tau", "III"),
            ],
            "L_L:nu",
            None,
            y["L_L"],
            None,
        ),
    ]

    matter = []
    for family, entries, left_component, right_component, y_left, y_right in rows:
        charge = _charge_from_pair(q, left_component, right_component)
        for symbol, name, particle_name, generation in entries:
            matter.append(MatterTile(
                symbol=symbol,
                name=name,
                particle_name=particle_name,
                family=family,
                generation=generation,
                charge=charge,
                y_left=y_left,
                y_right=y_right,
                mass_label=_fmt_mass(particle_name),
            ))

    bosons = [
        BosonTile("g", "gluon", "gluon", "gauge boson", "Q=0", "spin input: 1", _fmt_mass("gluon")),
        BosonTile("photon", "photon", "gamma", "gauge boson", "Q=0", "spin input: 1", _fmt_mass("gamma")),
        BosonTile("Z", "Z boson", "Z", "gauge boson", "Q=0", "spin input: 1", _fmt_mass("Z")),
        BosonTile("W", "W boson", "W+", "gauge boson", "Q=+/-1", "spin input: 1", _fmt_mass("W+")),
        BosonTile("H", "Higgs", "h", "scalar boson", "Q=0", "spin input: 0", _fmt_mass("h")),
    ]

    return DiagramData(
        matter=matter,
        bosons=bosons,
        solver_summary={k: y[k] for k in ["Q_L", "u_R", "d_R", "L_L", "e_R", "H"]},
        is_unique=bool(result.is_unique),
    )


def _text(x, y, text, cls="", anchor="start") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(text)}</text>'


def _rect(x, y, w, h, rx, cls) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'


def _matter_tile(tile: MatterTile, x: int, y: int) -> str:
    charge = f"Q={_fmt_fraction(tile.charge, signed=True)}"
    y_left = f"Y_L={_fmt_fraction(tile.y_left, signed=True)}"
    y_right = "Y_R=n/a" if tile.y_right is None else f"Y_R={_fmt_fraction(tile.y_right, signed=True)}"
    family_cls = "quark" if tile.family.startswith("quark") else "lepton"
    return "\n".join([
        f'<g class="tile matter {family_cls}">',
        _rect(x, y, 198, 146, 18, "tile-bg"),
        _text(x + 16, y + 26, tile.mass_label, "micro muted"),
        f'<circle cx="{x + 99}" cy="{y + 62}" r="34" class="symbol-ring"/>',
        _text(x + 99, y + 74, tile.symbol, "symbol", "middle"),
        _text(x + 99, y + 114, tile.name, "name", "middle"),
        f'<g class="chips">',
        _rect(x + 14, y + 122, 52, 18, 9, "chip-derived"),
        _text(x + 40, y + 135, charge, "chip-text", "middle"),
        _rect(x + 72, y + 122, 54, 18, 9, "chip-derived"),
        _text(x + 99, y + 135, y_left, "chip-text", "middle"),
        _rect(x + 132, y + 122, 54, 18, 9, "chip-derived"),
        _text(x + 159, y + 135, y_right, "chip-text tiny", "middle"),
        "</g>",
        "</g>",
    ])


def _boson_tile(tile: BosonTile, x: int, y: int, wide: bool = False) -> str:
    w = 198 if not wide else 414
    symbol_size = "symbol boson-symbol" if len(tile.symbol) <= 2 else "symbol boson-word"
    return "\n".join([
        f'<g class="tile boson {tile.family.replace(" ", "-")}">',
        _rect(x, y, w, 122, 18, "tile-bg"),
        _text(x + 16, y + 26, tile.mass_label, "micro muted"),
        f'<circle cx="{x + 70}" cy="{y + 62}" r="32" class="symbol-ring"/>',
        _text(x + 70, y + 74, tile.symbol, symbol_size, "middle"),
        _text(x + 142, y + 54, tile.name, "name small"),
        _text(x + 142, y + 82, tile.charge_label, "metric derived-label"),
        _text(x + 142, y + 104, tile.spin_label, "micro muted"),
        "</g>",
    ])


def _wrap_lines(text: str, max_chars: int) -> list[str]:
    lines = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _text_block(x: int, y: int, text: str, cls: str, max_chars: int, line_gap: int = 21) -> list[str]:
    return [_text(x, y + i * line_gap, line, cls) for i, line in enumerate(_wrap_lines(text, max_chars))]


def render_svg(data: DiagramData) -> str:
    """Render the full explanatory figure as an SVG string."""
    width, height = 1320, 1120
    matter_x, matter_y = 160, 252
    tile_w, tile_h, gap_x, gap_y = 198, 146, 20, 18
    col_x = [matter_x + i * (tile_w + gap_x) for i in range(3)]
    row_y = [matter_y + i * (tile_h + gap_y) for i in range(4)]
    generation = {"I": 0, "II": 1, "III": 2}
    row_for_family = {
        "quark up-type": 0,
        "quark down-type": 1,
        "charged lepton": 2,
        "neutrino": 3,
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Derived Standard Model charge map</title>",
        "<desc id=\"desc\">A Standard Model-style grid showing solver-derived hypercharges and electric charges for fermions, plus input boson data and consistency notes.</desc>",
        """<style>
        svg { background: #f7f3e8; font-family: Inter, Arial, sans-serif; color: #122033; }
        .title { font-size: 34px; font-weight: 800; fill: #122033; }
        .subtitle { font-size: 17px; fill: #405064; }
        .section { font-size: 20px; font-weight: 800; fill: #122033; }
        .axis { font-size: 18px; font-weight: 800; fill: #39485d; }
        .row-label { font-size: 17px; font-weight: 800; fill: #39485d; }
        .tile-bg { fill: #fffaf0; stroke-width: 4; filter: drop-shadow(0 4px 0 rgba(20, 32, 45, 0.08)); }
        .quark .tile-bg { stroke: #9b6de3; }
        .lepton .tile-bg { stroke: #54b948; }
        .gauge-boson .tile-bg { stroke: #ef7a5a; }
        .scalar-boson .tile-bg { stroke: #d8b31f; }
        .symbol-ring { fill: #f2efe4; stroke: #ffffff; stroke-width: 5; }
        .quark .symbol-ring { fill: #d7b5ff; }
        .lepton .symbol-ring { fill: #a7ec78; }
        .boson .symbol-ring { fill: #ff9a78; }
        .scalar-boson .symbol-ring { fill: #f7e46e; }
        .symbol { font-size: 42px; fill: #07111f; }
        .boson-symbol { font-size: 41px; }
        .boson-word { font-size: 21px; font-weight: 800; }
        .name { font-size: 24px; font-weight: 800; fill: #07111f; }
        .small { font-size: 21px; }
        .micro { font-size: 13px; fill: #334155; }
        .muted { fill: #536173; }
        .metric { font-size: 17px; font-weight: 800; }
        .chip-derived { fill: #dff5ff; stroke: none; }
        .chip-text { font-size: 11px; font-weight: 800; fill: #0b3954; }
        .tiny { font-size: 9px; }
        .derived-label { fill: #0b6580; font-weight: 800; }
        .input-label { fill: #7b5d12; font-weight: 800; }
        .panel { fill: #ffffff; stroke: #d9d1c0; stroke-width: 2; }
        .panel-title { font-size: 16px; font-weight: 800; fill: #122033; }
        .panel-text { font-size: 14px; fill: #405064; }
        .callout { fill: #e7f7ee; stroke: #59b37a; stroke-width: 2; }
        .warning { fill: #fff2df; stroke: #e0a03c; stroke-width: 2; }
        .flow { stroke: #6c7a89; stroke-width: 2; fill: none; marker-end: url(#arrow); }
        </style>""",
        """<defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#6c7a89"/>
        </marker>
        </defs>""",
        _text(62, 64, "Derived Standard Model charge map", "title"),
        _text(62, 94, "The familiar particle grid, but Q and Y are generated from anomaly cancellation, declared Yukawa U(1) charge sums, and Q = T3 + Y/2.", "subtitle"),
    ]

    pipeline = [
        ("reps", "SU(3) x SU(2) reps"),
        ("anomaly", "anomaly + Witten checks"),
        ("yukawa", "Yukawa invariance"),
        ("gmn", "GMN relation"),
        ("result", "derived Y and Q"),
    ]
    px, py = 62, 126
    for i, (_, label) in enumerate(pipeline):
        x = px + i * 220
        parts.append(_rect(x, py, 176, 54, 12, "panel"))
        parts.append(_text(x + 88, py + 24, label, "panel-title", "middle"))
        if i < len(pipeline) - 1:
            parts.append(f'<path d="M{x + 182},{py + 27} L{x + 214},{py + 27}" class="flow"/>')
    parts.append(_rect(1152, 126, 106, 54, 12, "callout"))
    parts.append(_text(1205, 149, f"unique: {data.is_unique}", "panel-title", "middle"))
    parts.append(_text(1205, 169, "solver result", "micro", "middle"))

    parts.append(_text(62, 224, "Three generations of matter", "section"))
    for gen, idx in generation.items():
        parts.append(_text(col_x[idx] + tile_w / 2, 242, gen, "axis", "middle"))
    for label, idx in [
        ("up quarks", 0),
        ("down quarks", 1),
        ("charged leptons", 2),
        ("neutrinos", 3),
    ]:
        parts.append(_text(144, row_y[idx] + 82, label, "row-label", "end"))

    for tile in data.matter:
        x = col_x[generation[tile.generation]]
        y = row_y[row_for_family[tile.family]]
        parts.append(_matter_tile(tile, x, y))

    bx = 832
    parts.append(_text(bx, 224, "Force carriers and scalar", "section"))
    boson_positions = {
        "gluon": (bx, 252, False),
        "gamma": (bx + 218, 252, False),
        "Z": (bx, 392, False),
        "W+": (bx + 218, 392, False),
        "h": (bx, 532, True),
    }
    for tile in data.bosons:
        x, y, wide = boson_positions[tile.particle_name]
        parts.append(_boson_tile(tile, x, y, wide=wide))

    sx, sy = 832, 690
    parts.append(_rect(sx, sy, 414, 170, 16, "panel"))
    parts.append(_text(sx + 20, sy + 32, "What is actually derived", "panel-title"))
    summary_lines = [
        f"Y(Q_L)={_fmt_fraction(data.solver_summary['Q_L'], signed=True)}",
        f"Y(u_R)={_fmt_fraction(data.solver_summary['u_R'], signed=True)}",
        f"Y(d_R)={_fmt_fraction(data.solver_summary['d_R'], signed=True)}",
        f"Y(L_L)={_fmt_fraction(data.solver_summary['L_L'], signed=True)}",
        f"Y(e_R)={_fmt_fraction(data.solver_summary['e_R'], signed=True)}",
        f"Y(H)={_fmt_fraction(data.solver_summary['H'], signed=True)}",
    ]
    for i, line in enumerate(summary_lines):
        parts.append(_text(sx + 24 + (i % 2) * 178, sy + 64 + (i // 2) * 30, line, "metric derived-label"))
    parts.append(_text(sx + 24, sy + 154, "Masses and spins are inputs; exact values are not SM-derived here.", "panel-text"))

    bx2, by2 = 62, 934
    panels = [
        ("anomaly-only", "2 solutions: u_R and d_R can swap without Yukawa constraints.", "warning"),
        ("free nu_R", "non-unique: a B-L flat direction opens.", "warning"),
        ("Majorana nu_R", "unique again: Y(nu_R)=0 and Q(nu_R)=0.", "callout"),
        ("process layer", "mu decay uses a W tree; mu -> e gamma is loop/mixing only.", "callout"),
    ]
    for i, (title, text, cls) in enumerate(panels):
        x = bx2 + i * 306
        parts.append(_rect(x, by2, 276, 112, 14, cls))
        parts.append(_text(x + 18, by2 + 30, title, "panel-title"))
        parts.extend(_text_block(x + 18, by2 + 60, text, "panel-text", 34))

    parts.append(_text(62, 1082, "Generated by nsm.visualize from derive_hypercharges(standard_model_skeleton()).", "micro muted"))
    parts.append("</svg>")
    return "\n".join(parts)


def render_html(svg: str) -> str:
    return "\n".join([
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>Derived Standard Model charge map</title>",
        "<style>body{margin:0;background:#f7f3e8;}main{max-width:1320px;margin:0 auto;}svg{width:100%;height:auto;display:block;}</style>",
        "</head>",
        "<body><main>",
        svg,
        "</main></body>",
        "</html>",
    ])


def write_visualization(out_dir: Path | str = "figures") -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_diagram_data()
    svg = render_svg(data)
    svg_path = out / "derived-standard-model.svg"
    html_path = out / "derived-standard-model.html"
    svg_path.write_text(svg, encoding="utf-8")
    html_path.write_text(render_html(svg), encoding="utf-8")
    return svg_path, html_path


def main() -> None:
    svg_path, html_path = write_visualization()
    print(f"wrote {svg_path}")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
