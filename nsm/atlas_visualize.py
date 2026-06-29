"""Generate an interactive repair-census landscape.

The figure projects each primitive U(1)' charge assignment into a two-dimensional
view of its six X-sector anomaly coefficients. It is a schematic projection of a
higher-dimensional anomaly vector, not a physical coordinate system.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from html import escape
from pathlib import Path

from nsm.anomalies import gauge_anomaly_coefficients
from nsm.extensions import gauge_charge_fermions
from nsm.repair_map import FIELDS, TIER_ORDER, RepairAtlas, build_atlas, charge_map

X_ANOMALIES = ("SU3^2-X", "SU2^2-X", "grav^2-X", "Y^2-X", "Y-X^2", "X^3")


@dataclass(frozen=True)
class AtlasPoint:
    id: int
    charges: tuple[int, ...]
    tier: str
    x: float
    y: float
    radius: float
    certified: bool
    solver_uncertified: bool
    anomalies: dict[str, str]
    anomaly_values: dict[str, float]
    added: tuple[tuple[str, str], ...]
    note: str
    rung_count: int


@dataclass(frozen=True)
class AtlasVisualData:
    title: str
    subtitle: str
    parameters: str
    raw_count: int
    distinct_count: int
    dedup_ratio: float
    uncertified_count: int
    counts: dict[str, int]
    axes: dict[str, dict[str, float]]
    points: tuple[AtlasPoint, ...]


def _frac_to_str(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _signed_log(value: Fraction) -> float:
    v = float(value)
    if v == 0:
        return 0.0
    return math.copysign(math.log1p(abs(v)), v)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(_dot(v, v))
    if norm == 0:
        return [0.0 for _ in v]
    return [x / norm for x in v]


def _mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def _power_component(matrix: list[list[float]], seed: list[float]) -> list[float]:
    v = _normalize(seed)
    for _ in range(80):
        nxt = _normalize(_mat_vec(matrix, v))
        if sum(abs(a - b) for a, b in zip(v, nxt)) < 1e-10:
            break
        v = nxt
    return v


def _principal_axes(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    if not rows:
        return [1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]
    n, d = len(rows), len(rows[0])
    means = [sum(row[j] for row in rows) / n for j in range(d)]
    centered = [[row[j] - means[j] for j in range(d)] for row in rows]
    cov = [[sum(row[i] * row[j] for row in centered) / max(1, n - 1)
            for j in range(d)] for i in range(d)]
    pc1 = _power_component(cov, [1.0] + [0.25] * (d - 1))
    lam1 = _dot(pc1, _mat_vec(cov, pc1))
    deflated = [[cov[i][j] - lam1 * pc1[i] * pc1[j] for j in range(d)] for i in range(d)]
    pc2 = _power_component(deflated, [0.15, 1.0] + [0.25] * (d - 2))
    # Gram-Schmidt cleanup, mostly for numerical stability.
    overlap = _dot(pc1, pc2)
    pc2 = _normalize([v - overlap * pc1[i] for i, v in enumerate(pc2)])
    return pc1, pc2


def _anomalies_for(charges: tuple[int, ...]) -> dict[str, Fraction]:
    coeffs = gauge_anomaly_coefficients(gauge_charge_fermions(charge_map(charges)))
    return {key: Fraction(coeffs[key]) for key in X_ANOMALIES}


def build_visual_data(radius: int = 2, max_added: int = 6, timeout_ms: int = 8000,
                      workers: int = 1) -> AtlasVisualData:
    atlas = build_atlas(radius=radius, max_added=max_added,
                        timeout_ms=timeout_ms, workers=workers)
    return visual_data_from_atlas(atlas)


def visual_data_from_atlas(atlas: RepairAtlas) -> AtlasVisualData:
    anomalies = [_anomalies_for(cell.charges) for cell in atlas.cells]
    rows = [[_signed_log(a[key]) for key in X_ANOMALIES] for a in anomalies]
    pc1, pc2 = _principal_axes(rows)
    projections = [(_dot(row, pc1), _dot(row, pc2)) for row in rows]
    max_abs_x = max((abs(x) for x, _ in projections), default=1.0) or 1.0
    max_abs_y = max((abs(y) for _, y in projections), default=1.0) or 1.0

    points = []
    for idx, (cell, anomaly, (x_raw, y_raw)) in enumerate(zip(atlas.cells, anomalies, projections)):
        total = sum(abs(float(v)) for v in anomaly.values())
        points.append(AtlasPoint(
            id=idx,
            charges=cell.charges,
            tier=cell.tier,
            x=x_raw / max_abs_x,
            y=y_raw / max_abs_y,
            radius=max(3.0, min(11.0, 2.7 + math.log1p(total))),
            certified=cell.certified,
            solver_uncertified=cell.has_uncertified_rung,
            anomalies={k: _frac_to_str(v) for k, v in anomaly.items()},
            anomaly_values={k: float(v) for k, v in anomaly.items()},
            added=cell.added,
            note=cell.note,
            rung_count=len(cell.rungs),
        ))

    return AtlasVisualData(
        title="Repair Census Landscape",
        subtitle=("A spatial projection of primitive U(1)' charge assignments by "
                  "their anomaly vector and minimum repair tier."),
        parameters=atlas.parameters(),
        raw_count=atlas.raw_count,
        distinct_count=atlas.distinct_count,
        dedup_ratio=round(atlas.dedup_ratio, 3),
        uncertified_count=atlas.uncertified_count,
        counts={tier: atlas.counts.get(tier, 0) for tier in TIER_ORDER},
        axes={
            "pc1": {k: round(v, 4) for k, v in zip(X_ANOMALIES, pc1)},
            "pc2": {k: round(v, 4) for k, v in zip(X_ANOMALIES, pc2)},
        },
        points=tuple(points),
    )


def _json_default(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def render_html(data: AtlasVisualData) -> str:
    payload = json.dumps(data, default=_json_default, separators=(",", ":"))
    title = escape(data.title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f2e8;
  --ink: #172033;
  --muted: #647084;
  --panel: #fffdf7;
  --line: #d7cfbf;
  --zero: #6b7280;
  --sterile: #2b6cb0;
  --colorless: #1f8a70;
  --colored: #c77700;
  --budget: #b64242;
  --blocked: #6b3fa0;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }}
main {{ min-height: 100vh; display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 0; }}
.stage {{ padding: 24px 26px 26px; min-width: 0; }}
.side {{ border-left: 1px solid var(--line); background: rgba(255,255,255,.46); padding: 22px; overflow: auto; }}
h1 {{ margin: 0 0 6px; font-size: 28px; line-height: 1.08; letter-spacing: 0; }}
.subtitle {{ margin: 0; color: var(--muted); font-size: 14px; max-width: 900px; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0 14px; }}
.pill {{ border: 1px solid var(--line); background: var(--panel); border-radius: 6px; padding: 7px 9px; font-size: 12px; color: #394255; }}
.viz-wrap {{ position: relative; height: min(72vh, 760px); min-height: 520px; border: 1px solid var(--line); background: #fffaf0; border-radius: 8px; overflow: hidden; }}
canvas {{ width: 100%; height: 100%; display: block; }}
.legend {{ position: absolute; left: 14px; top: 14px; display: flex; flex-wrap: wrap; gap: 8px; max-width: calc(100% - 28px); }}
.legend label {{ display: inline-flex; align-items: center; gap: 6px; border: 1px solid rgba(23,32,51,.14); background: rgba(255,253,247,.92); border-radius: 6px; padding: 6px 8px; font-size: 12px; cursor: pointer; }}
.legend input {{ margin: 0; accent-color: #172033; }}
.swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.axis-note {{ position: absolute; right: 14px; bottom: 12px; color: var(--muted); background: rgba(255,253,247,.88); border: 1px solid rgba(23,32,51,.12); border-radius: 6px; padding: 7px 9px; font-size: 12px; }}
.side h2 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
.side .tier {{ margin: 7px 0 16px; font-size: 13px; font-weight: 800; }}
.section {{ margin-top: 18px; padding-top: 15px; border-top: 1px solid var(--line); }}
.section h3 {{ margin: 0 0 9px; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: #445166; }}
.vector {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 5px; }}
.vec-cell {{ background: var(--panel); border: 1px solid var(--line); border-radius: 5px; padding: 7px 5px; text-align: center; min-width: 0; }}
.vec-cell b {{ display: block; font-size: 11px; color: var(--muted); }}
.vec-cell span {{ display: block; font-size: 16px; font-weight: 800; margin-top: 2px; }}
.bar-row {{ display: grid; grid-template-columns: 72px minmax(0, 1fr) 52px; align-items: center; gap: 8px; margin: 7px 0; font-size: 12px; }}
.bar-track {{ position: relative; height: 14px; border-radius: 4px; background: #ede6d8; overflow: hidden; }}
.bar-zero {{ position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: rgba(23,32,51,.35); }}
.bar {{ position: absolute; top: 0; bottom: 0; background: #b64242; }}
.bar.pos {{ left: 50%; }}
.bar.neg {{ right: 50%; }}
.repair-list {{ margin: 0; padding: 0; list-style: none; display: grid; gap: 6px; }}
.repair-list li {{ background: var(--panel); border: 1px solid var(--line); border-radius: 5px; padding: 7px 8px; font-size: 13px; }}
.note {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
.small {{ font-size: 12px; color: var(--muted); line-height: 1.4; }}
.counts {{ display: grid; grid-template-columns: 1fr auto; gap: 6px 12px; font-size: 13px; }}
.counts b {{ text-align: right; }}
@media (max-width: 920px) {{
  main {{ grid-template-columns: 1fr; }}
  .side {{ border-left: 0; border-top: 1px solid var(--line); }}
  .viz-wrap {{ height: 62vh; min-height: 430px; }}
}}
</style>
</head>
<body>
<main>
  <section class="stage">
    <h1>{title}</h1>
    <p class="subtitle">{escape(data.subtitle)}</p>
    <div class="meta">
      <div class="pill"><b id="distinct"></b> representatives</div>
      <div class="pill"><b id="raw"></b> raw cells</div>
      <div class="pill"><b id="uncertified"></b> solver-uncertified</div>
      <div class="pill">basis-dependent integer slice</div>
    </div>
    <div class="viz-wrap">
      <canvas id="plot"></canvas>
      <div class="legend" id="legend"></div>
      <div class="axis-note">2D PCA projection of signed-log anomaly vectors</div>
    </div>
  </section>
  <aside class="side">
    <h2 id="point-title">Selected assignment</h2>
    <div class="tier" id="point-tier"></div>
    <div class="vector" id="vector"></div>
    <div class="section">
      <h3>Anomaly vector</h3>
      <div id="bars"></div>
    </div>
    <div class="section">
      <h3>Repair</h3>
      <ul class="repair-list" id="repair"></ul>
      <p class="note" id="note"></p>
    </div>
    <div class="section">
      <h3>Census counts</h3>
      <div class="counts" id="counts"></div>
    </div>
    <div class="section">
      <p class="small" id="params"></p>
    </div>
  </aside>
</main>
<script>
const DATA = {payload};
const FIELDS = {json.dumps(FIELDS)};
const ANOMALIES = {json.dumps(X_ANOMALIES)};
const COLORS = {{
  "already-consistent": "#6b7280",
  "sterile-repairable": "#2b6cb0",
  "colorless-exotic-repairable": "#1f8a70",
  "colored-exotic-required": "#c77700",
  "budget-limited": "#b64242",
  "structurally-blocked": "#6b3fa0"
}};
const ORDER = {json.dumps(TIER_ORDER)};
let selected = DATA.points.find(p => p.tier === "sterile-repairable") || DATA.points[0];
let hover = null;
const enabled = new Set(ORDER);

const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const legend = document.getElementById("legend");

document.getElementById("distinct").textContent = DATA.distinct_count;
document.getElementById("raw").textContent = DATA.raw_count;
document.getElementById("uncertified").textContent = `${{DATA.uncertified_count}}/${{DATA.distinct_count}}`;
document.getElementById("params").textContent = DATA.parameters;

for (const tier of ORDER) {{
  if (!DATA.counts[tier]) continue;
  const label = document.createElement("label");
  label.innerHTML = `<input type="checkbox" checked data-tier="${{tier}}"><span class="swatch" style="background:${{COLORS[tier]}}"></span>${{tier}}`;
  legend.appendChild(label);
}}
legend.addEventListener("change", e => {{
  const tier = e.target.dataset.tier;
  if (!tier) return;
  if (e.target.checked) enabled.add(tier); else enabled.delete(tier);
  draw();
}});

function cssVar(name) {{
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}}

function resize() {{
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function project(p) {{
  const rect = canvas.getBoundingClientRect();
  const pad = Math.max(46, Math.min(rect.width, rect.height) * 0.08);
  return {{
    x: pad + (p.x + 1) * 0.5 * (rect.width - 2 * pad),
    y: pad + (1 - (p.y + 1) * 0.5) * (rect.height - 2 * pad)
  }};
}}

function draw() {{
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#fffaf0";
  ctx.fillRect(0, 0, rect.width, rect.height);
  const pad = Math.max(46, Math.min(rect.width, rect.height) * 0.08);
  ctx.strokeStyle = "rgba(23,32,51,.16)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, rect.height / 2); ctx.lineTo(rect.width - pad, rect.height / 2);
  ctx.moveTo(rect.width / 2, pad); ctx.lineTo(rect.width / 2, rect.height - pad);
  ctx.stroke();
  ctx.fillStyle = "rgba(23,32,51,.55)";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillText("PC1", rect.width - pad - 24, rect.height / 2 - 8);
  ctx.fillText("PC2", rect.width / 2 + 8, pad + 14);

  const visible = DATA.points.filter(p => enabled.has(p.tier));
  for (const p of visible) {{
    const pos = project(p);
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, p.radius, 0, Math.PI * 2);
    ctx.fillStyle = COLORS[p.tier] || "#444";
    ctx.globalAlpha = p.solver_uncertified ? 0.42 : 0.72;
    ctx.fill();
    ctx.globalAlpha = 1;
    if (!p.certified && p.tier !== "budget-limited") {{
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }}
  }}
  for (const p of [hover, selected]) {{
    if (!p || !enabled.has(p.tier)) continue;
    const pos = project(p);
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, p.radius + 5, 0, Math.PI * 2);
    ctx.strokeStyle = p === selected ? "#172033" : "rgba(23,32,51,.45)";
    ctx.lineWidth = p === selected ? 2.5 : 1.5;
    ctx.stroke();
  }}
}}

function nearestPoint(event) {{
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let best = null, bestD = Infinity;
  for (const p of DATA.points) {{
    if (!enabled.has(p.tier)) continue;
    const pos = project(p);
    const d = Math.hypot(pos.x - x, pos.y - y);
    if (d < bestD) {{ bestD = d; best = p; }}
  }}
  return bestD < 18 ? best : null;
}}

canvas.addEventListener("mousemove", e => {{
  hover = nearestPoint(e);
  canvas.style.cursor = hover ? "pointer" : "default";
  draw();
}});
canvas.addEventListener("mouseleave", () => {{ hover = null; draw(); }});
canvas.addEventListener("click", e => {{
  const p = nearestPoint(e);
  if (p) {{ selected = p; renderSide(); draw(); }}
}});

function fmtSigned(v) {{
  if (v > 0) return `+${{v}}`;
  return String(v);
}}

function renderSide() {{
  document.getElementById("point-title").textContent = `X = (${{selected.charges.join(", ")}})`;
  const tier = document.getElementById("point-tier");
  tier.textContent = selected.tier;
  tier.style.color = COLORS[selected.tier] || "#172033";
  const vector = document.getElementById("vector");
  vector.innerHTML = FIELDS.map((field, i) => `<div class="vec-cell"><b>${{field}}</b><span>${{fmtSigned(selected.charges[i])}}</span></div>`).join("");

  const maxAbs = Math.max(1, ...ANOMALIES.map(k => Math.abs(selected.anomaly_values[k])));
  const bars = document.getElementById("bars");
  bars.innerHTML = ANOMALIES.map(k => {{
    const v = selected.anomaly_values[k];
    const w = Math.min(50, Math.abs(v) / maxAbs * 50);
    const cls = v >= 0 ? "pos" : "neg";
    return `<div class="bar-row"><span>${{k}}</span><div class="bar-track"><div class="bar-zero"></div><div class="bar ${{cls}}" style="width:${{w}}%"></div></div><b>${{selected.anomalies[k]}}</b></div>`;
  }}).join("");

  const repair = document.getElementById("repair");
  repair.innerHTML = selected.added.length
    ? selected.added.map(([name, x]) => `<li>${{name}} <b>X=${{x}}</b></li>`).join("")
    : `<li>No added field in this result</li>`;
  document.getElementById("note").textContent = selected.note || "No note.";

  const counts = document.getElementById("counts");
  counts.innerHTML = ORDER.filter(t => DATA.counts[t]).map(t => `<span style="color:${{COLORS[t]}}">${{t}}</span><b>${{DATA.counts[t]}}</b>`).join("");
}}

window.addEventListener("resize", resize);
renderSide();
resize();
</script>
</body>
</html>
"""


def write_repair_atlas_visualization(out_dir: str | Path = "figures", radius: int = 2,
                                     max_added: int = 6, timeout_ms: int = 8000,
                                     workers: int = 1) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_visual_data(radius=radius, max_added=max_added,
                             timeout_ms=timeout_ms, workers=workers)
    html = render_html(data)
    path = out / "repair-census.html"
    path.write_text(html, encoding="utf-8")
    return path


if __name__ == "__main__":
    import os
    write_repair_atlas_visualization(workers=os.cpu_count() or 1)
