#!/usr/bin/env python3

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "target" / "curriculum_harness" / "learning_metrics.csv"
OUT_PATH = ROOT / "target" / "curriculum_harness" / "learning_curves.html"


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_int(value, default=0):
    if value in (None, "", "NA"):
        return default
    return int(value)


def split_key(row):
    return row.get("split", "")


def mode_key(row):
    return row.get("mode", "unknown")


def diagnostic_rows(rows):
    return [row for row in rows if split_key(row) == "diagnostic_baseline"]


def metric_rows(rows):
    return [row for row in rows if split_key(row) != "diagnostic_baseline"]


def mode_rows(rows, mode):
    return [row for row in rows if mode_key(row) == mode]


def unique_modes(rows):
    modes = sorted({mode_key(row) for row in rows if mode_key(row)})
    return [m for m in modes if m != "unknown"]


def make_series(rows, metric, include_splits):
    points = []
    for index, row in enumerate(rows, start=1):
        if split_key(row) not in include_splits:
            continue
        seq = parse_int(row.get("sequence_index"), index)
        points.append((seq, parse_int(row.get(metric), 0)))
    points.sort(key=lambda item: item[0])
    return points


def make_mode_recovery_points(rows):
    points = []
    for row in rows:
        if split_key(row) != "trained_recovery":
            continue
        points.append(
            (
                mode_key(row),
                parse_int(row.get("converged_iteration"), 0),
                row.get("id", ""),
            )
        )
    return points


def make_coherence_recovery_points(rows):
    points = []
    for row in rows:
        if split_key(row) != "trained_recovery":
            continue
        points.append(
            (
                mode_key(row),
                parse_int(row.get("converged_iteration"), 0),
                parse_int(row.get("self_consistency"), 0),
                row.get("id", ""),
            )
        )
    return points


def make_domain_recovery_series(rows, mode):
    data = {}
    filtered = [
        row
        for row in rows
        if mode_key(row) == mode and split_key(row) == "trained_recovery"
    ]
    for row in filtered:
        domain = row.get("domain", "Unknown")
        seq = parse_int(row.get("sequence_index"), 0)
        val = parse_int(row.get("converged_iteration"), 0)
        data.setdefault(domain, []).append((seq, val))
    for domain in data:
        data[domain].sort(key=lambda item: item[0])
    return data


def make_tier_recovery_series(rows, mode):
    data = {}
    filtered = [
        row
        for row in rows
        if mode_key(row) == mode and split_key(row) == "trained_recovery"
    ]

    for row in filtered:
        wobble = parse_int(row.get("wobble_strength"), 0)
        contradiction = parse_int(row.get("contradiction_strength"), 0)
        seq = parse_int(row.get("sequence_index"), 0)
        val = parse_int(row.get("converged_iteration"), 0)

        if wobble >= 48 or contradiction >= 40:
            tier = "hard"
        elif wobble >= 36 or contradiction >= 28:
            tier = "medium"
        else:
            tier = "light"

        data.setdefault(tier, []).append((seq, val))

    for tier in data:
        data[tier].sort(key=lambda item: item[0])
    return data


def baseline_budget_map(rows):
    result = {}
    for row in diagnostic_rows(rows):
        result[mode_key(row)] = parse_int(row.get("canonical_recovery_budget"), 0)
    return result


def baseline_derived_map(rows):
    result = {}
    for row in diagnostic_rows(rows):
        result[mode_key(row)] = parse_int(
            row.get("diagnostic_derived_recovery_budget_2x_median"), 0
        )
    return result


def baseline_stage_d_median_map(rows):
    result = {}
    for row in diagnostic_rows(rows):
        result[mode_key(row)] = parse_int(
            row.get("diagnostic_stage_d_recovery_median_iteration"), 0
        )
    return result


def svg_line_chart(
    points,
    width=900,
    height=240,
    stroke="#2563eb",
    baseline=None,
    baseline_label=None,
):
    if not points:
        return f'<svg width="{width}" height="{height}"></svg>'

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if baseline is not None:
        ys = ys + [baseline]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1

    def scale_x(value):
        return 40 + (value - min_x) * (width - 60) / (max_x - min_x)

    def scale_y(value):
        return height - 30 - (value - min_y) * (height - 50) / (max_y - min_y)

    point_str = " ".join(f"{scale_x(x):.1f},{scale_y(y):.1f}" for x, y in points)

    ticks = []
    for index in range(5):
        fraction = index / 4
        y_value = min_y + (max_y - min_y) * fraction
        y_pos = scale_y(y_value)
        ticks.append(
            f'<line x1="40" y1="{y_pos:.1f}" x2="{width - 20}" y2="{y_pos:.1f}" stroke="#e5e7eb" />'
            f'<text x="8" y="{y_pos + 4:.1f}" font-size="12" fill="#374151">{int(y_value)}</text>'
        )

    x_labels = []
    for x in sorted(set(xs)):
        x_pos = scale_x(x)
        x_labels.append(
            f'<text x="{x_pos:.1f}" y="{height - 8}" font-size="11" text-anchor="middle" fill="#374151">{x}</text>'
        )

    baseline_line = ""
    baseline_text = ""
    if baseline is not None:
        yb = scale_y(baseline)
        baseline_line = (
            f'<line x1="40" y1="{yb:.1f}" x2="{width - 20}" y2="{yb:.1f}" '
            f'stroke="#dc2626" stroke-dasharray="8 6" stroke-width="2" />'
        )
        label = baseline_label if baseline_label else f"baseline={baseline}"
        baseline_text = (
            f'<text x="{width - 24}" y="{yb - 6:.1f}" text-anchor="end" font-size="12" '
            f'fill="#b91c1c">{html.escape(label)}</text>'
        )

    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
      {''.join(ticks)}
      <line x1="40" y1="10" x2="40" y2="{height - 30}" stroke="#111827" />
      <line x1="40" y1="{height - 30}" x2="{width - 20}" y2="{height - 30}" stroke="#111827" />
      <polyline fill="none" stroke="{stroke}" stroke-width="3" points="{point_str}" />
      {baseline_line}
      {baseline_text}
      {''.join(x_labels)}
    </svg>
    """


def svg_scatter(points, width=900, height=260):
    if not points:
        return f'<svg width="{width}" height="{height}"></svg>'

    xs = [point[1] for point in points]
    ys = [point[2] for point in points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1

    def scale_x(value):
        return 50 + (value - min_x) * (width - 80) / (max_x - min_x)

    def scale_y(value):
        return height - 30 - (value - min_y) * (height - 60) / (max_y - min_y)

    colors = {
        "full_stack": "#2563eb",
        "no_meta": "#16a34a",
    }

    grid = []
    for index in range(5):
        frac = index / 4
        y_value = min_y + (max_y - min_y) * frac
        y_pos = scale_y(y_value)
        grid.append(
            f'<line x1="50" y1="{y_pos:.1f}" x2="{width - 20}" y2="{y_pos:.1f}" stroke="#e5e7eb" />'
            f'<text x="8" y="{y_pos + 4:.1f}" font-size="12" fill="#374151">{int(y_value)}</text>'
        )

    points_svg = []
    for mode, recovery_iter, coherence, label in points:
        cx = scale_x(recovery_iter)
        cy = scale_y(coherence)
        color = colors.get(mode, "#6b7280")
        tooltip = html.escape(f"{mode} | {label} | recovery={recovery_iter} | coherence={coherence}")
        points_svg.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}"><title>{tooltip}</title></circle>'
        )

    x_ticks = []
    for value in sorted(set(xs)):
        xpos = scale_x(value)
        x_ticks.append(
            f'<text x="{xpos:.1f}" y="{height - 8}" font-size="11" text-anchor="middle" fill="#374151">{value}</text>'
        )

    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
      {''.join(grid)}
      <line x1="50" y1="10" x2="50" y2="{height - 30}" stroke="#111827" />
      <line x1="50" y1="{height - 30}" x2="{width - 20}" y2="{height - 30}" stroke="#111827" />
      {''.join(points_svg)}
      {''.join(x_ticks)}
      <text x="{width / 2:.1f}" y="{height - 2}" text-anchor="middle" font-size="12" fill="#374151">Recovery iterations</text>
      <text x="14" y="22" font-size="12" fill="#374151">Self-coherence</text>
    </svg>
    """


def svg_mode_recovery(points, baseline_by_mode, width=900, height=260):
    if not points:
        return f'<svg width="{width}" height="{height}"></svg>'

    mode_order = ["full_stack", "no_meta"]
    mode_positions = {m: i + 1 for i, m in enumerate(mode_order)}

    ys = [p[1] for p in points]
    baseline_vals = [v for v in baseline_by_mode.values() if v > 0]
    ys = ys + baseline_vals
    min_y, max_y = min(ys), max(ys)
    if max_y == min_y:
        max_y += 1

    def scale_x(mode):
        pos = mode_positions.get(mode, 1)
        return 140 + (pos - 1) * (width - 280)

    def scale_y(value):
        return height - 30 - (value - min_y) * (height - 60) / (max_y - min_y)

    grid = []
    for idx in range(5):
        frac = idx / 4
        y_value = min_y + (max_y - min_y) * frac
        y_pos = scale_y(y_value)
        grid.append(
            f'<line x1="60" y1="{y_pos:.1f}" x2="{width - 30}" y2="{y_pos:.1f}" stroke="#e5e7eb" />'
            f'<text x="14" y="{y_pos + 4:.1f}" font-size="12" fill="#374151">{int(y_value)}</text>'
        )

    points_svg = []
    colors = {"full_stack": "#2563eb", "no_meta": "#16a34a"}
    for mode, value, label in points:
        cx = scale_x(mode)
        cy = scale_y(value)
        color = colors.get(mode, "#6b7280")
        tooltip = html.escape(f"{mode} | {label} | recovery={value}")
        points_svg.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}"><title>{tooltip}</title></circle>'
        )

    baseline_lines = []
    for mode in mode_order:
        baseline = baseline_by_mode.get(mode, 0)
        if baseline <= 0:
            continue
        cx = scale_x(mode)
        y = scale_y(baseline)
        baseline_lines.append(
            f'<line x1="{cx - 70:.1f}" y1="{y:.1f}" x2="{cx + 70:.1f}" y2="{y:.1f}" '
            f'stroke="#dc2626" stroke-dasharray="8 6" stroke-width="2" />'
            f'<text x="{cx:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="11" fill="#b91c1c">budget={baseline}</text>'
        )

    x_labels = []
    for mode in mode_order:
        x_labels.append(
            f'<text x="{scale_x(mode):.1f}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#374151">{mode}</text>'
        )

    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
      {''.join(grid)}
      <line x1="60" y1="10" x2="60" y2="{height - 30}" stroke="#111827" />
      <line x1="60" y1="{height - 30}" x2="{width - 30}" y2="{height - 30}" stroke="#111827" />
      {''.join(baseline_lines)}
      {''.join(points_svg)}
      {''.join(x_labels)}
    </svg>
    """


def chart_block(title, subtitle, svg):
    return f"""
    <section class="card">
      <h2>{html.escape(title)}</h2>
      <p>{html.escape(subtitle)}</p>
      {svg}
    </section>
    """


def table_rows(rows):
    return "".join(
        f"<tr><td>{html.escape(row.get('split', ''))}</td><td>{html.escape(row.get('mode', ''))}</td><td>{html.escape(row.get('sequence_index', ''))}</td><td>{html.escape(row.get('domain', ''))}</td><td>{html.escape(row.get('id', ''))}</td><td>{html.escape(row.get('meta_revision_count', ''))}</td><td>{html.escape(row.get('converged_iteration', ''))}</td><td>{html.escape(row.get('self_consistency', ''))}</td></tr>"
        for row in rows
        if split_key(row) != "diagnostic_baseline"
    )


def summary_block(rows):
    modes = unique_modes(rows)
    stage_medians = baseline_stage_d_median_map(rows)
    derived_budgets = baseline_derived_map(rows)
    canonical_budgets = baseline_budget_map(rows)

    lines = []
    for mode in modes:
        lines.append(
            f"{mode}: stage_d_median={stage_medians.get(mode, 0)}, derived_2x={derived_budgets.get(mode, 0)}, canonical_budget={canonical_budgets.get(mode, 0)}"
        )
    return "<br/>".join(html.escape(line) for line in lines)


def main():
    rows = load_rows(CSV_PATH)
    data_rows = metric_rows(rows)

    modes = unique_modes(rows)
    domains = sorted({row.get("domain", "Unknown") for row in data_rows})

    training_plus_trained = {"training", "trained_holdout", "trained_recovery"}

    anchors_full = make_series(mode_rows(data_rows, "full_stack"), "active_anchors", training_plus_trained)
    anchors_no_meta = make_series(mode_rows(data_rows, "no_meta"), "active_anchors", training_plus_trained)

    recovery_mode_points = make_mode_recovery_points(data_rows)
    coherence_recovery_points = make_coherence_recovery_points(data_rows)

    baseline_by_mode = baseline_budget_map(rows)
    baseline_full = baseline_by_mode.get("full_stack")

    domain_sections = []
    for mode in modes:
        by_domain = make_domain_recovery_series(data_rows, mode)
        for domain, points in sorted(by_domain.items()):
            domain_sections.append(
                chart_block(
                    f"Recovery by Domain ({mode} / {domain})",
                    "Trained recovery convergence across sequence index",
                    svg_line_chart(
                        points,
                        stroke="#0f766e" if mode == "full_stack" else "#a16207",
                        baseline=baseline_by_mode.get(mode),
                        baseline_label=f"budget={baseline_by_mode.get(mode, 0)}",
                    ),
                )
            )

    tier_sections = []
    for mode in modes:
        by_tier = make_tier_recovery_series(data_rows, mode)
        colors = {"hard": "#dc2626", "medium": "#d97706", "light": "#2563eb"}
        for tier, points in sorted(by_tier.items()):
            tier_sections.append(
                chart_block(
                    f"Recovery by Difficulty Tier ({mode} / {tier})",
                    "Tier split from holdout wobble+contradiction strengths",
                    svg_line_chart(
                        points,
                        stroke=colors.get(tier, "#374151"),
                        baseline=baseline_by_mode.get(mode),
                        baseline_label=f"budget={baseline_by_mode.get(mode, 0)}",
                    ),
                )
            )

    html_doc = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>RUGC Curriculum Learning Curves</title>
      <style>
        body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f8fafc; color: #111827; }}
        h1 {{ margin-bottom: 8px; }}
        p.meta {{ margin-top: 0; color: #4b5563; }}
        .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
        .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 13px; }}
        th {{ background: #f3f4f6; }}
      </style>
    </head>
    <body>
      <h1>RUGC Curriculum Learning Curves</h1>
      <p class="meta">Source: {html.escape(str(CSV_PATH))}</p>
      <p class="meta">Modes: {html.escape(', '.join(modes))}</p>
      <p class="meta">Domains: {html.escape(', '.join(domains))}</p>
      <p class="meta">Diagnostic baselines: {summary_block(rows)}</p>
      <div class="grid">
        {chart_block('Anchor Growth (full_stack)', 'Training plus trained holdout/recovery episodes', svg_line_chart(anchors_full, stroke='#2563eb'))}
        {chart_block('Anchor Growth (no_meta)', 'Training plus trained holdout/recovery episodes', svg_line_chart(anchors_no_meta, stroke='#16a34a'))}
        {chart_block('Recovery Iterations vs Mode', 'Trained recovery episodes grouped by mode with canonical budget line', svg_mode_recovery(recovery_mode_points, baseline_by_mode))}
        {chart_block('Self-Coherence vs Recovery Iterations', 'Scatter for trained recovery episodes; color encodes mode', svg_scatter(coherence_recovery_points))}
        {chart_block('Full Stack Recovery Curve', 'Trained recovery convergence with baseline line', svg_line_chart(make_series(mode_rows(data_rows, 'full_stack'), 'converged_iteration', {'trained_recovery'}), stroke='#2563eb', baseline=baseline_full, baseline_label=f'budget={baseline_full or 0}'))}
        {chart_block('No Meta Recovery Curve', 'Trained recovery convergence with baseline line', svg_line_chart(make_series(mode_rows(data_rows, 'no_meta'), 'converged_iteration', {'trained_recovery'}), stroke='#16a34a', baseline=baseline_by_mode.get('no_meta'), baseline_label=f"budget={baseline_by_mode.get('no_meta', 0)}"))}
        {''.join(domain_sections)}
        {''.join(tier_sections)}
      </div>
      <table>
        <thead>
          <tr>
            <th>Split</th><th>Mode</th><th>Seq</th><th>Domain</th><th>ID</th><th>Meta Revisions</th><th>Converged Iter</th><th>Self Consistency</th>
          </tr>
        </thead>
        <tbody>
          {table_rows(rows)}
        </tbody>
      </table>
    </body>
    </html>
    """

    OUT_PATH.write_text(html_doc, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
