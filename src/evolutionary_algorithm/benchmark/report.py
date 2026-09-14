"""Text rendering of benchmark results.

The report is a plain ASCII table so it can be printed, redirected and
diffed — reproducible runs produce byte-identical reports.
"""

from __future__ import annotations

from evolutionary_algorithm.benchmark.model import BenchmarkResult


def render_report(result: BenchmarkResult) -> str:
    """Render a benchmark result as an ASCII comparison table.

    Columns:
    - Solver: the configuration name
    - Best: the best fitness the configuration reached
    - % GT: best fitness relative to the scenario's ground truth (if any)
    - % Best: best fitness relative to the best achieved by any config
    - one column per target: rounds needed to reach it ("-" = never,
      "n/a" = target could not be resolved)
    """
    scenario = result.scenario
    ground_truth = scenario.ground_truth
    best_achieved = result.best_achieved()

    header_titles = ["Solver", "Best", "% GT", "% Best"]
    target_labels = [t.label for t in scenario.targets]
    header_titles.extend(target_labels)

    # Collect row cell strings first so column widths match
    rows: list[list[str]] = []
    for run in result.runs:
        best = run.best if run.best is not None else float("-inf")
        pct_gt = f"{best / ground_truth:.1%}" if ground_truth else "-"
        pct_best = f"{best / best_achieved:.1%}" if best_achieved else "-"

        cells = [run.config_name, f"{best:g}", pct_gt, pct_best]
        for target in scenario.targets:
            if target.resolve(ground_truth, best_achieved) is None:
                cells.append("n/a")
            else:
                rounds = run.milestones.get(target.label)
                cells.append("-" if rounds is None else f"{rounds}")
        rows.append(cells)

    widths = [
        max(
            len(header_titles[col]),
            *(len(row[col]) for row in rows),
        )
        for col in range(len(header_titles))
    ]

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(
            cells[i].rjust(widths[i]) if i > 0 else cells[i].ljust(widths[i])
            for i in range(len(cells))
        )

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"Benchmark result – scenario: {scenario.name}")
    gt_text = f"{ground_truth:g}" if ground_truth is not None else "n/a"
    ba_text = f"{best_achieved:g}" if best_achieved is not None else "n/a"
    lines.append(
        f"Rounds per run: {scenario.iterations} | "
        f"ground truth: {gt_text} | best achieved: {ba_text}"
    )
    lines.append("=" * 78)
    lines.append(fmt_row(header_titles))
    lines.append("-" * 78)
    for row in rows:
        lines.append(fmt_row(row))
    lines.append("=" * 78)

    return "\n".join(lines)


__all__ = ["render_report"]
