"""VCG mechanism for purchasing the cheapest directed path from A to F.

The network and costs are taken from Figure 7.3. Each directed edge is an
agent whose true cost is the cost shown in the figure. The script saves the
truthful allocation and VCG payments, then evaluates every report in
{0, 0.5, ..., 10} for one edge on the selected path (AB) and one edge off the
selected path (BD).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Iterable


Edge = tuple[str, str]


@dataclass(frozen=True)
class PathResult:
    path: tuple[str, ...]
    cost: float

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(zip(self.path, self.path[1:]))


# Figure 7.3: directed edges and their private true costs.
TRUE_COSTS: dict[Edge, float] = {
    ("A", "B"): 3.0,
    ("A", "C"): 2.0,
    ("B", "D"): 2.0,
    ("B", "E"): 1.0,
    ("C", "E"): 3.0,
    ("C", "F"): 5.0,
    ("D", "F"): 2.0,
    ("E", "F"): 1.0,
}


def shortest_path(costs: dict[Edge, float], source: str = "A", target: str = "F") -> PathResult:
    """Return a minimum-cost source-to-target path using Dijkstra's algorithm."""
    distances: dict[str, float] = {source: 0.0}
    paths: dict[str, tuple[str, ...]] = {source: (source,)}
    unvisited = set({source})

    while unvisited:
        node = min(unvisited, key=lambda item: (distances.get(item, inf), item))
        unvisited.remove(node)
        if node == target:
            return PathResult(paths[node], distances[node])

        for (start, end), edge_cost in costs.items():
            if start != node:
                continue
            candidate_cost = distances[node] + edge_cost
            candidate_path = paths[node] + (end,)
            old_cost = distances.get(end, inf)
            if candidate_cost < old_cost or (
                candidate_cost == old_cost and candidate_path < paths.get(end, ("~",))
            ):
                distances[end] = candidate_cost
                paths[end] = candidate_path
                unvisited.add(end)

    raise ValueError(f"No path exists from {source} to {target}")


def vcg_payments(costs: dict[Edge, float], source: str = "A", target: str = "F") -> tuple[PathResult, dict[Edge, float]]:
    """Compute the shortest path and truthful VCG payments.

    For a winning edge e, payment(e) is

        reported_cost(e) + (best cost without e - winning path cost).

    Losing edges receive zero.  This is the cost-minimization version of the
    VCG payment rule.
    """
    allocation = shortest_path(costs, source, target)
    payments = {edge: 0.0 for edge in costs}

    for edge in allocation.edges:
        without_edge = {candidate: value for candidate, value in costs.items() if candidate != edge}
        alternative = shortest_path(without_edge, source, target)
        payments[edge] = costs[edge] + alternative.cost - allocation.cost

    return allocation, payments


def utility(
    reported_costs: dict[Edge, float],
    true_costs: dict[Edge, float],
    agent: Edge,
) -> tuple[PathResult, float, float]:
    """Return allocation, payment, and utility for one agent.

    Utility is payment minus the agent's true cost when its edge is selected;
    a losing agent has utility zero.
    """
    allocation, payments = vcg_payments(reported_costs)
    if agent not in allocation.edges:
        return allocation, payments[agent], 0.0
    return allocation, payments[agent], payments[agent] - true_costs[agent]


def reports_0_to_10(step: float = 0.5) -> Iterable[float]:
    """Generate 0, 0.5, ..., 10 without floating-point drift."""
    for index in range(21):
        yield index * step


def utility_curve(agent: Edge) -> list[tuple[float, float]]:
    """Return (reported cost, utility) points for one agent."""
    curve = []
    for report in reports_0_to_10():
        reported = dict(TRUE_COSTS)
        reported[agent] = report
        _, _, agent_utility = utility(reported, TRUE_COSTS, agent)
        curve.append((report, agent_utility))
    return curve


def save_utility_table(output_path: Path | None = None) -> Path:
    """Save all misreport results as a CSV table."""
    output_path = output_path or Path(__file__).with_name("vcg_utility_table.csv")
    rows: list[dict[str, object]] = []

    for agent, label in ((("A", "B"), "AB (on path)"), (("B", "D"), "BD (off path)")):
        for report in reports_0_to_10():
            reported = dict(TRUE_COSTS)
            reported[agent] = report
            allocation, payment, agent_utility = utility(reported, TRUE_COSTS, agent)
            rows.append(
                {
                    "agent": label,
                    "reported_cost": report,
                    "selected_path": "-".join(allocation.path),
                    "payment": payment,
                    "true_cost": TRUE_COSTS[agent],
                    "utility": agent_utility,
                }
            )

    with output_path.open("w", newline="", encoding="utf-8") as table_file:
        writer = csv.DictWriter(table_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def save_truthful_result(output_path: Path | None = None) -> Path:
    """Save the truthful path and every edge's VCG payment as a CSV table."""
    output_path = output_path or Path(__file__).with_name("vcg_truthful_payments.csv")
    allocation, payments = vcg_payments(TRUE_COSTS)
    with output_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.writer(result_file)
        writer.writerow(["chosen_path", "path_cost", "edge", "true_cost", "selected", "payment"])
        for edge, true_cost in TRUE_COSTS.items():
            writer.writerow(
                [
                    "-".join(allocation.path),
                    allocation.cost,
                    f"{edge[0]}{edge[1]}",
                    true_cost,
                    edge in allocation.edges,
                    payments[edge],
                ]
            )
    return output_path


def save_utility_graph(output_path: Path | None = None) -> Path:
    """Save a self-contained SVG graph of the two requested utility curves."""
    output_path = output_path or Path(__file__).with_name("vcg_utility_graph.svg")
    width, height = 900, 560
    left, right, top, bottom = 90, 40, 70, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min, y_max = -2.5, 1.5

    def x_pixel(report: float) -> float:
        return left + report / 10 * plot_width

    def y_pixel(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    curves = {
        "AB (on path)": (utility_curve(("A", "B")), "#1565c0"),
        "BD (off path)": (utility_curve(("B", "D")), "#c62828"),
    }
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222} .tick{font-size:12px} .label{font-size:15px} .title{font-size:20px;font-weight:bold}</style>',
        f'<text x="{width / 2:g}" y="30" text-anchor="middle" class="title">VCG utility under cost misreports</text>',
    ]

    # Grid and axes.
    for report in range(11):
        x = x_pixel(report)
        svg.append(f'<line x1="{x:g}" y1="{top}" x2="{x:g}" y2="{top + plot_height}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{x:g}" y="{top + plot_height + 24}" text-anchor="middle" class="tick">{report}</text>')
    for value in range(-2, 2):
        y = y_pixel(value)
        stroke = "#444" if value == 0 else "#e5e7eb"
        svg.append(f'<line x1="{left}" y1="{y:g}" x2="{left + plot_width}" y2="{y:g}" stroke="{stroke}"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:g}" text-anchor="end" class="tick">{value}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>')
    svg.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222"/>')
    svg.append(f'<text x="{left + plot_width / 2:g}" y="{height - 18}" text-anchor="middle" class="label">Reported cost</text>')
    svg.append(f'<text x="20" y="{top + plot_height / 2:g}" text-anchor="middle" transform="rotate(-90 20 {top + plot_height / 2:g})" class="label">Utility</text>')

    for label, (curve, color) in curves.items():
        points = " ".join(f"{x_pixel(report):g},{y_pixel(value):g}" for report, value in curve)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for report, value in curve:
            svg.append(f'<circle cx="{x_pixel(report):g}" cy="{y_pixel(value):g}" r="3" fill="{color}"/>')

    # Mark truthful costs on the x-axis.
    for report, label, color in ((3, "AB true=3", "#1565c0"), (2, "BD true=2", "#c62828")):
        x = x_pixel(report)
        svg.append(f'<line x1="{x:g}" y1="{top}" x2="{x:g}" y2="{top + plot_height}" stroke="{color}" stroke-dasharray="5,5" opacity="0.65"/>')
        svg.append(f'<text x="{x + 6:g}" y="{top + 16:g}" class="tick" fill="{color}">{label}</text>')

    legend_x, legend_y = width - 230, 48
    for index, (label, (_, color)) in enumerate(curves.items()):
        y = legend_y + index * 22
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 25}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 32}" y="{y + 4}" class="tick">{label}</text>')

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def main() -> None:
    # AB is on A-B-E-F; BD is not on the truthful path.
    save_truthful_result()
    save_utility_graph()
    save_utility_table()


if __name__ == "__main__":
    main()
