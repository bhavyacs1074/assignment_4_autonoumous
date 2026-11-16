"""First-price sealed-bid auction with n bidders and one deviating bidder.

Setup (Part 2 of the assignment)
--------------------------------
* n bidders, every valuation is drawn independently from U[0, 1].
* Bidders 2..n play the symmetric Bayes-Nash equilibrium bid
  beta(v) = (n - 1) / n * v.
* Bidder 1 deviates and bids k * v1 for k in the grid 0.1, 0.2, ..., 2.0.
* The highest bid wins and the winner pays its own bid (first price).
  Ties (probability zero) are broken uniformly at random.

For every n in {2, 3, 4} and every k the script estimates

* the expected utility of bidder 1, E[(v1 - k v1) * 1{bidder 1 wins}], and
* the fraction of auctions in which the winner could have paid less and still
  won, i.e. the winning bid is strictly above the second-highest bid.  The
  average gap (winning bid - second-highest bid) is the "money left on the
  table".

Methodology
-----------
* Common random numbers: the same valuation draws are reused for every k, so
  differences between grid points reflect k rather than sampling noise.
* Every Monte Carlo estimate is reported with its standard error and compared
  to the closed-form expected utility derived below.  ``verify`` fails if any
  estimate is more than 4 standard errors from theory.

Closed form
-----------
Let c = k n / (n - 1).  Bidder 1 with value v wins iff the highest rival value
M = max(v2..vn) satisfies M < c v, so P(win | v) = min(1, c v)^(n-1) and

    E[U | v, k] = v (1 - k) min(1, c v)^(n-1).

Integrating over v ~ U[0, 1]:

    c <= 1:  E[U | k] = (1 - k) c^(n-1) / (n + 1)
    c >  1:  E[U | k] = (1 - k) (1/2 - (n - 1) / (2 (n + 1) c^2))

On c <= 1 the utility is proportional to (1 - k) k^(n-1), which is maximised at
k* = (n - 1) / n, the equilibrium.  There E[U] = 1 / (n (n + 1)).  The
conditional utility E[U | v, k] has the same maximiser for every v, so no
fixed value of v makes deviating profitable.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
BIDDER_COUNTS = (2, 3, 4)
K_GRID = tuple(round(0.1 * i, 1) for i in range(1, 21))  # 0.1 ... 2.0
FIXED_VALUES = (0.2, 0.5, 0.8)
NUM_AUCTIONS = 200_000
SEED = 2026


# --------------------------------------------------------------------------
# Theory
# --------------------------------------------------------------------------
def equilibrium_fraction(n: int) -> float:
    return (n - 1) / n


def theoretical_utility(n: int, k: float) -> float:
    """E[U1] for bid k*v1 when v1 ~ U[0, 1] and the rivals play equilibrium."""
    c = k / equilibrium_fraction(n)
    if c <= 1.0:
        return (1.0 - k) * c ** (n - 1) / (n + 1)
    return (1.0 - k) * (0.5 - (n - 1) / (2.0 * (n + 1) * c * c))


def theoretical_conditional_utility(n: int, k: float, v: float) -> float:
    """E[U1 | v1 = v] for bid k*v when the rivals play equilibrium."""
    c = k / equilibrium_fraction(n)
    return v * (1.0 - k) * min(1.0, c * v) ** (n - 1)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Draw:
    """One auction's random inputs, shared across all k (common random numbers)."""

    v1: float
    top_rival_bid: float
    second_rival_bid: float  # 0 when there is only one rival
    tie_coin: float


def draw_auctions(n: int, count: int, rng: random.Random) -> list[Draw]:
    shade = equilibrium_fraction(n)
    draws = []
    for _ in range(count):
        v1 = rng.random()
        rival_bids = sorted((shade * rng.random() for _ in range(n - 1)), reverse=True)
        second = rival_bids[1] if len(rival_bids) > 1 else 0.0
        draws.append(Draw(v1, rival_bids[0], second, rng.random()))
    return draws


@dataclass(frozen=True)
class Estimate:
    n: int
    k: float
    mean_utility: float
    utility_se: float
    theory_utility: float
    win_rate: float
    could_pay_less: float
    mean_overpayment: float

    @property
    def z_score(self) -> float:
        if self.utility_se == 0.0:
            return 0.0 if math.isclose(self.mean_utility, self.theory_utility, abs_tol=1e-9) else math.inf
        return (self.mean_utility - self.theory_utility) / self.utility_se


def simulate_k(n: int, k: float, draws: list[Draw], fixed_v: float | None = None) -> Estimate:
    """Run every auction in ``draws`` with bidder 1 bidding k*v1.

    If ``fixed_v`` is given, bidder 1's value is set to it instead of v1.
    """
    total = total_sq = 0.0
    wins = could_pay_less = 0
    overpayment = 0.0

    for d in draws:
        v1 = d.v1 if fixed_v is None else fixed_v
        b1 = k * v1
        rival = d.top_rival_bid
        if b1 > rival or (b1 == rival and d.tie_coin < 0.5):
            bidder1_wins = True
        else:
            bidder1_wins = False

        if bidder1_wins:
            u = v1 - b1
            winning_bid, second_bid = b1, rival
            wins += 1
        else:
            u = 0.0
            winning_bid, second_bid = rival, max(b1, d.second_rival_bid)

        total += u
        total_sq += u * u
        gap = winning_bid - second_bid
        if gap > 0.0:
            could_pay_less += 1
        overpayment += gap

    m = len(draws)
    mean = total / m
    variance = max(total_sq / m - mean * mean, 0.0) * m / (m - 1)
    theory = (
        theoretical_utility(n, k)
        if fixed_v is None
        else theoretical_conditional_utility(n, k, fixed_v)
    )
    return Estimate(
        n=n,
        k=k,
        mean_utility=mean,
        utility_se=math.sqrt(variance / m),
        theory_utility=theory,
        win_rate=wins / m,
        could_pay_less=could_pay_less / m,
        mean_overpayment=overpayment / m,
    )


def run_experiments(
    num_auctions: int = NUM_AUCTIONS, seed: int = SEED
) -> tuple[dict[int, list[Estimate]], dict[tuple[int, float], list[Estimate]]]:
    rng = random.Random(seed)
    uniform_v: dict[int, list[Estimate]] = {}
    fixed: dict[tuple[int, float], list[Estimate]] = {}
    for n in BIDDER_COUNTS:
        draws = draw_auctions(n, num_auctions, rng)
        uniform_v[n] = [simulate_k(n, k, draws) for k in K_GRID]
        for v in FIXED_VALUES:
            fixed[(n, v)] = [simulate_k(n, k, draws, fixed_v=v) for k in K_GRID]
    return uniform_v, fixed


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def check_closed_form() -> None:
    """Check the closed form against direct numerical integration over v."""
    steps = 20_000
    for n in BIDDER_COUNTS:
        for k in K_GRID + (equilibrium_fraction(n),):
            midpoint = sum(
                theoretical_conditional_utility(n, k, (i + 0.5) / steps) for i in range(steps)
            ) / steps
            assert math.isclose(midpoint, theoretical_utility(n, k), abs_tol=1e-6), (n, k)
        assert math.isclose(
            theoretical_utility(n, equilibrium_fraction(n)), 1 / (n * (n + 1)), rel_tol=1e-12
        )


def verify(uniform_v: dict[int, list[Estimate]], fixed: dict[tuple[int, float], list[Estimate]]) -> float:
    """Assert all Monte Carlo estimates agree with theory; return the max |z|."""
    check_closed_form()
    worst = 0.0
    for estimates in list(uniform_v.values()) + list(fixed.values()):
        for e in estimates:
            worst = max(worst, abs(e.z_score))
            assert abs(e.z_score) < 4.0, f"n={e.n} k={e.k}: z={e.z_score:.2f}"
            assert 0.0 <= e.could_pay_less <= 1.0
    for n, estimates in uniform_v.items():
        # Over-bidding (k > 1) wins but loses money; k = 1 gives exactly zero.
        for e in estimates:
            if e.k > 1.0:
                assert e.mean_utility < 0.0
            if e.k == 1.0:
                assert e.mean_utility == 0.0
        # The best grid point must be the one closest to the equilibrium k*.
        best = max(estimates, key=lambda e: e.theory_utility)
        k_star = equilibrium_fraction(n)
        assert abs(best.k - k_star) <= 0.05 + 1e-9, (n, best.k)
    return worst


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def save_csv(uniform_v: dict[int, list[Estimate]], fixed: dict[tuple[int, float], list[Estimate]]) -> None:
    fields = [
        "n", "k", "bid_rule", "mc_expected_utility", "std_error", "theory_expected_utility",
        "z_score", "bidder1_win_rate", "frac_winner_could_pay_less", "mean_money_left_on_table",
    ]
    with (OUTPUT_DIR / "first_price_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for n, estimates in uniform_v.items():
            for e in estimates:
                writer.writerow([
                    n, e.k, f"{e.k:g}*v", f"{e.mean_utility:.6f}", f"{e.utility_se:.6f}",
                    f"{e.theory_utility:.6f}", f"{e.z_score:.2f}", f"{e.win_rate:.4f}",
                    f"{e.could_pay_less:.4f}", f"{e.mean_overpayment:.6f}",
                ])

    with (OUTPUT_DIR / "first_price_fixed_value_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "v1", "k", "mc_expected_utility", "std_error", "theory_expected_utility", "z_score"])
        for (n, v), estimates in fixed.items():
            for e in estimates:
                writer.writerow([
                    n, v, e.k, f"{e.mean_utility:.6f}", f"{e.utility_se:.6f}",
                    f"{e.theory_utility:.6f}", f"{e.z_score:.2f}",
                ])


COLORS = {2: "#1565c0", 3: "#c62828", 4: "#2e7d32"}


def svg_line_chart(
    path: Path,
    title: str,
    y_label: str,
    series: dict[int, list[tuple[float, float]]],
    y_range: tuple[float, float],
    y_ticks: list[float],
    theory: dict[int, list[tuple[float, float]]] | None = None,
    markers: dict[int, float] | None = None,
    legend_left: bool = False,
) -> None:
    """Self-contained SVG line chart (no plotting library required)."""
    width, height = 900, 560
    left, right, top, bottom = 90, 40, 70, 80
    pw, ph = width - left - right, height - top - bottom
    x_min, x_max = 0.0, 2.0
    y_min, y_max = y_range

    def xp(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * pw

    def yp(y: float) -> float:
        y = min(max(y, y_min), y_max)
        return top + (y_max - y) / (y_max - y_min) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222} .tick{font-size:12px} '
        '.label{font-size:15px} .title{font-size:20px;font-weight:bold}</style>',
        f'<text x="{width / 2:g}" y="30" text-anchor="middle" class="title">{title}</text>',
    ]
    for i in range(11):
        x = i * 0.2
        out.append(f'<line x1="{xp(x):.1f}" y1="{top}" x2="{xp(x):.1f}" y2="{top + ph}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{xp(x):.1f}" y="{top + ph + 24}" text-anchor="middle" class="tick">{x:.1f}</text>')
    for y in y_ticks:
        stroke = "#444" if y == 0 else "#e5e7eb"
        out.append(f'<line x1="{left}" y1="{yp(y):.1f}" x2="{left + pw}" y2="{yp(y):.1f}" stroke="{stroke}"/>')
        out.append(f'<text x="{left - 12}" y="{yp(y) + 4:.1f}" text-anchor="end" class="tick">{y:g}</text>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + ph}" stroke="#222"/>')
    out.append(f'<line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}" stroke="#222"/>')
    out.append(f'<text x="{left + pw / 2:g}" y="{height - 18}" text-anchor="middle" class="label">'
               'k (bidder 1 bids k·v)</text>')
    out.append(f'<text x="22" y="{top + ph / 2:g}" text-anchor="middle" class="label" '
               f'transform="rotate(-90 22 {top + ph / 2:g})">{y_label}</text>')

    for n, x in (markers or {}).items():
        out.append(f'<line x1="{xp(x):.1f}" y1="{top}" x2="{xp(x):.1f}" y2="{top + ph}" '
                   f'stroke="{COLORS[n]}" stroke-dasharray="5,5" opacity="0.6"/>')
    for n, points in (theory or {}).items():
        pts = " ".join(f"{xp(x):.1f},{yp(y):.1f}" for x, y in points)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[n]}" stroke-width="1.5" '
                   'stroke-dasharray="2,3" opacity="0.9"/>')
    for n, points in series.items():
        pts = " ".join(f"{xp(x):.1f},{yp(y):.1f}" for x, y in points)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[n]}" stroke-width="2.5"/>')
        for x, y in points:
            out.append(f'<circle cx="{xp(x):.1f}" cy="{yp(y):.1f}" r="3.5" fill="{COLORS[n]}"/>')

    lx, ly = (left + 20 if legend_left else width - 250), top + 18
    for i, n in enumerate(series):
        y = ly + i * 22
        label = f"n = {n}" + (f" (k* = {markers[n]:.3f})" if markers else "")
        out.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 25}" y2="{y}" stroke="{COLORS[n]}" stroke-width="3"/>')
        out.append(f'<text x="{lx + 32}" y="{y + 4}" class="tick">{label}</text>')
    if theory:
        y = ly + len(series) * 22
        out.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 25}" y2="{y}" stroke="#444" stroke-dasharray="2,3"/>')
        out.append(f'<text x="{lx + 32}" y="{y + 4}" class="tick">closed form</text>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


def save_charts(uniform_v: dict[int, list[Estimate]]) -> None:
    fine_k = [i / 200 for i in range(1, 401)]
    svg_line_chart(
        OUTPUT_DIR / "first_price_expected_utility.svg",
        "First-price auction: bidder 1 expected utility vs bid fraction k",
        "Expected utility of bidder 1",
        {n: [(e.k, e.mean_utility) for e in est] for n, est in uniform_v.items()},
        y_range=(-0.50, 0.20),
        y_ticks=[-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2],
        theory={n: [(k, theoretical_utility(n, k)) for k in fine_k] for n in uniform_v},
        markers={n: equilibrium_fraction(n) for n in uniform_v},
    )
    svg_line_chart(
        OUTPUT_DIR / "first_price_could_pay_less.svg",
        "Fraction of auctions where the winner could have paid less and still won",
        "Fraction of auctions",
        {n: [(e.k, e.could_pay_less) for e in est] for n, est in uniform_v.items()},
        y_range=(0.9, 1.0),
        y_ticks=[0.9, 0.92, 0.94, 0.96, 0.98, 1.0],
    )
    svg_line_chart(
        OUTPUT_DIR / "first_price_money_left_on_table.svg",
        "Average money left on the table (winning bid - second-highest bid)",
        "Mean overpayment by the winner",
        {n: [(e.k, e.mean_overpayment) for e in est] for n, est in uniform_v.items()},
        y_range=(0.0, 0.8),
        y_ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        legend_left=True,
    )


def print_summary(uniform_v: dict[int, list[Estimate]], fixed: dict[tuple[int, float], list[Estimate]], worst_z: float) -> None:
    for n, estimates in uniform_v.items():
        k_star = equilibrium_fraction(n)
        print(f"\n=== n = {n}  (equilibrium k* = {k_star:.4f}, E[U] at k* = {1 / (n * (n + 1)):.4f}) ===")
        print(f"{'k':>4} {'MC E[U]':>9} {'SE':>8} {'theory':>9} {'z':>6} {'win%':>7} {'pay-less%':>10} {'left':>7}")
        for e in estimates:
            print(f"{e.k:4.1f} {e.mean_utility:9.5f} {e.utility_se:8.5f} {e.theory_utility:9.5f} "
                  f"{e.z_score:6.2f} {100 * e.win_rate:6.2f}% {100 * e.could_pay_less:9.3f}% {e.mean_overpayment:7.4f}")
        best = max(estimates, key=lambda e: e.mean_utility)
        print(f"best grid k (simulated) = {best.k:.1f}, E[U] = {best.mean_utility:.5f}")
        for v in FIXED_VALUES:
            best_v = max(fixed[(n, v)], key=lambda e: e.mean_utility)
            print(f"  fixed v1 = {v}: best grid k = {best_v.k:.1f}, E[U|v] = {best_v.mean_utility:.5f}")
    print(f"\nVerification passed: all estimates within 4 SE of theory (max |z| = {worst_z:.2f}).")


def main() -> None:
    uniform_v, fixed = run_experiments()
    worst_z = verify(uniform_v, fixed)
    save_csv(uniform_v, fixed)
    save_charts(uniform_v)
    print_summary(uniform_v, fixed, worst_z)


if __name__ == "__main__":
    main()
