"""
Shared building blocks for the second-price sealed-bid auction assignment.

Used by part_a.py, part_b.py and part_c.py.
"""
import csv

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

VALUES = [0.2, 0.5, 0.8]                 # bidder 1's possible true values
STRATEGIES = ["truthful", "half", "random"]   # rival strategies (a), (b), (c)
BIDS = np.arange(101) / 100.0            # 0.00, 0.01, ..., 1.00
N_AUCTIONS = 100_000                     # auctions simulated per (v, strategy)
SEED = 42                                # fixed seed -> reproducible results
TOL = 1e-12                              # float tolerance for "equal utility"
def rival_bids(strategy, n, rng):
    """Draw n bids for ONE rival following the given strategy."""
    value = rng.random(n)                # rival's private value ~ Uniform[0, 1]
    if strategy == "truthful":           # (a) bid the true value
        return value
    if strategy == "half":               # (b) bid half of the true value
        return value / 2.0
    if strategy == "random":             # (c) ignore value, bid a fresh U[0, 1]
        return rng.random(n)
    raise ValueError(f"unknown strategy: {strategy}")


def highest_rival_bid(strategy, n_rivals, n, rng):
    """
    Bidder 1 only ever pays the highest rival bid (the second-highest bid
    overall, whenever bidder 1 wins).  With several rivals who all use the same
    strategy, that is the element-wise maximum of their bids.
    """
    bids = np.stack([rival_bids(strategy, n, rng) for _ in range(n_rivals)])
    return bids.max(axis=0)


def utility(v, b, r):
    """
    Bidder 1's profit in a second-price auction.
      wins  (b > r): pays the rival's bid r, profit = v - r
      loses (b <= r): profit = 0   (ties are treated as a loss)
    Works with scalars or numpy arrays.
    """
    return np.where(b > r, v - r, 0.0)


def average_utility_curve(v, r):
    """Average utility of bidder 1 for every bid in BIDS, given an array r of
    simulated rival bids (one entry per auction)."""
    return np.array([utility(v, b, r).mean() for b in BIDS])


def theoretical_utility(v, strategy, n_rivals):
    """
    Exact expected utility for every bid in BIDS (used to sanity-check the
    simulation).  Let n = number of rivals.
      truthful / random : highest rival bid has density n*r^(n-1) on [0, 1]
                          E(b) = v*b^n - n*b^(n+1)/(n+1)
      half              : highest rival bid lives on [0, 0.5], density
                          n*2^n*r^(n-1), so cap b at 0.5 and multiply by 2^n
    """
    n = n_rivals
    if strategy == "half":
        b = np.minimum(BIDS, 0.5)
        scale = 2.0 ** n
    else:
        b = BIDS
        scale = 1.0
    return scale * (v * b ** n - n * b ** (n + 1) / (n + 1))


def fixed_bid_experiment(rival_sets):
    """
    Part B style check: rivals bid FIXED numbers (no randomness).
    rival_sets is a list of tuples, e.g. [(0.1,), (0.4,)] for one rival or
    [(0.1, 0.4), ...] for two rivals.  Bidder 1 pays the highest rival bid.
    """
    results, checks, curves = [], [], {}
    for v in VALUES:
        for rs in rival_sets:
            r = max(rs)
            curve = utility(v, BIDS, r)              # utility for each bid
            curves[(v, rs)] = curve
            for b, u in zip(BIDS, curve):
                results.append([v, *rs, b, u])
            best = curve.max()
            u_truth = curve[round(v * 100)]          # utility when b == v
            optimal = BIDS[curve >= best - TOL]      # every bid that is optimal
            checks.append([v, *rs, u_truth, best, optimal.min(), optimal.max(),
                           bool(u_truth >= best - TOL)])
    return results, checks, curves


# ----------------------------------------------------------------------------
# CSV + plotting helpers
# ----------------------------------------------------------------------------
def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([f"{x:.6f}" if isinstance(x, (float, np.floating)) else x
                        for x in row])
    print(f"wrote {path}  ({len(rows)} rows)")


def plot_strategy_grid(curves, theory, title, path):
    """3x3 grid: rows = bidder 1's value, columns = rival strategy."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for i, v in enumerate(VALUES):
        for j, s in enumerate(STRATEGIES):
            ax = axes[i][j]
            ax.plot(BIDS, curves[(v, s)], label="simulated")
            ax.plot(BIDS, theory[(v, s)], "--", linewidth=1, label="theory")
            ax.axvline(v, color="red", linestyle=":", label="b = v")
            ax.set_title(f"v = {v}, rival: {s}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == 2:
                ax.set_xlabel("bid b of bidder 1")
            if j == 0:
                ax.set_ylabel("average utility")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def plot_fixed_grid(curves, rival_sets, title, path):
    """Rows = bidder 1's value, columns = the fixed rival bid(s)."""
    ncols = len(rival_sets)
    fig, axes = plt.subplots(3, ncols, figsize=(3.4 * ncols + 1, 9),
                             sharex=True, sharey="row", squeeze=False)
    for i, v in enumerate(VALUES):
        for j, rs in enumerate(rival_sets):
            ax = axes[i][j]
            ax.plot(BIDS, curves[(v, rs)])
            ax.axvline(v, color="red", linestyle=":", label="b = v")
            label = ", ".join(str(x) for x in rs)
            ax.set_title(f"v = {v}, rival bid(s): {label}", fontsize=9)
            ax.grid(alpha=0.3)
            if i == 2:
                ax.set_xlabel("bid b of bidder 1")
            if j == 0:
                ax.set_ylabel("utility")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")
