"""
Part C: repeat Parts A and B with TWO rivals (bidder 2 and bidder 3).
Both rivals use the same strategy and draw their values from U[0, 1].
Bidder 1 now has to beat - and pays - the HIGHER of the two rival bids.
"""
from itertools import combinations_with_replacement

import numpy as np

from auction_common import (BIDS, N_AUCTIONS, SEED, STRATEGIES, VALUES,
                            average_utility_curve, fixed_bid_experiment,
                            highest_rival_bid, plot_fixed_grid,
                            plot_strategy_grid, theoretical_utility, utility,
                            write_csv)

N_RIVALS = 2
FIXED_BIDS = [0.1, 0.4, 0.9]
RIVAL_SETS = list(combinations_with_replacement(FIXED_BIDS, 2))  # 6 pairs


def part_a_with_two_rivals():
    rng = np.random.default_rng(SEED)
    curves, theory, rows, summary = {}, {}, [], []

    for s in STRATEGIES:
        # max of the two rivals' bids, 100,000 auctions
        r = highest_rival_bid(s, N_RIVALS, N_AUCTIONS, rng)

        for v in VALUES:
            curve = average_utility_curve(v, r)
            th = theoretical_utility(v, s, N_RIVALS)
            curves[(v, s)], theory[(v, s)] = curve, th

            for b, u, t in zip(BIDS, curve, th):
                rows.append([v, s, b, u, t])

            i_v = round(v * 100)
            best = curve.max()
            se = utility(v, v, r).std() / np.sqrt(N_AUCTIONS)
            gap = best - curve[i_v]
            summary.append([v, s, BIDS[curve.argmax()], best, curve[i_v], gap,
                            bool(gap <= 3 * se)])

    write_csv("partC_results.csv",
              ["v", "rival_strategy", "bid", "avg_utility_simulated",
               "utility_theory"], rows)
    write_csv("partC_summary.csv",
              ["v", "rival_strategy", "best_bid_simulated", "max_avg_utility",
               "avg_utility_at_b_equals_v", "gap", "truthful_within_noise"],
              summary)
    plot_strategy_grid(curves, theory,
                       "Part C (A repeated): average utility vs bid (2 rivals)",
                       "partC_A_plots.png")


def part_b_with_two_rivals():
    results, checks, curves = fixed_bid_experiment(RIVAL_SETS)

    write_csv("partC_fixed_results.csv",
              ["v", "rival2_bid", "rival3_bid", "bid", "utility"], results)
    write_csv("partC_fixed_check.csv",
              ["v", "rival2_bid", "rival3_bid", "utility_at_b_equals_v",
               "max_utility", "optimal_bid_low", "optimal_bid_high",
               "truthful_is_optimal"], checks)
    plot_fixed_grid(curves, RIVAL_SETS,
                    "Part C (B repeated): utility vs bid, fixed rival bids (2 rivals)",
                    "partC_B_plots.png")
    print("\nTruthful bidding optimal in every case?",
          all(row[-1] for row in checks))


if __name__ == "__main__":
    part_a_with_two_rivals()
    part_b_with_two_rivals()
