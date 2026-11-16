"""
Part A: one rival (bidder 2), three rival strategies, three values for bidder 1.
For every (v, strategy) pair we try bids 0.00 ... 1.00 and average the utility
over 100,000 simulated auctions.
"""
import numpy as np

from auction_common import (BIDS, N_AUCTIONS, SEED, STRATEGIES, VALUES,
                            average_utility_curve, highest_rival_bid,
                            plot_strategy_grid, theoretical_utility, utility,
                            write_csv)

N_RIVALS = 1


def main():
    rng = np.random.default_rng(SEED)
    curves, theory, rows, summary = {}, {}, [], []

    for s in STRATEGIES:
        # 100,000 independent rival bids drawn according to the strategy.
        # The same draws are reused for every bid b and every v, which makes
        # the curves smooth (and the comparison between bids fair).
        r = highest_rival_bid(s, N_RIVALS, N_AUCTIONS, rng)

        for v in VALUES:
            curve = average_utility_curve(v, r)          # 101 average utilities
            th = theoretical_utility(v, s, N_RIVALS)
            curves[(v, s)], theory[(v, s)] = curve, th

            for b, u, t in zip(BIDS, curve, th):
                rows.append([v, s, b, u, t])

            i_v = round(v * 100)                          # index of b == v
            best = curve.max()
            # standard error of the average utility at b == v
            se = utility(v, v, r).std() / np.sqrt(N_AUCTIONS)
            gap = best - curve[i_v]
            summary.append([v, s, BIDS[curve.argmax()], best, curve[i_v], gap,
                            bool(gap <= 3 * se)])

    write_csv("partA_results.csv",
              ["v", "rival_strategy", "bid", "avg_utility_simulated",
               "utility_theory"], rows)
    write_csv("partA_summary.csv",
              ["v", "rival_strategy", "best_bid_simulated", "max_avg_utility",
               "avg_utility_at_b_equals_v", "gap", "truthful_within_noise"],
              summary)
    plot_strategy_grid(curves, theory,
                       "Part A: bidder 1's average utility vs bid (1 rival)",
                       "partA_plots.png")


if __name__ == "__main__":
    main()
