"""
Part B: the rival bids a FIXED number (0.1, 0.4 or 0.9) - no randomness.
Check that bidding b = v gives bidder 1 the highest possible utility.
"""
from auction_common import fixed_bid_experiment, plot_fixed_grid, write_csv

RIVAL_SETS = [(0.1,), (0.4,), (0.9,)]      # one rival, three fixed bids


def main():
    results, checks, curves = fixed_bid_experiment(RIVAL_SETS)

    write_csv("partB_results.csv",
              ["v", "rival2_bid", "bid", "utility"], results)
    write_csv("partB_check.csv",
              ["v", "rival2_bid", "utility_at_b_equals_v", "max_utility",
               "optimal_bid_low", "optimal_bid_high", "truthful_is_optimal"],
              checks)
    plot_fixed_grid(curves, RIVAL_SETS,
                    "Part B: utility vs bid against fixed rival bids (1 rival)",
                    "partB_plots.png")

    print("\nTruthful bidding optimal in every case?",
          all(row[-1] for row in checks))


if __name__ == "__main__":
    main()
