# Part 2 — First-price auction, n bidders, one deviator

Run: `python3 first_price_auction.py` (standard library only; about 35 s).

## Setup
- All values v ~ U[0,1], independent. Rivals 2..n bid the equilibrium (n−1)/n·v. Bidder 1 bids k·v, k ∈ {0.1, …, 2.0}.
- Highest bid wins and pays its own bid. Ties have probability zero and are broken at random.
- 200,000 auctions per n. The same random draws are reused for every k (common random numbers).

## Closed form used for checking
With c = k·n/(n−1), E[U | v] = v(1−k)·min(1, cv)^(n−1), so

- c ≤ 1: E[U] = (1−k)·c^(n−1)/(n+1)
- c > 1: E[U] = (1−k)·(1/2 − (n−1)/(2(n+1)c²))

The maximum is at k* = (n−1)/n, where E[U] = 1/(n(n+1)). This holds for every fixed v, not only on average.

**Check:** all 300 Monte Carlo estimates (uniform v and fixed v ∈ {0.2, 0.5, 0.8}) are within 4 standard errors of the closed form, and the closed form matches numerical integration.

## Results (v₁ ~ U[0,1])

| n | k* | E[U] at k* (theory) | best grid k (sim) | E[U] there (sim / theory) | Winner could pay less | Mean money left on table at k* |
|---|----|----|----|----|----|----|
| 2 | 0.500 | 0.1667 | 0.5 | 0.1664 / 0.1667 | 100 % | 0.167 |
| 3 | 0.667 | 0.0833 | 0.7 | 0.0816 / 0.0820 | 100 % | ≈0.165 |
| 4 | 0.750 | 0.0500 | 0.7 | 0.0486 / 0.0488 | 100 % | ≈0.15 |

The tables for every k are in `first_price_results.csv` and `first_price_fixed_value_results.csv`. The plots are the three `.svg` files.

## Interpretation
- **Deviating does not pay.** Expected utility peaks at the equilibrium shading k* = (n−1)/n. For n = 3 and 4, k* is not on the 0.1 grid, so the best grid point is the neighbour closest to it.
- **Bidding too low** (k < k*) rarely wins. **Bidding too high** (k > k*) wins more often but pays too much. k = 1 always gives exactly 0, and k > 1 always loses money.
- **Competition shades bids less.** As n grows, k* rises towards 1 and the bidder's surplus 1/(n(n+1)) shrinks.
- **The winner could almost always have paid less.** With continuous values, the winning bid is above the second-highest bid with probability 1, so the measured fraction is 100 % for every k and n. This is the cost of a first-price rule: the winner pays its own bid, not the lowest bid that would still have won. A second-price auction removes this gap by charging the second-highest bid. The size of the gap (`mean_money_left_on_table`) is smallest near k* and grows quickly when bidder 1 over-bids.
