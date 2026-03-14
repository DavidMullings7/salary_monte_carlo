# Retirement Salary Monte Carlo Simulator

A Monte Carlo retirement planner that solves for the minimum salary required to reach a target portfolio with a given probability, accounting for federal taxes, FICA, pre-tax contributions, employer match, spousal income with Shapley-fair tax splitting, inheritance timing, and stochastic market returns.

## Use Cases

- **Salary negotiation floor** — determine the minimum salary you need to retire on time with acceptable probability, given your household's full financial picture
- **Retirement timeline sensitivity** — understand how adding or removing years changes required salary
- **Inheritance planning** — model the impact of a future inheritance on retirement feasibility, including the effect of when it arrives
- **Partner income tradeoffs** — assess how changes in a partner's income (career break, part-time, etc.) affect your required salary
- **Market assumption stress testing** — see how sensitive your retirement plan is to return assumptions and volatility

## How It Works

The simulator solves a root-finding problem: it searches for the salary at which your Monte Carlo success rate equals your target (e.g. 75%). Internally it:

1. Generates 10,000 simulated return paths using a lognormal model
2. For each candidate salary, computes annual contributions after taxes, pre-tax deductions, and spending
3. Simulates portfolio growth across all paths
4. Measures the fraction of paths that hit the target portfolio
5. Uses Brent's method to find the exact salary where that fraction equals the target success rate

## Inputs

### Household

| Parameter         | Description                                                                            |
| ----------------- | -------------------------------------------------------------------------------------- |
| Partner Income    | Your partner's gross annual income. Used to compute joint federal tax liability.       |
| Annual Spending   | Total household spending per year in today's dollars. The model does not inflate this. |
| Current Portfolio | Total investable assets today (401k, brokerage, IRA, etc.).                            |

### Goal

| Parameter           | Description                                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Target Portfolio    | The portfolio value you want to reach by end of horizon. A common rule of thumb is 25x annual spending (4% rule).            |
| Years               | Investment horizon in years.                                                                                                 |
| Target Success Rate | The fraction of Monte Carlo scenarios that must reach the target. 75% is moderately conservative; 90%+ is very conservative. |

### Inheritance

| Parameter        | Description                                               |
| ---------------- | --------------------------------------------------------- |
| Inheritance      | Expected lump-sum inheritance in today's dollars.         |
| Inheritance Year | The year within the horizon when the inheritance arrives. |

### Market Assumptions

| Parameter         | Description                                          |
| ----------------- | ---------------------------------------------------- |
| Mean Return       | Expected annualized real return on the portfolio.    |
| Return Volatility | Annualized standard deviation of log returns.        |
| Salary Growth     | Real (inflation-adjusted) annual salary growth rate. |

## Tax Model

Federal income tax is computed using current brackets and the standard deduction. Joint tax liability is split between spouses using the **Shapley value** — each person bears the average marginal cost of their income across both orderings of who earns first. This is mathematically the fairest allocation of the marriage penalty (or bonus) and avoids arbitrarily assigning the joint tax to one earner.

FICA taxes (Social Security and Medicare) are computed separately on your salary only, including the additional Medicare surcharge above $250,000.

Pre-tax contributions (401k and HSA) are maximized subject to the constraint that after-tax income covers spending. Contributions above the pre-tax limit flow into taxable after-tax savings.

**Note:** State income tax is not modeled. For Texas residents this is accurate; for other states you should treat the required salary output as a lower bound.

## Return Model

Returns are modeled as **lognormally distributed**, meaning gross returns `(1 + r)` follow a lognormal distribution and log returns are normally distributed. This is more realistic than a normal model because:

- Returns can never go below -100% (portfolio can't go negative)
- The distribution is right-skewed, matching empirical return distributions
- Long-run compounding is correctly captured

The `mean_return` parameter is interpreted as the arithmetic mean annual return. It is internally converted to the correct lognormal `μ` parameter so that the expected gross return equals `1 + mean_return`.

## Interpreting Results

### Required Salary

The minimum gross salary at which your Monte Carlo success rate meets the target. This is a **floor**, not a recommendation — it assumes you save every dollar of after-tax surplus above spending.

### Success Rate

The fraction of 10,000 simulated market scenarios in which the portfolio reaches the target. A 75% success rate means 1 in 4 scenarios falls short. Higher is safer but requires a higher salary.

### Median Ending Portfolio

The 50th percentile terminal portfolio across all simulations. This will typically be well above the target portfolio if the success rate is at 75%, because the median scenario is better than the 25th percentile scenario the solver is actually targeting.

### Wealth Paths Chart

Shows the 10th, 50th, and 90th percentile portfolio trajectories over time. The wide spread between percentiles reflects return volatility — the fan shape is expected and is not a cause for concern. The 10th percentile path is the one most relevant to your retirement security.

### Probability of FI Over Time

Shows the fraction of scenarios that have already crossed the target portfolio in each year. This is useful for understanding whether you might retire early in good market conditions.

### Contributions by Year

Annual contributions to the portfolio, including pre-tax deductions, employer match, and after-tax surplus. The spike in the inheritance year reflects the lump-sum addition. If contributions are flat or declining in real terms, this reflects the salary growth rate and tax drag.

## Inheritance Timing

A common intuition is that receiving inheritance earlier is always better because it compounds longer. This is true in expectation (median outcome) but **not necessarily true at a given success percentile** over typical retirement horizons.

Receiving $2M in year 1 instead of year 18 exposes that capital to 17 additional years of market risk. At the 25th percentile of outcomes, the left tail of 17 years of lognormal compounding can be worse than receiving the same $2M with certainty at year 18. The crossover horizon — at which earlier inheritance reliably helps the 25th percentile — is approximately:

```
T* = (0.674 * σ / μ)² / 4
```

With default parameters (σ=0.20, μ≈0.019), this is roughly 49 years. Over an 18-year horizon, the model correctly shows that later inheritance may require a lower salary at a 75% success target.

## Assumptions and Limitations

- Spending is constant in real terms and not inflation-adjusted within the model
- Social Security income is not modeled
- State income taxes are not modeled
- The model assumes you save 100% of after-tax surplus above spending — no lifestyle inflation
- Returns are lognormal with weak AR(1) mean reversion (φ=−0.15) — no regime switching or sequence-of-returns modeling beyond volatility
- The 401k limit, HSA limit, and tax brackets reflect current law and will need updating over time
- Inheritance is treated as a certain event in a known year — probability of inheritance is not modeled

## Installation

```bash
pip install streamlit numpy scipy plotly
streamlit run dashboard.py
```

## License

MIT
