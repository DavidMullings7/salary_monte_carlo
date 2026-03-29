# Retirement Planning: Income & Savings Solver

A Monte Carlo retirement planner that answers the core question: **how much do you need to earn and save to achieve a target monthly spending amount in retirement?**

The solver works end-to-end: it finds the minimum starting salary at which the joint probability of accumulating enough *and* sustaining the full retirement drawdown equals the target success rate.

## Use Cases

- **Salary floor** — determine the minimum salary needed to retire on time and sustain your target monthly spending for the full retirement horizon
- **Legacy planning** — set a terminal wealth floor and solve for the salary that achieves it at the target success rate
- **Retirement spending calibration** — understand the salary implications of different monthly spend targets
- **Social Security gap planning** — model the portfolio cost of retiring before Social Security begins, with a configurable delay
- **Inheritance planning** — model the impact of a future lump-sum, including timing effects
- **Market assumption stress testing** — see how sensitive the plan is to accumulation vs. retirement return assumptions and volatility
- **Partner income tradeoffs** — assess how a partner's income affects your required salary

## How It Works

### Joint Accumulation + Decumulation Solver

The solver finds the minimum starting salary at which the joint probability of accumulating enough AND sustaining the full retirement drawdown equals the target success rate. At each candidate salary, Brent's method pipes the 10,000 accumulation ending portfolios directly into the retirement drawdown simulation and measures the fraction of end-to-end paths that satisfy the success condition. This treats the full financial lifecycle as a single process — a bad late-accumulation sequence depresses early retirement returns in the same simulation path, correctly capturing sequence-of-returns risk at the transition.

When a legacy goal is set, the success condition is `ending portfolio ≥ legacy target` rather than `ending portfolio > $0`. The solver optimizes for the fraction of paths meeting the legacy floor.

### Return Model

A **single AR(1) lognormal process** spans the full horizon (accumulation years + retirement years) and is split at the retirement date. Both the unconditional mean and volatility shift at retirement to reflect a different asset allocation (e.g. lower-return, lower-volatility). The AR(1) autocorrelation structure and latent market state remain continuous across the boundary. A single seed controls the full path set for reproducibility.

## Inputs

### Household

| Parameter | Description |
| --- | --- |
| Partner Income | Gross annual income of the non-solver spouse. Used to compute joint federal tax liability. |
| Working-Years Spending | Annual household spending during accumulation, in today's dollars. |
| Current Portfolio | Total investable assets today (401k, brokerage, IRA, etc.). |

### Accumulation

| Parameter | Description |
| --- | --- |
| Years to Retirement | Investment horizon in years. |

### Retirement Income Goal

| Parameter | Description |
| --- | --- |
| Monthly Spend in Retirement | Target monthly spending once retired, in today's dollars. The required salary is derived from this. |
| Years in Retirement | Length of the retirement drawdown horizon. |
| Success Rate | Fraction of Monte Carlo end-to-end scenarios that must satisfy the success condition. |
| Social Security ($/mo) | Expected monthly Social Security benefit, which offsets the required portfolio draw once it begins. |
| SS Delay (years) | Years after retirement before Social Security begins. During this gap the full monthly spend is drawn from the portfolio. Hidden when Social Security is $0. |

### Legacy Goal

| Parameter | Description |
| --- | --- |
| Legacy Goal ($) | Minimum terminal portfolio value to leave at end of retirement. When set above $0, the solver optimizes for the fraction of paths ending at or above this floor rather than merely above $0. |

### Inheritance

| Parameter | Description |
| --- | --- |
| Amount | Expected lump-sum inheritance in today's dollars. |
| Year | Year within the accumulation horizon when the inheritance arrives. Hidden when amount is $0. |

### Market Assumptions

| Parameter | Description |
| --- | --- |
| Accum. Return | Expected mean annual return during the accumulation phase. |
| Accum. Volatility (σ) | Annualized standard deviation of log returns during accumulation. |
| Retirement Return | Expected mean annual return during the retirement drawdown phase. Typically lower, reflecting a more conservative allocation. |
| Retire Volatility (σ) | Annualized standard deviation of log returns during retirement. Typically lower than accumulation volatility, reflecting a more conservative allocation. |
| Salary Growth | Annual salary growth rate. |

### Tax & Plan Settings

Collapsed by default. These reflect current US tax law.

| Parameter | Description |
| --- | --- |
| 401(k) Limit | Annual pre-tax 401(k) contribution limit. |
| HSA Limit | Annual HSA contribution limit. |
| Employer Match | 401(k) employer match as a percentage of gross salary. |
| Standard Deduction | Federal standard deduction applied before bracket tax. |
| SS Rate | Employee Social Security payroll tax rate. |
| SS Wage Base | Maximum wages subject to Social Security tax. |
| Medicare Rate | Base Medicare payroll tax rate. |
| Add'l Medicare Rate | Additional Medicare surcharge rate above the threshold. |
| Add'l Medicare Threshold | Wage threshold above which the surcharge applies. |
| Bracket Thresholds | Upper income boundary for each federal tax bracket. Rates are fixed; thresholds are editable. |

## Tax Model

Federal income tax is computed using current brackets and the standard deduction. Joint tax liability is split using the **Shapley value** — each earner bears the average marginal cost of their income across both orderings of who earns first. This is the fairest allocation of the marriage penalty or bonus.

FICA (Social Security and Medicare) is computed on the solver's salary only, including the additional Medicare surcharge above the threshold.

Pre-tax contributions (401k and HSA) are maximized subject to the constraint that after-tax income covers working-years spending. After-tax surplus flows into additional portfolio contributions.

**Note:** State income tax is not modeled. Treat the required salary as a lower bound if you pay state income tax.

## Interpreting Results

### Required Starting Salary

The minimum gross salary at which the joint success rate meets the target. Assumes every dollar of after-tax surplus above spending is invested.

### Median Portfolio at Retirement

The median ending accumulation portfolio across all simulations at the solved salary. Display only — the solver does not optimize against this value.

### Joint Success Rate / Legacy Success

When no legacy goal is set: the fraction of end-to-end simulations that both accumulate enough and sustain the full retirement drawdown without depletion.

When a legacy goal is set: the fraction of simulations ending retirement with a portfolio at or above the legacy floor. This is the metric the solver optimizes against.

### Portfolio Survival Rate

The fraction of simulations where the portfolio is not fully depleted by end of retirement, regardless of the legacy floor. Always greater than or equal to the legacy success rate.

### Retirement Journey Chart

A unified timeline showing both phases. Blue lines are accumulation (10th/50th/90th percentile); orange lines are the retirement drawdown. The vertical dashed line marks retirement; the red dashed horizontal marks the median portfolio at retirement. When a legacy goal is set, a green dashed horizontal marks the legacy floor.

### Retirement Drawdown Paths

The 10th, 50th, and 90th percentile portfolio trajectories through retirement. The 10th percentile path is most relevant to plan durability.

### Paths Above 10th-Percentile Outcome

Fraction of accumulation paths staying above the 10th-percentile ending portfolio value in each year. Useful for understanding how early retirement might be possible in strong market conditions.

## Social Security Gap

When `SS Delay > 0`, the annual draw schedule is not uniform across retirement:

- **Years 0 → delay − 1**: full monthly spend drawn from the portfolio
- **Years delay → end**: draw reduced by the monthly Social Security benefit

This increases the required portfolio relative to immediate SS, with the magnitude depending on benefit size, delay length, and return assumptions during the gap.

## Assumptions and Limitations

- Working-years and retirement spending are constant in real terms; inflation is not modeled
- The model assumes 100% of after-tax surplus above spending is invested
- Social Security is modeled as a fixed flat amount starting at a fixed year; COLA adjustments are not modeled
- State income taxes are not modeled
- Volatility (σ) can differ between phases but is constant within each phase
- Inheritance is treated as a certain event in a known year; probability of receipt is not modeled
- The 401k limit, HSA limit, and tax brackets reflect current law and will need updating over time

## Installation

```bash
pip install streamlit numpy scipy plotly
streamlit run dashboard.py
```
