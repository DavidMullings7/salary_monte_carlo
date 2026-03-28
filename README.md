# Retirement Planning: Income & Savings Solver

A two-phase Monte Carlo retirement planner that answers the core question: **how much do you need to earn and save to achieve a target monthly spending amount in retirement?**

The app works backwards from your retirement spending goal. It derives the portfolio required at retirement accounting for return volatility and sequence-of-returns risk, then solves for the minimum starting salary needed to accumulate that portfolio.

## Use Cases

- **Salary floor** — determine the minimum salary needed to retire on time and sustain your target monthly spending for the full retirement horizon
- **Retirement spending calibration** — understand the salary implications of different monthly spend targets
- **Social Security gap planning** — model the portfolio cost of retiring before Social Security begins, with a configurable delay
- **Inheritance planning** — model the impact of a future lump-sum, including timing effects
- **Market assumption stress testing** — see how sensitive the plan is to accumulation vs. retirement return assumptions and volatility
- **Partner income tradeoffs** — assess how a partner's income affects your required salary

## How It Works

### Phase 1 — Decumulation Solver

Given the target monthly spend, Social Security offset, delay, and survival rate target, the solver finds the portfolio P\* at retirement such that the Monte Carlo portfolio survival rate equals the target (e.g. 90%). The solver runs Brent's method over 10,000 simulated retirement drawdown paths, each starting at a candidate portfolio value, until it finds P\* where exactly the target fraction survive the full retirement horizon.

### Phase 2 — Accumulation Solver

Given P\* from Phase 1, the solver finds the minimum starting salary at which the Monte Carlo accumulation success rate equals the target (e.g. 70%). At each candidate salary it computes annual after-tax contributions, simulates portfolio growth across all 10,000 paths, and measures the fraction reaching P\*.

### Return Model

Both phases draw from a **single AR(1) lognormal process** spanning the full horizon (accumulation years + retirement years), split at the retirement date. This preserves the market state at the transition: a crash in the final accumulation years feeds directly into early retirement returns. The unconditional mean shifts at retirement to reflect a different asset allocation, while the autocorrelation structure remains continuous. A single seed controls the full path set for reproducibility.

## Inputs

### Household

| Parameter | Description |
| --- | --- |
| Partner Income | Gross annual income of the non-solver spouse. Used to compute joint federal tax liability. |
| Working-Years Spending | Annual household spending during accumulation, in today's dollars. |
| Current Portfolio | Total investable assets today (401k, brokerage, IRA, etc.). |

### Retirement Income Goal

| Parameter | Description |
| --- | --- |
| Monthly Spend in Retirement | Target monthly spending once retired, in today's dollars. The required portfolio and salary are derived from this. |
| Years in Retirement | Length of the retirement drawdown horizon. |
| Portfolio Survival Rate | Fraction of Monte Carlo scenarios in which the portfolio must last the full retirement horizon. |
| Social Security ($/mo) | Expected monthly Social Security benefit, which offsets the required portfolio draw once it begins. |
| SS Delay (years) | Years after retirement before Social Security begins. During this gap the full monthly spend is drawn from the portfolio. Hidden when Social Security is $0. |

### Accumulation

| Parameter | Description |
| --- | --- |
| Years to Retirement | Investment horizon in years. |
| Accumulation Success Rate | Fraction of Monte Carlo scenarios that must reach the target portfolio by retirement. |

### Inheritance

| Parameter | Description |
| --- | --- |
| Amount | Expected lump-sum inheritance in today's dollars. |
| Year | Year within the accumulation horizon when the inheritance arrives. |

### Market Assumptions

| Parameter | Description |
| --- | --- |
| Accum. Return | Expected mean annual return during the accumulation phase. |
| Retirement Return | Expected mean annual return during the retirement drawdown phase. Typically lower, reflecting a more conservative allocation. |
| Volatility (σ) | Annualized standard deviation of log returns. Applied across both phases. |
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

The minimum gross salary at which the accumulation success rate meets the target. Assumes every dollar of after-tax surplus above spending is invested.

### Target Portfolio at Retirement

The portfolio required at retirement to sustain the target monthly spend for the full horizon at the target survival rate. Derived from the Monte Carlo decumulation solver, accounting for sequence-of-returns risk.

### Accumulation Success Rate

The fraction of 10,000 simulated accumulation paths that reach the target portfolio by retirement.

### Portfolio Survival Rate

The fraction of simulated retirement paths — each starting from its own accumulation ending value — in which the portfolio is not depleted by end of the retirement horizon.

### Retirement Journey Chart

A unified timeline showing both phases. Blue bands are accumulation (10th/50th/90th percentile); orange bands are the retirement drawdown. The vertical line marks retirement; the dashed horizontal marks the target portfolio.

### Retirement Drawdown Paths

The 10th, 50th, and 90th percentile portfolio trajectories through retirement. The 10th percentile path is most relevant to plan durability.

### Probability of Reaching Target Portfolio

Fraction of accumulation scenarios that have crossed the target portfolio in each year. Useful for understanding how early retirement might be possible in strong market conditions.

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
- Volatility (σ) is shared across both phases; only the mean return shifts at retirement
- Inheritance is treated as a certain event in a known year; probability of receipt is not modeled
- The 401k limit, HSA limit, and tax brackets reflect current law and will need updating over time

## Installation

```bash
pip install streamlit numpy scipy plotly
streamlit run dashboard.py
```
