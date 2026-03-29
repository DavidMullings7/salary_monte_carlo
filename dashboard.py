import logging
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brentq

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SIMULATIONS = 10000

# Default tax/plan constants (overridden by UI below)
STANDARD_DEDUCTION = 30000
BRACKETS = [
    (23850, 0.10),
    (96950, 0.12),
    (206700, 0.22),
    (394600, 0.24),
    (501050, 0.32),
    (751600, 0.35),
    (float("inf"), 0.37),
]
SOCIAL_SECURITY_RATE = 0.062
SOCIAL_SECURITY_WAGE_BASE = 168600
MEDICARE_RATE = 0.0145
ADDITIONAL_MEDICARE_RATE = 0.009
ADDITIONAL_MEDICARE_THRESHOLD = 250000
K401_LIMIT = 24500
HSA_LIMIT = 4500

# -----------------------------
# TAX CALCULATIONS
# -----------------------------

def federal_tax(taxable_income):
    tax = 0
    prev_limit = 0
    for limit, rate in BRACKETS:
        if taxable_income > limit:
            tax += (limit - prev_limit) * rate
            prev_limit = limit
        else:
            tax += (taxable_income - prev_limit) * rate
            break
    return max(tax, 0)


def tax_liability(income):
    taxable = max(0, income - STANDARD_DEDUCTION)
    return federal_tax(taxable)


def shapley_tax(incomes):
    a, b = incomes
    tax_a_first  = tax_liability(a) - tax_liability(0)
    tax_a_second = tax_liability(a + b) - tax_liability(b)
    tax_b_first  = tax_liability(b) - tax_liability(0)
    tax_b_second = tax_liability(a + b) - tax_liability(a)
    shapley_a = 0.5 * (tax_a_first + tax_a_second)
    shapley_b = 0.5 * (tax_b_first + tax_b_second)
    return [shapley_a, shapley_b]

# -----------------------------
# FICA
# -----------------------------

def fica_tax(wages):
    ss_tax = min(wages, SOCIAL_SECURITY_WAGE_BASE) * SOCIAL_SECURITY_RATE
    medicare_tax = wages * MEDICARE_RATE
    additional_medicare = max(0, wages - ADDITIONAL_MEDICARE_THRESHOLD) * ADDITIONAL_MEDICARE_RATE
    return ss_tax + medicare_tax + additional_medicare

# -----------------------------
# AFTER TAX INCOME
# -----------------------------

def personal_after_tax_income(salary, partner_income, k401, hsa):
    pretax = k401 + hsa
    incomes = [salary - pretax, partner_income]
    tax_split = shapley_tax(incomes)
    my_federal_tax = tax_split[0]
    payroll_tax = fica_tax(salary)
    return salary - pretax - my_federal_tax - payroll_tax

# -----------------------------
# SALARY PATH
# -----------------------------

def generate_salary_path(start_salary, years, salary_growth):
    return np.array([start_salary * (1 + salary_growth) ** y for y in range(years)])

# -----------------------------
# PRETAX CONTRIBUTION SOLVER
# -----------------------------

def max_pretax_contribution(salary, partner_income, spending):
    def objective(pretax):
        net_income = personal_after_tax_income(salary, partner_income, pretax, 0)
        return net_income - spending
    try:
        pretax = float(brentq(objective, 0, K401_LIMIT + HSA_LIMIT))  # type: ignore[arg-type]
    except ValueError:
        pretax = K401_LIMIT + HSA_LIMIT
    return pretax

# -----------------------------
# CONTRIBUTIONS
# -----------------------------

def compute_contributions(salaries, partner_income, spending, match_rate):
    years = len(salaries)
    contributions = np.zeros(years)
    for y in range(years):
        salary = salaries[y]
        pretax = max_pretax_contribution(salary, partner_income, spending)
        k401 = min(pretax, K401_LIMIT)
        hsa = min(max(0, pretax - k401), HSA_LIMIT)
        net_income = personal_after_tax_income(salary, partner_income, k401, hsa)
        match = salary * match_rate
        excess = net_income - spending
        if excess < -1.0:
            return None
        excess = max(excess, 0)
        contributions[y] = pretax + match + excess
    return contributions

# -----------------------------
# RETURN PATHS
# -----------------------------

def generate_return_paths(years, mean_return, return_std, phi=-0.15):
    log_mean = np.log(1 + mean_return) - 0.5 * return_std ** 2
    noise_std = return_std * np.sqrt(1 - phi ** 2)
    log_returns = np.zeros((SIMULATIONS, years))
    log_returns[:, 0] = np.random.normal(log_mean, return_std, SIMULATIONS)
    for t in range(1, years):
        noise = np.random.normal(0, noise_std, SIMULATIONS)
        log_returns[:, t] = log_mean + phi * (log_returns[:, t-1] - log_mean) + noise
    return np.exp(log_returns)


def generate_joint_return_paths(accum_years, retire_years, accum_mean, retire_mean,
                                 accum_std, retire_std, phi=-0.15):
    """Generate a single AR(1) process spanning accumulation and retirement, then split.

    Using one continuous process means the market state at the retirement boundary
    carries over: a crash in the final accumulation years depresses early retirement
    returns through the AR(1) dynamics, correctly modelling sequence-of-returns risk
    at the transition. Two independently seeded draws would miss this correlation
    entirely.

    The unconditional mean and volatility both shift at year accum_years to reflect
    a different asset allocation in retirement. The AR(1) autocorrelation structure
    and latent state remain continuous across the boundary.

    Returns:
        accum_growth: (SIMULATIONS, accum_years)
        retire_growth: (SIMULATIONS, retire_years)
    """
    accum_years = int(accum_years)
    retire_years = int(retire_years)
    total_years = accum_years + retire_years

    # Per-period volatility vector
    stds = np.empty(total_years)
    stds[:accum_years] = accum_std
    stds[accum_years:] = retire_std

    log_mean_accum  = np.log(1 + accum_mean)  - 0.5 * accum_std ** 2
    log_mean_retire = np.log(1 + retire_mean) - 0.5 * retire_std ** 2

    # Per-period mean vector
    log_means = np.empty(total_years)
    log_means[:accum_years]  = log_mean_accum
    log_means[accum_years:]  = log_mean_retire

    log_returns = np.zeros((SIMULATIONS, total_years))
    log_returns[:, 0] = np.random.normal(log_means[0], stds[0], SIMULATIONS)

    for t in range(1, total_years):
        noise_std = stds[t] * np.sqrt(1 - phi ** 2)
        noise = np.random.normal(0, noise_std, SIMULATIONS)
        # AR(1): deviation is measured against the mean of the *previous* period so
        # the autocorrelation term is well-defined even when mean/std shift at t=accum_years.
        log_returns[:, t] = log_means[t] + phi * (log_returns[:, t-1] - log_means[t-1]) + noise

    growth = np.exp(log_returns)
    return growth[:, :accum_years], growth[:, accum_years:]

# -----------------------------
# FAST FINAL PORTFOLIOS
# -----------------------------

def compute_final_portfolios(current_portfolio, contributions, growth):
    cum_growth = np.cumprod(growth, axis=1)
    start_component = current_portfolio * cum_growth[:, -1]
    reverse_growth = np.cumprod(growth[:, ::-1], axis=1)[:, ::-1]
    future_growth = np.ones_like(growth)
    future_growth[:, :-1] = reverse_growth[:, 1:]
    contribution_component = (contributions * future_growth).sum(axis=1)
    return start_component + contribution_component

# -----------------------------
# FULL PATHS (for charts)
# -----------------------------

def simulate_wealth_paths(current_portfolio, contributions, growth):
    sims, years = growth.shape
    wealth = np.zeros((sims, years + 1))
    wealth[:, 0] = current_portfolio
    for y in range(years):
        wealth[:, y+1] = wealth[:, y] * growth[:, y] + contributions[y]
    return wealth

# -----------------------------
# DECUMULATION
# -----------------------------



def make_annual_draws(monthly_spend, social_security_monthly, ss_start_year, retirement_years):
    """Build the per-year portfolio draw schedule for retirement.

    During the SS gap (years 0..ss_start_year-1) the full monthly_spend must be
    funded from the portfolio. Once SS begins, the draw is reduced by
    social_security_monthly. ss_start_year=0 means SS starts immediately.

    Returns an array of shape (retirement_years,)."""
    full_draw    = monthly_spend * 12
    reduced_draw = max(0.0, (monthly_spend - social_security_monthly) * 12)
    draws = np.full(retirement_years, reduced_draw)
    draws[:ss_start_year] = full_draw
    return draws


def simulate_decumulation_paths(starting_portfolios, annual_draws, retirement_years, ret_growth, legacy_target):
    """Monte Carlo drawdown using pre-generated return paths.
    Each simulation uses its own ending accumulation value as the starting portfolio.
    annual_draws is a (retirement_years,) array of per-year withdrawals, allowing
    the draw to vary (e.g. higher before Social Security begins).
    Returns wealth_paths (SIMULATIONS, retirement_years+1) and the portfolio survival rate."""
    sims = len(starting_portfolios)
    wealth = np.zeros((sims, retirement_years + 1))
    wealth[:, 0] = starting_portfolios
    for y in range(retirement_years):
        wealth[:, y + 1] = np.maximum(wealth[:, y] * ret_growth[:, y] - annual_draws[y], 0)
    
    if legacy_target > 0:
        survival_rate = float(np.mean(wealth[:, -1] >= legacy_target))
    else:
        survival_rate = float(np.mean(wealth[:, -1] > 0))

    return wealth, survival_rate


# -----------------------------
# SALARY MODEL
# -----------------------------

def simulate_salary_model(
    start_salary, partner_income, years, spending,
    current_portfolio, growth, salary_growth, match_rate, inheritance, inheritance_year
):
    logger.info(
        "simulate_salary_model | start_salary=%s partner_income=%s years=%s spending=%s "
        "current_portfolio=%s salary_growth=%s match_rate=%s inheritance=%s inheritance_year=%s",
        start_salary, partner_income, years, spending,
        current_portfolio, salary_growth, match_rate, inheritance, inheritance_year,
    )
    salaries = generate_salary_path(start_salary, years, salary_growth)
    contributions = compute_contributions(salaries, partner_income, spending, match_rate)
    if contributions is None:
        return None, None, None, None
    if inheritance_year is not None:
        contributions[inheritance_year - 1] += inheritance
    portfolios = compute_final_portfolios(current_portfolio, contributions, growth)
    wealth_paths = simulate_wealth_paths(current_portfolio, contributions, growth)
    if inheritance_year is not None:
        contributions[inheritance_year - 1] -= inheritance
    return portfolios, salaries, contributions, wealth_paths

# -----------------------------
# SALARY SOLVER
# -----------------------------

def solve_required_salary(
    partner_income, years, spending, current_portfolio,
    salary_growth, match_rate, target_success, inheritance, inheritance_year,
    accum_growth, annual_draws, retirement_years, ret_growth, legacy_target
):
    """Solve for the salary where the joint probability of accumulating enough AND
    sustaining the full retirement drawdown equals target_success.

    Each Brent iteration pipes the 10,000 accumulation ending portfolios directly into
    the decumulation simulation and measures the fraction that survive the full retirement
    horizon. This treats accumulation and decumulation as a single end-to-end process,
    avoiding the independence assumption of the two-stage approach."""

    def objective(salary):
        result = simulate_salary_model(
            salary, partner_income, years, spending, current_portfolio,
            accum_growth, salary_growth, match_rate, inheritance, inheritance_year
        )
        portfolios = result[0]
        if portfolios is None:
            # Salary can't cover working-years spending; treat joint success as 0.
            return -target_success
        _, joint_success = simulate_decumulation_paths(
            portfolios, annual_draws, retirement_years, ret_growth, legacy_target
        )
        return joint_success - target_success

    lo = 50_000.0
    hi = 600_000.0

    f_lo = objective(lo)

    # If joint success is already above the target at the minimum salary,
    # the current portfolio alone is likely sufficient — return lo.
    if f_lo >= 0:
        return lo

    # Expand the upper bound until objective(hi) > 0 or we hit an absolute ceiling.
    # This handles scenarios where a large current portfolio means even modest salaries
    # come close to the target, requiring a higher bound to confirm the bracket.
    MAX_SALARY = 5_000_000.0
    f_hi = objective(hi)
    while f_hi <= 0 and hi < MAX_SALARY:
        hi = min(hi * 2.0, MAX_SALARY)
        f_hi = objective(hi)

    if f_hi <= 0:
        raise ValueError(
            f"The {target_success:.0%} joint success target is not achievable even at a "
            f"${hi:,.0f} salary with the current inputs. "
            f"Try lowering the success rate, reducing retirement spending, or "
            f"increasing the retirement return assumption."
        )

    return float(brentq(objective, lo, hi))  # type: ignore[arg-type]

# -----------------------------
# SENSITIVITY HELPER
# -----------------------------

def _sensitivity_salary(overrides, base_params, base_accum_growth, base_ret_growth):
    """Solve required salary with one parameter overridden.

    Reuses precomputed return paths when the override does not affect path
    generation (i.e. non-return, non-horizon parameters). Regenerates paths
    with seed 42 otherwise so results are comparable across cases."""
    p = {**base_params, **overrides}
    p["years"] = int(p["years"])
    p["retirement_years"] = int(p["retirement_years"])
    p["ss_start_year"] = int(p["ss_start_year"])
    if p["inheritance_year"] is not None:
        p["inheritance_year"] = min(int(p["inheritance_year"]), p["years"])
    path_keys = {"mean_return", "retirement_return", "accum_std", "retire_std", "years", "retirement_years"}
    if path_keys & set(overrides):
        np.random.seed(42)
        ag, rg = generate_joint_return_paths(
            p["years"], p["retirement_years"],
            p["mean_return"], p["retirement_return"], p["accum_std"], p["retire_std"],
        )
    else:
        ag, rg = base_accum_growth, base_ret_growth
    draws = make_annual_draws(
        p["monthly_retirement_spend"], p["social_security_monthly"],
        p["ss_start_year"], p["retirement_years"],
    )
    try:
        return solve_required_salary(
            p["partner_income"], p["years"], p["spending"], p["current_portfolio"],
            p["salary_growth"], p["match_rate"], p["target_success"],
            p["inheritance"], p["inheritance_year"],
            ag, draws, p["retirement_years"], rg, p["legacy_target"],
        )
    except ValueError:
        return None

# ==============================
# STREAMLIT UI
# ==============================

# -----------------------------
# SIDEBAR — all inputs
# -----------------------------

with st.sidebar:

    st.title("Inputs")

    # Household
    st.subheader("Household")
    partner_income    = st.number_input("Partner Income ($)",           0, 1_000_000, 300_000, step=5_000)
    spending          = st.number_input("Working-Years Spending ($/yr)", 0,   500_000,  50_000, step=1_000)
    current_portfolio = st.number_input("Current Portfolio ($)",         0, 10_000_000, 240_000, step=10_000)

    st.divider()

    # Accumulation
    st.subheader("Accumulation")
    years = st.slider("Years to Retirement", 5, 40, 25)

    st.divider()

    # Retirement Income Goal — the primary input
    st.subheader("Retirement Income Goal")
    monthly_retirement_spend = st.number_input(
        "Monthly Withdrawal in Retirement ($)", 0, 50_000, 9_000, step=500,
        help="Target monthly withdrawal once retired, in today's dollars."
    )
    rc1, rc2 = st.columns(2)
    with rc1:
        retirement_years = st.slider("Years in Retirement", 10, 50, 40)
        target_success = st.slider("Success Rate", 50, 99, 70, format="%d%%") / 100
    with rc2:
        social_security_monthly = st.number_input(
            "Social Security ($/mo)", 0, 10_000, 2_500, step=100,
            help="Expected monthly Social Security benefit. Offsets required portfolio draw."
        )
        if social_security_monthly == 0:
            ss_start_year = 0
            st.caption("SS Delay (years) is ignored when Social Security monthly benefit is $0.")
        else:
            ss_start_year = st.number_input(
                "SS Delay (years)", 0, retirement_years, 20, step=1,
                help="Years after retirement before Social Security begins. "
                     "During this gap the full monthly spend is drawn from the portfolio."
            )

    st.divider()

    # Legacy Goal
    st.subheader("Legacy Goal")
    enable_legacy = st.checkbox("Set a legacy goal")
    legacy_target = st.number_input("Legacy Goal ($)", 0, 10_000_000, 1_000_000, step=50_000) if enable_legacy else 0

    st.divider()

    # Inheritance
    st.subheader("Inheritance")
    ic1, ic2 = st.columns(2)
    with ic1:
        inheritance = st.number_input("Amount ($)", 0, 10_000_000, 1_500_000, step=50_000)
    with ic2:
        # inheritance_year = st.number_input("Year", 1, years, years)
        if inheritance == 0:
            inheritance_year = 0
            st.caption("Inheritance year is ignored when inheritance is $0.")
        else:
            inheritance_year = st.number_input("Year", 1, years, years, help="The year in which the inheritance is received.")

    st.divider()

    # Market
    st.subheader("Market")
    mc1, mc2 = st.columns(2)
    with mc1:
        mean_return       = st.slider("Accumulation Return",    0.0, 10.0, 3.5, step=0.5, format="%.1f%%") / 100
        accum_std         = st.slider("Accum. Volatility (σ)",  0,   30,   18,             format="%d%%")   / 100
        salary_growth     = st.slider("Salary Growth",          0.0, 10.0,  1.5, step=0.5, format="%.1f%%") / 100
    with mc2:
        retirement_return = st.slider("Retirement Return",      0.0, 10.0,  2.5, step=0.5, format="%.1f%%") / 100
        retire_std        = st.slider("Retire Volatility (σ)",  0,   30,   10,             format="%d%%")   / 100

    st.divider()

    # Tax & Plan Settings
    with st.expander("Tax & Plan Settings"):

        st.caption("Defaults reflect current US tax law.")

        st.markdown("**401(k) & HSA**")
        p1, p2, p3 = st.columns(3)
        with p1:
            K401_LIMIT = st.number_input("401(k) ($)", 0, 100_000, 24_500, step=500)
        with p2:
            HSA_LIMIT  = st.number_input("HSA ($)",    0,  20_000,  4_500, step=100)
        with p3:
            match_rate = st.slider("Match", 0, 8, 7, format="%d%%") / 100

        st.markdown("**Federal Tax**")
        STANDARD_DEDUCTION = st.number_input("Standard Deduction ($)", 0, 100_000, 30_000, step=500)

        st.markdown("**FICA**")
        f1, f2 = st.columns(2)
        with f1:
            SOCIAL_SECURITY_RATE      = st.number_input("SS Rate",          0.0, 0.20,      0.062,   step=0.001, format="%.3f")
            SOCIAL_SECURITY_WAGE_BASE = st.number_input("SS Wage Base ($)", 0,   500_000,   168_600, step=1_000)
        with f2:
            MEDICARE_RATE                 = st.number_input("Medicare Rate",       0.0, 0.10, 0.0145,  step=0.001, format="%.4f")
            ADDITIONAL_MEDICARE_RATE      = st.number_input("Add'l Medicare Rate", 0.0, 0.10, 0.009,   step=0.001, format="%.3f")
            ADDITIONAL_MEDICARE_THRESHOLD = st.number_input("Add'l Threshold ($)", 0, 1_000_000, 250_000, step=10_000)

        st.markdown("**Bracket Thresholds** *(rates fixed)*")
        bracket_defaults = [23_850, 96_950, 206_700, 394_600, 501_050, 751_600]
        bracket_rates    = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
        bracket_labels   = ["10%", "12%", "22%", "24%", "32%", "35%"]
        new_limits = []
        bl1, bl2 = st.columns(2)
        bcols = [bl1, bl2, bl1, bl2, bl1, bl2]
        for i, (lbl, lim) in enumerate(zip(bracket_labels, bracket_defaults)):
            with bcols[i]:
                new_limits.append(st.number_input(f"{lbl} top ($)", 0, 2_000_000, lim, step=500, key=f"bracket_{i}"))
        BRACKETS = list(zip(new_limits, bracket_rates[:-1])) + [(float("inf"), 0.37)]

    st.divider()

    run = st.button("▶  Solve Required Salary", use_container_width=True, type="primary")

# -----------------------------
# MAIN AREA
# -----------------------------

st.title("Retirement Planning: Income & Savings Solver")
st.caption(
    "How much do you need to earn and save to achieve your target monthly spending in retirement?"
)

if not run:
    st.info("Configure inputs in the sidebar, then click **Solve Required Salary**.")
    st.stop()

# -----------------------------
# SOLVE + RESULTS
# -----------------------------

logger.info(
    "Inputs | monthly_retirement_spend=$%s retirement_years=%s "
    "social_security_monthly=$%s ss_start_year=%s partner_income=$%s years=%s spending=$%s "
    "current_portfolio=$%s salary_growth=%s match_rate=%s inheritance=$%s inheritance_year=%s "
    "legacy_target=$%s accum_std=%s retire_std=%s",
    f"{monthly_retirement_spend:,.0f}", retirement_years,
    f"{social_security_monthly:,.0f}", ss_start_year, f"{partner_income:,.0f}", years,
    f"{spending:,.0f}", f"{current_portfolio:,.0f}", f"{salary_growth:.2%}",
    f"{match_rate:.2%}", f"{inheritance:,.0f}", inheritance_year,
    f"{legacy_target:,.0f}", f"{accum_std:.0%}", f"{retire_std:.0%}",
)

with st.spinner("Running Monte Carlo simulation..."):

    # One AR(1) process spanning both phases, split at retirement.
    np.random.seed(42)
    accum_growth, ret_growth = generate_joint_return_paths(
        years, retirement_years, mean_return, retirement_return, accum_std, retire_std
    )

    # Draw schedule: higher before Social Security begins, lower once it does.
    annual_draws = make_annual_draws(
        monthly_retirement_spend, social_security_monthly, ss_start_year, retirement_years
    )

    # Solve for salary where the joint probability of accumulating enough AND
    # sustaining the full retirement drawdown equals target_success.
    # Each Brent iteration runs accumulation → decumulation end-to-end on all 10,000 paths.
    try:
        required_salary = solve_required_salary(
            partner_income, years, spending, current_portfolio,
            salary_growth, match_rate, target_success, inheritance, inheritance_year,
            accum_growth, annual_draws, retirement_years, ret_growth, legacy_target
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Final full simulation for charts and display metrics.
    portfolios, salaries, contributions, wealth_paths = simulate_salary_model(
        required_salary, partner_income, years, spending, current_portfolio,
        accum_growth, salary_growth, match_rate, inheritance, inheritance_year
    )
    assert portfolios is not None and wealth_paths is not None and contributions is not None

    # target_portfolio is display-only: the median ending accumulation portfolio
    # across all simulations at the solved salary.
    target_portfolio = float(np.median(portfolios))

    # Decumulation — each sim's ending portfolio seeds its own drawdown on the
    # continuation of the same AR(1) path.
    decum_paths, joint_success = simulate_decumulation_paths(
        portfolios, annual_draws, retirement_years, ret_growth, legacy_target
    )

    # Survival rate is always fraction ending above $0, regardless of legacy goal.
    # When legacy_target=0 this equals joint_success; otherwise it is higher.
    survival_rate = float(np.mean(decum_paths[:, -1] > 0))

    logger.info(
        "required_salary=$%s target_portfolio(median)=$%s joint_success=%s survival_rate=%s",
        f"{required_salary:,.0f}", f"{target_portfolio:,.0f}",
        f"{joint_success:.1%}", f"{survival_rate:.1%}",
    )

years_axis  = list(range(years + 1))
contrib_x   = years_axis[1:]
retire_axis = list(range(years, years + retirement_years + 1))

# -----------------------------
# TOP METRICS
# -----------------------------

st.subheader("Results")

cols = st.columns(4 if legacy_target > 0 else 3)

cols[0].metric("Required Starting Salary", f"${required_salary:,.0f}")
cols[1].metric(
    "Median Portfolio at Retirement", f"${target_portfolio:,.0f}",
    help="Median ending accumulation portfolio across all simulations. Display only."
)
cols[2].metric(
    "Portfolio Survival Rate", f"{survival_rate:.1%}",
    help="Fraction of simulations where the portfolio is not fully depleted by end of retirement."
)

if legacy_target > 0:
    cols[3].metric(
        f"Legacy Success (≥ ${legacy_target/1e6:.1f}M)",
        f"{joint_success:.1%}",
        help=f"Fraction of simulations ending retirement with at least ${legacy_target:,.0f}. "
             f"This is the target the solver optimizes against."
    )

st.divider()

# -----------------------------
# UNIFIED TIMELINE
# -----------------------------

st.subheader("Retirement Journey")

accum_p10 = np.percentile(wealth_paths, 10, axis=0)
accum_p50 = np.percentile(wealth_paths, 50, axis=0)
accum_p90 = np.percentile(wealth_paths, 90, axis=0)

decum_p10 = np.percentile(decum_paths, 10, axis=0)
decum_p50 = np.percentile(decum_paths, 50, axis=0)
decum_p90 = np.percentile(decum_paths, 90, axis=0)

BLUE   = "#2196F3"
ORANGE = "#FF9800"

fig_timeline = go.Figure()

# Accumulation phase — blue
fig_timeline.add_trace(go.Scatter(
    x=years_axis, y=accum_p90, name="Accum 90th",
    line=dict(dash="dot", width=1, color=BLUE), opacity=0.45, showlegend=False,
))
fig_timeline.add_trace(go.Scatter(
    x=years_axis, y=accum_p50, name="Accumulation (median)",
    line=dict(width=2.5, color=BLUE),
))
fig_timeline.add_trace(go.Scatter(
    x=years_axis, y=accum_p10, name="Accum 10th",
    line=dict(dash="dot", width=1, color=BLUE), opacity=0.45, showlegend=False,
))

# Decumulation phase — orange
fig_timeline.add_trace(go.Scatter(
    x=retire_axis, y=decum_p90, name="Retire 90th",
    line=dict(dash="dot", width=1, color=ORANGE), opacity=0.45, showlegend=False,
))
fig_timeline.add_trace(go.Scatter(
    x=retire_axis, y=decum_p50, name="Retire 50th",
    line=dict(width=2.5, color=ORANGE),
))
fig_timeline.add_trace(go.Scatter(
    x=retire_axis, y=decum_p10, name="Retire 10th",
    line=dict(dash="dot", width=1, color=ORANGE), opacity=0.45, showlegend=False,
))

fig_timeline.add_vline(
    x=years, line_dash="dash", line_color="gray",
    annotation_text="Retirement", annotation_position="top right",
)
fig_timeline.add_hline(
    y=target_portfolio, line_dash="dot", line_color="red", opacity=0.5,
    annotation_text=f"Median ${target_portfolio/1e6:.1f}M", annotation_font_size=11,
)
if legacy_target > 0:
    fig_timeline.add_hline(
        y=legacy_target, line_dash="dot", line_color="green", opacity=0.6,
        annotation_text=f"Legacy ${legacy_target/1e6:.1f}M", annotation_font_size=11,
    )
fig_timeline.update_layout(
    xaxis_title="Year",
    yaxis_title="Portfolio Value",
    yaxis=dict(tickprefix="$", tickformat=",.0f"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=440,
)
st.plotly_chart(fig_timeline, use_container_width=True)

st.divider()

# -----------------------------
# SALARY PATH
# -----------------------------

fig_salary = go.Figure()
fig_salary.add_trace(go.Bar(x=contrib_x, y=salaries, name="Salary"))
fig_salary.update_layout(
    title="Salary Path",
    xaxis_title="Year", yaxis_title="Salary",
    xaxis=dict(dtick=1),
    yaxis=dict(tickprefix="$", tickformat=",.0f"),
    height=300,
)
st.plotly_chart(fig_salary, use_container_width=True)

st.divider()

# -----------------------------
# PORTFOLIO DISTRIBUTION + CONTRIBUTIONS
# -----------------------------

r1c1, r1c2 = st.columns(2, gap="medium")

with r1c1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=portfolios, nbinsx=50))
    fig.add_vline(x=target_portfolio, line_dash="dash", line_color="gray",
                  annotation_text="Median", annotation_position="top right")
    fig.update_layout(
        title="Portfolio at Retirement (Distribution)",
        xaxis_title="Portfolio Value", yaxis_title="Count",
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=contrib_x, y=contributions, name="Contributions"))
    fig4.update_layout(
        title="Annual Contributions",
        xaxis_title="Year", yaxis_title="Contribution",
        xaxis=dict(dtick=1),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        height=380,
    )
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# DRAWDOWN PATHS + FI PROBABILITY
# -----------------------------

r2c1, r2c2 = st.columns(2, gap="medium")

with r2c1:
    fig_decum = go.Figure()
    fig_decum.add_trace(go.Scatter(
        x=retire_axis, y=decum_p90, name="90th Percentile",
        line=dict(dash="dot", width=1), opacity=0.6,
    ))
    fig_decum.add_trace(go.Scatter(
        x=retire_axis, y=decum_p50, name="Median",
        line=dict(width=2.5),
    ))
    fig_decum.add_trace(go.Scatter(
        x=retire_axis, y=decum_p10, name="10th Percentile",
        line=dict(dash="dot", width=1), opacity=0.6,
    ))
    fig_decum.add_hline(y=0, line_color="red", opacity=0.3,
                        annotation_text="Depleted", annotation_font_size=11)
    if legacy_target > 0:
        fig_decum.add_hline(
            y=legacy_target, line_color="green", opacity=0.5,
            annotation_text=f"Legacy ${legacy_target/1e6:.1f}M", annotation_font_size=11,
        )
    fig_decum.update_layout(
        title="Retirement Drawdown Paths",
        xaxis_title="Year", yaxis_title="Portfolio Value",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        height=380,
    )
    st.plotly_chart(fig_decum, use_container_width=True)

with r2c2:
    # Threshold: 10th percentile of ending portfolios. Shows the fraction of paths
    # staying above the worst-decile outcome over the accumulation horizon.
    p10_final = float(np.percentile(portfolios, 10))
    prob_by_year = np.mean(wealth_paths >= p10_final, axis=0)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=years_axis, y=prob_by_year,
        mode="lines", name="P(above p10)", fill="tozeroy", line=dict(width=2),
    ))
    fig3.update_layout(
        title=f"Paths Above 10th-Percentile Outcome (${p10_final/1e6:.1f}M)",
        xaxis_title="Year", yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=380,
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# -----------------------------
# SENSITIVITY ANALYSIS
# -----------------------------

st.subheader("Sensitivity Analysis")
st.caption(
    "Required starting salary when one input is shifted by a fixed step. "
    "Blue = salary goes down (favorable); orange = salary goes up (unfavorable). "
    "Sorted by total swing."
)

_base_params = dict(
    monthly_retirement_spend=monthly_retirement_spend,
    retirement_years=retirement_years,
    social_security_monthly=social_security_monthly,
    ss_start_year=int(ss_start_year),
    partner_income=partner_income,
    years=years,
    spending=spending,
    current_portfolio=current_portfolio,
    salary_growth=salary_growth,
    match_rate=match_rate,
    inheritance=inheritance,
    inheritance_year=inheritance_year,
    mean_return=mean_return,
    retirement_return=retirement_return,
    accum_std=accum_std,
    retire_std=retire_std,
    target_success=target_success,
    legacy_target=legacy_target,
)

# (display label, param key, low override value, high override value, step label)
_cases: list[tuple[str, str, float, float, str]] = [
    ("Monthly Spend",     "monthly_retirement_spend", max(0.0, monthly_retirement_spend - 1_500), monthly_retirement_spend + 1_500, "±$1,500/mo"),
    ("Accum. Return",     "mean_return",              max(0.0, mean_return - 0.01),               mean_return + 0.01,               "±1 pp"),
    ("Retirement Return", "retirement_return",        max(0.0, retirement_return - 0.01),         retirement_return + 0.01,         "±1 pp"),
    ("Accum. Volatility",  "accum_std",               max(0.0, accum_std - 0.05),                 accum_std + 0.05,                 "±5 pp"),
    ("Retire Volatility",  "retire_std",              max(0.0, retire_std - 0.05),                 retire_std + 0.05,                "±5 pp"),
    ("Years to Retire",   "years",                    float(max(5, years - 5)),                   float(years + 5),                 "±5 yrs"),
    ("Current Portfolio", "current_portfolio",        max(0.0, current_portfolio - 100_000),      current_portfolio + 100_000,      "±$100k"),
    ("Partner Income",    "partner_income",            max(0.0, partner_income - 50_000),          partner_income + 50_000,          "±$50k"),
    ("Salary Growth",     "salary_growth",            max(0.0, salary_growth - 0.01),             salary_growth + 0.01,             "±1 pp"),
    ("Working Spending",  "spending",                 max(0.0, spending - 10_000),                spending + 10_000,                "±$10k/yr"),
]
if social_security_monthly > 0:
    _cases.append(("Social Security", "social_security_monthly",
                   max(0.0, social_security_monthly - 1_000), social_security_monthly + 1_000, "±$1,000/mo"))
if ss_start_year > 0:
    _cases.append(("SS Delay", "ss_start_year",
                   float(max(0, ss_start_year - 5)), float(min(retirement_years, ss_start_year + 5)), "±5 yrs"))
if legacy_target > 0:
    _cases.append(("Legacy Goal", "legacy_target",
                   max(0.0, legacy_target - 250_000), legacy_target + 250_000, "±$250k"))

with st.spinner("Computing sensitivity..."):
    _rows = []
    for _lbl, _key, _lo_val, _hi_val, _step_lbl in _cases:
        _s_lo = _sensitivity_salary({_key: _lo_val}, _base_params, accum_growth, ret_growth)
        _s_hi = _sensitivity_salary({_key: _hi_val}, _base_params, accum_growth, ret_growth)
        if _s_lo is not None and _s_hi is not None:
            _rows.append((_lbl, _s_lo, _s_hi, abs(_s_hi - _s_lo), _step_lbl))

_rows.sort(key=lambda r: r[3], reverse=True)

if _rows:
    _ylabels  = [f"{r[0]}<br><sup>{r[4]}</sup>" for r in _rows]
    _sal_mins = [min(r[1], r[2]) for r in _rows]
    _sal_maxs = [max(r[1], r[2]) for r in _rows]

    # Favorable portion: sal_min → baseline (blue)
    _fav_base  = _sal_mins
    _fav_width = [max(0.0, required_salary - s) for s in _sal_mins]
    # Unfavorable portion: baseline → sal_max (orange)
    _bad_width = [max(0.0, s - required_salary) for s in _sal_maxs]

    _hover_fav = [
        f"<b>{r[0]}</b> ({r[4]})<br>Favorable end: ${min(r[1],r[2]):,.0f}<br>Swing: ${r[3]:,.0f}"
        for r in _rows
    ]
    _hover_bad = [
        f"<b>{r[0]}</b> ({r[4]})<br>Unfavorable end: ${max(r[1],r[2]):,.0f}<br>Swing: ${r[3]:,.0f}"
        for r in _rows
    ]

    fig_tornado = go.Figure()
    fig_tornado.add_trace(go.Bar(
        y=_ylabels, x=_fav_width, base=_fav_base,
        orientation="h",
        marker_color=BLUE, opacity=0.75,
        name="Favorable",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=_hover_fav,
    ))
    fig_tornado.add_trace(go.Bar(
        y=_ylabels, x=_bad_width, base=[required_salary] * len(_rows),
        orientation="h",
        marker_color=ORANGE, opacity=0.75,
        name="Unfavorable",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=_hover_bad,
    ))
    fig_tornado.add_vline(
        x=required_salary, line_dash="dash", line_color="black", opacity=0.6,
        annotation_text=f"Baseline ${required_salary:,.0f}",
        annotation_position="top right",
    )
    fig_tornado.update_layout(
        barmode="overlay",
        xaxis_title="Required Starting Salary",
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=80 + 36 * len(_rows),
    )
    st.plotly_chart(fig_tornado, use_container_width=True)
